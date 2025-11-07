"""
Benchmark Helix-DB vs Gemini File Search for a shared workload.

The script:
  * loads the prepared dataset + queries
  * executes search workloads across varying concurrency levels
  * records per-query latency and aggregate percentiles
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

import httpx
from google import genai
from google.api_core import exceptions as g_exceptions
from google.ai.generativelanguage import file_search_service_pb2

from scripts import utils


def percentile(values: List[float], pct: float) -> float:
    """Return percentile using linear interpolation."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    rank = (len(values_sorted) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values_sorted[int(rank)]
    return values_sorted[low] + (values_sorted[high] - values_sorted[low]) * (rank - low)


def extract_top_hit(payload: Any) -> Dict[str, Any] | None:
    """Best-effort extraction of the first hit from a response object."""
    if isinstance(payload, list):
        for item in payload:
            hit = extract_top_hit(item)
            if hit:
                return hit
        return None
    if isinstance(payload, dict):
        # Direct list container
        for key in ("results", "chunks", "items", "matches"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, dict):
                    return first
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            content = first.get("content") or {}
            parts = content.get("parts") or []
            for part in parts:
                if isinstance(part, dict) and part.get("text"):
                    return {"text": part["text"]}
        # Nested recursion
        for value in payload.values():
            hit = extract_top_hit(value)
            if hit:
                return hit
    return None


def format_hit(hit: Dict[str, Any] | None) -> str:
    if not hit:
        return "<no results>"
    doc = hit.get("doc_id") or hit.get("document_id") or hit.get("chunk", {}).get("document_id")
    score = hit.get("score") or hit.get("chunk", {}).get("score")
    text = hit.get("text") or hit.get("chunk", {}).get("text", "")
    snippet = (text or "").replace("\n", " ")[:120]
    if doc or score:
        return f"doc={doc} score={score} text='{snippet}...'"
    return f"text='{snippet}...'"


async def run_in_executor(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))


def serialize_gemini_response(response: Any) -> Any:
    for attr in ("to_dict", "model_dump"):
        if hasattr(response, attr):
            return getattr(response, attr)()
    if hasattr(response, "to_json"):
        try:
            return json.loads(response.to_json())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(response, "text"):
        return {"text": response.text}
    return str(response)


def gemini_content(query_text: str) -> List[genai_types.Content]:
    return [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=query_text)],
        )
    ]


def build_file_search_tool(store_name: str, top_k: int) -> genai_types.Tool:
    spec = file_search_service_pb2.FileSearchSpec(
        file_search_store_names=[store_name],
        max_num_results=top_k,
    )
    return genai_types.Tool(
        file_search=file_search_service_pb2.FileSearch(file_search_spec=spec)
    )


def build_helix_runner(
    embedder: utils.OpenAIEmbedder,
    helix_client: utils.HelixClient,
    search_query: str,
    top_k: int,
) -> Any:
    def _runner(query: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        embedding = embedder.embed([query["text"]])[0]
        response = helix_client.run(
            search_query,
            {"embedding": embedding, "top_k": top_k, "text": query["text"]},
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return {"latency_ms": latency_ms, "payload": response}

    return _runner


def build_gemini_runner(
    gemini_client: genai.Client,
    store_name: str,
    model_name: str,
    top_k: int,
) -> Any:
    tool = build_file_search_tool(store_name, top_k)
    generation_config = genai_types.GenerateContentConfig(
        tools=[tool],
        response_modalities=["TEXT"],
    )

    def _runner(query: Dict[str, Any]) -> Dict[str, Any]:
        start = time.perf_counter()
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=gemini_content(query["text"]),
            config=generation_config,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        payload = serialize_gemini_response(response)
        return {"latency_ms": latency_ms, "payload": payload}

    return _runner


async def execute_load(
    system_name: str,
    concurrency: int,
    attempts: int,
    queries: List[Dict[str, Any]],
    runner,
) -> Tuple[List[Dict[str, Any]], float]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async def _task(query_meta: Dict[str, Any], attempt_idx: int) -> None:
        async with sem:
            outcome = await run_in_executor(runner, query_meta)
            results.append(
                {
                    "system": system_name,
                    "query_id": query_meta["id"],
                    "concurrency": concurrency,
                    "attempt": attempt_idx,
                    "latency_ms": outcome["latency_ms"],
                    "timestamp": utils.utc_timestamp_ms(),
                    "payload": outcome["payload"],
                }
            )

    level_start = time.perf_counter()
    tasks = [
        asyncio.create_task(_task(query_meta, attempt_idx))
        for attempt_idx in range(attempts)
        for query_meta in queries
    ]
    await asyncio.gather(*tasks)
    level_duration = time.perf_counter() - level_start
    return results, level_duration


def write_results_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["system", "query_id", "concurrency", "attempt", "latency_ms", "timestamp"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def summarize(results: List[Dict[str, Any]], durations: Dict[Tuple[str, int], float]) -> None:
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for row in results:
        grouped[(row["system"], row["concurrency"])].append(row["latency_ms"])

    logging.info("=== Aggregate Latency ===")
    for (system, conc), values in sorted(grouped.items()):
        q50 = percentile(values, 0.50)
        q95 = percentile(values, 0.95)
        q99 = percentile(values, 0.99)
        duration = max(durations.get((system, conc), float("nan")), 1e-9)
        throughput = len(values) / duration
        logging.info(
            "%s @ %sx concurrency | p50=%.1fms p95=%.1fms p99=%.1fms | QPS=%.2f",
            system,
            conc,
            q50,
            q95,
            q99,
            throughput,
        )


def sanity_print(results: List[Dict[str, Any]], sample_size: int) -> None:
    logging.info("=== Sanity Samples ===")
    samples = []
    seen = set()
    for row in results:
        key = (row["system"], row["query_id"])
        if key in seen:
            continue
        seen.add(key)
        samples.append(row)
        if len(samples) >= sample_size * 2:
            break
    for row in samples[: sample_size * 2]:
        hit_str = format_hit(extract_top_hit(row.get("payload")))
        logging.info("%s | %s | %s", row["system"], row["query_id"], hit_str)


async def main_async() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = utils.load_config()
    benchmark_cfg = config["benchmark"]
    data_cfg = config["data"]
    helix_cfg = config["helix"]
    gemini_cfg = config["gemini"]
    embed_cfg = config["embeddings"]

    queries_path = utils.PROJECT_ROOT / data_cfg["queries_file"]
    queries = list(utils.read_jsonl(queries_path)) if queries_path.suffix == ".jsonl" else None
    if queries is None:
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if not queries:
        raise RuntimeError("No queries loaded for benchmarking.")

    embedder = utils.OpenAIEmbedder(
        api_key=utils.ensure_env(embed_cfg["api_key_env"]),
        model=embed_cfg["model"],
        api_base=embed_cfg.get("api_base", "https://api.openai.com/v1"),
        timeout=embed_cfg.get("timeout_seconds", 60),
        max_retries=embed_cfg.get("max_retries", 5),
        retry_backoff=embed_cfg.get("retry_backoff_seconds", 2.0),
    )
    helix_api_key = None
    if helix_cfg.get("api_key_env"):
        helix_api_key = utils.ensure_env(helix_cfg["api_key_env"])
    helix_client = utils.HelixClient(
        base_url=helix_cfg["base_url"],
        api_key=helix_api_key,
        timeout=helix_cfg.get("timeout_seconds", 30),
    )
    gemini_client = genai.Client(api_key=utils.ensure_env("GEMINI_API_KEY"))
    store_name = gemini_cfg.get("store_name")
    if not store_name:
        raise RuntimeError(
            "Gemini store name is not configured. "
            "Run scripts/ingest_gemini.py and update config.gemini.store_name with the logged identifier."
        )

    concurrency_levels = benchmark_cfg["concurrency"]
    attempts = benchmark_cfg.get("attempts_per_level", 3)
    top_k = benchmark_cfg.get("report_top_k", 5)
    sanity_sample = benchmark_cfg.get("sanity_sample", 2)
    output_dir = utils.PROJECT_ROOT / benchmark_cfg.get("output_dir", "results")

    helix_runner = build_helix_runner(embedder, helix_client, helix_cfg["search_query"], top_k)
    gemini_runner = build_gemini_runner(
        gemini_client,
        store_name,
        gemini_cfg["model"],
        gemini_cfg.get("results_per_query", top_k),
    )

    all_results: List[Dict[str, Any]] = []
    level_durations: Dict[Tuple[str, int], float] = {}

    for conc in concurrency_levels:
        logging.info("--- Helix concurrency=%s ---", conc)
        helix_rows, helix_duration = await execute_load("helix", conc, attempts, queries, helix_runner)
        all_results.extend(helix_rows)
        level_durations[("helix", conc)] = helix_duration

        logging.info("--- Gemini concurrency=%s ---", conc)
        gemini_rows, gemini_duration = await execute_load("gemini", conc, attempts, queries, gemini_runner)
        all_results.extend(gemini_rows)
        level_durations[("gemini", conc)] = gemini_duration

    helix_rows = [row for row in all_results if row["system"] == "helix"]
    gemini_rows = [row for row in all_results if row["system"] == "gemini"]
    write_results_csv(output_dir / "helix_results.csv", helix_rows)
    write_results_csv(output_dir / "gemini_results.csv", gemini_rows)

    summarize(all_results, level_durations)
    sanity_print(all_results, sanity_sample)

    await embedder.aclose()


def main() -> None:
    try:
        asyncio.run(main_async())
    except httpx.HTTPStatusError as exc:
        logging.error("HTTP error: %s %s", exc.response.status_code, exc.response.text)
        raise
    except g_exceptions.GoogleAPICallError as exc:
        logging.error("Gemini API error: %s", exc)
        raise


if __name__ == "__main__":
    main()
