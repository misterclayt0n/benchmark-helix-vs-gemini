"""
Upload the local corpus to Gemini File Search using the official google-genai SDK.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
import sys
from typing import List

from google import genai
from google.api_core import exceptions as g_exceptions
from google.genai import types as genai_types

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import utils


def iter_raw_files(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    files = [p for p in sorted(raw_dir.glob("**/*")) if p.is_file()]
    if not files:
        raise RuntimeError(f"No files to ingest under {raw_dir}")
    return files


def ensure_store(client: genai.Client, store_name: str | None, display_name: str) -> str:
    if store_name:
        try:
            store = client.file_search_stores.get(name=store_name)
            logging.info("Reusing existing File Search store %s", store.name)
            return store.name
        except g_exceptions.NotFound:
            logging.warning(
                "Configured store %s not found. A new store will be created.", store_name
            )
        except g_exceptions.GoogleAPICallError as exc:
            logging.warning("Lookup for store %s failed: %s", store_name, exc)

    store = client.file_search_stores.create(config={"display_name": display_name})
    logging.info("Created File Search store %s", store.name)
    return store.name


def wait_for_operation(
    client: genai.Client,
    operation: genai_types.Operation,
    poll_interval: float,
) -> genai_types.Operation:
    op = operation
    op_name = getattr(op, "name", None)
    if not op_name:
        raise RuntimeError("Gemini operation missing name identifier.")

    while not getattr(op, "done", False):
        time.sleep(poll_interval)
        op = client.operations.get(op)

    error = getattr(op, "error", None)
    if error:
        raise RuntimeError(f"Gemini operation {op_name} failed: {error}")
    return op


def sanitized_display_name(path: Path, raw_root: Path) -> str:
    rel = path.relative_to(raw_root).as_posix()
    return rel.replace("/", "_")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = utils.load_config()
    data_cfg = config["data"]
    gemini_cfg = config["gemini"]

    raw_dir = utils.PROJECT_ROOT / data_cfg["raw_dir"]
    api_key = utils.ensure_env("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    poll_interval = gemini_cfg.get("poll_interval_seconds", 5)
    store_name_cfg = gemini_cfg.get("store_name") or None
    display_name = gemini_cfg.get("store_display_name", "Benchmark Store")

    files = iter_raw_files(raw_dir)
    timer = utils.Timer.start_now()
    total_bytes = 0

    store_name = ensure_store(client, store_name_cfg, display_name)

    for path in files:
        logging.info("Uploading %s", path)
        operation = client.file_search_stores.upload_to_file_search_store(
            file=str(path),
            file_search_store_name=store_name,
            config={"display_name": sanitized_display_name(path, raw_dir)},
        )
        wait_for_operation(client, operation, poll_interval)
        total_bytes += path.stat().st_size

    elapsed = timer.elapsed()
    if elapsed == 0:
        elapsed = 1e-6
    logging.info(
        "Gemini ingestion complete for store %s (%.2f MB in %.2fs, %.2f MB/s)",
        store_name,
        total_bytes / 1e6,
        elapsed,
        (total_bytes / 1e6) / elapsed,
    )
    logging.info("File Search store ready: %s", store_name)
    logging.info("Update config.gemini.store_name with this value to reuse the store.")


if __name__ == "__main__":
    main()
