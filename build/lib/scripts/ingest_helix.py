"""
Ingest processed chunks into a local Helix-DB instance.

The script expects:
  - OpenAI embeddings configured in config.toml
  - Helix dev server reachable at helix.base_url
  - dataset.jsonl created via prepare_dataset.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Iterable

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import utils


def load_chunks(path: Path) -> Iterable[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    return utils.read_jsonl(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = utils.load_config()
    data_cfg = config["data"]
    helix_cfg = config["helix"]
    embed_cfg = config["embeddings"]

    chunks_path = utils.PROJECT_ROOT / data_cfg["chunks_file"]
    helix_api_key = None
    api_key_env = helix_cfg.get("api_key_env")
    if api_key_env:
        helix_api_key = utils.ensure_env(api_key_env)

    embedder = utils.OpenAIEmbedder(
        api_key=utils.ensure_env(embed_cfg["api_key_env"]),
        model=embed_cfg["model"],
        api_base=embed_cfg.get("api_base", "https://api.openai.com/v1"),
        timeout=embed_cfg.get("timeout_seconds", 60),
        max_retries=embed_cfg.get("max_retries", 5),
        retry_backoff=embed_cfg.get("retry_backoff_seconds", 2.0),
    )
    helix_client = utils.HelixClient(
        base_url=helix_cfg["base_url"],
        api_key=helix_api_key,
        timeout=helix_cfg.get("timeout_seconds", 30),
    )

    batch_size = helix_cfg.get("batch_size", 64)
    total = 0
    timer = utils.Timer.start_now()
    for batch in utils.batched(load_chunks(chunks_path), batch_size):
        embeddings = embedder.embed([row["text"] for row in batch])
        for chunk, vector in zip(batch, embeddings, strict=True):
            payload = {
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": vector,
            }
            response = helix_client.run(
                helix_cfg["insert_query"],
                payload,
            )
            logging.debug("Helix insert response: %s", response)
            total += 1

    elapsed = timer.elapsed()
    if elapsed == 0:
        elapsed = 1e-6
    logging.info(
        "Inserted %s chunks in %.2fs (%.1f chunks/sec)",
        total,
        elapsed,
        total / elapsed,
    )


if __name__ == "__main__":
    main()
