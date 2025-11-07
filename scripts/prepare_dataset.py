"""
Create deterministic RAG-style chunks from the raw corpus.

Usage:
    uv run python scripts/prepare_dataset.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Iterable, Iterator, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

TARGET_CHUNK_SIZE = 400  # ~tokens approximated by word count
MIN_CHUNK_SIZE = 120
MAX_CHARS = 2800
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "chunks" / "dataset.jsonl"


def iter_source_files() -> Iterator[Path]:
    """Yield candidate source files under data/raw."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw corpus directory not found: {RAW_DIR}")
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            yield path


def _paragraphs(text: str) -> List[str]:
    """Split raw text into trimmed paragraphs."""
    return [segment.strip() for segment in text.split("\n\n") if segment.strip()]


def chunk_paragraphs(paragraphs: Iterable[str]) -> Iterator[str]:
    """Greedy paragraph-based chunking with a soft size target."""
    chunk: List[str] = []
    token_budget = 0
    for para in paragraphs:
        para_tokens = len(para.split())
        para_chars = len(para)
        prospective_tokens = token_budget + para_tokens
        prospective_chars = sum(len(p) for p in chunk) + para_chars

        if (
            chunk
            and (prospective_tokens >= TARGET_CHUNK_SIZE or prospective_chars >= MAX_CHARS)
            and token_budget >= MIN_CHUNK_SIZE
        ):
            yield "\n\n".join(chunk)
            chunk = [para]
            token_budget = para_tokens
            continue

        chunk.append(para)
        token_budget += para_tokens

    if chunk:
        yield "\n\n".join(chunk)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for source in iter_source_files():
            text = source.read_text(encoding="utf-8")
            doc_id = source.relative_to(RAW_DIR).as_posix()
            paragraphs = _paragraphs(text)
            for idx, chunk_text in enumerate(chunk_paragraphs(paragraphs)):
                chunk_id = f"{doc_id.replace('/', '_')}__{idx:04d}"
                record = {"doc_id": doc_id, "chunk_id": chunk_id, "text": chunk_text}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1
    logging.info("Wrote %s chunks to %s", total_chunks, OUTPUT_PATH)


if __name__ == "__main__":
    main()
