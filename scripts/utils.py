"""
Shared helpers for benchmark scripts.

This module keeps configuration loading, batching utilities, HTTP client
helpers, and lightweight wrappers around the OpenAI embedding API as
building blocks for the rest of the harness.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import asyncio
import httpx

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load TOML configuration once per invocation."""
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as fh:
        return tomllib.load(fh)


def ensure_env(var_name: str) -> str:
    """Fetch a required environment variable or fail loudly."""
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return value


def batched(iterable: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Yield fixed-size batches from an iterable."""
    if size <= 0:
        raise ValueError("Batch size must be positive")
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSONL entries."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def utc_timestamp_ms() -> int:
    """Return current UTC timestamp in milliseconds."""
    return int(time.time() * 1000)


@dataclass
class Timer:
    """Simple timing helper."""

    start: float

    @classmethod
    def start_now(cls) -> "Timer":
        return cls(start=time.perf_counter())

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


class OpenAIEmbedder:
    """Minimal client for the OpenAI embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        max_retries: int = 5,
        retry_backoff: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._sync_client = httpx.Client(timeout=timeout)
        self._async_client: Optional[httpx.AsyncClient] = None

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Synchronously embed a batch of texts."""
        if not texts:
            return []
        payload = self._request_with_retries_sync(list(texts))
        return [row["embedding"] for row in payload.get("data", [])]

    async def embed_async(self, texts: Sequence[str]) -> List[List[float]]:
        """Asynchronously embed a batch of texts."""
        if not texts:
            return []
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        payload = await self._request_with_retries_async(list(texts))
        return [row["embedding"] for row in payload.get("data", [])]

    async def aclose(self) -> None:
        """Close async resources when done."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
        self._sync_client.close()

    def _should_retry(self, status_code: int) -> bool:
        return status_code in {429, 500, 502, 503, 504}

    def _compute_sleep(self, attempt: int) -> float:
        return self.retry_backoff * (2 ** attempt)

    def _request_with_retries_sync(self, texts: Sequence[str]) -> Dict[str, Any]:
        payload = {"model": self.model, "input": list(texts)}
        url = f"{self.api_base}/embeddings"
        for attempt in range(self.max_retries):
            try:
                response = self._sync_client.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if attempt < self.max_retries - 1 and self._should_retry(status):
                    sleep_time = self._compute_sleep(attempt)
                    logging.warning(
                        "Embedding request failed with status %s. Retrying in %.1fs",
                        status,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue
                raise
            except httpx.RequestError as exc:
                if attempt < self.max_retries - 1:
                    sleep_time = self._compute_sleep(attempt)
                    logging.warning(
                        "Embedding request error %s. Retrying in %.1fs",
                        exc,
                        sleep_time,
                    )
                    time.sleep(sleep_time)
                    continue
                raise
        raise RuntimeError("Exhausted retries for embeddings request.")

    async def _request_with_retries_async(self, texts: Sequence[str]) -> Dict[str, Any]:
        if self._async_client is None:
            raise RuntimeError("Async client not initialized")
        payload = {"model": self.model, "input": list(texts)}
        url = f"{self.api_base}/embeddings"
        for attempt in range(self.max_retries):
            try:
                response = await self._async_client.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if attempt < self.max_retries - 1 and self._should_retry(status):
                    sleep_time = self._compute_sleep(attempt)
                    logging.warning(
                        "Embedding request failed with status %s. Retrying in %.1fs",
                        status,
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                raise
            except httpx.RequestError as exc:
                if attempt < self.max_retries - 1:
                    sleep_time = self._compute_sleep(attempt)
                    logging.warning(
                        "Embedding request error %s. Retrying in %.1fs",
                        exc,
                        sleep_time,
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                raise
        raise RuntimeError("Exhausted retries for embeddings request.")


class HelixClient:
    """Lightweight wrapper over Helix-DB's per-query HTTP endpoints."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def run(self, query_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{query_name.lstrip('/')}"
        response = self._client.post(url, json=params)
        response.raise_for_status()
        return response.json()

