"""Embeddings: local BGE-M3 or Azure AI Foundry, with an offline fake for tests.

BGE-M3 is the Addendum's confirmed embedding model — multilingual including
Thai, runs on the dev Mac, no API key. The model is loaded lazily on first use:
importing this module must never pull weights, or the test suite stops working
offline.

`POC_EMBEDDING_PROVIDER=azure_foundry` swaps in `AzureEmbedder` instead — for a
machine that can't download/run BGE-M3 locally, this calls out to an Azure AI
Foundry / Azure OpenAI embedding deployment over HTTP. Same `Embedder` shape,
so `store.py`/`ingest.py` need no changes.

`POC_FAKE_EMBEDDINGS=1` swaps in `FakeEmbedder`, a deterministic hashed
bag-of-features vector. It is not semantic, but nearest-neighbour on shared
vocabulary is close enough to exercise the whole ingest → retrieve pipeline with
no download. Checked before `embedding_provider`, so it wins in tests regardless
of which real backend is configured.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Protocol, runtime_checkable

import httpx

from socratic_tutor.config import get_settings

#: Used only when settings carry no model id; the configured default is the same.
DEFAULT_MODEL_ID = "BAAI/bge-m3"

FAKE_ENV_FLAG = "POC_FAKE_EMBEDDINGS"
FAKE_DIMENSION = 256

_THAI_CHARS = re.compile(r"[฀-๿]+")
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+")


def text_features(text: str) -> list[str]:
    """Lexical features for offline scoring: Latin words + Thai character 3-grams.

    Thai has no word boundaries, so n-grams stand in for tokens. Used by
    `FakeEmbedder` and by the keyword half of the store's hybrid ranking.
    """
    features = [word.lower() for word in _LATIN_WORD.findall(text)]
    for run in _THAI_CHARS.findall(text):
        if len(run) <= 3:
            features.append(run)
        else:
            features.extend(run[i : i + 3] for i in range(len(run) - 2))
    return features


@runtime_checkable
class Embedder(Protocol):
    """What `store` and `ingest` need from an embedding backend."""

    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _default_model_id() -> str:
    return get_settings().embedding_model or DEFAULT_MODEL_ID


def use_fake_embeddings() -> bool:
    return os.environ.get(FAKE_ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


class BGEM3Embedder:
    """Local `sentence-transformers` BGE-M3. Weights load on first embed call."""

    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        self.model_id = model_id or _default_model_id()
        self.device = device
        self._model = None
        self._dimension = 0

    @property
    def dimension(self) -> int:
        if not self._dimension:
            self._dimension = int(self._load().get_sentence_embedding_dimension())
        return self._dimension

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class EmbedderError(Exception):
    """An embedding backend failed — network, auth, or an unexpected response shape."""


class AzureEmbedder:
    """Azure AI Foundry / Azure OpenAI embeddings, for a machine that can't run
    BGE-M3 locally. Mirrors `BGEM3Embedder`'s shape (lazy dimension, batched
    `embed_texts`, `embed_query` as a one-item call).

    Azure's contract needs three things plain OpenAI/OpenRouter don't: an
    `api-key` header (not `Authorization: Bearer`), a mandatory `api-version`
    query parameter, and the deployment name in the request path.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        deployment: str,
        api_key: str,
        api_version: str = "2024-10-21",
        timeout_s: float = 60.0,
        batch_size: int = 96,
        client: httpx.Client | None = None,
    ) -> None:
        if not endpoint:
            raise EmbedderError(
                "embedding_provider is 'azure_foundry' but POC_EMBEDDING_AZURE_ENDPOINT is empty"
            )
        if not deployment:
            raise EmbedderError(
                "embedding_provider is 'azure_foundry' but POC_EMBEDDING_AZURE_DEPLOYMENT is empty"
            )
        if not api_key:
            raise EmbedderError(
                "embedding_provider is 'azure_foundry' but POC_EMBEDDING_AZURE_API_KEY is not set; "
                "put it in .env at the repo root"
            )
        self._deployment = deployment
        self._batch_size = max(1, batch_size)
        self._dimension = 0
        if client is None:
            client = httpx.Client(
                base_url=f"{endpoint.rstrip('/')}/openai/deployments/{deployment}",
                headers={"api-key": api_key},
                params={"api-version": api_version},
                timeout=timeout_s,
            )
        self.client = client

    def __repr__(self) -> str:
        return f"AzureEmbedder(deployment={self._deployment!r}, api_key=<hidden>)"

    @property
    def dimension(self) -> int:
        if not self._dimension:
            self._dimension = len(self.embed_query("."))
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self.client.post("/embeddings", json={"input": batch})
        except httpx.TimeoutException as exc:
            raise EmbedderError("azure_foundry: embeddings request timed out") from exc
        except httpx.TransportError as exc:
            raise EmbedderError(f"azure_foundry: cannot reach the endpoint: {exc}") from exc
        if response.status_code >= 400:
            raise EmbedderError(
                f"azure_foundry: HTTP {response.status_code} from embeddings deployment "
                f"{self._deployment!r}: {response.text[:500]}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise EmbedderError(f"azure_foundry: embeddings response was not JSON: {exc}") from exc
        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise EmbedderError(
                f"azure_foundry: expected {len(batch)} embedding(s), got "
                f"{len(rows) if isinstance(rows, list) else 'a non-list response'}"
            )
        by_index = sorted(rows, key=lambda row: row.get("index", 0))
        return [[float(v) for v in row["embedding"]] for row in by_index]


class FakeEmbedder:
    """Deterministic hashed features — offline stand-in for BGE-M3.

    Features are lowercase Latin words plus Thai character 3-grams, because Thai
    has no word boundaries to split on. Two texts sharing vocabulary land close
    together, which is all the pipeline tests need.
    """

    def __init__(self, dimension: int = FAKE_DIMENSION) -> None:
        self.dimension = dimension

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for feature in text_features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An empty or symbol-only text still needs a valid unit vector.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


_cache: dict[str, Embedder] = {}


def get_embedder(model_id: str | None = None) -> Embedder:
    """Return the configured embedder, cached per model id.

    Honours `POC_FAKE_EMBEDDINGS` first (tests always get the fake, regardless
    of `embedding_provider`). Otherwise dispatches on `settings.embedding_provider`:
    `"local"` builds a `BGEM3Embedder` (still without loading weights — that
    happens on first embed call); `"azure_foundry"` builds an `AzureEmbedder`
    for a machine that can't run BGE-M3 locally.
    """
    if use_fake_embeddings():
        key = "fake"
        _cache.setdefault(key, FakeEmbedder())
        return _cache[key]
    settings = get_settings()
    if settings.embedding_provider == "azure_foundry":
        key = f"azure_foundry:{settings.embedding_azure_deployment}"
        if key not in _cache:
            _cache[key] = AzureEmbedder(
                endpoint=settings.embedding_azure_endpoint,
                deployment=settings.embedding_azure_deployment,
                api_key=settings.embedding_azure_api_key or "",
                api_version=settings.embedding_azure_api_version,
                timeout_s=settings.embedding_azure_timeout_s,
                batch_size=settings.embedding_azure_batch_size,
            )
        return _cache[key]
    key = model_id or _default_model_id()
    _cache.setdefault(key, BGEM3Embedder(key))
    return _cache[key]


def reset_embedder_cache() -> None:
    """Drop cached embedders — used by tests that flip `POC_FAKE_EMBEDDINGS`."""
    _cache.clear()


__all__ = [
    "DEFAULT_MODEL_ID",
    "FAKE_DIMENSION",
    "FAKE_ENV_FLAG",
    "AzureEmbedder",
    "BGEM3Embedder",
    "Embedder",
    "EmbedderError",
    "FakeEmbedder",
    "get_embedder",
    "reset_embedder_cache",
    "text_features",
    "use_fake_embeddings",
]
