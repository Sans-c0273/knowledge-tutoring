"""Drop-in replacement for the real `sentence_transformers` package, backed by
an Azure AI Foundry / Azure OpenAI embeddings deployment instead of a locally
downloaded and executed model.

Why this exists: `socratic-tutor-poc/src/socratic_tutor/domain/rag/embeddings.py`'s
`BGEM3Embedder` hard-codes exactly one call shape —

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, device=device)
    model.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    model.get_sentence_embedding_dimension()

— against whatever package `sentence_transformers` resolves to at import time.
Put this directory earlier on `PYTHONPATH` than site-packages (see this
package's INTEGRATION.md) and that import resolves here instead of the real
library, with **zero edits to socratic-tutor-poc's own files**: `BGEM3Embedder`
runs completely unmodified, calling this shim without knowing the difference.

This does not remove `sentence-transformers`/`torch` from what `uv sync`
installs (they're still declared in that project's `pyproject.toml`) — it only
stops the actual multi-gigabyte model weights from ever being downloaded or
run, since this shim's `SentenceTransformer` never touches the real library.

Reads `AZURE_FOUNDRY_EMBEDDING_*` env vars, falling back to the plain
`AZURE_FOUNDRY_*` ones for `ENDPOINT`/`API_KEY`/`API_VERSION`/`API_STYLE` so
you don't have to repeat them if the chat model and the embedding model live
on the same Azure resource — only `AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT` must be
set on its own, since an embedding model is always a separate deployment from
a chat model.

**Dimension warning**: switching embedding backends changes the vector
dimension (BGE-M3 = 1024; Azure `text-embedding-3-small` = 1536, `-large` =
3072 by default). Any existing Chroma collection built with BGE-M3 vectors is
incompatible with this shim's output — point `POC_DATA_DIR`/`POC_CHROMA_DIR`
at a fresh directory (or delete the old one) and re-ingest before using this.
"""

from __future__ import annotations

import math
import os
from typing import Any

from azure_foundry_adapter.client import AzureFoundrySettings, embed_sync

_DEFAULT_BATCH_SIZE = 96


def _env(name: str, fallback_name: str | None = None, default: str = "") -> str:
    value = os.environ.get(name)
    if value:
        return value
    if fallback_name:
        return os.environ.get(fallback_name, default)
    return default


def _settings_from_env() -> AzureFoundrySettings:
    return AzureFoundrySettings(
        endpoint=_env("AZURE_FOUNDRY_EMBEDDING_ENDPOINT", "AZURE_FOUNDRY_ENDPOINT"),
        deployment=_env("AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT"),
        api_key=_env("AZURE_FOUNDRY_EMBEDDING_API_KEY", "AZURE_FOUNDRY_API_KEY") or None,
        api_version=_env("AZURE_FOUNDRY_EMBEDDING_API_VERSION", "AZURE_FOUNDRY_API_VERSION", "2024-10-21"),
        api_style=_env("AZURE_FOUNDRY_EMBEDDING_API_STYLE", "AZURE_FOUNDRY_API_STYLE", "azure_openai"),  # type: ignore[arg-type]
    )


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


class SentenceTransformer:
    """Matches the slice of the real `sentence_transformers.SentenceTransformer`
    API that `BGEM3Embedder` calls. `model_name_or_path` and `device` are
    accepted (for signature compatibility) but ignored — the actual deployment
    to call comes from `AZURE_FOUNDRY_EMBEDDING_*` env vars, not from this
    constructor argument, since Azure addresses deployments, not model names.
    """

    def __init__(self, model_name_or_path: str | None = None, device: str | None = None, **_kwargs: Any) -> None:
        self.model_name_or_path = model_name_or_path
        self._settings: AzureFoundrySettings | None = None
        self._dimension = 0
        self._client: Any = None  # test hook: inject an httpx.Client via _client

    def _resolved_settings(self) -> AzureFoundrySettings:
        if self._settings is None:
            self._settings = _settings_from_env()
        return self._settings

    def get_sentence_embedding_dimension(self) -> int:
        if not self._dimension:
            self._dimension = len(self._embed_batch(["."], normalize=False)[0])
        return self._dimension

    def encode(
        self,
        sentences: str | list[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        **_kwargs: Any,
    ) -> list[float] | list[list[float]]:
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not texts:
            return [] if not single else []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), max(1, batch_size)):
            vectors.extend(self._embed_batch(texts[start : start + batch_size], normalize=normalize_embeddings))
        return vectors[0] if single else vectors

    def _embed_batch(self, batch: list[str], *, normalize: bool) -> list[list[float]]:
        data = embed_sync(self._resolved_settings(), batch, client=self._client)
        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise RuntimeError(
                f"azure_foundry embeddings shim: expected {len(batch)} embedding(s), got "
                f"{len(rows) if isinstance(rows, list) else 'a non-list response'}"
            )
        ordered = sorted(rows, key=lambda row: row.get("index", 0))
        vectors = [[float(v) for v in row["embedding"]] for row in ordered]
        return [_l2_normalize(v) for v in vectors] if normalize else vectors


__all__ = ["SentenceTransformer"]
