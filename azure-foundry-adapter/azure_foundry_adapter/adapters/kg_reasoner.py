"""Azure AI Foundry callable for `kg_reasoner`'s Tier-2 LLM re-rank.

`kg_reasoner` has zero import-time dependency on any provider SDK (see
`kg_reasoner.llm_reasoner.LLMCall`, a bare `Protocol`): you inject any callable
shaped `(*, system, user, schema_json) -> dict`. This module is that callable,
fully self-contained (only needs `httpx`, already a dependency of this
`azure-foundry-adapter` package) — no edits to kg_reasoner itself are needed.

Usage (from wherever you invoke kg_reasoner.pipeline.reason(...)):

    from azure_foundry_adapter.client import AzureFoundrySettings
    from azure_foundry_adapter.adapters.kg_reasoner import AzureFoundryLLMCall
    from kg_reasoner.pipeline import reason

    call = AzureFoundryLLMCall(AzureFoundrySettings.from_env())
    guidance = reason(ctx, index, llm_call=call)  # Tier 2 now re-ranks via Azure AI Foundry
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from azure_foundry_adapter.client import AzureFoundrySettings, complete_sync, first_message_content
from azure_foundry_adapter.errors import AzureFoundryResponseError


class AzureFoundryLLMCall:
    """Satisfies `kg_reasoner.llm_reasoner.LLMCall`.

    Every kg_reasoner Tier-2 call already passes a closed JSON schema and
    expects a dict back (or an exception, which the caller treats as
    best-effort failure and safely ignores — see `rerank()`'s docstring).
    """

    def __init__(
        self,
        settings: AzureFoundrySettings,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client  # optional injected client, e.g. httpx.Client(transport=...) in tests

    def __call__(self, *, system: str, user: str, schema_json: dict[str, Any]) -> dict[str, Any]:
        data = complete_sync(
            self._settings,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            schema_json=schema_json,
            schema_name="kg_reasoner_rerank",
            client=self._client,
        )
        content = first_message_content(data)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise AzureFoundryResponseError(f"azure_foundry: kg_reasoner call returned unparseable JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise AzureFoundryResponseError("azure_foundry: kg_reasoner call did not return a JSON object")
        return parsed


__all__ = ["AzureFoundryLLMCall"]
