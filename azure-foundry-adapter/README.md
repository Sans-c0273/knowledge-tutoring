# azure-foundry-adapter

Azure AI Foundry / Azure OpenAI adapter code for the three sibling projects in
`socratic_tutor/`. None of the three project repos are modified — this is a
standalone folder holding the client code, one adapter per project (each
matching that project's existing provider interface exactly), and copy-paste
instructions for wiring each one in.

```
azure-foundry-adapter/
├── azure_foundry_adapter/
│   ├── client.py                     # shared httpx-based REST client (chat + embeddings, sync + async)
│   ├── errors.py                     # provider-agnostic exception types
│   └── adapters/
│       ├── kg_reasoner.py            # LLMCall protocol — usable today, no repo edits
│       ├── knowledge_graph_poc.py    # Adapter protocol — copy into kg's adapters/ folder
│       └── socratic_tutor_poc.py     # LLMProvider ABC — copy into poc's providers/ folder
├── shims/
│   └── sentence_transformers/        # shadows the real package via PYTHONPATH —
│       └── __init__.py               # no repo edits — see INTEGRATION.md "embeddings"
├── INTEGRATION.md                    # exact per-project wiring steps
├── .env.example
└── pyproject.toml
```

## Why one folder, three adapters

The three projects were built independently and each already has its own
provider abstraction (see prior research in this conversation):

| Project | Its interface | Its existing providers |
|---|---|---|
| `kg_reasoner` | `LLMCall` Protocol (`kg_reasoner.llm_reasoner`) — inject any callable | none built in; provider-agnostic by design |
| `knowledge-graph-poc` | `Adapter` Protocol (`kg.llm.adapters.base`) | `claude_subscription`, `openrouter`, `anthropic_api` |
| `socratic-tutor-poc` | `LLMProvider` ABC (`socratic_tutor.providers.base`) | `claude_subscription`, `anthropic`, `openai_compat` |

There's no shared interface across the three, so a single generic "Azure
adapter" can't satisfy all of them — each file here is written specifically
against one project's own types (`RawCompletion`/`Msg` for
knowledge-graph-poc, `StructuredResult`/`StreamStats` for socratic-tutor-poc,
a bare dict for kg_reasoner), reusing `azure_foundry_adapter/client.py`'s
request-building logic where that doesn't conflict with a project's own SDK
choice.

## What Azure's contract needs that OpenRouter/generic OpenAI doesn't

All three adapters exist to bridge the same three differences:

1. **Auth header**: `api-key: <key>`, not `Authorization: Bearer <key>`
   (or an Entra ID/AAD bearer token, supported as an alternative — see `client.py`'s `entra_token_provider`).
2. **Mandatory `api-version` query parameter** on every request.
3. **Deployment name, not base model id**: Azure addresses whatever you named
   the model when you deployed it in Azure AI Foundry / Azure OpenAI Studio —
   passing `"gpt-4o"` where a deployment name is expected fails.

## Quick start

```bash
cd azure-foundry-adapter
uv venv && uv pip install -e .
cp .env.example .env   # fill in AZURE_FOUNDRY_ENDPOINT / API_KEY / DEPLOYMENT
```

Try it directly against `kg_reasoner` (no other repo edits required):

```python
from azure_foundry_adapter.client import AzureFoundrySettings
from azure_foundry_adapter.adapters.kg_reasoner import AzureFoundryLLMCall

call = AzureFoundryLLMCall(AzureFoundrySettings.from_env())
result = call(
    system="Reply with JSON only.",
    user='{"ping": "pong"}',
    schema_json={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False},
)
print(result)
```

For `knowledge-graph-poc` and `socratic-tutor-poc`, see **`INTEGRATION.md`** —
each needs its adapter file copied into that project's own package tree (since
it imports that project's internal modules) plus a handful of lines in a
config/factory file to register `"azure_foundry"` as a selectable provider,
exactly the same shape as how `openrouter`/`openai_compat` are already
registered there today.

## Two Azure resource shapes

Set `AZURE_FOUNDRY_API_STYLE` (or `AzureFoundrySettings.api_style`) to match
how you provisioned things:

- `azure_openai` (default) — a classic Azure OpenAI resource
  (`https://<resource>.openai.azure.com`), deployment name in the URL path.
- `foundry_models` — an Azure AI Foundry resource using the unified Foundry
  Models inference endpoint (`https://<resource>.services.ai.azure.com`),
  model/deployment name in the request body instead.

`client.py` (and therefore the `kg_reasoner` adapter) handles both via this
flag. The `knowledge-graph-poc` and `socratic-tutor-poc` adapters currently
assume `azure_openai`; INTEGRATION.md notes the small change needed for
`foundry_models`.

## Testing without real credentials

Every adapter accepts an injected `client` (mirroring each project's own
adapters, which all support this for tests), using `httpx.MockTransport`
instead of a real network call:

```python
import httpx
from azure_foundry_adapter.adapters.kg_reasoner import AzureFoundryLLMCall
from azure_foundry_adapter.client import AzureFoundrySettings

def fake_transport(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": '{"selections": []}'}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })

settings = AzureFoundrySettings(endpoint="https://fake", deployment="fake", api_key="fake")
call = AzureFoundryLLMCall(settings, client=httpx.Client(transport=httpx.MockTransport(fake_transport)))
print(call(system="...", user="...", schema_json={"type": "object"}))
```

This exact pattern (plus the sync/async `client.py` functions directly, a 404 →
`AzureFoundryModelNotFound` mapping, and the `foundry_models` URL/body shape)
is exercised in `scratch_test.py` at the root of this folder — run
`uv venv && uv pip install -e . && .venv/bin/python scratch_test.py` to see it
pass. It only proves the request/response shape is correct against a mocked
transport; **no live Azure AI Foundry credentials were available while
writing this**, so none of the three adapters have been exercised against a
real endpoint yet — do that before relying on them (see **Verify** steps in
`INTEGRATION.md`).
