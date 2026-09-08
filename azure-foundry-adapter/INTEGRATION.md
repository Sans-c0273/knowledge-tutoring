# Wiring this into each project

Nothing in `kg_reasoner/`, `knowledge-graph-poc/`, or `socratic-tutor-poc/` has
been touched. This file is the exact, copy-pasteable diff for each one — apply
whichever you actually want to use.

Install this package first (all three snippets below assume it's importable):

```bash
cd /home/sans/socratic_tutor/azure-foundry-adapter
uv venv && uv pip install -e .        # or: pip install -e .
```

---

## kg_reasoner — no repo edits needed

`kg_reasoner.llm_reasoner.LLMCall` is a bare `Protocol`; `reason(..., llm_call=...)`
takes any matching callable. Just pass one in from wherever you call it:

```python
from azure_foundry_adapter.client import AzureFoundrySettings
from azure_foundry_adapter.adapters.kg_reasoner import AzureFoundryLLMCall
from kg_reasoner.pipeline import reason

call = AzureFoundryLLMCall(AzureFoundrySettings.from_env())  # reads AZURE_FOUNDRY_* from env
guidance = reason(ctx, index, llm_call=call, top_n=3)
```

If you omit `llm_call` entirely, kg_reasoner just skips Tier 2 and returns
Tier-1 (rule-based) results — Azure AI Foundry is opt-in per call site.

---

## knowledge-graph-poc

1. Copy the adapter in:
   ```bash
   cp azure-foundry-adapter/azure_foundry_adapter/adapters/knowledge_graph_poc.py \
      knowledge-graph-poc/src/kg/llm/adapters/azure_foundry.py
   ```

2. In `knowledge-graph-poc/src/kg/config.py`:

   a. Add to the `PROVIDERS` tuple:
      ```python
      PROVIDERS: tuple[str, ...] = ("claude_subscription", "openrouter", "anthropic_api", "azure_foundry")
      ```

   b. Add to `CREDENTIAL_VARS` and `_CREDENTIAL_FOR_PROVIDER`:
      ```python
      CREDENTIAL_VARS: tuple[str, ...] = ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "AZURE_FOUNDRY_API_KEY")
      _CREDENTIAL_FOR_PROVIDER: dict[str, str | None] = {
          "claude_subscription": None,
          "openrouter": "OPENROUTER_API_KEY",
          "anthropic_api": "ANTHROPIC_API_KEY",
          "azure_foundry": "AZURE_FOUNDRY_API_KEY",
      }
      ```

   c. In `_parse_provider_settings`, add a branch alongside the existing `openrouter` one:
      ```python
      if provider == "azure_foundry":
          from kg.llm.adapters.azure_foundry import AzureFoundrySettings

          return AzureFoundrySettings(
              endpoint=_str(doc, f"{base}.endpoint"),
              api_version=_str(doc, f"{base}.api_version"),
              models=_models(doc, f"{base}.models"),
          )
      ```
      (Also add `AzureFoundrySettings` to the `ProviderSettings` union type near the top of the file.)

   d. In `adapter_factory`, add a branch alongside the existing `openrouter` one:
      ```python
      if provider == "azure_foundry":
          from kg.llm.adapters.azure_foundry import AzureFoundryAdapter

          return AzureFoundryAdapter(cfg)
      ```

3. In `knowledge-graph-poc/kg.yaml`, add a sibling block under `llm:` and select it:
   ```yaml
   llm:
     provider: azure_foundry        # was: claude_subscription | openrouter | anthropic_api

     azure_foundry:                 # Provider 4 — Azure AI Foundry / Azure OpenAI via openai.AzureOpenAI
       endpoint: https://<your-resource-name>.openai.azure.com
       api_version: "2024-10-21"
       models:                      # stage -> Azure *deployment name*, not the base model id
         describe: <deployment-name>
         atomize:  <deployment-name>
         edges:    <deployment-name>
         dedup:    <deployment-name>
   ```

4. Put your key in `knowledge-graph-poc/.env`:
   ```
   AZURE_FOUNDRY_API_KEY=...
   ```

5. Verify: `uv run kg selftest` (offline), then `uv run kg ingest --dry-run`.

---

## socratic-tutor-poc

1. Copy the adapter in:
   ```bash
   cp azure-foundry-adapter/azure_foundry_adapter/adapters/socratic_tutor_poc.py \
      socratic-tutor-poc/src/socratic_tutor/providers/azure_foundry.py
   ```

2. In `socratic-tutor-poc/src/socratic_tutor/config.py`:

   a. Add to the provider literal:
      ```python
      ProviderName = Literal["claude_subscription", "anthropic", "openai_compat", "azure_foundry"]
      ```

   b. Add settings fields (env-prefixed `POC_` automatically, per `model_config`):
      ```python
      # --- Azure AI Foundry / Azure OpenAI adapter
      azure_foundry_endpoint: str = ""
      azure_foundry_api_key: str | None = Field(default=None, repr=False)
      azure_foundry_deployment: str = ""
      azure_foundry_api_version: str = "2024-10-21"
      azure_foundry_timeout_s: float = 120.0
      ```
      Also add `"azure_foundry_api_key"` to the `exclude` set in `Settings.dump()`.

3. In `socratic-tutor-poc/src/socratic_tutor/providers/__init__.py`, add a branch in `_build`:
   ```python
   if provider_name == "azure_foundry":
       from socratic_tutor.providers.azure_foundry import AzureFoundryProvider

       return AzureFoundryProvider(
           endpoint=settings.azure_foundry_endpoint,
           deployment=settings.azure_foundry_deployment,
           api_key=settings.azure_foundry_api_key,
           api_version=settings.azure_foundry_api_version,
           timeout_s=settings.azure_foundry_timeout_s,
       )
   ```

4. In `socratic-tutor-poc/.env`, select it per role (any subset of the three) and set credentials:
   ```
   POC_GENERATION_PROVIDER=azure_foundry
   POC_INTENT_PROVIDER=azure_foundry
   POC_EVALUATION_PROVIDER=azure_foundry

   POC_AZURE_FOUNDRY_ENDPOINT=https://<your-resource-name>.openai.azure.com
   POC_AZURE_FOUNDRY_API_KEY=...
   POC_AZURE_FOUNDRY_DEPLOYMENT=<your-deployment-name>
   POC_AZURE_FOUNDRY_API_VERSION=2024-10-21
   ```
   You can mix providers per role — e.g. keep `claude_subscription` for
   `generation` and use `azure_foundry` only for `intent`/`evaluation`.

5. Verify: `uv run poc serve`, then exercise a chat turn and check the
   TurnTrace inspector shows `provider: azure_foundry` for the roles you swapped.

---

## socratic-tutor-poc — embeddings (no repo edits)

`socratic-tutor-poc`'s RAG store embeds locally via `sentence-transformers` /
`BAAI/bge-m3` (`domain/rag/embeddings.py`'s `BGEM3Embedder`). If your machine
can't download/run that model, you can redirect it to an Azure AI Foundry
embedding deployment **without touching that file**, by shadowing the
`sentence_transformers` import itself with `shims/sentence_transformers/` in
this folder — `BGEM3Embedder` calls `SentenceTransformer(...)`,
`.encode(...)`, `.get_sentence_embedding_dimension()` exactly as it always
does; it has no idea the implementation changed.

1. Deploy an embedding model in Azure AI Foundry (e.g. `text-embedding-3-small`)
   and note its deployment name.

2. Add to `socratic-tutor-poc/.env` (or export in the shell that runs it):
   ```
   AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT=<your-embedding-deployment-name>
   ```
   (Reuses `AZURE_FOUNDRY_ENDPOINT`/`API_KEY`/`API_VERSION`/`API_STYLE` from
   the chat setup above if you don't set embedding-specific overrides — see
   `.env.example` in this folder.)

3. Run the app with this folder's `shims/` directory prepended to
   `PYTHONPATH`, so `import sentence_transformers` resolves to the shim
   instead of (or in the absence of) the real package:
   ```bash
   cd socratic-tutor-poc
   PYTHONPATH="/home/sans/socratic_tutor/azure-foundry-adapter/shims:$PYTHONPATH" uv run poc serve
   ```
   Same for ingestion:
   ```bash
   PYTHONPATH="/home/sans/socratic_tutor/azure-foundry-adapter/shims:$PYTHONPATH" uv run poc ingest ...
   ```
   (exact ingest command may differ — check `uv run poc --help`).

**Dimension warning**: BGE-M3 produces 1024-dim vectors; Azure
`text-embedding-3-small` produces 1536 (3072 for `-large`, unless you pass a
`dimensions` override). A Chroma collection already populated with BGE-M3
vectors is incompatible with the new dimension — point `POC_DATA_DIR` (or
`POC_CHROMA_DIR`) at a fresh directory, or delete the existing
`data/chroma/`, before ingesting again under Azure embeddings.

**What this doesn't solve**: `uv sync` still installs the real
`sentence-transformers`/`torch` packages, since they're declared in
`socratic-tutor-poc/pyproject.toml` — the shim only stops the actual model
*weights* from being downloaded/run (that's usually the expensive/infeasible
part). If installing `torch` itself is the blocker on your machine, that
requires editing that `pyproject.toml` to drop the dependency, which is an
actual repo edit — say so if you want that instead.

---

## Foundry Models (unified endpoint) instead of classic Azure OpenAI

Everything above assumes a classic Azure OpenAI resource
(`https://<resource>.openai.azure.com`, deployment-in-URL). If instead you
provisioned an Azure AI Foundry resource using the unified **Foundry Models**
inference endpoint (`https://<resource>.services.ai.azure.com`), the
`kg_reasoner` path (via `azure_foundry_adapter.client`) already supports it —
set `AZURE_FOUNDRY_API_STYLE=foundry_models` in `.env`. The
`knowledge-graph-poc` adapter (via the `openai` SDK's `AzureOpenAI` class) and
the `socratic-tutor-poc` adapter (hand-rolled URL) both currently assume the
classic deployment-in-path shape — if you're on `foundry_models`, change
`knowledge_graph_poc.py`'s client construction to use
`openai.OpenAI(base_url=f"{endpoint}/models", ...)` with `model` in the request
body, and `socratic_tutor_poc.py`'s `base_url` to `f"{endpoint}/models"`
without appending `/openai/deployments/{deployment}`, sending `deployment` as
`model` in the JSON body instead.
