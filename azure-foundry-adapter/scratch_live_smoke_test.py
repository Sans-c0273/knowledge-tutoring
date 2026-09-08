"""Live smoke test against the REAL Azure endpoint in .env — makes 3 real API calls."""
import os
import sys

sys.path.insert(0, "shims")

from azure_foundry_adapter.client import AzureFoundrySettings, complete_sync, embed_sync, first_message_content

def env(name, fallback=None, default=""):
    v = os.environ.get(name)
    if v:
        return v
    if fallback:
        return os.environ.get(fallback, default)
    return default

chat_settings = AzureFoundrySettings(
    endpoint=os.environ["AZURE_FOUNDRY_ENDPOINT"],
    deployment=os.environ["AZURE_FOUNDRY_DEPLOYMENT"],
    api_key=os.environ["AZURE_FOUNDRY_API_KEY"],
    api_version=os.environ["AZURE_FOUNDRY_API_VERSION"],
    api_style=os.environ.get("AZURE_FOUNDRY_API_STYLE", "azure_openai"),
)

emb_settings = AzureFoundrySettings(
    endpoint=env("AZURE_FOUNDRY_EMBEDDING_ENDPOINT", "AZURE_FOUNDRY_ENDPOINT"),
    deployment=os.environ["AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT"],
    api_key=env("AZURE_FOUNDRY_EMBEDDING_API_KEY", "AZURE_FOUNDRY_API_KEY"),
    api_version=env("AZURE_FOUNDRY_EMBEDDING_API_VERSION", "AZURE_FOUNDRY_API_VERSION"),
    api_style=env("AZURE_FOUNDRY_EMBEDDING_API_STYLE", "AZURE_FOUNDRY_API_STYLE", "azure_openai"),
)

print(f"1) Plain chat completion -> deployment={chat_settings.deployment!r} api_version={chat_settings.api_version!r}")
try:
    data = complete_sync(
        chat_settings,
        system="Reply with exactly one word.",
        messages=[{"role": "user", "content": "Say 'pong'."}],
        max_tokens=10,
    )
    print("   OK ->", first_message_content(data).strip())
except Exception as exc:
    print("   FAILED:", type(exc).__name__, exc)
    raise SystemExit(1)

print(f"2) Strict JSON-schema completion (what the real adapters use)")
schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}
try:
    data = complete_sync(
        chat_settings,
        system="Reply with JSON only.",
        messages=[{"role": "user", "content": "Return {\"ok\": true}"}],
        max_tokens=20,
        schema_json=schema,
        schema_name="smoke_test",
    )
    print("   OK ->", first_message_content(data))
except Exception as exc:
    print("   FAILED:", type(exc).__name__, exc)
    raise SystemExit(1)

print(f"3) Embeddings -> deployment={emb_settings.deployment!r} api_version={emb_settings.api_version!r}")
try:
    data = embed_sync(emb_settings, ["hello world"])
    vec = data["data"][0]["embedding"]
    print(f"   OK -> dimension={len(vec)}, first 3 values={vec[:3]}")
except Exception as exc:
    print("   FAILED:", type(exc).__name__, exc)
    raise SystemExit(1)

print("\nALL LIVE CHECKS PASSED")
