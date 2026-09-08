"""Functional test of the sentence_transformers shim, mimicking exactly what
BGEM3Embedder does (import shape, call shape), against a mocked transport —
no real network / credentials."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "shims"))

import httpx

os.environ["AZURE_FOUNDRY_ENDPOINT"] = "https://fake.openai.azure.com"
os.environ["AZURE_FOUNDRY_API_KEY"] = "fake-key"
os.environ["AZURE_FOUNDRY_EMBEDDING_DEPLOYMENT"] = "text-embedding-3-small"

EMB_JSON = {
    "data": [
        {"embedding": [0.1, 0.2, 0.3], "index": 0},
        {"embedding": [0.4, 0.5, 0.6], "index": 1},
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 4, "total_tokens": 4},
}


def fake_transport(request: httpx.Request) -> httpx.Response:
    import json as _json

    assert request.headers.get("api-key") == "fake-key", request.headers
    assert request.url.params.get("api-version"), str(request.url)
    assert "/openai/deployments/text-embedding-3-small/embeddings" in str(request.url), str(request.url)
    body = _json.loads(request.read())
    n = len(body["input"])
    data = [{"embedding": [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)], "index": i} for i in range(n)]
    return httpx.Response(200, json={"data": data, "model": "text-embedding-3-small"})


def test_matches_bgem3_embedder_call_shape():
    # exact import/call shape BGEM3Embedder._load()/embed_texts()/dimension use:
    from sentence_transformers import SentenceTransformer  # resolves to our shim via sys.path

    model = SentenceTransformer("BAAI/bge-m3", device=None)
    model._client = httpx.Client(transport=httpx.MockTransport(fake_transport))  # test hook

    vectors = model.encode(
        ["hello", "world"], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    )
    assert len(vectors) == 2
    assert all(abs(sum(v * v for v in vec) - 1.0) < 1e-9 for vec in vectors)  # unit-normalized
    print("test_matches_bgem3_embedder_call_shape OK:", vectors)

    dim = model.get_sentence_embedding_dimension()
    assert dim == 3
    print("test_dimension OK:", dim)


def test_single_string_encode():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3")
    model._client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0], "index": 0}]})
        )
    )
    vec = model.encode("solo query", normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    assert isinstance(vec, list) and isinstance(vec[0], float)
    print("test_single_string_encode OK:", vec)


def test_fallback_env_vars():
    from sentence_transformers import _settings_from_env

    settings = _settings_from_env()
    assert settings.endpoint == "https://fake.openai.azure.com"  # fell back from AZURE_FOUNDRY_ENDPOINT
    assert settings.api_key == "fake-key"  # fell back from AZURE_FOUNDRY_API_KEY
    assert settings.deployment == "text-embedding-3-small"  # its own required var
    print("test_fallback_env_vars OK")


if __name__ == "__main__":
    test_matches_bgem3_embedder_call_shape()
    test_single_string_encode()
    test_fallback_env_vars()
    print("ALL OK")
