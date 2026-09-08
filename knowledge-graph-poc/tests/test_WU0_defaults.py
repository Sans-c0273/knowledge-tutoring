"""WU0 — shipped defaults are small-first and secret-free (PRD R18, R19; DESIGN §3.2, §20).

Reads the files at the repo root: kg.yaml, .env.example, .gitignore, .python-version.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
KG_YAML = REPO / "kg.yaml"
ENV_EXAMPLE = REPO / ".env.example"
GITIGNORE = REPO / ".gitignore"
PY_VERSION = REPO / ".python-version"

STAGES = ("describe", "atomize", "edges", "dedup")


@pytest.fixture(scope="module")
def shipped() -> dict:
    assert KG_YAML.exists(), "kg.yaml must ship at the repo root (DESIGN §20)"
    return yaml.safe_load(KG_YAML.read_text(encoding="utf-8"))


# ------------------------------------------------------------- kg.yaml


def test_R19_shipped_config_is_version_2(shipped):
    assert shipped["version"] == 2


def test_R21_shipped_provider_is_azure_foundry(shipped):
    # This is the `socratic-tutor-azure` fork's wired default (see azure-foundry-adapter/);
    # the upstream kg-mapper-poc project ships `claude_subscription` instead.
    assert shipped["llm"]["provider"] == "azure_foundry"


def test_R8_shipped_schema_is_one_of_the_two(shipped):
    assert shipped["corpus"]["schema"] in {"education", "general"}


def test_R18_shipped_paths_are_the_five_kinds_all_relative_under_data(shipped):
    paths = shipped["paths"]
    assert set(paths) == {"inbox", "converted", "graph", "processed", "runs"}
    for kind, rel in paths.items():
        assert not rel.startswith("/"), f"{kind}: absolute path"
        assert ".." not in rel, f"{kind}: escapes project"
        assert rel.startswith("data/"), f"{kind}: not under data/"
        assert "TK-PKA" not in rel and "01-knowledge-base" not in rel


def test_R19_shipped_chunk_sizes_match_design_3_2(shipped):
    assert shipped["chunking"] == {"target_tokens": 3000, "max_tokens": 4000, "min_tokens": 150}


def test_R19_shipped_limits_are_small_first(shipped):
    limits = shipped["limits"]
    assert limits["soft_budget_usd"] <= 2.00
    assert limits["max_nodes_per_chunk"] <= 25
    assert limits["max_dedup_pairs"] <= 20
    assert limits["roster_cap"] <= 400


def test_R21_shipped_config_has_a_full_model_map_for_every_provider(shipped):
    llm = shipped["llm"]
    for provider in ("claude_subscription", "openrouter", "anthropic_api"):
        models = llm[provider]["models"]
        assert set(models) == set(STAGES), provider
        assert all(isinstance(v, str) and v for v in models.values()), provider


def test_R21_shipped_openrouter_block_routes_only_to_structured_output_upstreams(shipped):
    orb = shipped["llm"]["openrouter"]
    assert orb["base_url"] == "https://openrouter.ai/api/v1"
    assert orb["provider"]["require_parameters"] is True
    assert orb["provider"]["data_collection"] == "deny"


def test_R21_shipped_repair_retries_is_bounded(shipped):
    assert 0 <= shipped["llm"]["repair_retries"] <= 3


def test_R18_shipped_config_contains_no_credential(shipped):
    text = KG_YAML.read_text(encoding="utf-8")
    assert not re.search(r"sk-(or|ant)-[A-Za-z0-9_-]{8,}", text)
    # Keys live in .env, never as kg.yaml values (comments may mention them).
    code_only = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    assert "OPENROUTER_API_KEY" not in code_only
    assert "ANTHROPIC_API_KEY" not in code_only


def test_R18_shipped_config_loads_through_kg_config():
    from kg.config import load_config

    # This fork ships azure_foundry as the default provider, which needs a
    # credential to load (unlike claude_subscription's OAuth-only flow) — this
    # repo's own .env supplies AZURE_FOUNDRY_API_KEY, so the load succeeds and
    # returns a real (non-None) api_key.
    cfg = load_config(KG_YAML)
    assert cfg.llm.provider == "azure_foundry"
    assert cfg.llm.api_key is not None
    assert cfg.sandbox.resolve("inbox", ".") == REPO / "data/inbox"


# --------------------------------------------------------- .env.example


def test_R18_env_example_exists_with_openrouter_placeholder():
    assert ENV_EXAMPLE.exists()
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert re.search(r"^OPENROUTER_API_KEY=\s*$", text, re.MULTILINE), "placeholder must be an empty assignment"


def test_R18_env_example_has_no_real_secret():
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        key, _, value = s.partition("=")
        assert key.strip() in {"OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "AZURE_FOUNDRY_API_KEY"}, f"unexpected key in .env.example: {key}"
        assert value.strip() == "", f"{key} must be empty in .env.example"


def test_D22_env_example_documents_anthropic_key_only_for_anthropic_api():
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" in text
    assert "anthropic_api" in text  # comment explaining when to set it
    # If present as an assignment it must be commented out, so copying the file
    # verbatim cannot trip the claude_subscription guard.
    for line in text.splitlines():
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            pytest.fail("ANTHROPIC_API_KEY must be commented out in .env.example (D22)")


# ----------------------------------------------- .gitignore / python pin


def test_R18_gitignore_excludes_env_and_runtime_data():
    assert GITIGNORE.exists()
    lines = {l.strip() for l in GITIGNORE.read_text(encoding="utf-8").splitlines()}
    assert ".env" in lines
    assert any(l in lines for l in ("data/", "data/**", "data")), ".gitignore must exclude data/"
    assert any(l in lines for l in (".venv/", ".venv"))


def test_D2_python_version_pinned_to_3_12():
    assert PY_VERSION.exists()
    assert PY_VERSION.read_text(encoding="utf-8").strip().startswith("3.12")
