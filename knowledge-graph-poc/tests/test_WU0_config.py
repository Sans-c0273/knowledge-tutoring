"""WU0 — kg.config: kg.yaml + .env loading, provider selection, credential guards.

Spec: PRD R18 (credentials in .env only; everything else in config; secrets never
written to output), R21 (provider selected in config; per-stage model IDs
configurable), R8 (config names exactly one schema), DESIGN §3.1 / §3.2 / §17
"Config guards", DECISIONS D22.

Names used (DESIGN §20 names config.py; the rest are the most obvious choices):
  kg.config.load_config(path_to_kg_yaml) -> Config
  kg.config.Config            .project_root, .corpus.schema, .paths.<kind>, .llm, .chunking, .limits,
                              .sandbox (kg.paths.Sandbox), .dump() -> redacted dict
  kg.config.Config.llm        .provider, .model_for(stage), .api_key (never in repr), .repair_retries
  kg.config.ConfigError       raised for every refusal below
  kg.config.STAGES            ("describe", "atomize", "edges", "dedup")
  kg.config.PROVIDERS         ("claude_subscription", "openrouter", "anthropic_api")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from kg.config import CREDENTIAL_VARS, PROVIDERS, STAGES, Config, ConfigError, load_config
from kg.paths import Sandbox, SandboxViolation

FAKE_OR_KEY = "sk-or-v1-TESTONLY-not-a-real-key-0123456789"
FAKE_ANTHROPIC_KEY = "sk-ant-TESTONLY-not-a-real-key-0123456789"


@pytest.fixture
def DELETE(make_project):
    """Sentinel understood by make_project overrides: remove this key from kg.yaml."""
    return make_project.DELETE


# ------------------------------------------------------------- constants


def test_R21_stage_names_are_the_four_model_stages():
    assert tuple(STAGES) == ("describe", "atomize", "edges", "dedup")


def test_R21_provider_vocabulary_is_exactly_three():
    assert set(PROVIDERS) == {"claude_subscription", "openrouter", "anthropic_api"}


# --------------------------------------------------------- happy loading


def test_R18_loads_default_config_from_kg_yaml(project_root: Path):
    cfg = load_config(project_root / "kg.yaml")
    assert isinstance(cfg, Config)
    assert cfg.project_root == project_root
    assert cfg.corpus.schema == "general"
    assert cfg.llm.provider == "claude_subscription"
    assert cfg.chunking.target_tokens == 3000
    assert cfg.limits.soft_budget_usd == pytest.approx(2.00)


def test_R18_config_exposes_a_sandbox_built_from_its_paths(project_root: Path):
    cfg = load_config(project_root / "kg.yaml")
    assert isinstance(cfg.sandbox, Sandbox)
    assert cfg.sandbox.resolve("inbox", "a.pdf") == project_root / "data/inbox/a.pdf"
    assert cfg.sandbox.resolve("runs", ".") == project_root / "data/runs"


def test_R8_schema_must_be_education_or_general(make_project):
    cfg = load_config(make_project({"corpus.schema": "education"}))
    assert cfg.corpus.schema == "education"
    with pytest.raises(ConfigError):
        load_config(make_project({"corpus.schema": "zettelkasten"}, subdir="bad"))


def test_R18_missing_config_file_is_a_config_error(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "kg.yaml")


def test_R18_malformed_yaml_is_a_config_error(tmp_path: Path):
    p = tmp_path / "kg.yaml"
    p.write_text("version: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_R18_unsupported_version_is_refused(make_project):
    with pytest.raises(ConfigError):
        load_config(make_project({"version": 1}))


def test_R18_missing_paths_key_is_refused(make_project, DELETE):
    with pytest.raises(ConfigError):
        load_config(make_project({"paths.processed": DELETE}))


def test_R18_absolute_path_in_config_is_refused(make_project, tmp_path: Path):
    # kg.yaml §3.2: "relative to the directory containing kg.yaml; absolute paths refused"
    with pytest.raises((ConfigError, SandboxViolation)):
        load_config(make_project({"paths.inbox": str(tmp_path / "proj/data/inbox")}))


def test_R18_path_escaping_project_root_is_refused(make_project):
    with pytest.raises((ConfigError, SandboxViolation)):
        load_config(make_project({"paths.graph": "../outside/graph"}))


def test_R18_path_touching_the_vault_is_refused(make_project):
    with pytest.raises((ConfigError, SandboxViolation)):
        load_config(make_project({"paths.graph": "TK-PKA/graph"}))


# ------------------------------------------------------ provider choice


@pytest.mark.parametrize("provider", ["claude_subscription", "openrouter", "anthropic_api"])
def test_R21_each_known_provider_is_accepted_when_its_credential_rule_is_met(make_project, monkeypatch, provider):
    if provider == "openrouter":
        monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OR_KEY)
    if provider == "anthropic_api":
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
    cfg = load_config(make_project({"llm.provider": provider}))
    assert cfg.llm.provider == provider


@pytest.mark.parametrize("bad", ["anthropic", "openai", "Claude_Subscription", "", None])
def test_R21_unknown_provider_is_refused(make_project, bad):
    with pytest.raises(ConfigError):
        load_config(make_project({"llm.provider": bad}))


def test_R21_missing_provider_key_is_refused(make_project, DELETE):
    with pytest.raises(ConfigError):
        load_config(make_project({"llm.provider": DELETE}))


# ------------------------------------------------- credential guards (D22)


def test_R21_openrouter_without_key_is_refused_and_names_the_variable(make_project):
    with pytest.raises(ConfigError) as ei:
        load_config(make_project({"llm.provider": "openrouter"}))
    assert "OPENROUTER_API_KEY" in str(ei.value)


def test_R21_openrouter_key_from_process_env_is_accepted(make_project, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OR_KEY)
    cfg = load_config(make_project({"llm.provider": "openrouter"}))
    assert cfg.llm.api_key == FAKE_OR_KEY


def test_R18_openrouter_key_from_dotenv_next_to_kg_yaml_is_accepted(make_project):
    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n"))
    assert cfg.llm.api_key == FAKE_OR_KEY


def test_R18_empty_dotenv_value_counts_as_missing(make_project):
    # `.env.example` ships `OPENROUTER_API_KEY=`; copying it unfilled must not pass the guard.
    with pytest.raises(ConfigError) as ei:
        load_config(make_project({"llm.provider": "openrouter"}, env_text="OPENROUTER_API_KEY=\n"))
    assert "OPENROUTER_API_KEY" in str(ei.value)


def test_R18_dotenv_values_are_read_literally_not_interpolated(make_project):
    # `.env` is a secrets file, not a template: `${BASE}` stays the literal string.
    env_text = f"BASE={FAKE_OR_KEY}\nOPENROUTER_API_KEY=${{BASE}}\n"
    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=env_text))
    assert cfg.llm.api_key == "${BASE}"
    assert FAKE_OR_KEY not in cfg.llm.api_key


def test_R18_dotenv_does_not_interpolate_from_the_process_environment(make_project, monkeypatch):
    monkeypatch.setenv("KG_TEST_UPSTREAM", FAKE_OR_KEY)
    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text="OPENROUTER_API_KEY=${KG_TEST_UPSTREAM}\n"))
    assert cfg.llm.api_key == "${KG_TEST_UPSTREAM}"


def test_R21_anthropic_api_without_key_is_refused_and_names_the_variable(make_project):
    with pytest.raises(ConfigError) as ei:
        load_config(make_project({"llm.provider": "anthropic_api"}))
    assert "ANTHROPIC_API_KEY" in str(ei.value)


def test_R21_anthropic_api_key_is_exposed_only_for_that_provider(make_project, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
    cfg = load_config(make_project({"llm.provider": "anthropic_api"}))
    assert cfg.llm.api_key == FAKE_ANTHROPIC_KEY


def test_D22_claude_subscription_refuses_to_run_when_ANTHROPIC_API_KEY_is_set(make_project, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
    with pytest.raises(ConfigError) as ei:
        load_config(make_project({"llm.provider": "claude_subscription"}))
    assert "ANTHROPIC_API_KEY" in str(ei.value)
    assert FAKE_ANTHROPIC_KEY not in str(ei.value)  # message must not echo the secret


def test_D22_claude_subscription_refuses_when_ANTHROPIC_API_KEY_comes_from_dotenv(make_project):
    with pytest.raises(ConfigError) as ei:
        load_config(make_project({"llm.provider": "claude_subscription"}, env_text=f"ANTHROPIC_API_KEY={FAKE_ANTHROPIC_KEY}\n"))
    assert "ANTHROPIC_API_KEY" in str(ei.value)


def test_D22_claude_subscription_needs_no_credential_and_exposes_none(make_project):
    cfg = load_config(make_project({"llm.provider": "claude_subscription"}))
    assert cfg.llm.api_key is None


def test_D22_claude_subscription_is_not_blocked_by_an_openrouter_key(make_project, monkeypatch):
    # Only ANTHROPIC_API_KEY can mis-route the Claude CLI; an OpenRouter key is irrelevant here.
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OR_KEY)
    cfg = load_config(make_project({"llm.provider": "claude_subscription"}))
    assert cfg.llm.provider == "claude_subscription"


# ------------------------------------------------- per-stage model maps


def test_R21_model_for_each_stage_follows_the_selected_provider_map(make_project, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OR_KEY)
    cfg = load_config(make_project({"llm.provider": "openrouter"}))
    assert cfg.llm.model_for("describe") == "google/gemini-2.5-flash"
    assert cfg.llm.model_for("atomize") == "anthropic/claude-sonnet-5"
    assert cfg.llm.model_for("edges") == "anthropic/claude-sonnet-5"
    assert cfg.llm.model_for("dedup") == "openai/gpt-5-mini"


def test_R21_switching_provider_switches_the_model_map(make_project):
    cfg = load_config(make_project({"llm.provider": "claude_subscription"}))
    assert {cfg.llm.model_for(s) for s in STAGES} == {"claude-sonnet-5"}


def test_R21_per_stage_override_in_yaml_is_honoured(make_project):
    cfg = load_config(make_project({"llm.claude_subscription.models.edges": "claude-opus-5"}))
    assert cfg.llm.model_for("edges") == "claude-opus-5"
    assert cfg.llm.model_for("atomize") == "claude-sonnet-5"


def test_R21_missing_stage_in_selected_provider_map_is_refused(make_project, DELETE):
    with pytest.raises(ConfigError) as ei:
        load_config(make_project({"llm.claude_subscription.models.dedup": DELETE}))
    assert "dedup" in str(ei.value)


def test_R21_missing_stage_in_an_unselected_provider_map_does_not_block(make_project, DELETE):
    # Only the selected provider's map must be complete; a partially filled other block is fine.
    cfg = load_config(make_project({"llm.openrouter.models.dedup": DELETE}))
    assert cfg.llm.provider == "claude_subscription"


def test_R21_unknown_stage_lookup_is_an_error(make_project):
    cfg = load_config(make_project())
    with pytest.raises((ConfigError, KeyError)):
        cfg.llm.model_for("summarise")


def test_R21_repair_retries_default_and_override(make_project):
    assert load_config(make_project()).llm.repair_retries == 2
    assert load_config(make_project({"llm.repair_retries": 0}, subdir="zero")).llm.repair_retries == 0


# ------------------------------------------------------ secret hygiene


def _all_text_views(cfg: Config) -> list[str]:
    views = [repr(cfg), str(cfg), repr(cfg.llm), str(cfg.llm), json.dumps(cfg.dump(), default=str)]
    return views


def test_R18_secret_never_appears_in_repr_str_or_dump_openrouter(make_project, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OR_KEY)
    cfg = load_config(make_project({"llm.provider": "openrouter"}))
    for view in _all_text_views(cfg):
        assert FAKE_OR_KEY not in view
        assert "TESTONLY" not in view


def test_R18_secret_never_appears_in_repr_str_or_dump_anthropic_api(make_project, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
    cfg = load_config(make_project({"llm.provider": "anthropic_api"}))
    for view in _all_text_views(cfg):
        assert FAKE_ANTHROPIC_KEY not in view
        assert "TESTONLY" not in view


def test_R18_dump_is_json_serialisable_and_names_provider_and_models(make_project):
    cfg = load_config(make_project())
    d = cfg.dump()
    text = json.dumps(d)  # must not raise
    assert '"claude_subscription"' in text
    assert "claude-sonnet-5" in text
    assert not re.search(r"sk-(or|ant)-", text)


def test_R18_dotenv_secret_never_appears_in_dump(make_project):
    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n"))
    assert FAKE_OR_KEY not in json.dumps(cfg.dump(), default=str)


# -------------------------------------------- dump()["credentials"] truthfulness


def _credentials(cfg: Config) -> dict[str, str]:
    creds = cfg.dump()["credentials"]
    assert set(creds) == set(CREDENTIAL_VARS), "every credential variable is reported, selected provider or not"
    assert set(creds.values()) <= {"present", "absent"}, f"values must be present/absent markers, got {creds}"
    return creds


def test_R18_dump_credentials_both_absent(make_project):
    creds = _credentials(load_config(make_project({"llm.provider": "claude_subscription"})))
    assert creds == {"OPENROUTER_API_KEY": "absent", "ANTHROPIC_API_KEY": "absent"}


def test_R18_dump_credentials_both_present_from_env(make_project, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OR_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
    cfg = load_config(make_project({"llm.provider": "openrouter"}))
    creds = _credentials(cfg)
    assert creds == {"OPENROUTER_API_KEY": "present", "ANTHROPIC_API_KEY": "present"}
    text = json.dumps(cfg.dump(), default=str)
    assert FAKE_OR_KEY not in text and FAKE_ANTHROPIC_KEY not in text and "TESTONLY" not in text


def test_R18_dump_credentials_mixed_dotenv_present_other_absent(make_project):
    cfg = load_config(make_project({"llm.provider": "openrouter"}, env_text=f"OPENROUTER_API_KEY={FAKE_OR_KEY}\n"))
    assert _credentials(cfg) == {"OPENROUTER_API_KEY": "present", "ANTHROPIC_API_KEY": "absent"}


def test_R18_dump_credentials_unselected_provider_key_is_still_reported(make_project, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OR_KEY)
    cfg = load_config(make_project({"llm.provider": "anthropic_api"}))
    assert _credentials(cfg)["OPENROUTER_API_KEY"] == "present"


def test_R18_dump_credentials_empty_value_reports_absent(make_project, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    cfg = load_config(make_project({"llm.provider": "claude_subscription"}))
    assert _credentials(cfg)["OPENROUTER_API_KEY"] == "absent"
