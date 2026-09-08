"""WU2 — `kg.config.adapter_factory(cfg)` (DESIGN §7.2, §17 "Config guards", §20 config.py;
D22, D26; PRD R21 "provider is selected in config").

WU0 deferred the factory; it returns the adapter for `llm.provider` without touching the network
or spawning anything. Unknown provider -> ConfigError; anthropic_api without the extra -> ImportError.
"""

from __future__ import annotations

import dataclasses
import sys

import pytest

from kg.config import ConfigError, adapter_factory
from kg.llm.adapters.anthropic_api import AnthropicApiAdapter
from kg.llm.adapters.base import Adapter
from kg.llm.adapters.claude_subscription import ClaudeSubscriptionAdapter
from kg.llm.adapters.openrouter import OpenRouterAdapter

from wu2_fakes import FAKE_OR_KEY, load_cfg


def test_R21_factory_returns_claude_subscription_adapter_for_default_config(make_project):
    cfg = load_cfg(make_project)  # provider claude_subscription
    adapter = adapter_factory(cfg)
    assert isinstance(adapter, ClaudeSubscriptionAdapter)
    assert isinstance(adapter, Adapter)
    assert adapter.provider == "claude_subscription"


def test_R21_factory_returns_openrouter_adapter_wired_from_config(make_project):
    cfg = load_cfg(make_project, "openrouter")
    adapter = adapter_factory(cfg)
    assert isinstance(adapter, OpenRouterAdapter)
    assert adapter.provider == "openrouter"
    assert str(adapter.client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    assert adapter.client.api_key == FAKE_OR_KEY
    assert adapter.client.max_retries == 3


def test_D26_factory_for_anthropic_api_without_the_extra_raises_import_error_naming_it(make_project, monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    cfg = load_cfg(make_project, "anthropic_api")
    with pytest.raises(ImportError, match="anthropic-api"):
        adapter_factory(cfg)


def test_D26_factory_returns_anthropic_api_adapter_when_a_client_can_be_built(make_project, monkeypatch):
    """With a stub `anthropic` module present the factory must build the adapter (no network)."""
    import types

    stub = types.ModuleType("anthropic")

    class Anthropic:  # minimal stand-in for anthropic.Anthropic
        def __init__(self, *a, **kw):
            self.kw = kw
            self.messages = None

    stub.Anthropic = Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", stub)
    cfg = load_cfg(make_project, "anthropic_api")
    adapter = adapter_factory(cfg)
    assert isinstance(adapter, AnthropicApiAdapter)
    assert adapter.provider == "anthropic_api"


def test_R21_factory_refuses_unknown_provider(make_project):
    cfg = load_cfg(make_project)
    bogus = dataclasses.replace(cfg, llm=dataclasses.replace(cfg.llm, provider="bedrock"))
    with pytest.raises(ConfigError) as ei:
        adapter_factory(bogus)
    assert "bedrock" in str(ei.value)


def test_D22_factory_refuses_claude_subscription_when_api_key_is_in_environment(make_project, monkeypatch):
    cfg = load_cfg(make_project)  # loaded while the env is clean
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TESTONLY-000")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        adapter_factory(cfg)


def test_R20_factory_never_spawns_or_connects(make_project):
    """Constructing every adapter must not trip conftest's zero-token poison."""
    for provider in ("claude_subscription", "openrouter"):
        adapter_factory(load_cfg(make_project, provider))  # would raise AssertionError if it touched the SDKs' entry points


def test_R21_factory_creates_the_agent_cwd_inside_the_sandboxed_runs_folder(make_project):
    cfg = load_cfg(make_project)
    adapter_factory(cfg)
    agent_cwd = (cfg.project_root / cfg.paths.runs / ".agent-cwd").resolve()
    assert agent_cwd.is_dir()
    assert not any(agent_cwd.rglob("CLAUDE.md")), "agent cwd must never contain a CLAUDE.md (D21)"
