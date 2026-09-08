"""Shared pytest fixtures for kg-mapper-poc (DESIGN §17, R20).

Everything here is offline: no network, no API keys, no tokens. The suite must
pass with ANTHROPIC_API_KEY / OPENROUTER_API_KEY unset; an autouse fixture
enforces that and restores the environment afterwards.

Fixture surface (WU0):
  clean_env        autouse — strips the same five variables `kg selftest` strips
                   (kg.config.CREDENTIAL_VARS + the three CLI/SDK token vars),
                   restores os.environ
  no_real_llm      autouse — monkeypatches claude_agent_sdk.query,
                   claude_agent_sdk.ClaudeSDKClient, openai.OpenAI.chat,
                   openai.AsyncOpenAI and openai.AsyncOpenAI.chat to raise if
                   ever touched (§17 zero-token guard)
  base_config()    helper — dict mirroring the shipped kg.yaml in DESIGN §3.2
  write_project()  helper — writes kg.yaml (+ optional .env) into a temp root
  project_root     fixture — temp project root with default kg.yaml + data/ dirs
  make_project     fixture — factory: make_project(overrides, env_text) -> kg.yaml path
  fake_adapter     fixture — FakeAdapter stub (see TODO below)

TODO(WU-llm): FakeOpenRouterAdapter and FakeClaudeSubscriptionAdapter scripted
fakes (DESIGN §17 rows "Repair-retry — OpenRouter path" / "Claude subscription
path" / "Rate-limit handling") land with the LLM boundary work unit. FakeAdapter
below is the fixture SURFACE only: it duck-types the Adapter protocol
(`complete(system, messages, schema_json, model, max_tokens)`) and records calls.
It deliberately does not import kg.llm.adapters.base so WU0 tests can run before
that module exists.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from kg.config import CREDENTIAL_VARS

STAGES = ("describe", "atomize", "edges", "dedup")
PATH_KINDS = ("inbox", "converted", "graph", "processed", "runs")
# kg.cli strips these in addition to CREDENTIAL_VARS (its tuple is module-private, so
# mirrored here; test_WU0_selftest_plumbing pins the two sets equal).
EXTRA_STRIPPED_VARS = ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY")
STRIPPED_VARS = tuple(CREDENTIAL_VARS) + EXTRA_STRIPPED_VARS


# --------------------------------------------------------------------------- env


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Selftest runs with every credential/token variable unset (DESIGN §17).

    Strips exactly what `kg selftest` strips from the pytest environment. Also
    snapshots and fully restores os.environ so a config loader that pushes
    `.env` values into the process environment cannot leak between tests.
    """
    saved = dict(os.environ)
    for var in STRIPPED_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch: pytest.MonkeyPatch):
    """Guard: no test may spawn the Claude CLI or open a socket to OpenRouter."""

    def _boom(*_a: Any, **_k: Any):  # pragma: no cover - only fires on a bug
        raise AssertionError("real LLM provider invoked during selftest (zero-token rule, DESIGN §17)")

    try:
        import claude_agent_sdk  # type: ignore

        monkeypatch.setattr(claude_agent_sdk, "query", _boom, raising=False)
        # ClaudeSDKClient is the other entry point (DESIGN §7.3). Poison the
        # class already imported by any adapter module, then the module attribute.
        client_cls = getattr(claude_agent_sdk, "ClaudeSDKClient", None)
        if isinstance(client_cls, type):
            monkeypatch.setattr(client_cls, "__init__", _boom, raising=False)
        monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _boom, raising=False)
    except ImportError:
        pass
    try:
        import openai  # type: ignore

        monkeypatch.setattr(openai.OpenAI, "chat", property(_boom), raising=False)
        # The OpenRouter adapter is async (DESIGN §7.4): poison AsyncOpenAI too —
        # `.chat` on the original class (covers `from openai import AsyncOpenAI`
        # done at import time) and the module attribute (covers `openai.AsyncOpenAI(...)`).
        async_cls = getattr(openai, "AsyncOpenAI", None)
        if isinstance(async_cls, type):
            monkeypatch.setattr(async_cls, "chat", property(_boom), raising=False)
        monkeypatch.setattr(openai, "AsyncOpenAI", _boom, raising=False)
    except ImportError:
        pass
    yield


# ------------------------------------------------------------------ config data


def base_config() -> dict[str, Any]:
    """A kg.yaml document equivalent to the shipped default in DESIGN §3.2."""
    return {
        "version": 2,
        "corpus": {"slug": "sample", "name": "Sample corpus", "schema": "general"},
        "paths": {
            "inbox": "data/inbox",
            "converted": "data/converted",
            "graph": "data/graph",
            "processed": "data/processed",
            "runs": "data/runs",
        },
        "llm": {
            "provider": "claude_subscription",
            "repair_retries": 2,
            "max_output_tokens": 8000,
            "stage_timeout_s": 300,
            "claude_subscription": {
                "models": {s: "claude-sonnet-5" for s in STAGES},
                "effort": "medium",
                "rate_limit_wait_minutes": 30,
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "models": {
                    "describe": "google/gemini-2.5-flash",
                    "atomize": "anthropic/claude-sonnet-5",
                    "edges": "anthropic/claude-sonnet-5",
                    "dedup": "openai/gpt-5-mini",
                },
                "provider": {
                    "require_parameters": True,
                    "data_collection": "deny",
                    "order": [],
                    "allow_fallbacks": True,
                },
                "app_title": "kg-mapper-poc",
            },
            "anthropic_api": {
                "models": {s: "claude-sonnet-5" for s in STAGES},
                "pricing_usd_per_mtok": {
                    "claude-opus-5": {"input": 5, "output": 25},
                    "claude-sonnet-5": {"input": 2, "output": 10},
                    "claude-haiku-4-5-20251001": {"input": 1, "output": 5},
                    "cache_read_multiplier": 0.10,
                    "cache_write_multiplier": 1.25,
                },
            },
        },
        "chunking": {"target_tokens": 3000, "max_tokens": 4000, "min_tokens": 150},
        "limits": {
            "max_nodes_per_chunk": 25,
            "max_dedup_pairs": 20,
            "roster_cap": 400,
            "soft_budget_usd": 2.00,
        },
        "dedup": {"threshold": 0.60, "write_same_as": False},
        "image": {"max_long_edge_px": 1568},
        "web": {"timeout_s": 20, "user_agent": "kg-mapper-poc/0.1"},
        "relevance_sample": {"n": 30, "bands": [[0, 39], [40, 70], [71, 100]], "seed": 1},
    }


def set_path(doc: dict[str, Any], dotted: str, value: Any) -> None:
    """Set doc['a']['b']['c'] = value for dotted == 'a.b.c'; creates intermediates."""
    keys = dotted.split(".")
    cur = doc
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def del_path(doc: dict[str, Any], dotted: str) -> None:
    keys = dotted.split(".")
    cur = doc
    for k in keys[:-1]:
        cur = cur[k]
    del cur[keys[-1]]


def write_project(root: Path, doc: dict[str, Any], env_text: str | None = None) -> Path:
    """Write kg.yaml (and optionally .env) under `root`; create data/ dirs. Returns kg.yaml path."""
    root.mkdir(parents=True, exist_ok=True)
    for rel in doc.get("paths", {}).values():
        if isinstance(rel, str) and not rel.startswith("/") and ".." not in rel:
            (root / rel).mkdir(parents=True, exist_ok=True)
    cfg_path = root / "kg.yaml"
    cfg_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    if env_text is not None:
        (root / ".env").write_text(env_text, encoding="utf-8")
    return cfg_path


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Temp project root holding the default kg.yaml and the five data/ folders."""
    write_project(tmp_path, base_config())
    return tmp_path


@pytest.fixture
def make_project(tmp_path: Path) -> Callable[..., Path]:
    """Factory: make_project({"llm.provider": "openrouter"}, env_text="...") -> kg.yaml path.

    `overrides` maps dotted keys to values; a value of `DELETE` removes the key.
    """

    def _make(overrides: dict[str, Any] | None = None, env_text: str | None = None, subdir: str = "proj") -> Path:
        doc = copy.deepcopy(base_config())
        for k, v in (overrides or {}).items():
            if v is DELETE:
                del_path(doc, k)
            else:
                set_path(doc, k, v)
        return write_project(tmp_path / subdir, doc, env_text)

    _make.DELETE = DELETE  # type: ignore[attr-defined]  # sentinel reachable without importing conftest
    return _make


class _Delete:
    def __repr__(self) -> str:  # pragma: no cover
        return "DELETE"


DELETE = _Delete()


# -------------------------------------------------------------- fake adapter


class FakeAdapter:
    """Minimal stand-in for the Adapter protocol (DESIGN §7.1).

    `complete()` returns whatever `canned` was given, in order, and records each
    call. Never spawns a process or opens a socket.

    TODO(WU-llm): replace the return type with kg.llm.adapters.base.RawCompletion
    once that module exists, and add FakeOpenRouterAdapter /
    FakeClaudeSubscriptionAdapter with the request-shape assertions from §17.
    """

    provider = "fake"

    def __init__(self, canned: list[Any] | None = None) -> None:
        self.canned = list(canned or [])
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, messages: list[Any], schema_json: dict, model: str, max_tokens: int) -> Any:
        self.calls.append(
            {"system": system, "messages": messages, "schema_json": schema_json, "model": model, "max_tokens": max_tokens}
        )
        if not self.canned:
            raise AssertionError("FakeAdapter exhausted: no canned response left")
        return self.canned.pop(0)


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    return FakeAdapter()
