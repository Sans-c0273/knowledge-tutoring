"""WU0 — self-test plumbing (PRD R20; DESIGN §17, §18, §20).

`kg selftest` wraps pytest and spends zero tokens. This file checks the wiring:
pyproject exposes the `kg` console script, the CLI module is runnable, `selftest`
is a subcommand, and the conftest fixture surface exists.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    assert PYPROJECT.exists(), "pyproject.toml must exist at the repo root"
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


# ------------------------------------------------------------- pyproject


def test_R20_console_script_kg_points_at_cli_main(pyproject):
    scripts = pyproject["project"]["scripts"]
    assert scripts["kg"] == "kg.cli:main"


def test_D2_requires_python_3_12_window(pyproject):
    rp = pyproject["project"]["requires-python"]
    assert ">=3.12" in rp.replace(" ", "")
    assert "<3.14" in rp.replace(" ", "")


def test_R20_pytest_configured_to_run_tests_dir(pyproject):
    ini = pyproject["tool"]["pytest"]["ini_options"]
    assert "tests" in ini["testpaths"]


def test_R21_core_dependencies_declared(pyproject):
    deps = " ".join(pyproject["project"]["dependencies"]).lower()
    for name in ("pydantic", "pyyaml", "python-dotenv", "claude-agent-sdk", "openai"):
        assert name in deps, f"{name} missing from [project.dependencies]"


def test_D26_anthropic_api_is_an_optional_extra_not_a_core_dependency(pyproject):
    core = [d.lower() for d in pyproject["project"]["dependencies"]]
    assert not any(d.split("[")[0].split(">")[0].split("=")[0].strip() == "anthropic" for d in core)
    extras = pyproject["project"]["optional-dependencies"]
    assert "anthropic-api" in extras
    assert any(d.lower().startswith("anthropic") for d in extras["anthropic-api"])


def test_R20_package_lives_under_src_kg(pyproject):
    assert (REPO / "src" / "kg" / "__init__.py").exists()
    assert (REPO / "src" / "kg" / "cli.py").exists()


# ------------------------------------------------------------------ CLI


def test_R20_cli_module_exposes_main():
    from kg.cli import main

    assert callable(main)


def test_R20_cli_selftest_help_exits_zero_in_process(capsys):
    from kg.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["selftest", "--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "selftest" in out


def test_R20_cli_lists_selftest_among_subcommands(capsys):
    from kg.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("ingest", "gates", "render", "stability", "sample-relevance", "selftest"):
        assert cmd in out, f"subcommand {cmd} missing from `kg --help`"


def test_R20_cli_unknown_subcommand_exits_nonzero(capsys):
    from kg.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["frobnicate"])
    assert ei.value.code != 0


# ------------------------------------------- selftest → pytest forwarding


@pytest.fixture
def pytest_runs(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the subprocess runner in kg.cli; record argv/env, report exit code 0."""
    import kg.cli

    calls: list[dict] = []

    def fake_run(cmd, **kw):
        calls.append({"cmd": list(cmd), **kw})
        return subprocess.CompletedProcess(list(cmd), 0)

    monkeypatch.setattr(kg.cli.subprocess, "run", fake_run)
    return calls


def _pytest_argv(call: dict) -> list[str]:
    cmd = call["cmd"]
    assert cmd[:3] == [sys.executable, "-m", "pytest"], cmd
    assert str(REPO / "tests") in cmd, "selftest must point pytest at the repo's tests/ directory"
    return cmd[3:]


def test_R20_selftest_forwards_dash_k_expression_to_pytest(pytest_runs):
    from kg.cli import main

    assert main(["selftest", "-k", "sandbox"]) == 0
    assert len(pytest_runs) == 1
    argv = _pytest_argv(pytest_runs[0])
    assert argv[-2:] == ["-k", "sandbox"]


def test_R20_selftest_forwards_args_after_double_dash_without_the_dash_dash(pytest_runs):
    from kg.cli import main

    assert main(["selftest", "--", "-x"]) == 0
    argv = _pytest_argv(pytest_runs[0])
    assert argv[-1] == "-x"
    assert "--" not in argv, "the `--` separator itself must not reach pytest"


def test_R20_selftest_with_no_extra_args_passes_only_the_tests_dir(pytest_runs):
    from kg.cli import main

    assert main(["selftest"]) == 0
    argv = _pytest_argv(pytest_runs[0])
    assert argv[-1] == str(REPO / "tests"), argv


def test_R20_selftest_propagates_pytest_exit_code(monkeypatch):
    import kg.cli

    monkeypatch.setattr(kg.cli.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(list(cmd), 3))
    assert kg.cli.main(["selftest"]) == 3


def test_R20_selftest_strips_every_credential_and_token_var_from_the_pytest_env(pytest_runs, monkeypatch):
    from conftest import STRIPPED_VARS

    from kg.cli import main

    for var in STRIPPED_VARS:
        monkeypatch.setenv(var, "leaked-TESTONLY-value")
    monkeypatch.setenv("KG_SELFTEST_SENTINEL", "keep-me")
    main(["selftest"])
    env = pytest_runs[0]["env"]
    assert env is not None
    leaked = [v for v in STRIPPED_VARS if v in env]
    assert leaked == [], f"credential/token vars reached pytest: {leaked}"
    assert env.get("KG_SELFTEST_SENTINEL") == "keep-me", "unrelated variables must survive"


def test_R20_conftest_clean_env_strips_the_same_vars_as_the_cli():
    from conftest import STRIPPED_VARS

    import kg.cli
    from kg.config import CREDENTIAL_VARS

    cli_stripped = set(CREDENTIAL_VARS) | set(getattr(kg.cli, "_EXTRA_STRIPPED_VARS", ()))
    assert set(STRIPPED_VARS) == cli_stripped, "tests/conftest.py must strip exactly what `kg selftest` strips"
    assert len(STRIPPED_VARS) == 5


def test_R20_python_dash_m_kg_cli_selftest_help_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "kg.cli", "selftest", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "selftest" in proc.stdout


# ------------------------------------------------------- fixture surface


def test_R20_conftest_fake_adapter_surface(fake_adapter):
    assert hasattr(fake_adapter, "complete")
    fake_adapter.canned.append({"ok": True})
    out = fake_adapter.complete("sys", [], {"type": "object"}, "fake-model", 100)
    assert out == {"ok": True}
    assert len(fake_adapter.calls) == 1
    assert fake_adapter.calls[0]["model"] == "fake-model"


def test_R20_conftest_project_root_has_default_config_and_folders(project_root: Path):
    assert (project_root / "kg.yaml").exists()
    for kind in ("inbox", "converted", "graph", "processed", "runs"):
        assert (project_root / "data" / kind).is_dir()


def test_R20_suite_runs_with_no_credentials_in_environment():
    import os

    from conftest import STRIPPED_VARS

    present = [v for v in STRIPPED_VARS if v in os.environ]
    assert present == [], f"clean_env must strip {STRIPPED_VARS}; still set: {present}"
