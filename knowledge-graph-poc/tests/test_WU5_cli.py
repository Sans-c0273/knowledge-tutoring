"""WU4/WU5 — `kg` CLI dispatch for `render`, `gates`, `stability`, `sample-relevance`, `summarise-relevance` (DESIGN §18, §23).

The command modules are monkeypatched: no pipeline runs, no model calls. Each command must
(1) resolve its paths from the sandbox of the loaded config, (2) print where it wrote, (3) exit 0
on success and non-zero on failure, and (4) refuse anything that would leave the sandbox.

Chosen: `kg sample-relevance --n N` → n_per_band = ceil(N / len(bands)); `--seed` defaults to
kg.yaml `relevance_sample.seed`; `kg stability --provider` is an argparse error (exit 2, §15).
"""

from __future__ import annotations

import math
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kg.cli import main
from kg.config import load_config
from kg.schemas import EDUCATION, GENERAL
from wu45_fixtures import RUN_ID


@pytest.fixture
def cfg_path(make_project) -> Path:
    return make_project()


@pytest.fixture
def cfg(cfg_path):
    return load_config(cfg_path)


def _install(monkeypatch, dotted: str, **attrs):
    """Import (or synthesize, if the module is not built yet) `dotted` and set attributes on it.

    Synthesising keeps the *dispatch* under test even before a module lands; the module's own
    tests cover its behaviour. The CLI must import lazily inside handlers (cli.py docstring).
    """
    try:
        mod = __import__(dotted, fromlist=["_"])
    except ImportError:
        mod = types.ModuleType(dotted)
        monkeypatch.setitem(sys.modules, dotted, mod)
        parent, _, child = dotted.rpartition(".")
        if parent:
            pmod = __import__(parent, fromlist=["_"])
            monkeypatch.setattr(pmod, child, mod, raising=False)
    for k, v in attrs.items():
        monkeypatch.setattr(mod, k, v, raising=False)
    return mod


class Recorder:
    def __init__(self, result=None):
        self.calls: list[tuple[tuple, dict]] = []
        self.result = result

    def __call__(self, *a, **kw):
        self.calls.append((a, kw))
        return self.result(*a, **kw) if callable(self.result) else self.result


# ------------------------------------------------------------------ render


def test_S18_render_calls_the_renderer_on_the_sandboxed_graph_and_prints_the_output_path(cfg_path, cfg, monkeypatch, capsys):
    out_path = cfg.sandbox.root("graph") / "graph.html"
    rec = Recorder(lambda graph_dir, out, **kw: out)
    _install(monkeypatch, "kg.render.html", render=rec)

    rc = main(["render", "--config", str(cfg_path)])
    assert rc == 0
    assert len(rec.calls) == 1
    (graph_dir, out), kw = rec.calls[0]
    assert Path(graph_dir) == cfg.sandbox.root("graph")
    assert Path(out) == out_path
    assert kw["schema"] is GENERAL
    assert str(out_path) in capsys.readouterr().out


def test_S18_render_passes_the_schema_from_config(make_project, monkeypatch):
    p = make_project({"corpus.schema": "education"}, subdir="edu")
    rec = Recorder(lambda graph_dir, out, **kw: out)
    _install(monkeypatch, "kg.render.html", render=rec)
    assert main(["render", "--config", str(p)]) == 0
    assert rec.calls[0][1]["schema"] is EDUCATION


def test_S18_render_reports_a_renderer_failure_as_nonzero_exit(cfg_path, monkeypatch, capsys):
    def boom(*a, **kw):
        raise RuntimeError("template missing")

    _install(monkeypatch, "kg.render.html", render=boom)
    rc = main(["render", "--config", str(cfg_path)])
    assert rc != 0
    assert "template missing" in capsys.readouterr().err


def test_S18_render_with_a_bad_config_path_fails_cleanly(tmp_path, capsys):
    rc = main(["render", "--config", str(tmp_path / "missing.yaml")])
    assert rc != 0
    assert "missing.yaml" in capsys.readouterr().err


# ------------------------------------------------------------------- gates


def test_S18_gates_validates_notes_on_disk_and_exits_zero_when_clean(cfg_path, cfg, monkeypatch, capsys):
    rec = Recorder([])
    _install(monkeypatch, "kg.gates", validate_graph=rec)
    rc = main(["gates", "--config", str(cfg_path)])
    assert rc == 0
    (graph_dir, schema), _ = rec.calls[0]
    assert Path(graph_dir) == cfg.sandbox.root("graph") and schema is GENERAL
    assert "0" in capsys.readouterr().out


def test_S18_gates_exits_nonzero_and_lists_findings_when_a_gate_fails(cfg_path, monkeypatch, capsys):
    finding = SimpleNamespace(gate="dag", reason="prerequisite_of cycle: sample-0001 -> sample-0002 -> sample-0001", edge=None)
    _install(monkeypatch, "kg.gates", validate_graph=Recorder([finding]))
    rc = main(["gates", "--config", str(cfg_path)])
    assert rc == 1
    out = capsys.readouterr()
    assert "dag" in out.out + out.err and "sample-0001 -> sample-0002" in out.out + out.err


def test_R7_gates_never_constructs_a_provider_adapter(cfg_path, monkeypatch):
    import kg.config as kc

    def no_adapter(*a, **kw):
        raise AssertionError("kg gates must not construct an adapter (DESIGN §2)")

    monkeypatch.setattr(kc, "adapter_factory", no_adapter, raising=False)
    _install(monkeypatch, "kg.gates", validate_graph=Recorder([]))
    assert main(["gates", "--config", str(cfg_path)]) == 0


# --------------------------------------------------------------- stability


def stability_result(cfg, **over):
    root = cfg.sandbox.root("runs") / f"stability-{RUN_ID}"
    base = dict(
        run_id=RUN_ID,
        root=root,
        jaccard=0.75,
        overlap_a=0.8,
        overlap_b=0.9,
        fuzzy_jaccard=0.8,
        passed=True,
        threshold=0.70,
        valid=True,
        invalid_reasons=[],
        only_in_a=[],
        only_in_b=[],
        common=["a"],
        relevance_deltas=SimpleNamespace(n=0, histogram={}, mean_abs=None, median_abs=None),
        legs={},
        report_md=root / "stability-report.md",
        report_json=root / "stability-report.json",
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_S18_stability_runs_two_legs_via_the_module_with_the_real_call_stage_and_prints_score_and_report(cfg_path, cfg, monkeypatch, capsys):
    fake_call_stage = object()
    _install(monkeypatch, "kg.llm", call_stage=fake_call_stage)
    rec = Recorder(stability_result(cfg))
    _install(monkeypatch, "kg.stability", run=rec)

    rc = main(["stability", "--config", str(cfg_path)])
    assert rc == 0
    (got_cfg,), kw = rec.calls[0]
    assert got_cfg.config_path == cfg.config_path
    assert kw["call_stage"] is fake_call_stage
    assert isinstance(kw["now"], datetime) and kw["now"].tzinfo is not None, "timestamp injected as an aware datetime"
    assert kw.get("keep", False) is False
    out = capsys.readouterr().out
    assert "0.75" in out and "stability-report.md" in out


def test_S18_stability_keep_flag_is_forwarded(cfg_path, cfg, monkeypatch):
    _install(monkeypatch, "kg.llm", call_stage=object())
    rec = Recorder(stability_result(cfg))
    _install(monkeypatch, "kg.stability", run=rec)
    assert main(["stability", "--keep", "--config", str(cfg_path)]) == 0
    assert rec.calls[0][1]["keep"] is True


def test_S15_stability_refuses_a_provider_override(cfg_path, cfg, monkeypatch, capsys):
    rec = Recorder(stability_result(cfg))
    _install(monkeypatch, "kg.stability", run=rec)
    with pytest.raises(SystemExit) as exc:
        main(["stability", "--provider", "openrouter", "--config", str(cfg_path)])
    assert exc.value.code == 2
    assert rec.calls == []
    assert "provider" in capsys.readouterr().err


def test_S18_stability_exit_code_reflects_a_failed_or_invalid_comparison(cfg_path, cfg, monkeypatch, capsys):
    _install(monkeypatch, "kg.llm", call_stage=object())
    _install(monkeypatch, "kg.stability", run=Recorder(stability_result(cfg, jaccard=0.4, passed=False)))
    assert main(["stability", "--config", str(cfg_path)]) == 1
    _install(monkeypatch, "kg.stability", run=Recorder(stability_result(cfg, valid=False, invalid_reasons=["leg b deferred a.md"])))
    assert main(["stability", "--config", str(cfg_path)]) == 1
    assert "INVALID" in capsys.readouterr().out


def test_S18_stability_on_the_subscription_provider_warns_about_the_shared_window(cfg_path, cfg, monkeypatch, capsys):
    _install(monkeypatch, "kg.llm", call_stage=object())
    _install(monkeypatch, "kg.stability", run=Recorder(stability_result(cfg)))
    assert cfg.llm.provider == "claude_subscription"
    main(["stability", "--config", str(cfg_path)])
    out = capsys.readouterr()
    assert "5-hour" in out.out + out.err or "window" in out.out + out.err


# --------------------------------------------------------- sample-relevance


def sample_result(cfg, rows: int = 6):
    path = cfg.sandbox.root("graph") / "_review" / f"relevance-sample-{RUN_ID}.md"
    return SimpleNamespace(rows=[object()] * rows, path=path, shortfall={"0-39": 0, "40-70": 0, "71-100": 0}, bands=((0, 39), (40, 70), (71, 100)))


def test_S18_sample_relevance_uses_config_defaults_and_prints_the_review_path(cfg_path, cfg, monkeypatch, capsys):
    rec = Recorder(sample_result(cfg))
    _install(monkeypatch, "kg.relevance_sample", sample=rec)
    rc = main(["sample-relevance", "--config", str(cfg_path)])
    assert rc == 0
    (graph_dir,), kw = rec.calls[0]
    assert Path(graph_dir) == cfg.sandbox.root("graph")
    assert kw["seed"] == cfg.relevance_sample.seed == 1
    assert [tuple(b) for b in kw["bands"]] == [tuple(b) for b in cfg.relevance_sample.bands]
    assert kw["n_per_band"] == math.ceil(cfg.relevance_sample.n / len(cfg.relevance_sample.bands)) == 10
    assert isinstance(kw["now"], datetime)
    assert str(sample_result(cfg).path) in capsys.readouterr().out


def test_S18_sample_relevance_n_and_seed_flags_override_config(cfg_path, cfg, monkeypatch):
    rec = Recorder(sample_result(cfg))
    _install(monkeypatch, "kg.relevance_sample", sample=rec)
    assert main(["sample-relevance", "--n", "7", "--seed", "42", "--config", str(cfg_path)]) == 0
    kw = rec.calls[0][1]
    assert kw["n_per_band"] == 3 and kw["seed"] == 42


def test_S18_sample_relevance_reports_shortfall_to_the_user(cfg_path, cfg, monkeypatch, capsys):
    res = sample_result(cfg, rows=4)
    res.shortfall = {"0-39": 2, "40-70": 0, "71-100": 0}
    _install(monkeypatch, "kg.relevance_sample", sample=Recorder(res))
    assert main(["sample-relevance", "--config", str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert "0-39" in out and "2" in out


def test_R17_sample_relevance_never_constructs_a_provider_adapter(cfg_path, cfg, monkeypatch):
    import kg.config as kc

    def no_adapter(*a, **kw):
        raise AssertionError("kg sample-relevance must not construct an adapter (DESIGN §2)")

    monkeypatch.setattr(kc, "adapter_factory", no_adapter, raising=False)
    _install(monkeypatch, "kg.relevance_sample", sample=Recorder(sample_result(cfg)))
    assert main(["sample-relevance", "--config", str(cfg_path)]) == 0


# ------------------------------------------------------ summarise-relevance


def summary():
    band = lambda a, d, u, r: SimpleNamespace(agree=a, disagree=d, unfilled=u, rate=r)  # noqa: E731
    return SimpleNamespace(
        bands={"0-39": band(1, 3, 0, 0.25), "40-70": band(2, 2, 0, 0.5), "71-100": band(4, 0, 0, 1.0)},
        monotonic=True,
        verdict="monotonic",
        filled=12,
        unfilled=0,
        malformed=0,  # fix cycle 1 (d38688b): Summary.malformed is printed by `kg summarise-relevance`
    )


def test_S18_summarise_relevance_is_registered_and_prints_rates_per_band_and_the_verdict(cfg_path, cfg, monkeypatch, capsys):
    review = cfg.sandbox.root("graph") / "_review" / f"relevance-sample-{RUN_ID}.md"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text("| edge | source | target | relevance | band | verdict | notes |\n|---|---|---|---|---|---|---|\n", encoding="utf-8")
    rec = Recorder(summary())
    _install(monkeypatch, "kg.relevance_sample", summarise=rec)

    rc = main(["summarise-relevance", str(review), "--config", str(cfg_path)])
    assert rc == 0
    (path,), _ = rec.calls[0]
    assert Path(path) == review
    out = capsys.readouterr().out
    for needle in ("0-39", "40-70", "71-100", "0.25", "0.50", "1.00", "monotonic", "malformed"):
        assert needle in out, needle


def test_S18_summarise_relevance_requires_a_file_argument(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["summarise-relevance"])
    assert exc.value.code == 2


def test_S18_summarise_relevance_missing_file_is_a_clean_error(cfg_path, cfg, monkeypatch, capsys):
    def missing(path):
        raise FileNotFoundError(path)

    _install(monkeypatch, "kg.relevance_sample", summarise=missing)
    target = cfg.sandbox.root("graph") / "_review" / "nope.md"
    rc = main(["summarise-relevance", str(target), "--config", str(cfg_path)])
    assert rc != 0
    assert "nope.md" in capsys.readouterr().err


def test_S23_summarise_relevance_refuses_a_file_in_a_forbidden_location(tmp_path, cfg_path, monkeypatch, capsys):
    rec = Recorder(summary())
    _install(monkeypatch, "kg.relevance_sample", summarise=rec)
    forbidden = tmp_path / "TK-PKA" / "01-knowledge-base" / "sample.md"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("| edge | source | target | relevance | band | verdict | notes |\n", encoding="utf-8")
    rc = main(["summarise-relevance", str(forbidden), "--config", str(cfg_path)])
    assert rc != 0
    assert rec.calls == [], "the file must not even be opened"
    assert "sandbox" in capsys.readouterr().err.lower()


# ------------------------------------------------------------ help surface


def test_S18_help_lists_every_documented_command(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("ingest", "gates", "render", "stability", "sample-relevance", "summarise-relevance", "selftest"):
        assert cmd in out, cmd


def test_S18_none_of_the_wu4_wu5_commands_report_not_implemented(cfg_path, cfg, monkeypatch, capsys):
    _install(monkeypatch, "kg.render.html", render=lambda g, o, **kw: o)
    _install(monkeypatch, "kg.gates", validate_graph=lambda g, s: [])
    _install(monkeypatch, "kg.llm", call_stage=object())
    _install(monkeypatch, "kg.stability", run=lambda *a, **kw: stability_result(cfg))
    _install(monkeypatch, "kg.relevance_sample", sample=lambda *a, **kw: sample_result(cfg), summarise=lambda p: summary())
    review = cfg.sandbox.root("graph") / "_review" / "r.md"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text("| edge | source | target | relevance | band | verdict | notes |\n", encoding="utf-8")
    for argv in (["render"], ["gates"], ["stability"], ["sample-relevance"], ["summarise-relevance", str(review)]):
        rc = main([*argv, "--config", str(cfg_path)])
        err = capsys.readouterr().err
        assert "not implemented" not in err, argv
        assert rc == 0, (argv, err)
