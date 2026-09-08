"""`kg` command-line entry point (DESIGN §18).

Subcommands: `ingest`, `gates`, `render`, `stability`, `sample-relevance`,
`summarise-relevance`, `selftest`. Every command resolves its paths through the
sandbox of the loaded config, prints where it wrote, exits 0 on success and
non-zero on failure, and turns the project's own exceptions into one-line errors.

Import-light on purpose: `python -m kg.cli` must start with nothing but the
standard library, so provider SDKs and config are imported inside handlers.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Mirror of kg.config.PROVIDERS for the argparse `choices`; kept import-light here
# and cross-checked against the real tuple in cmd_selftest (raises on drift).
_PROVIDER_CHOICES = ("claude_subscription", "openrouter", "anthropic_api", "azure_foundry")

# Stripped from the selftest environment in addition to kg.config.CREDENTIAL_VARS:
# the Claude Code CLI and the OpenAI SDK also honour these, and a leaked one would
# let a buggy test reach a real provider.
_EXTRA_STRIPPED_VARS = ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTS_DIR = _REPO_ROOT / "tests"

SUBSCRIPTION_PROVIDER = "claude_subscription"


def _fail(command: str, exc: BaseException) -> int:
    print(f"kg {command}: {exc}", file=sys.stderr)
    return 1


def _known_errors() -> tuple[type[BaseException], ...]:
    """The project's own error types plus OS errors: reported as one line, never as a traceback."""
    from kg.config import ConfigError
    from kg.notes import NoteError
    from kg.paths import SandboxViolation
    from kg.registry import RegistryError
    from kg.report import ReportRedactionError

    return (ConfigError, NoteError, SandboxViolation, RegistryError, ReportRedactionError, OSError, ValueError, RuntimeError)


def _load(args: argparse.Namespace, *, provider: str | None = None):
    from kg.config import ConfigError, load_config

    try:
        return load_config(args.config, provider=provider)
    except ConfigError as exc:
        print(f"kg {args.command}: {exc}", file=sys.stderr)
        return None


def cmd_ingest(args: argparse.Namespace) -> int:
    """Full pipeline S0–S8 (DESIGN §6); model calls go through `kg.llm.call_stage` unless --dry-run."""
    cfg = _load(args, provider=args.provider)
    if cfg is None:
        return 1
    from kg.llm import call_stage
    from kg.pipeline import PipelineError, run

    try:
        res = run(
            cfg,
            call_stage=call_stage,
            now=datetime.now(UTC),
            dry_run=args.dry_run,
            write_same_as=True if args.write_same_as else None,
            provider_overridden=args.provider is not None,
        )
    except (PipelineError, *_known_errors()) as exc:
        return _fail("ingest", exc)
    for row in res.inputs:
        print(f"{row.status:<12} {row.name}" + (f"  ({row.reason})" if row.reason else ""))
    print(f"run {res.run_id}: status {res.status}, {res.nodes_written} node(s) written; report {res.report_md}")
    return 0 if res.status == "ok" else 1


def cmd_gates(args: argparse.Namespace) -> int:
    """Validate the notes on disk; no model call (R7, DESIGN §10)."""
    cfg = _load(args)
    if cfg is None:
        return 1
    from kg.gates import validate_graph
    from kg.schemas import SCHEMAS

    try:
        findings = validate_graph(cfg.sandbox.root("graph"), SCHEMAS[cfg.corpus.schema])
    except _known_errors() as exc:
        return _fail("gates", exc)
    for f in findings:
        print(f"{f.gate}: {f.reason}")
    print(f"kg gates: {len(findings)} finding(s)")
    return 0 if not findings else 1


def cmd_render(args: argparse.Namespace) -> int:
    """Regenerate `_index.json` and the single-file graph.html from the notes; no model call, view-only (R14, DESIGN §13, §18 D32).

    Deterministic by default: identical notes give byte-identical output. `--stamp` opts
    into a `meta.generated` timestamp (in graph.html and _index.json) taken from the clock.
    """
    cfg = _load(args)
    if cfg is None:
        return 1
    from kg.render.html import render
    from kg.schemas import SCHEMAS

    try:
        out = cfg.sandbox.resolve("graph", "graph.html")
        written = render(
            cfg.sandbox.root("graph"),
            out,
            schema=SCHEMAS[cfg.corpus.schema],
            now=datetime.now(UTC) if args.stamp else None,
            sandbox=cfg.sandbox,
        )
    except _known_errors() as exc:
        return _fail("render", exc)
    print(f"kg render: wrote {written}")
    return 0


def _inbox_items(cfg) -> int:
    inbox = cfg.sandbox.root("inbox")
    return sum(1 for p in inbox.iterdir() if not p.name.startswith(".")) if inbox.is_dir() else 0


def cmd_stability(args: argparse.Namespace) -> int:
    """Two clean ingests of the same inbox, node-set agreement (R16, DESIGN §15). No --provider: both legs share kg.yaml."""
    cfg = _load(args)
    if cfg is None:
        return 1
    from kg.llm import call_stage
    from kg.pipeline import PipelineError
    from kg.stability import LEGS, run

    if cfg.llm.provider == SUBSCRIPTION_PROVIDER:
        from kg.config import STAGES

        items = _inbox_items(cfg)
        # Upper-bound estimate (DESIGN §15): every inbox item through every model stage, twice. Images add
        # a describe call and long files split into several atomize chunks, so the true count may differ.
        est_calls = items * len(STAGES) * len(LEGS)
        print(
            f"kg stability: provider {SUBSCRIPTION_PROVIDER} — up to ~{est_calls} model call(s) "
            f"({len(LEGS)} legs × {items} inbox item(s) × {len(STAGES)} stage(s)) share one 5-hour usage window; "
            "a rate limit hit mid-leg defers files and marks the comparison INVALID."
        )
    try:
        res = run(cfg, call_stage=call_stage, now=datetime.now(UTC), keep=args.keep, allow_served_drift=args.allow_served_drift)
    except (PipelineError, *_known_errors()) as exc:
        return _fail("stability", exc)
    verdict = "PASS" if res.passed else "FAIL"
    print(f"jaccard {res.jaccard:.2f} (threshold {res.threshold:.2f}) — {verdict}; overlap a {res.overlap_a:.2f} · b {res.overlap_b:.2f}; fuzzy {res.fuzzy_jaccard:.2f}")
    print(f"only in a: {len(res.only_in_a)} · only in b: {len(res.only_in_b)} · common: {len(res.common)} · related_to deltas: {res.relevance_deltas.n}")
    if not res.valid:
        print("INVALID for §8(b) — legs not equivalent:")
        for reason in res.invalid_reasons:
            print(f"  - {reason}")
    print(f"report {res.report_md}")
    return 0 if res.passed and res.valid else 1


def cmd_sample_relevance(args: argparse.Namespace) -> int:
    """Stratified `related_to` sample for human judgment (R17, DESIGN §16); no model call."""
    cfg = _load(args)
    if cfg is None:
        return 1
    from kg.relevance_sample import band_label, sample

    rs = cfg.relevance_sample
    bands = tuple(tuple(b) for b in rs.bands)
    n = args.n if args.n is not None else rs.n
    seed = args.seed if args.seed is not None else rs.seed
    if n < 1:
        print("kg sample-relevance: --n must be a positive integer", file=sys.stderr)
        return 2
    n_per_band = math.ceil(n / len(bands))
    try:
        res = sample(cfg.sandbox.root("graph"), n_per_band=n_per_band, bands=bands, seed=seed, now=datetime.now(UTC))
    except _known_errors() as exc:
        return _fail("sample-relevance", exc)
    print(f"kg sample-relevance: {len(res.rows)} edge(s) ({n_per_band} per band, seed {seed}) written to {res.path}")
    for band in res.bands:
        label = band_label(band)
        short = res.shortfall.get(label, 0)
        if short:
            print(f"  shortfall in band {label}: {short} fewer than requested (pool exhausted)")
    return 0


def cmd_summarise_relevance(args: argparse.Namespace) -> int:
    """Agreement per band and the §8(c) monotonicity verdict from a filled review file (R17)."""
    from kg.paths import is_denied

    path = Path(args.file)
    if is_denied(path) or is_denied(os.path.realpath(path)):
        print(f"kg summarise-relevance: refused by the sandbox guard — {path} lies in a forbidden location", file=sys.stderr)
        return 1
    from kg.relevance_sample import summarise

    try:
        s = summarise(path)
    except _known_errors() as exc:
        return _fail("summarise-relevance", exc)
    print(f"{'band':<10} {'agree':>6} {'disagree':>9} {'unfilled':>9} {'rate':>6}")
    for label, b in s.bands.items():
        rate = "n/a" if b.rate is None else f"{b.rate:.2f}"
        print(f"{label:<10} {b.agree:>6} {b.disagree:>9} {b.unfilled:>9} {rate:>6}")
    print(f"filled {s.filled} · unfilled {s.unfilled} · malformed rows {s.malformed}")
    print(f"verdict: {s.verdict}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run the offline test suite with every credential variable unset (DESIGN §17). Spends no tokens."""
    from kg.config import CREDENTIAL_VARS, PROVIDERS

    if set(PROVIDERS) != set(_PROVIDER_CHOICES):
        raise RuntimeError("kg.cli: --provider choices drifted from kg.config.PROVIDERS")
    if not _TESTS_DIR.is_dir():
        print(f"kg selftest: tests directory not found at {_TESTS_DIR} (run from a source checkout)", file=sys.stderr)
        return 1
    stripped = set(CREDENTIAL_VARS) | set(_EXTRA_STRIPPED_VARS)
    env = {k: v for k, v in os.environ.items() if k not in stripped}
    cmd = [sys.executable, "-m", "pytest", "-q", str(_TESTS_DIR), *args.pytest_args]
    try:
        proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=False)
    except OSError as exc:
        print(f"kg selftest: could not start pytest: {exc}", file=sys.stderr)
        return 1
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kg", description="Knowledge-graph extraction POC.")
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    p = sub.add_parser("ingest", help="Run the full pipeline on the inbox (model calls unless --dry-run).")
    p.add_argument("--config", default="kg.yaml", help="Path to kg.yaml (default: ./kg.yaml).")
    p.add_argument("--dry-run", action="store_true", help="Convert and estimate only; no model calls.")
    p.add_argument("--provider", choices=_PROVIDER_CHOICES, help="Override llm.provider for this run.")
    p.add_argument("--write-same-as", action="store_true", help="Write same_as edges for confirmed duplicates.")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("gates", help="Validate notes on disk (no model calls).")
    p.add_argument("--config", default="kg.yaml")
    p.set_defaults(func=cmd_gates)

    p = sub.add_parser("render", help="Regenerate _index.json and graph.html from the notes (view-only, no model calls).")
    p.add_argument("--config", default="kg.yaml")
    p.add_argument("--stamp", action="store_true", help="Record the current time as meta.generated (default: omitted, output deterministic).")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("stability", help="Ingest twice into clean folders and report node-set agreement (no --provider: both legs use kg.yaml).")
    p.add_argument("--config", default="kg.yaml")
    p.add_argument("--keep", action="store_true", help="Keep both legs' output folders.")
    p.add_argument(
        "--allow-served-drift",
        action="store_true",
        help="Treat a served-model mismatch between legs as a warning instead of INVALID (routers may serve different builds); the drift is recorded in the report.",
    )
    p.set_defaults(func=cmd_stability)

    p = sub.add_parser("sample-relevance", help="Write a stratified sample of related_to edges for human review.")
    p.add_argument("--config", default="kg.yaml")
    p.add_argument("--n", type=int, help="Total sample size, split evenly across bands (default from kg.yaml).")
    p.add_argument("--seed", type=int, help="Random seed (default from kg.yaml).")
    p.set_defaults(func=cmd_sample_relevance)

    p = sub.add_parser("summarise-relevance", help="Agreement rate per band and the monotonicity verdict of a filled review file.")
    p.add_argument("file", help="Path to a relevance-sample-<run_id>.md with the verdict column filled in.")
    p.add_argument("--config", default="kg.yaml", help="Accepted for uniformity; the file path is used as given.")
    p.set_defaults(func=cmd_summarise_relevance)

    p = sub.add_parser(
        "selftest",
        help="Run the offline test suite; spends no tokens.",
        epilog="Any further arguments (e.g. -k sandbox, -x, or after --) are passed to pytest.",
    )
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # `selftest` forwards unknown arguments to pytest (so `kg selftest -k sandbox`
    # and `kg selftest -x` work); every other subcommand keeps strict parsing.
    args, extra = parser.parse_known_args(argv)
    if args.command == "selftest":
        if extra and extra[0] == "--":
            extra = extra[1:]
        args.pytest_args = extra
    elif extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
