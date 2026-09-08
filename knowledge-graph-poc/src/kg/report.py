"""Run report, Markdown + JSON (DESIGN §14; D28; R15, R18, R21).

``build`` assembles a ``RunReport`` from plain rows; ``write`` renders both files
under ``<graph>/_reports/run-<run_id>.{md,json}`` — but only after scanning the
text for every credential value the configuration knows about (the selected
provider's key, every credential-shaped ``.env`` value, every credential variable in
the process environment — see ``secret_values``). A hit raises ``ReportRedactionError``
and nothing is written.
Cost semantics come from ``kg.llm.ledger.cost_display``: unknown is never "$0.00".
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from kg.config import CREDENTIAL_VARS, STAGES, Config
from kg.llm.ledger import SUBSCRIPTION_PROVIDER, UsageLedger, cost_display
from kg.prompts import prompt_set_hash

REPORTS_DIR = "_reports"
DRY_RUN_MODE = "dry-run"
REDACTED = "<redacted>"


class ReportRedactionError(Exception):
    """The rendered report would contain a credential value; nothing was written."""


@dataclass
class RunReport:
    run_id: str
    started: str
    finished: str
    mode: str
    status: str
    corpus: str
    schema: str
    provider: str
    provider_overridden: bool
    prompt_versions: dict[str, str]
    prompt_set_hash: str
    stages: dict[str, dict[str, Any]]
    inputs: list[dict[str, Any]]
    counts: dict[str, Any]
    rejected_edges: list[dict[str, Any]]
    review_queues: dict[str, int]
    usage: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    moves: dict[str, Any] | None = None

    @property
    def dry_run(self) -> bool:
        return self.mode == DRY_RUN_MODE


def _stage_rows(cfg: Config, ledger: UsageLedger, stage_models: Mapping[str, str] | None) -> dict[str, dict[str, Any]]:
    per = ledger.per_stage()
    rows: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        agg = per.get(stage)
        requested = (stage_models or {}).get(stage) or cfg.llm.model_for(stage)
        rows[stage] = {
            "provider": cfg.llm.provider,
            "model_requested": requested,
            "model_served": agg["model_served"] if agg else None,
            "calls": agg["calls"] if agg else 0,
            "attempts": agg["attempts"] if agg else 0,
            "repair_attempts": agg["repair_attempts"] if agg else 0,
            "input_tokens": agg["input_tokens"] if agg else 0,
            "output_tokens": agg["output_tokens"] if agg else 0,
            "cached_tokens": agg["cached_tokens"] if agg else 0,
            "reasoning_tokens": agg["reasoning_tokens"] if agg else 0,
            "cost_usd": agg["cost_usd"] if agg else None,
            "cost_estimate_usd": agg["cost_estimate_usd"] if agg else None,
            "cost_display": agg["cost_display"] if agg else cost_display(cfg.llm.provider, None, None),
        }
    return rows


def _default_status(inputs: Iterable[Mapping[str, Any]]) -> str:
    return "partial" if any(i.get("status") in ("deferred", "failed") for i in inputs) else "ok"


def build(
    cfg: Config,
    *,
    run_id: str,
    started: str,
    finished: str,
    mode: str,
    inputs: list[dict[str, Any]],
    nodes: dict[str, int],
    edges_by_type: dict[str, int],
    rejected_edges: list[dict[str, Any]],
    review_queues: dict[str, int],
    ledger: UsageLedger,
    prompt_versions: dict[str, str],
    stage_models: Mapping[str, str] | None = None,
    warnings: Iterable[str] = (),
    events: Iterable[dict[str, Any]] = (),
    moves: dict[str, Any] | None = None,
    provider_overridden: bool = False,
    status: str | None = None,
) -> RunReport:
    usage = ledger.to_dict()
    if not ledger.rows:
        usage["total"]["cost_display"] = cost_display(cfg.llm.provider, None, None)
    usage["soft_budget_applies"] = cfg.llm.provider != SUBSCRIPTION_PROVIDER
    return RunReport(
        run_id=run_id,
        started=started,
        finished=finished,
        mode=mode,
        status=status or _default_status(inputs),
        corpus=cfg.corpus.slug,
        schema=cfg.corpus.schema,
        provider=cfg.llm.provider,
        provider_overridden=provider_overridden,
        prompt_versions=dict(prompt_versions),
        prompt_set_hash=prompt_set_hash(),
        stages=_stage_rows(cfg, ledger, stage_models),
        inputs=[dict(i) for i in inputs],
        counts={"nodes": dict(nodes), "edges_by_type": dict(edges_by_type)},
        rejected_edges=[dict(r) for r in rejected_edges],
        review_queues=dict(review_queues),
        usage=usage,
        warnings=list(warnings),
        events=[dict(e) for e in events],
        moves=dict(moves) if moves is not None else None,
    )


def to_dict(rep: RunReport) -> dict[str, Any]:
    out = asdict(rep)
    out["dry_run"] = rep.dry_run
    return out


# ------------------------------------------------------------------ markdown


def cell(v: Any) -> str:
    """One Markdown table cell: pipes escaped, line breaks flattened (shared with the review queues)."""
    return "" if v is None else str(v).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines


def render_markdown(rep: RunReport) -> str:
    total = rep.usage["total"]
    lines = [
        f"# Run report {rep.run_id}",
        "",
        f"- mode: {rep.mode}" + (" (dry run)" if rep.dry_run else ""),
        f"- status: {rep.status}",
        f"- started: {rep.started}",
        f"- finished: {rep.finished}",
        f"- corpus: {rep.corpus}",
        f"- schema: {rep.schema}",
        f"- provider: {rep.provider}" + (" (overridden from the command line)" if rep.provider_overridden else ""),
        f"- prompt set: {rep.prompt_set_hash[:16]} (" + ", ".join(f"{k}={v}" for k, v in rep.prompt_versions.items()) + ")",
        "",
        "## Stages",
        "",
        *_table(
            ["stage", "model requested", "model served", "calls", "repair attempts", "input tokens", "output tokens", "cost"],
            [[s, r["model_requested"], r["model_served"] or "—", r["calls"], r["repair_attempts"], r["input_tokens"], r["output_tokens"], r["cost_display"]] for s, r in rep.stages.items()],
        ),
        "",
        "## Inputs",
        "",
        *_table(["input", "outcome", "units", "chunks", "reason"], [[i["name"], i["status"], i.get("units", 0), i.get("chunks", 0), i.get("reason") or ""] for i in rep.inputs]),
        "",
        "## Counts",
        "",
        f"- nodes: created {rep.counts['nodes'].get('created', 0)}, updated {rep.counts['nodes'].get('updated', 0)}, total {rep.counts['nodes'].get('total', 0)}",
        "- edges by type: " + (", ".join(f"{t} {n}" for t, n in rep.counts["edges_by_type"].items()) or "none"),
        "",
        "## Rejected edges",
        "",
    ]
    if rep.rejected_edges:
        lines += _table(["gate", "type", "source", "target", "reason"], [[r.get("gate"), r.get("type"), r.get("source_id"), r.get("target_id"), r.get("reason")] for r in rep.rejected_edges])
    else:
        lines.append("(none)")
    lines += [
        "",
        "## Review queues",
        "",
        *[f"- {k}: {v}" for k, v in rep.review_queues.items()],
        "",
        "## Usage and cost",
        "",
        f"- calls: {total['calls']}, attempts: {total['attempts']}, repair attempts: {total['repair_attempts']}",
        f"- tokens: input {total['input_tokens']}, output {total['output_tokens']}, cached {total['cached_tokens']}, reasoning {total['reasoning_tokens']}",
        f"- cost: {total['cost_display']}",
    ]
    if rep.usage.get("soft_budget_applies"):
        budget = rep.usage.get("soft_budget_usd")
        lines.append(f"- soft budget: {'not set' if budget is None else f'${budget:.2f}'}; spent so far: {cost_display(rep.provider, rep.usage.get('spent_usd'))}")
    else:
        lines.append("- soft budget: not applicable (subscription usage is not billed per token)")
    lines += ["", "## Provider events", ""]
    lines += [f"- {json.dumps(e, ensure_ascii=False)}" for e in rep.events] or ["(none)"]
    lines += ["", "## Warnings", ""]
    lines += [f"- {w}" for w in rep.warnings] or ["(none)"]
    if rep.moves is not None:
        lines += ["", "## File moves", "", f"- before: {json.dumps(rep.moves.get('before'))}", f"- after: {json.dumps(rep.moves.get('after'))}"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- redaction


#: A credential value shorter than this is not scanned for: `x` or `general` would match everywhere.
MIN_SECRET_LEN = 8
#: `.env` names that hold credentials; `KG_SCHEMA=general` or `PORT=8080` must not mark a healthy run failed.
_CREDENTIAL_NAME = re.compile(r"(_KEY|_TOKEN|_SECRET)$|PASSWORD", re.IGNORECASE)


def _is_secret(value: str | None) -> bool:
    return bool(value) and len(value.strip()) >= MIN_SECRET_LEN


def secret_values(cfg: Config) -> dict[str, str]:
    """variable name -> value for every credential this run could know about. Values never leave this module.

    Scanned: the selected provider's key, every credential-shaped `.env` variable (`*_KEY`,
    `*_TOKEN`, `*_SECRET`, `*PASSWORD*`) and every `CREDENTIAL_VARS` entry in the process
    environment — each only when at least `MIN_SECRET_LEN` characters long.
    """
    found: dict[str, str] = {}
    if _is_secret(cfg.llm.api_key):
        found[cfg.llm.credential_var or "api_key"] = cfg.llm.api_key.strip()
    dotenv = cfg.project_root / ".env"
    if dotenv.is_file():
        try:
            for k, v in dotenv_values(dotenv, interpolate=False).items():
                if (str(k) in CREDENTIAL_VARS or _CREDENTIAL_NAME.search(str(k))) and _is_secret(v):
                    found.setdefault(str(k), v.strip())
        except OSError:
            pass  # the report is still checked against the environment and the loaded key
    for var in CREDENTIAL_VARS:
        v = os.environ.get(var)
        if _is_secret(v):
            found.setdefault(var, v.strip())
    return found


def redact_check(text: str, cfg: Config) -> None:
    leaked = sorted(name for name, value in secret_values(cfg).items() if value in text)
    if leaked:
        raise ReportRedactionError(f"report text would contain the value of {', '.join(leaked)}; nothing written")


def _replace_deep(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, REDACTED)
        return value
    if isinstance(value, dict):
        return {k: _replace_deep(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_deep(v, secrets) for v in value]
    return value


def redact(rep: RunReport, cfg: Config) -> RunReport:
    """`rep` with every known credential value replaced by ``<redacted>`` and ``status: failed``.

    The pipeline uses this after ``write`` refused: notes and moves are already on disk,
    so a report must still be written (R15) — one that names the failure, not the secret.
    """
    secrets = sorted(secret_values(cfg).values(), key=len, reverse=True)
    data = _replace_deep(asdict(rep), secrets)
    data["status"] = "failed"
    data["warnings"] = [*data.get("warnings", []), f"report contained a credential value; it was replaced by {REDACTED}"]
    return RunReport(**data)


def write(rep: RunReport, cfg: Config) -> tuple[Path, Path]:
    md = render_markdown(rep)
    js = json.dumps(to_dict(rep), ensure_ascii=False, indent=2) + "\n"
    redact_check(md, cfg)
    redact_check(js, cfg)
    md_path = cfg.sandbox.resolve("graph", f"{REPORTS_DIR}/run-{rep.run_id}.md")
    js_path = cfg.sandbox.resolve("graph", f"{REPORTS_DIR}/run-{rep.run_id}.json")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    js_path.write_text(js, encoding="utf-8")
    return md_path, js_path


__all__ = ["DRY_RUN_MODE", "REDACTED", "REPORTS_DIR", "ReportRedactionError", "RunReport", "build", "cell", "redact", "redact_check", "render_markdown", "secret_values", "to_dict", "write"]
