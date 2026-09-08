"""KL Map draft review API (R18).

The Addendum's authoring rule is that a map is **LLM-extracted, then
human-reviewed before it goes live**. These endpoints are that review step, and
they enforce it server-side:

- `GET  /api/klmap/drafts` — every extracted draft with its validation report.
- `GET  /api/klmap/drafts/{course_id}` — one draft, for the review UI.
- `POST /api/klmap/drafts/{course_id}/approve` — `{approve, note}`; approving
  publishes the draft as the course's live map.
- `POST /api/klmap/drafts/{course_id}/revalidate` — re-read the YAML after a
  human edited it by hand (the POC's editing story) and re-run the loader.
- `GET  /api/klmap` — the live map, or the draft marked `approved: false`.

**Approval is refused while the draft has blocking errors.** The SPA disables
its button, but a disabled button is a hint, not a rule: a cyclic prerequisite
graph reaching the live map would make scaffolding non-terminating (Tech Spec
§2.3), so the check has to live here, where it cannot be bypassed.

Approval never edits the draft in place — it copies it to `content/klmap/` and
records the decision in the draft's sidecar, so the extracted artefact and the
published one stay distinguishable.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from socratic_tutor.config import get_settings
from socratic_tutor.domain.klmap import (
    KLMap,
    ValidationIssue,
    ValidationReport,
    load_klmap_with_report,
    save_klmap,
)
from socratic_tutor.domain.rag.kl_extract import draft_metadata_path, draft_path
from socratic_tutor.pedagogy.student_model import (
    StudentModelError,
    resolve_within,
    safe_path_segment,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/klmap", tags=["klmap"])

#: Draft lifecycle. These are the server's values; the SPA's `DraftStatus` union
#: is maintained separately in `web/src/types.ts`, so a change here is a contract
#: change that needs the same edit there. (It said `draft` for `pending` at
#: cutover — the divergence was invisible because this comment claimed a match
#: it never enforced.)
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"


class DraftDecision(BaseModel):
    """`POST .../approve` body."""

    approve: bool
    note: str | None = None


class KlMapDraftResponse(BaseModel):
    """One draft as the review UI renders it."""

    course_id: str
    course_name: str
    path: str
    extracted_at: str = ""
    status: str = STATUS_PENDING
    reviewed_at: str | None = None
    review_note: str | None = None
    sources: list[str] = Field(default_factory=list)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    report: dict[str, Any] = Field(default_factory=dict)


def drafts_dir() -> Path:
    return get_settings().content_dir / "drafts"


#: The course the seed pack ships under. `GET /api/klmap` falls back to the seed
#: for this id only; `chat_routes` names it in the log when a live map displaces
#: it. `ingest_routes.DEFAULT_COURSE_ID` happens to equal it today, but that one
#: is "where an upload goes when the client says nothing", a different question.
SEED_COURSE_ID = "MATH-SEED-01"

#: What a `course_id` may look like on the wire. Same alphabet as
#: `safe_path_segment`, stated here so request models can reject a bad id with a
#: 422 before it reaches a path helper; the cap keeps `klmap-<id>.yaml` and the
#: Chroma collection name inside filesystem limits (security review L1).
COURSE_ID_PATTERN = r"^[A-Za-z0-9._-]+$"
COURSE_ID_MAX_LENGTH = 64


def live_path(course_id: str) -> Path:
    """Where an approved map is published. Only approval writes here.

    Validated and resolved the way sessions and students are — `safe_path_segment`
    then `resolve_within` — rather than through a lossy slug. The slug used to
    fold `ALG 101`, `ALG-101` and `ALG/101` onto one file, so approving one
    course's draft could overwrite another's live map. Raises `StudentModelError`
    for an unsafe id; route handlers turn that into a 400 via `_checked`.
    """
    segment = safe_path_segment(course_id, "course_id")
    return resolve_within(get_settings().content_dir / "klmap", f"klmap-{segment}.yaml")


def _checked(course_id: str) -> str:
    """A path-parameter course id, or a 400 that says why it was refused."""
    try:
        return safe_path_segment(course_id, "course_id")
    except StudentModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def seed_klmap_path() -> Path:
    """The hand-authored seed map — the one course that ships without a review.

    Defined here, next to `live_path`, so the chat path and this API resolve the
    seed from one place; they used to name the file independently, which is how
    `GET /api/klmap` came to 404 for the course chat was teaching from (S2).
    """
    return get_settings().content_dir / "seed" / "kl-map-linear-equations.yaml"


def _read_metadata(course_id: str) -> dict[str, Any]:
    """The extraction sidecar, or an empty dict when a draft predates it."""
    path = draft_metadata_path(course_id, drafts_dir())
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("unreadable draft metadata for %s: %s", course_id, exc)
        return {}


def _write_metadata(course_id: str, metadata: dict[str, Any]) -> None:
    path = draft_metadata_path(course_id, drafts_dir())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _course_ids() -> list[str]:
    """Course ids that have a draft on disk.

    Read from the file, not the validator, so an invalid draft is still listed
    under its real course id rather than one guessed from the filename.
    """
    directory = drafts_dir()
    if not directory.is_dir():
        return []
    ids = []
    for path in sorted(directory.glob("klmap-*.yaml")):
        data = _read_draft_yaml(path)
        ids.append(str(data.get("course_id") or path.stem.removeprefix("klmap-")))
    return ids


def _read_draft_yaml(path: Path) -> dict[str, Any]:
    """The draft file as written, with no validation gate in front of it.

    `load_klmap_with_report` returns `(None, report)` for an invalid map, which
    is right for the serving path and wrong for review: the reviewer would get
    the list of problems next to an empty graph, with nothing to judge. The
    extractor deliberately preserves an invalid draft so a human can see what
    the model proposed, so the review API must not discard it.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        # Unparseable YAML has no content to show; the report says why.
        logger.warning("draft %s is not readable as YAML: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _display_nodes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Nodes exactly as authored, including any the validator rejected."""
    raw = data.get("nodes")
    if not isinstance(raw, list):
        return []
    nodes = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        nodes.append(
            {
                "id": str(entry.get("id", "")),
                "name": str(entry.get("name", "")),
                "name_th": entry.get("name_th"),
            }
        )
    return nodes


def _display_edges(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Edges as authored — dangling ones and out-of-vocabulary relations included.

    Passed through as raw strings rather than coerced to `Relation`: an edge
    naming a relation outside the closed vocabulary is precisely what the
    reviewer needs to see, and coercion would either drop it or raise.
    """
    raw = data.get("edges")
    if not isinstance(raw, list):
        return []
    edges = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        edges.append(
            {
                "from": str(entry.get("from", "")),
                "to": str(entry.get("to", "")),
                "relation": str(entry.get("relation", "")),
            }
        )
    return edges


def _build_draft(course_id: str) -> KlMapDraftResponse:
    """Load a draft for review, re-validating so a hand-edit is reflected.

    Content and verdict are read separately on purpose: the content comes from
    the YAML unconditionally, the verdict from the loader. A draft that fails
    validation still arrives with its graph intact.
    """
    path = draft_path(course_id, drafts_dir())
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No extracted draft for course {course_id!r}.")

    _, report = load_klmap_with_report(path)
    data = _read_draft_yaml(path)
    metadata = _read_metadata(course_id)
    stored = metadata.get("report") or {}

    # The loader's verdict is authoritative — the YAML may have been hand-edited
    # since extraction, so its errors are re-derived from the file every time.
    # Extraction-only findings (`KX` codes: what the model proposed and we
    # dropped) can never be rediscovered by re-reading the file, so they are
    # carried over from the sidecar.
    merged = ValidationReport(
        source=str(path), errors=list(report.errors), warnings=list(report.warnings)
    )
    for issue in stored.get("warnings", []):
        if str(issue.get("code", "")).startswith("KX"):
            merged.warnings.append(ValidationIssue.model_validate(issue))
    for issue in stored.get("errors", []):
        if str(issue.get("code", "")).startswith("KX"):
            merged.errors.append(ValidationIssue.model_validate(issue))

    return KlMapDraftResponse(
        course_id=str(data.get("course_id") or course_id),
        course_name=str(data.get("course_name") or metadata.get("course_name") or course_id),
        path=str(path),
        extracted_at=metadata.get("extracted_at", ""),
        status=metadata.get("status", STATUS_PENDING),
        reviewed_at=metadata.get("reviewed_at"),
        review_note=metadata.get("review_note"),
        sources=list(metadata.get("sources", [])),
        nodes=_display_nodes(data),
        edges=_display_edges(data),
        report=merged.model_dump(mode="json"),
    )


@router.get("/drafts", response_model=list[KlMapDraftResponse])
async def list_drafts() -> list[KlMapDraftResponse]:
    """Every draft awaiting or past review."""
    return [_build_draft(course_id) for course_id in _course_ids()]


@router.get("/drafts/{course_id}", response_model=KlMapDraftResponse)
async def get_draft(course_id: str) -> KlMapDraftResponse:
    return _build_draft(_checked(course_id))


@router.post("/drafts/{course_id}/revalidate", response_model=KlMapDraftResponse)
async def revalidate_draft(course_id: str) -> KlMapDraftResponse:
    """Re-read the draft YAML after a hand-edit and re-run the content rules."""
    return _build_draft(_checked(course_id))


@router.post("/drafts/{course_id}/approve", response_model=KlMapDraftResponse)
async def decide_draft(course_id: str, decision: DraftDecision) -> KlMapDraftResponse:
    """Approve or reject a draft. Approval publishes it; rejection only records.

    Refuses approval while the loader reports errors — a map that fails a Tech
    Spec §2.3 rule must never serve traffic, whatever the UI allowed the
    reviewer to click.
    """
    course_id = _checked(course_id)
    draft = _build_draft(course_id)
    errors = draft.report.get("errors", [])

    if decision.approve and errors:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot approve {course_id}: the draft has {len(errors)} blocking error(s). "
                "Fix them in the YAML and revalidate. "
                + "; ".join(str(issue.get("message", "")) for issue in errors[:3])
            ),
        )

    metadata = _read_metadata(course_id)
    metadata.update(
        {
            "course_id": course_id,
            "status": STATUS_APPROVED if decision.approve else STATUS_REJECTED,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "review_note": decision.note,
        }
    )

    if decision.approve:
        klmap, _ = load_klmap_with_report(draft_path(course_id, drafts_dir()))
        if klmap is None:  # pragma: no cover — guarded by the error check above
            raise HTTPException(status_code=409, detail="Draft is not loadable.")
        published = save_klmap(klmap, live_path(course_id))
        metadata["published_to"] = str(published)
        logger.info("KL Map for %s approved and published to %s", course_id, published)

    _write_metadata(course_id, metadata)
    return _build_draft(course_id)


@router.get("")
async def get_live_klmap(course_id: str) -> dict[str, Any]:
    """The course's live map, with where it came from.

    `approved` is *reviewed and published through this API*; nothing else earns
    it. `source` says which file answered — `live`, `draft` or `seed` — so the
    panel can tell a reviewed map from the shipped seed rather than inferring it
    from a boolean that was never designed to carry that distinction (security
    review L4).

    Resolution order is the chat path's, so the panel shows what the tutor is
    teaching from: published live map → pending draft → the seed pack, for the
    seed's own course id only. The seed is `approved: false` because no reviewer
    approved it; it is hand-authored content that ships with the product. Any
    other course with no live map and no draft is an honest 404, decided by the
    id before any YAML is read (review m4).
    """
    course_id = _checked(course_id)
    live = live_path(course_id)
    if live.is_file():
        klmap, report = load_klmap_with_report(live)
        if klmap is not None:
            return {**_wire(klmap), "approved": True, "source": SOURCE_LIVE}
        # Chat is teaching from the seed for this course right now; say so
        # somewhere an operator will look, since the response cannot.
        logger.warning(
            "live map %s for %s is not loadable (%d error(s)); falling through",
            live,
            course_id,
            len(report.errors),
        )

    draft = draft_path(course_id, drafts_dir())
    if draft.is_file():
        klmap, _ = load_klmap_with_report(draft)
        if klmap is None:
            raise HTTPException(
                status_code=409,
                detail=f"The draft map for {course_id!r} has validation errors and is not loadable.",
            )
        return {**_wire(klmap), "approved": False, "source": SOURCE_DRAFT}

    if course_id == SEED_COURSE_ID:
        seed = seed_klmap_path()
        if seed.is_file():
            klmap, _ = load_klmap_with_report(seed)
            if klmap is not None:
                return {**_wire(klmap), "approved": False, "source": SOURCE_SEED}

    raise HTTPException(status_code=404, detail=f"No KL Map for course {course_id!r}.")


#: `source` values on `GET /api/klmap`; mirrored by `KlMapSource` in web/src/types.ts.
SOURCE_LIVE = "live"
SOURCE_DRAFT = "draft"
SOURCE_SEED = "seed"


def _wire(klmap: KLMap) -> dict[str, Any]:
    return {
        "course_id": klmap.course_id,
        "course_name": klmap.course_name,
        "nodes": [node.model_dump(mode="json") for node in klmap.nodes],
        "edges": [edge.model_dump(by_alias=True, mode="json") for edge in klmap.edges],
    }


__all__ = [
    "COURSE_ID_MAX_LENGTH",
    "COURSE_ID_PATTERN",
    "SEED_COURSE_ID",
    "SOURCE_DRAFT",
    "SOURCE_LIVE",
    "SOURCE_SEED",
    "STATUS_APPROVED",
    "STATUS_PENDING",
    "STATUS_REJECTED",
    "DraftDecision",
    "KlMapDraftResponse",
    "drafts_dir",
    "live_path",
    "router",
    "seed_klmap_path",
]
