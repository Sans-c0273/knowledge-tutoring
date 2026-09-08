"""Permanent node ids and ``_registry.yaml`` (DESIGN §8.2, §8.5; R7).

Ids are ``<corpus.slug>-<NNNN>``; ``next_seq`` only ever increases, so a retired id
is never handed out again. Rows are looked up by normalised title or alias
(``kg.normalise.norm_title``); retired rows are listed forever but never matched.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kg.convert._common import write_files_atomically
from kg.normalise import norm_title

STATUSES: tuple[str, ...] = ("active", "retired")
MAX_STEM_LEN = 120
_NON_SLUG = re.compile(r"[^a-z0-9]+")


class RegistryError(Exception):
    """A registry operation was refused or the file on disk is unusable."""


def slugify(title: str) -> str:
    """Lower-case ASCII, hyphen-separated; accents folded, everything else dropped.

    Thai (and any other non-Latin script) slugifies to the empty string, in which
    case the note is named by its id alone (``stem_for``).
    """
    decomposed = unicodedata.normalize("NFKD", title or "")
    ascii_only = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").casefold()
    return _NON_SLUG.sub("-", ascii_only).strip("-")


def stem_for(node_id: str, title: str) -> str:
    """File stem ``<id>-<slug>`` (or ``<id>`` for an empty slug), bounded to ``MAX_STEM_LEN``."""
    slug = slugify(title)
    if not slug:
        return node_id
    budget = MAX_STEM_LEN - len(node_id) - 1
    if len(slug) > budget:
        cut = slug[:budget]
        slug = cut.rsplit("-", 1)[0] if "-" in cut else cut
        slug = slug.strip("-")
    return f"{node_id}-{slug}" if slug else node_id


@dataclass
class RegistryRow:
    id: str
    title: str
    norm_title: str
    file: str
    status: str
    created_run: str
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Registry:
    """In-memory registry; ``save``/``load`` round-trip ``_registry.yaml``."""

    def __init__(self, corpus: str, schema: str, *, next_seq: int = 1, rows: Iterable[RegistryRow] = ()) -> None:
        if not corpus or not corpus.strip():
            raise RegistryError("registry corpus slug must be non-empty")
        self.corpus = corpus
        self.schema = schema
        self._rows: list[RegistryRow] = list(rows)
        highest = max((self._seq_of(r.id) for r in self._rows), default=0)
        if next_seq < 1:
            raise RegistryError(f"next_seq must be >= 1 (got {next_seq})")
        self._next_seq = max(next_seq, highest + 1)

    # ------------------------------------------------------------ helpers

    def _seq_of(self, node_id: str) -> int:
        prefix = f"{self.corpus}-"
        if not node_id.startswith(prefix):
            return 0
        tail = node_id[len(prefix) :]
        return int(tail) if tail.isdigit() else 0

    def _row(self, node_id: str) -> RegistryRow:
        for row in self._rows:
            if row.id == node_id:
                return row
        raise RegistryError(f"unknown node id {node_id!r}")

    # ---------------------------------------------------------------- API

    @property
    def next_seq(self) -> int:
        return self._next_seq

    @property
    def rows(self) -> list[RegistryRow]:
        return list(self._rows)

    def active(self) -> list[RegistryRow]:
        return [r for r in self._rows if r.status == "active"]

    def find(self, title_or_alias: str) -> str | None:
        """Id of the active row whose normalised title or alias equals the query, else None."""
        key = norm_title(title_or_alias)
        if not key:
            return None
        for row in self._rows:
            if row.status != "active":
                continue
            if row.norm_title == key or any(norm_title(a) == key for a in row.aliases):
                return row.id
        return None

    def find_by_title(self, title: str) -> str | None:
        """Id of the active row whose normalised *title* equals the query (aliases not consulted), else None."""
        key = norm_title(title)
        if not key:
            return None
        return next((row.id for row in self._rows if row.status == "active" and row.norm_title == key), None)

    def allocate(self, title: str, *, run: str, aliases: Iterable[str] = ()) -> RegistryRow:
        clean_title = " ".join((title or "").split())
        key = norm_title(clean_title)
        if not key:
            raise RegistryError("cannot allocate an id for an empty title")
        existing = self.find(clean_title)
        if existing is not None:
            raise RegistryError(f"title {clean_title!r} is already registered as {existing}; use find()")
        node_id = f"{self.corpus}-{self._next_seq:04d}"
        row = RegistryRow(
            id=node_id,
            title=clean_title,
            norm_title=key,
            file=f"nodes/{stem_for(node_id, clean_title)}.md",
            status="active",
            created_run=run,
            aliases=[a for a in dict.fromkeys(" ".join(x.split()) for x in aliases) if a and norm_title(a) != key],
        )
        self._rows.append(row)
        self._next_seq += 1
        return row

    def add_aliases(self, node_id: str, aliases: Iterable[str]) -> RegistryRow:
        """Union new aliases into a row (never the title itself, never duplicates by normalisation)."""
        row = self._row(node_id)
        known = {row.norm_title, *(norm_title(a) for a in row.aliases)}
        for alias in aliases:
            clean = " ".join((alias or "").split())
            key = norm_title(clean)
            if clean and key and key not in known:
                row.aliases.append(clean)
                known.add(key)
        return row

    def retire(self, node_id: str) -> RegistryRow:
        row = self._row(node_id)
        row.status = "retired"
        return row

    def discard(self, node_id: str, *, run: str) -> RegistryRow:
        """Remove a row allocated in `run` whose note was never written; the id is not reused.

        ``next_seq`` is unchanged, so the discarded id stays burnt. Rows from earlier
        runs are refused: those are retired, never removed.
        """
        row = self._row(node_id)
        if row.created_run != run or row.status != "active":
            raise RegistryError(f"{node_id} was not allocated in run {run!r}; retire it instead of discarding")
        self._rows.remove(row)
        return row

    def file_for(self, node_id: str) -> str:
        return self._row(node_id).file

    # ------------------------------------------------------------ on disk

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "schema": self.schema,
            "next_seq": self._next_seq,
            "nodes": [r.to_dict() for r in self._rows],
        }

    def dump(self) -> str:
        """The ``_registry.yaml`` text (deterministic for the same rows)."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False)

    def save(self, path: Path) -> Path:
        """Write atomically (``.tmp`` + ``os.replace``): a crash never leaves a truncated registry."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_files_atomically({path: self.dump()})
        return path

    @classmethod
    def load(cls, path: Path) -> Registry:
        path = Path(path)
        if not path.is_file():
            raise RegistryError(f"registry file not found: {path}")
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RegistryError(f"{path.name}: malformed YAML: {exc}") from exc
        if not isinstance(doc, dict):
            raise RegistryError(f"{path.name}: top level must be a mapping")
        for key in ("corpus", "schema", "next_seq", "nodes"):
            if key not in doc:
                raise RegistryError(f"{path.name}: missing key {key!r}")
        if not isinstance(doc["nodes"], list):
            raise RegistryError(f"{path.name}: 'nodes' must be a list")
        rows: list[RegistryRow] = []
        for i, raw in enumerate(doc["nodes"]):
            if not isinstance(raw, dict):
                raise RegistryError(f"{path.name}: nodes[{i}] must be a mapping")
            try:
                rows.append(
                    RegistryRow(
                        id=str(raw["id"]),
                        title=str(raw["title"]),
                        norm_title=norm_title(str(raw["title"])),  # recomputed: the stored value is informational
                        file=str(raw["file"]),
                        status=str(raw.get("status", "active")),
                        created_run=str(raw.get("created_run", "")),
                        aliases=[str(a) for a in (raw.get("aliases") or [])],
                    )
                )
            except KeyError as exc:
                raise RegistryError(f"{path.name}: nodes[{i}] is missing {exc.args[0]!r}") from exc
            if rows[-1].status not in STATUSES:
                raise RegistryError(f"{path.name}: nodes[{i}] has unknown status {rows[-1].status!r}")
        next_seq = doc["next_seq"]
        if isinstance(next_seq, bool) or not isinstance(next_seq, int):
            raise RegistryError(f"{path.name}: 'next_seq' must be an integer")
        return cls(str(doc["corpus"]), str(doc["schema"]), next_seq=next_seq, rows=rows)


__all__ = ["MAX_STEM_LEN", "Registry", "RegistryError", "RegistryRow", "STATUSES", "slugify", "stem_for"]
