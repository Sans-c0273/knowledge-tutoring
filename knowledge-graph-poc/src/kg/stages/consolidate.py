"""S4 consolidate (DESIGN §6, §8.5, §9; R7, R16).

Candidates from every chunk are merged by exact normalised title (``norm_title``);
a title may also match an alias (in either direction), but two aliases never match
each other — ``PDF`` listed as an alias of two different concepts must stay two nodes
and is left to dedup. The registry is the source of truth for identity: a candidate
whose title or alias names an active row reuses that permanent id (``is_new=False``),
an id is never allocated for a title/alias that already names an active row, and
when the in-file grouping and the registry disagree the candidate is routed by the
registry id. A candidate that bridges two in-file groups merges them, so every draft
carries a distinct id. An alias that already names a *different* active row is
dropped (with a warning) rather than made ambiguous. New titles are allocated ids in
first-occurrence order, which is deterministic for a given input. No model call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kg.normalise import norm_title
from kg.notes import NodeSource
from kg.registry import Registry
from kg.stages.atomize import Candidate


@dataclass
class NodeDraft:
    id: str
    title: str
    aliases: list[str]
    definition: str
    sources: list[NodeSource]
    review: list[str]
    chunk_ids: list[str]
    source_files: list[str]
    is_new: bool
    provider: str | None = field(default=None, compare=False)
    model: str | None = field(default=None, compare=False)
    prompt_version: str | None = field(default=None, compare=False)


class _Group:
    """One future draft: the first candidate's surface form plus everything merged into it."""

    def __init__(self, cand: Candidate, existing_id: str | None) -> None:
        self.title = cand.title
        self.key = norm_title(cand.title)
        self.existing_id = existing_id
        self.definition = cand.definition
        self.aliases: list[str] = []
        self.alias_keys: set[str] = set()
        self.sources: list[NodeSource] = []
        self.review: list[str] = []
        self.chunk_ids: list[str] = []
        self.source_files: list[str] = []
        self.provider, self.model, self.prompt_version = cand.provider, cand.model, cand.prompt_version
        self.merge(cand)

    def _add_alias(self, alias: str) -> None:
        k = norm_title(alias)
        if k and k != self.key and k not in self.alias_keys:
            self.aliases.append(alias)
            self.alias_keys.add(k)

    def _add_sources(self, sources: list[NodeSource]) -> None:
        seen = {(s.locator, s.quote) for s in self.sources}
        for s in sources:
            if (s.locator, s.quote) not in seen:
                self.sources.append(s)
                seen.add((s.locator, s.quote))

    def _add_lists(self, review: list[str], chunk_ids: list[str], source_files: list[str]) -> None:
        for flag in review:
            if flag not in self.review:
                self.review.append(flag)
        for cid in chunk_ids:
            if cid and cid not in self.chunk_ids:
                self.chunk_ids.append(cid)
        for src in source_files:
            if src and src not in self.source_files:
                self.source_files.append(src)

    def merge(self, cand: Candidate) -> None:
        if not self.definition and cand.definition:
            self.definition = cand.definition
        for alias in cand.aliases:
            self._add_alias(alias)
        if cand.title:
            # A differently-worded title that matched through an alias is itself an alias.
            self._add_alias(cand.title)
        self._add_sources(cand.sources)
        self._add_lists(cand.review, [cand.chunk_id], [cand.source])

    def absorb(self, other: _Group) -> None:
        """Union another in-file group into this one (this group's title and definition win)."""
        if not self.definition and other.definition:
            self.definition = other.definition
        self._add_alias(other.title)
        for alias in other.aliases:
            self._add_alias(alias)
        self._add_sources(other.sources)
        self._add_lists(other.review, other.chunk_ids, other.source_files)
        if self.existing_id is None:
            self.existing_id = other.existing_id


def _resolve_existing(cand: Candidate, registry: Registry) -> str | None:
    """Active registry id named by the candidate: title vs title/alias, or alias vs title (never alias vs alias)."""
    existing = registry.find(cand.title)
    if existing is not None:
        return existing
    for alias in cand.aliases:
        existing = registry.find_by_title(alias)
        if existing is not None:
            return existing
    return None


def consolidate(candidates: list[Candidate], registry: Registry, *, run: str, warnings: list[str] | None = None) -> list[NodeDraft]:
    """Merge `candidates` into drafts with permanent ids. Dropped aliases are reported through `warnings`."""
    groups: list[_Group] = []
    by_title: dict[str, _Group] = {}  # normalised title -> group
    by_alias: dict[str, _Group] = {}  # normalised alias -> group (matched against titles only)
    by_existing: dict[str, _Group] = {}  # registry id -> group
    notes: list[str] = warnings if warnings is not None else []

    def in_file_matches(cand: Candidate) -> list[_Group]:
        key = norm_title(cand.title)
        found: list[_Group] = []
        for g in (by_title.get(key), by_alias.get(key), *(by_title.get(k) for k in map(norm_title, cand.aliases))):
            if g is not None and g not in found:
                found.append(g)
        return found

    def index(group: _Group) -> None:
        by_title[group.key] = group
        for k in group.alias_keys:
            by_alias.setdefault(k, group)
        if group.existing_id is not None:
            by_existing[group.existing_id] = group

    def repoint(old: _Group, new: _Group) -> None:
        for table in (by_title, by_alias, by_existing):
            for k, g in list(table.items()):
                if g is old:
                    table[k] = new

    for cand in candidates:
        key = norm_title(cand.title)
        if not key:
            continue
        existing = _resolve_existing(cand, registry)
        matches: list[_Group] = []
        if existing is not None and existing in by_existing:
            matches.append(by_existing[existing])
        for g in in_file_matches(cand):
            if g in matches:
                continue
            # Registry identity wins: the candidate's own registry id, else the first matched group's id,
            # decides; an in-file group already tied to a different id is not this concept.
            identity = existing if existing is not None else next((m.existing_id for m in matches if m.existing_id is not None), None)
            if g.existing_id is not None and identity is not None and g.existing_id != identity:
                continue
            matches.append(g)
        if not matches:
            group = _Group(cand, existing)
            groups.append(group)
        else:
            primary = next((g for g in matches if g.existing_id is not None), matches[0])
            for other in matches:
                if other is not primary:
                    primary.absorb(other)
                    groups.remove(other)
                    repoint(other, primary)
            primary.merge(cand)
            if primary.existing_id is None and existing is not None:
                primary.existing_id = existing  # an in-file group turns out to name a registered node
            group = primary
        index(group)

    drafts: list[NodeDraft] = []
    for g in groups:
        node_id = g.existing_id
        if node_id is None:
            # Never allocate when the title or any alias already names an active row: attach instead.
            node_id = registry.find(g.title)
            if node_id is None:
                node_id = next((r for r in (registry.find_by_title(a) for a in g.aliases) if r is not None), None)
        aliases = list(g.aliases)
        if node_id is not None:
            row = next(r for r in registry.rows if r.id == node_id)
            if row.norm_title != g.key:
                aliases.append(g.title)  # a differently-worded title that matched a registered node is its alias
        aliases = _drop_foreign_aliases(aliases, node_id, g.title, registry, notes)
        if node_id is not None:
            is_new = False
            if aliases:
                registry.add_aliases(node_id, aliases)
        else:
            node_id, is_new = registry.allocate(g.title, run=run, aliases=aliases).id, True
        drafts.append(
            NodeDraft(
                id=node_id,
                title=g.title,
                aliases=aliases,
                definition=g.definition,
                sources=list(g.sources),
                review=list(g.review),
                chunk_ids=list(g.chunk_ids),
                source_files=list(g.source_files),
                is_new=is_new,
                provider=g.provider,
                model=g.model,
                prompt_version=g.prompt_version,
            )
        )
    ids = [d.id for d in drafts]
    if len(set(ids)) != len(ids):
        raise AssertionError(f"consolidate produced duplicate draft ids: {sorted(i for i in set(ids) if ids.count(i) > 1)}")
    return drafts


def _drop_foreign_aliases(aliases: list[str], node_id: str | None, title: str, registry: Registry, notes: list[str]) -> list[str]:
    """Aliases minus any that already name a *different* active row (an alias is never shared)."""
    kept: list[str] = []
    for alias in aliases:
        owner = registry.find(alias)
        if owner is not None and owner != node_id:
            notes.append(f"alias {alias!r} of {title!r} dropped: it already names active node {owner}")
        else:
            kept.append(alias)
    return kept


__all__ = ["NodeDraft", "consolidate"]
