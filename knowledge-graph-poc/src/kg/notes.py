"""Node files: render, parse, wikilink weaving (DESIGN §8.2–8.4, §12; D13; R7, R9, R10, R11).

Frontmatter is canonical; ``## Relations`` and ``## Source`` are regenerated from
it on every write, and the definition is stored in memory as PLAIN prose — links
are stripped on read and woven on write, which is what makes re-rendering
byte-identical (R11). The two worked examples in DESIGN §8.3 / §8.4 are the
byte-exact targets of ``render``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import yaml

from kg.schemas import SCHEMAS

REQUIRED_KEYS: tuple[str, ...] = (
    "id",
    "title",
    "aliases",
    "schema",
    "created",
    "updated",
    "run",
    "prompt_version",
    "provider",
    "model",
    "review",
    "sources",
    "edges",
)
FRONTMATTER_ORDER: tuple[str, ...] = REQUIRED_KEYS[:4] + ("origin",) + REQUIRED_KEYS[4:]

_WIKILINK = re.compile(r"\[\[([^\[\]|]*?)(?:\|([^\[\]]*?))?\]\]")
# Word boundary that also treats Thai combining marks as word characters.
_WORD_CHAR = r"[\w฀-๿]"
_LINE_BREAKS = re.compile(r"[\r\n\x0b\x0c\x85\u2028\u2029]+")
#: A body line starting with one of these would be read back as structure, not prose (R4/R11).
_STRUCTURAL_PREFIXES: tuple[str, ...] = ("#", ">", "---")
#: An HTML comment hides text from every Markdown renderer; an unterminated one hides the rest of the text.
_HTML_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)


class NoteError(Exception):
    """A note on disk does not have the §8.2 shape."""


# ------------------------------------------------------------------- models


@dataclass
class NodeSource:
    locator: str
    file: str
    quote: str

    def __post_init__(self) -> None:
        # A locator is one `### <locator>` heading line; a line break inside it would let
        # the tail be read back as a forged heading or quote.
        self.locator = _LINE_BREAKS.sub(" ", self.locator).strip()


def safe_prose(text: str) -> str:
    """Model-authored prose made safe to embed in a note body: one line, no structure, no hidden text.

    HTML comments (`<!-- … -->`, closed or not) are removed and every remaining `<` is
    backslash-escaped, so text a renderer would hide cannot persist into a note or be
    fed back into the dedup prompt. Whitespace (including line breaks) collapses to
    single spaces, so a definition can never smuggle in a `## Source` section; a line
    that would still start with `#`, `>` or `---` is escaped with a backslash, which
    Markdown renders as the literal.
    """
    prose = _HTML_COMMENT.sub(" ", strip_wikilinks(text or ""))
    prose = " ".join(prose.split()).replace("<", "\\<")
    if prose.startswith(_STRUCTURAL_PREFIXES):
        prose = "\\" + prose
    return prose


@dataclass
class NodeEdge:
    type: str
    target: str
    relevance: int | None = None


@dataclass(frozen=True)
class LinkTarget:
    stem: str
    title: str
    aliases: tuple[str, ...] = ()


@dataclass
class Node:
    id: str
    title: str
    aliases: list[str]
    schema: str
    origin: str | None
    created: str
    updated: str
    run: str
    prompt_version: str
    provider: str
    model: str
    review: list[str]
    sources: list[NodeSource]
    edges: list[NodeEdge]
    definition: str


# ------------------------------------------------------------- YAML helpers


class _PlainLoader(yaml.SafeLoader):
    """SafeLoader without timestamp resolution: dates and datetimes stay strings."""


_PlainLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _load_yaml(text: str):
    return yaml.load(text, Loader=_PlainLoader)  # noqa: S506 - SafeLoader subclass


def _scalar(value: str, *, flow: bool = False) -> str:
    """Emit `value` as a plain YAML scalar when it round-trips unchanged, else double-quoted."""
    probe = f"k: [{value}]" if flow else f"k: {value}"
    if value and "\n" not in value and value == value.strip():
        try:
            back = _load_yaml(probe)
        except yaml.YAMLError:
            back = None
        expected = {"k": [value]} if flow else {"k": value}
        if back == expected:
            return value
    return json.dumps(value, ensure_ascii=False)


def _flow_list(items: Iterable[str]) -> str:
    return "[" + ", ".join(_scalar(i, flow=True) for i in items) + "]"


# ------------------------------------------------------------------- render


def _edge_types_in_order(node: Node) -> list[str]:
    schema = SCHEMAS.get(node.schema)
    order = list(schema.edges) if schema is not None else []
    seen = [t for t in order if any(e.type == t for e in node.edges)]
    seen += [e.type for e in node.edges if e.type not in seen and e.type not in order]
    return list(dict.fromkeys(seen))


def _frontmatter(node: Node) -> str:
    lines = [
        f"id: {_scalar(node.id)}",
        f"title: {_scalar(node.title)}",
        f"aliases: {_flow_list(node.aliases)}",
        f"schema: {_scalar(node.schema)}",
    ]
    if node.origin is not None:
        lines.append(f"origin: {_scalar(node.origin)}")
    lines += [
        f"created: {_scalar(node.created)}",
        f"updated: {_scalar(node.updated)}",
        f"run: {_scalar(node.run)}",
        f"prompt_version: {_scalar(node.prompt_version)}",
        f"provider: {_scalar(node.provider)}",
        f"model: {_scalar(node.model)}",
        f"review: {_flow_list(node.review)}",
    ]
    if node.sources:
        lines.append("sources:")
        for s in node.sources:
            lines.append(f"  - locator: {_scalar(s.locator)}")
            lines.append(f"    file: {_scalar(s.file)}")
    else:
        lines.append("sources: []")
    types = _edge_types_in_order(node)
    if types:
        lines.append("edges:")
        for t in types:
            edges = [e for e in node.edges if e.type == t]
            if any(e.relevance is not None for e in edges):
                lines.append(f"  {t}:")
                for e in edges:
                    rel = "null" if e.relevance is None else str(e.relevance)
                    lines.append(f"    - {{id: {_scalar(e.target, flow=True)}, relevance: {rel}}}")
            else:
                lines.append(f"  {t}: {_flow_list(e.target for e in edges)}")
    else:
        lines.append("edges: {}")
    return "\n".join(lines)


def _quote_lines(quote: str) -> list[str]:
    return [f"> {line}" if line else ">" for line in (quote.split("\n") if quote else [""])]


def render(node: Node, targets: Mapping[str, LinkTarget]) -> str:
    """The §8.2 file text for `node`.

    `targets` maps node id -> LinkTarget for every node that may be linked. Calling
    with an empty mapping while the node has edges is a programming error and is
    refused; a partial mapping degrades an unknown target to an id-only link
    (`[[<id>|<id>]]`) so a hand-edited or dangling note can still be written and
    then reported by the `dangling` gate. The pipeline itself never hits that
    path: unknown ids are rejected before anything is written.
    """
    if node.edges and not targets:
        raise KeyError(f"node {node.id}: no link targets given for edge target(s) {', '.join(e.target for e in node.edges)}")
    body = [f"# {node.title}", "", "## Definition", weave(strip_wikilinks(node.definition), targets, exclude=node.id), "", "## Relations"]
    for t in _edge_types_in_order(node):
        body.append(f"### {t}")
        for e in (e for e in node.edges if e.type == t):
            tgt = targets.get(e.target) or LinkTarget(stem=e.target, title=e.target)
            link = f"- [[{tgt.stem}|{tgt.title}]]"
            body.append(f"{link} — relevance {e.relevance}" if e.relevance is not None else link)
    body += ["", "## Source"]
    for s in node.sources:
        body.append(f"### {s.locator}")
        body.extend(_quote_lines(s.quote))
    return "---\n" + _frontmatter(node) + "\n---\n" + "\n".join(body) + "\n"


# -------------------------------------------------------------------- parse


def _split(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise NoteError("note has no YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise NoteError("frontmatter is not closed by a '---' line")
    return text[4:end], text[end + 5 :]


def _section(body: str, name: str) -> str | None:
    marker = f"\n## {name}\n"
    padded = "\n" + body
    start = padded.find(marker)
    if start < 0:
        return None
    rest = padded[start + len(marker) :]
    nxt = re.search(r"\n## ", "\n" + rest)
    return rest if nxt is None else rest[: nxt.start()]


def _parse_sources(section: str | None) -> dict[str, list[str]]:
    """locator -> quotes in order, from the `### <locator>` / `> quote` pairs."""
    out: dict[str, list[str]] = {}
    current: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if current is not None:
            out.setdefault(current, []).append("\n".join(lines))

    for line in (section or "").split("\n"):
        if line.startswith("### "):
            flush()
            current, lines = line[4:], []
        elif current is not None and (line.startswith("> ") or line == ">"):
            lines.append(line[2:] if line.startswith("> ") else "")
    flush()
    return out


def _require_str(fm: Mapping, key: str) -> str:
    value = fm.get(key)
    if isinstance(value, bool) or value is None:
        raise NoteError(f"frontmatter key {key!r} must be a string")
    return str(value)


def _str_list(fm: Mapping, key: str) -> list[str]:
    value = fm.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise NoteError(f"frontmatter key {key!r} must be a list")
    return [str(v) for v in value]


def _parse_edges(raw) -> list[NodeEdge]:
    if raw is None:
        return []
    if not isinstance(raw, Mapping):
        raise NoteError("frontmatter 'edges' must be a mapping of edge type to targets")
    edges: list[NodeEdge] = []
    for etype, items in raw.items():
        if not isinstance(items, list):
            raise NoteError(f"edges.{etype} must be a list")
        for item in items:
            if isinstance(item, Mapping):
                if "id" not in item:
                    raise NoteError(f"edges.{etype}: scored edge entry without an id")
                rel = item.get("relevance")
                edges.append(NodeEdge(type=str(etype), target=str(item["id"]), relevance=rel))
            else:
                edges.append(NodeEdge(type=str(etype), target=str(item)))
    return edges


def parse(text: str) -> Node:
    """Read a §8.2 note back into a `Node`; the definition comes back as plain prose."""
    fm_text, body = _split(text)
    try:
        fm = _load_yaml(fm_text)
    except yaml.YAMLError as exc:
        raise NoteError(f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(fm, Mapping):
        raise NoteError("frontmatter must be a mapping")
    missing = [k for k in REQUIRED_KEYS if k not in fm]
    if missing:
        raise NoteError(f"frontmatter is missing required key(s): {', '.join(missing)}")
    schema = _require_str(fm, "schema")
    if schema not in SCHEMAS:
        raise NoteError(f"unknown schema {schema!r}; expected one of {', '.join(SCHEMAS)}")
    origin = fm.get("origin")
    if origin is not None:
        allowed = SCHEMAS[schema].origin_values or ()
        if str(origin) not in allowed:
            raise NoteError(f"origin {origin!r} is not valid for schema {schema!r}")
        origin = str(origin)

    raw_sources = fm.get("sources")
    if not isinstance(raw_sources, list):
        raise NoteError("frontmatter 'sources' must be a list")
    quotes_by_locator = _parse_sources(_section(body, "Source"))
    used: dict[str, int] = {}
    sources: list[NodeSource] = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping) or "locator" not in raw:
            raise NoteError("each source needs a 'locator'")
        locator = str(raw["locator"])
        n = used.get(locator, 0)
        quotes = quotes_by_locator.get(locator, [])
        quote = quotes[n] if n < len(quotes) else ""
        used[locator] = n + 1
        sources.append(NodeSource(locator=locator, file=str(raw.get("file", "")), quote=quote))

    definition = _section(body, "Definition")
    if definition is None:
        raise NoteError("note has no '## Definition' section")
    return Node(
        id=_require_str(fm, "id"),
        title=_require_str(fm, "title"),
        aliases=_str_list(fm, "aliases"),
        schema=schema,
        origin=origin,
        created=_require_str(fm, "created"),
        updated=_require_str(fm, "updated"),
        run=_require_str(fm, "run"),
        prompt_version=_require_str(fm, "prompt_version"),
        provider=_require_str(fm, "provider"),
        model=_require_str(fm, "model"),
        review=_str_list(fm, "review"),
        sources=sources,
        edges=_parse_edges(fm.get("edges")),
        definition=strip_wikilinks(definition).strip(),
    )


# ---------------------------------------------------------------- wikilinks


def strip_wikilinks(text: str) -> str:
    """Replace every ``[[target|alias]]`` / ``[[target]]`` with its display text, however nested."""
    previous = None
    while previous != text:
        previous = text
        text = _WIKILINK.sub(lambda m: m.group(2) if m.group(2) is not None else m.group(1), text)
    return text


def weave(text: str, targets: Mapping[str, LinkTarget], *, exclude: str | None = None) -> str:
    """Link the first-class names of `targets` (titles and aliases) wherever they occur as whole words.

    Case-insensitive; the prose keeps its own casing; longest name wins; text already
    inside a wikilink is never touched, so the operation is idempotent.
    """
    names: dict[str, LinkTarget] = {}
    for node_id, tgt in targets.items():
        if node_id == exclude:
            continue
        for name in (tgt.title, *tgt.aliases):
            key = name.strip().casefold()
            if key and key not in names:
                names[key] = tgt
    if not names or not text:
        return text
    alternation = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    pattern = re.compile(rf"(?<!{_WORD_CHAR})(?:{alternation})(?!{_WORD_CHAR})", re.IGNORECASE)

    def link(m: re.Match[str]) -> str:
        shown = m.group(0)
        tgt = names.get(shown.casefold())
        if tgt is None:  # casefold/IGNORECASE mismatch (rare): find it the slow way
            tgt = next(t for n, t in names.items() if re.fullmatch(re.escape(n), shown, re.IGNORECASE))
        return f"[[{tgt.stem}|{shown}]]"

    out: list[str] = []
    pos = 0
    for existing in _WIKILINK.finditer(text):
        out.append(pattern.sub(link, text[pos : existing.start()]))
        out.append(existing.group(0))
        pos = existing.end()
    out.append(pattern.sub(link, text[pos:]))
    return "".join(out)


__all__ = [
    "FRONTMATTER_ORDER",
    "LinkTarget",
    "Node",
    "NodeEdge",
    "NodeSource",
    "NoteError",
    "REQUIRED_KEYS",
    "parse",
    "render",
    "safe_prose",
    "strip_wikilinks",
    "weave",
]
