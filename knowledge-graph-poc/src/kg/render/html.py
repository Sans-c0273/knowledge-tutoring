"""Single-file HTML relationship view (PRD R14, R9, R11; DESIGN §13, §8.6; D9).

`graph.html` = `template.html` with three substitutions: the page title, the
vendored `force-graph.min.js` (inlined, never referenced) and the graph payload
as a `<script id="graph-data" type="application/json">` block. The payload is the
§8.6 `_index.json` document plus `styles` (edge styling for the selected schema
only); `_index.json` itself is regenerated from the notes on every render.

Offline by construction: no `<script src>`, no `<link href>`, no fonts, no
`fetch`; a CSP meta tag (`default-src 'none'`, inline script/style, `img-src data:`)
backs that up in the browser. The JSON is made safe inside a script element by
escaping `<`, `>` and `&` as `\\uXXXX` sequences, so a title containing `</script>`
cannot terminate the data block. Placeholders are substituted in ONE pass, so a
corpus name that happens to contain a placeholder string is inserted verbatim rather
than expanded into JS or JSON. Output is byte-identical for identical input; the only
clock is the injected `now` (`meta.generated`), omitted when `now` is None.

`out_path` may be any non-forbidden location for library callers; when a `Sandbox`
is supplied (the CLI does) it must additionally lie inside the sandbox's graph root.
"""

from __future__ import annotations

import html as html_mod
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kg import graph_io
from kg.paths import Sandbox, SandboxViolation, is_denied
from kg.schemas import EdgeSchema

_HERE = Path(__file__).resolve().parent
TEMPLATE_PATH: Path = _HERE / "template.html"
VENDOR_JS_PATH: Path = _HERE.parents[2] / "vendor" / "force-graph.min.js"

#: Edge styling for every edge type of both schemas (DESIGN §13). `dash` is a canvas
#: line-dash pattern ([] = solid); `arrow` marks directed types. `related_to` width
#: and opacity are additionally scaled by relevance in the page script.
EDGE_STYLES: dict[str, dict[str, Any]] = {
    "related_to": {"color": "#7aa2f7", "dash": [], "arrow": False, "label": "related to (width/opacity by relevance)"},
    "part_of": {"color": "#9ece6a", "dash": [], "arrow": True, "label": "part of"},
    "same_as": {"color": "#bb9af7", "dash": [6, 4], "arrow": False, "label": "same as"},
    "prerequisite_of": {"color": "#f7768e", "dash": [], "arrow": True, "label": "prerequisite of"},
    "example_of": {"color": "#e0af68", "dash": [2, 3], "arrow": True, "label": "example of"},
    "refines": {"color": "#7dcfff", "dash": [], "arrow": True, "label": "refines"},
    "supersedes": {"color": "#ff9e64", "dash": [8, 3], "arrow": True, "label": "supersedes"},
}

_TITLE_TOKEN = "__KG_TITLE__"
_SUBTITLE_TOKEN = "__KG_SUBTITLE__"
_VENDOR_TOKEN = "__KG_VENDOR_JS__"
_PAYLOAD_TOKEN = "__KG_GRAPH_JSON__"


def _json_for_script(payload: dict[str, Any]) -> str:
    """JSON that is valid inside `<script type="application/json">` whatever the strings contain."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def build_payload(graph: graph_io.Graph, schema: EdgeSchema) -> dict[str, Any]:
    """The §8.6 index plus per-edge-type styles scoped to `schema`."""
    payload = graph.to_index()
    payload["styles"] = {etype: dict(EDGE_STYLES[etype]) for etype in schema.edges if etype in EDGE_STYLES}
    return payload


def _confine(out_path: Path, sandbox: Sandbox) -> None:
    """`out_path` must be inside the sandbox's graph root (lexically, then via Sandbox.resolve's symlink checks)."""
    graph_root = sandbox.root("graph")
    try:
        rel = Path(os.path.abspath(out_path)).relative_to(graph_root)
    except ValueError as exc:
        raise SandboxViolation(f"graph.html must be written inside the sandboxed graph folder {graph_root}: {out_path}") from exc
    sandbox.resolve("graph", rel)


def _substitute(template: str, tokens: dict[str, str]) -> str:
    """Replace every placeholder in one pass; inserted text is never rescanned for placeholders."""
    pattern = re.compile("|".join(re.escape(t) for t in tokens))
    return pattern.sub(lambda m: tokens[m.group(0)], template)


def render(graph_dir: Path, out_path: Path, *, schema: EdgeSchema, now: datetime | None = None, sandbox: Sandbox | None = None) -> Path:
    graph_dir, out_path = Path(graph_dir), Path(out_path)
    if is_denied(out_path) or is_denied(Path(out_path).resolve()):
        raise SandboxViolation(f"refusing to write graph.html into a forbidden location: {out_path}")
    if sandbox is not None:
        _confine(out_path, sandbox)
    graph = graph_io.load_graph(graph_dir)  # refuses a forbidden graph_dir, raises if missing
    if now is not None:
        graph.meta["generated"] = now.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    graph_io.write_index(graph, graph_dir / graph_io.INDEX_FILE)

    payload = build_payload(graph, schema)
    corpus = graph.meta.get("corpus") or "graph"
    title = f"{corpus} — knowledge graph"
    subtitle = f"schema {schema.name} · {len(graph.nodes)} nodes · {len(graph.edges)} edges"

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    vendor = VENDOR_JS_PATH.read_text(encoding="utf-8").strip()
    for token in (_TITLE_TOKEN, _SUBTITLE_TOKEN, _VENDOR_TOKEN, _PAYLOAD_TOKEN):
        if template.count(token) < 1:
            raise RuntimeError(f"{TEMPLATE_PATH.name}: placeholder {token} missing")
    if "</script" in vendor.lower():
        raise RuntimeError(f"{VENDOR_JS_PATH.name} contains '</script'; it cannot be inlined safely")
    page = _substitute(
        template,
        {
            _TITLE_TOKEN: html_mod.escape(title),
            _SUBTITLE_TOKEN: html_mod.escape(subtitle),
            _VENDOR_TOKEN: vendor,
            _PAYLOAD_TOKEN: _json_for_script(payload),
        },
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path


__all__ = ["EDGE_STYLES", "TEMPLATE_PATH", "VENDOR_JS_PATH", "build_payload", "render"]
