"""WU4 — `kg.render.html`: the single-file HTML relationship view (PRD R14, R9, R11; DESIGN §13, D9, §8.6).

Interface (chosen, see wu45_fixtures docstring):
  render(graph_dir, out_path, *, schema: EdgeSchema, now: datetime | None = None) -> Path
  EDGE_STYLES, TEMPLATE_PATH, VENDOR_JS_PATH
  DOM contract: <script id="graph-data" type="application/json">, #search, #min-relevance, #edge-filters, #inspector

A browser is not available in the suite, so R14 "opens with no console errors" is approximated by:
(1) the vendored force-graph bytes are inlined, (2) no external resource reference of any kind,
(3) `node --check` parses every inline script when Node is installed (skipped otherwise).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kg.schemas import EDUCATION, GENERAL, SCHEMAS
from wu45_fixtures import (
    EDUCATION_EDGES,
    EDUCATION_IDS,
    GENERAL_EDGES,
    GENERAL_IDS,
    NOW,
    NOW_2,
    TEMPLATE,
    THAI_TITLE,
    VENDOR_JS,
    canonical_edges,
    graph_payload,
    inline_scripts,
    js_scripts,
    write_education_graph,
    write_general_graph,
)

EXTERNAL_HOST_RE = re.compile(r"https?://(?!www\.w3\.org/)", re.IGNORECASE)


@pytest.fixture
def gen_graph(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "graph"
    d.mkdir(parents=True)
    write_general_graph(d)
    return d


@pytest.fixture
def edu_graph(tmp_path: Path) -> Path:
    d = tmp_path / "edu" / "graph"
    d.mkdir(parents=True)
    write_education_graph(d)
    return d


def render_to(graph_dir: Path, schema=GENERAL, *, now=NOW) -> tuple[Path, str]:
    from kg.render.html import render

    out = graph_dir / "graph.html"
    got = render(graph_dir, out, schema=schema, now=now)
    assert Path(got) == out
    return out, out.read_text(encoding="utf-8")


# ---------------------------------------------------------- packaging (D9)


def test_D9_vendored_force_graph_is_present_in_the_repo():
    assert VENDOR_JS.is_file(), f"expected vendored force-graph 1.51.4 at {VENDOR_JS}"
    assert VENDOR_JS.stat().st_size > 50_000
    assert (VENDOR_JS.parent / "force-graph.LICENSE").is_file(), "DESIGN §20: vendor/force-graph.LICENSE"


def test_D9_template_lives_next_to_the_renderer_and_declares_utf8():
    assert TEMPLATE.is_file(), f"expected {TEMPLATE}"
    text = TEMPLATE.read_text(encoding="utf-8")
    assert re.search(r'<meta\s+charset=["\']?utf-8', text, re.IGNORECASE)


def test_D9_module_constants_point_at_the_shipped_files():
    from kg.render import html as mod

    assert Path(mod.VENDOR_JS_PATH).resolve() == VENDOR_JS.resolve()
    assert Path(mod.TEMPLATE_PATH).resolve() == TEMPLATE.resolve()


# ----------------------------------------------------- single file, offline


def test_R14_output_is_one_file_that_inlines_the_vendored_library(gen_graph):
    out, html = render_to(gen_graph)
    assert out.is_file()
    assert VENDOR_JS.read_text(encoding="utf-8").strip() in html, "force-graph.min.js bytes must be inlined, not referenced"
    assert not any(p.suffix in (".js", ".css") for p in gen_graph.iterdir()), "no sidecar assets — a single self-contained file"


def test_R14_no_external_requests_of_any_kind(gen_graph):
    _, html = render_to(gen_graph)
    vendor = VENDOR_JS.read_text(encoding="utf-8").strip()
    glue = html.replace(vendor, "<!-- vendored -->")
    assert not re.search(r"<script[^>]+\bsrc\s*=", html, re.IGNORECASE), "no <script src=>"
    assert not re.search(r"<link[^>]+\bhref\s*=\s*[\"']?(https?:)?//", html, re.IGNORECASE), "no external <link href>"
    assert not re.search(r"@import\s+url\(\s*[\"']?(https?:)?//", html, re.IGNORECASE)
    assert "sourceMappingURL=http" not in html
    for needle in ("fetch(", "XMLHttpRequest", "import(", "navigator.sendBeacon", "new WebSocket"):
        assert needle not in glue, f"renderer glue must not contain {needle!r}"
    urls = [m.group(0) for m in re.finditer(r"https?://[^\s\"'<>)]+", glue)]
    external = [u for u in urls if EXTERNAL_HOST_RE.match(u) and "example.go.th" not in u]
    assert external == [], f"external URLs in glue/template: {external}"
    assert "<meta charset" in html.lower()


def test_R14_html_is_valid_utf8_and_thai_titles_render_unescaped(gen_graph):
    out, html = render_to(gen_graph)
    out.read_bytes().decode("utf-8")  # raises on bad encoding
    assert THAI_TITLE in html


# ------------------------------------------------------------ payload


def test_R14_every_node_and_edge_on_disk_appears_in_the_embedded_payload(gen_graph):
    _, html = render_to(gen_graph)
    payload = graph_payload(html)
    assert {n["id"] for n in payload["nodes"]} == GENERAL_IDS
    assert canonical_edges(payload["edges"]) == GENERAL_EDGES
    assert payload["meta"]["schema"] == "general"


def test_R14_education_payload_has_all_nodes_edges_and_origin(edu_graph):
    _, html = render_to(edu_graph, EDUCATION)
    payload = graph_payload(html)
    assert {n["id"] for n in payload["nodes"]} == EDUCATION_IDS
    assert canonical_edges(payload["edges"]) == EDUCATION_EDGES
    assert all(n["origin"] == "course_material" for n in payload["nodes"]), "R9: renderer may display origin"


def test_R9_general_payload_nodes_carry_no_origin(gen_graph):
    _, html = render_to(gen_graph)
    for n in graph_payload(html)["nodes"]:
        assert n.get("origin") is None


def test_R14_payload_nodes_carry_title_definition_excerpt_and_source_locators(gen_graph):
    _, html = render_to(gen_graph)
    node = next(n for n in graph_payload(html)["nodes"] if n["id"] == "kb-0001")
    assert node["title"] == "Cloud First Policy"
    assert "[[" not in node["definition_plain"] and node["definition_plain"].startswith("Cloud First requires")
    locators = [s["locator"] for s in node["sources"]]
    assert "sovereignty-notes.docx#heading=Cloud First > Data residency" in locators
    assert node["provider"] == "openrouter" and node["model"] == "anthropic/claude-sonnet-5", "§13: inspector shows provider · model"


def test_R10_R14_related_to_edges_carry_relevance_and_other_types_do_not(gen_graph):
    _, html = render_to(gen_graph)
    edges = graph_payload(html)["edges"]
    rel = [e for e in edges if e["type"] == "related_to"]
    assert len(rel) == 6 and all(isinstance(e["relevance"], int) for e in rel)
    assert all(e.get("relevance") is None for e in edges if e["type"] != "related_to")


def test_R14_payload_json_is_safe_inside_a_script_element(gen_graph):
    """`</script>` inside a title/definition must not terminate the data block."""
    from wu45_fixtures import note_spec, write_graph

    d = gen_graph.parent / "hostile"
    d.mkdir()
    write_graph(d, [note_spec("h-0001", "Tag </script><script>alert(1)</script>", schema="general", definition="x </script> y")], corpus="h", schema="general")
    _, html = render_to(d)
    payload = graph_payload(html)
    assert payload["nodes"][0]["title"] == "Tag </script><script>alert(1)</script>"
    assert "<script>alert(1)</script>" not in html


# -------------------------------------------------------- styles / filters


def test_R14_edge_style_map_covers_every_edge_type_of_both_schemas():
    from kg.render.html import EDGE_STYLES

    for schema in SCHEMAS.values():
        missing = set(schema.edges) - set(EDGE_STYLES)
        assert not missing, f"{schema.name}: no style for {missing}"
    for etype, style in EDGE_STYLES.items():
        assert isinstance(style, dict) and "color" in style, etype


def test_R14_payload_styles_are_scoped_to_the_selected_schema(gen_graph, edu_graph):
    _, html = render_to(gen_graph, GENERAL)
    assert set(graph_payload(html)["styles"]) == set(GENERAL.edges)
    _, html_e = render_to(edu_graph, EDUCATION)
    assert set(graph_payload(html_e)["styles"]) == set(EDUCATION.edges)


def test_R14_filters_search_and_inspector_are_present(gen_graph):
    _, html = render_to(gen_graph)
    assert re.search(r'id="search"', html), "title/alias search box"
    slider = re.search(r'<input[^>]*id="min-relevance"[^>]*>', html)
    assert slider and 'type="range"' in slider.group(0), "min-relevance slider"
    assert re.search(r'id="edge-filters"', html), "per-edge-type toggles container"
    assert re.search(r'id="inspector"', html), "node inspector panel"
    glue = "\n".join(js_scripts(html))
    assert "relevance" in glue and "aliases" in glue, "JS references relevance weighting and alias search"


def test_R14_glue_script_references_the_payload_and_force_graph_api(gen_graph):
    _, html = render_to(gen_graph)
    glue = "\n".join(js_scripts(html))
    assert "graph-data" in glue
    assert "ForceGraph" in glue
    assert "linkWidth" in glue or "linkColor" in glue, "edge styling by type / relevance through force-graph link callbacks"


# ------------------------------------------------------------ JS parses


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed; JS parse smoke test skipped")
def test_R14_every_inline_script_parses_under_node(gen_graph, tmp_path):
    _, html = render_to(gen_graph)
    scripts = js_scripts(html)
    assert len(scripts) >= 2, "expected vendored library + glue"
    for i, body in enumerate(scripts):
        f = tmp_path / f"inline-{i}.js"
        f.write_text(body, encoding="utf-8")
        proc = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
        assert proc.returncode == 0, f"inline script #{i} failed to parse:\n{proc.stderr[:800]}"


def test_R14_payload_block_is_application_json(gen_graph):
    _, html = render_to(gen_graph)
    attrs = next(a for a, _ in inline_scripts(html) if 'id="graph-data"' in a)
    assert 'type="application/json"' in attrs


# --------------------------------------------------------- idempotency


def test_R11_render_is_byte_identical_on_re_run(gen_graph):
    out, first = render_to(gen_graph, now=NOW)
    first_bytes = out.read_bytes()
    out2, _ = render_to(gen_graph, now=NOW)
    assert out2.read_bytes() == first_bytes
    idx = (gen_graph / "_index.json").read_bytes()
    render_to(gen_graph, now=NOW)
    assert (gen_graph / "_index.json").read_bytes() == idx


def test_R11_render_does_not_modify_the_notes(gen_graph):
    from wu45_fixtures import snapshot_tree

    before = snapshot_tree(gen_graph / "nodes")
    render_to(gen_graph)
    assert snapshot_tree(gen_graph / "nodes") == before


def test_S8_6_render_regenerates_index_json_matching_the_payload(gen_graph):
    _, html = render_to(gen_graph)
    idx = json.loads((gen_graph / "_index.json").read_text(encoding="utf-8"))
    payload = graph_payload(html)
    assert {n["id"] for n in idx["nodes"]} == {n["id"] for n in payload["nodes"]}
    assert canonical_edges(idx["edges"]) == canonical_edges(payload["edges"])


def test_R14_generated_timestamp_is_injected_not_read_from_the_clock(gen_graph):
    _, a = render_to(gen_graph, now=NOW)
    _, b = render_to(gen_graph, now=NOW_2)
    assert graph_payload(a)["meta"].get("generated") != graph_payload(b)["meta"].get("generated")


# ---------------------------------------------------------- unhappy paths


def test_R14_empty_graph_renders_a_valid_page_with_an_empty_payload(tmp_path):
    d = tmp_path / "empty" / "graph"
    (d / "nodes").mkdir(parents=True)
    _, html = render_to(d)
    payload = graph_payload(html)
    assert payload["nodes"] == [] and payload["edges"] == []


def test_R14_corrupt_note_is_skipped_and_listed_in_the_payload_problems(gen_graph):
    (gen_graph / "nodes" / "kb-0003-in-country-region.md").write_text("garbage", encoding="utf-8")
    _, html = render_to(gen_graph)
    payload = graph_payload(html)
    assert "kb-0003" not in {n["id"] for n in payload["nodes"]}
    assert payload["meta"]["problems"], "renderer surfaces skipped files instead of silently dropping them"


def test_S23_out_path_outside_the_graph_folder_tree_is_still_written_only_where_asked(gen_graph, tmp_path):
    from kg.render.html import render

    out = tmp_path / "elsewhere" / "graph.html"
    out.parent.mkdir()
    render(gen_graph, out, schema=GENERAL, now=NOW)
    assert out.is_file()
    assert not (gen_graph / "graph.html").exists()


def test_S23_refuses_to_write_into_a_forbidden_location(gen_graph, tmp_path):
    from kg.paths import SandboxViolation
    from kg.render.html import render

    forbidden = tmp_path / "TK-PKA" / "graph.html"
    forbidden.parent.mkdir(parents=True)
    with pytest.raises(SandboxViolation):
        render(gen_graph, forbidden, schema=GENERAL, now=NOW)
    assert not forbidden.exists()
