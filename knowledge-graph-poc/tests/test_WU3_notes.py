"""WU3 — `kg.notes`: node file render/parse, wikilink weaving, idempotency.

Spec: PRD R7 (atomic note: title, definition, `## Relations` grouped by edge type, `## Source`),
R3/R4 (`## Source` = verbatim quote + structured locator), R9 (`origin` education only),
R10 (`relevance` only on general `related_to`), R11 (wikilinks; strip-on-read/weave-on-write
idempotent); DESIGN §8.2–8.4 (worked examples are the byte-exact target), §12; D13.

Interface (chosen; see wu3_fixtures docstring): Node, NodeSource, NodeEdge, LinkTarget,
render(node, targets), parse(text), weave(text, targets, *, exclude=None), strip_wikilinks(text).
"""

from __future__ import annotations

import pytest
import yaml

from wu3_fixtures import (
    EDU_EXPECTED,
    GEN_EXPECTED,
    THAI_TITLE,
    edu_example_node,
    edu_example_targets,
    gen_example_node,
    gen_example_targets,
    simple_node,
)


# ------------------------------------------------------------ render: exact


def test_R7_R9_render_education_worked_example_byte_exact():
    from kg.notes import render

    assert render(edu_example_node(), edu_example_targets()) == EDU_EXPECTED


def test_R7_R10_render_general_worked_example_byte_exact():
    from kg.notes import render

    assert render(gen_example_node(), gen_example_targets()) == GEN_EXPECTED


def test_R9_general_note_has_no_origin_key_and_education_note_has_it():
    from kg.notes import render

    gen_fm = yaml.safe_load(render(gen_example_node(), gen_example_targets()).split("---")[1])
    edu_fm = yaml.safe_load(render(edu_example_node(), edu_example_targets()).split("---")[1])
    assert "origin" not in gen_fm
    assert edu_fm["origin"] == "course_material"


def test_R7_frontmatter_is_valid_yaml_with_the_canonical_keys():
    from kg.notes import render

    fm = yaml.safe_load(render(edu_example_node(), edu_example_targets()).split("---")[1])
    assert list(fm) == ["id", "title", "aliases", "schema", "origin", "created", "updated", "run", "prompt_version", "provider", "model", "review", "sources", "edges"]
    assert fm["sources"] == [
        {"locator": "week01-probability.pptx#slide=4", "file": "week01-probability.pptx"},
        {"locator": "week01-handout.pdf#page=2", "file": "week01-handout.pdf"},
    ]
    assert fm["edges"] == {"prerequisite_of": ["stat101-0005", "stat101-0009"], "part_of": ["stat101-0001"]}


def test_R10_general_related_to_frontmatter_carries_relevance_and_other_types_do_not():
    from kg.notes import render

    fm = yaml.safe_load(render(gen_example_node(), gen_example_targets()).split("---")[1])
    assert fm["edges"]["related_to"] == [{"id": "kb-0021", "relevance": 82}, {"id": "kb-0030", "relevance": 41}]
    assert fm["edges"]["part_of"] == ["kb-0002"]


def test_R7_relations_section_groups_by_edge_type_in_schema_order():
    from kg.notes import render

    text = render(gen_example_node(), gen_example_targets())
    rel = text.split("## Relations")[1].split("## Source")[0]
    assert rel.index("### related_to") < rel.index("### part_of")
    assert "— relevance 82" in rel and "— relevance 41" in rel
    assert rel.count("— relevance") == 2, "relevance shown on related_to only"


def test_R3_R4_source_section_has_locator_heading_and_verbatim_blockquote_per_quote():
    from kg.notes import render

    text = render(edu_example_node(), edu_example_targets())
    src = text.split("## Source")[1]
    assert "### week01-probability.pptx#slide=4\n> The sample space S is the set of all possible outcomes of an experiment." in src
    assert "### week01-handout.pdf#page=2\n> We write S (sometimes Ω)" in src


def test_R7_render_with_no_edges_and_no_aliases_still_emits_the_sections():
    from kg.notes import parse, render

    n = simple_node("kc-0001", "Lonely Concept")
    text = render(n, {})
    fm = yaml.safe_load(text.split("---")[1])
    assert fm["aliases"] == [] and fm["edges"] == {} and fm["review"] == []
    assert "## Definition" in text and "## Relations" in text and "## Source" in text
    assert parse(text) == n


def test_R7_render_refuses_a_node_whose_edge_target_is_not_in_targets():
    from kg.notes import NodeEdge, render

    n = simple_node("kc-0001", "A", edges=[NodeEdge(type="part_of", target="kc-0099")])
    with pytest.raises((KeyError, ValueError)):
        render(n, {})


# ------------------------------------------------------------- parse


def test_R7_parse_education_example_gives_the_model_back():
    from kg.notes import parse

    assert parse(EDU_EXPECTED) == edu_example_node()


def test_R7_parse_general_example_gives_the_model_back():
    from kg.notes import parse

    node = parse(GEN_EXPECTED)
    assert node == gen_example_node()
    assert node.origin is None
    assert [(e.type, e.target, e.relevance) for e in node.edges] == [("related_to", "kb-0021", 82), ("related_to", "kb-0030", 41), ("part_of", "kb-0002", None)]


def test_R7_parse_dates_and_run_are_strings_not_yaml_dates():
    from kg.notes import parse

    node = parse(EDU_EXPECTED)
    assert node.created == "2026-08-25" and isinstance(node.created, str)
    assert node.updated == "2026-08-25" and isinstance(node.updated, str)
    assert node.run == "2026-08-25T10-15-02Z"


def test_R11_parse_strips_wikilinks_from_the_definition():
    from kg.notes import parse

    node = parse(EDU_EXPECTED)
    assert "[[" not in node.definition and "]]" not in node.definition
    assert node.definition == edu_example_node().definition


def test_R4_parse_reads_quotes_from_the_source_section_paired_with_frontmatter_locators():
    from kg.notes import parse

    node = parse(GEN_EXPECTED)
    assert [s.locator for s in node.sources] == [
        "https://example.go.th/policy/cloud-first#heading=2. Scope > 2.1 Data classes",
        "sovereignty-notes.docx#heading=Cloud First > Data residency",
    ]
    assert node.sources[0].quote.startswith("Agencies shall adopt cloud services")
    assert node.sources[1].file == "sovereignty-notes.docx"


@pytest.mark.parametrize(
    "bad",
    [
        "no frontmatter at all\n",
        "---\nid: x\n---\n# X\n",  # missing required keys
        "---\nid: kc-0001\ntitle: X\naliases: []\nschema: nowhere\ncreated: 2026-08-25\nupdated: 2026-08-25\nrun: r\nprompt_version: atomize@1\nprovider: p\nmodel: m\nreview: []\nsources: []\nedges: {}\n---\n# X\n\n## Definition\nx\n",
    ],
)
def test_R7_parse_rejects_malformed_notes_with_an_error(bad: str):
    from kg.notes import parse

    with pytest.raises(Exception):
        parse(bad)


# ---------------------------------------------------------- round trip


@pytest.mark.parametrize("text,targets", [(EDU_EXPECTED, "edu"), (GEN_EXPECTED, "gen")])
def test_R11_render_parse_render_twice_is_byte_identical(text: str, targets: str):
    from kg.notes import parse, render

    tgt = edu_example_targets() if targets == "edu" else gen_example_targets()
    once = render(parse(text), tgt)
    twice = render(parse(once), tgt)
    assert once == text
    assert twice == once
    assert "[[[[" not in twice


def test_R11_title_with_yaml_special_characters_round_trips():
    from kg.notes import parse, render

    n = simple_node("kc-0001", "Bayes: Theorem #1 [draft]", aliases=["P(A|B): posterior"], definition="Ratio a: b — see #1.")
    text = render(n, {})
    assert parse(text) == n
    assert yaml.safe_load(text.split("---")[1])["title"] == "Bayes: Theorem #1 [draft]"


def test_R11_thai_title_and_definition_survive_round_trip():
    from kg.notes import parse, render

    n = simple_node("kc-0001", THAI_TITLE, definition=f"{THAI_TITLE} คือสาขาหนึ่งของการเรียนรู้ของเครื่อง")
    text = render(n, {})
    assert THAI_TITLE in text
    back = parse(text)
    assert back.title == THAI_TITLE
    assert back.definition == n.definition
    assert render(back, {}) == text


def test_R7_review_flags_round_trip():
    from kg.notes import parse, render

    n = simple_node("kc-0001", "Flagged", review=["quote_not_found"])
    text = render(n, {})
    assert yaml.safe_load(text.split("---")[1])["review"] == ["quote_not_found"]
    assert parse(text).review == ["quote_not_found"]


# ---------------------------------------------------------------- weave


def test_R11_weave_links_other_titles_case_insensitively_keeping_prose_casing():
    from kg.notes import weave

    out = weave(edu_example_node().definition, edu_example_targets(), exclude="stat101-0004")
    assert "Every [[stat101-0005-event|event]] is a subset" in out
    assert "a [[stat101-0009-probability-measure|probability measure]] assigns" in out


def test_R11_weave_never_links_the_node_itself():
    from kg.notes import weave

    out = weave(edu_example_node().definition, edu_example_targets(), exclude="stat101-0004")
    assert "[[stat101-0004-sample-space" not in out
    assert "sample space is the set" in out


def test_R11_weave_matches_aliases_too():
    from kg.notes import LinkTarget, weave

    targets = {"kc-0002": LinkTarget(stem="kc-0002-sample-space", title="Sample Space", aliases=("Outcome space",))}
    out = weave("Enumerate the outcome space first.", targets)
    assert out == "Enumerate the [[kc-0002-sample-space|outcome space]] first."


def test_R11_weave_prefers_the_longest_matching_title():
    from kg.notes import LinkTarget, weave

    targets = {
        "kc-0001": LinkTarget(stem="kc-0001-probability", title="Probability"),
        "kc-0002": LinkTarget(stem="kc-0002-probability-measure", title="Probability Measure"),
    }
    out = weave("A probability measure is a function.", targets)
    assert out == "A [[kc-0002-probability-measure|probability measure]] is a function."


def test_R11_weave_matches_whole_words_only():
    from kg.notes import LinkTarget, weave

    targets = {"kc-0001": LinkTarget(stem="kc-0001-event", title="Event")}
    assert weave("Eventually the event occurs.", targets) == "Eventually the [[kc-0001-event|event]] occurs."


def test_R11_weave_is_idempotent_and_never_nests():
    from kg.notes import weave

    once = weave(edu_example_node().definition, edu_example_targets(), exclude="stat101-0004")
    twice = weave(once, edu_example_targets(), exclude="stat101-0004")
    assert twice == once
    assert "[[[[" not in twice


def test_R11_weave_thai_title_inside_thai_prose():
    from kg.notes import LinkTarget, weave

    targets = {"kc-0009": LinkTarget(stem="kc-0009", title=THAI_TITLE)}
    out = weave(f"บทนำ {THAI_TITLE} เป็นหัวข้อสำคัญ", targets)
    assert out == f"บทนำ [[kc-0009|{THAI_TITLE}]] เป็นหัวข้อสำคัญ"


def test_R11_weave_with_no_targets_returns_text_unchanged():
    from kg.notes import weave

    text = "Nothing to link here."
    assert weave(text, {}) == text


# ---------------------------------------------------------------- strip


@pytest.mark.parametrize(
    "linked,plain",
    [
        ("Every [[stat101-0005-event|event]] is a subset.", "Every event is a subset."),
        ("See [[kc-0001-foo]] for more.", "See kc-0001-foo for more."),
        # DESIGN §12: strip is idempotent and never drops prose — the text before a
        # hand-nested link survives (consistent with the parse test below).
        ("Nested [[[[a|b]]]] junk", "Nested b junk"),
        ("no links", "no links"),
    ],
)
def test_R11_strip_wikilinks(linked: str, plain: str):
    from kg.notes import strip_wikilinks

    assert strip_wikilinks(linked) == plain


def test_R11_parse_of_a_hand_nested_link_recovers_plain_prose():
    from kg.notes import parse

    text = EDU_EXPECTED.replace("[[stat101-0005-event|event]]", "[[[[stat101-0005-event|event]]]]")
    assert parse(text).definition == edu_example_node().definition
