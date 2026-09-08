"""WU1 — locator resolvability (R3 acceptance: "a script can resolve ≥95% of locators
back to a real page/slide/sheet"; on the tiny fixture corpus we require 100%).

Interface under test (DESIGN §21 names `tests/test_locators.py`; identifier chosen):
  kg.convert.resolve_locator(locator: str, base_dir: Path) -> bool
    - file locators (`<file>#...`) are resolved against `base_dir/<file>`;
    - URL locators (`http(s)://...#heading=...`) are resolved against the archived
      raw HTML `base_dir/<host>-<slug>.html` written by the web converter;
    - returns False (never raises) for a missing file, an out-of-range page/slide,
      an unknown sheet/heading, or a malformed locator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kg.config import Config, load_config
from wu1_fixtures import (
    DOCX_WITH_HEADINGS,
    HTML_PAGE,
    WEB_URL,
    FakeCallStage,
    canned_describe,
    make_docx,
    make_md,
    make_pdf,
    make_png,
    make_pptx,
    make_text,
    make_xlsx,
)


@pytest.fixture
def corpus(project_root: Path):
    """Convert a tiny mixed corpus once; return (cfg, outcomes)."""
    from kg.convert import convert_inbox

    cfg: Config = load_config(project_root / "kg.yaml")
    inbox = cfg.sandbox.root("inbox")
    make_text(inbox / "notes.txt", n_lines=100)
    make_md(inbox / "notes.md")
    make_pdf(inbox / "deck.pdf", ["Page one", "", "Page three"])
    make_pptx(inbox / "deck.pptx", [("A", "a", None), ("B", "b", "notes"), ("C", "c", None)])
    make_docx(inbox / "spec.docx", DOCX_WITH_HEADINGS)
    make_xlsx(inbox / "terms.xlsx")
    make_png(inbox / "diagram.png", size=(200, 100))
    (inbox / "urls.txt").write_text(WEB_URL + "\n", encoding="utf-8")
    outcomes = convert_inbox(
        cfg, call_stage=FakeCallStage([canned_describe()]), fetch=lambda url: HTML_PAGE if url == WEB_URL else None
    )
    assert all(o.status == "converted" for o in outcomes), [(o.item.name, o.status, o.reason) for o in outcomes]
    return cfg, outcomes


def test_R3_every_locator_in_the_corpus_resolves(corpus):
    from kg.convert import resolve_locator

    cfg, outcomes = corpus
    inbox, converted = cfg.sandbox.root("inbox"), cfg.sandbox.root("converted")
    total, resolved, failures = 0, 0, []
    for o in outcomes:
        base = converted if o.item.kind == "web" else inbox
        for u in o.units:
            total += 1
            if resolve_locator(u.locator, base):
                resolved += 1
            else:
                failures.append(u.locator)
    assert total >= 15, "corpus too small to be meaningful"
    assert resolved == total, f"unresolved locators: {failures}"


@pytest.mark.parametrize(
    "locator",
    [
        "notes.md#heading=Intro > Terms",
        "notes.md#lines=1-2",
        "deck.pdf#page=3",
        "deck.pptx#slide=2",
        "spec.docx#heading=1 Scope > 1.2 Terms",
        "terms.xlsx#Sheet1!A1:C4",
        "terms.xlsx#Grades!A1:B3",
        "diagram.png",
    ],
)
def test_R3_hand_written_locators_of_each_format_resolve(corpus, locator: str):
    from kg.convert import resolve_locator

    cfg, _ = corpus
    assert resolve_locator(locator, cfg.sandbox.root("inbox")) is True


def test_R3_web_locator_resolves_against_archived_html(corpus):
    from kg.convert import resolve_locator

    cfg, _ = corpus
    assert resolve_locator(f"{WEB_URL}#heading=Probability Basics > Events", cfg.sandbox.root("converted")) is True


@pytest.mark.parametrize(
    "locator",
    [
        "missing.pdf#page=1",  # no such file
        "deck.pdf#page=0",  # pages are 1-based
        "deck.pdf#page=99",  # out of range
        "deck.pptx#slide=4",  # only 3 slides
        "terms.xlsx#Nope!A1:B2",  # unknown sheet
        "terms.xlsx#Sheet1!Z100:ZZ200",  # range outside used area
        "notes.md#heading=Nonexistent Heading",
        "spec.docx#heading=No Such Heading",
        "notes.txt#lines=900-950",  # beyond EOF
        "garbage-without-any-structure",
        "",
    ],
)
def test_R3_unresolvable_locators_return_false_not_raise(corpus, locator: str):
    from kg.convert import resolve_locator

    cfg, _ = corpus
    assert resolve_locator(locator, cfg.sandbox.root("inbox")) is False


def test_R3_web_locator_without_archive_returns_false(corpus, tmp_path: Path):
    from kg.convert import resolve_locator

    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_locator(f"{WEB_URL}#heading=Probability Basics", empty) is False
