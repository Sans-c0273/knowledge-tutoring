"""WU1 — `kg.convert` dispatch: S0 discover + S1 convert over a mixed inbox (R1, R2, R5-boundary, sandbox).

Spec: PRD R1 ("Unsupported files are reported, never silently skipped"; mixed
inbox ingests with a per-input outcome), R2 (converted/ written), R5 (nothing
deleted — the *move* to processed/ is S8 in pipeline.py per DESIGN §6, so here
we only pin that conversion moves and deletes nothing), DESIGN §4.2 row "else",
§6 S0/S1, §3.3 sandbox.

Interfaces under test (chosen where DESIGN §20 names only the module):
  kg.convert.SUPPORTED_EXTENSIONS: frozenset[str]   lower-case, with dot
  kg.convert.InputItem(name, kind, path, url)        kind ∈ text|markdown|pdf|pptx|docx|xlsx|image|web|unsupported
  kg.convert.ConversionOutcome(item, status, reason, units)
                                                      status ∈ converted|unsupported|failed|deferred
  kg.convert.discover(inbox_dir: Path) -> list[InputItem]
  kg.convert.convert_inbox(cfg: Config, *, call_stage, fetch=None) -> list[ConversionOutcome]
  kg.convert.convert_item(item, cfg, *, call_stage, fetch=None) -> ConversionOutcome
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg.config import Config, load_config
from kg.paths import SandboxViolation
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
    snapshot_tree,
)

SUPPORTED_FILES = ("notes.txt", "notes.md", "deck.pdf", "deck.pptx", "spec.docx", "terms.xlsx", "diagram.png")
UNSUPPORTED_FILES = ("archive.zip", "legacy.xls")


def populate_inbox(inbox: Path, *, with_unsupported: bool = True, with_urls: bool = True) -> None:
    make_text(inbox / "notes.txt", n_lines=30)
    make_md(inbox / "notes.md")
    make_pdf(inbox / "deck.pdf", ["Page one", "Page two"])
    make_pptx(inbox / "deck.pptx", [("Slide A", "Body A", "Notes A"), ("Slide B", "Body B", None)])
    make_docx(inbox / "spec.docx", DOCX_WITH_HEADINGS)
    make_xlsx(inbox / "terms.xlsx")
    make_png(inbox / "diagram.png", size=(2000, 1000))
    if with_unsupported:
        (inbox / "archive.zip").write_bytes(b"PK\x05\x06" + b"\0" * 18)
        (inbox / "legacy.xls").write_bytes(b"\xd0\xcf\x11\xe0 old binary excel")
    if with_urls:
        (inbox / "urls.txt").write_text(f"# course reading list\n\n{WEB_URL}\n\n", encoding="utf-8")


def _output_suffix(p: Path) -> str:
    """Artefact kind of a converted/ file: `.units.json`, `.html` or `.md` (stems may contain dots)."""
    return next((s for s in (".units.json", ".html", ".md") if p.name.endswith(s)), p.suffix)


def fetch_local(url: str) -> str | None:
    """Offline stand-in for trafilatura.fetch_url: serves the saved HTML for the one known URL."""
    return HTML_PAGE if url == WEB_URL else None


@pytest.fixture
def cfg(project_root: Path) -> Config:
    return load_config(project_root / "kg.yaml")


@pytest.fixture
def inbox(cfg: Config) -> Path:
    d = cfg.sandbox.root("inbox")
    populate_inbox(d)
    return d


# ================================================================ S0 discover


def test_R1_discover_lists_every_file_and_url_with_a_kind(inbox: Path):
    from kg.convert import discover

    items = discover(inbox)
    by_name = {i.name: i for i in items}
    assert set(by_name) == set(SUPPORTED_FILES) | set(UNSUPPORTED_FILES) | {WEB_URL}
    assert "urls.txt" not in by_name, "urls.txt is the URL list, not an input in its own right"
    expected_kind = {
        "notes.txt": "text",
        "notes.md": "markdown",
        "deck.pdf": "pdf",
        "deck.pptx": "pptx",
        "spec.docx": "docx",
        "terms.xlsx": "xlsx",
        "diagram.png": "image",
        WEB_URL: "web",
        "archive.zip": "unsupported",
        "legacy.xls": "unsupported",
    }
    for name, kind in expected_kind.items():
        assert by_name[name].kind == kind, f"{name}: kind {by_name[name].kind!r} != {kind!r}"
    web = by_name[WEB_URL]
    assert web.url == WEB_URL and web.path is None
    assert by_name["deck.pdf"].path == inbox / "deck.pdf" and by_name["deck.pdf"].url is None


def test_R1_discover_unsupported_extensions_are_items_not_silently_skipped(inbox: Path):
    from kg.convert import SUPPORTED_EXTENSIONS, discover

    items = discover(inbox)
    unsupported = [i for i in items if i.kind == "unsupported"]
    assert {i.name for i in unsupported} == set(UNSUPPORTED_FILES)
    assert ".xls" not in SUPPORTED_EXTENSIONS, "DESIGN §4.2: .xls unsupported"
    assert {".txt", ".md", ".pdf", ".pptx", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".gif", ".webp"} <= set(
        SUPPORTED_EXTENSIONS
    )
    assert all(e == e.lower() and e.startswith(".") for e in SUPPORTED_EXTENSIONS)


def test_R1_discover_matches_extensions_case_insensitively(cfg: Config):
    from kg.convert import discover

    inbox = cfg.sandbox.root("inbox")
    make_pdf(inbox / "UPPER.PDF", ["x"])
    items = discover(inbox)
    assert [i.kind for i in items] == ["pdf"]


def test_R1_discover_urls_txt_ignores_blank_lines_and_comments(cfg: Config):
    from kg.convert import discover

    inbox = cfg.sandbox.root("inbox")
    (inbox / "urls.txt").write_text(
        "# heading comment\n\n   \nhttps://a.example.org/one\n# https://commented.example.org\nhttps://b.example.org/two  \n",
        encoding="utf-8",
    )
    items = discover(inbox)
    assert [i.url for i in items] == ["https://a.example.org/one", "https://b.example.org/two"]
    assert all(i.kind == "web" for i in items)


def test_R1_discover_empty_inbox_yields_no_items(cfg: Config):
    from kg.convert import discover

    assert discover(cfg.sandbox.root("inbox")) == []


def test_R1_discover_is_deterministic_and_sorted(inbox: Path):
    from kg.convert import discover

    a = [i.name for i in discover(inbox)]
    b = [i.name for i in discover(inbox)]
    assert a == b
    files = [n for n in a if n != WEB_URL]
    assert files == sorted(files), "file items in sorted name order so runs are reproducible (R16)"


# ================================================================= S1 convert


def test_R1_convert_inbox_yields_exactly_one_outcome_per_input(cfg: Config, inbox: Path):
    from kg.convert import convert_inbox

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage([canned_describe()]), fetch=fetch_local)
    names = [o.item.name for o in outcomes]
    assert sorted(names) == sorted(set(SUPPORTED_FILES) | set(UNSUPPORTED_FILES) | {WEB_URL})
    assert len(names) == len(set(names)), "one outcome per input, no duplicates"
    status = {o.item.name: o.status for o in outcomes}
    for name in SUPPORTED_FILES + (WEB_URL,):
        assert status[name] == "converted", f"{name}: {status[name]} ({next(o.reason for o in outcomes if o.item.name == name)})"
    for name in UNSUPPORTED_FILES:
        assert status[name] == "unsupported"


def test_R1_unsupported_outcome_carries_a_reason_naming_the_extension(cfg: Config, inbox: Path):
    from kg.convert import convert_inbox

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage([canned_describe()]), fetch=fetch_local)
    xls = next(o for o in outcomes if o.item.name == "legacy.xls")
    assert xls.status == "unsupported"
    assert xls.reason and ".xls" in xls.reason
    assert xls.units == []
    assert (inbox / "legacy.xls").exists(), "left in inbox (DESIGN §4.2)"
    assert not (cfg.sandbox.root("converted") / "legacy.md").exists()


def test_R2_every_converted_outcome_has_md_and_units_json_on_disk(cfg: Config, inbox: Path):
    # Paths come from the outcome itself (`ConversionOutcome.outputs`), not from a
    # guessed `<stem>`: populate_inbox() has colliding stems (deck.pdf/deck.pptx,
    # notes.txt/notes.md), so the on-disk stem may carry a collision suffix.
    from kg.convert import convert_inbox

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage([canned_describe()]), fetch=fetch_local)
    converted_dir = cfg.sandbox.root("converted")
    all_outputs: list[Path] = []
    for o in outcomes:
        if o.status != "converted":
            assert o.outputs == (), f"{o.item.name}: {o.status} outcome must expose no outputs"
            continue
        assert o.units, f"{o.item.name}: converted but no units"
        by_suffix = {_output_suffix(p): p for p in o.outputs}
        expected_suffixes = {".html", ".md", ".units.json"} if o.item.kind == "web" else {".md", ".units.json"}
        assert set(by_suffix) == expected_suffixes, f"{o.item.name}: outputs {sorted(p.name for p in o.outputs)}"
        for p in o.outputs:
            assert p.is_file(), f"{o.item.name}: missing {p.name}"
            assert p.parent == converted_dir, f"{o.item.name}: {p} not directly under converted/"
        if o.item.kind == "web":
            assert by_suffix[".md"].name.startswith("learn.example.edu-"), "web page → converted/<host>-<slug>.*"
        rows = json.loads(by_suffix[".units.json"].read_text(encoding="utf-8"))
        assert [r["locator"] for r in rows] == [u.locator for u in o.units]
        all_outputs.extend(o.outputs)
    assert len(all_outputs) == len(set(all_outputs)), "two inputs must never share an output file"


def test_R2_colliding_stems_get_a_deterministic_extension_suffix(cfg: Config, inbox: Path):
    # DESIGN §4.1 names outputs `converted/<stem>.*`; when two inbox files share a stem
    # within one run the implementation appends `-<ext>` so neither overwrites the other.
    from kg.convert import convert_inbox

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage([canned_describe()]), fetch=fetch_local)
    converted_dir = cfg.sandbox.root("converted")
    md_for = {o.item.name: next(p for p in o.outputs if p.name.endswith(".md")) for o in outcomes if o.status == "converted"}
    assert md_for["deck.pdf"] == converted_dir / "deck-pdf.md"
    assert md_for["deck.pptx"] == converted_dir / "deck-pptx.md"
    assert md_for["notes.txt"] == converted_dir / "notes-txt.md"
    assert md_for["notes.md"] == converted_dir / "notes-md.md"
    for name in ("spec.docx", "terms.xlsx", "diagram.png"):
        assert md_for[name] == converted_dir / f"{Path(name).stem}.md", f"{name}: unique stem must NOT be suffixed"
    assert not (converted_dir / "deck.md").exists() and not (converted_dir / "notes.md").exists()
    for stem in ("deck-pdf", "deck-pptx", "notes-txt", "notes-md"):
        assert (converted_dir / f"{stem}.units.json").is_file(), f"{stem}.units.json follows the suffixed stem"


def test_R2_collision_suffix_is_stable_across_runs(cfg: Config, inbox: Path):
    from kg.convert import convert_inbox

    def run() -> dict[str, tuple[str, ...]]:
        outs = convert_inbox(cfg, call_stage=FakeCallStage([canned_describe()]), fetch=fetch_local)
        return {o.item.name: tuple(p.name for p in o.outputs) for o in outs}

    assert run() == run(), "output naming must be deterministic (R16)"


def test_R1_one_failing_input_does_not_abort_the_others(cfg: Config, inbox: Path):
    from kg.convert import convert_inbox

    (inbox / "broken.pdf").write_bytes(b"%PDF-1.4\nnot a pdf\n%%EOF\n")
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage([canned_describe()]), fetch=fetch_local)
    broken = next(o for o in outcomes if o.item.name == "broken.pdf")
    assert broken.status == "failed"
    assert broken.reason and broken.reason.strip()
    assert broken.units == []
    assert (inbox / "broken.pdf").exists(), "failed item stays in inbox (DESIGN §6 S1)"
    assert not (cfg.sandbox.root("converted") / "broken.md").exists()
    converted = [o.item.name for o in outcomes if o.status == "converted"]
    assert set(SUPPORTED_FILES) | {WEB_URL} <= set(converted)


def test_R1_web_fetch_failure_is_a_failed_outcome_with_reason(cfg: Config):
    from kg.convert import convert_inbox

    inbox = cfg.sandbox.root("inbox")
    (inbox / "urls.txt").write_text("https://down.example.org/page\n", encoding="utf-8")
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage(), fetch=lambda url: None)
    assert len(outcomes) == 1
    assert outcomes[0].status == "failed"
    assert outcomes[0].reason and outcomes[0].reason.strip()


def test_R6_image_in_inbox_goes_through_call_stage_with_config_max_edge(cfg: Config):
    import base64
    import io

    from PIL import Image

    from kg.convert import convert_inbox

    make_png(cfg.sandbox.root("inbox") / "diagram.png", size=(2400, 1200))
    fake = FakeCallStage([canned_describe()])
    outcomes = convert_inbox(cfg, call_stage=fake)
    assert [o.status for o in outcomes] == ["converted"]
    assert outcomes[0].units[0].locator == "diagram.png"
    assert len(fake.calls) == 1 and fake.calls[0]["stage"] == "describe"
    sent = Image.open(io.BytesIO(base64.b64decode(fake.calls[0]["images"][0].data_b64)))
    assert max(sent.size) <= cfg.image.max_long_edge_px  # 1568 from kg.yaml


def test_R6_image_model_failure_is_reported_and_file_stays_in_inbox(cfg: Config):
    # DESIGN §6: model-call failures mark the file failed/deferred; nothing from it is written.
    from kg.convert import convert_inbox

    src = make_png(cfg.sandbox.root("inbox") / "diagram.png", size=(300, 200))
    fake = FakeCallStage(raise_exc=RuntimeError("provider unavailable"))
    outcomes = convert_inbox(cfg, call_stage=fake)
    assert len(outcomes) == 1
    assert outcomes[0].status in ("failed", "deferred")
    assert outcomes[0].reason and "provider unavailable" in outcomes[0].reason
    assert src.exists()
    assert not (cfg.sandbox.root("converted") / "diagram.md").exists()


def test_R1_convert_inbox_without_urls_txt_converts_files_only(cfg: Config):
    from kg.convert import convert_inbox

    inbox = cfg.sandbox.root("inbox")
    populate_inbox(inbox, with_unsupported=False, with_urls=False)
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage([canned_describe()]))
    assert sorted(o.item.name for o in outcomes) == sorted(SUPPORTED_FILES)
    assert all(o.status == "converted" for o in outcomes)


def test_R1_convert_inbox_empty_inbox_returns_empty_list(cfg: Config):
    from kg.convert import convert_inbox

    fake = FakeCallStage()
    assert convert_inbox(cfg, call_stage=fake) == []
    assert fake.calls == []


# ================================================= R5 boundary: move nothing


def test_R5_conversion_moves_and_deletes_nothing_in_the_inbox(cfg: Config, inbox: Path):
    # The move to processed/ is S8 (pipeline.py) after a successful model run — DESIGN §6.
    # Conversion itself must leave the inbox byte-identical.
    from kg.convert import convert_inbox

    before = snapshot_tree(inbox)
    convert_inbox(cfg, call_stage=FakeCallStage([canned_describe()]), fetch=fetch_local)
    after = snapshot_tree(inbox)
    assert after == before
    assert snapshot_tree(cfg.sandbox.root("processed")) == {}


# ===================================================================== sandbox


def test_sandbox_conversion_writes_only_under_converted(cfg: Config, inbox: Path):
    from kg.convert import convert_inbox

    root = cfg.project_root
    before = snapshot_tree(root)
    convert_inbox(cfg, call_stage=FakeCallStage([canned_describe()]), fetch=fetch_local)
    after = snapshot_tree(root)
    new_or_changed = {p for p in after if p not in before or after[p] != before[p]}
    assert new_or_changed, "conversion must have written something"
    converted_rel = cfg.sandbox.root("converted").relative_to(root).as_posix() + "/"
    outside = sorted(p for p in new_or_changed if not p.startswith(converted_rel))
    assert outside == [], f"files written outside converted/: {outside}"


def test_sandbox_item_whose_name_escapes_converted_is_refused(cfg: Config, tmp_path: Path):
    from kg.convert import InputItem, convert_item

    real = make_md(tmp_path / "escape-src.md")
    item = InputItem(name="../escape.md", kind="markdown", path=real, url=None)
    root = cfg.project_root
    before = snapshot_tree(root)
    try:
        outcome = convert_item(item, cfg, call_stage=FakeCallStage())
    except SandboxViolation:
        pass
    else:
        assert outcome.status == "failed", "an escaping path must be refused, not converted"
        assert outcome.reason
    after = snapshot_tree(root)
    assert set(after) == set(before), f"unexpected files written: {set(after) ^ set(before)}"
    assert not (root / "data" / "escape.md").exists()


def test_sandbox_item_pointing_at_a_denylisted_path_string_is_refused(cfg: Config, tmp_path: Path):
    # String-only: nothing here touches the real vault (paths_sandbox tests use the same trick).
    from kg.convert import InputItem, convert_item

    real = make_md(tmp_path / "vault-src.md")
    item = InputItem(name="01-knowledge-base/note.md", kind="markdown", path=real, url=None)
    try:
        outcome = convert_item(item, cfg, call_stage=FakeCallStage())
    except SandboxViolation:
        return
    assert outcome.status == "failed" and outcome.reason
