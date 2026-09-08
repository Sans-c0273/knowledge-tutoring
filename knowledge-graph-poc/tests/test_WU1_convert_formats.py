"""WU1 — per-format converters (R2, R3, R6; DESIGN §4.1–§4.3, D3, D8).

Every converter writes `converted/<stem>.md` (with `<!-- unit: <locator> -->`
boundaries) and `converted/<stem>.units.json` (`[{uid, locator, heading_path,
text, kind}]`) and returns the same Units. Locators must match PRD R3 / DESIGN
§4.2 exactly. Fixtures are built in tmp dirs (tests/wu1_fixtures.py); the web
converter is fed local HTML, never a live URL; the image converter is given a
fake `call_stage` (DESIGN §7.1 surface) returning a canned DescribeOutput.

§4.2 "known limitations" are asserted as documented behaviour, not bugs.

Interfaces under test (chosen where DESIGN §20 names only the module):
  kg.convert.Unit(uid, locator, heading_path, text, kind, source)
  kg.convert.ConversionError
  kg.convert.text_md.convert(src, out_dir) -> list[Unit]     (.txt and .md)
  kg.convert.pdf.convert(src, out_dir)
  kg.convert.pptx.convert(src, out_dir)
  kg.convert.docx.convert(src, out_dir)
  kg.convert.xlsx.convert(src, out_dir)
  kg.convert.image.convert(src, out_dir, *, max_long_edge_px, call_stage)
  kg.convert.web.convert_html(url, html, out_dir)            (offline seam)
  kg.convert.web.convert(url, out_dir, *, fetch)             (fetch(url) -> str | None)
"""

from __future__ import annotations

import base64
import io
import json
import re
import unicodedata
from pathlib import Path

import pytest

from wu1_fixtures import (
    DOCX_NO_HEADINGS,
    DOCX_THAI,
    DOCX_WITH_HEADINGS,
    HTML_EMPTY,
    HTML_PAGE,
    MD_SETEXT,
    MD_THAI,
    THAI_PHRASE,
    WEB_URL,
    FakeCallStage,
    canned_describe,
    make_docx,
    make_jpeg,
    make_md,
    make_md_oversize_section,
    make_pdf,
    make_png,
    make_pptx,
    make_text,
    make_xlsx,
)

UNIT_KEYS = {"uid", "locator", "heading_path", "text", "kind"}


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "converted"
    d.mkdir()
    return d


@pytest.fixture
def src_dir(tmp_path: Path) -> Path:
    d = tmp_path / "inbox"
    d.mkdir()
    return d


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _assert_r2_artifacts(out_dir: Path, stem: str, units: list) -> list[dict]:
    """R2: both files exist; units.json mirrors the returned units; md carries unit boundaries."""
    md_path = out_dir / f"{stem}.md"
    units_path = out_dir / f"{stem}.units.json"
    assert md_path.is_file(), f"{md_path.name} not written (R2: intermediate Markdown must be inspectable)"
    assert units_path.is_file(), f"{units_path.name} not written (DESIGN §4.1)"
    data = json.loads(units_path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == len(units) > 0
    for row, unit in zip(data, units):
        assert UNIT_KEYS <= set(row), f"units.json row missing keys: {UNIT_KEYS - set(row)}"
        assert row["uid"] == unit.uid
        assert row["locator"] == unit.locator
        assert row["heading_path"] == list(unit.heading_path)
        assert row["text"] == unit.text
        assert row["kind"] == unit.kind
    md_text = md_path.read_text(encoding="utf-8")
    for unit in units:
        assert f"<!-- unit: {unit.locator} -->" in md_text, f"boundary comment missing for {unit.locator}"
    return data


def _assert_common_unit_shape(units: list, source_name: str) -> None:
    uids = [u.uid for u in units]
    assert len(set(uids)) == len(uids), "unit ids must be unique within a source"
    for u in units:
        assert isinstance(u.uid, str) and u.uid
        assert isinstance(u.locator, str) and u.locator.startswith(source_name), u.locator
        assert isinstance(u.heading_path, list)
        assert isinstance(u.text, str)
        assert isinstance(u.kind, str) and u.kind
        assert u.source == source_name


# =============================================================== .txt / .md


def test_R3_txt_units_are_line_ranges_of_at_most_60_lines(src_dir: Path, out_dir: Path):
    from kg.convert import text_md

    src = make_text(src_dir / "notes.txt", n_lines=150, para_lines=5)
    units = text_md.convert(src, out_dir)

    _assert_common_unit_shape(units, "notes.txt")
    assert len(units) >= 3  # 150 lines / ≤60 per group
    spans = []
    for u in units:
        m = re.fullmatch(r"notes\.txt#lines=(\d+)-(\d+)", u.locator)
        assert m, f"R3: text locator must be file.txt#lines=a-b, got {u.locator}"
        a, b = int(m.group(1)), int(m.group(2))
        assert 1 <= a <= b
        assert b - a + 1 <= 60, f"paragraph group longer than ~60 lines: {u.locator}"
        assert u.heading_path == []
        spans.append((a, b))
    assert spans[0][0] == 1, "first unit must start at line 1"
    for (_, b1), (a2, _) in zip(spans, spans[1:]):
        assert a2 > b1, "line ranges must be ascending and non-overlapping"
    # every non-blank source line appears in some unit text
    all_text = "\n".join(u.text for u in units)
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert line in all_text, f"lost line: {line!r}"
    _assert_r2_artifacts(out_dir, "notes", units)


def test_R3_md_units_follow_atx_heading_paths_with_ancestor_stack(src_dir: Path, out_dir: Path):
    from kg.convert import text_md

    src = make_md(src_dir / "notes.md")
    units = text_md.convert(src, out_dir)
    _assert_common_unit_shape(units, "notes.md")

    heading_locators = {u.locator for u in units if "#heading=" in u.locator}
    assert heading_locators == {
        "notes.md#heading=Intro",
        "notes.md#heading=Intro > Terms",
        "notes.md#heading=Intro > Scope",
        "notes.md#heading=Appendix",
    }
    terms = next(u for u in units if u.locator == "notes.md#heading=Intro > Terms")
    assert terms.heading_path == ["Intro", "Terms"]
    assert "sample space is the set of all possible outcomes" in terms.text
    _assert_r2_artifacts(out_dir, "notes", units)


def test_R3_md_preamble_before_first_heading_gets_a_line_range(src_dir: Path, out_dir: Path):
    from kg.convert import text_md

    units = text_md.convert(make_md(src_dir / "notes.md"), out_dir)
    preamble = [u for u in units if "#lines=" in u.locator]
    assert len(preamble) == 1, "preamble must be exactly one line-range unit (DESIGN §4.2)"
    assert re.fullmatch(r"notes\.md#lines=1-[23]", preamble[0].locator), preamble[0].locator
    assert "Preamble line one" in preamble[0].text and "Preamble line two" in preamble[0].text
    assert preamble[0].heading_path == []
    assert units[0] is preamble[0], "units are in document order; preamble first"


def test_R3_md_setext_headings_fall_back_to_line_ranges_documented_limitation(src_dir: Path, out_dir: Path):
    # DESIGN §4.2 known limitation: setext headings are not recognised; line ranges are used instead.
    from kg.convert import text_md

    units = text_md.convert(make_md(src_dir / "setext.md", MD_SETEXT), out_dir)
    assert units
    assert all("#heading=" not in u.locator for u in units)
    assert all(re.fullmatch(r"setext\.md#lines=\d+-\d+", u.locator) for u in units)
    assert "Body text under a setext heading" in "\n".join(u.text for u in units)


def test_R3_md_oversize_section_is_split_at_blank_lines_with_lines_suffix(src_dir: Path, out_dir: Path):
    from kg.convert import text_md

    units = text_md.convert(make_md_oversize_section(src_dir / "big.md"), out_dir)
    big = [u for u in units if u.locator.startswith("big.md#heading=Big")]
    assert len(big) >= 2, "a ~6k-word section must be split into several units"
    prev_end = 0
    for u in big:
        m = re.fullmatch(r"big\.md#heading=Big;lines=(\d+)-(\d+)", u.locator)
        assert m, f"split section locator must append ;lines=a-b, got {u.locator}"
        a, b = int(m.group(1)), int(m.group(2))
        assert a > prev_end and a <= b
        prev_end = b
        assert u.heading_path == ["Big"]
        assert u.text.strip(), "split must not produce an empty unit"
    # split at blank lines: no unit starts or ends mid-paragraph (paragraphs are 2 lines: wordNNN / tailNNN)
    for u in big:
        first, last = u.text.strip().splitlines()[0], u.text.strip().splitlines()[-1]
        assert first.startswith("word"), f"unit starts mid-paragraph: {first[:30]!r}"
        assert last.startswith("tail"), f"unit ends mid-paragraph: {last[:30]!r}"


def test_R3_md_thai_text_survives_conversion_after_nfkc(src_dir: Path, out_dir: Path):
    # DESIGN §22: real Eddi materials are Thai-heavy; NFKC is the normalisation of record.
    from kg.convert import text_md

    units = text_md.convert(make_md(src_dir / "thai.md", MD_THAI), out_dir)
    joined = nfkc("\n".join(u.text for u in units))
    assert nfkc(THAI_PHRASE) in joined
    md_text = nfkc((out_dir / "thai.md").read_text(encoding="utf-8"))
    assert nfkc(THAI_PHRASE) in md_text
    assert any(nfkc(u.locator) == nfkc("thai.md#heading=บทนำ") for u in units)


def test_R1_text_md_rejects_missing_file(src_dir: Path, out_dir: Path):
    from kg.convert import ConversionError, text_md

    with pytest.raises((ConversionError, FileNotFoundError)):
        text_md.convert(src_dir / "does-not-exist.md", out_dir)
    assert not list(out_dir.iterdir()), "nothing may be written for a failed conversion"


# ===================================================================== .pdf


def test_R3_pdf_one_unit_per_page_with_page_locator(src_dir: Path, out_dir: Path):
    from kg.convert import pdf

    src = make_pdf(src_dir / "deck.pdf", ["Page one text about sample spaces", "", "Page three text"])
    units = pdf.convert(src, out_dir)
    _assert_common_unit_shape(units, "deck.pdf")
    assert [u.locator for u in units] == ["deck.pdf#page=1", "deck.pdf#page=2", "deck.pdf#page=3"]
    assert "Page one text about sample spaces" in units[0].text
    assert "Page three text" in units[2].text
    assert all(u.heading_path == [] for u in units)
    _assert_r2_artifacts(out_dir, "deck", units)


def test_R3_pdf_empty_page_is_flagged_empty_page_not_dropped(src_dir: Path, out_dir: Path):
    # DESIGN §4.2: empty pages flagged `empty_page`; scanned PDFs yield nothing by design (no OCR).
    from kg.convert import pdf

    units = pdf.convert(make_pdf(src_dir / "deck.pdf", ["text", "", "more"]), out_dir)
    assert len(units) == 3, "empty page must still be a unit (so page numbering stays truthful)"
    assert units[1].kind == "empty_page"
    assert units[1].text.strip() == ""
    assert units[0].kind != "empty_page" and units[2].kind != "empty_page"


def test_R1_pdf_corrupt_file_raises_conversion_error_and_writes_nothing(src_dir: Path, out_dir: Path):
    from kg.convert import ConversionError, pdf

    bad = src_dir / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4\nthis is not a pdf body at all\n%%EOF\n")
    with pytest.raises(ConversionError) as ei:
        pdf.convert(bad, out_dir)
    assert str(ei.value).strip(), "failure reason must be non-empty (R1: reported, never silent)"
    assert not (out_dir / "broken.md").exists() and not (out_dir / "broken.units.json").exists()
    assert bad.exists(), "the input is never deleted (R5)"


# ==================================================================== .pptx


def test_R3_pptx_one_unit_per_slide_with_slide_locator_and_notes_kept_in_slide(src_dir: Path, out_dir: Path):
    from kg.convert import pptx

    src = make_pptx(
        src_dir / "deck.pptx",
        [
            ("Sample Space", "All possible outcomes", "Notes for slide one"),
            ("Events", "Subsets of the sample space", None),
            ("Complement", "One minus the probability", "Third notes"),
        ],
    )
    units = pptx.convert(src, out_dir)
    _assert_common_unit_shape(units, "deck.pptx")
    assert [u.locator for u in units] == ["deck.pptx#slide=1", "deck.pptx#slide=2", "deck.pptx#slide=3"]
    assert "Sample Space" in units[0].text and "All possible outcomes" in units[0].text
    assert "Notes for slide one" in units[0].text, "speaker notes stay in the slide unit (DESIGN §4.2)"
    assert "Notes for slide one" not in units[1].text
    assert "Third notes" in units[2].text
    _assert_r2_artifacts(out_dir, "deck", units)


def test_R1_pptx_corrupt_file_raises_conversion_error(src_dir: Path, out_dir: Path):
    from kg.convert import ConversionError, pptx

    bad = src_dir / "broken.pptx"
    bad.write_bytes(b"PK\x03\x04 definitely not a presentation")
    with pytest.raises(ConversionError):
        pptx.convert(bad, out_dir)
    assert not (out_dir / "broken.md").exists()


# ==================================================================== .docx


def test_R3_docx_units_follow_word_heading_styles(src_dir: Path, out_dir: Path):
    from kg.convert import docx

    units = docx.convert(make_docx(src_dir / "spec.docx", DOCX_WITH_HEADINGS), out_dir)
    _assert_common_unit_shape(units, "spec.docx")
    assert [u.locator for u in units] == [
        "spec.docx#heading=1 Scope",
        "spec.docx#heading=1 Scope > 1.2 Terms",
        "spec.docx#heading=2 Method",
    ]
    assert units[1].heading_path == ["1 Scope", "1.2 Terms"]
    assert "discrete case only" in units[0].text
    assert "set of all possible outcomes" in units[1].text and "subset of the sample space" in units[1].text
    assert "favourable outcomes" in units[2].text
    _assert_r2_artifacts(out_dir, "spec", units)


def test_R3_docx_without_headings_falls_back_to_paragraph_index_blocks(src_dir: Path, out_dir: Path):
    # DESIGN §4.2: zero headings → ~15-paragraph blocks, locator `#para=a-b`. Needs Word heading styles (limitation).
    from kg.convert import docx

    units = docx.convert(make_docx(src_dir / "plain.docx", DOCX_NO_HEADINGS), out_dir)
    assert len(units) >= 2, "40 paragraphs must yield more than one ~15-paragraph block"
    assert all("#heading=" not in u.locator for u in units)
    prev_end = 0
    for u in units:
        m = re.fullmatch(r"plain\.docx#para=(\d+)-(\d+)", u.locator)
        assert m, f"R3: DOCX fallback locator must be file.docx#para=a-b, got {u.locator}"
        a, b = int(m.group(1)), int(m.group(2))
        assert a == prev_end + 1, "paragraph blocks must be contiguous and 1-based"
        assert a <= b
        prev_end = b
        assert u.heading_path == []
    assert prev_end == 40, "every paragraph must be covered exactly once"
    assert "Paragraph 01" in units[0].text and "Paragraph 40" in units[-1].text


def test_R3_docx_thai_text_survives_conversion_after_nfkc(src_dir: Path, out_dir: Path):
    from kg.convert import docx

    units = docx.convert(make_docx(src_dir / "thai.docx", DOCX_THAI), out_dir)
    assert nfkc(THAI_PHRASE) in nfkc("\n".join(u.text for u in units))
    assert nfkc(THAI_PHRASE) in nfkc((out_dir / "thai.md").read_text(encoding="utf-8"))
    assert nfkc(units[0].locator) == nfkc("thai.docx#heading=บทนำ")


def test_R1_docx_corrupt_file_raises_conversion_error(src_dir: Path, out_dir: Path):
    from kg.convert import ConversionError, docx

    bad = src_dir / "broken.docx"
    bad.write_bytes(b"not a zip at all")
    with pytest.raises(ConversionError):
        docx.convert(bad, out_dir)
    assert not (out_dir / "broken.md").exists()


# ==================================================================== .xlsx


def test_R3_xlsx_units_are_sheet_blocks_with_a1_range_locators(src_dir: Path, out_dir: Path):
    from kg.convert import xlsx

    units = xlsx.convert(make_xlsx(src_dir / "terms.xlsx"), out_dir)
    _assert_common_unit_shape(units, "terms.xlsx")
    locators = [u.locator for u in units]
    assert locators[0] == "terms.xlsx#Sheet1!A1:C4", locators
    assert re.fullmatch(r"terms\.xlsx#Sheet1!A7:[BC]8", locators[1]), locators  # block after ≥2 empty rows
    assert locators[2] == "terms.xlsx#Grades!A1:B3", locators
    assert len(units) == 3
    _assert_r2_artifacts(out_dir, "terms", units)


def test_R2_xlsx_unit_text_is_markdown_table_with_row_numbers_and_column_letters(src_dir: Path, out_dir: Path):
    from kg.convert import xlsx

    units = xlsx.convert(make_xlsx(src_dir / "terms.xlsx"), out_dir)
    first = units[0].text
    assert re.search(r"\|\s*A\s*\|\s*B\s*\|\s*C\s*\|", first), "header row must be column letters"
    assert re.search(r"^\|\s*1\s*\|\s*Term\s*\|", first, re.M), "Excel row number is the first column"
    assert re.search(r"^\|\s*2\s*\|\s*Sample space\s*\|\s*All possible outcomes\s*\|\s*3\s*\|", first, re.M)
    grades = units[2].text
    assert re.search(r"^\|\s*3\s*\|\s*B\s*\|\s*75\s*\|", grades, re.M)


def test_R3_xlsx_formula_without_cached_value_is_blank_documented_limitation(src_dir: Path, out_dir: Path):
    # DESIGN §4.2: `data_only=True` → cached values only. openpyxl-written formulas have no cache → empty cell.
    from kg.convert import xlsx

    units = xlsx.convert(make_xlsx(src_dir / "terms.xlsx"), out_dir)
    second_block = units[1].text
    assert "SUM(" not in second_block, "formula source text must never leak into the unit"
    assert "Total" in second_block


def test_R1_xlsx_corrupt_file_raises_conversion_error(src_dir: Path, out_dir: Path):
    from kg.convert import ConversionError, xlsx

    bad = src_dir / "broken.xlsx"
    bad.write_bytes(b"PK\x03\x04 not a workbook")
    with pytest.raises(ConversionError):
        xlsx.convert(bad, out_dir)
    assert not (out_dir / "broken.md").exists()


# =================================================================== images


def _decode_image(images: list):
    from PIL import Image

    assert isinstance(images, list) and len(images) == 1, "exactly one ImageInput per image"
    img_in = images[0]
    assert hasattr(img_in, "media_type") and hasattr(img_in, "data_b64"), "ImageInput{media_type, data_b64} (DESIGN §4.3)"
    raw = base64.b64decode(img_in.data_b64, validate=True)
    return img_in.media_type, Image.open(io.BytesIO(raw))


def test_R6_image_unit_is_whole_image_locator_with_rendered_description(src_dir: Path, out_dir: Path):
    from kg.convert import image
    from kg.schemas import DescribeOutput

    src = make_png(src_dir / "diagram.png")
    fake = FakeCallStage([canned_describe()])
    units = image.convert(src, out_dir, max_long_edge_px=400, call_stage=fake)

    _assert_common_unit_shape(units, "diagram.png")
    assert len(units) == 1
    u = units[0]
    assert u.locator == "diagram.png", "R3: image locator is the whole file, no fragment"
    # §4.3 / §8.0: Markdown rendered from the structured fields — title as `#`, description, bullet lists.
    assert re.search(r"^# Venn diagram of two events\s*$", u.text, re.M), u.text
    assert "Two overlapping circles labelled A and B" in u.text
    for item in ("Circle A", "Circle B", "Rectangle S", "A overlaps B", "A and B lie inside S"):
        assert re.search(rf"^\s*[-*] .*{re.escape(item)}", u.text, re.M), f"bullet for {item!r} missing:\n{u.text}"
    # the call crossed the boundary with the right stage and schema
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["stage"] == "describe"
    assert call["schema"] is DescribeOutput
    assert isinstance(call["system"], str) and call["system"].strip()
    _assert_r2_artifacts(out_dir, "diagram", units)
    assert "Two overlapping circles" in (out_dir / "diagram.md").read_text(encoding="utf-8")


def test_R6_image_is_downscaled_to_max_long_edge_before_the_call(src_dir: Path, out_dir: Path):
    from kg.convert import image

    src = make_png(src_dir / "diagram.png", size=(2400, 1200))
    original_bytes = src.read_bytes()
    fake = FakeCallStage([canned_describe()])
    image.convert(src, out_dir, max_long_edge_px=400, call_stage=fake)

    media_type, sent = _decode_image(fake.calls[0]["images"])
    assert media_type == "image/png"
    assert max(sent.size) <= 400, f"Pillow downscale must happen BEFORE the call; sent {sent.size}"
    w, h = sent.size
    assert abs(w / h - 2.0) < 0.05, "aspect ratio preserved"
    assert src.read_bytes() == original_bytes, "the original file is untouched (R5)"


def test_R6_image_smaller_than_max_edge_is_not_upscaled(src_dir: Path, out_dir: Path):
    from kg.convert import image

    src = make_png(src_dir / "small.png", size=(300, 200))
    fake = FakeCallStage([canned_describe()])
    image.convert(src, out_dir, max_long_edge_px=1568, call_stage=fake)
    _, sent = _decode_image(fake.calls[0]["images"])
    assert sent.size == (300, 200)


def test_R6_jpeg_is_sent_with_jpeg_media_type(src_dir: Path, out_dir: Path):
    from kg.convert import image

    src = make_jpeg(src_dir / "photo.jpg")
    fake = FakeCallStage([canned_describe()])
    units = image.convert(src, out_dir, max_long_edge_px=1568, call_stage=fake)
    media_type, _ = _decode_image(fake.calls[0]["images"])
    assert media_type == "image/jpeg"
    assert units[0].locator == "photo.jpg"
    assert (out_dir / "photo.md").is_file()


def test_R6_image_adapter_failure_propagates_and_writes_nothing(src_dir: Path, out_dir: Path):
    from kg.convert import image

    src = make_png(src_dir / "diagram.png", size=(200, 100))
    fake = FakeCallStage(raise_exc=RuntimeError("provider unavailable"))
    with pytest.raises(Exception):
        image.convert(src, out_dir, max_long_edge_px=1568, call_stage=fake)
    assert not (out_dir / "diagram.md").exists(), "no Markdown for a failed description (nothing partial, D20/§6)"
    assert not (out_dir / "diagram.units.json").exists()
    assert src.exists()


def test_R6_image_corrupt_bytes_raise_conversion_error_without_calling_the_model(src_dir: Path, out_dir: Path):
    from kg.convert import ConversionError, image

    bad = src_dir / "broken.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n not really a png")
    fake = FakeCallStage([canned_describe()])
    with pytest.raises(ConversionError):
        image.convert(bad, out_dir, max_long_edge_px=1568, call_stage=fake)
    assert fake.calls == [], "no tokens spent on an unreadable image"


# ====================================================================== web


def test_R3_web_units_follow_heading_paths_prefixed_by_url(out_dir: Path):
    from kg.convert import web

    units = web.convert_html(WEB_URL, HTML_PAGE, out_dir)
    _assert_common_unit_shape(units, WEB_URL)
    assert {u.locator for u in units} == {
        f"{WEB_URL}#heading=Probability Basics",
        f"{WEB_URL}#heading=Probability Basics > Sample Space",
        f"{WEB_URL}#heading=Probability Basics > Events",
        f"{WEB_URL}#heading=Probability Basics > Events > Complement",
    }
    comp = next(u for u in units if u.locator.endswith("> Complement"))
    assert comp.heading_path == ["Probability Basics", "Events", "Complement"]
    assert "one minus the probability of the event" in comp.text


def test_R2_web_boilerplate_links_and_images_are_dropped(out_dir: Path):
    # trafilatura called with include_links=False, include_images=False (DESIGN §4.2)
    from kg.convert import web

    units = web.convert_html(WEB_URL, HTML_PAGE, out_dir)
    joined = "\n".join(u.text for u in units)
    assert "https://example.org/more" not in joined
    assert "diagram.png" not in joined
    assert "Copyright notice" not in joined
    assert "Home | About" not in joined and "](/" not in joined


def test_R2_web_raw_html_is_archived_and_markdown_written_under_host_slug(out_dir: Path):
    from kg.convert import web

    units = web.convert_html(WEB_URL, HTML_PAGE, out_dir)
    archives = sorted(out_dir.glob("*.html"))
    assert len(archives) == 1, "raw HTML archived to converted/<host>-<slug>.html"
    archive = archives[0]
    assert archive.name.startswith("learn.example.edu-"), archive.name
    assert archive.read_text(encoding="utf-8") == HTML_PAGE
    stem = archive.name[: -len(".html")]
    _assert_r2_artifacts(out_dir, stem, units)


def test_R1_web_page_without_extractable_text_fails_with_reason(out_dir: Path):
    # DESIGN §4.2 limitation: JS-rendered pages → `failed: no extractable text`
    from kg.convert import ConversionError, web

    with pytest.raises(ConversionError) as ei:
        web.convert_html("https://app.example.com/spa", HTML_EMPTY, out_dir)
    assert "no extractable text" in str(ei.value)
    assert not list(out_dir.glob("*.md")) and not list(out_dir.glob("*.units.json"))


def test_R1_web_convert_uses_injected_fetch_and_fails_when_fetch_returns_none(out_dir: Path):
    from kg.convert import ConversionError, web

    seen: list[str] = []

    def fetch_ok(url: str) -> str | None:
        seen.append(url)
        return HTML_PAGE

    units = web.convert(WEB_URL, out_dir, fetch=fetch_ok)
    assert seen == [WEB_URL]
    assert len(units) == 4

    def fetch_fail(url: str) -> str | None:
        return None

    with pytest.raises(ConversionError):
        web.convert("https://down.example.org/x", out_dir, fetch=fetch_fail)
