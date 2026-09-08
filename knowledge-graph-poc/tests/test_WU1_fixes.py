"""WU1 fix cycle 1 — regression pins for commits cb2a52a, 0725af6, e84f908, fbd69b5 (R1, R2, R3, R6).

Each test pins one behaviour the reviewers flagged and the implementer fixed, so a
later refactor cannot silently undo it:

  fix_web_stem_*        R2  web stems are `<host>-<slug>-<sha1[:8]>`: `/Page` vs `/page/`,
                            http vs https, two Thai paths no longer overwrite each other;
                            a file named like a web stem never shares an output
  fix_thai_tokens_*     R3  estimate_tokens counts one token per non-ASCII character
  fix_xlsx_range_*      R3  an xlsx range resolves only if the WHOLE range is in the used area
  fix_urls_txt_*        R1  urls.txt itself is validated: symlink outside inbox, non-UTF-8,
                            scheme-less / ftp lines each get a per-item outcome with a reason
  fix_cap_*             R1  parser resource caps: OOXML inflate / ratio, xlsx container part,
                            PDF page count, text byte size → `failed` with the cap in the reason
  fix_image_*           R6  a Pillow decompression bomb fails BEFORE the model boundary (no
                            call); a wrong-typed stage result AFTER it is `deferred`
  fix_atomic_*          R2  outputs land via .tmp + os.replace: a failed write leaves the
                            existing target untouched and no *.tmp behind
  fix_discover_*        R1  dotfiles skipped; dangling / directory symlinks reported as
                            `failed`; control characters in a file name refused by the sandbox
  fix_fence_*           R3  a ``` fence is closed only by a ``` fence, never by ~~~

Only public surfaces are exercised (kg.convert, kg.convert.web/text_md,
kg.convert.locators, kg.chunk); the cap constants are read from
kg.convert._common because that is where the implementation defines them.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kg.config import Config, load_config
from kg.convert._common import MAX_INFLATED_BYTES, MAX_PDF_PAGES, MAX_TEXT_BYTES, ZIP_RATIO_FLOOR_BYTES, ConversionError
from wu1_fixtures import HTML_PAGE, WEB_URL, FakeCallStage, canned_describe, make_md, make_pdf, make_png, make_xlsx

pytestmark = pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")


@pytest.fixture
def cfg(project_root: Path) -> Config:
    return load_config(project_root / "kg.yaml")


def fetch_any(url: str) -> str | None:
    """Offline fetcher: every URL serves the same saved page."""
    return HTML_PAGE


def outcome_by_name(outcomes) -> dict[str, object]:
    return {o.item.name: o for o in outcomes}


# ================================================================ 1. web stems

COLLIDING_URLS = (
    "https://h.example/Page",
    "https://h.example/page/",
    "http://h.example/page",
    "https://h.example/ไทย-หนึ่ง",
    "https://h.example/ไทย-สอง",
)


def test_R2_fix_web_stem_colliding_slugs_get_distinct_hashed_stems():
    from kg.convert import web

    stems = [web.stem_for(u) for u in COLLIDING_URLS]
    assert len(set(stems)) == len(stems), stems
    for stem in stems:
        host, _, rest = stem.partition("-")
        assert host == "h.example"
        slug, _, digest = rest.rpartition("-")
        assert slug, stem
        assert len(digest) == 8 and all(c in "0123456789abcdef" for c in digest), f"{stem}: tail must be sha1[:8]"


def test_R2_fix_web_stem_is_deterministic_across_calls():
    from kg.convert import web

    first = [web.stem_for(u) for u in COLLIDING_URLS]
    second = [web.stem_for(u) for u in COLLIDING_URLS]
    assert first == second


def test_R2_fix_web_stem_host_is_capped_at_64_chars():
    from kg.convert import web

    stem = web.stem_for("https://" + "a" * 200 + ".example/x")
    host = stem.rsplit("-x-", 1)[0]
    assert len(host) <= web.MAX_HOST_LEN == 64, host
    assert stem.endswith(tuple("0123456789abcdef")), "hash tail still present after host truncation"


def test_R2_fix_web_stem_file_named_like_a_web_stem_shares_no_output(cfg: Config):
    # Unhappy path: a PDF deliberately named `<web-stem>.pdf` alongside the URL itself.
    from kg.convert import InputItem, convert_inbox, output_stems, web

    inbox = cfg.sandbox.root("inbox")
    web_stem = web.stem_for(WEB_URL)
    make_pdf(inbox / f"{web_stem}.pdf", ["Page one"])
    (inbox / "urls.txt").write_text(WEB_URL + "\n", encoding="utf-8")

    items = [InputItem(name=WEB_URL, kind="web", path=None, url=WEB_URL), InputItem(name=f"{web_stem}.pdf", kind="pdf", path=Path("x"), url=None)]
    stems = output_stems(items)
    assert len(set(stems.values())) == 2, stems

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage(), fetch=fetch_any)
    assert [o.status for o in outcomes] == ["converted", "converted"], [(o.item.name, o.status, o.reason) for o in outcomes]
    all_outputs = [p for o in outcomes for p in o.outputs]
    assert len(all_outputs) == len(set(all_outputs)), "two inputs must never share an output file"
    assert all(p.is_file() for p in all_outputs)


def test_R2_fix_web_stem_colliding_urls_convert_to_distinct_files_deterministically(cfg: Config):
    from kg.convert import convert_inbox

    inbox = cfg.sandbox.root("inbox")
    (inbox / "urls.txt").write_text("\n".join(COLLIDING_URLS) + "\n", encoding="utf-8")

    def run() -> dict[str, tuple[str, ...]]:
        outs = convert_inbox(cfg, call_stage=FakeCallStage(), fetch=fetch_any)
        assert [o.status for o in outs] == ["converted"] * len(COLLIDING_URLS), [(o.item.name, o.status, o.reason) for o in outs]
        return {o.item.name: tuple(sorted(p.name for p in o.outputs)) for o in outs}

    first = run()
    md_files = {names for names in first.values()}
    assert len(md_files) == len(COLLIDING_URLS), "every URL must get its own .html/.md/.units.json triple"
    assert all(len(names) == 3 for names in first.values())
    assert run() == first, "web stems must be identical on a second run (R16)"


# ============================================================ 2. Thai tokens


def test_R3_fix_thai_tokens_one_token_per_non_ascii_character():
    from kg.chunk import estimate_tokens

    thai = "การเรียนรู้เชิงลึกคือสาขาหนึ่งของการเรียนรู้ของเครื่อง"
    assert " " not in thai
    assert estimate_tokens(thai) == len(thai)


def test_R3_fix_thai_tokens_ascii_behaviour_unchanged():
    from kg.chunk import estimate_tokens

    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") == 2, "one token per whitespace-separated ASCII word"
    assert estimate_tokens("a" * 20) == 1 + 20 // 8, "plus one per further 8 ASCII characters"
    assert estimate_tokens("  spaced   out  ") == 2


def test_R3_fix_thai_tokens_mixed_piece_counts_both_parts():
    from kg.chunk import estimate_tokens

    # "ab" + 3 Thai characters in one whitespace-free piece: 1 (ASCII word) + 3 (non-ASCII chars)
    assert estimate_tokens("abกขค") == 1 + 3


# ============================================================ 3. xlsx ranges


@pytest.fixture
def xlsx_dir(tmp_path: Path) -> Path:
    make_xlsx(tmp_path / "terms.xlsx")  # Sheet1 used area: 8 rows x 3 columns (A1:C8)
    return tmp_path


@pytest.mark.parametrize("rng", ["Sheet1!A1:ZZ999", "Sheet1!A1:C9", "Sheet1!A1:D8"])
def test_R3_fix_xlsx_range_partly_outside_used_area_does_not_resolve(xlsx_dir: Path, rng: str):
    from kg.convert import resolve_locator

    assert resolve_locator(f"terms.xlsx#{rng}", xlsx_dir) is False, rng


def test_R3_fix_xlsx_range_fully_inside_used_area_resolves(xlsx_dir: Path):
    from kg.convert import resolve_locator

    assert resolve_locator("terms.xlsx#Sheet1!A1:C4", xlsx_dir) is True
    assert resolve_locator("terms.xlsx#Sheet1!A1:C8", xlsx_dir) is True, "the whole used area is itself a valid range"


# ============================================================== 4. urls.txt


def test_R1_fix_urls_txt_symlink_outside_inbox_is_one_failed_item(cfg: Config):
    from kg.convert import convert_inbox

    inbox = cfg.sandbox.root("inbox")
    outside = cfg.project_root / "outside-urls.txt"
    outside.write_text("https://evil.example/x\n", encoding="utf-8")
    (inbox / "urls.txt").symlink_to(outside)

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert len(outcomes) == 1, [(o.item.name, o.status) for o in outcomes]
    assert outcomes[0].status == "failed"
    assert outcomes[0].reason and "sandbox refused" in outcomes[0].reason
    assert outcomes[0].outputs == () and outcomes[0].units == []


def test_R1_fix_urls_txt_non_utf8_is_one_failed_item_naming_utf8(cfg: Config):
    from kg.convert import convert_inbox

    (cfg.sandbox.root("inbox") / "urls.txt").write_bytes(b"\xff\xfe https://x.example/a\n")
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.item.name, o.status, o.reason) for o in outcomes]
    assert outcomes[0].reason and "UTF-8" in outcomes[0].reason


def test_R1_fix_urls_txt_ftp_and_scheme_less_lines_are_unsupported_with_reason(cfg: Config):
    from kg.convert import convert_inbox

    (cfg.sandbox.root("inbox") / "urls.txt").write_text(
        "ftp://x.example/a\nlearn.example.edu/no-scheme\n" + WEB_URL + "\n", encoding="utf-8"
    )
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage(), fetch=fetch_any)
    by = outcome_by_name(outcomes)
    assert set(by) == {"ftp://x.example/a", "learn.example.edu/no-scheme", WEB_URL}
    for bad in ("ftp://x.example/a", "learn.example.edu/no-scheme"):
        assert by[bad].status == "unsupported", (bad, by[bad].status, by[bad].reason)
        assert by[bad].reason and bad in by[bad].reason, "reason must quote the offending line"
        assert by[bad].outputs == ()
    assert by[WEB_URL].status == "converted", "a bad line must not take the good URL down with it"


# ========================================================== 5. resource caps


def _zip_with(path: Path, parts: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
    return path


def test_R1_fix_cap_docx_inflating_over_max_bytes_fails_naming_the_cap(cfg: Config):
    from kg.convert import convert_inbox

    _zip_with(cfg.sandbox.root("inbox") / "bomb.docx", {"word/document.xml": b"\0" * (MAX_INFLATED_BYTES + 1)})
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.status, o.reason) for o in outcomes]
    assert outcomes[0].reason and "cap" in outcomes[0].reason, outcomes[0].reason
    assert not list(cfg.sandbox.root("converted").glob("bomb*"))


def test_R1_fix_cap_docx_extreme_compression_ratio_fails_naming_the_ratio(cfg: Config):
    from kg.convert import convert_inbox

    size = 2 * 2**20
    assert size > ZIP_RATIO_FLOOR_BYTES and size <= MAX_INFLATED_BYTES, "must trip the ratio check, not the size cap"
    _zip_with(cfg.sandbox.root("inbox") / "ratio.docx", {"word/document.xml": b"\0" * size})
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.status, o.reason) for o in outcomes]
    assert outcomes[0].reason and "ratio" in outcomes[0].reason, outcomes[0].reason


def test_R1_fix_cap_xlsx_zip_without_workbook_part_fails(cfg: Config):
    from kg.convert import convert_inbox

    _zip_with(cfg.sandbox.root("inbox") / "fake.xlsx", {"hello.txt": b"nope"})
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.status, o.reason) for o in outcomes]
    assert outcomes[0].reason and "xl/workbook.xml" in outcomes[0].reason, outcomes[0].reason


def test_R1_fix_cap_pdf_over_max_pages_fails_naming_the_cap(cfg: Config):
    from kg.convert import convert_inbox

    make_pdf(cfg.sandbox.root("inbox") / "big.pdf", ["p"] * (MAX_PDF_PAGES + 1))
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.status, o.reason) for o in outcomes]
    assert outcomes[0].reason and "cap" in outcomes[0].reason and str(MAX_PDF_PAGES) in outcomes[0].reason, outcomes[0].reason
    assert not list(cfg.sandbox.root("converted").glob("big*"))


def test_R1_fix_cap_pdf_at_exactly_max_pages_is_not_refused_by_the_cap(cfg: Config):
    # Boundary: the cap is "over", not "at or over".
    from kg.convert import convert_inbox

    make_pdf(cfg.sandbox.root("inbox") / "edge.pdf", ["p"] * MAX_PDF_PAGES)
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["converted"], [(o.status, o.reason) for o in outcomes]
    assert len(outcomes[0].units) == MAX_PDF_PAGES


def test_R1_fix_cap_text_file_over_max_bytes_fails_naming_the_cap(cfg: Config):
    from kg.convert import convert_inbox

    big = cfg.sandbox.root("inbox") / "big.txt"
    with big.open("wb") as fh:
        fh.truncate(MAX_TEXT_BYTES + 1)  # sparse: nothing is actually read if the cap works
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.status, o.reason) for o in outcomes]
    assert outcomes[0].reason and "cap" in outcomes[0].reason, outcomes[0].reason


# ==================================================================== 6. image


def test_R6_fix_image_decompression_bomb_fails_before_the_model_boundary(cfg: Config, monkeypatch: pytest.MonkeyPatch):
    from PIL import Image

    from kg.convert import convert_inbox

    src = make_png(cfg.sandbox.root("inbox") / "bomb.png", size=(200, 200))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # 200*200 > 2*100 → Pillow raises DecompressionBombError
    fake = FakeCallStage([canned_describe()])
    outcomes = convert_inbox(cfg, call_stage=fake)
    assert [o.status for o in outcomes] == ["failed"], [(o.status, o.reason) for o in outcomes]
    assert outcomes[0].reason and outcomes[0].reason.strip()
    assert fake.calls == [], "no tokens may be spent on an image Pillow refuses to decode"
    assert src.exists()
    assert not (cfg.sandbox.root("converted") / "bomb.md").exists()


def test_R6_fix_image_wrong_typed_stage_result_is_deferred(cfg: Config):
    from kg.convert import convert_inbox

    make_png(cfg.sandbox.root("inbox") / "d.png", size=(50, 50))
    calls: list[str] = []

    def bad_stage(stage: str, *a, **k):
        calls.append(stage)
        return SimpleNamespace(data={"not": "a DescribeOutput"})

    outcomes = convert_inbox(cfg, call_stage=bad_stage)
    assert [o.status for o in outcomes] == ["deferred"], [(o.status, o.reason) for o in outcomes]
    assert calls == ["describe"], "the model boundary was reached exactly once"
    assert outcomes[0].reason and "DescribeOutput" in outcomes[0].reason
    assert outcomes[0].outputs == ()
    assert not (cfg.sandbox.root("converted") / "d.md").exists()


# ============================================================ 7. atomic writes


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permission bits")
def test_R2_fix_atomic_write_into_read_only_dir_leaves_target_and_no_tmp(tmp_path: Path):
    from kg.convert import web

    ro = tmp_path / "ro"
    ro.mkdir()
    target = ro / f"{web.stem_for(WEB_URL)}.md"
    target.write_text("old", encoding="utf-8")
    os.chmod(ro, 0o500)
    try:
        with pytest.raises(ConversionError) as excinfo:
            web.convert_html(WEB_URL, HTML_PAGE, ro)
        assert "cannot write" in str(excinfo.value)
        assert target.read_text(encoding="utf-8") == "old", "pre-existing target must be untouched"
        assert sorted(p.name for p in ro.iterdir()) == [target.name], "no *.tmp or partial output left behind"
    finally:
        os.chmod(ro, 0o700)


def test_R2_fix_atomic_write_success_leaves_no_tmp_files(tmp_path: Path):
    from kg.convert import web

    out = tmp_path / "out"
    out.mkdir()
    web.convert_html(WEB_URL, HTML_PAGE, out)
    names = sorted(p.name for p in out.iterdir())
    assert len(names) == 3 and not any(n.endswith(".tmp") for n in names), names


# ================================================================ 8. discovery


def test_R1_fix_discover_dotfile_skipped_symlinks_reported_as_failed(cfg: Config):
    from kg.convert import convert_inbox

    inbox = cfg.sandbox.root("inbox")
    (inbox / ".DS_Store").write_bytes(b"\0Bud1")
    (inbox / "dangling.pdf").symlink_to(cfg.project_root / "nowhere.pdf")
    (inbox / "linkdir").symlink_to(cfg.project_root)
    (inbox / "realdir").mkdir()

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    by = outcome_by_name(outcomes)
    assert set(by) == {"dangling.pdf", "linkdir"}, sorted(by)
    for name in ("dangling.pdf", "linkdir"):
        assert by[name].status == "failed", (name, by[name].status)
        assert by[name].reason and name in by[name].reason, "reason must name the offending entry"
    assert ".DS_Store" not in by and "realdir" not in by


def test_R1_fix_discover_control_char_in_name_is_refused_by_sandbox(cfg: Config):
    from kg.convert import InputItem, convert_item

    real = make_md(cfg.sandbox.root("inbox") / "real.md")
    item = InputItem(name="a\nb.md", kind="markdown", path=real, url=None)
    outcome = convert_item(item, cfg, call_stage=FakeCallStage())
    assert outcome.status == "failed"
    assert outcome.reason and outcome.reason.startswith("sandbox refused"), outcome.reason
    assert outcome.outputs == ()
    assert list(cfg.sandbox.root("converted").iterdir()) == []


def test_R1_fix_discover_control_char_file_in_inbox_gets_a_failed_outcome(cfg: Config):
    from kg.convert import convert_inbox

    inbox = cfg.sandbox.root("inbox")
    try:
        make_md(inbox / "evil\x01.md")
    except OSError:
        pytest.skip("filesystem refuses control characters in file names")
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert len(outcomes) == 1
    assert outcomes[0].status == "failed" and outcomes[0].reason and "sandbox refused" in outcomes[0].reason
    assert list(cfg.sandbox.root("converted").iterdir()) == []


# ============================================================ 9. fence tracking


def test_R3_fix_fence_tilde_does_not_close_a_backtick_fence(tmp_path: Path):
    from kg.convert import text_md

    lines = ["```", "# x", "~~~", "# y", "```", "# Real"]
    md = make_md(tmp_path / "fence.md", "\n".join(lines) + "\n")
    assert text_md.markdown_heading_paths(md) == {"Real"}


def test_R3_fix_fence_matching_fence_still_closes(tmp_path: Path):
    from kg.convert import text_md

    md = make_md(tmp_path / "fence.md", "```\n# x\n```\n# Real\n~~~\n# y\n~~~\n# Also\n")
    assert text_md.markdown_heading_paths(md) == {"Real", "Also"}


# =====================================================================================
# WU1 fix cycle 2 — regression pins for commit 8939d36 (R1, R2)
#
#   fix2_tmp_*        R2  the `.tmp` staging path is sandbox-resolved and created O_EXCL|O_NOFOLLOW:
#                         a planted `<target>.tmp` symlink is refused (outside) or fails the write
#                         (inside); the planted file is never followed, never deleted
#   fix2_reason_*     R1  reasons never carry raw control characters and a generic exception
#                         reason is `<Type>: <first line>` capped at 200 chars
#   fix2_stems_*      R2  output_stems judges collisions under casefold() (APFS), keeps case
#   fix2_fragment_*   R2  a urls.txt `<url>#fragment` line is fetched/hashed without the fragment;
#                         item.name keeps the line; every unit locator resolves
#   fix2_ratio_*      R1  MAX_ZIP_RATIO is 300: ~1000:1 still trips, ~220:1 no longer does
# =====================================================================================

import random
import re

from kg.convert._common import MAX_ZIP_RATIO

_RAW_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _plant_a_txt(cfg: Config) -> Path:
    src = cfg.sandbox.root("inbox") / "a.txt"
    src.write_text("plain text body about sample spaces\n", encoding="utf-8")
    return src


# ------------------------------------------------------------- (a)(b) tmp symlinks


def test_R2_fix2_tmp_symlink_to_outside_is_refused_by_sandbox_and_victim_untouched(cfg: Config, tmp_path_factory: pytest.TempPathFactory):
    from kg.convert import convert_inbox

    _plant_a_txt(cfg)
    victim = tmp_path_factory.mktemp("victim") / "victim"
    victim.write_text("victim original", encoding="utf-8")
    assert not str(victim).startswith(str(cfg.project_root)), "victim must lie outside the project root"
    converted = cfg.sandbox.root("converted")
    planted = converted / "a.md.tmp"
    planted.symlink_to(victim)

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.item.name, o.status, o.reason) for o in outcomes]
    reason = outcomes[0].reason or ""
    assert "sandbox refused" in reason and "a.md.tmp" in reason, reason
    assert outcomes[0].outputs == () and outcomes[0].units == []
    assert victim.read_text(encoding="utf-8") == "victim original", "the symlink must never be written through"
    assert not (converted / "a.md").exists() and not (converted / "a.units.json").exists()
    assert planted.is_symlink(), "foreign files in converted/ are never deleted"
    assert sorted(p.name for p in converted.iterdir()) == ["a.md.tmp"], "no other staging file may be left behind"


def test_R2_fix2_tmp_symlink_inside_converted_fails_write_and_is_not_followed_or_deleted(cfg: Config):
    from kg.convert import convert_inbox

    _plant_a_txt(cfg)
    converted = cfg.sandbox.root("converted")
    other = converted / "other.md"
    other.write_text("other original", encoding="utf-8")
    planted = converted / "a.md.tmp"
    planted.symlink_to(other)

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.item.name, o.status, o.reason) for o in outcomes]
    reason = outcomes[0].reason or ""
    assert "cannot write" in reason, reason
    assert outcomes[0].outputs == ()
    assert other.read_text(encoding="utf-8") == "other original", "O_NOFOLLOW/O_EXCL: the link target must not be written through"
    assert planted.is_symlink() and os.readlink(planted) == str(other), "the planted symlink is foreign: never unlinked"
    assert not (converted / "a.md").exists(), "os.replace must not have moved anything into place"
    assert sorted(p.name for p in converted.iterdir()) == ["a.md.tmp", "other.md"], "no *.units.json.tmp left behind"


def test_R2_fix2_tmp_plain_file_pre_planted_is_also_refused_not_overwritten(cfg: Config):
    # Unhappy path without a symlink: an ordinary stale `.tmp` is O_EXCL-refused too, and kept.
    from kg.convert import convert_inbox

    _plant_a_txt(cfg)
    converted = cfg.sandbox.root("converted")
    stale = converted / "a.md.tmp"
    stale.write_text("stale staging file", encoding="utf-8")

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.item.name, o.status, o.reason) for o in outcomes]
    assert "cannot write" in (outcomes[0].reason or ""), outcomes[0].reason
    assert stale.read_text(encoding="utf-8") == "stale staging file"
    assert not (converted / "a.md").exists()


# ----------------------------------------------------------- (c)(d) reason hygiene


def test_R1_fix2_reason_exception_from_model_boundary_is_deferred_capped_and_control_free(cfg: Config):
    from kg.convert import convert_inbox

    make_png(cfg.sandbox.root("inbox") / "d.png", size=(50, 50))
    fake = FakeCallStage(raise_exc=RuntimeError("provider unavailable\x07 " + "x" * 1000))

    outcomes = convert_inbox(cfg, call_stage=fake)
    assert [o.status for o in outcomes] == ["deferred"], [(o.status, (o.reason or "")[:60]) for o in outcomes]
    reason = outcomes[0].reason or ""
    assert len(fake.calls) == 1, "the model boundary was reached exactly once"
    assert len(reason) <= 200, len(reason)
    assert reason.endswith("..."), "a truncated reason must say so"
    assert reason.startswith("RuntimeError: provider unavailable x"), reason[:60]
    assert "\x07" not in reason and not _RAW_CONTROL.search(reason), reason
    assert outcomes[0].outputs == () and not (cfg.sandbox.root("converted") / "d.md").exists()


def test_R1_fix2_reason_exception_multiline_keeps_first_line_only(cfg: Config):
    # SDK errors carry response bodies after the first line; only the first line is a reason.
    from kg.convert import convert_inbox

    make_png(cfg.sandbox.root("inbox") / "d.png", size=(50, 50))
    fake = FakeCallStage(raise_exc=RuntimeError("HTTP 529 overloaded\n{\"body\": \"" + "y" * 5000 + "\"}"))

    outcomes = convert_inbox(cfg, call_stage=fake)
    assert [o.status for o in outcomes] == ["deferred"]
    assert outcomes[0].reason == "RuntimeError: HTTP 529 overloaded", outcomes[0].reason


def test_R1_fix2_reason_exception_with_empty_message_is_just_the_type(cfg: Config):
    from kg.convert import convert_inbox

    make_png(cfg.sandbox.root("inbox") / "d.png", size=(50, 50))
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage(raise_exc=RuntimeError()))
    assert [o.status for o in outcomes] == ["deferred"]
    assert outcomes[0].reason == "RuntimeError", outcomes[0].reason


def _fetch_raising(url: str) -> str | None:
    raise OSError("connection\x07 reset\nby peer")


def _fetch_empty(url: str) -> str | None:
    return None


@pytest.mark.parametrize("fetch", [_fetch_raising, _fetch_empty], ids=["fetch-raises", "fetch-returns-none"])
def test_R1_fix2_reason_url_with_control_char_is_reported_without_it(cfg: Config, fetch):
    from kg.convert import convert_inbox

    line = "https://h.example/a\x07b"
    (cfg.sandbox.root("inbox") / "urls.txt").write_text(line + "\n", encoding="utf-8")

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage(), fetch=fetch)
    assert [o.status for o in outcomes] == ["failed"], [(o.status, o.reason) for o in outcomes]
    reason = outcomes[0].reason or ""
    assert "fetch failed" in reason, reason
    assert "h.example/ab" in reason, "the URL is still shown, minus the control character"
    assert not _RAW_CONTROL.search(reason), repr(reason)
    assert outcomes[0].item.name == line, "the item keeps the line as written; only the report text is cleaned"


def test_R1_fix2_reason_dangling_symlink_with_control_char_in_name_is_escaped(cfg: Config):
    from kg.convert import convert_inbox

    inbox = cfg.sandbox.root("inbox")
    try:
        (inbox / "dang\x07.pdf").symlink_to(cfg.project_root / "nowhere.pdf")
    except OSError:
        pytest.skip("filesystem refuses control characters in file names")

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.item.name, o.status, o.reason) for o in outcomes]
    reason = outcomes[0].reason or ""
    assert "dangling symlink" in reason and "dang" in reason, reason
    assert not _RAW_CONTROL.search(reason), repr(reason)
    assert list(cfg.sandbox.root("converted").iterdir()) == []


# --------------------------------------------------------------- (e) casefold stems


def test_R2_fix2_stems_case_variants_are_distinct_under_casefold_and_keep_case():
    from kg.convert import InputItem, output_stems

    a_md = InputItem(name="A.md", kind="markdown", path=Path("A.md"), url=None)
    a_txt = InputItem(name="a.txt", kind="text", path=Path("a.txt"), url=None)
    stems = output_stems([a_md, a_txt])

    folded = [s.casefold() for s in stems.values()]
    assert len(set(folded)) == 2, stems
    assert stems[a_md].startswith("A"), "emitted stem keeps the source's case"
    assert stems[a_txt].startswith("a"), "emitted stem keeps the source's case"
    assert stems[a_md].endswith("-md") and stems[a_txt].endswith("-txt"), stems


def test_R2_fix2_stems_same_extension_case_variants_get_numeric_suffix():
    # `A.md` vs `a.md` would still fold to the same `-md` stem; the counter must judge case-insensitively too.
    from kg.convert import InputItem, output_stems

    items = [InputItem(name="A.md", kind="markdown", path=Path("A.md"), url=None), InputItem(name="a.md", kind="markdown", path=Path("a.md"), url=None)]
    stems = output_stems(items)
    folded = [s.casefold() for s in stems.values()]
    assert len(set(folded)) == 2, stems


def test_R2_fix2_stems_case_variants_convert_to_distinct_files(cfg: Config):
    from kg.convert import convert_inbox

    inbox = cfg.sandbox.root("inbox")
    make_md(inbox / "A.md")
    (inbox / "a.txt").write_text("plain text body about sample spaces\n", encoding="utf-8")
    if not (inbox / "A.md").exists() or not (inbox / "a.txt").exists():
        pytest.skip("filesystem collapsed the two names")

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["converted", "converted"], [(o.item.name, o.status, o.reason) for o in outcomes]
    all_outputs = [p for o in outcomes for p in o.outputs]
    assert len(all_outputs) == 4
    assert len({p.name.casefold() for p in all_outputs}) == 4, "outputs must not collide on a case-insensitive filesystem"
    assert all(p.is_file() for p in all_outputs)
    assert len({p.read_bytes() for p in all_outputs if p.suffix == ".md"}) == 2, "the two .md files hold different sources"


# ------------------------------------------------------------- (f) URL fragments


class _RecordingFetch:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, url: str) -> str | None:
        self.calls.append(url)
        return HTML_PAGE


def test_R2_fix2_fragment_discover_keeps_line_as_name_and_strips_it_from_url(cfg: Config):
    from kg.convert import discover

    line = WEB_URL + "#section-2"
    (cfg.sandbox.root("inbox") / "urls.txt").write_text(line + "\n", encoding="utf-8")
    items = discover(cfg.sandbox.root("inbox"))
    assert len(items) == 1
    assert items[0].kind == "web"
    assert items[0].name == line, "the report shows the line as written"
    assert items[0].url == WEB_URL, "hash/fetch/locate use the fragment-less URL"


def test_R2_fix2_fragment_url_converts_and_every_locator_resolves(cfg: Config):
    from kg.convert import convert_inbox, resolve_locator, web

    line = WEB_URL + "#section-2"
    (cfg.sandbox.root("inbox") / "urls.txt").write_text(line + "\n", encoding="utf-8")
    fetch = _RecordingFetch()

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage(), fetch=fetch)
    assert [o.status for o in outcomes] == ["converted"], [(o.item.name, o.status, o.reason) for o in outcomes]
    out = outcomes[0]
    assert out.item.name == line and out.item.url == WEB_URL
    assert fetch.calls == [WEB_URL], "the fragment is never sent to the server"
    assert {p.name for p in out.outputs} == {f"{web.stem_for(WEB_URL)}{sfx}" for sfx in (".html", ".md", ".units.json")}
    converted_dir = cfg.sandbox.root("converted")
    assert out.units, "the page has extractable text"
    for u in out.units:
        assert resolve_locator(u.locator, converted_dir) is True, u.locator


def test_R2_fix2_fragment_only_variants_of_one_url_share_a_stem_hash(cfg: Config):
    # `#a` and `#b` are the same page: they must hash to the same web stem (and then be de-collided).
    from kg.convert import InputItem, discover, output_stems, web

    (cfg.sandbox.root("inbox") / "urls.txt").write_text(f"{WEB_URL}#a\n{WEB_URL}#b\n", encoding="utf-8")
    items = discover(cfg.sandbox.root("inbox"))
    assert [i.url for i in items] == [WEB_URL, WEB_URL]
    assert all(isinstance(i, InputItem) for i in items)
    stems = output_stems(items)
    assert len(set(stems.values())) == 2, stems
    assert all(s.startswith(web.stem_for(WEB_URL)) for s in stems.values()), stems


# ------------------------------------------------------------------ (g) zip ratio


def _ratio_of(path: Path) -> int:
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
    return sum(i.file_size for i in infos) // max(1, sum(i.compress_size for i in infos))


def test_R1_fix2_ratio_cap_is_300():
    assert MAX_ZIP_RATIO == 300


def test_R1_fix2_ratio_about_1000_to_1_still_trips(cfg: Config):
    from kg.convert import convert_inbox

    size = 2 * 2**20
    bomb = _zip_with(cfg.sandbox.root("inbox") / "ratio.docx", {"word/document.xml": b"\0" * size})
    measured = _ratio_of(bomb)
    assert measured >= 1000, f"fixture drifted: zeros should deflate at ~1000:1, got {measured}:1"
    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.status, o.reason) for o in outcomes]
    reason = outcomes[0].reason or ""
    assert "ratio" in reason and f"{MAX_ZIP_RATIO}:1" in reason, reason


def test_R1_fix2_ratio_between_100_and_300_is_no_longer_refused_by_the_ratio_cap(cfg: Config):
    # A sparse-sheet-like container at ~220:1 (over the old 100 cap, under the new 300) passes the
    # ratio check; the payload is not a real workbook so it fails later for a different reason.
    from kg.convert import convert_inbox

    rnd = random.Random(1)
    block = 1024
    payload = b"".join(b"\0" * (block - 1) + bytes([rnd.randrange(256)]) for _ in range(2 * 2**20 // block))
    sparse = _zip_with(cfg.sandbox.root("inbox") / "sparse.xlsx", {"xl/workbook.xml": payload})
    measured = _ratio_of(sparse)
    assert 100 < measured < MAX_ZIP_RATIO, f"fixture must sit between the old and new caps, got {measured}:1"
    assert len(payload) > ZIP_RATIO_FLOOR_BYTES, "must be large enough for the ratio to be judged at all"

    outcomes = convert_inbox(cfg, call_stage=FakeCallStage())
    assert [o.status for o in outcomes] == ["failed"], [(o.status, o.reason) for o in outcomes]
    reason = outcomes[0].reason or ""
    assert "ratio" not in reason and "cap" not in reason, f"the ratio cap must not be what refused it: {reason}"
