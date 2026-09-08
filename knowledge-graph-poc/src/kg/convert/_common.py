"""Pieces shared by every converter: the Unit model, errors, NFKC, heading/line
splitting and the two output files (DESIGN §4.1, §4.2, §22; D3, D8).

Every converter ends in ``write_outputs()``: ``converted/<stem>.md`` with a
``<!-- unit: <locator> -->`` comment before each unit, and
``converted/<stem>.units.json`` (``[{uid, locator, heading_path, text, kind,
source}]``). Extraction consumes only the JSON; the Markdown is for people (R2).

Text is NFKC-normalised on read (DESIGN §22) — Thai is preserved; fullwidth
Latin and compatibility forms are folded.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

#: Paragraph groups for line-range units are capped at about this many lines (DESIGN §4.2).
MAX_GROUP_LINES = 60
UNIT_BOUNDARY = "<!-- unit: {locator} -->"
HEADING_SEP = " > "

# Resource caps on untrusted inputs: every parser below runs on whatever lands in the
# inbox, so a zip bomb, a 100k-page PDF or a multi-GB "text" file must fail with a
# reason instead of exhausting memory. Over-cap → ConversionError → status `failed`.
# TODO(config): move to kg.yaml limits.* once WU2 lands
MAX_INFLATED_BYTES = 200 * 2**20  # total uncompressed size of an OOXML container
# uncompressed / compressed; only judged above ZIP_RATIO_FLOOR_BYTES. Sparse xlsx sheets
# (mostly empty <c> cells) legitimately exceed 100:1; MAX_INFLATED_BYTES bounds memory anyway.
MAX_ZIP_RATIO = 300
ZIP_RATIO_FLOOR_BYTES = 1 * 2**20  # tiny, highly repetitive XML compresses legitimately well
MAX_PDF_PAGES = 500
MAX_TEXT_BYTES = 20 * 2**20


class ConversionError(Exception):
    """One input could not be converted. The message is the reason shown for the item (R1)."""


@dataclass
class Unit:
    """One addressable piece of a source: a page, slide, sheet block, heading section, ... (D8)."""

    uid: str
    locator: str
    heading_path: list[str]
    text: str
    kind: str
    source: str

    def to_row(self) -> dict[str, object]:
        return {
            "uid": self.uid,
            "locator": self.locator,
            "heading_path": list(self.heading_path),
            "text": self.text,
            "kind": self.kind,
            "source": self.source,
        }


def nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def number_units(units: list[Unit]) -> list[Unit]:
    """Assign ``U1, U2, ...`` in document order (ids are unique within one source)."""
    for i, unit in enumerate(units, start=1):
        unit.uid = f"U{i}"
    return units


# ------------------------------------------------------------------- reading


def require_file(src: Path) -> Path:
    src = Path(src)
    if not src.is_file():
        raise ConversionError(f"{src.name}: file not found")
    return src


def read_text_nfkc(src: Path) -> str:
    """UTF-8 (BOM tolerated) text of `src`, NFKC-normalised. Non-UTF-8 input is a failure, not garbage."""
    src = require_file(src)
    try:
        size = src.stat().st_size
        if size > MAX_TEXT_BYTES:
            raise ConversionError(f"{src.name}: text file is {size} bytes, over the {MAX_TEXT_BYTES}-byte cap")
        raw = src.read_bytes()
    except OSError as exc:
        raise ConversionError(f"{src.name}: cannot read: {exc.strerror or exc.__class__.__name__}") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConversionError(f"{src.name}: not UTF-8 text (byte {exc.start}); re-save as UTF-8") from exc
    return nfkc(text)


def split_lines(text: str) -> list[str]:
    """Lines numbered as an editor would: split on \\n only, CR stripped."""
    return [line.rstrip("\r") for line in text.split("\n")]


def require_ooxml_part(src: Path, part: str, what: str) -> Path:
    """Refuse files that are not the OOXML container they claim to be.

    markitdown falls back to plain-text extraction for unreadable input instead of
    raising, so a corrupt .pptx/.docx would otherwise 'convert' to its own bytes.
    """
    src = require_file(src)
    try:
        with zipfile.ZipFile(src) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise ConversionError(f"{src.name}: not a valid {what} (not an OOXML zip: {exc})") from exc
    if part not in {info.filename for info in infos}:
        raise ConversionError(f"{src.name}: not a valid {what} ({part} missing)")
    inflated = sum(info.file_size for info in infos)
    compressed = sum(info.compress_size for info in infos)
    if inflated > MAX_INFLATED_BYTES:
        raise ConversionError(f"{src.name}: {what} inflates to {inflated} bytes, over the {MAX_INFLATED_BYTES}-byte cap")
    if inflated > ZIP_RATIO_FLOOR_BYTES and inflated > MAX_ZIP_RATIO * max(compressed, 1):
        raise ConversionError(f"{src.name}: {what} compression ratio {inflated // max(compressed, 1)}:1 exceeds {MAX_ZIP_RATIO}:1")
    return src


_MARKITDOWN = None


def markitdown_text(src: Path) -> str:
    """Markdown from markitdown 0.1.7 for PPTX/DOCX (D3). Imported lazily: it loads an ONNX model."""
    global _MARKITDOWN
    from markitdown import MarkItDown  # heavy import (magika/onnxruntime)

    if _MARKITDOWN is None:
        _MARKITDOWN = MarkItDown(enable_plugins=False)
    try:
        return _MARKITDOWN.convert(str(src)).text_content or ""
    except Exception as exc:  # markitdown wraps many library errors; report, never swallow
        raise ConversionError(f"{src.name}: markitdown could not convert: {exc}") from exc


# ------------------------------------------------------ heading / line splitting

_ATX = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_CLOSING_HASHES = re.compile(r"\s+#+$")
_FENCE = re.compile(r"^(`{3,}|~{3,})")


@dataclass
class Section:
    """A run of lines under one ATX heading (or the preamble when `heading_path` is empty)."""

    heading_path: list[str]
    heading_line: int | None  # 1-based line of the heading; None for the preamble
    start: int  # 1-based line number of lines[0]
    lines: list[str] = field(default_factory=list)

    @property
    def path_text(self) -> str:
        return HEADING_SEP.join(self.heading_path)


def split_sections(lines: list[str]) -> list[Section]:
    """Split Markdown lines at ATX headings; `heading_path` is the ancestor stack.

    Setext headings are not recognised (DESIGN §4.2 known limitation). Headings
    inside fenced code blocks are ignored; a fence closes only on the same fence
    character at least as long as the opener (CommonMark), so ``~~~`` never
    closes a ``````` block.
    """
    sections = [Section([], None, 1)]
    stack: list[tuple[int, str]] = []
    open_fence: str | None = None
    for lineno, line in enumerate(lines, start=1):
        fence = _FENCE.match(line)
        if fence is not None:
            marker = fence.group(1)
            if open_fence is None:
                open_fence = marker
            elif marker[0] == open_fence[0] and len(marker) >= len(open_fence):
                open_fence = None
        match = None if open_fence is not None else _ATX.match(line)
        if match is None:
            sections[-1].lines.append(line)
            continue
        level = len(match.group(1))
        title = _CLOSING_HASHES.sub("", match.group(2)).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        sections.append(Section([t for _, t in stack], lineno, lineno + 1))
    return sections


def heading_paths(lines: list[str]) -> set[str]:
    return {sec.path_text for sec in split_sections(lines) if sec.heading_path}


def paragraphs(lines: list[str]) -> list[tuple[int, int]]:
    """Maximal runs of non-blank lines as 0-based inclusive index pairs."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, line in enumerate(lines):
        if line.strip():
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(lines) - 1))
    return runs


def group_lines(lines: list[str], first_line_no: int, max_lines: int = MAX_GROUP_LINES) -> list[tuple[int, int, str]]:
    """Greedy paragraph groups spanning at most `max_lines` source lines.

    Returns ``(a, b, text)`` with 1-based inclusive line numbers into the source
    (`first_line_no` is the number of ``lines[0]``). Groups split only at blank
    lines, so a single paragraph longer than `max_lines` stays whole.
    """
    groups: list[list[int]] = []
    for ps, pe in paragraphs(lines):
        if groups and pe - groups[-1][0] + 1 <= max_lines:
            groups[-1][1] = pe
        else:
            groups.append([ps, pe])
    return [(first_line_no + a, first_line_no + b, "\n".join(lines[a : b + 1])) for a, b in groups]


# ------------------------------------------------------------------- writing


def render_unit_markdown(unit: Unit) -> str:
    head = UNIT_BOUNDARY.format(locator=unit.locator)
    if unit.kind == "section" and unit.heading_path:
        head += f"\n{'#' * len(unit.heading_path)} {unit.heading_path[-1]}\n"
    body = unit.text.rstrip()
    return f"{head}\n{body}\n" if body else f"{head}\n"


#: Staging suffix for atomic writes; the dispatcher sandbox-resolves ``<target>TMP_SUFFIX`` too.
TMP_SUFFIX = ".tmp"
# Create-only, never follow: a pre-planted ``<name>.tmp`` (plain file or symlink) makes
# the write fail instead of being written through and renamed into place.
_TMP_OPEN_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)


def _tmp_path(path: Path) -> Path:
    return path.with_name(path.name + TMP_SUFFIX)


def write_files_atomically(files: dict[Path, str]) -> None:
    """Write every ``path: text`` in `files` as one step: all to ``<name>.tmp`` first, then ``os.replace`` each.

    Temporaries are created exclusively (``O_EXCL | O_NOFOLLOW``, mode 0600), so an
    existing ``.tmp`` — in particular a planted symlink — is an ``OSError``, never
    followed. A failure while staging removes the temporaries this call created and
    leaves every target untouched, so the converted/ folder never holds a
    half-written document.
    """
    staged: list[Path] = []
    try:
        for path, text in files.items():
            tmp = _tmp_path(path)
            fd = os.open(tmp, _TMP_OPEN_FLAGS, 0o600)
            staged.append(tmp)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
        for path in files:
            os.replace(_tmp_path(path), path)
    except OSError:
        for tmp in staged:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass  # best-effort cleanup; the original error is what the caller reports
        raise


def write_outputs(out_dir: Path, stem: str, units: list[Unit], *, extra_files: dict[str, str] | None = None) -> tuple[Path, ...]:
    """Write ``<stem>.md`` and ``<stem>.units.json`` under `out_dir` (R2, DESIGN §4.1).

    `extra_files` maps a suffix (``".html"``) to text written as ``<stem><suffix>``
    in the same step (the web converter archives its raw HTML this way). All
    documents are rendered before anything is written and land atomically
    (``.tmp`` + ``os.replace``), so a failure leaves nothing partial behind. The
    caller has already sandbox-resolved `out_dir`.
    """
    if not units:
        raise ConversionError(f"{stem}: no extractable text")
    markdown = "\n".join(render_unit_markdown(u) for u in units)
    rows = json.dumps([u.to_row() for u in units], ensure_ascii=False, indent=1)
    out_dir = Path(out_dir)
    files: dict[Path, str] = {out_dir / f"{stem}.md": markdown, out_dir / f"{stem}.units.json": rows + "\n"}
    for suffix, text in (extra_files or {}).items():
        files[out_dir / f"{stem}{suffix}"] = text
    try:
        write_files_atomically(files)
    except OSError as exc:
        raise ConversionError(f"{stem}: cannot write converted output: {exc.strerror or exc.__class__.__name__}") from exc
    return tuple(files)


def make_convert(extract: Callable[[Path], list[Unit]]) -> Callable[..., list[Unit]]:
    """The ``convert(src, out_dir, *, stem=None)`` every file converter exposes: extract, then write_outputs."""

    def convert(src: Path, out_dir: Path, *, stem: str | None = None) -> list[Unit]:
        src = Path(src)
        units = extract(src)
        write_outputs(Path(out_dir), stem or src.stem, units)
        return units

    convert.__doc__ = f"Convert `src` with ``{extract.__module__}.extract`` and write ``<stem>.md`` + ``<stem>.units.json`` under `out_dir`."
    return convert
