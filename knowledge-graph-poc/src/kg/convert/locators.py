"""Resolve a locator back to the real page / slide / sheet / heading (R3 acceptance).

``resolve_locator(locator, base_dir)`` returns ``True`` when the locator points
at something that exists in the original under `base_dir`; file locators are
resolved against ``base_dir/<file>``, URL locators against the archived raw HTML
``base_dir/<host>-<slug>.html`` written by the web converter. It returns
``False`` — and never raises — for a missing file, an out-of-range page/slide,
an unknown sheet/heading, a range outside the used area, or a malformed locator.
"""

from __future__ import annotations

import re
from pathlib import Path

from openpyxl.utils.cell import range_boundaries

from kg.convert import docx, pdf, pptx, text_md, web, xlsx

_RANGE = re.compile(r"^(\d+)-(\d+)$")
_INT = re.compile(r"^\d+$")
_XLSX_FRAGMENT = re.compile(r"^(?:'((?:[^']|'')+)'|([^'!]+))!([A-Za-z]{1,3}\d+(?::[A-Za-z]{1,3}\d+)?)$")


def resolve_locator(locator: str, base_dir: Path) -> bool:
    try:
        return _resolve(locator, Path(base_dir))
    except Exception:  # contract: a broken locator or unreadable file is "unresolved", never an exception
        return False


def _resolve(locator: str, base_dir: Path) -> bool:
    if not isinstance(locator, str) or not locator.strip():
        return False
    if locator.startswith(("http://", "https://")):
        return _resolve_url(locator, base_dir)

    file_part, _, fragment = locator.partition("#")
    if not file_part or Path(file_part).name != file_part:  # a bare file name, no directories
        return False
    path = base_dir / file_part
    if not path.is_file():
        return False
    if not fragment:
        return True  # whole-file locator (images)

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return _resolve_xlsx(path, fragment)

    key, sep, value = fragment.partition("=")
    if not sep:
        return False
    if key == "page" and suffix == ".pdf":
        return _in_range(value, pdf.page_count(path))
    if key == "slide" and suffix == ".pptx":
        return _INT.match(value) is not None and int(value) in pptx.slide_numbers(path)
    if key == "lines" and suffix in text_md.MARKDOWN_SUFFIXES | {".txt"}:
        return _span_within(value, text_md.line_count(path))
    if key == "para" and suffix == ".docx":
        return _span_within(value, docx.paragraph_count(path))
    if key == "heading":
        heading, _, tail = value.partition(";lines=")
        if suffix in text_md.MARKDOWN_SUFFIXES:
            if heading not in text_md.markdown_heading_paths(path):
                return False
            return _span_within(tail, text_md.line_count(path)) if tail else True
        if suffix == ".docx" and not tail:
            return heading in docx.docx_heading_paths(path)
    return False


def _resolve_url(locator: str, base_dir: Path) -> bool:
    url, _, fragment = locator.partition("#")
    archive = base_dir / f"{web.stem_for(url)}.html"
    if not archive.is_file():
        return False
    if not fragment:
        return True
    key, sep, value = fragment.partition("=")
    if not sep or key != "heading":
        return False
    return value in web.html_heading_paths(url, archive.read_text(encoding="utf-8"))


def _resolve_xlsx(path: Path, fragment: str) -> bool:
    match = _XLSX_FRAGMENT.match(fragment)
    if match is None:
        return False
    sheet = (match.group(1) or "").replace("''", "'") if match.group(1) else match.group(2)
    areas = xlsx.used_areas(path)
    if sheet not in areas:
        return False
    used_row, used_col = areas[sheet]
    min_col, min_row, max_col, max_row = range_boundaries(match.group(3).upper())
    if None in (min_col, min_row, max_col, max_row):
        return False
    # The whole range must lie inside the sheet's used area, not just its top-left corner.
    return 1 <= min_row <= max_row <= used_row and 1 <= min_col <= max_col <= used_col


def _in_range(value: str, count: int) -> bool:
    return _INT.match(value) is not None and 1 <= int(value) <= count


def _span_within(value: str, count: int) -> bool:
    match = _RANGE.match(value)
    if match is None:
        return False
    a, b = int(match.group(1)), int(match.group(2))
    return 1 <= a <= b <= count


__all__ = ["resolve_locator"]
