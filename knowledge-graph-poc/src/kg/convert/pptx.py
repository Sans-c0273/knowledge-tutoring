"""``.pptx`` → one Unit per slide via markitdown (DESIGN §4.2, D3; R3 ``file.pptx#slide=N``).

markitdown emits ``<!-- Slide number: N -->`` before every slide; we split on
that marker and keep speaker notes inside the slide's unit. Pictures contribute
alt text only (v1 limitation, TECH_DEBT). A slide with no text at all is kept
as ``kind="empty_slide"`` so slide numbers stay truthful.
"""

from __future__ import annotations

import re
from pathlib import Path

from kg.convert._common import ConversionError, Unit, make_convert, markitdown_text, nfkc, number_units, require_ooxml_part

_SLIDE_MARKER = re.compile(r"^<!-- Slide number: (\d+) -->[ \t]*$", re.M)
_TITLE = re.compile(r"^# (.+?)\s*$", re.M)


def _slides(src: Path) -> list[tuple[int, str]]:
    src = require_ooxml_part(Path(src), "ppt/presentation.xml", "PowerPoint presentation")
    text = markitdown_text(src)
    parts = _SLIDE_MARKER.split(text)
    # parts = [before-first-marker, "1", slide-1-text, "2", slide-2-text, ...]
    if len(parts) < 3:
        raise ConversionError(f"{src.name}: no slides found in markitdown output")
    return [(int(parts[i]), nfkc(parts[i + 1]).strip()) for i in range(1, len(parts) - 1, 2)]


def extract(src: Path) -> list[Unit]:
    src = Path(src)
    name = src.name
    units: list[Unit] = []
    for slide_no, body in _slides(src):
        title = _TITLE.search(body)
        heading = [title.group(1).strip()] if title else []
        kind = "slide" if body else "empty_slide"
        units.append(Unit("", f"{name}#slide={slide_no}", heading, body, kind, name))
    return number_units(units)


convert = make_convert(extract)


def slide_numbers(src: Path) -> set[int]:
    return {n for n, _ in _slides(src)}


__all__ = ["convert", "extract", "slide_numbers"]
