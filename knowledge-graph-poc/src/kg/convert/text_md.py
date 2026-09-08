"""``.txt`` and ``.md`` → Units (DESIGN §4.2 rows 1–2; R3).

* ``.txt``: paragraph groups of at most ~60 lines → ``file.txt#lines=a-b``.
* ``.md``: ATX heading sections, heading path = ancestor stack →
  ``file.md#heading=Intro > Terms``; text before the first heading → ``#lines=a-b``;
  a section longer than ~60 lines is split at blank lines and each part gets
  ``;lines=a-b`` appended. Setext headings are not recognised and fall back to
  line ranges (documented limitation).

Line numbers are 1-based into the source file, so ``#lines`` locators resolve
against the original.
"""

from __future__ import annotations

from pathlib import Path

from kg.convert._common import (
    HEADING_SEP,
    Section,
    Unit,
    group_lines,
    heading_paths,
    make_convert,
    number_units,
    read_text_nfkc,
    split_lines,
    split_sections,
)

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})


def extract(src: Path) -> list[Unit]:
    src = Path(src)
    name = src.name
    lines = split_lines(read_text_nfkc(src))
    if src.suffix.lower() in MARKDOWN_SUFFIXES:
        sections = split_sections(lines)
    else:
        sections = [Section([], None, 1, lines)]

    units: list[Unit] = []
    for sec in sections:
        groups = group_lines(sec.lines, sec.start)
        if not groups:
            continue  # heading with no body, or an all-blank preamble
        if not sec.heading_path:
            for a, b, text in groups:
                units.append(Unit("", f"{name}#lines={a}-{b}", [], text, "text", name))
        elif len(groups) == 1:
            units.append(Unit("", f"{name}#heading={sec.path_text}", list(sec.heading_path), groups[0][2], "section", name))
        else:
            for a, b, text in groups:
                locator = f"{name}#heading={sec.path_text};lines={a}-{b}"
                units.append(Unit("", locator, list(sec.heading_path), text, "section", name))
    return number_units(units)


convert = make_convert(extract)


def line_count(src: Path) -> int:
    return len(split_lines(read_text_nfkc(src)))


def markdown_heading_paths(src: Path) -> set[str]:
    """Every ``A > B`` heading path in a Markdown file (used by locator resolution)."""
    return heading_paths(split_lines(read_text_nfkc(src)))


__all__ = ["HEADING_SEP", "MARKDOWN_SUFFIXES", "convert", "extract", "line_count", "markdown_heading_paths"]
