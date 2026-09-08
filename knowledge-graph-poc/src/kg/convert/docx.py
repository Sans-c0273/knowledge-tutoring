"""``.docx`` → Units via markitdown (mammoth → HTML → Markdown) (DESIGN §4.2, D3).

Word heading styles become ATX headings; units follow the heading path
(``file.docx#heading=1 Scope > 1.2 Terms``). A document with no headings falls
back to blocks of ~15 paragraphs (``file.docx#para=31-45``); text before the
first heading, when headings exist, is likewise ``#para=1-k``. Needs real Word
heading styles (documented limitation); images contribute alt text only.
"""

from __future__ import annotations

from pathlib import Path

from kg.convert._common import (
    Unit,
    heading_paths,
    make_convert,
    markitdown_text,
    nfkc,
    number_units,
    paragraphs,
    require_ooxml_part,
    split_lines,
    split_sections,
)

PARAGRAPHS_PER_BLOCK = 15


def _markdown_lines(src: Path) -> list[str]:
    src = require_ooxml_part(Path(src), "word/document.xml", "Word document")
    return split_lines(nfkc(markitdown_text(src)))


def _paragraph_blocks(name: str, lines: list[str], first_index: int = 1) -> list[Unit]:
    """~15-paragraph blocks; paragraph indices are 1-based and contiguous."""
    runs = paragraphs(lines)
    units: list[Unit] = []
    for start in range(0, len(runs), PARAGRAPHS_PER_BLOCK):
        block = runs[start : start + PARAGRAPHS_PER_BLOCK]
        a = first_index + start
        b = a + len(block) - 1
        text = "\n\n".join("\n".join(lines[ps : pe + 1]) for ps, pe in block)
        units.append(Unit("", f"{name}#para={a}-{b}", [], text, "para_block", name))
    return units


def extract(src: Path) -> list[Unit]:
    src = Path(src)
    name = src.name
    lines = _markdown_lines(src)
    sections = split_sections(lines)
    if not any(sec.heading_path for sec in sections):
        return number_units(_paragraph_blocks(name, lines))

    units: list[Unit] = []
    for sec in sections:
        if not sec.heading_path:
            units.extend(_paragraph_blocks(name, sec.lines))
            continue
        body = "\n".join(sec.lines).strip()
        if not body:
            continue  # heading with no text under it
        units.append(Unit("", f"{name}#heading={sec.path_text}", list(sec.heading_path), body, "section", name))
    return number_units(units)


convert = make_convert(extract)


def docx_heading_paths(src: Path) -> set[str]:
    return heading_paths(_markdown_lines(src))


def paragraph_count(src: Path) -> int:
    return len(paragraphs(_markdown_lines(src)))


__all__ = ["PARAGRAPHS_PER_BLOCK", "convert", "docx_heading_paths", "extract", "paragraph_count"]
