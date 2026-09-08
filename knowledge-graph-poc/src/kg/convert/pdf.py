"""``.pdf`` → one Unit per page via pdfplumber (DESIGN §4.2, D3; R3 ``file.pdf#page=N``).

Pages without extractable text are kept as ``kind="empty_page"`` units with empty
text so page numbering stays truthful; scanned PDFs therefore yield nothing
usable by design (no OCR — PRD §4). Multi-column reading order is whatever
pdfplumber produces.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from kg.convert._common import MAX_PDF_PAGES, ConversionError, Unit, make_convert, nfkc, number_units, require_file


def extract(src: Path) -> list[Unit]:
    src = require_file(Path(src))
    name = src.name
    units: list[Unit] = []
    try:
        with pdfplumber.open(str(src)) as doc:
            if len(doc.pages) > MAX_PDF_PAGES:
                raise ConversionError(f"{name}: PDF has {len(doc.pages)} pages, over the {MAX_PDF_PAGES}-page cap")
            for page_no, page in enumerate(doc.pages, start=1):
                text = nfkc((page.extract_text() or "").strip())
                kind = "page" if text else "empty_page"
                units.append(Unit("", f"{name}#page={page_no}", [], text, kind, name))
    except ConversionError:
        raise
    except Exception as exc:  # pdfminer raises many unrelated types for damaged files
        raise ConversionError(f"{name}: cannot read PDF: {exc}") from exc
    if not units:
        raise ConversionError(f"{name}: PDF has no pages")
    return number_units(units)


convert = make_convert(extract)


def page_count(src: Path) -> int:
    with pdfplumber.open(str(require_file(Path(src)))) as doc:
        return len(doc.pages)


__all__ = ["convert", "extract", "page_count"]
