"""URLs from ``inbox/urls.txt`` → Units via trafilatura (DESIGN §4.2, D3; R3 ``<url>#heading=A > B``).

``convert_html()`` is the offline seam (tests feed saved HTML); ``convert()``
takes an injected ``fetch(url) -> str | None`` so nothing here opens a socket
unless the caller asks for the default fetcher (``make_fetch``).

The raw HTML is archived to ``converted/<host>-<slug>-<hash8>.html`` next to the
Markdown, so locators can be resolved later without re-fetching. Pages whose
text is rendered by JavaScript have no extractable text and fail with a reason
(documented limitation); no authentication is attempted.
"""

from __future__ import annotations

import configparser
import hashlib
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import trafilatura
from trafilatura.settings import DEFAULT_CONFIG

from kg.convert._common import ConversionError, Unit, heading_paths, nfkc, number_units, split_lines, split_sections, write_outputs

Fetch = Callable[[str], "str | None"]

_EXTRACT_KW = dict(output_format="markdown", include_tables=True, include_links=False, include_images=False)
_SLUG_JUNK = re.compile(r"[^a-z0-9]+")
_HOST_JUNK = re.compile(r"[^a-z0-9.-]+")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
MAX_SLUG_LEN = 80
MAX_HOST_LEN = 64


def _shown(url: str) -> str:
    """`url` as it may appear in a reason: control characters stripped (they end up in the report)."""
    return _CONTROL_CHARS.sub("", url)


def stem_for(url: str) -> str:
    """``<host>-<slug>-<hash8>`` for every artefact of one page.

    The slug keeps the stem readable; the ``sha1(url)[:8]`` tail is what makes it
    unique — ``/Page`` vs ``/page/``, ``http`` vs ``https`` or two non-ASCII paths
    all slug identically and would otherwise overwrite each other. Deterministic.
    """
    parts = urlsplit(url)
    host = _HOST_JUNK.sub("-", (parts.hostname or "").lower()).strip("-")[:MAX_HOST_LEN].strip("-") or "unknown-host"
    slug = _SLUG_JUNK.sub("-", parts.path.lower()).strip("-")[:MAX_SLUG_LEN].strip("-") or "index"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{host}-{slug}-{digest}"


def extract_markdown(url: str, html: str) -> str:
    text = trafilatura.extract(html, url=url, **_EXTRACT_KW)
    if not text or not text.strip():
        raise ConversionError(f"{_shown(url)}: no extractable text (JS-rendered or empty page)")
    return nfkc(text)


def units_from_markdown(url: str, markdown: str) -> list[Unit]:
    units: list[Unit] = []
    for sec in split_sections(split_lines(markdown)):
        body = "\n".join(sec.lines).strip()
        if not body:
            continue
        if sec.heading_path:
            units.append(Unit("", f"{url}#heading={sec.path_text}", list(sec.heading_path), body, "section", url))
        else:
            units.append(Unit("", url, [], body, "text", url))
    return number_units(units)


def convert_html(url: str, html: str, out_dir: Path, *, stem: str | None = None) -> list[Unit]:
    units = units_from_markdown(url, extract_markdown(url, html))
    if not units:
        raise ConversionError(f"{_shown(url)}: no extractable text")
    # The raw-HTML archive is written in the same atomic step as .md/.units.json.
    write_outputs(Path(out_dir), stem or stem_for(url), units, extra_files={".html": html})
    return units


def convert(url: str, out_dir: Path, *, fetch: Fetch, stem: str | None = None) -> list[Unit]:
    try:
        html = fetch(url)
    except Exception as exc:  # network layer errors become a reported failure for this URL
        detail = _shown(str(exc).splitlines()[0]) if str(exc) else type(exc).__name__
        raise ConversionError(f"{_shown(url)}: fetch failed: {detail}") from exc
    if not html:
        raise ConversionError(f"{_shown(url)}: fetch failed (no content returned)")
    return convert_html(url, html, out_dir, stem=stem)


def make_fetch(*, timeout_s: int, user_agent: str) -> Fetch:
    """trafilatura.fetch_url bound to kg.yaml's ``web`` settings (the only network path in this module)."""
    config = configparser.ConfigParser()
    config.read_dict({"DEFAULT": dict(DEFAULT_CONFIG["DEFAULT"])})
    config.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(timeout_s))
    config.set("DEFAULT", "USER_AGENTS", user_agent)

    def fetch(url: str) -> str | None:
        return trafilatura.fetch_url(url, config=config)

    return fetch


def html_heading_paths(url: str, html: str) -> set[str]:
    return heading_paths(split_lines(extract_markdown(url, html)))


__all__ = ["Fetch", "convert", "convert_html", "html_heading_paths", "make_fetch", "stem_for", "units_from_markdown"]
