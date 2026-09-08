"""Conversion stage: S0 discover + S1 convert (R1, R2, R3, R5, R6; DESIGN §4, §6).

``discover()`` lists every file in the inbox plus every URL in ``inbox/urls.txt``
as an ``InputItem`` with a kind; unsupported files are items too, never skipped.
``convert_inbox()`` / ``convert_item()`` produce exactly one ``ConversionOutcome``
per item — ``converted | unsupported | failed | deferred`` — and write
``converted/<stem>.md`` + ``<stem>.units.json`` through the sandbox. Conversion
moves and deletes nothing; the move to ``processed/`` is S8 (pipeline).

Only the image converter calls the model, through the injected ``call_stage``
(DESIGN §7.1); a failure at or after the model boundary marks that file
``deferred`` (DESIGN §6), a failure before it (unreadable image) ``failed``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

from kg.config import Config
from kg.convert import docx, image, pdf, pptx, text_md, web, xlsx
from kg.convert._common import HEADING_SEP, MAX_GROUP_LINES, TMP_SUFFIX, UNIT_BOUNDARY, ConversionError, Unit, nfkc
from kg.convert.image import CallStage, ImageInput
from kg.convert.locators import resolve_locator
from kg.paths import SandboxViolation

URLS_FILE = "urls.txt"
URL_SCHEMES: tuple[str, ...] = ("http://", "https://")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
#: Reasons built from an arbitrary exception are cut here: SDK errors can carry whole response bodies.
MAX_EXCEPTION_REASON_LEN = 200

KIND_BY_EXTENSION: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".pdf": "pdf",
    ".pptx": "pptx",
    ".docx": "docx",
    ".xlsx": "xlsx",
    **{ext: "image" for ext in image.SUPPORTED_IMAGE_SUFFIXES},
}
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(KIND_BY_EXTENSION)
KINDS: tuple[str, ...] = ("text", "markdown", "pdf", "pptx", "docx", "xlsx", "image", "web", "unsupported")
STATUSES: tuple[str, ...] = ("converted", "unsupported", "failed", "deferred")


@dataclass(frozen=True)
class InputItem:
    """One thing found by S0: a file in the inbox (`path`) or a URL from urls.txt (`url`)."""

    name: str
    kind: str
    path: Path | None
    url: str | None
    #: Set by S0 when the item is already known to be unusable (dangling symlink,
    #: unreadable urls.txt, ...); S1 reports it as ``failed`` with this reason.
    error: str | None = None


@dataclass
class ConversionOutcome:
    item: InputItem
    status: str
    reason: str | None
    units: list[Unit] = field(default_factory=list)
    #: Files written under converted/ for this item (empty unless status == "converted").
    outputs: tuple[Path, ...] = ()


# ------------------------------------------------------------------ S0 discover


def kind_for(name: str) -> str:
    return KIND_BY_EXTENSION.get(Path(name).suffix.lower(), "unsupported")


def read_urls(urls_file: Path) -> list[str]:
    """Non-blank, non-comment lines of urls.txt, de-duplicated in order.

    Raises ``UnicodeDecodeError`` / ``OSError`` for an unreadable file; ``discover()``
    turns those into one ``failed`` item so the run continues.
    """
    seen: list[str] = []
    for raw in urls_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        seen.append(line)
    return seen


def _is_inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
    except ValueError:
        return False
    return True


def _url_item(line: str) -> InputItem:
    if not line.lower().startswith(URL_SCHEMES):
        return InputItem(name=line, kind="unsupported", path=None, url=line)
    # The fragment is never sent to the server and resolve_locator() recomputes the
    # stem from the fragment-less URL, so hash/fetch/locate on that; the report keeps
    # the line as written (item.name).
    url, _fragment = urldefrag(line)
    return InputItem(name=line, kind="web", path=None, url=url)


def _urls_items(inbox_dir: Path, urls_file: Path) -> list[InputItem]:
    """Items for urls.txt: one per URL, or one ``failed`` item when the list itself cannot be used."""

    def refused(reason: str) -> list[InputItem]:
        return [InputItem(name=URLS_FILE, kind="web", path=urls_file, url=None, error=reason)]

    if urls_file.is_symlink() and not _is_inside(urls_file, inbox_dir):
        return refused(f"sandbox refused: {URLS_FILE} is a symlink resolving outside the inbox")
    try:
        lines = read_urls(urls_file)
    except UnicodeDecodeError as exc:
        return refused(f"{URLS_FILE}: not UTF-8 text (byte {exc.start}); re-save as UTF-8")
    except OSError as exc:
        return refused(f"{URLS_FILE}: cannot read: {exc.strerror or exc.__class__.__name__}")
    return [_url_item(line) for line in lines]


def discover(inbox_dir: Path) -> list[InputItem]:
    """Files (sorted by name) then URLs (in urls.txt order).

    Dotfiles (``.DS_Store``) and real sub-directories are not inputs and are skipped
    silently; a dangling symlink or a symlink to a directory is reported as a
    ``failed`` item, never skipped (R1). Symlinks to files are items and are checked
    against the inbox root at conversion time.
    """
    inbox_dir = Path(inbox_dir)
    if not inbox_dir.is_dir():
        return []
    items: list[InputItem] = []
    for path in sorted(inbox_dir.iterdir(), key=lambda p: p.name):
        if path.name == URLS_FILE or path.name.startswith("."):
            continue
        error: str | None = None
        if path.is_symlink():  # names go into the report: control characters are escaped via !r
            if not path.exists():
                error = f"{path.name!r}: dangling symlink"
            elif path.is_dir():
                error = f"{path.name!r}: symlink to a directory; only files are inputs"
        elif not path.is_file():
            continue  # a real sub-directory (or a socket/fifo) is not an input
        items.append(InputItem(name=path.name, kind=kind_for(path.name), path=path, url=None, error=error))
    urls_file = inbox_dir / URLS_FILE
    if urls_file.is_symlink() or urls_file.is_file():
        items.extend(_urls_items(inbox_dir, urls_file))
    return items


# ------------------------------------------------------------------- S1 convert


def _plain_stem(item: InputItem) -> str:
    return web.stem_for(item.url) if item.kind == "web" and item.url else Path(item.name).stem


def output_stems(items: list[InputItem]) -> dict[InputItem, str]:
    """``converted/<stem>`` per item, unique within the run so nothing is overwritten.

    Files sharing a stem get ``-<ext>`` (``deck-pdf``, ``deck-pptx``); a web stem
    that collides with anything gets ``-web``; anything still colliding after that
    gets ``-2``, ``-3``, ... in discovery order. Deterministic for a given inbox.
    """
    # Collisions are judged case-insensitively: on APFS/NTFS `A.md` and `a.txt` would
    # otherwise write the same converted/a.* files. The emitted stem keeps its case.
    plain = {item: _plain_stem(item) for item in items}
    counts: dict[str, int] = {}
    for stem in plain.values():
        counts[stem.casefold()] = counts.get(stem.casefold(), 0) + 1
    stems: dict[InputItem, str] = {}
    for item, stem in plain.items():
        if counts[stem.casefold()] > 1:
            tag = "web" if item.kind == "web" else Path(item.name).suffix.lower().lstrip(".")
            stem = f"{stem}-{tag}" if tag else stem
        stems[item] = stem
    taken: set[str] = set()
    for item, stem in stems.items():
        unique, n = stem, 1
        while unique.casefold() in taken:
            n += 1
            unique = f"{stem}-{n}"
        stems[item] = unique
        taken.add(unique.casefold())
    return stems


def _require_inside_inbox(cfg: Config, path: Path) -> None:
    inbox_real = os.path.realpath(cfg.sandbox.root("inbox"))
    try:
        rel = Path(os.path.realpath(path)).relative_to(inbox_real)
    except ValueError:
        raise SandboxViolation(f"inbox: input lies outside the inbox root ({path.name})") from None
    cfg.sandbox.resolve("inbox", rel)


def _unsupported(item: InputItem) -> ConversionOutcome:
    if item.url is not None:
        reason = f"unsupported URL scheme in {URLS_FILE}: {item.url!r}; only {' and '.join(URL_SCHEMES)} URLs are fetched"
    else:
        ext = Path(item.name).suffix.lower() or "(no extension)"
        reason = (
            f"unsupported file type {ext!r}; left in inbox. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}, URLs in {URLS_FILE}"
        )
    return ConversionOutcome(item, "unsupported", reason, [])


class _BoundaryProbe:
    """Wraps ``call_stage`` so the dispatcher knows whether the model boundary was reached.

    Anything raised once the model has been called — including a stage result of the
    wrong type — is a model-boundary failure and defers the file (DESIGN §6); anything
    raised before (unreadable or oversized image) is a plain conversion failure.
    """

    def __init__(self, call_stage: CallStage) -> None:
        self.call_stage = call_stage
        self.reached = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.reached = True
        return self.call_stage(*args, **kwargs)


def _exception_reason(exc: BaseException) -> str:
    """``<Type>: <first line of the message>``, control characters stripped, capped (never a raw SDK body)."""
    text = str(exc)
    first = _CONTROL_CHARS.sub("", text.splitlines()[0]).strip() if text.strip() else ""
    reason = f"{type(exc).__name__}: {first}" if first else type(exc).__name__
    if len(reason) > MAX_EXCEPTION_REASON_LEN:
        reason = reason[: MAX_EXCEPTION_REASON_LEN - 3] + "..."
    return reason


def convert_item(
    item: InputItem,
    cfg: Config,
    *,
    call_stage: CallStage,
    fetch: web.Fetch | None = None,
    out_stem: str | None = None,
) -> ConversionOutcome:
    """Convert one item. Never raises for a bad input: the failure becomes the outcome (R1)."""
    if item.error:
        return ConversionOutcome(item, "failed", item.error, [])
    if item.kind == "unsupported" or item.kind not in KINDS:
        return _unsupported(item)
    probe = _BoundaryProbe(call_stage)
    try:
        out_dir = cfg.sandbox.root("converted")
        if item.kind == "web":
            if not item.url:
                raise ConversionError(f"{item.name!r}: web item without a URL")
            stem = out_stem or web.stem_for(item.url)
            targets = (f"{stem}.html", f"{stem}.md", f"{stem}.units.json")
        else:
            if item.path is None:
                raise ConversionError(f"{item.name!r}: file item without a path")
            if _CONTROL_CHARS.search(item.name):
                raise SandboxViolation(f"inbox: file name contains control characters ({item.name!r})")
            cfg.sandbox.resolve("converted", item.name)  # the name itself must not escape or touch the denylist
            _require_inside_inbox(cfg, item.path)
            stem = out_stem or Path(item.name).stem
            targets = (f"{stem}.md", f"{stem}.units.json")
        outputs = tuple(cfg.sandbox.resolve("converted", t) for t in targets)
        for t in targets:  # the staging files are written too: a planted `<t>.tmp` symlink must not lead outside
            cfg.sandbox.resolve("converted", t + TMP_SUFFIX)
        out_dir.mkdir(parents=True, exist_ok=True)

        if item.kind == "web":
            units = web.convert(
                item.url,
                out_dir,
                fetch=fetch or web.make_fetch(timeout_s=cfg.web.timeout_s, user_agent=cfg.web.user_agent),
                stem=stem,
            )
        elif item.kind == "image":
            units = image.convert(item.path, out_dir, max_long_edge_px=cfg.image.max_long_edge_px, call_stage=probe, stem=stem)
        else:
            units = _FILE_CONVERTERS[item.kind](item.path, out_dir, stem=stem)
        return ConversionOutcome(item, "converted", None, units, outputs)
    except SandboxViolation as exc:
        return ConversionOutcome(item, "failed", f"sandbox refused: {exc}", [])
    except ConversionError as exc:
        if probe.reached:  # e.g. the describe stage returned something other than a DescribeOutput
            return ConversionOutcome(item, "deferred", f"model boundary: {exc}", [])
        return ConversionOutcome(item, "failed", str(exc), [])
    except Exception as exc:  # one bad input must not abort the run (R1); the reason is a capped summary, not the raw message
        status = "deferred" if probe.reached else "failed"  # model-boundary failures defer the file (DESIGN §6)
        return ConversionOutcome(item, status, _exception_reason(exc), [])


_FILE_CONVERTERS: dict[str, Callable[..., list[Unit]]] = {
    "text": text_md.convert,
    "markdown": text_md.convert,
    "pdf": pdf.convert,
    "pptx": pptx.convert,
    "docx": docx.convert,
    "xlsx": xlsx.convert,
}


def convert_inbox(cfg: Config, *, call_stage: CallStage, fetch: web.Fetch | None = None) -> list[ConversionOutcome]:
    """S0 + S1 over the whole inbox: one outcome per input, in discovery order."""
    items = discover(cfg.sandbox.root("inbox"))
    stems = output_stems(items)
    return [convert_item(item, cfg, call_stage=call_stage, fetch=fetch, out_stem=stems[item]) for item in items]


__all__ = [
    "HEADING_SEP",
    "KINDS",
    "KIND_BY_EXTENSION",
    "MAX_GROUP_LINES",
    "STATUSES",
    "SUPPORTED_EXTENSIONS",
    "UNIT_BOUNDARY",
    "URLS_FILE",
    "URL_SCHEMES",
    "CallStage",
    "ConversionError",
    "ConversionOutcome",
    "ImageInput",
    "InputItem",
    "Unit",
    "convert_inbox",
    "convert_item",
    "discover",
    "docx",
    "image",
    "kind_for",
    "nfkc",
    "output_stems",
    "pdf",
    "pptx",
    "read_urls",
    "resolve_locator",
    "text_md",
    "web",
    "xlsx",
]
