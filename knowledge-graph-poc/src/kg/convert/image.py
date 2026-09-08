"""Raster images → one Unit holding the vision-model description (R6; DESIGN §4.2, §4.3).

Pillow opens the file (an unreadable image fails here, before any tokens are
spent), downscales so the long edge is at most ``image.max_long_edge_px``
(never upscales, keeps aspect ratio), re-encodes JPEG as JPEG and everything
else as PNG (GIF: first frame), and base64-encodes it. The encoded payload is
capped at ``MAX_IMAGE_BYTES`` (4.5 MiB, under Anthropic's 5 MB per-image limit):
an oversize image is first re-encoded as RGB JPEG q85 at full size (alpha
composited on white, never black), and if still too large the long edge is
halved and re-encoded until it fits; ``media_type`` always names the
encoding actually sent. The injected
``call_stage`` (DESIGN §7.1 surface) is called with stage ``describe`` and the
``DescribeOutput`` schema; the six returned fields are rendered to Markdown
(title as ``#``, description, bullet lists) which becomes the unit text.
Locator is the whole file (``diagram.png``).

Failures from the model boundary propagate unchanged; nothing is written for a
failed description (D20, §6).
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from kg.convert._common import ConversionError, Unit, nfkc, require_file, write_outputs
from kg.llm.adapters.base import ImageInput  # re-exported; defined by the LLM boundary (DESIGN §20)
from kg.prompts import describe as describe_prompt
from kg.schemas import DescribeOutput

SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
#: Hard cap on the encoded payload handed to a provider (Anthropic: 5 MB per image; keep headroom).
MAX_IMAGE_BYTES = int(4.5 * 1024 * 1024)
#: Halvings of the long edge tried before giving up (1568 → 3 px in ten steps; never reached in practice).
MAX_SHRINK_STEPS = 10
FALLBACK_JPEG_QUALITY = 85
_FILENAME_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")

#: DESIGN §7.1: call_stage(stage, system, user_text, schema, images, **kw) -> StageResult with `.data`.
CallStage = Callable[..., Any]


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def _flatten_on_white(img: Image.Image) -> Image.Image:
    """RGB copy with any alpha composited on white (a bare ``convert("RGB")`` would turn transparency black)."""
    if not _has_alpha(img):
        return img.convert("RGB")
    rgba = img.convert("RGBA")
    bg = Image.new("RGB", rgba.size, "white")
    bg.paste(rgba, mask=rgba.getchannel("A"))
    return bg


def _encode(img: Image.Image, *, as_jpeg: bool, quality: int) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if as_jpeg:
        _flatten_on_white(img).save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), "image/jpeg"
    if img.mode not in ("RGB", "RGBA", "L", "LA", "P", "1"):
        img = img.convert("RGBA")
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


def prepare_image(src: Path, *, max_long_edge_px: int) -> ImageInput:
    """Open, downscale and re-encode `src`; the original file is never modified.

    Byte cap: if the first encoding exceeds `MAX_IMAGE_BYTES`, fall back to RGB
    JPEG (q85) at full size — for a PNG that is a format change, for a JPEG a
    quality drop from q90 — then halve the long edge and re-encode until it fits
    (bounded by `MAX_SHRINK_STEPS`). The returned `media_type` matches the final
    encoding. Transparency is composited on white before any JPEG encoding.
    """
    src = require_file(Path(src))
    try:
        with Image.open(src) as opened:
            opened.load()  # force a full decode so truncated files fail here
            img = opened.copy()
            source_format = opened.format
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ConversionError(f"{src.name}: cannot decode image: {exc}") from exc

    if max(img.size) > max_long_edge_px:
        img.thumbnail((max_long_edge_px, max_long_edge_px), Image.Resampling.LANCZOS)

    as_jpeg = source_format == "JPEG"
    data, media_type = _encode(img, as_jpeg=as_jpeg, quality=90)
    if len(data) > MAX_IMAGE_BYTES:
        # Cheapest first: q85 JPEG at full size (PNG → JPEG, or JPEG q90 → q85) before any resolution is lost.
        data, media_type = _encode(img, as_jpeg=True, quality=FALLBACK_JPEG_QUALITY)
    steps = 0
    while len(data) > MAX_IMAGE_BYTES:
        if steps >= MAX_SHRINK_STEPS or max(img.size) < 2:
            raise ConversionError(f"{src.name}: cannot encode the image under {MAX_IMAGE_BYTES} bytes ({len(data)} bytes after {steps} halvings)")
        half = max(1, max(img.size) // 2)
        img.thumbnail((half, half), Image.Resampling.LANCZOS)
        data, media_type = _encode(img, as_jpeg=True, quality=FALLBACK_JPEG_QUALITY)
        steps += 1
    return ImageInput.from_bytes(data, media_type)


def _safe_name(name: str) -> str:
    """The filename as shown to the model: control characters and line breaks stripped (WU1 item 12)."""
    return _FILENAME_CONTROL.sub("", name).strip() or "image"


def render_description(desc: DescribeOutput) -> str:
    """Markdown from the six DescribeOutput fields (DESIGN §4.3 / §8.0)."""
    lines = [f"# {nfkc(desc.title).strip() or 'Untitled image'}", "", f"_Kind: {desc.kind}_", ""]
    description = nfkc(desc.description).strip()
    if description:
        lines += [description, ""]
    for heading, items in (("Elements", desc.elements), ("Visible text", desc.text_visible), ("Relationships", desc.relationships)):
        cleaned = [nfkc(item).strip() for item in items if item and item.strip()]
        if cleaned:
            lines += [f"## {heading}", *[f"- {item}" for item in cleaned], ""]
    return "\n".join(lines).rstrip() + "\n"


def describe(src: Path, *, max_long_edge_px: int, call_stage: CallStage, **call_kw: Any) -> DescribeOutput:
    src = Path(src)
    payload = prepare_image(src, max_long_edge_px=max_long_edge_px)
    user_text = (
        f"Describe the informational content of the attached image (file: {_safe_name(src.name)}). "
        "Fill every field of the schema; use empty lists where nothing applies."
    )
    result = call_stage("describe", describe_prompt.SYSTEM, user_text, DescribeOutput, [payload], **call_kw)
    data = getattr(result, "data", None)
    if not isinstance(data, DescribeOutput):
        raise ConversionError(f"{src.name}: describe stage returned {type(data).__name__}, expected DescribeOutput")
    return data


def extract(src: Path, *, max_long_edge_px: int, call_stage: CallStage, **call_kw: Any) -> list[Unit]:
    src = Path(src)
    desc = describe(src, max_long_edge_px=max_long_edge_px, call_stage=call_stage, **call_kw)
    text = render_description(desc)
    return [Unit("U1", src.name, [nfkc(desc.title).strip()] if desc.title.strip() else [], text, "image", src.name)]


def convert(
    src: Path, out_dir: Path, *, max_long_edge_px: int, call_stage: CallStage, stem: str | None = None, **call_kw: Any
) -> list[Unit]:
    src = Path(src)
    units = extract(src, max_long_edge_px=max_long_edge_px, call_stage=call_stage, **call_kw)
    write_outputs(Path(out_dir), stem or src.stem, units)
    return units


__all__ = [
    "CallStage",
    "ImageInput",
    "MAX_IMAGE_BYTES",
    "SUPPORTED_IMAGE_SUFFIXES",
    "convert",
    "describe",
    "extract",
    "prepare_image",
    "render_description",
]
