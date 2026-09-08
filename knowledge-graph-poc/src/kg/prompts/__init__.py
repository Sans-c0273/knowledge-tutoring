"""Per-stage prompts (DESIGN §7.8; D15): each module exposes ``VERSION``, ``SYSTEM`` and ``TEXT_SHA``.

``TEXT_SHA`` is the literal SHA-256 of ``SYSTEM`` recorded when the prompt was
last edited; ``text_sha()`` recomputes it so the self-test can fail on a stale
hash. ``prompt_set_hash()`` combines the four so a run report pins the whole
prompt set. Prompts are provider-neutral.
"""

from __future__ import annotations

import hashlib
import importlib

STAGES: tuple[str, ...] = ("describe", "atomize", "edges", "dedup")


def text_sha(system: str) -> str:
    return hashlib.sha256(system.encode("utf-8")).hexdigest()


def versions() -> dict[str, str]:
    """stage -> VERSION for every stage prompt."""
    return {stage: importlib.import_module(f"kg.prompts.{stage}").VERSION for stage in STAGES}


def prompt_set_hash() -> str:
    """SHA-256 over the four stage TEXT_SHAs, in stage order."""
    shas = [importlib.import_module(f"kg.prompts.{stage}").TEXT_SHA for stage in STAGES]
    return hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()


__all__ = ["STAGES", "prompt_set_hash", "text_sha", "versions"]
