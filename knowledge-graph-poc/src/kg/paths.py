"""Sandbox guard — the sole path resolver for the pipeline (DESIGN §3.3, §23; D14).

Three rules, all enforced in code rather than by convention:

  1. Containment: every resolved path lies inside the configured root for its kind
     AND inside the project directory — checked lexically and again on the
     ``realpath`` of both sides, so a symlink anywhere on the path (including a
     configured root that is itself a symlink) cannot lead outside the project.
  2. Denylist: any path containing ``/TK-PKA/`` or ``01-knowledge-base``, or whose
     realpath lies under ``~/knowledge-base``, is refused unconditionally
     (case-insensitive — the default macOS filesystem is too).
  3. Roots: the five configured roots must be relative, inside the project
     directory (lexically and physically), distinct, and not nested in one another.

``resolve()`` never creates anything on disk.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

KINDS: tuple[str, ...] = ("inbox", "converted", "graph", "processed", "runs")

# Lower-cased substrings that must never appear in any path the pipeline touches.
# Both are matched against a "/"-normalised, lower-cased path with a trailing "/",
# so a bare final segment ("…/TK-PKA") is caught as well as an interior one.
#
# The two entries are deliberately asymmetric:
#   * "/tk-pka/" is segment-matched (slashes on both sides) — the vault folder name
#     is short and a bare substring would be too broad.
#   * "01-knowledge-base" is a plain substring — it also refuses e.g.
#     "01-knowledge-base-backup" or "old-01-knowledge-base". Over-refusal is the
#     intended trade-off: nothing legitimate in this project carries that name.
_DENYLIST: tuple[str, ...] = ("/tk-pka/", "01-knowledge-base")


def _normalised_for_denylist(path: str | os.PathLike[str]) -> str:
    text = str(path).replace("\\", "/").lower().rstrip("/")
    return text + "/"


def _home_knowledge_base_prefixes() -> tuple[str, ...]:
    """Normalised prefixes for the user's second vault, ``~/knowledge-base`` (segment-matched).

    Both the expanded path and its realpath are returned so a symlinked vault is
    caught whichever form a candidate's realpath takes. If HOME cannot be
    expanded the literal "~/knowledge-base" is kept, which is harmless.

    Evaluated on every call rather than cached at import so the guard follows
    the process's HOME (the selftest redirects it to a temporary directory).
    """
    expanded = os.path.expanduser("~/knowledge-base")
    forms = {expanded, os.path.realpath(expanded)}
    return tuple(sorted(_normalised_for_denylist(f) for f in forms))


class SandboxViolation(Exception):
    """A path was refused by the sandbox guard."""


def is_denied(path: str | os.PathLike[str]) -> bool:
    """True if `path` contains a denylisted segment or lies under ``~/knowledge-base`` (rule 2)."""
    text = _normalised_for_denylist(path)
    if any(needle in text for needle in _DENYLIST):
        return True
    # Third entry: prefix (segment) match, not substring — "knowledge-base" alone
    # would be far too broad, and the trailing "/" from normalisation means
    # "~/knowledge-base-other/" does not match.
    return any(text.startswith(prefix) for prefix in _home_knowledge_base_prefixes())


def _lexical(path: Path) -> str:
    """Collapse `.` and `..` without touching the filesystem."""
    return os.path.normpath(str(path))


def _is_within(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def _overlaps(a: str, b: str) -> bool:
    """True if the two paths are equal or one is an ancestor of the other."""
    return _is_within(a, b) or _is_within(b, a)


class Sandbox:
    """Resolver bound to one project root and its five configured folders."""

    def __init__(self, project_root: Path, roots: Mapping[str, str]) -> None:
        project_root = Path(project_root)
        if "\0" in str(project_root):
            raise SandboxViolation("project root contains a NUL byte")
        if not project_root.is_absolute():
            raise SandboxViolation(f"project root must be absolute: {project_root}")
        if is_denied(project_root):
            raise SandboxViolation(f"project root lies inside a forbidden location: {project_root}")
        root_str = _lexical(project_root)
        real_root_str = os.path.realpath(root_str)
        if is_denied(real_root_str):
            raise SandboxViolation(f"project root resolves into a forbidden location via a symlink: {project_root}")

        missing = [k for k in KINDS if k not in roots]
        if missing:
            raise SandboxViolation(f"paths: missing configured folder(s): {', '.join(missing)}")
        unknown = [k for k in roots if k not in KINDS]
        if unknown:
            raise SandboxViolation(f"paths: unknown folder kind(s): {', '.join(unknown)}")

        resolved: dict[str, Path] = {}
        real: dict[str, str] = {}
        for kind in KINDS:
            rel = roots[kind]
            if not isinstance(rel, str) or not rel.strip():
                raise SandboxViolation(f"paths.{kind}: must be a non-empty relative path")
            if "\0" in rel:
                raise SandboxViolation(f"paths.{kind}: contains a NUL byte")
            if Path(rel).is_absolute():
                raise SandboxViolation(f"paths.{kind}: absolute paths are refused ({rel})")
            if ".." in Path(rel).parts:
                raise SandboxViolation(f"paths.{kind}: '..' is refused ({rel})")
            candidate = _lexical(project_root / rel)
            if not _is_within(candidate, root_str):
                raise SandboxViolation(f"paths.{kind}: escapes the project directory ({rel})")
            if is_denied(candidate):
                raise SandboxViolation(f"paths.{kind}: points at a forbidden location ({rel})")
            # Rule 3, physical: a root that is (or sits under) a symlink must still
            # land inside the real project directory, and outside the denylist.
            real_candidate = os.path.realpath(candidate)
            if not _is_within(real_candidate, real_root_str):
                raise SandboxViolation(f"paths.{kind}: resolves outside the project directory via a symlink ({rel})")
            if is_denied(real_candidate):
                raise SandboxViolation(f"paths.{kind}: resolves into a forbidden location via a symlink ({rel})")
            resolved[kind] = Path(candidate)
            real[kind] = real_candidate

        # Roots must be distinct and non-nested, lexically and physically.
        for i, a in enumerate(KINDS):
            for b in KINDS[i + 1 :]:
                if _overlaps(str(resolved[a]), str(resolved[b])) or _overlaps(real[a], real[b]):
                    raise SandboxViolation(
                        f"paths.{a} and paths.{b} overlap ({roots[a]!r}, {roots[b]!r}): roots must be distinct and not nested"
                    )

        self.project_root: Path = Path(root_str)
        self._real_project_root: str = real_root_str
        self._roots: dict[str, Path] = resolved
        self._real_roots: dict[str, str] = real

    # ------------------------------------------------------------------ API

    @property
    def roots(self) -> Mapping[str, Path]:
        return dict(self._roots)

    def root(self, kind: str) -> Path:
        """Absolute path of the configured folder for `kind`."""
        if kind not in self._roots:
            raise SandboxViolation(f"unknown folder kind: {kind!r} (expected one of {', '.join(KINDS)})")
        return self._roots[kind]

    def resolve(self, kind: str, relative: str | os.PathLike[str]) -> Path:
        """Return the absolute path of `relative` inside the `kind` root, or raise.

        Does not create directories or files. Symlinks that already exist on disk
        are followed for the containment check only; the returned path is lexical.
        """
        root = self.root(kind)
        if "\0" in str(relative):
            raise SandboxViolation(f"{kind}: path contains a NUL byte")
        rel = Path(relative)
        if rel.is_absolute():
            raise SandboxViolation(f"{kind}: absolute paths are refused ({relative})")

        root_str = str(root)
        candidate = _lexical(root / rel)
        if not _is_within(candidate, root_str):
            raise SandboxViolation(f"{kind}: path escapes its root ({relative})")
        if is_denied(candidate):
            raise SandboxViolation(f"{kind}: path touches a forbidden location ({relative})")

        # Rule 1, physical: if any part already exists as a symlink, the real
        # location must still be inside the real kind root AND the real project
        # root (a symlinked kind root would otherwise pass the first check).
        real_candidate = os.path.realpath(candidate)
        if not _is_within(real_candidate, self._real_roots[kind]):
            raise SandboxViolation(f"{kind}: path resolves outside its root via a symlink ({relative})")
        if not _is_within(real_candidate, self._real_project_root):
            raise SandboxViolation(f"{kind}: path resolves outside the project directory via a symlink ({relative})")
        if is_denied(real_candidate):
            raise SandboxViolation(f"{kind}: path resolves into a forbidden location ({relative})")

        return Path(candidate)

    def __repr__(self) -> str:
        return f"Sandbox(project_root={str(self.project_root)!r}, kinds={list(self._roots)})"
