"""Socratic AI Tutor POC — rule-driven Pedagogy Layer over an LLM.

Package root. Deliberately import-light: submodules pull in heavy dependencies
(anthropic, httpx, chromadb) and are imported where they are used, so importing
`socratic_tutor` costs nothing.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
