"""Stage ``dedup`` (S6; R12; DESIGN §6, §8.0, §11): pairwise same-concept adjudication."""

from __future__ import annotations

VERSION = "dedup@1"

SYSTEM = """You decide whether two concept notes describe the same concept.

You receive a numbered list of pairs. Each pair shows two notes with their ids, titles, aliases and
definitions. For every pair, return exactly one judgement with the two ids copied verbatim and a verdict:
- same: the two notes are the same concept under different wording, spelling, language or granularity of
  title, such that a reader would expect a single note.
- different: they are distinct concepts, even if closely related (a theorem and its proof, a set and an
  element, a topic and one of its parts).
- unsure: the definitions do not give enough information to decide.

Give a one-sentence reason for each judgement. Judge only the pairs you were given; do not add pairs."""

TEXT_SHA = "2d277821569725a53c94d870699f33fd6e128f1b1bfe340cc30b019b8cc49892"
