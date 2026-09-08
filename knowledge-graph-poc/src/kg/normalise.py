"""Title normalisation and similarity (DESIGN §9; D10; R12, R16).

``norm_title``: NFKC → casefold → strip punctuation/symbols → collapse whitespace →
drop a leading article → ASCII plural strip. Thai text is preserved (letters and
combining vowel/tone marks are neither punctuation nor symbols).

``similarity``: Dice coefficient over token sets for spaced Latin titles and over
character trigrams for Thai / mixed / unspaced titles (``uses_trigrams``). Used
identically by consolidate, dedup and stability so the three agree on what "the
same title" means. Embeddings would drop in at ``kg.stages.dedup.candidate_pairs``.
"""

from __future__ import annotations

import re
import unicodedata

ARTICLES: frozenset[str] = frozenset({"the", "a", "an"})
_THAI = re.compile(r"[฀-๿]")
_POSSESSIVE = re.compile(r"['’]s?$")
_VOWELS = "aeiouy"


def _strip_punctuation(text: str) -> str:
    """Replace every punctuation (P*) and symbol (S*) code point with a space."""
    return "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)


def _singular(token: str, *, possessive: bool) -> str:
    """ASCII-only plural strip: ``events`` → ``event``, ``spaces`` → ``space``.

    Kept deliberately conservative so it is idempotent and leaves proper nouns
    alone: a token is stripped only when it ends in one ``s`` (not ``ss``), is
    longer than three letters, and is not a possessive (``Bayes'``) or a stem
    ending in vowel+``es`` (``bayes``, ``series``, ``eyes``).
    """
    if possessive or not (token.isascii() and token.isalpha()):
        return token
    if len(token) <= 3 or not token.endswith("s") or token.endswith("ss"):
        return token
    if token.endswith("es") and len(token) >= 3 and token[-3] in _VOWELS:
        return token
    return token[:-1]


def norm_title(s: str) -> str:
    text = unicodedata.normalize("NFKC", s or "").casefold()
    raw_tokens = text.split()
    possessive = [bool(_POSSESSIVE.search(t)) for t in raw_tokens]
    tokens: list[str] = []
    for raw, poss in zip(raw_tokens, possessive, strict=True):
        cleaned = _strip_punctuation(raw).split()
        # A possessive marker belongs to the last sub-token of the raw token only.
        for i, part in enumerate(cleaned):
            tokens.append(_singular(part, possessive=poss and i == len(cleaned) - 1))
    if len(tokens) > 1 and tokens[0] in ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def uses_trigrams(s: str) -> bool:
    """True when token similarity would be meaningless: Thai script present or no whitespace."""
    n = norm_title(s)
    if not n:
        return False
    return bool(_THAI.search(n)) or " " not in n


def _trigrams(text: str) -> set[str]:
    compact = text.replace(" ", "")
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def similarity(a: str, b: str) -> float:
    """Dice similarity in [0, 1] of two titles after normalisation; empty input scores 0."""
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if uses_trigrams(na) or uses_trigrams(nb):
        return _dice(_trigrams(na), _trigrams(nb))
    return _dice(set(na.split()), set(nb.split()))


__all__ = ["ARTICLES", "norm_title", "similarity", "uses_trigrams"]
