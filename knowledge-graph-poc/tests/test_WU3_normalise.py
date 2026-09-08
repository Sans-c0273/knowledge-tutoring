"""WU3 — `kg.normalise`: title normalisation and similarity (DESIGN §9, D10; R12, R16).

§9: NFKC → casefold → strip punctuation → collapse whitespace → drop leading article → ASCII
plural strip. Token-set similarity for spaced Latin titles; character-trigram similarity for
Thai / unspaced titles. The exact coefficient (Dice vs Jaccard) is not pinned here — the
properties that consolidate/dedup/stability rely on are.

Interface (chosen): norm_title(s) -> str, similarity(a, b) -> float in [0, 1], uses_trigrams(s) -> bool.
"""

from __future__ import annotations

import pytest

from wu3_fixtures import THAI_TITLE, THAI_TITLE_2


# --------------------------------------------------------------- norm_title


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Sample Space", "sample space"),
        ("  Sample   Space ", "sample space"),
        ("The Sample Spaces", "sample space"),
        ("A Sample Space", "sample space"),
        ("An Event", "event"),
        ("Bayes' Theorem", "bayes theorem"),
        ("Cloud-First Policy!", "cloud first policy"),
        ("ＡＩ Ｔｅｒｍｓ", "ai term"),  # NFKC fullwidth → ASCII, then plural strip
        ("Events", "event"),
        ("Probability Measures", "probability measure"),
        ("Straße", "strasse"),  # casefold, not lower
    ],
)
def test_S9_norm_title(raw: str, expected: str):
    from kg.normalise import norm_title

    assert norm_title(raw) == expected


def test_S9_norm_title_is_idempotent():
    from kg.normalise import norm_title

    for raw in ("The Sample Spaces", "Bayes' Theorem", THAI_TITLE, "ＡＩ"):
        assert norm_title(norm_title(raw)) == norm_title(raw)


def test_S9_norm_title_keeps_thai_text_and_collapses_whitespace():
    from kg.normalise import norm_title

    assert norm_title(f"  {THAI_TITLE}  ") == THAI_TITLE
    assert norm_title(f"{THAI_TITLE}   {THAI_TITLE_2}") == f"{THAI_TITLE} {THAI_TITLE_2}"


def test_S9_norm_title_does_not_strip_a_word_that_merely_starts_like_an_article():
    from kg.normalise import norm_title

    assert norm_title("Theorem") == "theorem"
    assert norm_title("Anomaly Detection") == "anomaly detection"


def test_S9_norm_title_of_empty_or_punctuation_only_is_empty():
    from kg.normalise import norm_title

    assert norm_title("") == ""
    assert norm_title("!!! ---") == ""


# --------------------------------------------------------------- similarity


def test_S9_similarity_is_one_for_identical_and_normalised_equal_titles():
    from kg.normalise import similarity

    assert similarity("Sample Space", "Sample Space") == 1.0
    assert similarity("The Sample Spaces", "sample space") == 1.0
    assert similarity(THAI_TITLE, f" {THAI_TITLE} ") == 1.0


def test_S9_similarity_is_zero_for_disjoint_titles():
    from kg.normalise import similarity

    assert similarity("Sample Space", "Bayes Theorem") == 0.0


def test_S9_similarity_is_symmetric_and_bounded():
    from kg.normalise import similarity

    pairs = [("Sample Space", "Sample Space Definition"), (THAI_TITLE, THAI_TITLE_2), ("Cloud First Policy", "Policy")]
    for a, b in pairs:
        s = similarity(a, b)
        assert 0.0 <= s <= 1.0
        assert s == similarity(b, a)


def test_S9_latin_partial_overlap_is_strictly_between_zero_and_one_and_monotone():
    from kg.normalise import similarity

    two_of_three = similarity("Sample Space", "Sample Space Definition")
    one_of_four = similarity("Sample Set", "Sample Space Definition")
    assert 0.0 < one_of_four < two_of_three < 1.0


def test_S9_latin_similarity_is_token_based_not_character_based():
    from kg.normalise import similarity

    # Same tokens, different order → identical token set → 1.0
    assert similarity("Space Sample", "Sample Space") == 1.0
    # Sharing characters but no tokens → 0 (a char method would score this well above 0)
    assert similarity("Sampler Tool", "Sample Space") == 0.0


def test_D10_thai_titles_sharing_substrings_score_above_zero():
    """Token methods return 0 for any two non-identical Thai titles (no word delimiters); trigrams must not."""
    from kg.normalise import similarity

    s = similarity(THAI_TITLE, THAI_TITLE_2)  # การเรียนรู้เชิงลึก vs การเรียนรู้ของเครื่อง share "การเรียนรู้"
    assert 0.0 < s < 1.0


def test_D10_thai_unrelated_titles_score_low():
    from kg.normalise import similarity

    related = similarity(THAI_TITLE, THAI_TITLE_2)
    unrelated = similarity(THAI_TITLE, "สถิติ")
    assert unrelated < related


def test_D10_mixed_thai_latin_and_unspaced_titles_use_trigrams():
    from kg.normalise import similarity, uses_trigrams

    assert uses_trigrams(THAI_TITLE) is True
    assert uses_trigrams(f"AI {THAI_TITLE}") is True
    assert uses_trigrams("SampleSpace") is True, "no whitespace → trigrams (§9)"
    assert uses_trigrams("Sample Space") is False
    assert similarity("SampleSpace", "SampleSpaces") > 0.5


def test_S9_similarity_with_empty_input_is_zero_not_an_error():
    from kg.normalise import similarity

    assert similarity("", "Sample Space") == 0.0
    assert similarity("", "") == 0.0
