"""WU3 — `kg.registry`: permanent ids, retirement, `_registry.yaml` round-trip, slugs, title/alias index.

Spec: PRD R7 ("Nodes have permanent IDs that are never reused"); DESIGN §8.2 (file name
`<id>-<slug>.md`, or `<id>.md` when the slug is empty), §8.5 (registry shape, `next_seq` only
increases, `retired` ids stay listed forever).

Interface (chosen): Registry(corpus, schema) .allocate(title, *, run, aliases=()) -> RegistryRow,
.retire(id), .find(title_or_alias) -> id | None, .rows, .active(), .next_seq, .file_for(id),
.save(path), Registry.load(path); RegistryRow(id, title, norm_title, file, status, created_run, aliases);
slugify(title) -> str; stem_for(node_id, title) -> str.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wu3_fixtures import RUN_ID, RUN_ID_2, THAI_TITLE


@pytest.fixture
def reg():
    from kg.registry import Registry

    return Registry(corpus="kc", schema="education")


# ------------------------------------------------------------ allocation


def test_R7_ids_are_corpus_slug_plus_four_digit_seq_allocated_monotonically(reg):
    rows = [reg.allocate(t, run=RUN_ID) for t in ("Sample Space", "Event", "Probability Measure")]
    assert [r.id for r in rows] == ["kc-0001", "kc-0002", "kc-0003"]
    assert reg.next_seq == 4


def test_R7_row_carries_the_8_5_fields(reg):
    row = reg.allocate("Sample Space", run=RUN_ID, aliases=("Outcome space",))
    assert row.id == "kc-0001"
    assert row.title == "Sample Space"
    assert row.norm_title == "sample space"
    assert row.file == "nodes/kc-0001-sample-space.md"
    assert row.status == "active"
    assert row.created_run == RUN_ID
    assert list(row.aliases) == ["Outcome space"]


def test_R7_allocate_refuses_a_title_already_registered(reg):
    reg.allocate("Sample Space", run=RUN_ID)
    with pytest.raises(Exception):
        reg.allocate("sample spaces", run=RUN_ID)  # same normalised title → use find(), not allocate()


def test_R7_allocate_refuses_an_empty_title(reg):
    with pytest.raises(Exception):
        reg.allocate("   ", run=RUN_ID)


# ------------------------------------------------------------ retirement


def test_R7_retired_ids_stay_listed_and_are_never_reused(reg):
    a = reg.allocate("Old Idea", run=RUN_ID)
    reg.retire(a.id)
    b = reg.allocate("New Idea", run=RUN_ID_2)
    assert b.id == "kc-0002", "retired kc-0001 must not be handed out again"
    by_id = {r.id: r for r in reg.rows}
    assert by_id["kc-0001"].status == "retired"
    assert [r.id for r in reg.active()] == ["kc-0002"]
    assert reg.find("Old Idea") is None, "retired titles are not matched for reuse"


def test_R7_retire_unknown_id_is_an_error(reg):
    with pytest.raises(Exception):
        reg.retire("kc-0042")


def test_R7_a_title_can_be_re_registered_after_retirement_under_a_new_id(reg):
    a = reg.allocate("Sample Space", run=RUN_ID)
    reg.retire(a.id)
    b = reg.allocate("Sample Space", run=RUN_ID_2)
    assert b.id == "kc-0002"
    assert reg.find("Sample Space") == "kc-0002"


# ------------------------------------------------------------- round trip


def test_R7_registry_yaml_round_trip(reg, tmp_path: Path):
    from kg.registry import Registry

    reg.allocate("Sample Space", run=RUN_ID, aliases=("Outcome space",))
    e = reg.allocate("Event", run=RUN_ID)
    reg.allocate(THAI_TITLE, run=RUN_ID)
    reg.retire(e.id)
    path = tmp_path / "_registry.yaml"
    reg.save(path)

    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc["corpus"] == "kc" and doc["schema"] == "education" and doc["next_seq"] == 4
    assert [n["id"] for n in doc["nodes"]] == ["kc-0001", "kc-0002", "kc-0003"]
    assert {"id", "title", "norm_title", "file", "status", "created_run"} <= set(doc["nodes"][0])
    assert doc["nodes"][1]["status"] == "retired"
    assert doc["nodes"][2]["title"] == THAI_TITLE

    back = Registry.load(path)
    assert back.corpus == "kc" and back.schema == "education"
    assert back.next_seq == reg.next_seq
    assert list(back.rows) == list(reg.rows)
    assert back.allocate("Fourth", run=RUN_ID_2).id == "kc-0004"


def test_R7_next_seq_only_increases_even_when_rows_are_sparse(tmp_path: Path):
    from kg.registry import Registry

    path = tmp_path / "_registry.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "corpus": "stat101",
                "schema": "education",
                "next_seq": 12,
                "nodes": [
                    {"id": "stat101-0004", "title": "Sample Space", "norm_title": "sample space", "file": "nodes/stat101-0004-sample-space.md", "status": "active", "created_run": RUN_ID}
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    reg = Registry.load(path)
    assert reg.allocate("Something New", run=RUN_ID_2).id == "stat101-0012"
    assert reg.find("sample space") == "stat101-0004"


def test_R7_load_missing_file_is_a_clear_error(tmp_path: Path):
    from kg.registry import Registry

    with pytest.raises(Exception):
        Registry.load(tmp_path / "nope.yaml")


def test_R7_save_is_deterministic(reg, tmp_path: Path):
    reg.allocate("A", run=RUN_ID)
    reg.allocate("B", run=RUN_ID)
    p1, p2 = tmp_path / "r1.yaml", tmp_path / "r2.yaml"
    reg.save(p1)
    reg.save(p2)
    assert p1.read_bytes() == p2.read_bytes()


# ------------------------------------------------------------ title index


def test_R7_find_matches_normalised_titles_and_aliases(reg):
    reg.allocate("Sample Space", run=RUN_ID, aliases=("Outcome space", "Ω"))
    reg.allocate("Event", run=RUN_ID)
    assert reg.find("sample space") == "kc-0001"
    assert reg.find("The Sample Spaces") == "kc-0001"
    assert reg.find("outcome   SPACE") == "kc-0001"
    assert reg.find("ω") == "kc-0001"
    assert reg.find("Events") == "kc-0002"
    assert reg.find("Nothing Like It") is None


def test_R7_file_for_returns_the_registered_relative_path(reg):
    reg.allocate("Sample Space", run=RUN_ID)
    assert reg.file_for("kc-0001") == "nodes/kc-0001-sample-space.md"


# ------------------------------------------------------------------ slugs


@pytest.mark.parametrize(
    "title,slug",
    [
        ("Sample Space", "sample-space"),
        ("Bayes' Theorem", "bayes-theorem"),
        ("  Cloud   First: Policy!  ", "cloud-first-policy"),
        ("ＡＩ Terms", "ai-terms"),
        (THAI_TITLE, ""),
        (f"AI {THAI_TITLE}", "ai"),
        ("P(A|B)", "p-a-b"),
    ],
)
def test_R7_slugify(title: str, slug: str):
    from kg.registry import slugify

    assert slugify(title) == slug


def test_R7_slug_is_lowercase_ascii_hyphenated_only():
    from kg.registry import slugify

    s = slugify("Weird—Dashes & Symbols_100%")
    assert s and all(c.islower() or c.isdigit() or c == "-" for c in s)
    assert "--" not in s and not s.startswith("-") and not s.endswith("-")


def test_R7_stem_for_uses_id_only_when_slug_is_empty():
    from kg.registry import stem_for

    assert stem_for("stat101-0004", "Sample Space") == "stat101-0004-sample-space"
    assert stem_for("kc-0001", THAI_TITLE) == "kc-0001"


def test_R7_very_long_titles_produce_a_bounded_stem():
    from kg.registry import stem_for

    stem = stem_for("kc-0001", " ".join(["word"] * 60))
    assert stem.startswith("kc-0001-")
    assert len(stem) <= 120
