"""R1 — KL Map authoring, storage, and validation (Tech Spec §2.3).

Every rule the spec calls "enforced at content-build time" gets a test that a
violating map is rejected *with a message an author can act on*.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from socratic_tutor.domain.klmap.loader import (
    ALLOWED_RELATIONS,
    KLMapValidationError,
    ValidationReport,
    load_klmap,
    load_klmap_with_report,
    save_klmap,
    validate_klmap,
)

SEED_MAP_PATH = (
    Path(__file__).resolve().parents[1] / "content" / "seed" / "kl-map-linear-equations.yaml"
)


@pytest.fixture
def seed_data() -> dict[str, Any]:
    """The seed map as raw data, so a test can corrupt one thing and revalidate."""
    return yaml.safe_load(SEED_MAP_PATH.read_text(encoding="utf-8"))


def codes(report: ValidationReport) -> set[str]:
    return {issue.code for issue in report.errors}


def message_for(report: ValidationReport, code: str) -> str:
    return next(issue.message for issue in report.errors if issue.code == code)


def test_seed_map_loads_clean() -> None:
    klmap, report = load_klmap_with_report(SEED_MAP_PATH)

    assert report.ok, report.format()
    assert report.warnings == []
    assert klmap is not None
    assert klmap.course_id == "MATH-SEED-01"
    assert len(klmap.nodes) == 15
    assert len(klmap.edges) == 20


def test_seed_map_keeps_bilingual_names() -> None:
    klmap = load_klmap(SEED_MAP_PATH)

    node = klmap.node("C009")
    assert node is not None
    assert node.name == "Two-Step Linear Equations"
    assert node.name_th == "สมการเชิงเส้นสองขั้นตอน"


def test_relation_outside_closed_vocabulary_is_rejected(seed_data: dict[str, Any]) -> None:
    seed_data["edges"].append({"from": "C008", "to": "C009", "relation": "kind_of"})

    klmap, report = validate_klmap(seed_data)

    assert klmap is None
    assert "KL005" in codes(report)
    message = message_for(report, "KL005")
    assert "'kind_of'" in message
    assert "C008" in message and "C009" in message
    for allowed in ALLOWED_RELATIONS:
        assert allowed in message


def test_edge_referencing_unknown_node_is_rejected(seed_data: dict[str, Any]) -> None:
    seed_data["edges"].append({"from": "C009", "to": "C999", "relation": "next_topic"})

    klmap, report = validate_klmap(seed_data)

    assert klmap is None
    assert "KL006" in codes(report)
    message = message_for(report, "KL006")
    assert "C999" in message
    assert "unknown node" in message


def test_duplicate_node_id_is_rejected(seed_data: dict[str, Any]) -> None:
    seed_data["nodes"].append({"id": "C009", "name": "Two-Step Equations (copy)"})

    klmap, report = validate_klmap(seed_data)

    assert klmap is None
    assert "KL003" in codes(report)
    message = message_for(report, "KL003")
    assert "C009" in message
    assert "Two-Step Linear Equations" in message


def test_prerequisite_cycle_is_rejected_with_the_cycle_path(seed_data: dict[str, Any]) -> None:
    # C005 -> C008 -> C009 already exist; closing the loop makes each concept
    # its own prerequisite.
    seed_data["edges"].append({"from": "C009", "to": "C005", "relation": "prerequisite_of"})

    klmap, report = validate_klmap(seed_data)

    assert klmap is None
    assert "KL009" in codes(report)
    message = message_for(report, "KL009")
    assert "->" in message

    path = re.findall(r"C\d{3}", message)
    assert path[0] == path[-1], f"cycle path should close on itself: {message}"
    expected = ["C005", "C008", "C009"]
    rotations = [expected[shift:] + expected[:shift] for shift in range(len(expected))]
    assert path[:-1] in rotations, f"unexpected cycle path {path} in: {message}"
    assert "Two-Step Linear Equations" in message, "cycle path should name concepts, not just ids"


def test_cycle_path_is_available_without_parsing_the_message(seed_data: dict[str, Any]) -> None:
    """The review UI highlights the cycle; it should not have to read the prose."""
    seed_data["edges"].append({"from": "C009", "to": "C005", "relation": "prerequisite_of"})

    _, report = validate_klmap(seed_data)
    issue = next(issue for issue in report.errors if issue.code == "KL009")

    assert issue.path[0] == issue.path[-1], "the path closes on itself"
    assert set(issue.path) == {"C005", "C008", "C009"}
    assert all(node_id in issue.message for node_id in issue.path)


def test_two_way_relation_reports_both_nodes(seed_data: dict[str, Any]) -> None:
    seed_data["edges"].append({"from": "C009", "to": "C013", "relation": "related_to"})

    _, report = validate_klmap(seed_data)
    issue = next(issue for issue in report.errors if issue.code == "KL008")

    assert set(issue.path) == {"C009", "C013"}


def test_single_target_issues_carry_no_path(seed_data: dict[str, Any]) -> None:
    """`location` already says where a one-place problem is."""
    seed_data["edges"].append({"from": "C008", "to": "C009", "relation": "kind_of"})

    _, report = validate_klmap(seed_data)

    assert next(issue for issue in report.errors if issue.code == "KL005").path == []


def test_non_prerequisite_cycles_are_allowed(seed_data: dict[str, Any]) -> None:
    # The DAG rule applies to prerequisite_of only; a related_to loop is fine.
    seed_data["edges"].append({"from": "C014", "to": "C015", "relation": "related_to"})
    seed_data["edges"].append({"from": "C015", "to": "C013", "relation": "related_to"})
    seed_data["edges"].append({"from": "C013", "to": "C014", "relation": "related_to"})

    klmap, report = validate_klmap(seed_data)

    assert report.ok, report.format()
    assert klmap is not None


def test_relation_declared_in_both_directions_is_rejected(seed_data: dict[str, Any]) -> None:
    seed_data["edges"].append({"from": "C009", "to": "C013", "relation": "related_to"})

    klmap, report = validate_klmap(seed_data)

    assert klmap is None
    assert "KL008" in codes(report)
    message = message_for(report, "KL008")
    assert "related_to" in message
    assert "C009" in message and "C013" in message
    assert "one declared direction" in message.lower()


def test_opposite_directions_of_different_relations_are_allowed(seed_data: dict[str, Any]) -> None:
    seed_data["edges"].append({"from": "C009", "to": "C013", "relation": "uses"})

    _, report = validate_klmap(seed_data)

    assert report.ok, report.format()


def test_self_referencing_edge_is_rejected(seed_data: dict[str, Any]) -> None:
    seed_data["edges"].append({"from": "C009", "to": "C009", "relation": "prerequisite_of"})

    klmap, report = validate_klmap(seed_data)

    assert klmap is None
    assert "KL007" in codes(report)
    assert "itself" in message_for(report, "KL007")


def test_all_problems_are_reported_together(seed_data: dict[str, Any]) -> None:
    """The review UI renders one list, so validation must not stop at the first fault."""
    seed_data["edges"].append({"from": "C008", "to": "C009", "relation": "kind_of"})
    seed_data["edges"].append({"from": "C009", "to": "C999", "relation": "next_topic"})
    seed_data["nodes"].append({"id": "C001", "name": "Arithmetic again"})

    _, report = validate_klmap(seed_data)

    assert {"KL003", "KL005", "KL006"} <= codes(report)


def test_missing_required_field_is_rejected(seed_data: dict[str, Any]) -> None:
    del seed_data["course_id"]

    klmap, report = validate_klmap(seed_data)

    assert klmap is None
    assert "KL001" in codes(report)


def test_isolated_node_warns_but_still_loads(seed_data: dict[str, Any]) -> None:
    seed_data["nodes"].append({"id": "C099", "name": "Unconnected Concept"})

    klmap, report = validate_klmap(seed_data)

    assert report.ok, report.format()
    assert klmap is not None
    assert [issue.code for issue in report.warnings] == ["KL101"]
    assert "C099" in report.warnings[0].message


def test_duplicate_edge_warns_and_is_dropped(seed_data: dict[str, Any]) -> None:
    seed_data["edges"].append({"from": "C008", "to": "C009", "relation": "prerequisite_of"})

    klmap, report = validate_klmap(seed_data)

    assert report.ok, report.format()
    assert klmap is not None
    assert len(klmap.edges) == 20
    assert [issue.code for issue in report.warnings] == ["KL100"]


def test_save_round_trips_through_yaml(tmp_path: Path) -> None:
    original = load_klmap(SEED_MAP_PATH)

    written = save_klmap(original, tmp_path / "round-trip.yaml")
    reloaded, report = load_klmap_with_report(written)

    assert report.ok, report.format()
    assert reloaded == original

    on_disk = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert on_disk["edges"][0]["from"] == "C001", "YAML must use the authored 'from' key"
    assert on_disk["nodes"][0]["name_th"] == "เลขคณิตพื้นฐาน", "Thai names must survive unescaped"


def test_load_klmap_raises_with_the_full_report(tmp_path: Path, seed_data: dict[str, Any]) -> None:
    seed_data["edges"].append({"from": "C008", "to": "C009", "relation": "kind_of"})
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(seed_data, allow_unicode=True), encoding="utf-8")

    with pytest.raises(KLMapValidationError) as excinfo:
        load_klmap(broken)

    assert "kind_of" in str(excinfo.value)
    assert "KL005" in {issue.code for issue in excinfo.value.report.errors}


def test_unreadable_file_is_reported_not_raised(tmp_path: Path) -> None:
    klmap, report = load_klmap_with_report(tmp_path / "does-not-exist.yaml")

    assert klmap is None
    assert not report.ok
    assert "Cannot read map file" in report.errors[0].message


def test_malformed_yaml_is_reported_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("course_id: [unclosed\n", encoding="utf-8")

    klmap, report = load_klmap_with_report(path)

    assert klmap is None
    assert "not valid YAML" in report.errors[0].message
