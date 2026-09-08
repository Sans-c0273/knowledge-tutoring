"""WU0 — kg.paths sandbox guard.

Spec: PRD §3 ("never reads from or writes to the TK-PKA vault or
01-knowledge-base/"), PRD §9 ("No part of the pipeline touches paths outside its
own configured folders"), DESIGN §3.3 / §23, DECISIONS D14.

Three rules, all exercised:
  (1) every resolved path is contained in the configured root for its kind;
  (2) hard-coded denylist: anything containing `/TK-PKA/` or `01-knowledge-base`
      is refused unconditionally, even if it would otherwise be inside a root;
  (3) roots themselves must be inside the project directory (no absolute, no `..`).

The forbidden path strings below are strings only. They never touch the real
vault: fake user names / fake prefixes are used so nothing resolves to
/Users/tanat/TK-PKA even on a case-insensitive filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kg.paths import KINDS, Sandbox, SandboxViolation

ROOTS = {
    "inbox": "data/inbox",
    "converted": "data/converted",
    "graph": "data/graph",
    "processed": "data/processed",
    "runs": "data/runs",
}


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    for rel in ROOTS.values():
        (tmp_path / rel).mkdir(parents=True)
    return Sandbox(project_root=tmp_path, roots=ROOTS)


# ------------------------------------------------------------ happy path


def test_R18_sandbox_kinds_are_exactly_the_five_configured_folders():
    assert tuple(KINDS) == ("inbox", "converted", "graph", "processed", "runs")


@pytest.mark.parametrize("kind", ROOTS)
def test_R18_resolve_returns_absolute_path_inside_kind_root(sandbox: Sandbox, tmp_path: Path, kind: str):
    p = sandbox.resolve(kind, "week01-probability.pptx")
    assert p.is_absolute()
    assert p == tmp_path / ROOTS[kind] / "week01-probability.pptx"
    assert p.relative_to(tmp_path / ROOTS[kind])  # contained


def test_R18_resolve_allows_nested_relative_paths(sandbox: Sandbox, tmp_path: Path):
    p = sandbox.resolve("graph", "nodes/stat101-0004-sample-space.md")
    assert p == tmp_path / "data/graph/nodes/stat101-0004-sample-space.md"


def test_R18_resolve_dot_returns_the_root_itself(sandbox: Sandbox, tmp_path: Path):
    assert sandbox.resolve("inbox", ".") == tmp_path / "data/inbox"


def test_R18_resolve_does_not_create_anything_on_disk(sandbox: Sandbox, tmp_path: Path):
    p = sandbox.resolve("runs", "stability-x/a/report.md")
    assert not p.exists()
    assert not (tmp_path / "data/runs/stability-x").exists()


# --------------------------------------------------- rule 1: containment


def test_R18_refuses_dotdot_escape_from_kind_root(sandbox: Sandbox):
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", "../../etc/passwd")


def test_R18_refuses_dotdot_that_lands_in_a_sibling_kind(sandbox: Sandbox):
    # inbox/../graph is inside the project but outside the *inbox* root.
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", "../graph/nodes/x.md")


def test_R18_refuses_absolute_path_elsewhere(sandbox: Sandbox):
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", "/etc/passwd")


def test_R18_refuses_unknown_kind(sandbox: Sandbox):
    with pytest.raises(SandboxViolation):
        sandbox.resolve("vault", "anything.md")


def test_R18_refuses_symlink_that_escapes_the_root(sandbox: Sandbox, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.md").write_text("x", encoding="utf-8")
    os.symlink(outside, tmp_path / "data/inbox/link")
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", "link/secret.md")


# ------------------------------------------------- rule 2: denylist


@pytest.mark.parametrize(
    "relative",
    [
        "TK-PKA/x.md",  # -> <root>/TK-PKA/x.md contains "/TK-PKA/"
        "sub/TK-PKA/x.md",
        "01-knowledge-base/x.md",
        "sub/01-knowledge-base/deeper/x.md",
        "01-knowledge-base",  # bare directory name
    ],
)
def test_R18_denylist_refuses_forbidden_segments_even_inside_a_root(sandbox: Sandbox, relative: str):
    # These would pass containment (they are under data/inbox) — the denylist must still win.
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", relative)


def test_R18_denylist_refuses_project_root_inside_a_vault_path():
    # String only; /Users/someone does not exist on this machine.
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=Path("/Users/someone/TK-PKA/kg-mapper-poc"), roots=ROOTS)


def test_R18_denylist_refuses_project_root_inside_knowledge_base():
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=Path("/x/y/01-knowledge-base/poc"), roots=ROOTS)


def test_R18_denylist_refuses_a_configured_root_pointing_at_a_vault_segment(tmp_path: Path):
    bad = dict(ROOTS, inbox="TK-PKA/inbox")
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=tmp_path, roots=bad)


def test_R18_denylist_is_case_insensitive(sandbox: Sandbox):
    # macOS default filesystem is case-insensitive: "tk-pka" IS the vault. Guard must not be fooled.
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", "tk-pka/x.md")


# --------------------------------- rule 3: roots inside project directory


def test_R18_refuses_absolute_root_in_configuration(tmp_path: Path):
    bad = dict(ROOTS, inbox=str(tmp_path / "data/inbox"))  # absolute even though inside
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=tmp_path, roots=bad)


def test_R18_refuses_root_that_escapes_project_directory(tmp_path: Path):
    bad = dict(ROOTS, processed="../elsewhere/processed")
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=tmp_path, roots=bad)


def test_R18_refuses_missing_kind_in_roots(tmp_path: Path):
    incomplete = {k: v for k, v in ROOTS.items() if k != "runs"}
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=tmp_path, roots=incomplete)


def test_R18_sandbox_violation_is_a_distinct_exception_type():
    assert issubclass(SandboxViolation, Exception)
    assert SandboxViolation is not ValueError and SandboxViolation is not OSError


# ------------------------------------------ rule 3, physical: symlinked roots
#
# A configured root that is itself a symlink is lexically inside the project but
# physically elsewhere. Every path derived from it would be read/written outside
# the sandbox, so the guard must refuse — at construction, or failing that at
# root()/resolve() for that kind. Under no circumstances may a usable Path be
# handed out for that kind.


def _make_roots_on_disk(project_root: Path, roots: dict[str, str], *, except_kind: str) -> None:
    for kind, rel in roots.items():
        if kind != except_kind:
            (project_root / rel).mkdir(parents=True, exist_ok=True)
    (project_root / roots[except_kind]).parent.mkdir(parents=True, exist_ok=True)


def _assert_kind_root_refused(project_root: Path, roots: dict[str, str], kind: str) -> None:
    """Refusal at construction is preferred; otherwise root() AND resolve() must refuse."""
    try:
        sb = Sandbox(project_root=project_root, roots=roots)
    except SandboxViolation:
        return
    with pytest.raises(SandboxViolation):
        sb.root(kind)
    with pytest.raises(SandboxViolation):
        sb.resolve(kind, "x.md")


def test_R18_refuses_kind_root_that_is_a_symlink_outside_the_project(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    _make_roots_on_disk(tmp_path, ROOTS, except_kind="graph")
    os.symlink(elsewhere, tmp_path / "data/graph")  # data/graph -> <tmp>/elsewhere
    _assert_kind_root_refused(tmp_path, ROOTS, "graph")


def test_R18_refuses_kind_root_symlinked_into_a_vault_path(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory):
    # Real path of the target contains "/TK-PKA/" — name only, under a tmp dir; never the real vault.
    vaultish = tmp_path_factory.mktemp("vaultish") / "TK-PKA" / "02-works"
    vaultish.mkdir(parents=True)
    _make_roots_on_disk(tmp_path, ROOTS, except_kind="inbox")
    os.symlink(vaultish, tmp_path / "data/inbox")
    _assert_kind_root_refused(tmp_path, ROOTS, "inbox")


def test_R18_refuses_kind_root_symlinked_into_knowledge_base_dir(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory):
    # Third denylist entry mirrors ~/knowledge-base. Real vault dir under a tmp dir, never the real one.
    kbish = tmp_path_factory.mktemp("kbish") / "01-knowledge-base"
    kbish.mkdir(parents=True)
    _make_roots_on_disk(tmp_path, ROOTS, except_kind="processed")
    os.symlink(kbish, tmp_path / "data/processed")
    _assert_kind_root_refused(tmp_path, ROOTS, "processed")


# ------------------------------------ rule 2, third entry: ~/knowledge-base
#
# The third denylist entry is the second vault at ~/knowledge-base. HOME is
# redirected to a tmp dir so the real one is never touched; the implementation
# may expand `~` lazily (Path.home()/HOME) or match the `/knowledge-base/`
# segment — both are covered by the same fake-home layout.


@pytest.fixture
def fake_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_R18_denylist_refuses_kind_root_symlinked_into_home_knowledge_base(tmp_path: Path, fake_home: Path):
    target = fake_home / "knowledge-base" / "notes"
    target.mkdir(parents=True)
    _make_roots_on_disk(tmp_path, ROOTS, except_kind="graph")
    os.symlink(target, tmp_path / "data/graph")
    _assert_kind_root_refused(tmp_path, ROOTS, "graph")


def test_R18_denylist_refuses_project_root_inside_home_knowledge_base(fake_home: Path):
    # String only; nothing is created under the fake home.
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=fake_home / "knowledge-base" / "poc", roots=ROOTS)


def test_R18_denylist_refuses_file_symlink_into_home_knowledge_base(sandbox: Sandbox, tmp_path: Path, fake_home: Path):
    target = fake_home / "knowledge-base" / "note.md"
    target.parent.mkdir(parents=True)
    target.write_text("x", encoding="utf-8")
    os.symlink(target, tmp_path / "data/inbox/note.md")
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", "note.md")


# --------------------------------- rule 1, physical: symlinked files in a root


def test_R18_refuses_file_symlink_that_points_outside_the_project(sandbox: Sandbox, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.md").write_text("x", encoding="utf-8")
    os.symlink(outside / "secret.md", tmp_path / "data/inbox/leak.md")  # file, not directory
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", "leak.md")


def test_R18_refuses_file_symlink_that_lands_in_a_sibling_kind(sandbox: Sandbox, tmp_path: Path):
    # Inside the project, but outside the *inbox* root: still a containment breach.
    (tmp_path / "data/graph/nodes.md").write_text("x", encoding="utf-8")
    os.symlink(tmp_path / "data/graph/nodes.md", tmp_path / "data/inbox/alias.md")
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", "alias.md")


def test_R18_refuses_symlink_in_an_intermediate_directory(sandbox: Sandbox, tmp_path: Path, tmp_path_factory: pytest.TempPathFactory):
    outside = tmp_path_factory.mktemp("outside")
    (outside / "deep").mkdir()
    (outside / "deep/secret.md").write_text("x", encoding="utf-8")
    (tmp_path / "data/inbox/sub").mkdir()
    os.symlink(outside, tmp_path / "data/inbox/sub/hop")
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", "sub/hop/deep/secret.md")


# --------------------------------- rule 3: roots must be distinct and disjoint
#
# Two kinds sharing a directory, or one kind's root nested inside another's,
# means a write for one kind lands inside another kind's tree (e.g. processed
# originals appearing as inbox input). Refused at construction.


@pytest.mark.parametrize(
    "overrides",
    [
        {"converted": "data/inbox"},  # exact duplicate
        {"converted": "data/inbox/"},  # trailing slash alias
        {"converted": "data/./inbox"},  # dot-segment alias
        {"converted": "./data/inbox"},  # leading-dot alias
    ],
    ids=["exact", "trailing-slash", "dot-segment", "leading-dot"],
)
def test_R18_refuses_duplicate_roots(tmp_path: Path, overrides: dict[str, str]):
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=tmp_path, roots=dict(ROOTS, **overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"converted": "data/inbox/converted"},  # child inside a sibling kind
        {"inbox": "data"},  # parent of every other kind
        {"runs": "data/graph/runs/deep"},  # grandchild
    ],
    ids=["child", "parent-of-all", "grandchild"],
)
def test_R18_refuses_nested_roots(tmp_path: Path, overrides: dict[str, str]):
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=tmp_path, roots=dict(ROOTS, **overrides))


def test_R18_refuses_roots_made_duplicate_by_a_symlink(tmp_path: Path):
    # data/converted -> data/inbox: lexically distinct, physically the same directory.
    _make_roots_on_disk(tmp_path, ROOTS, except_kind="converted")
    os.symlink(tmp_path / "data/inbox", tmp_path / "data/converted")
    _assert_kind_root_refused(tmp_path, ROOTS, "converted")


def test_R18_accepts_five_distinct_sibling_roots(tmp_path: Path):
    # Control: the standard layout is not caught by the duplicate/nested check.
    sb = Sandbox(project_root=tmp_path, roots=ROOTS)
    assert len({sb.root(k) for k in KINDS}) == 5


# ------------------------------------------------ malformed input: NUL bytes
#
# A NUL byte can never be part of a real path; the OS layer raises ValueError
# on contact. The guard must convert that into SandboxViolation so callers have
# one exception to catch and no path is half-processed.


@pytest.mark.parametrize("relative", ["a\x00b.md", "\x00", "sub/\x00/x.md", "TK-PKA\x00/x.md"])
def test_R18_refuses_nul_byte_in_relative_with_sandbox_violation(sandbox: Sandbox, relative: str):
    with pytest.raises(SandboxViolation):
        sandbox.resolve("inbox", relative)


def test_R18_refuses_nul_byte_in_configured_root_with_sandbox_violation(tmp_path: Path):
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=tmp_path, roots=dict(ROOTS, inbox="data/in\x00box"))


def test_R18_refuses_nul_byte_in_project_root_with_sandbox_violation(tmp_path: Path):
    with pytest.raises(SandboxViolation):
        Sandbox(project_root=Path(str(tmp_path) + "\x00x"), roots=ROOTS)
