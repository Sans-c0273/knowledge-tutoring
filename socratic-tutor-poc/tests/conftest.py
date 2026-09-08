"""Shared test isolation.

**Why this exists.** Chroma keeps a course collection in a single SQLite file
under `data/chroma/`. Any test that reaches the store without an explicit
`persist_dir` opens the *repository's* database, so two `pytest` processes at
once — routine here, several agents share this tree — block on the same SQLite
write lock. That wait happens inside SQLite's C code, where `pytest-timeout`'s
SIGALRM cannot interrupt it, so the suite appears to hang forever with no
traceback rather than failing. It also left ~70 stray files in `data/uploads`
from ingestion-API tests.

So every test run gets its own `POC_DATA_DIR`, **and its own copy of
`content/`**. The copy carries the real inputs the suite is supposed to read
(rule tables, syllabus, seed map, seed corpus) but not the runtime-written
directories — `content/klmap/` (published live maps) and `content/drafts/`
(extraction drafts). Both are produced by ordinary use of the product on this
machine and both are gitignored, so leaving `content_dir` pointed at the
repository meant a developer who had approved one upload got a suite that
resolved topics against their live map instead of the seed: green in CI, red
on the laptop, nothing in `git status` to explain it (code review 2026-09-02,
M1). A test that needs to *write* content still redirects `POC_CONTENT_DIR`
itself; `monkeypatch.setenv` restores this session-wide value afterwards.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from socratic_tutor.api import chat_routes
from socratic_tutor.config import get_settings
from socratic_tutor.domain.rag import store

REPO_CONTENT = Path(__file__).resolve().parents[1] / "content"

#: Written at runtime by approval and extraction; never inputs to a test that
#: did not create them.
RUNTIME_CONTENT_DIRS = ("klmap", "drafts")


@pytest.fixture(scope="session", autouse=True)
def isolated_dirs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point runtime state and content at this run's temp dir."""
    data_dir = tmp_path_factory.mktemp("poc-data")
    content_dir = tmp_path_factory.mktemp("poc-content") / "content"
    shutil.copytree(REPO_CONTENT, content_dir, ignore=shutil.ignore_patterns(*RUNTIME_CONTENT_DIRS))

    previous = {name: os.environ.get(name) for name in ("POC_DATA_DIR", "POC_CONTENT_DIR")}
    os.environ["POC_DATA_DIR"] = str(data_dir)
    os.environ["POC_CONTENT_DIR"] = str(content_dir)
    _reset_caches()

    yield data_dir

    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _reset_caches()


def _reset_caches() -> None:
    """Everything that memoises a path derived from the settings."""
    get_settings.cache_clear()
    store.reset_client_cache()
    chat_routes._default_dependencies.cache_clear()
    chat_routes._live_courses.clear()
    chat_routes._rejected_live.clear()
