"""The Claude Agent SDK provider — isolation, parsing, cleanup. No subprocess spawned.

The isolation assertions are the important ones. This provider runs a real Claude
Code subprocess, which by default can read the filesystem and load `~/.claude`
settings, hooks, memory and `CLAUDE.md`. For a system whose premise is
controlling exactly what the model sees, each of those is a hole, so each is
asserted rather than assumed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ResultMessage, StreamEvent

from socratic_tutor.models.enums import Intent, LearnerState
from socratic_tutor.models.intent import IntentClassification, SpecialHandlingResult
from socratic_tutor.providers.base import (
    Msg,
    ProviderError,
    ProviderRateLimited,
    ProviderResponseError,
    StreamStats,
)
from socratic_tutor.providers.claude_subscription import (
    ENV_ALLOWLIST,
    ClaudeSubscriptionProvider,
    cleanup_transcripts,
    project_slug,
    subprocess_env,
    sweep_orphans,
    transcript_dirs,
)

MESSAGES = [Msg(role="user", content="What is a linear equation?")]

WIRE = IntentClassification(
    intent=Intent.EXPLAIN,
    learner_state=LearnerState.NORMAL,
    special_handling=SpecialHandlingResult(),
    confidence=0.9,
)


def result_message(**overrides: Any) -> ResultMessage:
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 10,
        "duration_api_ms": 8,
        "is_error": False,
        "num_turns": 1,
        "session_id": "sess",
        "total_cost_usd": None,
        "usage": {"input_tokens": 120, "output_tokens": 18},
        "result": "",
        "structured_output": WIRE.model_dump(mode="json"),
        "model_usage": {"claude-haiku-4-5": {}},
    }
    fields.update(overrides)
    return ResultMessage(**{k: v for k, v in fields.items() if k in ResultMessage.__annotations__})


def fake_query(*messages: Any) -> Any:
    """A `query` stand-in that records its options and replays canned messages."""
    seen: dict[str, Any] = {}

    def query(*, prompt: Any, options: Any) -> AsyncIterator[Any]:
        seen["options"] = options

        async def stream() -> AsyncIterator[Any]:
            seen["prompt"] = [chunk async for chunk in prompt]
            for message in messages:
                yield message

        return stream()

    query.seen = seen  # type: ignore[attr-defined]
    return query


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


def provider_with(data_dir: Path, *messages: Any) -> tuple[ClaudeSubscriptionProvider, Any]:
    query = fake_query(*messages)
    return ClaudeSubscriptionProvider(data_dir=data_dir, query_fn=query), query


# ------------------------------------------------------------- isolation


async def test_the_subprocess_gets_no_tools_settings_skills_or_mcp(data_dir: Path) -> None:
    """Every default capability of a Claude Code session is switched off."""
    provider, query = provider_with(data_dir, result_message())
    await provider.complete_structured(
        model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
    )
    options = query.seen["options"]

    assert options.tools == []
    assert options.allowed_tools == []
    # Explicit empty list, not None: None leaves the CLI's own defaults in force.
    assert options.skills == []
    assert options.setting_sources == []
    assert options.mcp_servers == {}
    assert options.permission_mode is None
    assert options.continue_conversation is False
    assert options.resume is None


async def test_the_subprocess_environment_is_an_allowlist(data_dir: Path) -> None:
    """Everything not on the allowlist is blanked, so a new secret is excluded by default.

    The SDK builds `{**os.environ, **options.env}` and cannot remove an inherited
    variable, so an allowlist has to be expressed as explicit blanks. This is what
    makes it safe not to refuse to start when a provider key is set on the machine:
    the subprocess cannot see it.
    """
    provider, query = provider_with(data_dir, result_message())
    await provider.complete_structured(
        model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
    )
    env = query.seen["options"].env
    for name in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        assert env.get(name) == ""
    assert "PATH" not in env  # allowlisted, so left inherited rather than blanked


def test_this_repos_own_secret_is_blanked_by_the_allowlist() -> None:
    """A denylist of provider keys would have missed `POC_OPENAI_COMPAT_API_KEY`."""
    fake_environ = {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "POC_OPENAI_COMPAT_API_KEY": "sk-secret",
        "AWS_SECRET_ACCESS_KEY": "also-secret",
        "SOME_FUTURE_TOKEN": "not-yet-invented",
    }
    env = subprocess_env(fake_environ)
    assert env["POC_OPENAI_COMPAT_API_KEY"] == ""
    assert env["AWS_SECRET_ACCESS_KEY"] == ""
    assert env["SOME_FUTURE_TOKEN"] == ""
    assert "PATH" not in env
    assert "HOME" not in env


def test_the_allowlist_keeps_only_process_basics() -> None:
    """Nothing resembling a credential may be on the allowlist."""
    assert not any(
        marker in name.upper()
        for name in ENV_ALLOWLIST
        for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    )


async def test_each_call_gets_its_own_empty_directory(data_dir: Path) -> None:
    """Empty by construction every call, rather than checked empty once at startup."""
    provider, query = provider_with(data_dir, result_message())

    seen: list[Path] = []
    for _ in range(2):
        await provider.complete_structured(
            model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
        )
        seen.append(Path(query.seen["options"].cwd))

    assert seen[0] != seen[1], "a shared cwd gives concurrent calls the same transcript slug"
    for cwd in seen:
        assert cwd.is_relative_to(data_dir)
        assert not cwd.exists(), "the run directory is removed after the call"


async def test_the_run_directory_is_removed_even_when_the_call_fails(data_dir: Path) -> None:
    provider, query = provider_with(data_dir)  # no ResultMessage -> raises
    with pytest.raises(ProviderResponseError):
        await provider.complete_structured(
            model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
        )
    assert not Path(query.seen["options"].cwd).exists()


def test_orphans_from_a_killed_process_are_swept(data_dir: Path) -> None:
    """A SIGKILL skips the cleanup; a restart must not inherit the leftovers."""
    root = data_dir / "agent-runs"
    root.mkdir(parents=True)
    orphan = root / "run-abandoned"
    orphan.mkdir()
    (orphan / "stale").write_text("x", encoding="utf-8")

    assert sweep_orphans(root) == 1
    assert not orphan.exists()


def test_construction_sweeps_orphans(data_dir: Path) -> None:
    root = data_dir / "agent-runs"
    root.mkdir(parents=True)
    (root / "run-abandoned").mkdir()

    ClaudeSubscriptionProvider(data_dir=data_dir, query_fn=fake_query())
    assert not (root / "run-abandoned").exists()


async def test_the_transcript_is_deleted_after_every_call(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI writes the student's message under `<config>/projects/<slug>/`.

    That is personal data outside the application's data directory, so it must not
    survive the call. Simulated by planting a transcript at the slug the CLI would
    use for this call's cwd, which the fake query cannot do for itself.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))

    planted: list[Path] = []

    def planting_query(*, prompt: Any, options: Any) -> AsyncIterator[Any]:
        for directory in transcript_dirs(Path(options.cwd)):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "session.jsonl").write_text(
                '{"prompt": "What is 2x + 4 = 10?"}', encoding="utf-8"
            )
            planted.append(directory)

        async def stream() -> AsyncIterator[Any]:
            yield result_message()

        return stream()

    provider = ClaudeSubscriptionProvider(data_dir=data_dir, query_fn=planting_query)
    await provider.complete_structured(
        model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
    )

    assert planted, "the test must actually plant a transcript"
    assert not any(directory.exists() for directory in planted)


def test_cleanup_refuses_to_delete_outside_the_projects_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "config"))
    victim = tmp_path / "not-a-transcript"
    victim.mkdir()
    assert cleanup_transcripts(victim) == []
    assert victim.exists()


def test_project_slug_matches_the_cli_scheme() -> None:
    assert project_slug("/Users/x/src/poc") == "-Users-x-src-poc"
    assert project_slug("/a b/c.d") == "-a-b-c-d"
    long = "/" + "x" * 300
    slug = project_slug(long)
    assert len(slug) > 200 and slug.startswith("-" + "x" * 199)


# --------------------------------------------------------- structured call


async def test_structured_output_uses_the_sdk_json_schema_mode(data_dir: Path) -> None:
    """No prompt-and-retry hack: the SDK takes the schema natively and re-prompts itself."""
    provider, query = provider_with(data_dir, result_message())
    result = await provider.complete_structured(
        model="claude-haiku-4-5",
        system="classify this",
        messages=MESSAGES,
        schema=IntentClassification,
    )
    options = query.seen["options"]

    assert options.output_format["type"] == "json_schema"
    schema = options.output_format["schema"]
    assert set(schema["required"]) == {
        "intent",
        "learner_state",
        "special_handling",
        "confidence",
    }
    assert schema["additionalProperties"] is False
    assert options.system_prompt == "classify this"
    assert options.model == "claude-haiku-4-5"

    assert isinstance(result.value, IntentClassification)
    assert result.value.intent is Intent.EXPLAIN
    assert result.usage.input_tokens == 120
    assert result.model == "claude-haiku-4-5"
    assert result.provider == "claude_subscription"
    assert result.latency_ms >= 0.0


async def test_structured_output_accepts_a_json_string(data_dir: Path) -> None:
    payload = json.dumps(WIRE.model_dump(mode="json"))
    provider, _ = provider_with(data_dir, result_message(structured_output=payload))
    result = await provider.complete_structured(
        model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
    )
    assert result.value == WIRE


async def test_output_that_does_not_match_the_schema_is_an_error(data_dir: Path) -> None:
    provider, _ = provider_with(data_dir, result_message(structured_output={"intent": "nope"}))
    with pytest.raises(ProviderResponseError, match="does not match IntentClassification"):
        await provider.complete_structured(
            model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
        )


async def test_exhausted_structured_output_retries_is_a_named_error(data_dir: Path) -> None:
    """The SDK re-prompts on mismatch; when it gives up, say so rather than guess."""
    provider, _ = provider_with(
        data_dir,
        result_message(
            subtype="error_max_structured_output_retries", is_error=True, structured_output=None
        ),
    )
    with pytest.raises(ProviderResponseError, match="exhausted its structured-output retries"):
        await provider.complete_structured(
            model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
        )


async def test_a_missing_result_message_is_an_error(data_dir: Path) -> None:
    provider, _ = provider_with(data_dir)
    with pytest.raises(ProviderResponseError, match="without a result"):
        await provider.complete_structured(
            model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
        )


async def test_an_api_429_becomes_a_rate_limit_error(data_dir: Path) -> None:
    provider, _ = provider_with(
        data_dir,
        result_message(subtype="error", is_error=True, structured_output=None),
    )
    message = result_message(subtype="error", is_error=True, structured_output=None)
    message.api_error_status = 429  # type: ignore[attr-defined]
    provider, _ = provider_with(data_dir, message)
    with pytest.raises(ProviderRateLimited):
        await provider.complete_structured(
            model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
        )


async def test_a_failed_result_carries_its_detail(data_dir: Path) -> None:
    provider, _ = provider_with(
        data_dir,
        result_message(subtype="error_during_execution", is_error=True, structured_output=None),
    )
    with pytest.raises(ProviderError, match="error_during_execution"):
        await provider.complete_structured(
            model="claude-haiku-4-5", system="s", messages=MESSAGES, schema=IntentClassification
        )


# ------------------------------------------------------------- streaming


def stream_event(text: str) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id="s",
        event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
    )


async def test_streaming_yields_text_deltas_and_records_usage(data_dir: Path) -> None:
    provider, query = provider_with(
        data_dir,
        stream_event("Let's "),
        stream_event("work through it."),
        result_message(structured_output=None),
    )
    stats = StreamStats()
    chunks = [
        delta
        async for delta in provider.stream_text(
            model="claude-sonnet-5", system="teach", messages=MESSAGES, stats=stats
        )
    ]

    assert "".join(chunks) == "Let's work through it."
    assert query.seen["options"].include_partial_messages is True
    assert query.seen["options"].output_format is None
    assert stats.usage.input_tokens == 120
    assert stats.ttft_ms is not None
    assert stats.total_ms is not None
    assert stats.provider == "claude_subscription"


async def test_streaming_ignores_non_text_events(data_dir: Path) -> None:
    noise = StreamEvent(uuid="u", session_id="s", event={"type": "message_start"})
    provider, _ = provider_with(
        data_dir, noise, stream_event("hi"), result_message(structured_output=None)
    )
    chunks = [
        delta
        async for delta in provider.stream_text(
            model="claude-sonnet-5", system="teach", messages=MESSAGES
        )
    ]
    assert chunks == ["hi"]


# ------------------------------------------------------------- the prompt


async def test_the_conversation_collapses_into_one_user_message(data_dir: Path) -> None:
    provider, query = provider_with(data_dir, result_message())
    await provider.complete_structured(
        model="claude-haiku-4-5",
        system="s",
        messages=[
            Msg(role="user", content="How do I start?"),
            Msg(role="assistant", content="Subtract 4 from both sides."),
            Msg(role="user", content="ok what next"),
        ],
        schema=IntentClassification,
    )
    sent = query.seen["prompt"]
    assert len(sent) == 1
    text = sent[0]["message"]["content"][0]["text"]
    assert "How do I start?" in text
    assert "assistant: Subtract 4 from both sides." in text
    assert text.endswith("ok what next")


async def test_an_empty_prompt_is_refused(data_dir: Path) -> None:
    provider, _ = provider_with(data_dir, result_message())
    with pytest.raises(ProviderError, match="no user turn"):
        await provider.complete_structured(
            model="claude-haiku-4-5",
            system="s",
            messages=[Msg(role="user", content="   ")],
            schema=IntentClassification,
        )


# ----------------------------------------------------- the serve bind guard


def test_loopback_binds_are_allowed() -> None:
    from socratic_tutor.cli import check_bind

    for host in ("127.0.0.1", "::1", "localhost"):
        check_bind(host, {})


def test_a_public_bind_is_refused_because_there_is_no_auth() -> None:
    from socratic_tutor.cli import check_bind

    with pytest.raises(SystemExit, match="no authentication"):
        check_bind("0.0.0.0", {})
    with pytest.raises(SystemExit):
        check_bind("192.168.1.10", {})


def test_a_public_bind_can_be_opted_into_deliberately() -> None:
    from socratic_tutor.cli import ALLOW_PUBLIC_BIND_VAR, check_bind

    check_bind("0.0.0.0", {ALLOW_PUBLIC_BIND_VAR: "1"})


# ------------------------------------------- M10: path traversal (review fix)


@pytest.mark.parametrize("segment", ["..", ".", "../etc", "../../etc/passwd", ""])
def test_traversal_segments_are_rejected(segment: str) -> None:
    """`.` is in the allowed character set, so `..` passed the string check."""
    from socratic_tutor.pedagogy.student_model import StudentModelError, safe_path_segment

    with pytest.raises(StudentModelError):
        safe_path_segment(segment, "course_id")


def test_ordinary_ids_still_work() -> None:
    from socratic_tutor.pedagogy.student_model import safe_path_segment

    for value in ("MATH-101", "BIO101.v2", "student_1", "a.b-c_d"):
        assert safe_path_segment(value, "course_id") == value


def test_resolve_within_refuses_to_leave_the_root(tmp_path: Path) -> None:
    """The containment check, not the string check, is what guards the filesystem."""
    from socratic_tutor.pedagogy.student_model import StudentModelError, resolve_within

    root = tmp_path / "data"
    root.mkdir()
    assert resolve_within(root, "course", "s.json").is_relative_to(root.resolve())

    with pytest.raises(StudentModelError, match="outside the data root"):
        resolve_within(root, "..", "escaped.json")


def test_resolve_within_catches_a_symlink_out_of_the_root(tmp_path: Path) -> None:
    """A symlink defeats every string-only check; resolving the path does not."""
    from socratic_tutor.pedagogy.student_model import StudentModelError, resolve_within

    root = tmp_path / "data"
    root.mkdir()
    (tmp_path / "outside").mkdir()
    (root / "course").symlink_to(tmp_path / "outside", target_is_directory=True)

    with pytest.raises(StudentModelError, match="outside the data root"):
        resolve_within(root, "course", "s.json")
