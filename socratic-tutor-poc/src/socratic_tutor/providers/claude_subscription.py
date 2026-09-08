"""Claude Agent SDK on the owner's existing `claude login` — the default provider.

No API key, no `ant auth login`, no provisioning: the SDK spawns the Claude Code
CLI, which reuses the subscription credentials already on the machine. This is
the path `~/src/kg-mapper-poc` proved out, and the isolation below is copied from
it deliberately rather than redesigned.

**Isolation is the reason this file is careful.** The SDK runs a real Claude Code
subprocess, which by default can read the filesystem, load `~/.claude` settings,
hooks, memory and `CLAUDE.md`, and leaves a session transcript on disk. For a
system whose entire premise is controlling exactly what the model sees, that is
unacceptable, so every call runs with:

- `tools=[]`, `allowed_tools=[]`, `skills=[]` — no filesystem, no shell, no skills;
- `setting_sources=[]`, `mcp_servers={}` — no `~/.claude` settings, hooks, memory
  or `CLAUDE.md` reaches the prompt;
- a **fresh, empty `cwd` per call**, removed afterwards. Per call rather than
  shared because the CLI derives the transcript directory name from the cwd, so a
  shared cwd gives concurrent calls the same name and cleanup races;
- an **environment allowlist**: everything outside `ENV_ALLOWLIST` is blanked, so
  a secret this repo gains tomorrow is excluded by default. The SDK builds
  `{**os.environ, **options.env}` and cannot remove an inherited variable, so the
  allowlist is expressed as explicit blanks;
- the CLI transcript deleted after every call, plus `sweep_orphans` at
  construction for anything a killed process left behind.

**Known residual risk — the transcript.** The CLI writes the prompt, which
contains student messages, to `<config>/projects/<slug>/` *before* we delete it.
The obvious fix, pointing `CLAUDE_CONFIG_DIR` at an ephemeral directory, was
tried and **breaks authentication** ("Not logged in"): the subscription
credential is resolved relative to the real config directory. So the write cannot
currently be prevented, only made short-lived and precisely reversible. For a
system handling minors' personal data this needs a product decision, not just a
code one — see the report accompanying this change.

Structured output (Calls A and B) needs no prompting hack: the SDK takes
`output_format={"type": "json_schema", ...}` natively, re-prompts the model
itself on a schema mismatch, and returns the parsed object on
`ResultMessage.structured_output`.

Cost: not billed per token. Rate limits are the subscription's rolling window,
which is why the eval suites run sequentially (Addendum §1).

**Latency: ~7 s per structured call, and do not try to optimise the subprocess.**
Measured live on Haiku 4.5 at `effort=low`: 9.0 s cold, 7.1 s warm (5.8-9.6 s),
against a Tech Spec §10 pre-stream budget of 800 ms for *all* of Calls A + B +
steps 2-4. The breakdown is what matters: only ~1.3-1.9 s is subprocess spawn,
while ~5.3-5.6 s is API time across `num_turns=2`.

So the cost is structural, not startup overhead. The Agent SDK is an **agent
harness**: it runs a loop of two model turns and wraps our ~600-token prompt in
~1,537 tokens of its own scaffolding, to produce a four-field JSON. Calls A and B
are not agentic tasks — classification and comparison are single-shot — so we are
paying for a loop we do not want. Warming, pooling or pre-spawning cannot recover
it; even at zero spawn cost this path stays ~5.4 s.

Consequence: this provider is a **development and evaluation convenience**,
chosen because it needs zero provisioning, and it carries roughly a 10x latency
cost. The 800 ms target was written against direct HTTP calls and remains the
right target for `anthropic` and `openai_compat`. Quote it in a pilot document
only alongside which path it describes.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
import unicodedata
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from socratic_tutor.providers.base import (
    LLMProvider,
    Msg,
    ProviderConfigError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimited,
    ProviderResponseError,
    StreamStats,
    StructuredResult,
    Usage,
    strict_json_schema,
)

log = logging.getLogger(__name__)

PROVIDER = "claude_subscription"
AGENT_CWD_NAME = "agent-runs"
MAX_TURNS = 3
STRUCTURED_OUTPUT_EXHAUSTED = "error_max_structured_output_retries"
CLAUDE_CONFIG_DIR_VAR = "CLAUDE_CONFIG_DIR"

#: The only parent-environment variables the CLI subprocess may see. This is an
#: **allowlist, not a denylist**, and that is the whole point: a denylist has to
#: be updated every time the application gains a secret, and this repo already
#: has one a denylist of provider keys would miss (`POC_OPENAI_COMPAT_API_KEY`).
#: Anything not named here is blanked, so a secret added tomorrow is excluded by
#: default rather than by someone remembering.
#:
#: Verified live: a call with 62 inherited variables blanked and these 8 kept
#: authenticates and succeeds.
ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        # process basics the CLI and Node need to start at all
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "TMP",
        "TEMP",
        # locale, or Node warns and Thai text can mis-encode
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        # Node and TLS trust, for proxied or MITM-inspected networks
        "NODE_OPTIONS",
        "NODE_EXTRA_CA_CERTS",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    }
)


#: Always blanked, present in the parent or not — the credentials whose leakage
#: would be worst, named explicitly so the guarantee is unconditional.
ALWAYS_BLANKED: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "POC_OPENAI_COMPAT_API_KEY",
)


def subprocess_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Blank every parent variable outside `ENV_ALLOWLIST`.

    The SDK spawns with `{**os.environ, **options.env}` and offers no way to
    *remove* an inherited variable, so an allowlist has to be expressed as
    explicit blanks for everything else. Verified against the installed
    transport.
    """
    source = os.environ if environ is None else environ
    blanked = {name: "" for name in source if name not in ENV_ALLOWLIST}
    # Blanked whether or not the parent has them, so the guarantee does not depend
    # on the ambient environment and a test can assert it unconditionally.
    blanked.update(dict.fromkeys(ALWAYS_BLANKED, ""))
    return blanked


PROJECT_SLUG_MAX_LEN = 200
_SLUG_KEEP = re.compile(r"[a-zA-Z0-9]")
_BASE36 = "0123456789abcdefghijklmnopqrstuvwxyz"


# ------------------------------------------------------ transcript cleanup
# Slug scheme read from the CLI bundled with claude-agent-sdk 0.2.144: every
# character outside [a-zA-Z0-9] in the cwd path becomes '-'; a slug over 200
# characters is cut to 200 and suffixed '-<base36(abs(java_string_hash(path)))>'.
# Ported from kg-mapper-poc, which established it.


def _java_string_hash(text: str) -> int:
    """`(h << 5) - h + charCode | 0` over UTF-16 code units, as the CLI computes it."""
    h = 0
    units = text.encode("utf-16-le")
    for i in range(0, len(units), 2):
        h = (h * 31 + int.from_bytes(units[i : i + 2], "little")) & 0xFFFFFFFF
    return h - 0x100000000 if h >= 0x80000000 else h


def _base36(n: int) -> str:
    if n == 0:
        return "0"
    digits: list[str] = []
    while n:
        n, r = divmod(n, 36)
        digits.append(_BASE36[r])
    return "".join(reversed(digits))


def project_slug(cwd: str) -> str:
    """The directory name the CLI uses under `<config>/projects/` for a session in `cwd`."""
    text = unicodedata.normalize("NFC", cwd)
    units = text.encode("utf-16-le")
    chars = [chr(int.from_bytes(units[i : i + 2], "little")) for i in range(0, len(units), 2)]
    slug = "".join(c if _SLUG_KEEP.fullmatch(c) else "-" for c in chars)
    if len(slug) <= PROJECT_SLUG_MAX_LEN:
        return slug
    return f"{slug[:PROJECT_SLUG_MAX_LEN]}-{_base36(abs(_java_string_hash(text)))}"


def claude_config_dir() -> Path:
    override = os.environ.get(CLAUDE_CONFIG_DIR_VAR, "").strip()
    return Path(override) if override else Path.home() / ".claude"


def transcript_dirs(cwd: Path) -> list[Path]:
    """Transcript directories for `cwd` — the literal path and its realpath when they differ."""
    projects = claude_config_dir() / "projects"
    candidates = dict.fromkeys([str(cwd), os.path.realpath(cwd)])
    return [projects / project_slug(c) for c in candidates]


def cleanup_transcripts(cwd: Path) -> list[Path]:
    """Delete the CLI transcript directory for this sandbox `cwd` only.

    Best-effort: a missing directory is not an error and an `OSError` is logged
    rather than raised, because failing a tutoring turn over transcript hygiene
    would be the wrong trade. Never touches anything outside
    `<config>/projects/<slug-of-cwd>/`.
    """
    removed: list[Path] = []
    projects = claude_config_dir() / "projects"
    for target in transcript_dirs(cwd):
        if target.parent != projects or not target.name or target.name in (".", ".."):
            log.warning("claude_subscription: refusing to clean outside %s: %s", projects, target)
            continue
        if not target.is_dir():
            continue
        try:
            shutil.rmtree(target)
            removed.append(target)
        except OSError as exc:
            log.warning(
                "claude_subscription: could not remove transcript dir %s: %s",
                target,
                exc.strerror or exc.__class__.__name__,
            )
    return removed


def runs_root(data_dir: Path) -> Path:
    """Parent of the per-call sandbox directories, inside the application's data dir."""
    path = data_dir / AGENT_CWD_NAME
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProviderConfigError(
            f"agent runs directory {path}: {exc.strerror or exc.__class__.__name__}"
        ) from exc
    return path


@contextmanager
def run_directory(root: Path) -> Iterator[Path]:
    """A fresh, empty cwd for exactly one call, removed with its transcript after.

    Per-call rather than shared, for two reasons. The CLI derives the transcript
    directory name from the cwd path, so a shared cwd gives concurrent calls the
    same slug and cleaning up after one deletes another's in-flight transcript.
    And a directory checked empty once at construction says nothing about what is
    in it an hour later in a long-running server; a directory created per call is
    empty by construction, every call, with nothing to re-check.
    """
    cwd = Path(tempfile.mkdtemp(dir=root, prefix="run-"))
    try:
        yield cwd
    finally:
        cleanup_transcripts(cwd)
        shutil.rmtree(cwd, ignore_errors=True)


def sweep_orphans(root: Path) -> int:
    """Remove run directories and transcripts left by a crashed or killed process.

    `run_directory` cleans up in a `finally`, which a SIGKILL does not run. This
    runs at construction so a restart clears anything the last process left in
    `~/.claude/projects/`.
    """
    swept = 0
    for leftover in sorted(root.glob("run-*")):
        if not leftover.is_dir():
            continue
        cleanup_transcripts(leftover)
        shutil.rmtree(leftover, ignore_errors=True)
        swept += 1
    if swept:
        log.info("claude_subscription: swept %d orphaned run director(ies)", swept)
    return swept


# --------------------------------------------------------------- provider


class ClaudeSubscriptionProvider(LLMProvider):
    """Calls Claude through the Agent SDK on the machine's existing `claude login`."""

    name = PROVIDER

    def __init__(
        self,
        *,
        data_dir: Path,
        effort: str = "low",
        query_fn: Callable[..., AsyncIterator[Any]] | None = None,
    ) -> None:
        self._effort = effort
        self._query_fn = query_fn
        self._runs_root = runs_root(data_dir)
        sweep_orphans(self._runs_root)

    def __repr__(self) -> str:
        return (
            f"ClaudeSubscriptionProvider(effort={self._effort!r}, "
            f"runs_root={str(self._runs_root)!r})"
        )

    # ------------------------------------------------------------- options

    def _options(
        self, system: str, *, cwd: Path, schema: dict[str, Any] | None, streaming: bool
    ) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        options: dict[str, Any] = {
            "system_prompt": system,
            "tools": [],
            "allowed_tools": [],
            # Explicit empty list: None would leave the CLI's own defaults in force.
            "skills": [],
            "setting_sources": [],
            "mcp_servers": {},
            "env": subprocess_env(),
            "permission_mode": None,
            "max_turns": MAX_TURNS,
            "cwd": str(cwd),
            "continue_conversation": False,
            "resume": None,
        }
        if schema is not None:
            options["output_format"] = {"type": "json_schema", "schema": schema}
            options["effort"] = self._effort
        if streaming:
            options["include_partial_messages"] = True
        return ClaudeAgentOptions(**options)

    def _query(self) -> Callable[..., AsyncIterator[Any]]:
        if self._query_fn is not None:
            return self._query_fn
        import claude_agent_sdk

        return claude_agent_sdk.query

    @staticmethod
    def _prompt(messages: Sequence[Msg]) -> Any:
        """Collapse the turn list into the single user message the SDK receives."""
        parts = [f"{m.role}: {m.content}" if m.role == "assistant" else m.content for m in messages]
        text = "\n\n".join(parts).strip()
        if not text:
            raise ProviderError("claude_subscription: no user turn to send")

        async def stream() -> AsyncIterator[dict[str, Any]]:
            yield {
                "type": "user",
                "message": {"role": "user", "content": [{"type": "text", "text": text}]},
            }

        return stream()

    # ------------------------------------------------------------- calls

    async def complete_structured(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Msg],
        schema: type[BaseModel],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> StructuredResult:
        """Calls A and B. `max_tokens` and `temperature` are accepted for interface
        parity and ignored: the SDK exposes neither, and output length is bounded
        by the schema."""
        from claude_agent_sdk import ResultMessage

        started = time.perf_counter()
        result: Any = None
        with run_directory(self._runs_root) as cwd:
            options = self._options(
                system, cwd=cwd, schema=strict_json_schema(schema), streaming=False
            )
            async for message in self._run(options, messages, model):
                if isinstance(message, ResultMessage):
                    result = message
        latency_ms = (time.perf_counter() - started) * 1000.0

        if result is None:
            raise ProviderResponseError("claude_subscription: stream ended without a result")
        self._raise_for_result(result, model)

        payload = result.structured_output
        if payload is None:
            raise ProviderResponseError(
                "claude_subscription: the SDK returned success but no structured_output"
            )
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ProviderResponseError(
                    f"claude_subscription: structured_output was not valid JSON: {exc}"
                ) from exc
        try:
            value = schema.model_validate(payload)
        except ValidationError as exc:
            raise ProviderResponseError(
                f"claude_subscription: output does not match {schema.__name__}: {exc}"
            ) from exc

        return StructuredResult(
            value=value,
            usage=_usage_of(result),
            latency_ms=latency_ms,
            model=_model_served(result) or model,
            provider=PROVIDER,
        )

    async def stream_text(
        self,
        *,
        model: str,
        system: str,
        messages: Sequence[Msg],
        max_tokens: int = 2000,
        stats: StreamStats | None = None,
    ) -> AsyncIterator[str]:
        """Call C. Text deltas come from `StreamEvent`, whose `event` is the raw
        Anthropic stream event, so `content_block_delta` carries the text."""
        from claude_agent_sdk import ResultMessage, StreamEvent

        if stats is not None:
            stats.model, stats.provider = model, PROVIDER
        started = time.perf_counter()
        try:
            with run_directory(self._runs_root) as cwd:
                options = self._options(system, cwd=cwd, schema=None, streaming=True)
                async for message in self._run(options, messages, model):
                    if isinstance(message, StreamEvent):
                        delta = _delta_text(message.event)
                        if not delta:
                            continue
                        if stats is not None and stats.ttft_ms is None:
                            stats.ttft_ms = (time.perf_counter() - started) * 1000.0
                        yield delta
                    elif isinstance(message, ResultMessage):
                        self._raise_for_result(message, model)
                        if stats is not None:
                            stats.usage = _usage_of(message)
                            stats.stop_reason = getattr(message, "stop_reason", None)
                            stats.model = _model_served(message) or model
        finally:
            if stats is not None:
                stats.total_ms = (time.perf_counter() - started) * 1000.0

    async def _run(self, options: Any, messages: Sequence[Msg], model: str) -> AsyncIterator[Any]:
        """Drive one query, translating SDK failures into this package's errors."""
        from claude_agent_sdk import (
            ClaudeSDKError,
            CLIConnectionError,
            CLINotFoundError,
            RateLimitEvent,
        )

        options.model = model
        try:
            async for message in self._query()(prompt=self._prompt(messages), options=options):
                if isinstance(message, RateLimitEvent):
                    info = message.rate_limit_info
                    if getattr(info, "status", None) == "rejected":
                        raise ProviderRateLimited(
                            f"claude_subscription: subscription rate limit rejected the call "
                            f"(type={getattr(info, 'rate_limit_type', None)!r}, "
                            f"resets_at={getattr(info, 'resets_at', None)!r})"
                        )
                    continue
                yield message
        except CLINotFoundError as exc:
            raise ProviderConfigError(
                "claude_subscription: the Claude Code CLI is not installed or not on PATH. "
                "Install Claude Code and run `claude login` once."
            ) from exc
        except CLIConnectionError as exc:
            raise ProviderConnectionError(f"claude_subscription: {exc}") from exc
        except ClaudeSDKError as exc:
            raise ProviderError(f"claude_subscription: {exc.__class__.__name__}: {exc}") from exc

    @staticmethod
    def _raise_for_result(result: Any, model: str) -> None:
        if not getattr(result, "is_error", False) and getattr(result, "subtype", None) == "success":
            return
        subtype = getattr(result, "subtype", None)
        if subtype == STRUCTURED_OUTPUT_EXHAUSTED:
            raise ProviderResponseError(
                f"claude_subscription: the SDK exhausted its structured-output retries on {model!r}"
            )
        if getattr(result, "api_error_status", None) == 429:
            raise ProviderRateLimited(f"claude_subscription: rate limited on {model!r}")
        detail = "; ".join(getattr(result, "errors", None) or []) or getattr(result, "result", "")
        raise ProviderError(
            f"claude_subscription: {subtype or 'failed'} on {model!r}: {detail or 'no detail'}"
        )

    def cleanup(self) -> int:
        """Sweep orphaned run directories. Already done per call and at construction."""
        return sweep_orphans(self._runs_root)


def _delta_text(event: dict[str, Any]) -> str:
    if event.get("type") != "content_block_delta":
        return ""
    delta = event.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return ""
    text = delta.get("text")
    return text if isinstance(text, str) else ""


def _usage_of(result: Any) -> Usage:
    usage = getattr(result, "usage", None) or {}
    if not isinstance(usage, dict):
        return Usage()
    return Usage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )


def _model_served(result: Any) -> str | None:
    """First key of `ResultMessage.model_usage`, which is keyed by model ID."""
    model_usage = getattr(result, "model_usage", None)
    if isinstance(model_usage, dict) and model_usage:
        return str(next(iter(model_usage)))
    return None


__all__ = [
    "AGENT_CWD_NAME",
    "ALWAYS_BLANKED",
    "ENV_ALLOWLIST",
    "PROVIDER",
    "ClaudeSubscriptionProvider",
    "claude_config_dir",
    "cleanup_transcripts",
    "project_slug",
    "run_directory",
    "runs_root",
    "subprocess_env",
    "sweep_orphans",
    "transcript_dirs",
]
