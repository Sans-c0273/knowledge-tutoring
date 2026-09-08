"""`poc` command line — the POC's one entry point (pyproject `[project.scripts]`).

`poc serve` runs the API and, when `web/dist` exists, the SPA with it, on
localhost only (Addendum §"Repo & stack": no auth, single user). Reload is off
by default because it restarts the process mid-ingestion.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poc", description="Socratic AI Tutor POC")
    subcommands = parser.add_subparsers(dest="command", required=True)

    serve = subcommands.add_parser("serve", help="run the API and the web UI")
    serve.add_argument("--host", default=DEFAULT_HOST, help=f"default {DEFAULT_HOST}")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    serve.add_argument(
        "--reload",
        action="store_true",
        help="restart on code changes; interrupts running ingestions",
    )
    serve.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
    )
    return parser


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

ALLOW_PUBLIC_BIND_VAR = "POC_ALLOW_PUBLIC_BIND"


def check_bind(host: str, environ: Mapping[str, str] | None = None) -> None:
    """Refuse a non-loopback bind, because this application has no authentication.

    `--host 0.0.0.0` here publishes the chat endpoint, the ingestion endpoint and
    the glass-box inspector — which renders session content — to every machine on
    the network, with no auth and no rate limit. The POC is specified as
    localhost, single user (Addendum §"Repo & stack"), so a non-loopback bind is
    far more likely a mistake than an intent, and it should fail loudly rather
    than appear to work.

    `POC_ALLOW_PUBLIC_BIND=1` is the deliberate override for someone who has read
    that and means it.
    """
    source = os.environ if environ is None else environ
    if host in LOOPBACK_HOSTS or source.get(ALLOW_PUBLIC_BIND_VAR, "").strip():
        return
    raise SystemExit(
        f"refusing to bind {host}: this application has no authentication and no rate "
        f"limit, so a non-loopback bind exposes chat, ingestion and the inspector "
        f"(which renders session content) to the whole network.\n"
        f"Bind {DEFAULT_HOST} instead, or set {ALLOW_PUBLIC_BIND_VAR}=1 if you mean it."
    )


def serve(host: str, port: int, reload: bool, log_level: str) -> int:
    import uvicorn

    check_bind(host)
    uvicorn.run(
        "socratic_tutor.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return serve(args.host, args.port, args.reload, args.log_level)
    return 1  # unreachable: argparse rejects unknown subcommands


if __name__ == "__main__":
    sys.exit(main())
