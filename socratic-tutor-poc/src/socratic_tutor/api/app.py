"""FastAPI application root.

Localhost, single user, no auth (Addendum §"Repo & stack"). One SPA is served
from `web/dist` when it has been built; in development Vite serves it on :5173
and proxies `/api` here, so CORS is configured for that origin.

Every router the product needs is included in `create_app` below. A new feature
area adds one `api_router.include_router(...)` line there *and* a test that
drives `create_app`, never a hand-built `FastAPI()` — a test that assembles its
own app cannot catch a router that was never mounted, which is exactly how
`/api/chat/turn` once shipped returning `index.html` with HTTP 200.

The static mount claims `/`, so it goes on last: anything registered after it is
unreachable, and any API path that is *not* registered gets silently answered
with the SPA's HTML instead of a 404.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from socratic_tutor.api.chat_routes import router as chat_router
from socratic_tutor.api.ingest_routes import IngestHub, IngestPipeline, RagPipeline
from socratic_tutor.api.ingest_routes import router as ingest_router
from socratic_tutor.api.klmap_routes import router as klmap_router
from socratic_tutor.config import PROJECT_ROOT, get_settings
from socratic_tutor.providers import reset_providers

logger = logging.getLogger(__name__)

#: The Vite dev server. Its proxy makes CORS unnecessary for `npm run dev`, but
#: a browser pointed straight at :5173 with `VITE_API_BASE` set needs it.
DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

WEB_DIST = PROJECT_ROOT / "web" / "dist"


def create_app(
    pipeline: IngestPipeline | None = None,
    web_dist: Path | None = None,
) -> FastAPI:
    """Build the app. Tests pass a fake `pipeline` to keep models and network out."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        get_settings().ensure_dirs()
        try:
            yield
        finally:
            await app.state.ingest_hub.drain()
            await reset_providers()

    app = FastAPI(
        title="Socratic AI Tutor — POC",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.ingest_hub = IngestHub()
    app.state.ingest_pipeline = pipeline or RagPipeline()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_router = APIRouter(prefix="/api")
    api_router.include_router(ingest_router)
    api_router.include_router(klmap_router)
    # Chat and the glass-box inspector (R19/R20). `chat_routes` builds its own
    # dependencies from seed content on first use unless `set_dependencies` was
    # called, so no wiring is needed here.
    api_router.include_router(chat_router)

    @api_router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api_router)

    # Mounted last: it claims "/", so every API path must already be registered
    # above or it will be answered with the SPA's HTML instead of a 404.
    dist = web_dist or WEB_DIST
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
    else:
        logger.info("web/dist not built; serving the API only (run `npm run build` in web/)")

    return app


__all__ = ["DEV_ORIGINS", "WEB_DIST", "create_app"]
