"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bob_sentinel import __version__
from bob_sentinel.api.routes import router
from bob_sentinel.config import get_settings

log = logging.getLogger(__name__)

DESCRIPTION = """
Dark-vessel screening for the Bangladesh EEZ, fusing Sentinel-1 SAR detections
with live AIS.

**These outputs are probabilistic decision support, not evidence of illegal
fishing.** A detection with no AIS match may be a vessel legitimately exempt
from carrying AIS, a vessel whose transponder failed, or an artefact of sea
state. Detections are labelled `matched`, `dark`, or `indeterminate`; the last
means AIS coverage was too thin to interpret the absence of a match at all.
"""


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("BoB Sentinel %s starting; AOI=%s", __version__, settings.aoi_name)
    missing = [
        name
        for name, value in (
            ("CDSE_USERNAME", settings.cdse_username),
            ("AISSTREAM_API_KEY", settings.aisstream_api_key),
            ("GFW_API_TOKEN", settings.gfw_api_token),
        )
        if not value
    ]
    if missing:
        # Not fatal: each subsystem is independent, and a missing GFW token
        # should never stop AIS ingestion.  /healthz reports what is armed.
        log.warning("unconfigured credentials: %s", ", ".join(missing))
    yield
    log.info("BoB Sentinel shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="BoB Sentinel",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # MVP runs behind a local compose network
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(router)

    frontend = Path(__file__).resolve().parent.parent / "frontend"
    if frontend.is_dir():
        app.mount("/static", StaticFiles(directory=frontend), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(frontend / "index.html")

    return app


app = create_app()
