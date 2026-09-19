"""Vercel serverless entrypoint.

Only the **read-only API and map** are deployed here. The two heavy
subsystems deliberately stay off serverless:

* **SAR processing** needs rasterio/GDAL, scipy and scikit-image — together
  they approach Vercel's 250 MB unzipped bundle limit, and a full GRD scene
  takes far longer than any serverless invocation is allowed to run.
* **AIS ingestion** is a long-lived WebSocket consumer, though
  ``/api/ais/collect`` drains it in short bursts so the hosted map is still
  fed by the real feed.

Both run where they belong: ``python -m bob_sentinel.workers.sar_process`` and
``python -m bob_sentinel.workers.ais_ingest`` against the same database.

``app`` is built by a function and bound with a single top-level assignment
on purpose: Vercel's Python builder statically scans this module for a
top-level ``app``/``application``/``handler``, and an assignment nested
inside a ``try`` block is not recognised.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The repository root is the Vercel project root; make the package importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _listing(path: Path) -> list[str]:
    try:
        return sorted(p.name for p in path.iterdir())[:50]
    except OSError as exc:
        return [f"<unreadable: {exc}>"]


def _bootstrap_failure_app(error: str):
    """Serve the import traceback instead of a blank 500.

    A failed import inside a serverless function surfaces only as a generic
    FUNCTION_INVOCATION_FAILED, and this account's runtime logs are not
    reachable through the API — so the cause has to arrive over HTTP.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    fallback = FastAPI(title="BoB Sentinel (bootstrap failure)")

    @fallback.get("/{full_path:path}")
    def bootstrap_error(full_path: str) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": "bob_sentinel failed to import in the serverless bundle",
                "traceback": error.splitlines()[-25:],
                "entrypoint": __file__,
                "resolved_root": str(ROOT),
                "root_contents": _listing(ROOT),
                "cwd": os.getcwd(),
                "cwd_contents": _listing(Path.cwd()),
                "sys_path": sys.path[:15],
                "python": sys.version,
            },
        )

    return fallback


def _build_app():
    try:
        from bob_sentinel.main import app as application
    except Exception:  # noqa: BLE001 — report it rather than dying silently
        import traceback

        return _bootstrap_failure_app(traceback.format_exc())
    return application


app = _build_app()

__all__ = ["app"]
