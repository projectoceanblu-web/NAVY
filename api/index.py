"""Vercel serverless entrypoint.

Only the **read-only API and map** are deployed here. The two heavy
subsystems deliberately stay off serverless:

* **SAR processing** needs rasterio/GDAL, scipy and scikit-image — together
  they approach Vercel's 250 MB unzipped bundle limit, and a full GRD scene
  takes far longer than any serverless invocation is allowed to run.
* **AIS ingestion** is a long-lived WebSocket consumer. Serverless functions
  have no persistent process to hold a socket open.

Both run where they belong: `python -m bob_sentinel.workers.sar_process` and
`python -m bob_sentinel.workers.ais_ingest`, against the same database. The
deployment reads what they write.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The repository root is the Vercel project root; make the package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bob_sentinel.main import app  # noqa: E402

# Vercel's Python runtime looks for a module-level ASGI callable named `app`.
__all__ = ["app"]
