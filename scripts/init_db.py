#!/usr/bin/env python3
"""Create the schema and load the area of interest.

Idempotent: safe to run on every container start.

    python scripts/init_db.py [--fallback-eez] [--eez-geojson path.geojson]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from bob_sentinel.config import get_settings  # noqa: E402
from bob_sentinel.db import get_engine, session_scope  # noqa: E402
from bob_sentinel.ingest.eez import (  # noqa: E402
    load_fallback_bangladesh_eez,
    load_geojson_region,
)
from bob_sentinel.main import configure_logging  # noqa: E402
from bob_sentinel.models import Base  # noqa: E402
from bob_sentinel.repository import get_region  # noqa: E402

log = logging.getLogger("init_db")


def wait_for_db(attempts: int = 30, delay: float = 2.0) -> None:
    """Compose starts the API before Postgres finishes recovery; retry."""
    engine = get_engine()
    for attempt in range(1, attempts + 1):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            log.info("database is up")
            return
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            log.info("waiting for database (%d/%d): %s", attempt, attempts, type(exc).__name__)
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eez-geojson", type=Path,
                        help="GeoJSON of the authoritative VLIZ EEZ polygon")
    parser.add_argument("--fallback-eez", action="store_true",
                        help="Load the built-in APPROXIMATE outline instead")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    wait_for_db()
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)
    log.info("schema ready: %s", ", ".join(sorted(Base.metadata.tables)))

    with session_scope() as session:
        if args.eez_geojson:
            region = load_geojson_region(
                session, args.eez_geojson, name=settings.aoi_name,
                kind="eez", mrgid=settings.aoi_mrgid,
            )
            log.info("loaded EEZ %r (id=%s)", region.name, region.id)
        elif args.fallback_eez:
            if get_region(session, mrgid=settings.aoi_mrgid) is None:
                load_fallback_bangladesh_eez(session)
            else:
                log.info("AOI already present; leaving it alone")
        else:
            log.info(
                "no EEZ loaded. Download World EEZ v12 from "
                "https://www.marineregions.org/downloads.php and pass "
                "--eez-geojson, or use --fallback-eez for the approximate outline."
            )


if __name__ == "__main__":
    main()
