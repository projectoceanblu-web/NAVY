"""Loading maritime boundary polygons into PostGIS.

The authoritative source is the Flanders Marine Institute (VLIZ) Maritime
Boundaries Geodatabase, World EEZ v12 (2023-10-25, DOI 10.14284/632), from
https://www.marineregions.org/downloads.php — a free download that requires
accepting their licence, so it is not fetched automatically.  Bangladesh is
gazetteer MRGID 8481.

Headline EEZ areas differ by source (~141,000 km2 per FAO, ~166,000 km2 per
others) because they count different zones.  Always filter on the polygon
geometry, never on a headline number.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from bob_sentinel.models import Region
from bob_sentinel.repository import upsert_region

log = logging.getLogger(__name__)

MARINE_REGIONS_URL = "https://www.marineregions.org/downloads.php"
BANGLADESH_MRGID = 8481

#: Coarse fallback outline of the Bangladesh EEZ, used when the licensed VLIZ
#: geodatabase has not been downloaded.  It is a convex approximation of the
#: claim following the 2012 ITLOS and 2014 PCA delimitations — good enough to
#: bound an AIS subscription and to smoke-test the stack end to end, and NOT
#: good enough for any statement about whether a vessel was inside the EEZ.
BANGLADESH_EEZ_APPROX = {
    "type": "Polygon",
    "coordinates": [
        [
            [92.20, 21.05],   # Myanmar land boundary terminus (Naf river mouth)
            [91.00, 20.72],
            [90.00, 20.40],
            [89.20, 20.30],
            [88.90, 20.62],   # India (West Bengal) boundary, seaward
            [89.10, 21.62],   # coastal India/Bangladesh land terminus
            [89.85, 21.78],   # Sundarbans / Meghna estuary
            [90.62, 22.10],
            [91.40, 22.40],
            [91.85, 22.30],   # Chittagong coast
            [92.32, 21.42],   # Teknaf peninsula
            [92.20, 21.05],
        ]
    ],
}


def geojson_to_wkt(geometry: dict) -> str:
    """Minimal GeoJSON -> WKT for (Multi)Polygon, no geo dependencies needed."""
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "Polygon":
        return f"POLYGON({_rings(coordinates)})"
    if kind == "MultiPolygon":
        polygons = ",".join(f"({_rings(polygon)})" for polygon in coordinates)
        return f"MULTIPOLYGON({polygons})"
    raise ValueError(f"unsupported geometry type: {kind!r}")


def _rings(rings) -> str:
    return ",".join(
        "(" + ",".join(f"{float(x)} {float(y)}" for x, y, *_ in ring) + ")"
        for ring in rings
    )


def load_geojson_region(
    session: Session,
    path: str | Path,
    *,
    name: str,
    kind: str = "eez",
    mrgid: int | None = None,
) -> Region:
    """Load a region from a GeoJSON file (Feature, FeatureCollection or bare geometry).

    Multiple features are dissolved into one MULTIPOLYGON, which is what the
    VLIZ EEZ layer needs when a country's claim comes as several parts.
    """
    path = Path(path)
    payload = json.loads(path.read_text())

    geometries = []
    if payload.get("type") == "FeatureCollection":
        geometries = [f["geometry"] for f in payload["features"]]
    elif payload.get("type") == "Feature":
        geometries = [payload["geometry"]]
    else:
        geometries = [payload]

    polygons: list[list] = []
    for geometry in geometries:
        if geometry["type"] == "Polygon":
            polygons.append(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            polygons.extend(geometry["coordinates"])
        else:
            raise ValueError(f"unsupported geometry type: {geometry['type']!r}")

    wkt = geojson_to_wkt({"type": "MultiPolygon", "coordinates": polygons})
    log.info("loading region %r (%d polygon(s)) from %s", name, len(polygons), path)
    return upsert_region(
        session, name=name, kind=kind, geom_wkt=wkt, mrgid=mrgid, source=str(path)
    )


def load_fallback_bangladesh_eez(session: Session) -> Region:
    """Install the coarse approximate EEZ so the stack runs before the download."""
    log.warning(
        "loading the APPROXIMATE Bangladesh EEZ outline — replace it with the "
        "VLIZ World EEZ v12 polygon (%s, MRGID %d) before drawing any "
        "conclusion about whether a vessel was inside the zone",
        MARINE_REGIONS_URL,
        BANGLADESH_MRGID,
    )
    return upsert_region(
        session,
        name="Bangladesh EEZ (approximate)",
        kind="eez",
        mrgid=BANGLADESH_MRGID,
        geom_wkt=geojson_to_wkt(BANGLADESH_EEZ_APPROX),
        source="built-in approximation — NOT the authoritative VLIZ boundary",
    )
