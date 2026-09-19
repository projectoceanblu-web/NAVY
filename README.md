# BoB Sentinel

Dark-vessel / IUU screening MVP for the Bay of Bengal (Bangladesh EEZ), fusing
free Sentinel-1 SAR detections with live AIS.

The pipeline is: pull Sentinel-1 GRD scenes from the Copernicus Data Space
Ecosystem → run CA-CFAR ship detection → interpolate AIS tracks to the SAR
acquisition instant → match detections against AIS → flag what cannot be
explained.

## What this is, and is not

**Outputs are probabilistic screening signals, not evidence of illegal
fishing.** A SAR detection without an AIS match may be a vessel legitimately
exempt from carrying AIS, a failed transponder, a naval or government vessel,
or sea-state clutter. This is a dual-use surveillance prototype built entirely
on open data; treat every output as a prompt to look, never as a finding.

The system deliberately reports **three** verdicts, not two:

| Verdict | Meaning |
| --- | --- |
| `matched` | A SAR detection sits within the match gate of an interpolated AIS position. |
| `dark` | No AIS match, **and** other vessels nearby were reporting AIS — so the silence is notable. |
| `indeterminate` | No AIS match, but no AIS coverage nearby either. This says something about the feed, not about the vessel. |

Collapsing `indeterminate` into `dark` is the single easiest way to turn this
tool into a false-accusation machine, which is why the distinction is enforced
in the data model (`detections.is_dark` is nullable and `NULL` means "unknown"),
in the API, and on the map.

## Live deployment

**https://bob-sentinel.vercel.app**

The hosted map serves live Global Fishing Watch data over the Bangladesh EEZ —
Sentinel-1 SAR dark detections, AIS vessel presence, apparent fishing effort,
and encounter/loitering events — with the GFW token held server-side.

Only the read-only API and map run on Vercel. The SAR pipeline needs GDAL and
minutes of CPU per scene, and AIS ingestion is a long-lived WebSocket
consumer; both stay batch/daemon jobs against the same database. See
[Hosting](#hosting).

## Quick start

### 1. Run it offline with synthetic data (no accounts needed)

Useful for developing the detector and fusion logic without credentials or
network. It is **not** representative data — see
[the live deployment](#live-deployment) for real observations.

```bash
cp .env.example .env          # defaults are fine for the demo
docker compose up -d db
docker compose run --rm api python scripts/init_db.py --fallback-eez
docker compose run --rm api python scripts/demo_pipeline.py
docker compose up -d api
open http://localhost:8000
```

The demo synthesises a SAR scene and an AIS picture designed to produce one of
each verdict, then runs the real pipeline over them. It is the fastest way to
confirm the stack is wired correctly.

### 2. Run it with live data

Register for four free accounts, then fill in `.env`:

| Service | What it provides | Sign-up |
| --- | --- | --- |
| Copernicus Data Space Ecosystem | Sentinel-1 GRD SAR | <https://dataspace.copernicus.eu/> |
| aisstream.io | Live AIS over WebSocket | <https://aisstream.io/> |
| Global Fishing Watch | Independent SAR/AIS cross-check | <https://globalfishingwatch.org/our-apis/tokens> |
| xView3-SAR *(optional)* | Labelled SAR training data | <https://iuu.xview.us/> |

```bash
docker compose up -d                 # db + api + live AIS ingester
docker compose run --rm sar-worker python -m bob_sentinel.workers.sar_process --days 7 --limit 1
docker compose run --rm anomaly-scan
```

### Load the real EEZ boundary

The built-in Bangladesh outline is a **coarse approximation** good only for
bounding an AIS subscription. For anything that depends on whether a vessel was
inside the zone, download the authoritative polygon — Flanders Marine Institute
(VLIZ) Maritime Boundaries Geodatabase, World EEZ v12, MRGID 8481 — from
<https://www.marineregions.org/downloads.php>, export the Bangladesh feature to
GeoJSON, and load it:

```bash
docker compose run --rm api python scripts/init_db.py --eez-geojson /data/bangladesh_eez.geojson
```

## Architecture

```
aisstream.io ──► ais-ingest worker ──┐
   (WebSocket)                       ├──► PostGIS ──► FastAPI ──► Leaflet map
CDSE ──► sar-worker ──► CA-CFAR ─────┤        ▲
   (OData/OAuth2)      detections    │        │
                                     └── fusion (interpolate → match → verdict)
GFW API v3 ──────────────────────────────────┘  (independent cross-check, proxied)
```

| Component | Module |
| --- | --- |
| Sentinel-1 search & download | `bob_sentinel/sar/cdse.py` |
| GRD reading, tiling, geolocation | `bob_sentinel/sar/raster.py` |
| CA-CFAR detector | `bob_sentinel/sar/cfar.py` |
| Scene → geolocated detections | `bob_sentinel/sar/pipeline.py` |
| AIS ingestion | `bob_sentinel/ingest/aisstream.py` |
| AIS interpolation | `bob_sentinel/fusion/interpolate.py` |
| Matching & dark-vessel verdict | `bob_sentinel/fusion/matcher.py` |
| Behavioural heuristics | `bob_sentinel/fusion/anomalies.py` |
| GFW client | `bob_sentinel/clients/gfw.py` |
| HTTP API | `bob_sentinel/api/routes.py` |

## API

Interactive docs at `/docs`. All geometry is GeoJSON (`lon, lat`).

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Liveness plus per-credential readiness |
| `GET /api/stats` | Row counts and data freshness |
| `GET /api/ais/latest?minutes=` | Newest position per vessel |
| `GET /api/ais/track/{mmsi}?hours=` | One vessel's track |
| `GET /api/detections?status=dark\|matched\|indeterminate` | SAR detections with verdicts |
| `GET /api/scenes` | Processed SAR scenes |
| `GET /api/anomalies?kind=` | AIS behavioural flags |
| `GET /api/regions` | Boundary polygons |
| `GET /api/gfw/tile/{z}/{x}/{y}` | GFW heatmap tiles, token kept server-side |

Live layers, proxied to Global Fishing Watch per request (token server-side,
cached 5 minutes because GFW rate-limits per user across all tokens):

| Endpoint | Purpose |
| --- | --- |
| `GET /api/live/sar?dark_only=true` | Real Sentinel-1 detections; `dark_only` applies `matched='false'` |
| `GET /api/live/presence` | Real AIS vessel presence |
| `GET /api/live/fishing` | Real apparent fishing effort |
| `GET /api/live/events?types=` | Real encounters, loitering, fishing events |
| `GET /api/live/diagnostics` | Upstream envelope shapes — for debugging an empty layer |
| `POST /api/ais/collect?seconds=` | Drain the aisstream feed briefly and persist it |

## Detection

CA-CFAR compares each pixel against clutter statistics from a surrounding
training ring, with guard cells so a vessel's own energy cannot inflate its
background estimate. Ring statistics use summed-area tables, so a full GRD
scene is a handful of vectorised NumPy passes rather than a Python sliding
window.

Three clutter models are available. Measured on 640k px of synthetic Rayleigh
sea clutter:

| Model | Empirical FA rate at `pfa=1e-6` | Notes |
| --- | --- | --- |
| `gaussian` | 7.5e-5 (**75× design**) | Rayleigh is heavy-tailed vs. normal; over-calls. |
| `lognormal` | ≤ design | Conservative; the classic sea-clutter choice. |
| `gamma` *(default)* | ≤ design | Found every planted target with zero false alarms. |

Speckle is rejected by the `min_area_px` area filter during clustering, **not**
by morphological opening: a 3×3 erosion annihilates any vessel thinner than
3 px, which at 10 m GRD spacing is most of the Bay of Bengal fishing fleet.

For higher recall on small and clustered vessels, swap in a learned detector —
the [xView3 reference model](https://github.com/DIUx-xView/xview3-reference),
the [first-place solution](https://github.com/DIUx-xView/xView3_first_place),
or AI2's [`sar_vessel_detect`](https://github.com/allenai/sar_vessel_detect).
Those need a GPU; CFAR does not.

## Matching

AIS is a ragged per-vessel time series; a SAR scene is one instant. Every track
is interpolated to the acquisition midpoint — great-circle interpolation when
the instant is bracketed by real fixes, dead reckoning along SOG/COG when it is
not (and refused entirely beyond `MATCH_MAX_GAP_S`, where the estimate would
describe the model rather than the vessel).

The match gate is adaptive, starting at `MATCH_RADIUS_M` and widening for:

- **AIS position age** — drift allowance while the vessel was unobserved.
- **Vessel length** — a 300 m hull's centroid is not its AIS antenna.
- **Doppler azimuth displacement** — a SAR processor maps a moving target's
  radial velocity to an along-track offset of `R·v/V`. For Sentinel-1 a 10 kn
  radial component displaces a target ~575 m, already past a fixed 500 m gate.
  Ignoring this manufactures "dark" vessels out of fast movers.

## Behavioural heuristics

`gaps`, `impossible_speed`, `loitering`, `encounter`, `identity_anomaly`. Each
carries the numbers that produced it so an analyst can dismiss it quickly.

The gap rule follows Global Fishing Watch's template: a silence is only flagged
if the vessel was being heard reliably beforehand (≥14 positions in the prior
12 hours). Without that precondition every patchy receiver becomes an
accusation.

## Hosting

The deployment splits along what serverless can actually do:

| Component | Where | Why |
| --- | --- | --- |
| Read-only API + map | Vercel (`api/index.py`) | Stateless and fast |
| Live GFW layers | Vercel | REST, fits an invocation |
| PostGIS | Supabase (transaction pooler) | The direct host is IPv6-only on the free tier |
| SAR detection | Batch, off-Vercel | rasterio/GDAL + scipy; minutes of CPU per scene |
| AIS ingestion | Daemon, off-Vercel | A persistent WebSocket has no home in a function |

`api/requirements.txt` deliberately excludes rasterio, scipy and
scikit-image; CI asserts the API still imports without them, because a stray
import there fails the function at boot in production rather than at build.

For continuous AIS rather than per-request bursts, run
`python -m bob_sentinel.workers.ais_ingest` against the same database on any
always-on host.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest                      # unit tests; integration tests skip
.venv/bin/ruff check bob_sentinel tests scripts
```

Integration tests need PostGIS and are skipped without it:

```bash
TEST_DATABASE_URL=postgresql+psycopg://sentinel:sentinel@localhost:5432/bobsentinel \
  .venv/bin/pytest tests/test_integration_db.py
```

## Operational notes

- **Sentinel-1 constellation (2026).** Sentinel-1A concluded operations on
  30 June 2026; Sentinel-1D became fully operational on 5 May 2026. The
  constellation is now 1C + 1D at a 6-day nominal revisit, and the CDSE client
  filters to `S1C`/`S1D` by default.
- **CDSE tokens expire in ~10 minutes.** The client refreshes transparently;
  for very long jobs prefer S3 access, whose keys do not expire.
- **aisstream has no coverage in the Bay of Bengal.** Measured 2026-09-19: a
  global subscription returns ~1,700 messages in 15 s, while the Bangladesh
  EEZ returns zero — as does the whole bay (5–25°N, 78–100°E). The nearest
  contributing receivers are around the Malacca Strait. The feed is
  crowd-contributed from shore stations, so this is a gap in reception, not
  an absence of vessels, and the collector says so rather than drawing an
  empty layer that would read as empty water. It is also precisely why an
  unmatched SAR detection here is reported `indeterminate`, not `dark`.
  Real vessel tracks over this AOI need a source that covers it — GFW's
  Vessels API or a satellite-AIS provider.
- **GFW is non-commercial use only**, rate-limited per user across all tokens,
  and its SAR layer runs to roughly 5 days ago — it is not a live feed.
- **Geolocation accuracy.** GRD products are georeferenced by GCPs; the
  approximate transform is good to a few tens of metres over open water. For
  metre-accurate work, terrain-correct with SNAP/pyroSAR first — `open_scene`
  reads either.

## Licence and data attribution

Contains modified Copernicus Sentinel data. AIS via aisstream.io. Maritime
boundaries © Flanders Marine Institute, Maritime Boundaries Geodatabase v12
(DOI 10.14284/632). Global Fishing Watch data used under their non-commercial
API terms.
