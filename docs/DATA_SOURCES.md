# Data sources — registration, endpoints, and limits

Everything here is free. Four accounts; none require payment.

---

## 1. Copernicus Data Space Ecosystem (Sentinel-1 SAR)

**Register:** <https://dataspace.copernicus.eu/> → "Register".

### Authentication

Direct product download uses the OAuth2 **password grant** against the CDSE
Keycloak realm:

```
POST https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
  client_id=cdse-public
  grant_type=password
  username=<...>
  password=<...>
```

Returns an `access_token` valid ~10 minutes plus a `refresh_token`.
`bob_sentinel/sar/cdse.py` refreshes ahead of expiry and falls back to a full
re-login if the refresh token is rejected.

Sentinel Hub and openEO instead use **client credentials**: create an OAuth
client under Dashboard → User settings (the secret is shown once).

### Access methods

| Method | Endpoint | Best for |
| --- | --- | --- |
| OData *(used here)* | `https://catalogue.dataspace.copernicus.eu/odata/v1` | Rich `$filter` queries by product type, geometry, date |
| STAC | `https://stac.dataspace.copernicus.eu/v1` | Standards-based discovery (the legacy `/stac` endpoint was deprecated 17 Nov 2025) |
| Zipper | `https://zipper.dataspace.copernicus.eu/odata/v1/Products(<id>)/$value` | Downloading the product zip |
| S3 | `https://eodata.dataspace.copernicus.eu` | High-throughput parallel access; keys do not expire |
| openEO | `openeo` client | Server-side subsetting, no bulk download |

### Constellation status (2026)

| Satellite | Status |
| --- | --- |
| Sentinel-1A | **Concluded operations 30 June 2026** after 12 years |
| Sentinel-1B | Retired December 2021 |
| Sentinel-1C | Operational since May 2025 |
| Sentinel-1D | Open data 17 April 2026; fully operational 5 May 2026 |

The constellation is now **1C + 1D at a 6-day nominal revisit**. At Bangladesh's
latitude overlapping swaths give effective passes every few days. `ACTIVE_PLATFORMS`
in `cdse.py` encodes this; update it when the constellation changes again.

Sentinel-1C acquisitions were briefly suspended during the June 2026
reconfiguration (approx. 9–23 June), so verify the current acquisition scenario
before assuming a fixed revisit.

### Products and quotas

Use `IW_GRDH_1S` (Interferometric Wide swath, high-resolution ground range
detected, Level-1) — systematically produced, ~10 m pixel spacing. NRT products
arrive in under 3 hours.

Downloads are summed per user over a rolling 30-day window and checked hourly.
Exceeding the quota moves you to a slower interface rather than cutting you off.
Sentinel Hub processing units reset on the 1st of each month and do not roll over.

---

## 2. aisstream.io (live AIS)

**Register:** <https://aisstream.io/> — sign in with GitHub, create a key under
Account. WebSocket only; there is no REST API.

**Endpoint:** `wss://stream.aisstream.io/v0/stream`

```json
{
  "APIKey": "<YOUR_KEY>",
  "BoundingBoxes": [[[20.5, 88.0], [22.8, 92.7]]],
  "FilterMessageTypes": ["PositionReport", "ShipStaticData"]
}
```

### Protocol details that cost time if learned the hard way

- The subscription must be sent **within 3 seconds** of the socket opening.
- Bounding boxes are `[[lat, lon], [lat, lon]]` — **latitude first**, the
  opposite of GeoJSON. Getting this wrong yields a silent, empty stream.
- Keys are **case-sensitive**: `APIKey`, not `apiKey`. Wrong casing is accepted
  and then sends no data — the classic "subscribed but no data" symptom.
- Frames arrive as **binary** WebSocket messages carrying UTF-8 JSON.
- Re-sending a subscription **replaces** the previous one; it does not merge.
- `FiltersShipMMSI` accepts at most 50 vessels.
- Do not send subscription updates faster than once per second.
- Community-run, **no SLA**. Reconnection with backoff is mandatory.

### Sentinel values

`Sog = 102.3`, `Cog = 360.0` and `TrueHeading = 511` all mean "not available",
not a real reading. The parser maps them to `None`.

### Fallbacks

- **AISHub** (<https://www.aishub.net/>) — cooperative: you must run a receiver
  and stream NMEA (≥10 vessels average over 7 days, ≥90% uptime) to get the
  aggregated feed. Budget for hardware.
- **RTL-SDR** — a local receiver plus `pyais` gives a resilient coastal feed.
- **GFW** — historical and near-real-time AIS-derived layers.

---

## 3. Global Fishing Watch API v3

**Register:** <https://globalfishingwatch.org/our-apis/tokens> (3-step
self-registration, non-commercial use).

**Base URL:** `https://gateway.api.globalfishingwatch.org/v3`
**Auth:** `Authorization: Bearer <TOKEN>`

### Datasets

| Purpose | Dataset ID |
| --- | --- |
| SAR presence | `public-global-sar-presence:latest` |
| AIS presence | `public-global-presence:latest` |
| Fishing effort | `public-global-fishing-effort:latest` |
| Vessel identity | `public-global-vessel-identity:latest` |
| Encounters | `public-global-encounters-events:latest` |
| Fishing events | `public-global-fishing-events:latest` |
| Port visits | `public-global-port-visits-events:latest` |
| Loitering | `public-global-loitering-events:latest` |
| Gaps (AIS-off) | `public-global-gaps-events:latest` |

The loitering and gaps IDs follow the v3 naming convention but were not
verbatim-confirmed against a live v3 response. **Confirm with
`GET /v3/datasets` before relying on them** — `GFWClient.list_datasets()` exists
for that.

### Dark detections

`POST /v3/4wings/report` with `filters[0]=matched='false'`, region
`{"dataset": "public-eez-areas", "id": 8481}` (Bangladesh).

### Limits

- Non-commercial use only; GFW may revoke access.
- Rate limits are **per user across all tokens**; max 5 tokens per user.
- `date-range` caps at 366 days.
- SAR layer runs 2017 → roughly 5 days ago. **Not live.**
- Per-detection extras (`length_m`, `presence_score`, `matching_score`) are in
  the Data Download Portal only, not the API.
- GFW reported SAR freshness disruption in August 2026 tied to Sentinel-1A's
  retirement — a thin result may be upstream, not empty seas.

---

## 4. xView3-SAR (training data, optional)

**Register:** <https://iuu.xview.us/> (credentialed, US export-control
compliance). Code: <https://github.com/DIUx-xView>.

991 full-size Sentinel-1 scenes averaging 29,400 × 24,400 px — 1,400 gigapixels
total, with 243,018 verified maritime objects over 43.2 million km². Labels
carry `is_vessel`, `is_fishing`, `confidence` and vessel length; co-located
bathymetry (GEBCO) and wind (Sentinel-1 L2 OCN) rasters accompany each scene.

The Hugging Face mirror `ConnorLuckettDSTG/SARFish` extends it with SLC.

Only needed if you replace CFAR with a learned detector.

---

## 5. Maritime boundaries

**Marine Regions** (<https://www.marineregions.org/downloads.php>) — Flanders
Marine Institute (VLIZ) Maritime Boundaries Geodatabase, World EEZ v12
(2023-10-25, 122 MB, DOI 10.14284/632). Bangladesh is **MRGID 8481**.

Headline areas differ by source — FAO gives ~141,000 km² for the EEZ within
~166,000 km² of marine waters; other sources cite ~166,000 km² for the EEZ
alone. **Always filter on the polygon geometry, never on a headline number.**

Also useful: Protected Planet / WDPA (<https://www.protectedplanet.net/>) for
marine protected areas.
