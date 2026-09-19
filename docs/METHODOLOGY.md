# Methodology, limitations, and responsible use

## The claim this system makes

For each Sentinel-1 detection, BoB Sentinel answers one narrow question:

> Was there a vessel broadcasting AIS at this place, at this instant?

It does **not** answer whether a vessel was fishing, whether it was licensed,
or whether anything illegal occurred. Every stage below introduces error, and
the verdicts are calibrated to acknowledge that rather than to hide it.

---

## Stage 1 — Detection (CA-CFAR)

**What it does.** Compares each pixel's intensity against clutter statistics
estimated from a surrounding ring of training cells, separated from the cell
under test by guard cells so a large vessel's own energy cannot inflate its own
background estimate.

**Clutter models.** Sea clutter in linear intensity is heavy-tailed, so the
choice of distribution matters more than the threshold. Measured on 640k px of
synthetic Rayleigh clutter:

| Model | Design PFA 1e-6 | Empirical | Verdict |
| --- | --- | --- | --- |
| `gaussian` | 1e-6 | 7.5e-5 | Over-calls by ~75×. Use only on heavily multi-looked data. |
| `lognormal` | 1e-6 | 0 | Conservative. Classic sea-clutter choice. |
| `gamma` | 1e-6 | 0 | **Default.** Found every planted target, zero false alarms. |

**Known failure modes.**

- **Wind and sea state.** High winds roughen the sea surface and raise
  backscatter; the adaptive threshold absorbs some of this, but a storm still
  drives the false-alarm rate up. Sentinel-1 L2 OCN wind products (shipped with
  xView3) can gate detections by wind speed.
- **Land and coastline.** Bright land will be detected as vessels. Pass a
  coastline mask via `cfar_detect(..., mask=...)`; masked pixels are zeroed
  *before* ring statistics so land cannot poison nearby sea thresholds.
- **Small vessels.** The wooden and fibreglass boats that dominate the Bay of
  Bengal fleet have low radar cross-sections and may be below the noise floor
  entirely. **Absence of a detection is not absence of a vessel.**
- **Azimuth ambiguities.** Strong targets produce ghost replicas displaced
  along-track — a known source of duplicate detections.
- **Size estimation.** `length_m` is the longest bounding-box side at ~10 m
  spacing. It saturates below ~20 m and should be read as an order of
  magnitude, not a measurement.

**Upgrade path.** CFAR is a screening prior. For production recall on small and
clustered vessels, use a learned detector (xView3 reference, the first-place
solution, or AI2 `sar_vessel_detect`) — all need a GPU.

---

## Stage 2 — Geolocation

GRD products are georeferenced by ground control points rather than a clean
affine transform. `rasterio.transform.from_gcps` fits an approximate transform
accurate to a few tens of metres over open water — adequate against a 500 m
match gate, and far cheaper than a full SNAP terrain-correction pass.

For metre-accurate geolocation, pre-process with SNAP or pyroSAR and feed the
terrain-corrected GeoTIFF in; `open_scene` reads either.

---

## Stage 3 — AIS interpolation

A SAR scene is one instant. AIS is a ragged per-vessel series: Class A reports
every 2–10 s under way, Class B every 30 s, and far less when satellite or
terrestrial coverage is thin.

| Situation | Method | Trusted? |
| --- | --- | --- |
| Instant falls between two fixes | Great-circle (slerp) interpolation | Yes, if the gap ≤ 30 min |
| Instant falls outside the track | Dead reckoning along SOG/COG | **No** |
| Nearest fix > `MATCH_MAX_GAP_S` away | Refused — returns `None` | n/a |

Great-circle interpolation is used rather than linear lon/lat averaging because
the latter is wrong over long gaps and breaks entirely across the antimeridian.

Dead-reckoned positions are excluded from matching by default. An extrapolated
guess should not be able to explain away a detection — that would convert a
modelling assumption into an exoneration.

---

## Stage 4 — Matching

The gate starts at `MATCH_RADIUS_M` (default 500 m) and widens for:

| Factor | Why |
| --- | --- |
| AIS position age | Drift while the vessel was unobserved (2 m/s allowance) |
| Vessel length | A 300 m hull's SAR centroid is not where its AIS antenna sits |
| Speed over ground | **Doppler azimuth displacement** (below) |

Gates are capped (`max_gate_m`, default 5 km) — an unbounded gate would
eventually match a detection to the entire sea.

### Doppler azimuth displacement

A SAR processor maps a moving target's radial velocity to an **along-track
position offset**:

```
offset ≈ R · v_radial / V_platform
```

For Sentinel-1 (slant range ~850 km, platform speed ~7.6 km/s), a 10 kn radial
component displaces a target by **~575 m** — already beyond a fixed 500 m gate.
A system that ignores this systematically manufactures "dark" vessels out of
fast-moving cooperative ones. This is why the gate is adaptive rather than
constant.

### Greedy assignment

Pairs are matched shortest-distance-first, and an MMSI is consumed once
matched, so two detections cannot both be explained by the same transponder.

---

## Stage 5 — The verdict

This is the part with ethical consequences.

```
                     ┌─ AIS position within the gate? ──► matched
SAR detection ───────┤
                     └─ no match ─┬─ AIS coverage nearby? ──► dark
                                  └─ no coverage nearby ───► indeterminate
```

**Coverage test.** If *no* vessel at all was reporting AIS within 50 km of the
detection, the absence of a match tells you about the feed, not about the
vessel. aisstream is terrestrial and crowd-contributed; it genuinely does go
thin offshore in the northern Bay of Bengal.

`detections.is_dark` is **nullable** and `NULL` means indeterminate, so these
rows are naturally excluded from `WHERE is_dark` queries. Collapsing
indeterminate into dark would roughly match how a naive implementation behaves
and would be the single largest source of false accusations in this system.

### Legitimate reasons a vessel appears dark

- Below the AIS carriage threshold (most small artisanal fishing boats).
- Transponder failure, power loss, or antenna damage.
- Naval, coast guard, or government vessels exempt from broadcasting.
- AIS reception gaps — satellite coverage, message collision in dense traffic.
- The detection is not a vessel at all: a wind streak, an oil platform, an
  azimuth ambiguity, a breaking wave.

Switching AIS off is also a documented tactic in IUU fishing. **Both
explanations are consistent with the same observation**, which is precisely why
this is a screening tool and not an enforcement one.

---

## Stage 6 — Behavioural heuristics

| Kind | Rule | Innocent explanation |
| --- | --- | --- |
| `ais_gap` | ≥6 h silence, after ≥14 positions in the prior 12 h | Receiver outage, transponder fault |
| `impossible_speed` | Implied SOG > 60 kn between fixes | Duplicated MMSI, bad decode, clock skew |
| `loitering` | ≤2 kn within 10 km for ≥3 h | Fishing gear soaking, engine trouble, weather |
| `encounter` | Two vessels ≤500 m and ≤2 kn for ≥2 h | Crew transfer, resupply, assistance at sea |
| `identity_anomaly` | Malformed MMSI/MID, bad IMO check digit | Data entry error, receiver corruption |

The gap precondition matters most. Without requiring prior good reception,
every vessel in a poorly covered area is flagged for going "dark" when in
reality it was never being heard well. This follows Global Fishing Watch's own
rule and is tested explicitly
(`test_gap_on_a_poorly_heard_vessel_is_not_flagged`).

Each anomaly stores the numbers that produced it so an analyst can dismiss it
in seconds.

---

## Validation

The system has **not** been validated against ground truth. Doing so properly
would need:

1. **Detector benchmarking** against xView3-SAR's labelled scenes, reporting
   precision/recall by vessel length — CFAR's weakness is small vessels, and
   that is exactly the fleet that matters here.
2. **Match-rate baselining** against GFW's `public-global-sar-presence` layer
   over the same water and window. A large divergence in dark-detection rate
   indicates a problem in this pipeline, not a discovery.
3. **Coverage characterisation** of aisstream over the Bangladesh EEZ
   specifically, to calibrate the 50 km coverage radius empirically rather than
   by assumption.

Until (1) and (2) are done, treat detection counts as relative indicators, not
absolute measurements.

---

## Responsible use

- **Label outputs as probabilistic.** The API description, the map sidebar, and
  every dark-detection popup say so; keep it that way in anything downstream.
- **Never present a dark detection as proof.** It is a prompt to task another
  sensor or a patrol, nothing more.
- **Small-boat bias.** The fleet most likely to be missed by SAR is also the
  most economically vulnerable. A system that reliably sees industrial vessels
  and misses artisanal ones will produce a skewed picture of who is fishing.
  Be explicit about this whenever detection counts inform policy.
- **Open data only.** Everything here is public. That keeps the work auditable
  and reproducible, and it is a deliberate constraint.
- **Dual use.** Maritime domain awareness protects fisheries and also enables
  surveillance of people going about lawful work. Be deliberate about who gets
  access to outputs and what they are permitted to do with them.
