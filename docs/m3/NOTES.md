# M3 findings: diagnostic samples, schema validation, size estimate

Evidence: `smoke_test.json` (all checks), `field_dictionary.csv`,
`request_examples.json`, `size_estimate.csv`. Re-run: `scripts/smoke_test.py --site <id>`
(cached chunks are reused; `--skip-fetch` validates cache only).

## Samples (production acquisition path, outer 7 km AOI)
| Product | Granule (acq) | Outcome |
|---|---|---|
| ATL03 | 2021-09-10 RGT1199 c12 | ok, 848,186 photons (109,796 in core) |
| ATL03 | 2021-09-21 RGT1374 c12 | ok, 22,044 photons |
| ATL03 | 2021-07-21 RGT429 c12 (pre-emergence) | ok, 5,268 photons |
| ATL03 | 2021-12-09 RGT1199 c13 (name-lookup partner) | no_samples_in_aoi |
| ATL03 | 2026-07-12 RGT429 c32 (latest) | ok, 1,090,495 photons |
| ATL09 | via 2021-09-10 / 2021-09-21 ATL03 | ok, 278 / 268 native profiles |
| ATL09 | via 2025-03-04 ATL03 | no_granules (no ATL09 partner in catalog) |
| ATL08 | 2021-09-10 / 2021-09-21 | ok, 751 / 96 segments |
| ATL08 | 2021-07-21 | no_samples_in_aoi (ATL03 photons exist; no land segments) |
| ATL24 | 2021-09-10 / 2021-09-21 | ok, 227,966 / 18,088 photons |
| GEDI L2A | 2021-09-19 orbit 15692 | ok, 1,900 footprints (304 in core) |

## Validated
- **Time:** for ATL03, ATL08, ATL24 and GEDI, SlideRule `time_ns` equals
  `2018-01-01T00:00:00Z + delta_time`, with |residual| < 0.25 µs. Native
  `delta_time` is retained in every chunk.
- **Vertical reference:** ATL03 `height` equals native `h_ph` exactly (ellipsoidal,
  ITRF2014); GEDI `elevation_lm` equals native `elev_lowestmode` exactly (WGS84
  ellipsoid). Open-ocean ATL03 high-confidence photons sit +0.46 to +0.86 m
  above the ATL03 `geoid` (EGM2008), which is consistent with ellipsoidal
  heights. ATL24 `ellipse_h - ortho_h` is about 47.7 m with a 0.5–0.7 m range
  along track, matching the local geoid. Ellipsoidal and orthometric variables
  are kept separately.
- **Identifiers:** ATL03 `ph_index` is uint32 and `segment_id` is int32; there
  are no duplicate (srcid, gt, ph_index). GEDI `shot_number` is uint64 with no
  duplicates. ATL24 `index_ph` is int32, unique per beam. It refers to ATL03
  r006, so it is not used for joins.
- **Filter overrides took effect:** ATL03 returns confidence classes -1 through 4,
  including background (0) and not-considered (-1). ATL24 returns classes
  0/40/41. GEDI returns footprints whose L2 quality flag is not set.
- **ATL09 reduction:** ATL09 values are constant within every (beam, profile
  time), so the reduction from photons to profiles is lossless.
  `layer_top`/`layer_bot` are truncated to the first layer, as noted in M0.
- **Behaviour:**
  - An empty AOI gives `no_samples_in_aoi`; no matching granules gives
    `no_granules`.
  - A server or catalog failure gives `retrieval_failed`. A missing ATL09
    partner gives `no_granules`.
  - A tampered checksum or stray partial file means the chunk is not complete,
    so it is re-fetched.
  - A re-run skips every verified chunk.
  - Repeating a resource inside one request doubles the rows on the server,
    so the pipeline issues one request per unique granule. The inventory has
    no duplicate granules.

## Limitations / observations
- **ATL03 `quality_ph`:** every photon in all samples is 0 (nominal). A request
  for `[1,2,3]` only returns no rows rather than an error, which is consistent
  with non-nominal photons being absent here. A server-side filter upstream of
  the request cannot be fully excluded.
- **Photons just outside the outer AOI:** about 300 per pass, all within 5.4 m
  beyond the 7 km edge, because the server includes them by geolocation
  segment. They stay in the raw subsets; zones are assigned locally from photon
  coordinates and labeled `outer_margin`.
- **Fill values:** ATL08 terrain statistics contain float32-max fills and NaNs,
  and canopy fields are NaN over water. ATL09 `surface_height` and the layer
  fields contain float32-max fills. Raw values are kept and masks are applied
  in M5.
- **2021-07-21 pre-emergence pass:** the high-confidence photons have a median
  of about -132 m relative to the geoid, which is not a water surface. This is
  not interpreted until the M6 profiles.
- GEDI `landsat_water_persistence` is removed from requests (unsupported, M0).

## Full-run estimate
Totals across 67 ATL03/ATL09 resources, 66 ATL08, 54 ATL24 and 36 GEDI:
- **Disk:** about 1.3 GB expected, up to about 2.8 GB. ATL03 dominates
  (15 MB/granule on average, 33 MB maximum).
- **Runtime:** about 5.0 hours if run serially, so roughly 2.5–3 hours with 2
  workers. ATL09 requests are the slowest at about 2 minutes each.
