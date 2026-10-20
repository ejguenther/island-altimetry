# M8 notes — PlanetScope imagery coincident with ATL03 / GEDI

## Purpose
M5–M7 could not tell new land from a pumice raft or the sea surface, and could
not date when the 2021 island disappeared. Dated, mostly cloud-free optical
imagery taken close to each lidar overpass is the independent evidence needed.

## Method
1. **Targets** (`planet_targets.csv`): each ATL03 and GEDI L2A granule with
   `footprint_intersects_outer`.
   - `overpass_utc` is the median `time_ns` of the cached AOI subset
     (`time_source=subset_median`).
   - Without a subset it is the granule start (`granule_start`). An ATL03 granule
     spans about 7 minutes and a GEDI granule about 90, so these 7 times are coarse.
2. **Catalog**: one Data API quick search per calendar year over the outer AOI.
   - Filters: `PSScene`, scene `cloud_cover <= 0.5`, and items the account can
     download.
   - The standard-quality filter is off (`standard_quality_only: false`). Every
     scene over this site is `quality_category=test` with `ground_control=false`:
     open ocean has no ground control points. With the filter on, the search
     returns zero scenes. The quality fields are recorded per match.
   - Raw results are cached in `data/<site>/imagery/planet/<run_id>/catalog/`.
3. **Per-AOI cloud**: the Data API coverage endpoint (`coverage_mode: UDM2`)
   gives the clear percentage inside the `match_aoi`. That is the
   2 km core, where the island is. Scene-wide
   `cloud_cover` describes a ~25 × 12 km scene and cannot be trusted for a
   2–7 km target.
   - Estimates are cached in `coverage_cache.json`, keyed by AOI hash and item id.
   - They are only requested for items within 30 days of a target.
4. **Strips**: scenes sharing a `strip_id` are combined. Footprint coverage is
   the union. AOI clear is the area-weighted mean.
5. **Match** (`planet_matches.csv`): the closest strip with core footprint >= 0.95
   and core clear >= 0.80. Orders are still clipped to the 7 km disk
   (`clip_aoi`), which a strip may cover only partly.
   - Tiers: `same_day` ±12 h (by time, not calendar date), `d1`, `d3`, `d7`, `d30`.
   - `dt_hours` is signed: image time minus lidar time.
   - `near_miss_*` holds the closest rejected strip and why it was rejected
     (`cloud`, `footprint`, `no_estimate`).
6. **Order plan** (`planet_order_plan.csv`): unique strips across targets.
   - Bundle: the best one every item in the strip supports, in the order
     `analytic_8b_sr_udm2` → `analytic_sr_udm2` → `analytic_udm2` →
     `analytic_3b_udm2`. The last is for Dove-Classic scenes that only have RGB
     assets, such as 2021-09-24.
   - Quota estimate: the area where the strip overlaps the 7 km disk (at most about
     154 km²).
7. **QC** (`planet_scene_qc.csv`, `figures/planet/`): measured UDM2 band-1
   clear fraction over the core and outer AOIs, and quicklooks with the ATL03
   core centerlines and GEDI footprints overlaid.

## Caveats
- **UDM2 and pumice or glint:** after the August 2021 eruption, pumice rafts,
  discoloured water and sun glint may be classed as haze or cloud. Before
  concluding that no clear image exists, look at the near misses around
  2021-08-13 to 2021-12.
- **Timing:** PlanetScope passes at roughly 09:30–11:30 local time (about
  00:30–02:30 UTC here). Lidar passes at other times of day cannot have a
  `same_day` image closer than a few hours.
- **Sensor generations:** sensors changed over 2018–2022 (Dove-Classic → Dove-R →
  SuperDove). SR bundles are harmonized to Sentinel-2. `analytic_udm2` strips are
  top-of-atmosphere radiance and must not be mixed with SR in quantitative work.
- **Geolocation:** all scenes are `test` quality without ground control, so
  their geolocation may be off by tens to hundreds of metres. Check it against a
  fixed feature, such as Minami-Ioto about 6 km SSW, before overlaying
  lidar-derived outlines.

## Results (search 2026-10-20, run fukutoku_okanoba_20261019T120708Z)
- **Targets:** 89 (66 ATL03, 23 GEDI L2A).
- **Catalog:** 1,358 PSScene items with scene cloud <= 0.5. By year: 2019 4,
  2020 29, 2021 122, 2022 327, 2023 427, 2024 334, 2025 66, 2026 (to Aug) 49.
  There are none in late 2018. Planet rarely imaged this ocean before 2021.
- **Rule change:** the first run judged footprint and clear over the 7 km disk.
  Footprint was the main reason for rejection (52 of the near misses), so the
  rule moved to the core: 552 strips judged and 0 failed coverage estimates.

| tier | same_day | d1 | d3 | d7 | d30 | unmatched |
|---|---|---|---|---|---|---|
| all targets (89) | 7 | 5 | 15 | 11 | 20 | 31 |
| usable-core ATL03 (44) | 4 | 4 | 8 | 5 | 8 | 15 |

- **Unmatched:** 25 of the 31 are from 2018–2020, when imagery is sparse.
- **Near misses:** 20 rejected for footprint and 17 for cloud.
- **Elevated-return dates:**
  - 2021-09-10 matches only at +333 h (d30, the same 0.88-clear strip as other
    September targets). Its nearest rejected strip, at −50 h, is 99% clear but
    does not cover the core.
  - 2021-09-21 matches at +58 h (d3).
- **Order plan:** 48 unique strips. 46 are `analytic_8b_sr_udm2`, 1 is
  `analytic_sr_udm2`, and 1 has no downloadable analytic bundle.

| order up to | strips | ~km² | usable-core only: strips | ~km² |
|---|---|---|---|---|
| same_day | 7 | 1,018 | 4 | 563 |
| d1 | 12 | 1,730 | 8 | 1,122 |
| d3 | 25 | 3,642 | 15 | 2,165 |
| d7 | 34 | 4,978 | 20 | 2,888 |
| d30 | 47 | 6,826 | 27 | 3,953 |

### Orders (placed 2026-10-20, up to d7)
- **Ordered:** 35 strips, about 5,106 km² (budget 5,200), chosen with the
  `estimate` coverage mode. 34 are `analytic_8b_sr_udm2` (harmonized). One is
  `analytic_3b_udm2`: 2021-09-24, the Dove-Classic RGB strip that matches five
  eruption-period targets.
- **Downloaded:** all 35 succeeded and passed MD5 checks. That is 51 scenes and
  6.1 GB in `data/<site>/imagery/planet/scenes/`. Order IDs are in `orders.json`.

### Coverage-mode finding
- The default `estimate` mode overstated clear sky, worst under light haze. For
  2024-08-02 it gave 91% clear over the core, while the downloaded UDM2 shows
  about 80% light haze and 20% cloud (0.9% clear).
- `mode=UDM2` uses no quota and agrees with the measured UDM2 core-clear
  fraction to within 0.05 for 34 of 35 strips (the other differs by 0.065).
- The search now defaults to `UDM2` mode. Coverage results are cached per mode.
- **Re-run with UDM2 mode** (core >= 0.95 footprint, >= 0.80 clear):

  | tier | same_day | d1 | d3 | d7 | d30 | unmatched |
  |---|---|---|---|---|---|---|
  | all targets (89) | 3 | 4 | 10 | 7 | 12 | 53 |

  All 24 targets still matched within d7 use strips that were already ordered,
  so there are no better strips to order within ±7 d.

### Measured QC (`planet_scene_qc.csv`, `figures/planet/`)
- **Core clear:** 21 of 35 strips are at least 80% clear, 8 are 60–80%, and 6 are
  under 60%. The worst are 2024-08-02 (0.9%), 2022-12-03 (32%) and
  2024-11-27 (32%).
- Strips under 80% are kept. Partial views can still be useful for outlines;
  check the quicklook before using one.
- An earlier QC bug merged UDM2 with nodata=0 (0 is a valid UDM2 value). It is
  now fixed by recoding blackfill to 255 before merging.
- **2021-09-24:** core clear is 80%. The quicklook shows the island as a dark
  crescent, crossed by the ATL03 tracks of 2021-09-10 and 2021-09-21.
- **2021-11-09:** a discoloured-water plume is visible around the vent.

