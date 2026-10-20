# M0 findings: environment and product contracts

Accessed 2026-10-18. Evidence: `environment.json`, `product_support.csv`,
`product_contracts.json` (regenerate with `scripts/check_environment.py`).

## Environment
- RHEL 8.10; `.venv` from `/usr/bin/python3.12`; deps in `requirements.lock.txt`.
- SlideRule client v5.6.3 = server v5.6.3 (public cluster, no credentials).
- Disk free ~287 GB.

## Product support summary

| Product | Route | Served release | Latest CMR release | Status |
|---|---|---|---|---|
| ATL03 | `atl03x`, asset `icesat2` | 007 | 007 | verified |
| ATL08 | `atl08x`, asset `icesat2-atl08` (standard product, not PhoREAL) | 007 | 007 | verified |
| ATL09 | `atl09_fields` ancillary on `atl03x` | 007 | 007 | **partial ATL09 extraction** |
| ATL24 | `atl24x`, asset `icesat2-atl24` (standard product) | 002 (built on ATL03 r006) | 002 | verified, see release note |
| GEDI L2A | `gedi02ax`, asset `gedil2a` | **002** | **003** | verified on v002 only |

## Findings that change implementation
1. **Silent empty results.** On failure (bad field, unreadable granule) the client
   returns an empty frame without raising. `sr_client.run` captures server
   `exceptrec` alerts and classifies `retrieval_failed` / `field_unsupported` /
   `no_granules` / `no_samples_in_aoi` / `service_unsupported` explicitly.
2. **Hidden server defaults overridden** (`configs/products.yaml`):
   - ATL03 default `quality_ph` keeps nominal photons only → set `[0,1,2,3]`;
     `cnf: -2`, `srt: -1` (dynamic), `podppd: 255`.
   - ATL24 default `class_ph` is `["bathymetry"]` only → set `[0, 40, 41]`;
     `compact: false` to get uncertainties and flags.
   - GEDI quality/degrade/surface filters default off (kept off).
3. **Discovery cap:** server `max_resources` default is 300 granules per request;
   inventory must split time windows (or raise the cap) and verify no window hits it.
4. **Field path conventions:** ATL08 `atl08_fields` are relative to
   `<beam>/land_segments/`; ATL24 ancillary fields must be nested as
   `atl24.anc_fields` (top-level `anc_fields` is silently ignored). The probe now
   requires every requested field to come back as a column.
5. **No `atl09x` endpoint** (server: "Failed to load script atl09x.lua").

## Limitations
- **ATL09 is partial.** Values are attached per ATL03 photon from the ATL09
  `high_rate` group (one value per ~25 Hz profile; ~15 distinct profiles per beam
  across the 4 km core). Native profile sampling is recoverable by de-duplicating
  on `high_rate/delta_time`. Multi-layer fields (`layer_top`, `layer_bot`, native
  shape [n_profiles, 10]) are returned as **length-1 arrays**; only one layer is
  accessible, so layer heights are reported as truncated and not used for
  multi-layer cloud context. Coverage is limited to where ATL03 photons exist;
  no independent ATL09 coverage claim is made. Beam-to-profile mapping is
  performed server-side and is not exposed.
- **ATL24 release mismatch.** ATL24 v002 granules are produced from ATL03
  release 006 (`ATL24_..._006_xx_002_01`), while ATL03 r006 is no longer in CMR
  and SlideRule serves ATL03 r007. ATL24 is therefore retrieved as an
  independent standard product; photon-level association to r007 ATL03 photons
  (`index_ph`) is not release-compatible and is not used for joins. Spatial and
  temporal (same RGT/cycle/beam) association is allowed, labeled as such.
- **GEDI L2A v002, not v003.** LP DAAC's latest release is v003; SlideRule's
  `gedil2a` asset resolves v002 granules, and direct v003 resources fail server-side
  (`H5Coro read failure on BEAMxxxx/quality_flag`). Per the plan, no silent switch
  to direct downloads: GEDI is retrieved from v002, with v003 recorded as
  unavailable through the chosen service. v002 collection ends 2025-07-10; the
  latest v003 granule is 2025-07-09, so the temporal record is effectively the same.
- **GEDI `landsat_water_persistence`** cannot be read by the server (field unsupported).
- **GEDI elevation reference:** `elevation_lm` maps to native `elev_lowestmode`
  (WGS84 ellipsoid); verified mapping and units to be re-checked in M3.
- Server-side CMR queries for `ATL03` r006 return nothing (release withdrawn).

## Probe acquisitions
ICESat-2 RGT 1199 cycle 12 (2021-09-10, `11991201`) and GEDI orbit O15692
(2021-09-19) cross the 2 km core zone and were used for field probes.
