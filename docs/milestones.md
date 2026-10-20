# Milestone completion notes

## M0 — environment and product contracts
See `docs/m0/NOTES.md`. User accepted (2026-10-19): GEDI L2A via v002 (v003
unavailable through SlideRule); ATL24 as an independent product (spatial/temporal
association only); ATL09 labeled a partial extraction.

## M1 — site geometry and temporal scope
- Run `fukutoku_okanoba_20261019T120708Z`, query end frozen at 2026-10-19T12:07:08Z.
- Core 2 km / outer 7 km geodesic disks (vertex distances verified to <0.5 m);
  AOI hash `edec330ff7fcf425` reproducible across rebuilds.
- Events sourced from NASA Earth Observatory (onset ~2021-08-12T21:00Z;
  new island confirmed 2021-08-15) and the reference study (Aug 2021). No
  disappearance date (unsourced). GVP page returned HTTP 403 at access time.
- Minami-Ioto (~6.0 km from center) lies inside the outer buffer.

## M2 — inventory
- SlideRule discovery and the read-only CMR check agree exactly for all products
  (66/57/121/53/36 by AOI footprint search).
- Catalog footprints differ between products, so AOI search alone misses some
  partners (e.g. ATL03 2021-12-09 RGT 1199 c13 found only via its ATL08). A
  cross-product completion step looks partners up by exact name in CMR and labels
  them `cmr_name_lookup`. Final candidates: ATL03 67, ATL08 66, ATL09 125,
  ATL24 54, GEDI L2A 36.
- ATL03 RGTs: 429 and 1199 throughout; 1374, 254, 757, 871 appear only
  Sep 2021–Mar 2022 (cause not asserted).
- ATL24 v002 has no granules after 2024-10-14 (13 later ATL03 passes lack ATL24).
- ATL03 2025-03-04 (RGT 1199 c26) has no published ATL08 or ATL09 granule.
- Latest observations: ICESat-2 2026-07-12 (~99 d before run end); GEDI v002
  2025-04-24 in this AOI. A mid-2024 ICESat-2 gap is visible; not explained here.
- Discovery quirks: server returns `[]` for empty windows; cap exceedance raises
  an explicit error (windows split automatically; no splits were needed).

## M3 — diagnostic samples and schema validation
See `docs/m3/NOTES.md`. All validation checks pass; full run is estimated at
~1.3 GB (≤2.8 GB) and ~2.5–3 h with 2 workers. Implementation fixes found here:
the client truncates long output paths, so server output now goes to a short
temp path first; and a missing ATL09 partner is classified `no_granules`.

## M4 — full acquisition
Run `fukutoku_okanoba_20261019T120708Z`, 2 workers, ~2.5 h, ~750 MB cache.
Ledger (ok / no_samples_in_aoi / no_granules / retrieval_failed):
ATL03 62/5/0/0, ATL08 42/18/0/6, ATL09 61/5/1/0, ATL24 50/4/0/0, GEDI L2A 26/10/0/0.
- A resume pass skipped every verified chunk and retried only the failures.
- The 6 ATL08 failures are deterministic: the server reports
  `H5Coro::Future read failure on gtX/land_segments/latitude` for some beams and
  then drops the whole dataframe. The same 6 failed 8 attempts across two runs,
  and requesting only the other beams fails the same way, while known-good
  granules still return. Likely those beams have no `land_segments` group (ocean
  only); not confirmed, since direct granule download is out of scope. They stay
  `retrieval_failed` (never zero); the ledger has an `alerts_summary` column.

## M5–M7 — summaries, figures, comparisons, report
- Reconciliation passes for all products (view rows == ledger rows).
- 44 usable ICESat-2 dates (coherent core surface >= 100 m); 16 pre-onset, 28 post.
- Elevated coherent returns (> 5 m above the geoid) only on 2021-09-10 (RGT 1199,
  max 28.7 m) and 2021-09-21 (RGT 1374, max 23.1 m); their elevated bins coincide
  within ~1 m, ~150–270 m west of the GVP point. Pre-onset maximum 3.2 m.
- Revisits within 30 m of those bins (2021-11-11 and 2022-02-09 RGT 757,
  2022-03-22 RGT 1374) show only 0.7–1.4 m median above the geoid.
- GEDI did not sample the feature: the nearest baseline post-onset footprints
  were 554 m (2021-09-09) and 185 m (2021-09-19) away.
- Consistency (not accuracy): ATL08−ATL03 median 0.00 m (MAD 0.10);
  ATL24 sea surface−ATL03 0.01 m (MAD 0.01–0.02); GEDI−ICESat-2 (<=30 m, <=30 d,
  481 pairs) +0.10 m buffer / +0.23 m core, tides/DAC not applied.
- The report computes revisits of the elevated locations directly; most of the
  227 post-onset repeat pairs repeat the sea surface.
