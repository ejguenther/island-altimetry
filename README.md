# island-altimetry

Reproducible, configurable workflow for inventorying, acquiring, and exploring
satellite altimetry (ICESat-2 ATL03/ATL08/ATL09/ATL24, GEDI L2A) over volcanic
islands via [SlideRule](https://docs.slideruleearth.io/). First site:
Fukutoku-Okanoba (`fukutoku_okanoba`). See `plan/initial_plan.md` for the
scientific scope and rules.

## Environment (RHEL 8, no Conda, no system package changes)

```bash
/usr/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.lock.txt   # exact versions
.venv/bin/pip install -e .                        # this package
```

`requirements.in` lists direct dependencies; `requirements.lock.txt` is the
`pip freeze` of the verified environment.

Access uses the SlideRule **public cluster**; no credentials are required or
stored. Read-only NASA CMR collection metadata queries are unauthenticated.

## M0: environment and product contracts

```bash
.venv/bin/python scripts/check_environment.py
```

Writes `docs/m0/environment.json`, `docs/m0/product_support.csv`, and
`docs/m0/product_contracts.json`. Product intent (assets, endpoints, permissive
request parameters, requested fields) lives in `configs/products.yaml`; the
script records live evidence of what the server actually serves and returns.
See `docs/m0/NOTES.md` for findings and limitations.

## Workflow

All scripts take `--site <id>` (or `--all` where noted) and default to the
site's latest run. Each run is an immutable snapshot with a frozen UTC end.

```bash
PY=.venv/bin/python
$PY scripts/check_environment.py                       # M0 contracts (docs/m0/)
$PY scripts/build_aoi.py   --site fukutoku_okanoba     # M1 new run: AOIs, events, manifest
$PY scripts/inventory.py   --site fukutoku_okanoba     # M2 candidate granules + CMR check
$PY scripts/smoke_test.py  --site fukutoku_okanoba     # M3 diagnostic samples + validation (docs/m3/)
$PY scripts/acquire.py     --site fukutoku_okanoba     # M4 full retrieval (resumable)
$PY scripts/summarize.py   --site fukutoku_okanoba     # M5 coverage/quality tables
$PY scripts/plot_coverage.py   --site fukutoku_okanoba # M6 maps + timelines
$PY scripts/plot_profiles.py   --site fukutoku_okanoba # M6 photon profiles
$PY scripts/compare_products.py --site fukutoku_okanoba # M6 qualified comparisons
$PY scripts/build_report.py    --site fukutoku_okanoba # M7 feasibility.md
$PY -m pytest                                          # focused validation tests
```

### M8: coincident PlanetScope imagery (requires Planet login)
Run `.venv/bin/planet auth login` once, or export `PL_API_KEY`. Search is metadata
only and uses no quota. Orders draw down quota, so they are a dry run unless you
pass `--submit` together with a km² budget. Criteria live in
`configs/islands.yaml` under `defaults.imagery.planet`; see `docs/m8/NOTES.md`.

```bash
$PY scripts/check_environment.py --planet-only         # SDK + auth check
$PY scripts/planet_search.py --site fukutoku_okanoba   # M8a targets, candidates, matches, order plan
$PY scripts/planet_order.py  --site fukutoku_okanoba --max-tier d1 --budget-km2 3000            # dry run
$PY scripts/planet_order.py  --site fukutoku_okanoba --max-tier d1 --budget-km2 3000 --submit   # M8b
$PY scripts/planet_order.py  --site fukutoku_okanoba --download                                  # wait + download
$PY scripts/planet_qc.py     --site fukutoku_okanoba   # M8c measured UDM2 clear + quicklooks
```

### M9: lidar profiles with coincident imagery (no network)
One figure per core-crossing ATL03 or GEDI L2A pass. Each beam's height profile
sits beside a downloaded Planet strip, with the beams drawn on the image in matching
colours. The default `--pick tiered` takes the clearest strip within ±7 d, else the
clearest within ±32 d (`--tier-days 7 32`), else the temporally closest downloaded
strip at any distance. The map title names the rule used; the last fallback is shown
in red with Δt in days. `--pick nearest|clearest` limits the choice to `--max-dt-days`.

```bash
$PY scripts/plot_track_imagery.py --site fukutoku_okanoba                  # all passes
$PY scripts/plot_track_imagery.py --site fukutoku_okanoba --product gedi_l2a --max-dt-days 14
$PY scripts/plot_track_imagery.py --site fukutoku_okanoba --granule ATL03_20210921145752_13741207_007_01
# gap-fill: clearest strip within ±32 d for passes without imagery, then re-plot them
$PY scripts/planet_gapfill.py --site fukutoku_okanoba --budget-km2 3000            # dry run
$PY scripts/planet_gapfill.py --site fukutoku_okanoba --budget-km2 3000 --submit --download
$PY scripts/plot_track_imagery.py --site fukutoku_okanoba --max-dt-days 32 --pick clearest --missing-only
```
Figures go to `outputs/<site>/<run>/figures/track_imagery/` and the pairings to
`tables/track_imagery_pairs.csv`.

### Coastal elevation experiment (PlanetScope + RF vs simulated tracks)
A one-AOI reconstruction test (`plan/coastal_elevation_feasibility_plan.md`). The AOI is the
Florida Panhandle 2022 NCMP topobathy lidar, east strip, configured in `configs/coastal_experiment.yaml`.
The target is the mean elevation of class 2/29 lidar returns per 3 m Planet cell (NAVD88), because the
1 m DTM is clamped in deeper water. Land/water routing uses imagery NIR only. Spatial blocks with
150 m guards are frozen before labelling. Methods: B0 constant, B1 IDW from track labels, and
R1/R2/R3 random forests on dense or track labels.
```bash
$PY scripts/audit_coastal_inputs.py                    # tile/LAZ audit, datum, footprint
$PY scripts/planet_search.py --coastal                 # strips in the configured window (no quota)
$PY scripts/planet_order.py --coastal --strip 5730968  # dry run; add --submit, then --download
$PY scripts/prepare_coastal_reference.py               # exact Planet grid, target, support, masks, QA
$PY scripts/make_spatial_splits.py                     # frozen split + maps (before any label)
$PY scripts/simulate_label_tracks.py                   # dense + 9 track layouts
$PY scripts/run_interpolation.py                       # B0/B1 (IDW chosen on validation)
$PY scripts/train_coastal_rf.py                        # R1/R2/R3 (add --smoke for a quick check)
$PY scripts/train_coastal_rf.py --only T1,T2           # terrain features (texture/shape/orientation 3/9/15 + distance)
$PY scripts/ablate_coastal_features.py                 # drop-one-family + permutation importance (validation only)
$PY scripts/plot_terrain_features.py                   # QA figure of feature layers over a subset
$PY scripts/evaluate_coastal_models.py --partition val
$PY scripts/evaluate_coastal_models.py --partition test   # once, after choices are frozen
```
Cached inputs go to `data/florida_panhandle/{coastal,imagery/planet}/`. Results go to
`outputs/florida_panhandle/coastal_rf/` (`feasibility_report.md`, `tables/`, `figures/`). The raw DTM, DSM
and LAZ are only read. After the download, a rerun takes about 15 minutes with a peak RSS of about 5 GB.

Layout: `data/<site>/aoi/<aoi_hash>/`, `data/<site>/inventory/<run_id>/`,
`data/<site>/subsets/<product>/<release>/aoi_<hash>/req_<hash>/<granule>.{parquet,json}`
(immutable raw chunks + sidecars), `data/<site>/analysis/<run_id>/`, and
`outputs/<site>/<run_id>/{run_manifest.json, feasibility.md, tables/, figures/, plot_index.csv}`.
Planet imagery: `data/<site>/imagery/planet/{<run_id>/catalog/, coverage_cache.json, orders.json, scenes/<YYYYMMDD>_<strip>/}`.
`data/` and `outputs/` are not committed.

### Rerun, resume, update
- **Resume:** re-run `acquire.py`; verified chunks (sidecar status + sha256) are
  skipped, failed or missing ones are retried with bounded exponential backoff.
- **Regenerate analysis:** M5–M7 scripts read only local caches (no network).
- **Incremental update:** run `build_aoi.py` again to create a new run (new frozen
  end), then `inventory.py` and `acquire.py`. Chunks with identical AOI, request
  parameters and processing version are reused from the cache; changing filters or
  fields creates a new request-hash namespace instead of overwriting.
- **Overrides:** `build_aoi.py --set core_radius_m=3000 --set buffer_m=8000`.

### Adding an island
Add an entry under `sites:` in `configs/islands.yaml` (id, names, center,
sourced events, optional radius/product overrides). Circular AOIs around a center
are implemented; supplied-polygon sites are not yet supported. No acquisition code
changes are needed; see `tests/test_multisite_quality.py`.

### Outcome codes
`ok`, `no_granules`, `no_samples_in_aoi`, `all_filtered`, `service_unsupported`,
`field_unsupported`, `retrieval_failed`, `not_yet_queried`. Failures are never
recorded as scientific zeros.
