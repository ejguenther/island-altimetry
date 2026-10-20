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
