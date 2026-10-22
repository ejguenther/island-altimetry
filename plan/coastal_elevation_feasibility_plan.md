# Coastal topobathy elevation feasibility experiment — plan for Claude

## Purpose

Implement a small, interpretable experiment in the existing `island-altimetry` repository to test whether PlanetScope imagery can help estimate coastal terrain elevation from sparse elevation labels. Use an existing airborne topobathymetric lidar DTM as the reference and as the source of simulated training labels.

The user will identify the first AOI and the DTM location. Do not choose a different site or acquire other AOIs without direction. Start with one AOI only; geographic transfer is a later question.

Primary questions:

1. Can a simple imagery-based random forest predict elevation in spatially withheld parts of this AOI when training elevations are plentiful?
2. How does performance change when labels are restricted to simulated satellite-lidar-like tracks?
3. Do image features improve prediction over simple sparse-height interpolation?

These are controlled reconstruction experiments, not instrument simulations or demonstrations of temporal change detection. A negative result for this model does not disprove all imagery-based reconstruction methods.

## Agreed constraints and working defaults

- Work in a **new branch of `island-altimetry`**. Suggested branch: `experiment/coastal-elevation-feasibility`.
- RHEL workstation; Python `venv`; Python scripts and reusable modules only, no notebooks.
- Approximately **40 GB currently available RAM**, 64 GB total. Target a conservative working budget, initially about 20 GB for the entire experiment process tree, and inspect actual availability before execution.
- Reuse existing PlanetScope acquisition scripts, credentials, storage conventions, logging, and configuration where practical. Do not implement a second downloader.
- Use one user-selected AOI, one suitable PlanetScope scene initially, and dense-label versus simulated-track-label training.
- Include simple interpolation and other inexpensive diagnostic comparisons before considering more complicated models.
- Defer additional AOIs, actual ICESat-2/ATL24/GEDI acquisition or modeling, CNNs, background-DEM predictors, and volcanic-island transfer.
- Working target: **mean terrain elevation within each prediction pixel**, in meters in the reference DTM's documented vertical datum. This is elevation, not water depth relative to the image-date sea surface.
- Start with separate land and water regressors if both domains have enough valid data, using an imagery-derived surface mask. Report shoreline/ambiguous pixels separately. Inadequate coverage of one domain must remain visible rather than blocking the other.
- Do not prescribe mandatory milestone review stops or unattended completion. Coordinate execution with the user; milestone dependencies determine technical order.

## Experiment matrix

Use a fixed validation/test layout for all experiments. Run land and water evaluations separately.

| ID | Method | Training information | Role |
| --- | --- | --- | --- |
| B0 | Constant prediction | Mean elevation from the corresponding training subset and surface domain | Sanity baseline; fit separately for dense and sparse labels |
| B1 | Sparse-height interpolation | Only simulated-track labels and their coordinates | Tests what sparse heights alone can recover |
| R1 | Random forest, dense labels | Spectral bands plus elevations sampled throughout the training region | Optimistic benchmark for this model and image |
| R2 | Random forest, sparse labels | Same bands/model settings, labels limited to simulated tracks in the training region | Isolates spatial sampling constraints |
| R3 | Random forest with simple context | Bands plus predefined ratios and local statistics, using the same sparse labels | Tests inexpensive feature engineering |
| Optional R4 | Interpolation plus learned residual | Sparse heights and imagery; residual training must be cross-fitted | A later simple extension if R2/R3 and B1 justify it |

R4 is not required for initial completion. No CNN work is needed to finish this plan. R1 is a diagnostic benchmark, not a mathematical upper bound. Do not tune models to make a preferred method win.

## Reference grid and sampling rules

1. Preserve the original 1 m DTM unchanged. Read it by windows or a virtual mosaic when possible.
2. Use the selected PlanetScope scene as the grid template if it is a suitable projected, approximately 3 m product. Match its CRS, affine transform, extent, and dimensions exactly. Confirm the actual metadata; do not silently relabel a different resolution as 3 m.
3. Aggregate DTM elevations by area-weighted mean into the target cells. Do not simply select a central 1 m pixel or upsample the image to 1 m and claim 1 m prediction detail.
4. Produce valid source-area fraction and within-cell elevation standard deviation on the same grid. Default minimum valid fraction: 0.90, configurable. The fraction denominator is the full destination-cell area, including source gaps. A separately warped valid-data mask must represent gaps as zero, not ignored NoData.
5. Keep NoData distinct from valid zero and negative elevations. Check vertical units, datum consistency between tiles, and whether submerged cells are actual bathymetric terrain, interpolation, hydroflattening, or water-surface values.
6. Grid alignment is not geolocation correction. Inspect image/reference correspondence using available stable land features. Document positional uncertainty; do not align changing shorelines by force or use validation elevation errors to optimize a shift.
7. Simulated sparse labels are **selected 3 m reference cells intersected by track centerlines**, not synthetic photons. Record one label per unique cell even if multiple tracks cross it. This simplifies spatial sampling and does not reproduce the real laser footprint or segment support.
8. Keep the 1 m DTM available for later support/footprint experiments, but do not add those to the initial scope.

## Spatial split and leakage controls

- Define the split before extracting training labels or tuning models. Store split geometry, cell IDs, random seeds, and exclusion buffers.
- Suggested starting proportions: approximately 60% training, 20% validation, 20% final test among usable cells, adjusted for AOI geometry and both surface domains. Spatial independence and adequate representation matter more than exact percentages.
- Use several spatial blocks or coherent regions, not a random per-pixel split. Avoid a fine checkerboard that puts every test pixel immediately beside training data.
- Put excluded buffers between training and held-out regions. Set buffer width in meters based on feature-neighborhood radius and local spatial dependence; it must exceed the largest feature radius. Document the chosen value and rationale without tuning it on final test performance.
- Require adequate land/water and elevation-range coverage in training, validation, and test; use descriptive reference summaries to diagnose split support, not to tune prediction accuracy. Flag extrapolation beyond training elevation ranges rather than silently dropping difficult test cells.
- Track simulations may span the AOI for visualization, but labels are extracted **only from the training partition**. Validation/test track crossings never enter fitting or interpolation. Count any repeated grid cells once.
- Freeze a common final evaluation mask for cross-method metrics. If interpolation cannot predict everywhere, additionally report coverage and compare all methods on a declared common-supported subset. Do not improve scores by silently restricting evaluation differently for each method.
- Image-neighborhood features must not use reference elevations. Exclude neighborhoods crossing held-out boundaries for the main split, or apply an equivalent documented guard, so contextual feature windows do not overlap training and held-out samples.
- Do not use x/y coordinates as random-forest predictors initially. Interpolation necessarily uses coordinates; that is a distinct baseline.
- Reference DTM elevations may be used for labels, evaluation, and explicit diagnostic stratification. Never use reference slope, elevation-derived shore distance, or the sign of reference elevation to route pixels into land/water prediction models.
- Hold final test labels out of model/feature/track-density selection. Use validation for decisions. If the test influences redesign, document that it has become development data and establish a new untouched holdout when possible.

## M0 — Create branch and inspect the existing implementation

Tasks:

- Read repository instructions and inspect the working tree before edits. Preserve uncommitted work and running data jobs. Never reset, clean, or stash unrelated work automatically.
- Create the new experiment branch from an appropriate base containing the existing PlanetScope downloader. Record the base commit. If checking out would disturb an active download or dirty tree, use a sibling worktree with the new branch and configure explicit paths to reusable data.
- Inspect existing `venv`, Python version, dependencies, downloader, AOI/config conventions, and output layout. Extend existing patterns instead of rebuilding infrastructure.
- Obtain the user-selected AOI/DTM paths if not already supplied. Implement independent setup while awaiting them; do not launch site-dependent acquisition on an invented location.
- Check available RAM, cores, and disk. Establish resource limits, a run manifest, and experiment configuration.

Deliverables: branch/worktree, implementation notes, resolved environment instructions, and draft experiment config.

Acceptance: existing workflows are preserved; branch/base are recorded; data and credentials are excluded from Git; no undefined AOI is treated as resolved.

## M1 — Audit DTM and select coincident imagery

Tasks:

- Inventory only relevant AOI DTM tiles and metadata: source survey, actual acquisition date/range, CRS, vertical datum/units, resolution, NoData, coverage, and quality limitations. DTM publication date is not its acquisition date.
- If multiple surveys overlap, select a consistent survey or document a deterministic priority rule; do not mosaic inconsistent dates/datums indiscriminately.
- Reuse existing Planet search/download scripts to find one suitable analytic scene covering the chosen experiment crop. Prefer a consistent reflectance product and band configuration; avoid visual RGB products.
- Search initially near the lidar survey dates, inspect metadata and previews, and expand the date window only when needed. Choose based on temporal plausibility, cloud/haze/shadow, glint, water clarity, coverage, and registration quality. Record exact date offsets and selection rationale.
- Review imagery quality masks carefully: water can be a valid study surface, not an automatic exclusion. Preserve reflectance scaling, band names, acquisition time, sensor generation, and geometric-quality metadata.
- Verify that land terrain and submerged features are plausible targets. Mask vegetation/buildings where necessary because the reference is bare-earth terrain; optical appearance is not automatically a ground measurement.
- Identify a modest crop within the selected AOI if full coverage would be excessive. The crop should include useful relief and shallow-water variation, not only near-zero elevations.

Deliverables: source manifest, chosen scene, AOI/crop, and concise suitability audit including unresolved datum/date/registration issues.

Acceptance: source provenance and image quality are known sufficiently to define a controlled experiment. Missing metadata or unavailable image coverage is reported as a limitation, not silently assumed away.

## M2 — Build the aligned reference and quality masks

Tasks:

- Generate the mean-elevation target, coverage fraction, and within-cell elevation standard deviation on the exact Planet grid. Use windowed processing; avoid full 1 m mosaics in RAM.
- Build image-valid and reference-valid masks. Create an imagery-derived land/water/ambiguous classification with a simple documented rule and visual inspection; do not expect NIR/NDWI to perfectly separate all shallow water, wet sand, and dark land.
- Buffer ambiguous shoreline/surf pixels as needed and report the excluded area. Retain separate masks so exclusion can be understood and later relaxed.
- Create overlays and transect plots to check extent, coordinate order, registration, NoData, and elevation distributions. Any applied image shift must have independent justification and be frozen before evaluation.
- Record valid-cell counts by domain, target resolution, vertical reference, and units. Keep image-date water level distinct from the vertical datum. Do not convert elevation to depth by assuming datum zero is the instantaneous sea surface.

Deliverables: aligned rasters/masks, reference-grid specification, preprocessing config, and QA figures.

Acceptance: exact grid equality is checked programmatically; valid-data fractions handle gaps correctly; no zero/negative elevation is lost through NoData handling; land/water routing uses imagery rather than reference terrain.

## M3 — Freeze splits and generate dense and sparse label sets

Tasks:

- Implement the spatial splits and buffers above. Publish split maps and reference-distribution summaries before model fitting.
- Dense means spatially distributed labels throughout the training regions, not necessarily every cell. Set a configurable, reproducible sample cap (initially up to about 200,000 total training cells, smaller for the smoke run); balance land/water sufficiently without changing the evaluation population.
- For sparse labels, use configurable straight parallel centerlines with origin, azimuth, spacing, and count. Start with a sparse layout with roughly 3 track centerlines across the crop, then moderate/denser layouts (roughly 6 and 12) as scale permits. These are experiment settings, not claims about real ICESat-2 beam spacing or revisit coverage.
- Record meters of spacing, number of selected unique cells, available height range, and training-domain coverage. Actual sparsity is determined by geometry and counts, not a layout name.
- Use approximately three predefined translations/orientations per density if computationally affordable. Fix the variants in configuration before scoring; do not cherry-pick the most favorable layout. Very small domains may require adjusting the proposed counts.
- For a cleaner dense-versus-sparse comparison, include each sparse label set in the dense set when within the cap. Optionally add a count-matched spatially distributed sample to distinguish label count from track geometry; this is inexpensive but not required.
- Preserve native cell IDs and duplicate checks. Sparse training cells and held-out cells must be disjoint.

Deliverables: split definitions, track GeoJSON, label tables/cell indices, distance-to-training-label rasters or sampled distances, and sampling summary.

Acceptance: training labels occur only in training regions; layouts and split seeds are reproducible; no repeated cell is an independent example; evaluation regions are identical across dense and sparse runs.

## M4 — Run sanity checks and interpolation baselines

Tasks:

- Fit training-mean baselines separately for each domain and label regime.
- Implement a practical sparse-height interpolation baseline, initially inverse-distance weighting with a spatial index and bounded neighbors. Use only training track labels; predict in chunks. Avoid a dense pairwise-distance matrix.
- Restrict land/water interpolation to corresponding training-domain labels where practical. Record unsupported cells and distances; avoid blending through ambiguous shoreline regions without disclosure.
- Select a minimal interpolation configuration on validation data. Map nearest-label distance and where predictions involve poorly supported extrapolation. A nearest-neighbor comparison is optional.
- Run a small end-to-end fixture/smoke experiment before processing the whole crop. Check interpolation behavior at known coordinates and confirm excluded evaluation labels cannot affect outputs.

Deliverables: B0/B1 validation predictions, coverage maps, interpolation parameters, and initial diagnostic profiles.

Acceptance: predictions use no held-out elevations; runtime/memory are practical; failures, missing predictions, and poor spatial support remain visible.

## M5 — Fit random-forest benchmarks

Tasks:

- Use `RandomForestRegressor` or the established repository equivalent. Start with a modest constrained forest, for example 200 trees, bounded depth (initially 20), minimum leaf size (initially 5), and about 4 workers. Treat these as resource-conscious starting settings, not optimized scientific choices.
- Use float32 image feature arrays where appropriate and prediction chunks. Keep cell indices/IDs exact. Preserve masks rather than feeding arbitrary fill values into the model.
- R1/R2 first use spectral bands only and the same declared model settings. For the primary comparison, avoid separate large hyperparameter searches for dense and sparse cases.
- R3 adds only a small predefined feature set: a few safe band ratios/normalized differences and local means/standard deviations in 3-by-3 and 5-by-5 windows. Handle near-zero denominators and invalid neighbors explicitly. Band choices depend on the actual scene; no assumptions about absent bands.
- Fit separate land/water regressors. If a domain has insufficient labels or elevation-range coverage, mark that model unsupported. Do not silently substitute reference-derived masks or fabricate missing training data.
- Fit and inspect predictions on validation regions. Use a small declared parameter comparison only if necessary; no broad search or architecture escalation.
- Save model, feature order/definitions, data/config hashes, dependency versions, sample counts, random seed, and run IDs. Predict the valid AOI in windows for maps, including a mask identifying train/validation/test/excluded areas.

Deliverables: R1/R2/R3 models and predictions, validation tables, feature/config records, and memory/runtime summaries.

Acceptance: a complete dense/sparse comparison runs within the resource budget; imagery features are the only RF predictors; no hidden reference-derived covariate or held-out label enters fitting.

## M6 — Evaluate the frozen experiments and interpret feasibility

Tasks:

- Freeze model/feature choices using validation, then evaluate the declared final experiments on the untouched test partition. Predeclared layout variants may all be evaluated, but do not choose a winning layout on the test set.
- Report bias (prediction minus reference), MAE, RMSE, sample counts, and prediction coverage separately for land, water, and any retained ambiguous pixels. Combined scores are secondary.
- Show error by reference elevation/depth-related strata, nearest-training-label distance, within-cell terrain variability, and sparse layout. Reference elevation bins are diagnostic strata, not predictors; submerged elevation magnitude is not necessarily image-date water depth.
- Report uncertainty across layouts descriptively. Do not treat millions of correlated pixels as millions of independent replicates.
- Plot reference/predicted/error maps on consistent color scales, density scatterplots, and several fixed profiles crossing withheld regions. Include split and track geometry on the maps.
- Compare R1 vs R2 for sampling effects, R2 vs R3 for feature effects, and imagery-based models vs B1 for information beyond interpolation. Distinguish supported common-domain comparisons from full-domain coverage limitations.
- Optionally calculate integrated elevation-error volume over a fixed, fully supported test domain as a secondary diagnostic. Label it reconstruction error relative to the reference, not measured erosion or temporal volume change.
- State the practical verdict: promising locally, sampling-limited, information/model-limited, data-quality-limited, or unresolved. Avoid a single arbitrary universal RMSE success threshold; relate errors to relief, baseline improvement, spatial support, and intended future use.

Deliverables: figures, CSV metrics, layout comparison, and `feasibility_report.md`.

Acceptance: all comparisons have the same declared evaluation support or explicitly report differences; land performance cannot be hidden by good water performance; no claim of cross-AOI or temporal generalization is made.

## M7 — Optional inexpensive extensions and handoff

Complete the core report before expanding. Choose only a small extension justified by an unresolved result:

- Count-matched distributed labels to separate sample number from track placement.
- A slightly larger sparse-label set to reveal whether a sampling bottleneck eases.
- Deliberate image-to-label offsets (e.g. 3 and 10 m in predefined directions) to examine positional sensitivity; maintain the original split and report as perturbations, not estimated scene errors.
- R4 interpolation-plus-residual modeling. Construct residual labels using spatially cross-fitted interpolation that excludes a block/segment around each held-out training location. Never compute residuals from an interpolation surface containing the point itself. Use all permitted training labels for the baseline surface only at final prediction time.

Do not turn this phase into CNN development, additional-AOI acquisition, or actual ICESat-2/GEDI work.

Finalize README commands, output index, source/processing manifest, parameter defaults, and instructions for rerunning from cached inputs. Summarize what a second AOI or actual satellite labels would test later. Leave merge/push decisions to the user's normal Git workflow; do not merge to main automatically.

Acceptance: the one-AOI experiment is reproducible through scripts/configuration, with a clear explanation of what worked, what failed, and the smallest useful next experiment.

## Resource and engineering guidance

- Adapt filenames to the repository; suggested script responsibilities are `audit_coastal_inputs.py`, `prepare_coastal_reference.py`, `make_spatial_splits.py`, `simulate_label_tracks.py`, `run_interpolation.py`, `train_coastal_rf.py`, and `evaluate_coastal_models.py`. Shared logic belongs in small reusable modules.
- Suggested config keys: AOI/DTM paths, scene ID, survey dates/datum, crop, output root, masks/thresholds, grid template, split geometry/buffer, dense sample cap, track layouts/seeds, features, forest limits, interpolation settings, workers, memory target, and run ID.
- Use the existing dependency workflow and `venv`; record added package versions. Likely additions include rasterio, scipy, scikit-learn, and joblib if not already present. Verify compatibility with the actual RHEL/Python setup rather than pinning guessed versions.
- Keep raw data read-only; store derived rasters, cached label/features, models, and reports separately. Use tiled/compressed GeoTIFFs, compact tables, and JSON configuration/provenance. Keep large data/model artifacts out of Git.
- Parallelize at one level at a time. Do not run multiple fully parallel forests together. Measure a small fit before raising sample size, depth, or workers; reduce the crop/sample cap when needed rather than consuming all available RAM.
- Atomic writes and run/config hashes should prevent partial results or stale caches from appearing complete. Reuse valid existing downloads and processing outputs.
- Focus validation on scientific failure modes: exact grids and resampling, coverage fractions, NoData/zero handling, vertical units, split/feature-window separation, disjoint cell IDs, reproducible sampling, interpolation leakage, and metric-count reconciliation. Avoid an elaborate test suite that merely mirrors implementation.

## Completion checklist

- [ ] New branch/worktree preserves existing `island-altimetry` work.
- [ ] User-selected AOI and source DTM/Planet scene are documented.
- [ ] 3 m working-grid reference and quality masks pass QA; original 1 m DTM is preserved.
- [ ] Spatial train/validation/test masks and guard buffers are fixed.
- [ ] Dense and multiple simulated sparse-track label sets are reproducible.
- [ ] Constant and interpolation baselines run.
- [ ] Bands-only dense/sparse RF and one sparse contextual-feature RF run.
- [ ] Land/water results, distance-to-label diagnostics, and layout sensitivity are reported.
- [ ] Final evaluation is held out from development; coverage differences are explicit.
- [ ] Report and scripts support a decision about the next experiment without requiring CNNs, a second AOI, or actual satellite-lidar acquisition.

## Reference documentation

Consult the installed tool versions and source metadata during implementation:

- GDAL average resampling: https://gdal.org/en/stable/programs/gdalwarp.html
- Rasterio exact-grid reprojection: https://rasterio.readthedocs.io/en/stable/topics/reproject.html
- PlanetScope products and metadata: https://docs.planet.com/data/imagery/planetscope/
- Random forest parameters: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

The reference DTM supplies the experiment's labels and evaluation reference. It remains a measured/processed product with uncertainty, not perfect ground truth. This experiment measures local spatial reconstruction under the selected scene and survey conditions; it does not establish recovery of changing volcanic topography.
