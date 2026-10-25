# CNN terrain feasibility: center-label training and fully convolutional inference

## Assignment for Claude

Extend the existing `island-altimetry` repository on a new branch to test whether spatial context improves above-water terrain reconstruction relative to the existing random forest experiments. Use eight coastal AOIs with airborne topobathy lidar DTMs and approximately coincident PlanetScope imagery. Train a PyTorch model on image patches with **only one elevation target at each patch center**, then apply the exact same model fully convolutionally to produce elevation rasters.

The previous RF experiment was promising for bathymetry and weaker for land. The present experiment focuses on **above-water bare-earth terrain**, not bathymetry. Preserve the existing RF pipeline and rerun comparable baselines where needed; do not assume previous numerical results used the new split or inputs.

## Confirmed scope and resources

- Existing repository: `island-altimetry`; suggested new branch `experiment/cnn-terrain-fcn`.
- RHEL workstation, existing Python `venv`, scripts and modules only; no notebooks.
- GPU: NVIDIA RTX A4500, reported 20,470 MB VRAM. Verify free VRAM, driver, CUDA support, and compatible PyTorch installation at execution time.
- Host RAM: 64 GB total, previously about 40 GB available. Work conservatively; do not preload all raster patches or exhaust available memory.
- Eight AOIs selected from twelve available. The user identifies the relevant AOI/DTM paths or site list. If unspecified, propose a documented eight-site subset from the existing inventory based on data suitability and geographic independence, then resolve ambiguous site identities before downloading.
- Reuse the existing PlanetScope acquisition workflow and authentication. Acquire additional AOI scenes near actual lidar survey acquisition dates, preferring the nearest suitable clear/low-cloud scene.
- Compare spatially distributed DTM labels (the dense-label regime) with simulated-track DTM labels (the sparse-label regime).
- Evaluate both spatial reconstruction within represented AOIs and transfer to entirely withheld AOIs.
- Defer actual ICESat-2/GEDI labels, bathymetry, additional sites beyond the eight, temporal-change claims, pretrained foundation models, and extensive architecture searches.
- Milestones are sequential technical dependencies. This plan does not impose mandatory approval stops or assume unattended completion; coordinate pacing with the user.

## Scientific question and interpretation

Does image context help predict terrain elevations at locations without training labels, and does that improvement persist on an unseen coastal site?

The CNN initially predicts absolute mean terrain elevation at the 3 m image-grid scale, in a documented common vertical reference. It receives imagery only. Sparse elevations enter as supervision; they are **not supplied as an elevation raster at inference**. This is analogous to sparse lidar training labels, not local height anchoring during deployment. Explicit height-input fusion is a separate possible future formulation.

No method can infer arbitrary terrain uniquely from reflectance. Terrain type, material, vegetation, illumination, survey timing, and site-specific correlations can determine performance. Failure of a small model does not disprove the broader approach. Success on these coastal sites does not establish transfer to a new volcanic island or recovery of elevation change.

## Required model contract

Use a small, finite-receptive-field, **stride-one fully convolutional regressor** as the first architecture. Avoid flattening, linear layers tied to patch dimensions, global/adaptive pooling, strided downsampling, and spatially global operations. Do not start with a U-Net: downsampling and padding introduce extra alignment concerns that are unnecessary for this test.

Recommended first architecture, to implement and verify:

| Layer | Input/output channels | Kernel | Dilation | Padding | Activation |
| --- | --- | --- | --- | --- | --- |
| 1 | C → 32 | 3 × 3 | 1 | valid / 0 | ReLU |
| 2 | 32 → 32 | 3 × 3 | 2 | valid / 0 | ReLU |
| 3 | 32 → 64 | 3 × 3 | 4 | valid / 0 | ReLU |
| 4 | 64 → 64 | 3 × 3 | 8 | valid / 0 | ReLU |
| 5 | 64 → 32 | 3 × 3 | 1 | valid / 0 | ReLU |
| Head | 32 → 1 | 1 × 1 | 1 | valid / 0 | linear |

All strides are 1. The receptive-field width is `R = 1 + sum((kernel - 1) * dilation) = 33` pixels. A 33 × 33 input produces a single scalar at the location of its center pixel. At 3 m posting this is a 99 m-wide context window, with a 16-pixel / 48 m center-to-center halo on each side. Report context in both pixels and meters.

For a larger H × W input, the same weights produce `(H - 32) × (W - 32)` outputs, aligned to original input indices beginning at row/column 16. No conversion, retraining, separate inference head, or per-pixel sliding loop is required.

Use fixed channel-wise training-set scaling. **No BatchNorm, GroupNorm, InstanceNorm, or spatial LayerNorm in the first implementation.** Normalization that depends on batch or spatial extent can make patch and whole-image behavior differ. Pointwise activations and fixed scalers keep the model local. Avoid dropout initially as well.

If one context comparison is warranted, use the same five-layer structure with dilations `[1, 1, 2, 3, 1]`, giving R=17. Keep the initial architecture search limited to these two context sizes. Freeze split buffers for the largest planned receptive field.

## Data contract

- One good PlanetScope analytic reflectance scene per AOI initially; no temporal stacks or visual RGB products. Record scene ID, acquisition time, band order, sensor generation, scaling, cloud/quality metadata, and survey-date offset.
- Use common available bands across all eight AOIs. Default to blue/green/red/NIR if the inventory mixes four- and eight-band scenes. Do not fabricate missing bands. Acquire consistent products when practical, and record sensor differences as a transfer limitation.
- Start the imagery search close to each survey's actual acquisition dates; prefer the nearest scene that is suitable over the selected land region. A low scene-wide cloud percentage is insufficient if clouds cover the AOI. Expand a configurable date window when needed and record exclusions/reasons rather than silently switching dates.
- Retain the original 1 m DTM. Reuse or build a valid-area-weighted mean elevation raster on each scene's exact approximately 3 m grid, plus source-valid fraction and within-cell elevation variability. Verify alignment rather than trusting matching pixel sizes alone.
- Within each AOI, imagery and targets must share CRS, affine transform, dimensions, and extent. Different AOIs may retain their own projected grids, provided meter-scale posting and band semantics are consistent; there is no need for a giant cross-site mosaic.
- Verify vertical datum, units, survey dates, measured versus interpolated terrain, and tile compatibility. Different vertical datums cannot be pooled as though they were equivalent. Use documented conversions where needed; if unresolved, mark the site incompatible for pooled absolute-elevation training.
- Use imagery-derived land/water masks and label-quality masks. Restrict loss/evaluation centers to suitable above-water terrain; context may include valid water pixels. Do not infer image-date land purely from DTM elevation >0, because datum zero is not instantaneous sea level.
- Flag or exclude buildings and substantial vegetation for the first bare-terrain test using defensible available masks or manual image interpretation. Report the retained terrain types and area; do not quietly redefine the task to easy pixels after scoring.
- Invalid reference labels outside a center do not invalidate image context. Conversely, clouds/NoData anywhere in the receptive field invalidate the center for the initial conservative implementation. Implement this consistently in patch training and dense inference, with a context-valid mask.
- Never use reference DTM values, slopes, elevation-derived shoreline distance, per-site target means, or target-derived masks as predictors. Reference-only diagnostics must remain separate from inference inputs.

## Eight-AOI evaluation design

Default site assignment, fixed before model fitting:

| Site role | Count | Use |
| --- | --- | --- |
| Fitting sites | 5 | Labels from spatial training blocks; spatial validation and test blocks held out within each site |
| Transfer-validation site | 1 | No fitting labels; used to choose settings/checkpoint for unseen-site behavior |
| Transfer-test sites | 2 | No fitting or tuning labels; final unseen-site assessment only |

AOI identifiers are not necessarily independent sites. Adjacent/overlapping AOIs from the same landscape or imagery footprint must be grouped in one site role. If eight AOIs do not provide eight independent groups, report the actual group count and revise the assignment explicitly without pretending there are eight independent experiments.

Within the five fitting sites, use coherent spatial train/validation/test blocks, roughly 70/15/15 where geography permits. Reserve exclusion buffers and enforce **non-overlapping image-context footprints across partitions**. For the 33-pixel model, centers on opposite sides must be separated enough that their 16-pixel-radius contexts do not overlap; spatial autocorrelation may warrant a larger guard. Choose guards in meters before training, record them, and use the same guards for both context sizes.

Every scene/date/tile belonging to a site stays in the same site group. Keep transfer sites out of normalization, label sampling, model fitting, feature selection, and RF fitting. Quality audits may inspect metadata and source suitability, but do not select final test sites based on model performance.

Fit input channel means/scales using only the fitting sites' training partitions. Use the same fixed image scalers for dense and sparse experiments. For optional target scaling, use only a declared common sparse-label subset available to both regimes, or use a fixed unit conversion with no fitted target statistics. Do not let dense-only labels determine the sparse experiment's target normalization. Never normalize targets per site or subtract a held-out site's mean terrain height. Convert predictions/loss summaries back to meters.

Use a predeclared checkpoint criterion, initially equal weighting of (a) mean per-site validation RMSE across the five fitting sites and (b) transfer-validation RMSE. Log both components; no hidden test-set selection. This makes the validation objective explicit without letting a large site dominate.

## Dense and sparse supervision

**Both regimes use exactly one center target per training patch.** Dense refers to the distribution of eligible labeled centers through training blocks, not dense pixelwise loss over the patch. Do not quietly use a full DTM loss in the dense regime.

- Dense: select a reproducible, spatially distributed pool of valid training centers per fitting AOI. Cap it for tractability, for example 50,000 per site initially, and report actual counts and relief coverage.
- Sparse: select unique valid centers where simulated track centerlines intersect training cells. Reuse the prior simulation conventions and, where sensible, a small predefined set of track densities/layouts. Save geometries, meter spacing, azimuth, translation, random seed, and cell IDs.
- No simulated labels in transfer sites or spatial validation/test blocks enter fitting. Labels at multiple points in the same 3 m cell count once.
- Use comparable AOI-balanced minibatch sampling so the largest site does not dominate. Sparse cells may be revisited across epochs, but report distinct centers separately from optimizer examples.
- Compare dense/sparse models using the same architecture, splits, feature bands, eligible evaluation masks, loss, and validation criterion. Keep the primary optimizer-step budget comparable, and report the amount of repeated sampling. Different label pools need not have equal unique sample counts.
- Initially use one declared sparse layout and one training seed to establish the pipeline; after freezing the model, repeat dense and sparse runs with three seeds if practical. Additional track-layout variants are a distinct sensitivity experiment, not an unreported source of cherry-picking.
- Rotations by multiples of 90 degrees and reflections may be used consistently as simple augmentation, with center targets unchanged. Record that these alter directional illumination cues; keep augmentation fixed across dense/sparse comparisons. Avoid aggressive color augmentation that changes physical reflectance relationships initially.

## M0 — Branch, GPU environment, and repository integration

Tasks:

- Read repository instructions; inspect active branch, dirty files, existing RF/Planet scripts, configuration, input manifests, and output conventions. Preserve running jobs and uncommitted work.
- Create the experiment branch from an appropriate base containing required existing functionality. Use a worktree if switching would disrupt another job. Record base commit; do not merge to main automatically.
- Verify A4500 CUDA execution with a small PyTorch tensor/convolution, versions, driver compatibility, free VRAM, and host memory. Install a compatible PyTorch build in the existing environment workflow without replacing system drivers.
- Inspect existing sample/mask/split artifacts and determine what can be reused scientifically. Do not modify earlier experiment results in place.
- Create a config with eight AOIs, product/band policy, architecture, receptive field, masks, split roles, sampling, optimization, inference tile size, and run paths. Resolve AOI IDs/paths with the user if needed.

Deliverables: working branch/environment, GPU smoke result, config skeleton, and integration notes.

Acceptance: GPU training works; prior pipelines remain reproducible; unknown AOI locations are explicit.

## M1 — Assemble and audit eight AOI datasets

Tasks:

- Inventory DTMs, survey dates, vertical references, and available imagery for the eight sites. Acquire needed Planet scenes through the existing downloader, with AOI-level quality screening and recorded date offsets.
- Establish common bands, reflectance scaling, measured terrain support, and land/context validity. Reuse aligned references only after checking their processing metadata.
- Build per-AOI mean-elevation targets and masks as needed. Inspect overlays/profiles for horizontal offsets; fix obvious processing issues before training, using independent image/reference evidence rather than final test errors.
- Record usable land area, relief distribution, mask losses, survey/scene differences, geographic grouping, and unresolved uncertainties.

Deliverables: eight-site manifest, scene selection ledger, aligned image/reference/mask products, and QA figures.

Acceptance: all included sites have compatible target definitions and common image inputs; incompatible sites are identified rather than silently included. No full imagery history is required.

## M2 — Freeze site/spatial splits, scalers, and sample pools

Tasks:

- Assign site groups to the 5/1/2 roles; generate spatial blocks and context guards inside fitting sites.
- Create immutable train/validation/test center-index tables and fixed diagnostic transects. Use windowed reads or compact indices; do not export millions of duplicated patch arrays.
- Fit permitted scalers on training data only. Implement a dataset returning normalized C×R×R image context, one normalized center elevation, AOI/cell identity, and mask metadata.
- Generate dense and sparse training pools. Ensure AOI-balanced sampling, deterministic worker seeding, unique-cell accounting, and no held-out label access in training data loaders.
- Create fixed validation subsets that represent area and relief without concentrating only on near-zero terrain. Distinguish any elevation-stratified diagnostic subset from area-representative headline metrics.

Deliverables: split maps/JSON, sample pools, scalers, track geometries, and split/sample audit.

Acceptance: no training/evaluation context footprints overlap; transfer sites cannot affect fitted preprocessing; dense/sparse runs share evaluation masks.

## M3 — Implement and verify center-label/FCN equivalence

Tasks:

- Implement the valid-convolution architecture above with a linear output head. Derive receptive field and output shape programmatically from the configured layers.
- For R×R patches, use the single output and one center-label regression loss. Default to mean squared error in normalized elevation, with meter-scale metrics tracked separately. No neighboring DTM labels enter the loss.
- Prove numerical equivalence in evaluation mode on a small image: compare selected dense outputs with outputs from the corresponding independent patches using the same weights, scalers, and valid contexts.
- Test constant/coordinate-ramp fixtures to verify output indexing and the 16-pixel geospatial offset. Confirm that changing an unsupervised neighboring label cannot alter the center-only loss.
- Test gradients, finite outputs, and overfitting of a tiny fixed training subset before running a scientific experiment.

Deliverables: PyTorch model/dataset, focused equivalence/alignment tests, and tiny-subset training curves.

Acceptance: patch and dense outputs agree to a documented numerical tolerance; output coordinates match label centers; tiny-subset loss decreases; no architecture conversion is needed for inference.

## M4 — Train the controlled experiments and comparable baselines

Run a small matrix first:

| Experiment | Fit labels | Inputs | Purpose |
| --- | --- | --- | --- |
| Constant | Matching dense/sparse permitted pool | Training elevation mean | Sanity baseline |
| RF dense | Dense | Common Planet bands; existing declared RF features if applicable | Dense-label feature-model benchmark |
| RF sparse | Sparse | Same band/feature definitions | Matched sparse RF benchmark |
| CNN dense | Dense center labels | Common Planet image context | Main context-model test |
| CNN sparse | Sparse center labels | Identical context architecture | Sparse-supervision test |
| Interpolation sparse | Sparse training centers | Coordinates/heights only | Within-site geometry baseline |

RFs must be fitted to the corresponding pooled training sites and exact center pools, or differences must be explicitly labeled. Existing local RF results may be supplementary but are not the pooled-model transfer baseline. Interpolation has no local training heights in entirely withheld sites: mark transfer interpolation unsupported rather than interpolating across distant AOIs or borrowing held-out labels. Local calibration of a transfer site is a different future experiment.

Training defaults, to adjust after a measured smoke run:

- AdamW; initial learning rate 1e-3, weight decay 1e-4; fixed starting settings across dense/sparse models.
- Initial batch size 128 for 33×33 patches; reduce if VRAM or I/O dictates. Measure rather than assume throughput.
- Mixed precision using APIs supported by the installed PyTorch; compare a small float32 reference for stability. Save scaler state if applicable.
- Approximately 50,000 sampled centers per epoch, balanced across fitting sites; maximum 50 epochs, early-stopping patience 8 by the declared validation criterion. Treat these as starting budgets, not requirements to consume all steps.
- Save the best validation checkpoint, last resumable checkpoint, optimizer/AMP state, seeds, scalers, feature order, sample/split hashes, software versions, and timing/memory summaries.
- Log train loss, within-site validation metrics, transfer-validation metrics, per-site metrics, and early stopping. Do not inspect final test scores during development.
- Start with one context size. Add R=17 only if validation results or memory/context concerns justify the comparison; no large hyperparameter sweep.

Deliverables: models/checkpoints, training history, baseline predictions, and validation comparison.

Acceptance: dense/sparse differences are traceable to supervision availability; models train on GPU within measured limits; comparable RF and CNN runs use the same scientific partitions.

## M5 — Implement efficient fully convolutional raster inference

Tasks:

- Support one-call inference for rasters/crops that fit GPU memory, using evaluation/inference mode and training-fitted scalers.
- Support larger images using **output tiles with a context halo**, not one model invocation per output pixel. With R=33, an output tile T×T needs an input window `(T+32)×(T+32)` including a 16-pixel halo on each side.
- Start with output tiles around 256 or 512 pixels and choose based on measured VRAM. Do not require a whole Planet scene to fit in one GPU allocation.
- Write only the valid output cores to their exact positions on the original image grid; overlapping input halos provide context and do not require averaging predictions. Process border tiles carefully for arbitrary dimensions.
- Preserve the original geotransform in a full-size output with NoData on the outer 16-pixel border and other unsupported centers. For a cropped valid-output file, shift its geotransform by the appropriate input-row/column offset.
- Apply a consistent context-validity mask. Output only valid above-water centers; surrounding water imagery may remain context. Do not fill missing image context with reference elevations.
- Numerically compare tiled inference against a full forward pass on a manageable crop, including tile joins. Compare several tile sizes. Test float32 first; document separately any mixed-precision differences.
- Benchmark FCN inference against a small patch-by-patch reference for timing and correctness; the slow reference is a test only.

Deliverables: reusable inference script, GeoTIFF predictions/validity masks, seam/equivalence results, and throughput/memory report.

Acceptance: whole-crop, tiled, and center-patch predictions agree within stated tolerances at valid pixels; there are no tile-boundary seams or pixel offsets; large-raster inference does not materialize all patches.

## M6 — Final within-site and unseen-site evaluation

Tasks:

- Freeze selected settings using validation. Evaluate the declared final runs on spatial test blocks of the five fitting sites and the two untouched transfer-test sites.
- Report bias (prediction minus reference), MAE, RMSE, valid evaluation area/count, and unsupported fraction **per AOI**, separately for dense/sparse and model type.
- Report equal-site aggregate metrics alongside area-weighted metrics, keeping within-site and unseen-site results separate. Eight AOIs do not justify treating all pixels as independent replicates.
- Compare paired CNN–RF and CNN–interpolation errors on identical valid cells. Report prediction coverage whenever model/context masks differ; provide common-mask comparisons explicitly.
- Diagnose error versus reference elevation, terrain variability/slope (evaluation only), distance to sparse labels within fitting sites, vegetation/land-cover strata if available, and sensor/date mismatch. Do not compute a local-label-distance claim for unseen sites that supplied no labels.
- Plot consistent reference/prediction/error maps, profiles crossing held-out relief, predicted-versus-reference distributions, and residual maps. Inspect whether the model recovers relief or regresses to a near-constant surface.
- If multiple seeds are run, report their variability rather than selecting a seed using test performance.
- Explicitly distinguish: spatial-context benefit, sparse-label penalty, unseen-site degradation, elevation-range extrapolation, registration/data-quality limits, and remaining uncertainty.

Deliverables: metric tables, diagnostic figures, per-site findings, and `cnn_terrain_feasibility_report.md`.

Acceptance: the result states whether context improves on RF under matched conditions, whether sparse training remains useful, and how much performance changes on unseen sites. Neither bathymetry success nor large low-relief areas can obscure terrain-shape failures.

## M7 — Reproducible handoff and bounded next steps

- Document exact preparation, split, sampling, training, resumption, inference, and evaluation commands; match existing repository CLI style.
- Provide config and output index, dependencies, checkpoint provenance, source scene/survey ledger, and completed milestone notes.
- Make spatial exclusions and effective label counts inspectable. Preserve old RF outputs and keep large imagery/model files outside Git.
- Recommend one next experiment based on the evidence: another context size, simple imagery-plus-height-input fusion, better registration, expanded terrain types, actual lidar labels, or later bathymetry extension. Do not implement these expansions automatically.
- No automatic merge into main. Follow the user's normal branch/review workflow.

Acceptance: someone can reproduce the frozen eight-AOI experiment and map a new compatible image with the stored weights/scalers, while understanding that new-site accuracy is not guaranteed.

## Resource management and implementation layout

- Use windowed raster reads, per-AOI caches or memory maps, and lightweight center-index datasets. Avoid holding all full scenes, all target rasters, or all extracted patches in RAM simultaneously.
- Start with 2–4 data-loader workers and bounded prefetch; inspect whether parallel raster reads multiply memory use. Use per-worker raster handles safely. Pin memory only when beneficial and within the budget.
- Reserve VRAM headroom for activations and optimizer state; initially target roughly 14–16 GB peak or less after checking other GPU processes. Start with a small representative crop/batch before scaling.
- Do not train multiple GPU jobs or fully parallel RFs concurrently by default. Run independent experiments sequentially on the one GPU.
- Use fixed seeds for Python/NumPy/PyTorch/data loaders and record reproducibility settings. Document GPU numerical tolerances instead of claiming universal bitwise reproducibility.
- Suggested modules/scripts: `prepare_cnn_sites.py`, `make_cnn_splits.py`, `sample_cnn_centers.py`, `train_terrain_cnn.py`, `predict_terrain_fcn.py`, `evaluate_terrain_cnn.py`, with shared model/dataset/inference modules. Adapt these names to existing conventions.
- Use run/config hashes and atomic outputs so partial or stale artifacts are not mistaken for completed results. Preserve site role, grid, mask, and band order in every prediction manifest.

## Required scientific checks

1. Target is one center elevation per example in both regimes; changing neighboring labels does not change the loss.
2. No train/evaluation image-context overlap within fitting sites; no site-group leakage across roles.
3. Global fixed preprocessing uses fitting-site training data only, with no per-test-site elevation centering.
4. RF/CNN comparisons use matching labels, inputs where applicable, and evaluation support.
5. Patch, crop-FCN, and halo-tiled predictions align and agree numerically.
6. No label-derived terrain predictor or prediction-time land mask.
7. NoData/cloud masking and outer borders are consistent between training and inference.
8. Raster dimensions, CRS, geotransform, vertical reference, and meter-scale outputs are verified.

Prioritize these focused tests over broad infrastructure or tests that simply mirror code.

## References for implementation

- Fully convolutional prediction: Long, Shelhamer, and Darrell (2015), https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html
- PyTorch convolution shapes/dilation: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- PyTorch mixed precision: https://docs.pytorch.org/docs/stable/amp.html
- PyTorch reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
- PlanetScope product metadata: https://docs.planet.com/data/imagery/planetscope/

Verify APIs against the installed versions. The specific regression architecture, site partition, and training budgets above are proposed experiment defaults, not architectures or performance claims taken from those references.
