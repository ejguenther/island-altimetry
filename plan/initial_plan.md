Volcanic island altimetry exploration: implementation plan for Claude
Objective and agreed scope
Build a reproducible, configurable Python-script workflow for inventorying, acquiring, and exploring satellite altimetry over volcanic islands. Implement Fukutoku-Okanoba first, then allow additional islands through configuration rather than island-specific code.
The immediate question is: What observations actually exist, where and when are they usable, and could they support later elevation reconstruction and elevation-change analysis? This project does not yet train an elevation model or estimate island-wide volume change.
User decisions:
Execution: RHEL workstation; Python virtual environment (`venv`).
Acquisition: SlideRule.
Products: ICESat-2 ATL03, standard ATL08, ATL09, standard ATL24, and GEDI L2A footprint-level ground elevation.
Dates: complete available mission records through the latest available observations at execution time. ICESat-2 discovery begins October 2018; GEDI uses its own actual collection start.
Geography: island vicinity plus surrounding ocean, with configurable buffers.
Implementation: Python scripts and reusable modules; no notebooks.
Deliverables: inventory, cached subsets and provenance, maps, timelines, photon profiles, product comparisons, quality summaries, and a feasibility assessment.
Imagery and independent DEM acquisition/comparison are deferred.
Milestones are sequential dependencies. This document does not impose a mandatory stop/approval policy or require unattended completion; coordinate execution with the user.
First study site
Stable ID: `fukutoku_okanoba`.
Names: Fukutoku-Okanoba / Fukutoku-Oka-no-Ba; the 2021 emerging island called Niijima in the reference study.
Reference: Simurda et al. (2022), ICESat-2 Applications for Investigating Emerging Volcanoes, DOI: https://doi.org/10.3390/geosciences12010040.
Initial center: longitude 141.481, latitude 24.285 (WGS84), from Smithsonian Global Volcanism Program volcano 284130. This is a volcano reference location, not a surveyed island centroid or shoreline.
Starting spatial defaults: a 2 km radius core search zone and a 5 km additional ocean buffer around that zone, giving a 7 km radius outer discovery AOI. These are adjustable exploration defaults, not known island boundaries.
Retain a `zone` attribute distinguishing the core search zone from the outer buffer. Neither zone is automatically a land/ocean label. Nearby Minami-Ioto may enter the outer AOI; its observations must not be described as measurements of the 2021 island.
Include pre-emergence, eruption/emergence, and later observations. Use sourced event annotations; do not assign an exact disappearance date without evidence. Keep events separate from data availability and interpretation.
Scientific and implementation rules
All available means all discoverable acquisitions in the AOI and requested period, not every global granule, historical processing version, or variable. Inventory completeness and field completeness are separate. Define a documented field selection for exploration and report fields unavailable through the chosen service.
Use the latest suitable published release per product at run start; freeze collection IDs, releases, retrieval parameters, and an absolute UTC end time in the run manifest. An incremental update creates a new inventory snapshot. Do not silently mix releases or replace cached results.
Use SlideRule for data access and supported discovery. If its catalog cannot establish collection-level completeness, a read-only NASA CMR metadata query is acceptable as an explicitly labeled discovery check, not an alternate data-download pipeline. Record the distinction. Do not silently switch to direct granule downloads when a product is unsupported.
Verify current client/server capabilities and schemas. Documentation and endpoints evolve; examples below are planning references, not guaranteed executable signatures.
Separate standard products from custom algorithms: a PhoREAL output is not the standard ATL08 product, and an on-demand bathymetry calculation is not a published ATL24 subset. Keep any optional derived products in distinct namespaces.
ATL09 is atmospheric context, not a terrain-height product. Its profiles/layers have different sampling and beam/profile relationships from ATL03. Preserve native sampling when accessible; document any service-resampled ancillary fields.
Standard land and bathymetry products may omit ocean observations or a newly emerged island. Lack of product coverage is a finding, not evidence of absent land, shallow water, or atmospheric clarity.
Begin with a permissive diagnostic subset and apply named quality screens afterward. Never require ATL08 or ATL24 availability to retain base ATL03 data.
Separate `no_granules`, `no_samples_in_aoi`, `all_filtered`, `service_unsupported`, `field_unsupported`, `retrieval_failed`, and `not_yet_queried`. An API error must never become a scientific zero.
No credentials in code, configuration, logs, or exported manifests. Inspect existing workstation/repository conventions first. Use local credentials through supported authentication mechanisms; document any interactive setup actually required.
Suggested repository and configuration
Adapt to the existing repository if one is provided. Otherwise use a small new project, not a framework or web application.
```text
README.md
requirements.in
requirements.lock.txt
configs/islands.yaml
src/island_altimetry/
scripts/
  check_environment.py
  build_aoi.py
  inventory.py
  smoke_test.py
  acquire.py
  summarize.py
  plot_coverage.py
  plot_profiles.py
  compare_products.py
  build_report.py
data/<island_id>/
  aoi/
  inventory/<run_id>/
  subsets/<product>/<release>/
  analysis/<run_id>/
outputs/<island_id>/<run_id>/
  figures/
  tables/
  feasibility.md
  run_manifest.json
```
Configuration should include site ID/aliases, center or optional supplied geometry, core radius, buffer distance, product toggles, mission-specific starts, end=`latest`, collection/release policy, requested fields, quality profiles, output/data roots, worker limit, timeout/retry settings, and comparison tolerances. Resolve `latest` once per run to UTC now; report each product's newest actual observation separately. Allow one-site and all-configured-site commands.
Default to modest concurrency and per-granule or small resource-group requests. Persist one successful chunk before starting dependent processing. Prefer Parquet/GeoParquet for point/segment subsets, CSV for compact inventories and summaries, JSON for machine-readable provenance, GeoJSON for AOIs, and PNG/PDF for plots. ATL09 multidimensional data may require a separate appropriate format or a documented field-limited representation; do not force profile arrays into misleading point rows.
M0 — Establish environment, access, and product contracts
Purpose: establish what can actually be requested before coding bulk retrieval.
Tasks:
Inspect RHEL version, available Python interpreters, existing repository instructions, available disk space, and existing SlideRule setup without exposing credentials.
Select a Python version supported by current dependencies and available on the workstation; do not assume the system Python can be replaced. Create `.venv` with that interpreter; no Conda or `uv` requirement, no system package changes.
Install a minimal stack: SlideRule client, pandas, NumPy, PyArrow, GeoPandas/Shapely, pyproj, matplotlib, configuration parsing, and only other libraries justified by the implementation. Prefer compatible wheels on RHEL; document genuine compatibility blockers.
Record environment, dependency versions, and SlideRule client/server versions.
Verify published collection names, versions, temporal extents, asset mappings, and supported fields for every requested product against current SlideRule and NASA documentation.
Produce a product-support matrix: product, collection/version, native record unit, desired variables, access route, known transformations/default filters, and unresolved limitations.
In particular, establish whether ATL09 can be obtained natively or only attached/resampled through ancillary fields. If only the latter is supported, label it a partial ATL09 extraction; do not claim full independent ATL09 coverage.
Deliverables: working environment instructions, reproducible dependency record, `product_support.csv`, and `product_contracts.json` including source links and access dates.
Acceptance: client connects; all five products have either a verified route to test or a precise documented blocker. Unsupported routes do not prevent exploration of supported products.
M1 — Define reusable site geometry and temporal scope
Purpose: make every subsequent query traceable to the same AOI and dates.
Tasks:
Implement configuration validation, site selection, and parameter overrides.
Generate the core zone and buffered AOI using metric buffering in a suitable local projection or a geodesic construction; never buffer longitude/latitude by a meter value directly.
Save geographic polygons plus projection and construction metadata. Handle dateline crossing for future sites or explicitly reject unsupported geometry instead of producing a malformed query.
Freeze run ID, query end time, mission-specific starts, release policy, and config hash.
Add source-linked event annotations emphasizing the 2021 emergence period without excluding any mission dates.
Deliverables: AOI files, resolved config, event table, and initial run manifest.
Acceptance: geometries are valid, center is correct, radii are verified in meters, and rerunning identical inputs produces the same AOI/config hashes. The core zone is visibly labeled as a search zone, not an island outline.
M2 — Inventory the complete available record
Purpose: discover all candidate acquisitions before expensive subsetting.
Tasks:
Search each requested product independently from its mission/collection start through frozen run end. Include all available beam types, day/night acquisitions, and orbit cycles.
Follow pagination, split dates when services impose result caps, and persist unmodified discovery responses plus normalized metadata.
Deduplicate repeated catalog records and processing revisions using a documented policy. Retain lineage to discarded revisions; select one revision consistently for retrieval.
Record product, collection/release, granule/resource ID, acquisition start/end, production/revision metadata where available, query interval, query timestamp, spatial-search method, and discovery status.
Treat granule polygon intersection as candidate coverage only. Actual point/segment/profile intersections are determined after retrieval.
Establish ATL03-to-ATL08/ATL09/ATL24 associations using documented identifiers and release compatibility, not unverified filename substitution. GEDI remains an independent mission inventory.
Produce annual and monthly candidate counts, earliest/latest actual dates per product, and data-latency summaries relative to run end. Explain mission gaps only when supported by authoritative documentation.
Deliverables: per-product inventories, inventory completeness ledger, association table, and preliminary acquisition timeline.
Acceptance: every requested product/date partition has a recorded outcome; caps/pagination and version-selection rules are explicit. A catalog or network failure cannot be labeled no coverage.
M3 — Retrieve small diagnostic samples and validate schemas
Purpose: validate requests, units, provenance, and permissive filtering before retrieving all history.
Tasks:
Use a few inventory resources: prioritize the reference study's September 2021 ATL03 acquisitions if available, then a pre-emergence and a later acquisition. Test at least one candidate from each other product where one exists.
Use the study's known acquisition dates as a useful check, not a hard-coded restriction or a guarantee that those granules remain exposed under the selected release.
Validate native fields and fill values, time conversion, beam/profile identity, coordinate reference, vertical reference, units, and stable row identifiers. Keep raw native times alongside normalized UTC.
Inspect and explicitly override default filters that would hide noise, ocean returns, low-confidence photons, or low-quality GEDI shots. Document any unavoidable server-side losses. Estimate data volume from the small runs.
Validate output schemas and provenance mapping before scaling up. Test an empty AOI response, duplicate resource, interrupted chunk, and resume behavior with small fixtures or a limited smoke run.
Deliverables: diagnostic subsets, resolved field dictionary, request examples, smoke-test results, and full-run size/runtime estimate.
Acceptance: sample rows trace to source resources; no datum/time ambiguity is hidden; empty/error outcomes differ; no full-mission request is needed to diagnose schema failures.
M4 — Acquire all available AOI subsets with restart support
Purpose: retrieve and cache the full inventory without losing diagnostic information.
Tasks:
Implement small product adapters behind a common request/status interface, using verified SlideRule routes.
Retrieve all selected inventory resources over the outer AOI, including surrounding ocean, retaining the agreed diagnostic fields. Do not restrict all products to their intersection or to the 2021 island lifetime.
Write atomic chunk outputs and checksums; maintain a completion ledger keyed by product, source revision/release, AOI hash, field/filter configuration, and processing version.
Use bounded exponential retries with a finite budget. Distinguish transient failures from invalid fields/authentication errors. Re-running skips only verified complete chunks; metadata or parameter changes invalidate the appropriate cache entries.
Keep raw subsets immutable. Generate filtered views separately. Any plotting subsampling must not reduce the archived subset or reported counts.
Product-specific requirements:
Product	Retain/inspect, subject to verified schema	Interpretation rule
ATL03	Photon height/location/time and native photon identity; all beams/spots; confidence and quality fields; solar/background context; granule/RGT/cycle identifiers; useful correction metadata	Do not collapse to fitted heights or require a land classification. Preserve enough context to inspect ocean/land/background returns.
ATL08	Native terrain-height statistics, uncertainties, segment IDs/extent information, photon counts, quality/context flags	Retrieve standard product segments. Missing ATL08 classifications must not remove ATL03 photons.
ATL09	Accessible atmospheric layer/profile context, flags, native profile/time coordinates, and sampling/alignment metadata	Keep profile/ancillary distinction explicit. Atmospheric altitude is not terrain elevation.
ATL24	Published classes, corrected coordinates/heights, confidence/uncertainty, relevant surface/depth fields, native photon association where exposed	Document corrected versus uncorrected variables and compatible ATL03 release. Do not infer water depth directly from an unlabeled height column.
GEDI L2A	Footprint ground elevation, coordinates/time, exact shot ID, orbit/beam, quality/degrade/sensitivity and other relevant flags exposed by the service	Verify mapping to the native ground-elevation variable, commonly `elev_lowestmode`. Preserve IDs as exact integers or strings, never floating point.
A service field limitation is not permission to invent a substitute. Add missing desired fields to the limitations table; retain accessible supported records.
Deliverables: cached subsets, acquisition ledger, requests/provenance, checksums, and field-completeness table.
Acceptance: every selected resource is complete, scientifically empty, or explicitly unresolved; no silently dropped failures. Re-running resumes without duplicating records or redownloading completed chunks.
M5 — Summarize coverage and evaluate quality
Purpose: quantify usable support rather than just total returned rows.
Tasks:
Normalize product-specific tables while retaining native variables, units, fill masks, source IDs, and raw values. Apply version-specific flag definitions.
Define named quality profiles: unfiltered diagnostic, documented baseline, and optional stricter sensitivity screen. Record rejection reasons and per-filter counts; avoid a single unexplained `good` flag.
Summarize candidate granules, actual intersecting acquisitions, returned rows, finite valid heights, baseline-screened rows, and stricter-screened rows separately.
Tabulate by product, year/month, acquisition, beam, core/buffer zone, and relevant quality classes. Keep units clear: photon counts, segment counts, atmospheric profiles, and GEDI footprints are different quantities.
Quantify actual observation gaps, spatial sampling, repeat-track proximity, and data available around the 2021 event. Evaluate repeated coverage spatially; a shared RGT alone is not a repeat measurement of the same surface.
Do not count multiple products from the same ICESat-2 pass as independent temporal observations.
Keep unknown surface types unknown. Without imagery or supplied historical outlines, core-zone counts are not verified on-island counts.
Deliverables: coverage tables, quality/rejection tables, spatial/repeat-support summaries, and unresolved-data ledger.
Acceptance: summary counts reconcile with cached rows and acquisition ledger; every zero is distinguishable from a failed/unqueried request. Coverage is reported for both the complete mission and the short-lived island's documented event period.
M6 — Generate maps, timelines, profiles, and qualified comparisons
Purpose: make the observations inspectable without notebooks or imagery downloads.
Tasks:
Create regional and core-zone coverage maps with AOI boundaries, site marker, date/product/beam legends, and counts. Use a plain coordinate/grid background; do not add imagery or online map tiles in this phase.
Produce acquisition timelines by product, monthly counts, and gap summaries with event annotations and separate candidate versus actual usable coverage.
Plot ATL03 height against along-track distance for informative passes; show diagnostic photons and named quality views, with full-range and surface-detail panels when useful. Preserve row counts when plotting a thinned display.
Overlay standard ATL08 terrain segments and ATL24 classes/heights only where their spatial/temporal associations are verified. Show ATL09 atmospheric context separately with its sampling and alignment limits.
Plot GEDI footprints and native terrain heights with quality context. Cross-mission comparisons are optional when valid matches exist, not an assumed deliverable of paired differences.
Before differences, establish compatible vertical references, correction status, horizontal support, and acquisition timing. Retain ellipsoidal and orthometric variables separately; unknown datum means no quantitative height-difference claim.
Match using configurable spatial/time tolerances and footprint/segment support. Log distances and time offsets. Do not interpret a GEDI-to-ICESat-2 mismatch as sensor error or change when topography, support, or date differs.
Report inter-product consistency, not independent accuracy: ATL03/ATL08/ATL24 share measurement ancestry. No independent DEM truth is included yet.
Deliverables: reproducible PNG/PDF figures, plot index linking figures to inputs/parameters, optional valid-match table, and documented reasons when a comparison is inappropriate.
Acceptance: every figure labels units, time, product/release, quality profile, and spatial scope. No unsupported terrain, bathymetry, accuracy, or change claims are implied by overlays.
M7 — Assemble the feasibility assessment and reusable handoff
Purpose: decide what can be attempted next and preserve a workflow that works for additional sites.
Tasks:
Generate `feasibility.md` with linked figures/tables and a concise product-by-product verdict: support verified, limited support, no actual coverage, or technically unresolved.
Answer: Which acquisitions could sample the 2021 island? How many distinct usable dates and independent spatial transects exist? Are observations repeated at comparable locations? What ocean/bathymetry/atmospheric context exists? Which gaps arise from sampling, quality, product masks, service limits, or retrieval errors?
Separate three decisions: feasibility of descriptive altimetry exploration, feasibility of repeat elevation-change analysis, and suitability for later full-surface reconstruction. Do not infer the third from one successful elevation profile.
State what remains unknown without imagery/historical outlines and independent DEMs. Do not claim island-wide volume feasibility from sparse transects alone.
Recommend a next step: proceed with this site, use it primarily as a diagnostic benchmark and add another island, or resolve specific access/sampling blockers first. Base the recommendation on observed support rather than a predetermined success threshold.
Document exact environment setup and script commands, dependencies, field/flag dictionaries, acquisition and quality choices, rerun/update instructions, and limitations.
Demonstrate multi-island design by validating a second synthetic config entry or supplied site config without downloading an unrequested second site's full record.
Deliverables: final report, final run manifest, working scripts/configuration, README, dependency record, and milestone completion notes.
Acceptance: a new island requires configuration rather than new acquisition logic; plots/summaries regenerate from local caches; all five requested products have an evidence-backed status. A constrained or negative feasibility result is a valid completion.
Validation priorities
Use focused checks for failure modes that can corrupt scientific conclusions: geometry/coordinate order, UTC conversion, exact identifiers, pagination and deduplication, resume/cache integrity, fill-value handling, schema changes, product associations, quality-count reconciliation, and vertical-reference compatibility. Do not build a large test suite that merely mirrors every implementation function. Successful completion means a reproducible real-data exploration plus transparent unresolved limitations, not just passing mocked API tests.
Sources and capability notes
References checked while preparing this plan; Claude must verify the installed/live service before implementation. Exact field contracts and product-version choices belong in M0, not in assumptions copied from older examples.
Study: https://doi.org/10.3390/geosciences12010040
Site coordinates and context: https://volcano.si.edu/volcano.cfm?vn=284130
Current SlideRule documentation: https://docs.slideruleearth.io/
ICESat-2 endpoints and ancillary fields: https://docs.slideruleearth.io/user-guide/icesat2
GEDI subsetting: https://docs.slideruleearth.io/user-guide/gedi
SlideRule source/client: https://github.com/SlideRuleEarth/sliderule
NASA NSIDC collection discovery guidance: https://nsidc.github.io/NSIDC-Data-Cookbook/reference-guides/collection-granule-search.html
NASA CMR API documentation: https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html
The documentation currently describes native ATL08 and ATL24 subsetting endpoints, ATL09 ancillary-field access, and GEDI L2A subsetting. These are distinct access contracts; verify their completeness and the selected releases rather than assuming the same extraction method works for every product. Obtain collection-specific user guides/ATBDs and GEDI dictionaries through official NASA providers during M0.
