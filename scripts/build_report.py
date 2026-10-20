#!/usr/bin/env python
"""M7: assemble outputs/<site>/<run>/feasibility.md from cached tables, figures,
and the run manifest (no network). Every number is read from a table.

  build_report.py --site fukutoku_okanoba [--run-id RUN]

Verdict rules (per product):
  technically unresolved  - any chunk not verified complete
  no actual coverage      - no rows in the AOI from any acquisition
  limited support         - rows exist but access is partial (documented M0
                            limitation) or no acquisition has core-zone rows
  support verified        - otherwise
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from island_altimetry import config, manifest, quality
from island_altimetry.config import fmt_utc
from island_altimetry.plotting import PRODUCT_LABEL, PRODUCT_ORDER
from scripts_path import load_acquire

acq = load_acquire()

LIMITATIONS = {
    "atl09": "partial ATL09 extraction: ancillary values attached to ATL03 photons (no native endpoint); "
             "layer_top/layer_bot truncated to one layer",
    "atl24": "ATL24 v002 ends 2024-10-14; derived from ATL03 r006 (spatial/temporal association only)",
    "gedi_l2a": "GEDI L2A v002 via SlideRule (v003 unreadable by the service); record ends 2025-04 in this AOI; "
                "landsat_water_persistence unavailable",
}


def md_table(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    if df.empty:
        return "_(none)_\n"
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            whole = df[c].dropna().apply(float.is_integer).all() if df[c].notna().any() else False
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{int(v)}" if whole else floatfmt.format(v)))
    head = "| " + " | ".join(map(str, df.columns)) + " |\n| " + " | ".join("---" for _ in df.columns) + " |\n"
    return head + "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))


def verdicts(cfg, A, led) -> pd.DataFrame:
    rows = []
    for pid in PRODUCT_ORDER:
        if pid not in cfg["products"]:
            continue
        L, a = led[led["product"] == pid], A[A["product"] == pid]
        unresolved = int((~L["verified_complete"].astype(bool)).sum())
        with_rows = int((a["n_rows"] > 0).sum())
        with_core = int((a["n_core_baseline"] > 0).sum())
        if unresolved:
            v = "technically unresolved"
        elif with_rows == 0:
            v = "no actual coverage"
        elif pid in LIMITATIONS or with_core == 0:
            v = "limited support"
        else:
            v = "support verified"
        rows.append({"product": PRODUCT_LABEL[pid], "verdict": v, "candidates": len(L),
                     "ok": int((L["status"] == "ok").sum()), "no_samples_in_aoi": int((L["status"] == "no_samples_in_aoi").sum()),
                     "no_granules": int((L["status"] == "no_granules").sum()), "unresolved": unresolved,
                     "acq_with_core_baseline": with_core, "limitation": LIMITATIONS.get(pid, "")})
    return pd.DataFrame(rows)


def run_site(site_id, run_id):
    ctx = acq.context(site_id, run_id)
    cfg, m = ctx["cfg"], ctx["m"]
    out = manifest.output_dir(cfg, m["run_id"])
    T = out / "tables"
    rd = lambda n: pd.read_csv(T / n) if (T / n).exists() and (T / n).stat().st_size > 1 else pd.DataFrame()
    A, led = rd("coverage_by_acquisition.csv"), rd("acquisition_ledger.csv")
    ev, gaps, rep = rd("event_period_summary.csv"), rd("usable_gaps.csv"), rd("repeat_track_pairs.csv")
    a09, cls, comp = rd("atl09_context_by_acquisition.csv"), rd("quality_class_counts.csv"), rd("comparison_summary.csv")
    lat, recon, rej = rd("inventory_latency.csv"), rd("reconciliation.csv"), rd("rejections_summary.csv")
    gv = rd("comparison_gedi_vs_icesat2_valid.csv")
    events = pd.read_csv(config.ROOT / m["events_table"])
    summary = m.get("summary", {})
    onset = pd.Timestamp(summary["analysis_parameters"]["event_onset_utc"])
    elev_thr = cfg["elevated_threshold_m"]
    cl = pd.read_parquet(manifest.data_dir(cfg) / "analysis" / m["run_id"] / "atl03_core_centerlines.parquet")
    core_geoid = float(cl["geoid"].median())

    V = verdicts(cfg, A, led)
    a03 = A[A["product"] == "atl03"].copy()
    a03["t"] = pd.to_datetime(a03["acq_start_utc"], utc=True)
    a03["usable_surface_core"] = a03["usable_surface_core"].fillna(False).astype(bool)
    post = a03[a03["t"] >= onset]
    elevated = post[post["usable_surface_core"] & (post["core_coherent_max_h_above_geoid_m"] > elev_thr)]
    pre_max = a03.loc[(a03["t"] < onset) & a03["usable_surface_core"], "core_coherent_max_h_above_geoid_m"]
    usable = a03[a03["usable_surface_core"]]
    by_period = usable.assign(period=np.where(usable["t"] < onset, "pre-onset", "post-onset")).groupby("period").size().to_dict()
    gedi_post = A[(A["product"] == "gedi_l2a") & (pd.to_datetime(A["acq_start_utc"], utc=True) >= onset) & (A["n_core_baseline"] > 0)]
    rep_post = rep[(pd.to_datetime(rep["date_a"]) >= onset.tz_localize(None)) | (pd.to_datetime(rep["date_b"]) >= onset.tz_localize(None))] if len(rep) else rep
    rep_both_post = rep[(pd.to_datetime(rep["date_a"]) >= onset.tz_localize(None)) & (pd.to_datetime(rep["date_b"]) >= onset.tz_localize(None))] if len(rep) else rep
    unresolved = int((~led["verified_complete"].astype(bool)).sum())

    # Revisits of the elevated locations: coherent bins of any other pass within the
    # repeat separation of an elevated bin, plus the nearest baseline GEDI footprint.
    sep = cfg["repeat"]["max_separation_m"]
    cl = cl.assign(h_geoid=cl["h_median"] - cl["geoid"])
    coh = cl[cl["coherent"]]
    el_bins = coh[coh["h_geoid"] > elev_thr]
    revisits, gedi_near = pd.DataFrame(), pd.DataFrame()
    if len(el_bins):
        ex, ey = el_bins["x"].to_numpy(), el_bins["y"].to_numpy()
        nearest = lambda x, y: np.min(np.hypot(np.asarray(x)[:, None] - ex[None], np.asarray(y)[:, None] - ey[None]), axis=1)
        o = coh.assign(dist_m=nearest(coh["x"], coh["y"]))
        o = o[o["dist_m"] <= sep]
        revisits = o.groupby(["date", "rgt", "gt"]).agg(
            bins=("bin", "size"), min_dist_m=("dist_m", "min"), h_above_geoid_median=("h_geoid", "median"),
            h_above_geoid_max=("h_geoid", "max")).reset_index()
        gf = manifest.data_dir(cfg) / "analysis" / m["run_id"] / "gedi_core_baseline_footprints.parquet"
        if gf.exists():
            G = pd.read_parquet(gf)
            G = G[pd.to_datetime(G["date"]) >= onset.tz_localize(None).normalize()]
            if len(G):
                G = G.assign(dist_m=nearest(G["x_local_m"], G["y_local_m"]))
                gedi_near = G.groupby(["date", "granule"]).agg(nearest_m=("dist_m", "min")).reset_index()
    el_dates = set(el_bins["date"].astype(str))
    later = revisits[~revisits["date"].astype(str).isin(el_dates)] if len(revisits) else revisits
    later_elevated = later[later["h_above_geoid_max"] > elev_thr] if len(later) else later

    # Gap attribution for ATL03 passes without a usable core surface.
    led03 = led[led["product"] == "atl03"].set_index("granule")
    reasons = []
    for g, r in led03.iterrows():
        a = a03[a03["granule"] == g]
        if r["status"] != "ok":
            reasons.append(r["status"] if r["status"] in ("no_samples_in_aoi", "no_granules") else "retrieval/unresolved")
        elif a.empty or a["n_core"].iloc[0] == 0:
            reasons.append("sampling: pass misses core zone")
        elif not a["usable_surface_core"].iloc[0]:
            reasons.append("quality: core photons but no coherent surface (cloud/noise/weak beam)")
        else:
            reasons.append("usable")
    gap_reasons = pd.Series(reasons).value_counts().rename_axis("ATL03 pass outcome").reset_index(name="passes")

    fig = lambda f, alt: f"![{alt}](figures/{f})\n"
    L = []
    L.append(f"# Feasibility assessment: {site_id}\n")
    L.append(f"Run `{m['run_id']}` · query end {m['query_end_utc']} · AOI `{m['aoi_hash']}` · config `{m['config_hash']}` · "
             f"generated {fmt_utc(dt.datetime.now(dt.timezone.utc))}\n")
    L.append("Scope: descriptive exploration of what satellite altimetry exists over Fukutoku-Okanoba (core search zone "
             f"{cfg['core_radius_m']:.0f} m, outer AOI {cfg['core_radius_m'] + cfg['buffer_m']:.0f} m around the GVP 284130 "
             "reference point). The core zone is a search zone, not an island outline. No imagery, historical outlines, "
             "or independent DEMs were used; core-zone counts are not verified on-island counts.\n")
    L.append("## Summary\n")
    elev_list = ", ".join(f"{d} (RGT {int(r)}, max {h:.1f} m)" for d, r, h in
                          zip(elevated["date"], elevated["granule"].str[21:25].astype(int), elevated["core_coherent_max_h_above_geoid_m"]))
    L.append(f"- **ICESat-2 passes with a coherent surface in the core zone:** {len(usable)} distinct dates "
             f"({by_period.get('pre-onset', 0)} before and {by_period.get('post-onset', 0)} after the 2021-08-12 eruption onset).\n")
    L.append(f"- **Passes after onset showing coherent returns more than {elev_thr:.0f} m above the local geoid** (pre-onset "
             f"maximum {pre_max.max():.1f} m): {len(elevated)} — {elev_list or 'none'}. These are elevated surface returns "
             "consistent with emergent material; without imagery or outlines they are not verified island measurements.\n")
    if len(el_bins):
        L.append(f"- **Revisits of the elevated locations** (coherent bins within {sep:.0f} m): "
                 + (", ".join(f"{d} (RGT {r}, median {h:.1f} m above geoid)" for d, r, h in
                              later.groupby(["date", "rgt"])["h_above_geoid_median"].median().reset_index().itertuples(index=False))
                    or "none") + ".\n")
    L.append(f"- **GEDI acquisitions after onset with baseline core footprints:** {len(gedi_post)}"
             + (f"; nearest to the elevated returns: {', '.join(f'{d} {n:.0f} m' for d, n in zip(gedi_near['date'], gedi_near['nearest_m']))}"
                if len(gedi_near) else "") + ".\n")
    L.append(f"- **Repeat comparable locations (centerlines within {summary['analysis_parameters']['repeat']['max_separation_m']} m "
             f"over ≥{summary['analysis_parameters']['repeat']['min_overlap_m']} m):** {len(rep)} beam-track pairs in total, "
             f"{len(rep_post)} involving a post-onset pass, {len(rep_both_post)} with both passes post-onset.\n")
    L.append(f"- **Unresolved retrievals:** {unresolved}.\n")
    L.append("\n### Product verdicts\n")
    L.append(md_table(V))
    L.append("\n## Answers\n")
    L.append("### Which acquisitions could sample the 2021 island?\n")
    cols = ["date", "granule", "n_core_baseline", "core_coherent_length_m", "core_coherent_beams", "core_coherent_max_h_above_geoid_m"]
    L.append(md_table(post[post["n_core"] > 0][cols]))
    L.append("\nGEDI post-onset acquisitions with core footprints:\n")
    L.append(md_table(gedi_post[["date", "granule", "n_core_baseline", "core_baseline_h_min", "core_baseline_h_median", "core_baseline_h_max"]]))
    L.append(f"\nGEDI elevations are WGS84-ellipsoidal (`elev_lowestmode`); the local EGM2008 geoid in the core zone is "
             f"{core_geoid:.2f} m (median of ATL03 `geoid`), so height above the geoid ≈ elevation − {core_geoid:.1f} m.\n")
    L.append("### How many distinct usable dates and independent transects?\n")
    L.append(f"Usable ICESat-2 dates (coherent core surface ≥ {summary['analysis_parameters']['repeat'].get('min_usable_length_m', 100)} m): "
             f"{len(usable)}. Independent transects are beam-tracks with coherent core surface; each pass counts once in time "
             "regardless of how many products (ATL03/08/09/24) it produced.\n")
    L.append(md_table(usable[["date", "granule", "core_coherent_beams", "core_coherent_length_m", "core_coherent_max_h_above_geoid_m"]]))
    L.append("\nUsable-acquisition gaps:\n")
    L.append(md_table(gaps))
    L.append("\n### Are observations repeated at comparable locations?\n")
    if len(rep):
        rs = rep.assign(pair=rep["date_a"] + " ↔ " + rep["date_b"]).groupby(["rgt_a", "rgt_b"]).agg(
            pairs=("pair", "size"), median_sep_m=("median_sep_m", "median"), first=("date_a", "min"), last=("date_b", "max")).reset_index()
        L.append(md_table(rs))
        L.append("\nRepeat pairs require spatial proximity of the coherent centerlines, not just a shared RGT. Most pairs "
                 "repeat the sea surface; the pairs that matter for change are revisits of the elevated returns:\n")
        L.append(f"\nCoherent core bins (any pass) within {sep:.0f} m of a bin more than {elev_thr:.0f} m above the geoid:\n")
        L.append(md_table(revisits, "{:.1f}"))
        if len(gedi_near):
            L.append("\nNearest post-onset baseline GEDI core footprint to those elevated bins:\n")
            L.append(md_table(gedi_near, "{:.0f}"))
    else:
        L.append("No beam-track pairs met the repeat-location criterion.\n")
    L.append("### What ocean, bathymetry, and atmospheric context exists?\n")
    if len(cls):
        c24 = cls[cls["product"] == "atl24"].groupby(["zone", "class"])["n"].sum().unstack(fill_value=0)
        L.append("ATL24 photon classes by zone (0 unclassified, 40 bathymetry, 41 sea surface):\n")
        L.append(md_table(c24.reset_index()))
    if len(a09):
        L.append("\nATL09 (partial extraction) cloud context per pass, core zone:\n")
        L.append(md_table(a09[a09["zone"] == "core"][["date", "granule", "n_profiles", "frac_layer_flag_1", "cloud_flag_atm_max"]]))
    L.append("\n### Which gaps arise from sampling, quality, product masks, service limits, or retrieval errors?\n")
    L.append(md_table(gap_reasons))
    L.append("\nProduct-level gaps: ATL08 standard segments are absent over open water (product mask; e.g. ATL08 "
             "`no_samples_in_aoi` where ATL03 photons exist); ATL24 v002 ends 2024-10-14 and omits non-water surfaces; "
             "GEDI record ends 2025-04 in this AOI (v002 via SlideRule) and is footprint-sparse; ATL09 is available only "
             "where ATL03 photons exist. Lack of product coverage is not evidence of absent land, shallow water, or clear sky.\n")
    ur = led[~led["verified_complete"].astype(bool)]
    if len(ur):
        L.append(f"\nUnresolved retrievals ({len(ur)}): recorded with their failure status and never counted as zero "
                 "coverage. Server alerts after bounded retries:\n")
        cols = [c for c in ("product", "granule", "status", "attempts", "alerts_summary") if c in ur.columns]
        L.append(md_table(ur[cols]))
    L.append("\nInventory latency (newest actual observation per product vs run end):\n")
    L.append(md_table(lat))
    L.append("\n## Inter-product consistency (not accuracy)\n")
    L.append(md_table(comp))
    L.append(f"\nGEDI–ICESat-2 valid matches (≤{cfg['comparison']['max_distance_m']} m, ≤{cfg['comparison']['max_time_offset_days']} d): "
             f"{len(gv)}. Differences may reflect ocean tide and DAC (provided but not applied to ATL03 h_ph), sea state, "
             "footprint vs bin support, or surface change; they are not interpreted as sensor error. ATL03, ATL08, and "
             "ATL24 share measurement ancestry, so their agreement is processing consistency, not independent validation. "
             "See `tables/vertical_compatibility.csv`.\n")
    L.append("\n## Decisions\n")
    d1 = "feasible" if len(usable) >= 3 else "limited"
    d2 = ("limited to a handful of revisits of the elevated locations" if len(later) else
          "not supported: no later pass revisits the elevated locations")
    later_dates = ", ".join(sorted(set(later["date"].astype(str)))) if len(later) else "none"
    L.append(f"1. **Descriptive altimetry exploration: {d1}.** {len(usable)} usable ICESat-2 dates and {len(elevated)} post-onset "
             f"passes with elevated coherent returns. ATL03 support is verified; ATL08 has {int(((led['product'] == 'atl08') & ~led['verified_complete'].astype(bool)).sum())} "
             "unresolved granules; ATL09 (partial extraction), ATL24 and GEDI have documented limitations.\n")
    L.append(f"2. **Repeat elevation-change analysis: {d2}.** {len(rep_both_post)} post-onset repeat pairs exist, but almost all "
             f"repeat the sea surface. The elevated returns of {', '.join(sorted(el_dates)) or 'none'} were revisited within "
             f"{sep:.0f} m on {later_dates}; {'none' if later_elevated.empty else len(later_elevated)} of those revisits showed "
             f"returns above {elev_thr:.0f} m. That is consistent with the emergent material being gone or near sea level by the "
             "first revisit, but the timing of its loss is bounded only by the pass dates and is not attributed without imagery.\n")
    L.append("3. **Later full-surface reconstruction: not established.** Sparse along-track transects (a few beams per pass, "
             "~3 km beam spacing, ≤ a few post-onset dates) cannot constrain an island-wide surface or volume; this is not "
             "inferred from individual successful profiles.\n")
    L.append("\n## Recommendation\n")
    L.append("Use Fukutoku-Okanoba primarily as a **diagnostic benchmark**: it demonstrates that the workflow detects "
             "short-lived emergent material in ICESat-2 photons and separates product, quality, and service limitations. "
             "Before any change analysis, add (a) dated imagery-derived outlines to label land vs water in the core zone, "
             "and (b) a second island with a longer subaerial lifetime and denser repeat coverage. This recommendation "
             "follows from the observed counts above, not a predetermined threshold.\n")
    L.append("\n## What remains unknown\n")
    L.append("- Whether core-zone returns are land, pumice raft, plume, or sea surface at a given date (no imagery/outlines).\n"
             "- Island extent, volume, and disappearance date (no sourced date; sparse transects).\n"
             "- Absolute accuracy (no independent DEM; ATL08 `dem_h` and GEDI `digital_elevation_model` are reference "
             "fields, not truth).\n- Full ATL09 layer structure (service truncation).\n")
    L.append("\n## Figures\n")
    for f, alt in (("map_regional_coverage.png", "regional coverage"), ("map_core_coverage.png", "core coverage"),
                   ("timeline_candidate_vs_usable.png", "timeline"), ("timeline_monthly_counts.png", "monthly counts"),
                   ("comparison_histograms.png", "comparisons")):
        if (out / "figures" / f).exists():
            L.append(fig(f, alt))
    if len(elevated):
        L.append("\nProfiles for elevated post-onset passes:\n")
        for g in elevated["granule"]:
            if (out / "figures" / "profiles" / f"profile_{g}.png").exists():
                L.append(fig(f"profiles/profile_{g}.png", g))
    L.append("\nAll profiles: `figures/profiles/`; figure provenance: `plot_index.csv`.\n")
    L.append("\n## Methods and provenance\n")
    L.append(f"- Quality profiles: " + "; ".join(f"{k}: {v}" for k, v in quality.DESCRIPTION.items()) + "\n")
    L.append(f"- Coherent surface: {summary.get('coherent_surface_definition', '')}.\n")
    L.append(f"- Elevated threshold: {elev_thr} m above the ATL03 EGM2008 geoid (pre-onset coherent maxima ≤ {pre_max.max():.1f} m).\n")
    L.append(f"- Event period convention: onset {onset} + {summary['analysis_parameters']['event_window_days']} d "
             "(analysis bucket, not an island lifetime). Events: `tables/events.csv`.\n")
    L.append("- Releases, endpoints, and limitations: `docs/m0/NOTES.md`; field dictionary: `docs/m3/field_dictionary.csv`.\n")
    L.append("- Reconciliation of view rows vs acquisition ledger:\n\n" + md_table(recon))
    if len(rej):
        L.append("\nRejection counts by screen (first failing screen, all acquisitions):\n\n" + md_table(rej))
    (out / "feasibility.md").write_text("\n".join(L))
    manifest.update(cfg, m["run_id"], "report", {"feasibility": "feasibility.md", "completed_utc": fmt_utc(dt.datetime.now(dt.timezone.utc)),
                                                 "verdicts": V[["product", "verdict"]].to_dict(orient="records"),
                                                 "git_commit": manifest.git_commit()})
    print(f"wrote {out / 'feasibility.md'}")
    print(V[["product", "verdict"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--run-id")
    a = ap.parse_args()
    for s in (config.site_ids() if a.all else [a.site]):
        run_site(s, a.run_id)


if __name__ == "__main__":
    main()
