#!/usr/bin/env python
"""M5: summarize coverage and quality from cached chunks (no network).

  summarize.py --site fukutoku_okanoba [--run-id RUN]

Tables -> outputs/<site>/<run>/tables/; analysis views (track centerlines,
footprints) -> data/<site>/analysis/<run>/. Units differ by product: ATL03 and
ATL24 rows are photons, ATL08 rows are 100 m segments, ATL09 rows are native
atmospheric profiles, GEDI rows are footprints.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from island_altimetry import config, manifest, normalize, quality, status
from island_altimetry.config import fmt_utc
from scripts_path import load_acquire

acq = load_acquire()
UNIT = {"atl03": "photons", "atl08": "100m_segments", "atl09": "atm_profiles", "atl24": "photons", "gedi_l2a": "footprints"}
BEAM = {"gedi_l2a": "beam"}


def iter_views(ctx, pid, zoner):
    """Yield (ledger_row, normalized view, fill counts) for every ok chunk."""
    contract = ctx["products"][pid]
    from island_altimetry import adapters
    parms = adapters.build_parms(pid, contract, ctx["products"]["atl03"], ctx["poly"])
    store = acq.store_for(ctx, pid, parms)
    for _, r in acq.resources_for(ctx, pid).iterrows():
        m = store.read_meta(r["granule"])
        if not m or m["status"] != status.OK or not store.is_complete(r["granule"]):
            continue
        p = store.data_path(r["granule"])
        raw = pd.read_parquet(p) if pid == "atl09" else gpd.read_parquet(p)
        v, fills = normalize.view(pid, raw, r["granule"], zoner)
        yield r, m, v, fills


def centerlines(v: pd.DataFrame, bin_m: float, min_n: int, max_mad: float) -> pd.DataFrame:
    """Per (beam, along-track bin) of strict core photons: median position and
    height, robust spread, and a coherent-surface flag (>= min_n photons with
    MAD <= max_mad). Scattered cloud/noise photons do not form coherent bins."""
    c = v[v["q_strict"] & (v["zone"] == "core")]
    if c.empty:
        return pd.DataFrame()
    c = c.assign(bin=(c["x_atc"] // bin_m).astype("int64"))
    g = c.groupby(["gt", "bin"]).agg(x=("x_local_m", "median"), y=("y_local_m", "median"),
                                     h_median=("height", "median"), n=("height", "size"),
                                     geoid=("geoid", "median"),
                                     mad=("height", lambda h: float(np.median(np.abs(h - np.median(h)))))).reset_index()
    g["coherent"] = (g["n"] >= min_n) & (g["mad"] <= max_mad)
    return g


def repeat_pairs(tracks: pd.DataFrame, max_sep: float, min_overlap: float, bin_m: float) -> pd.DataFrame:
    """Pairs of beam tracks from different dates whose core centerlines lie within
    max_sep over at least min_overlap of along-track length."""
    keys = list(tracks.groupby(["granule", "gt"]).groups)
    pts = {k: tracks.loc[idx, ["x", "y"]].to_numpy() for k, idx in tracks.groupby(["granule", "gt"]).groups.items()}
    date = {k: tracks.loc[idx, "date"].iloc[0] for k, idx in tracks.groupby(["granule", "gt"]).groups.items()}
    rgt = {k: tracks.loc[idx, "rgt"].iloc[0] for k, idx in tracks.groupby(["granule", "gt"]).groups.items()}
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if date[a] == date[b]:
                continue
            A, B = pts[a], pts[b]
            d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
            close = d <= max_sep
            overlap = close.sum() * bin_m
            if overlap >= min_overlap:
                out.append({"granule_a": a[0], "gt_a": a[1], "date_a": date[a], "rgt_a": rgt[a],
                            "granule_b": b[0], "gt_b": b[1], "date_b": date[b], "rgt_b": rgt[b],
                            "median_sep_m": round(float(np.median(d[close])), 1), "overlap_m": float(overlap)})
    return pd.DataFrame(out)


def period(t: pd.Timestamp, onset: pd.Timestamp, window_days: int) -> str:
    if t < onset:
        return "pre_onset"
    if t < onset + pd.Timedelta(days=window_days):
        return f"onset_to_+{window_days}d"
    return "later"


def run_site(site_id: str, run_id: str | None):
    ctx = acq.context(site_id, run_id)
    cfg, m = ctx["cfg"], ctx["m"]
    run_id = m["run_id"]
    c = cfg["center"]
    zoner = normalize.Zoner(c["lon"], c["lat"], cfg["core_radius_m"], cfg["core_radius_m"] + cfg["buffer_m"])
    tables = manifest.output_dir(cfg, run_id) / "tables"
    adir = manifest.data_dir(cfg) / "analysis" / run_id
    adir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(config.ROOT / m["events_table"])
    onset = pd.Timestamp(events.sort_values("start").query("precision != 'month'")["start"].iloc[0])
    rp = cfg["repeat"]

    ledger = pd.read_csv(tables / "acquisition_ledger.csv")
    acq_rows, zone_rows, rej_rows, cls_rows, fill_rows, tracks, gedi_pts, a09_rows = [], [], [], [], [], [], [], []
    recon = []
    for pid in cfg["products"]:
        n_total = 0
        for r, meta, v, fills in iter_views(ctx, pid, zoner):
            n_total += len(v)
            v, rej = quality.apply(pid, v)
            t0 = pd.Timestamp(r["acq_start_utc"])
            base = {"product": pid, "granule": r["granule"], "acq_start_utc": r["acq_start_utc"],
                    "date": t0.strftime("%Y-%m-%d"), "unit": UNIT[pid]}
            for k, val in fills.items():
                fill_rows.append({**base, "column": k, "n_float_max_fill": val})
            for x in rej:
                rej_rows.append({**base, **x})
            hcol = "height_primary" if "height_primary" in v else None
            row = {**base, "n_rows": len(v),
                   "n_core": int((v["zone"] == "core").sum()), "n_buffer": int((v["zone"] == "buffer").sum()),
                   "n_outer_margin": int((v["zone"] == "outer_margin").sum()),
                   "n_finite_height": int(np.isfinite(v[hcol]).sum()) if hcol else None,
                   "n_baseline": int(v["q_baseline"].sum()), "n_strict": int(v["q_strict"].sum()),
                   "n_core_baseline": int((v["q_baseline"] & (v["zone"] == "core")).sum()),
                   "n_core_strict": int((v["q_strict"] & (v["zone"] == "core")).sum())}
            bcol = BEAM.get(pid, "gt")
            row["beams_with_core_baseline"] = int(v.loc[v["q_baseline"] & (v["zone"] == "core"), bcol].nunique())
            if hcol:
                cb = v.loc[v["q_baseline"] & (v["zone"] == "core"), hcol]
                row.update(core_baseline_h_min=float(cb.min()) if len(cb) else None,
                           core_baseline_h_median=float(cb.median()) if len(cb) else None,
                           core_baseline_h_max=float(cb.max()) if len(cb) else None,
                           height_ref=normalize.HEIGHTS[pid][normalize.PRIMARY_HEIGHT[pid]],
                           height_var=normalize.PRIMARY_HEIGHT[pid])
            if "x_atc" in v and pid != "atl09":
                cz = v[v["zone"] == "core"]
                row["core_alongtrack_extent_m_max_beam"] = float(cz.groupby("gt")["x_atc"].agg(lambda s: s.max() - s.min()).max()) if len(cz) else 0.0
            if pid == "atl09":
                for z, g in v.groupby("zone"):
                    a09_rows.append({**base, "zone": z, "n_profiles": len(g),
                                     "frac_layer_flag_1": float((g["high_rate/layer_flag"] == 1).mean()),
                                     "cloud_flag_atm_max": int(g["high_rate/cloud_flag_atm"].max()),
                                     "n_photons_represented": int(g["n_atl03_photons"].sum())})
            acq_rows.append(row)
            grp = [bcol, "zone"]
            zc = v.groupby(grp).agg(n=("q_diagnostic", "size"), n_baseline=("q_baseline", "sum"), n_strict=("q_strict", "sum")).reset_index()
            zc = zc.rename(columns={bcol: "beam"})
            for _, z in zc.iterrows():
                zone_rows.append({**base, "beam": z["beam"], "zone": z["zone"], "n": int(z["n"]),
                                  "n_baseline": int(z["n_baseline"]), "n_strict": int(z["n_strict"])})
            qcls = {"atl03": "atl03_cnf", "atl24": "class_ph", "atl08": "layer_flag", "gedi_l2a": "quality_flag"}.get(pid)
            if qcls:
                for (z, k), n in v.groupby(["zone", qcls]).size().items():
                    cls_rows.append({**base, "zone": z, "class_var": qcls, "class": int(k), "n": int(n)})
            if pid == "atl03":
                cl = centerlines(v, rp["centerline_bin_m"], rp["coherent_min_photons"], rp["coherent_max_mad_m"])
                coh = cl[cl["coherent"]] if len(cl) else cl
                row["core_coherent_bins"] = int(len(coh))
                row["core_coherent_length_m"] = float(len(coh) * rp["centerline_bin_m"])
                row["core_coherent_beams"] = int(coh["gt"].nunique()) if len(coh) else 0
                row["core_coherent_max_h_above_geoid_m"] = float((coh["h_median"] - coh["geoid"]).max()) if len(coh) else None
                row["usable_surface_core"] = bool(row["core_coherent_length_m"] >= rp["min_usable_length_m"])
                if not cl.empty:
                    tracks.append(cl.assign(granule=r["granule"], date=base["date"], rgt=int(v["rgt"].iloc[0]),
                                            cycle=int(v["cycle"].iloc[0])))
            if pid == "gedi_l2a":
                g = v[v["q_baseline"] & (v["zone"] == "core")]
                if len(g):
                    gedi_pts.append(g[["granule", "shot_number", "beam", "x_local_m", "y_local_m", "elevation_lm"]].assign(date=base["date"]))
        led_n = ledger.loc[(ledger["product"] == pid) & (ledger["status"] == status.OK), "n_rows"].sum()
        recon.append({"product": pid, "unit": UNIT[pid], "rows_in_views": n_total, "rows_in_ledger": int(led_n),
                      "reconciled": bool(n_total == led_n)})
        print(f"== {pid}: {n_total} {UNIT[pid]} (ledger {int(led_n)})", flush=True)

    A = pd.DataFrame(acq_rows)
    A.to_csv(tables / "coverage_by_acquisition.csv", index=False)
    pd.DataFrame(zone_rows).to_csv(tables / "coverage_by_beam_zone.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(tables / "quality_class_counts.csv", index=False)
    R = pd.DataFrame(rej_rows)
    R.to_csv(tables / "rejections_by_acquisition.csv", index=False)
    if len(R):
        R.groupby(["product", "profile", "screen"], sort=False)["rejected"].sum().reset_index().to_csv(
            tables / "rejections_summary.csv", index=False)
    pd.DataFrame(fill_rows).to_csv(tables / "fill_value_counts.csv", index=False)
    pd.DataFrame(a09_rows).to_csv(tables / "atl09_context_by_acquisition.csv", index=False)
    pd.DataFrame(recon).to_csv(tables / "reconciliation.csv", index=False)

    # Monthly / annual coverage (candidate vs queried vs actual vs usable).
    inv_dir = ctx["inv_dir"]
    cov = []
    for pid in cfg["products"]:
        src = "atl03" if pid == "atl09" else pid
        inv = pd.read_csv(inv_dir / f"inventory_{src}.csv")
        led = ledger[ledger["product"] == pid].copy()
        led["month"] = pd.to_datetime(led["acq_start_utc"], utc=True).dt.strftime("%Y-%m")
        a = A[A["product"] == pid].copy() if len(A) else pd.DataFrame(columns=["acq_start_utc"])
        a["month"] = pd.to_datetime(a["acq_start_utc"], utc=True).dt.strftime("%Y-%m")
        for mo, g in led.groupby("month"):
            ag = a[a["month"] == mo]
            cov.append({"product": pid, "unit": UNIT[pid], "month": mo, "candidate_granules": len(g),
                        "ok": int((g["status"] == status.OK).sum()),
                        "no_samples_in_aoi": int((g["status"] == status.NO_SAMPLES_IN_AOI).sum()),
                        "no_granules": int((g["status"] == status.NO_GRANULES).sum()),
                        "unresolved": int((~g["verified_complete"]).sum()),
                        "acq_with_core_rows": int((ag["n_core"] > 0).sum()) if len(ag) else 0,
                        "acq_with_core_baseline": int((ag["n_core_baseline"] > 0).sum()) if len(ag) else 0,
                        "rows": int(ag["n_rows"].sum()) if len(ag) else 0,
                        "rows_core_baseline": int(ag["n_core_baseline"].sum()) if len(ag) else 0,
                        "rows_core_strict": int(ag["n_core_strict"].sum()) if len(ag) else 0})
    C = pd.DataFrame(cov)
    C.to_csv(tables / "coverage_by_month.csv", index=False)
    C.assign(year=C["month"].str[:4]).groupby(["product", "unit", "year"]).sum(numeric_only=True).reset_index().to_csv(
        tables / "coverage_by_year.csv", index=False)

    # Gaps between usable acquisitions (core baseline) per product.
    gaps = []
    usable = A["n_core_baseline"] > 0
    if "usable_surface_core" in A:
        usable = np.where(A["product"] == "atl03", A["usable_surface_core"].fillna(False).astype(bool), usable)
    for pid, g in A[usable].groupby("product"):
        t = pd.to_datetime(g["acq_start_utc"], utc=True).sort_values()
        d = t.diff().dt.total_seconds().div(86400).dropna()
        gaps.append({"product": pid, "n_usable_acquisitions": len(t), "n_distinct_dates": t.dt.date.nunique(),
                     "first": fmt_utc(t.min()), "last": fmt_utc(t.max()),
                     "median_gap_days": round(float(d.median()), 1) if len(d) else None,
                     "max_gap_days": round(float(d.max()), 1) if len(d) else None,
                     "max_gap_start": fmt_utc(t.iloc[int(np.argmax(d.values))]) if len(d) else None})
    pd.DataFrame(gaps).to_csv(tables / "usable_gaps.csv", index=False)

    # Event-period buckets (analysis convention, not an island lifetime).
    A["period"] = [period(pd.Timestamp(t), onset, cfg["event_window_days"]) for t in A["acq_start_utc"]]
    A["usable_surface_core"] = A.get("usable_surface_core", pd.Series(np.nan, index=A.index))
    ev = A.groupby(["product", "period"]).agg(acquisitions=("granule", "size"),
                                              atl03_usable_surface=("usable_surface_core", lambda s: int(s.fillna(False).astype(bool).sum())),
                                              with_core_rows=("n_core", lambda s: int((s > 0).sum())),
                                              with_core_baseline=("n_core_baseline", lambda s: int((s > 0).sum())),
                                              with_core_strict=("n_core_strict", lambda s: int((s > 0).sum()))).reset_index()
    ev.to_csv(tables / "event_period_summary.csv", index=False)

    # Tracks, repeat locations (ICESat-2), GEDI footprint proximity.
    T = pd.concat(tracks, ignore_index=True) if tracks else pd.DataFrame()
    T.to_parquet(adir / "atl03_core_centerlines.parquet", index=False)
    P = repeat_pairs(T[T["coherent"]], rp["max_separation_m"], rp["min_overlap_m"], rp["centerline_bin_m"]) if len(T) else pd.DataFrame()
    P.to_csv(tables / "repeat_track_pairs.csv", index=False)
    G = pd.concat(gedi_pts, ignore_index=True) if gedi_pts else pd.DataFrame()
    G.to_parquet(adir / "gedi_core_baseline_footprints.parquet", index=False)
    gp = []
    if len(G):
        X = G[["x_local_m", "y_local_m"]].to_numpy()
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        same = G["date"].to_numpy()[:, None] == G["date"].to_numpy()[None]
        i, j = np.where(np.triu((D <= rp["max_separation_m"]) & ~same, 1))
        for a, b in zip(i, j):
            gp.append({"shot_a": str(G["shot_number"].iloc[a]), "date_a": G["date"].iloc[a],
                       "shot_b": str(G["shot_number"].iloc[b]), "date_b": G["date"].iloc[b], "dist_m": round(float(D[a, b]), 1)})
    pd.DataFrame(gp).to_csv(tables / "gedi_repeat_footprint_pairs.csv", index=False)

    # Unresolved-data ledger.
    U = ledger[~ledger["verified_complete"].astype(bool)]
    U.to_csv(tables / "unresolved_ledger.csv", index=False)

    usable_passes = sorted(A.loc[(A["product"] == "atl03") & A.get("usable_surface_core", False).fillna(False).astype(bool), "date"].unique())
    summary = {
        "completed_utc": fmt_utc(dt.datetime.now(dt.timezone.utc)),
        "quality_profiles": quality.DESCRIPTION,
        "analysis_parameters": {"event_onset_utc": str(onset), "event_window_days": cfg["event_window_days"],
                                "repeat": rp},
        "reconciliation": recon,
        "independent_icesat2_passes_with_core_coherent_surface": len(usable_passes),
        "usable_pass_dates": usable_passes,
        "coherent_surface_definition": (f"20 m along-track bins of strict-profile core photons with >= "
                                        f"{rp['coherent_min_photons']} photons and MAD <= {rp['coherent_max_mad_m']} m; "
                                        f"a pass is usable if coherent length >= {rp['min_usable_length_m']} m"),
        "repeat_track_pairs": len(P), "distinct_repeat_rgts": sorted({int(x) for x in P["rgt_a"]}) if len(P) else [],
        "gedi_repeat_footprint_pairs": len(gp), "unresolved_chunks": int(len(U)),
        "notes": ["ICESat-2 passes are counted once per ATL03 granule; ATL08/ATL09/ATL24 from the same pass are not "
                  "independent observations.",
                  "Core-zone counts are search-zone counts, not verified on-island counts (no imagery or outlines)."],
    }
    manifest.update(cfg, run_id, "summary", summary)
    print(json.dumps({k: summary[k] for k in ("reconciliation", "independent_icesat2_passes_with_core_coherent_surface",
                                              "repeat_track_pairs", "gedi_repeat_footprint_pairs", "unresolved_chunks")}, indent=1))
    print(ev.to_string(index=False))
    if not all(r["reconciled"] for r in recon):
        raise SystemExit("reconciliation failed: view rows differ from ledger")


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
