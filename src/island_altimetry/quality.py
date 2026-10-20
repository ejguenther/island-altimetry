"""Named quality profiles. Each profile is an ordered list of named screens; rows
failing a screen are counted under that screen's name (first failure only), so
rejection counts add up and there is no single unexplained `good` flag.

Profiles: `diagnostic` (no screens), `baseline` (documented product-recommended
minimum), `strict` (sensitivity screen). Screens use only native variables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite(c):
    return lambda d: np.isfinite(d[c].astype("float64"))


SCREENS = {
    "atl03": {
        "baseline": [
            ("finite_height", _finite("height")),
            ("signal_conf_ge_low", lambda d: d["atl03_cnf"] >= 2),
            ("quality_ph_nominal", lambda d: d["quality_ph"] == 0),
        ],
        "strict": [
            ("signal_conf_ge_medium", lambda d: d["atl03_cnf"] >= 3),
            ("podppd_nominal", lambda d: d["podppd_flag"] == 0 if "podppd_flag" in d else pd.Series(True, index=d.index)),
        ],
    },
    "atl08": {
        "baseline": [
            ("finite_h_te_best_fit", _finite("terrain/h_te_best_fit")),
        ],
        "strict": [
            ("h_te_uncertainty_le_2m", lambda d: d["h_te_uncertainty"].astype("float64") <= 2.0),
            ("layer_flag_clear", lambda d: d["layer_flag"] == 0),
            ("n_te_photons_ge_50", lambda d: d["n_te_photons"] >= 50),
        ],
    },
    "atl24": {
        "baseline": [
            ("classified_surface_or_bathy", lambda d: d["class_ph"].isin([40, 41])),
            ("not_low_confidence", lambda d: d["low_confidence_flag"] == 0),
        ],
        "strict": [
            ("sensor_depth_not_exceeded", lambda d: d["sensor_depth_exceeded"] == 0),
            ("valid_kd", lambda d: d["invalid_kd"] == 0),
            ("valid_wind_speed", lambda d: d["invalid_wind_speed"] == 0),
        ],
    },
    "gedi_l2a": {
        "baseline": [
            ("finite_elevation_lm", _finite("elevation_lm")),
            ("quality_flag_1", lambda d: d["quality_flag"] == 1),
            ("degrade_flag_0", lambda d: d["degrade_flag"] == 0),
        ],
        "strict": [
            ("sensitivity_ge_0.9", lambda d: d["sensitivity"].astype("float64") >= 0.9),
            ("surface_flag_1", lambda d: d["surface_flag"] == 1),
        ],
    },
}

DESCRIPTION = {
    "atl03": "baseline: finite h_ph, signal_conf_ph>=2 (low+), quality_ph nominal; strict: +conf>=3, podppd_flag 0",
    "atl08": "baseline: finite h_te_best_fit; strict: +h_te_uncertainty<=2 m, layer_flag 0, n_te_photons>=50",
    "atl24": "baseline: class 40/41, not low-confidence; strict: +sensor depth not exceeded, valid Kd and wind",
    "gedi_l2a": "baseline: finite elev_lowestmode, native quality_flag 1, degrade_flag 0; strict: +sensitivity>=0.9, surface_flag 1",
}


def apply(pid: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Adds boolean columns q_baseline, q_strict and rejection reason; returns
    (df, rejection rows). strict implies baseline."""
    screens = SCREENS.get(pid)
    df["q_diagnostic"] = True
    if not screens:
        df["q_baseline"] = df["q_strict"] = True
        df["reject_reason"] = ""
        return df, []
    alive = pd.Series(True, index=df.index)
    reason = pd.Series("", index=df.index, dtype=object)
    rows = []
    for profile in ("baseline", "strict"):
        for name, fn in screens[profile]:
            ok = fn(df).fillna(False).astype(bool)
            fail = alive & ~ok
            rows.append({"profile": profile, "screen": name, "rejected": int(fail.sum())})
            reason[fail] = f"{profile}:{name}"
            alive &= ok
        df[f"q_{profile}"] = alive.copy()
    df["reject_reason"] = reason
    return df, rows
