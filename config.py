"""
Settings shared by every lake + a loader for per-lake settings.

Per-lake settings (position, Sentinel-2 tile, radar track, minimum lake area, known events) live in
lakes/lakes.csv. Fill in lon/lat for a lake, then run `python setup_lake.py <lake_id>` to auto-detect and
verify the rest. Choose the lake with  GLOF_LAKE=<lake_id>  or the --lake option of the scripts.
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(BASE, "lakes", "lakes.csv")
DEFAULT_LAKE = "south_lhonak"

# ---- same for every lake ----
SEED_RADIUS_M = 150
MAX_CLOUD, MIN_VALID, MIN_LST_VALID = 0.5, 0.9, 0.5
PC_KEY = os.environ.get("PC_KEY", "")        # OPTIONAL Planetary Computer key (only loosens rate limits)
MIN_FEATURES = 2                             # need at least this many non-null features, else risk stays null
WATCH_PCT, ALERT_PCT = 80, 95
TRAIN_END = "2024-12-31"                     # the model learns ONLY from windows up to this date
THERMAL_WEIGHT = 0.6                         # temperature alone can reach at most 60% (= Low)
DEFAULT_MIN_AREA = 0.5


def make_bbox(lon, lat, half_km=1.5):
    """Square box around a point: [lon_min, lat_min, lon_max, lat_max] (same as your notebook)."""
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * np.cos(np.radians(lat)))
    return [float(round(lon - dlon, 4)), float(round(lat - dlat, 4)), float(round(lon + dlon, 4)), float(round(lat + dlat, 4))]


def registry():
    return pd.read_csv(REGISTRY, dtype=str).fillna("").set_index("lake_id", drop=False)


def save_registry(reg):
    reg.reset_index(drop=True).to_csv(REGISTRY, index=False)


def lakes_with_data():
    reg = registry()
    return [l for l in reg.index if os.path.exists(os.path.join(BASE, "data", l, "windows.csv"))]


def _f(x):
    if x is None or pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ("", "nan", "none", "null"):
        return None
    try:
        val = float(s)
        return None if np.isnan(val) else val
    except (ValueError, TypeError):
        return None


def use(lake_id):
    """Point every module (fetch, model, pipeline, app) at one lake."""
    g = globals()
    r = registry().loc[lake_id]
    g["LAKE"], g["LAKE_NAME"] = lake_id, r["name"] or lake_id
    if r["bbox"].strip():
        g["BBOX"] = [float(x) for x in r["bbox"].split(",")]
    elif _f(r["lon"]) is not None and _f(r["lat"]) is not None:
        g["BBOX"] = make_bbox(float(r["lon"]), float(r["lat"]), _f(r["half_km"]) or 1.5)
    else:
        g["BBOX"] = None                                          # position unknown: fetch is disabled for this lake
    sl, sa = _f(r["seed_lon"]), _f(r["seed_lat"])
    g["SEED"] = (sl, sa) if sl is not None and sa is not None else (
        (float(r["lon"]), float(r["lat"])) if _f(r["lon"]) is not None and _f(r["lat"]) is not None else None)
    g["TILE"] = r["tile"] or None
    g["S1_TRACK"] = (r["s1_orbit_state"], int(float(r["s1_rel_orbit"]))) if r["s1_orbit_state"] and r["s1_rel_orbit"] else None
    g["MIN_AREA"] = _f(r["min_area"]) or DEFAULT_MIN_AREA
    g["KNOWN_EVENT_WINDOWS"] = [x.strip() for x in r["event_windows"].split(";") if x.strip()]
    g["EVENT_LABEL"] = r["event_label"]
    d = os.path.join(BASE, "data", lake_id)
    g["DATA_DIR"] = d
    g["HISTORY_CSV"] = os.path.join(d, "windows.csv")             # your notebook output
    g["LIVE_CSV"] = os.path.join(d, "live.csv")                   # windows fetched since
    g["RISK_CSV"] = os.path.join(d, "risk.csv")
    g["BACKTEST_CSV"] = os.path.join(d, "backtest.csv")
    g["MODEL_FILE"] = os.path.join(BASE, "models", f"{lake_id}.joblib")
    os.makedirs(os.path.join(BASE, "models"), exist_ok=True)
    return lake_id


use(os.environ.get("GLOF_LAKE", DEFAULT_LAKE))
