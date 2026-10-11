"""
Unsupervised GLOF risk model (Isolation Forest). Missing values are kept as null, never filled.

Three small Isolation Forests, one per signal group, each trained only on windows where that
signal was observed:   AREA (optical)  |  RADAR (Sentinel-1 VV)  |  THERMAL (MODIS)
A group that has no data in a window simply does not vote. Overall risk = the most unusual
group that IS available, re-ranked against history windows that had the same number of groups.
"""
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
import config as C
import preprocessor as P
import glof_predictor as GP

BASE = ["lake_area_km2", "lake_area_min_km2", "vv_db", "sar_backscatter_change", "thermal_index"]
GROUPS = {   # group -> (column that must be observed, features the forest uses)
    "area":    ("lake_area_km2", ["area_dev", "area_step"]),
    "radar":   ("vv_db",         ["vv_dev", "vv_change"]),
    "thermal": ("thermal_index", ["thermal_z", "thermal_change"]),
}
Z = {"thermal_z": "thermal_index"}     # temperature is truly seasonal -> compare with same month
BASELINE_N, BASELINE_MIN, MAX_GAP_DAYS = 6, 3, 60   # "recent normal" = median of last 6 observations


def load_windows(path):
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "window"})
    df["window"] = pd.to_datetime(df["window"])
    for c in BASE + ["observability"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df.sort_values("window").drop_duplicates("window", keep="last").reset_index(drop=True)


def recent_dev(df, col, n=BASELINE_N, min_obs=BASELINE_MIN):
    """Value minus the median of the last n OBSERVED values before it (null stays null).
    Looks only backwards, so slow trends (a growing lake) are absorbed and sudden jumps stand out."""
    s = df.set_index("window")[col].dropna()
    base = s.shift(1).rolling(n, min_periods=min_obs).median()
    return (s - base).reindex(df["window"]).values


def step_from_last(df, col, max_gap=MAX_GAP_DAYS):
    """Change since the previous OBSERVED value, only if it was at most max_gap days ago."""
    s = df.set_index("window")[col].dropna()
    gap = s.index.to_series().diff().dt.days
    return s.diff().where(gap <= max_gap).reindex(df["window"]).values


def seasonal_stats(train):
    m = train["window"].dt.month
    return {c: train.groupby(m)[c].agg(["mean", "std"]) for c in Z.values()}


def build_features(df, stats):
    df = df.copy()
    m = df["window"].dt.month
    for z, col in Z.items():                      # distance from the same month's normal (in std units)
        mean, std = m.map(stats[col]["mean"]), m.map(stats[col]["std"]).replace(0, np.nan)
        df[z] = (df[col] - mean) / std
    df["area_dev"] = recent_dev(df, "lake_area_km2")
    df["area_step"] = step_from_last(df, "lake_area_km2")
    df["vv_dev"] = recent_dev(df, "vv_db")
    df["vv_change"] = df["vv_db"].diff()
    df["thermal_change"] = df["thermal_index"].diff()
    return df


MIN_TRAIN_ROWS = 25


def _subsets(feats):
    return [tuple(x for x, on in zip(feats, mask) if on)
            for mask in ((1, 1), (1, 0), (0, 1))]


def _group_raw(f, models):
    """Raw anomaly score per group. Each window is scored by the forest that matches the features
    it actually has (nulls are never filled). No matching forest / no data -> null."""
    out = pd.DataFrame(np.nan, index=f.index, columns=list(GROUPS))
    for g, (need, feats) in GROUPS.items():
        obs = f[list(feats)].notna()
        for sub in _subsets(feats):
            m = models.get((g, sub))
            if m is None:
                continue
            sel = f[need].notna() & (obs.sum(axis=1) == len(sub)) & obs[list(sub)].all(axis=1)
            if sel.any():
                out.loc[sel, g] = -m.score_samples(f.loc[sel, list(sub)])
        # score with the matching forest's own reference below (see _group_pct)
    return out


def _fit_group_models(f):
    models = {}
    for g, (need, feats) in GROUPS.items():
        for sub in _subsets(feats):
            rows = f[f[need].notna()][list(sub)].dropna()
            if len(rows) >= MIN_TRAIN_ROWS:
                models[(g, sub)] = IsolationForest(n_estimators=200, random_state=42).fit(rows)
    return models


def _group_pct(f, models, refs):
    """Percentile (0-100) of each group's anomaly score vs. that forest's own training history."""
    pct = pd.DataFrame(np.nan, index=f.index, columns=list(GROUPS))
    for g, (need, feats) in GROUPS.items():
        obs = f[list(feats)].notna()
        for sub in _subsets(feats):
            m = models.get((g, sub))
            if m is None:
                continue
            sel = f[need].notna() & obs[list(sub)].all(axis=1) & (obs.sum(axis=1) == len(sub))
            if sel.any():
                raw = -m.score_samples(f.loc[sel, list(sub)])
                pct.loc[sel, g] = _rank(refs[(g, sub)], raw)
    return pct


def _rank(ref, x):
    return np.searchsorted(ref, x, side="right") / len(ref) * 100


LAKE_GROUPS = ["area", "radar"]      # direct evidence about the lake itself


def _lake_top(pct):
    return pct[LAKE_GROUPS].max(axis=1), pct[LAKE_GROUPS].notna().sum(axis=1)


def fit(train_df):
    """Learn 'normal' from train_df only (callers pass windows <= TRAIN_END)."""
    stats = seasonal_stats(train_df)
    f = build_features(train_df, stats)
    models = _fit_group_models(f)
    refs = {}
    for (g, sub), m in models.items():
        need, feats = GROUPS[g]
        rows = f[f[need].notna()][list(sub)].dropna()
        refs[(g, sub)] = np.sort(-m.score_samples(rows))
    pct = _group_pct(f, models, refs)
    top, k = _lake_top(pct)
    ref_lake = {n: np.sort(top[k == n].dropna().values) for n in (1, 2)}
    ref_lake["all"] = np.sort(top[k > 0].dropna().values)
    return {"models": models, "stats": stats, "refs": refs, "ref_lake": ref_lake}


def score(df, art):
    """Calibrated GLOF threat % and risk per window.
    df must be the full continuous series (so change features work).
    
    Combines:
    1. Unsupervised Anomaly Scoring (Isolation Forest across Area, Radar, Thermal)
    2. Physical Glaciological Pre-Breach Early Warning (Freeboard capacity ratio, expansion surge, heatwave ablation)
    3. Catastrophic Outburst / Dam Drainage Detection
    4. Multi-Horizon Time-Series Forecasting (10d, 20d, 30d projections with confidence bands)
    """
    f = build_features(df, art["stats"])
    pct = _group_pct(f, art["models"], art["refs"])
    top, k = _lake_top(pct)
    lake = pd.Series(np.nan, index=df.index)
    for i in df.index[k > 0]:
        ref = art["ref_lake"].get(int(k[i]))
        ref = ref if ref is not None and len(ref) >= 20 else art["ref_lake"]["all"]
        lake[i] = _rank(ref, top[i])
    therm = pct["thermal"] * C.THERMAL_WEIGHT
    anomaly_risk = pd.concat([lake, therm], axis=1).max(axis=1, skipna=True)        # NaN only if both are NaN
    driver = pd.Series(np.nan, index=df.index, dtype=object)
    for i in df.index[anomaly_risk.notna()]:
        if pd.notna(lake[i]) and lake[i] >= (therm[i] if pd.notna(therm[i]) else -1):
            driver[i] = pct.loc[i, LAKE_GROUPS].astype(float).idxmax()
        else:
            driver[i] = "thermal"

    out = df[["window"] + BASE + ["observability"]].copy()
    for g in GROUPS:
        out[f"{g}_pct"] = pct[g].round(1).values
    out["signals_used"] = pct.notna().sum(axis=1).values
    out["anomaly_risk_pct"] = anomaly_risk.round(1).values

    # Glaciological Data Preprocessing & GLOF Prediction Engine
    prep = P.preprocess_lake_series(df, baseline_end=C.TRAIN_END)
    predictor = GP.GLOFPredictor(watch_threshold=C.WATCH_PCT, alert_threshold=C.ALERT_PCT)
    glof_pred = predictor.predict(prep)

    out["glof_threat_pct"] = glof_pred["glof_threat_pct"].values
    out["pre_breach_hazard"] = glof_pred["pre_breach_hazard"].values
    out["drainage_score"] = glof_pred["drainage_score"].values
    out["glof_alert_level"] = glof_pred["alert_level"].values
    out["primary_driver"] = glof_pred["primary_driver"].values
    out["lead_time_window"] = glof_pred["lead_time_window"].values
    out["capacity_ratio"] = glof_pred["capacity_ratio"].values
    out["vol_est_mcm"] = prep["vol_est_mcm"].values
    out["area_forecast_10d"] = glof_pred["area_forecast_10d"].values
    out["area_forecast_20d"] = glof_pred["area_forecast_20d"].values
    out["area_forecast_30d"] = glof_pred["area_forecast_30d"].values
    out["area_forecast_lower_30d"] = glof_pred["area_forecast_lower_30d"].values
    out["area_forecast_upper_30d"] = glof_pred["area_forecast_upper_30d"].values
    out["data_confidence"] = glof_pred["data_confidence"].values

    # Composite Operational Risk Score:
    # Reflects the highest of unsupervised anomaly and calibrated physical GLOF threat
    out["risk_pct"] = np.maximum(
        out["anomaly_risk_pct"].fillna(0),
        out["glof_threat_pct"].fillna(0)
    ).replace(0, np.nan).round(1).values

    out["main_driver"] = np.where(
        out["glof_threat_pct"] >= 50.0,
        out["primary_driver"],
        driver
    )
    out["risk_level"] = pd.cut(out["risk_pct"], [-1, C.WATCH_PCT, C.ALERT_PCT, 100], labels=["Low", "Watch", "Alert"])
    out["period"] = np.where(out["window"] <= pd.Timestamp(C.TRAIN_END), "train (2017-24)", "live scoring")
    return out


def train_production(df):
    """Final model: windows up to TRAIN_END (2017-2024) except the known outburst window(s)."""
    normal = df[(df["window"] <= pd.Timestamp(C.TRAIN_END))
                & ~df["window"].isin(pd.to_datetime(C.KNOWN_EVENT_WINDOWS))]
    art = fit(normal)
    joblib.dump(art, C.MODEL_FILE)
    return art


def backtest(df, train_end="2021-12-31", event=None):
    """Honest test: learn on early years only, then score later years the model never saw."""
    if not event and not C.KNOWN_EVENT_WINDOWS:
        return None, None                                    # no known event for this lake: nothing to test against
    event = pd.to_datetime(event or C.KNOWN_EVENT_WINDOWS[0])
    art = fit(df[df["window"] <= train_end])
    res = score(df, art)
    res["period"] = np.where(res["window"] <= train_end, "train",
                    np.where(res["window"] < event, "held-out normal", "event & after"))
    return res, art
