"""
Advanced Glaciological Data Preprocessor for Glacial Lake Outburst Flood (GLOF) Early Warning.

Features:
- Causal missing data reconstruction with observation age tracking
- Optical / SAR cross-validation and data confidence scoring
- Moraine dam storage capacity ratio (A / A_max) and freeboard depletion estimation
- Multi-window expansion surge velocity (10d, 30d, 60d) and acceleration
- Empirical lake volume calculation (Huggel / Cook scaling: V = 0.04355 * A^1.23)
- Cumulative thermal ablation forcing (degree-day heatwaves triggering melt & permafrost thaw)
- Hanging glacier tongue dynamics & acceleration anomaly (velocity_trend, velocity_accel)
- Sudden catastrophic drainage shock detector
"""

import numpy as np
import pandas as pd


def compute_baseline_stats(df, baseline_end="2024-12-31"):
    """
    Computes glaciological baseline statistics strictly on data <= baseline_end.
    These parameters are fixed during model training and reused at inference time
    to guarantee zero future data leakage.
    """
    d = df.copy()
    d["window"] = pd.to_datetime(d["window"])
    baseline_ts = pd.Timestamp(baseline_end)
    train_mask = d["window"] <= baseline_ts
    train_data = d[train_mask] if train_mask.any() else d

    # Lake Area statistics
    area_train = train_data["lake_area_km2"].dropna()
    has_area = len(area_train) >= 15
    if has_area:
        area_min = float(area_train.quantile(0.05))
        area_max = float(area_train.quantile(0.98))
        area_med = float(area_train.median())
    else:
        area_min, area_max, area_med = 0.0, 1.0, 0.5

    # Thermal seasonal statistics
    m = train_data["window"].dt.month
    therm_med = train_data.groupby(m)["thermal_index"].median()
    therm_std = train_data.groupby(m)["thermal_index"].std().replace(0, np.nan).fillna(15.0)

    # Radar statistics
    vv_train = train_data["vv_db"].dropna()
    if len(vv_train) >= 15:
        vv_med = float(vv_train.median())
        vv_std = float(vv_train.std()) or 1.0
    else:
        vv_med, vv_std = -10.0, 1.0

    # Glacier dynamics statistics
    vel_train = train_data.get("velocity_trend", pd.Series(dtype=float)).dropna()
    if len(vel_train) >= 15 and vel_train.std() > 0:
        vel_mean, vel_std = float(vel_train.mean()), float(vel_train.std())
    else:
        vel_mean, vel_std = 0.0, 1.0

    acc_train = train_data.get("velocity_accel", pd.Series(dtype=float)).dropna()
    if len(acc_train) >= 15 and acc_train.std() > 0:
        acc_mean, acc_std = float(acc_train.mean()), float(acc_train.std()) or 1.0
    else:
        acc_mean, acc_std = 0.0, 1.0

    # Observation difference residual
    obs_res = float(np.nanstd(train_data["lake_area_km2"].diff())) if train_data["lake_area_km2"].notna().sum() > 10 else 0.05
    obs_res = max(obs_res, 0.03)

    return {
        "area_min": area_min,
        "area_max": area_max,
        "area_med": area_med,
        "therm_med": therm_med,
        "therm_std": therm_std,
        "vv_med": vv_med,
        "vv_std": vv_std,
        "vel_mean": vel_mean,
        "vel_std": vel_std,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "obs_res": obs_res,
    }


def preprocess_lake_series(df, baseline_end="2024-12-31", baseline_stats=None):
    """
    Transforms raw 10-day satellite window observations into clean, causal, 
    glaciologically-grounded time-series features.
    
    Parameters:
        df: DataFrame containing at least 'window', 'lake_area_km2', 'vv_db', 'thermal_index'
        baseline_end: Date string for training baseline end (strictly causal, prevents data leakage)
        baseline_stats: Pre-computed baseline statistics dictionary from training
        
    Returns:
        DataFrame with enriched glaciological signals and data confidence indicators.
    """
    d = df.copy()
    d["window"] = pd.to_datetime(d["window"])
    d = d.sort_values("window").drop_duplicates("window", keep="last").reset_index(drop=True)
    
    # Standardize column names and types
    numeric_cols = [
        "lake_area_km2", "lake_area_min_km2", "vv_db", 
        "sar_backscatter_change", "thermal_index", 
        "velocity_trend", "velocity_accel", "observability"
    ]
    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            d[c] = np.nan

    # 1. Baseline statistics (learned during training or strictly <= baseline_end)
    if baseline_stats is None:
        baseline_stats = compute_baseline_stats(d, baseline_end=baseline_end)
    
    area_min = baseline_stats["area_min"]
    area_max = baseline_stats["area_max"]
    area_med = baseline_stats["area_med"]
    therm_med = baseline_stats["therm_med"]
    therm_std = baseline_stats["therm_std"]

    # 2. Causal Data Reconstruction (Forward State Tracking)
    # Optical area can be clouded for 1-3 windows in monsoon.
    # Forward-fill up to 4 windows (40 days) for state estimation.
    d["area_obs"] = d["lake_area_km2"]
    d["area_state"] = d["lake_area_km2"].ffill(limit=4)
    # Fallback to median if lake has no observations yet
    d["area_state_filled"] = d["area_state"].fillna(area_med)
    
    # Observation age: number of 10-day windows since the last confirmed optical reading
    is_obs = d["lake_area_km2"].notna().astype(int)
    obs_age = np.zeros(len(d), dtype=int)
    cur_age = 0
    for i in range(len(d)):
        if is_obs.iloc[i]:
            cur_age = 0
        else:
            cur_age += 1
        obs_age[i] = cur_age
    d["obs_age"] = obs_age
    
    # Data Confidence (0.0 to 1.0)
    # High if fresh optical + radar present; decays gracefully as observation ages.
    radar_avail = d["vv_db"].notna().astype(float)
    base_obs = d["observability"].fillna(0.6).clip(0.1, 1.0)
    conf = (0.6 * np.exp(-0.25 * d["obs_age"]) + 0.4 * radar_avail) * base_obs
    d["data_confidence"] = np.clip(conf, 0.05, 1.0).round(2)

    # 3. Glaciological Feature Engineering
    # A. Capacity Fill Ratio: ratio of current area between baseline min and historical max
    # As capacity_ratio approaches 1.0, the lake occupies its maximum basin volume and moraine freeboard is zero.
    denom = max(area_max - area_min, 0.05)
    d["capacity_ratio"] = np.clip((d["area_state_filled"] - area_min) / denom, 0.0, 1.5).round(3)
    
    # B. Expansion Surge Velocity (10d, 30d, 60d rates)
    d["expansion_rate_10d"] = d["area_state"].diff(1).round(4)
    d["expansion_rate_30d"] = d["area_state"].diff(3).round(4)
    d["expansion_rate_60d"] = d["area_state"].diff(6).round(4)
    d["expansion_accel"] = d["expansion_rate_30d"].diff(2).round(4)
    
    # Relative expansion rate (%) over 30 days
    base_shift = d["area_state"].shift(3).replace(0, np.nan)
    d["expansion_pct_30d"] = (d["expansion_rate_30d"] / base_shift * 100).clip(-100, 200).round(1)

    # C. Empirical Volume & Volume Growth (Huggel / Cook / IPCC empirical scaling)
    # V = 0.04355 * A^1.23 * 1000 (volume in thousand m³ / MCM)
    d["vol_est_mcm"] = (0.04355 * (np.maximum(d["area_state_filled"], 0.001) ** 1.23) * 1000.0).round(2)
    d["vol_delta_30d"] = d["vol_est_mcm"].diff(3).round(2)

    # D. Sudden Catastrophic Drainage Shock (Breach Detector)
    # A sudden drop > 15% and > 0.10 km² indicates rapid lake emptying / dam breach
    area_diff_1 = d["lake_area_km2"].diff(1)
    drainage_drop = (area_diff_1 < -0.10) & ((area_diff_1 / d["lake_area_km2"].shift(1)) < -0.15)
    d["drainage_shock"] = drainage_drop.fillna(False)

    # E. Thermal Melt Forcing & Heatwaves
    curr_m = d["window"].dt.month
    m_med = curr_m.map(therm_med).fillna(50.0)
    m_std = curr_m.map(therm_std).fillna(20.0)
    d["thermal_z"] = ((d["thermal_index"] - m_med) / m_std).round(2)
    d["thermal_excess"] = np.maximum(0, d["thermal_index"] - m_med).round(2)
    # 60-day cumulative heat wave melt forcing (rolling 6 windows)
    d["thermal_cum_60d"] = d["thermal_excess"].rolling(6, min_periods=1).sum().round(1)
    d["thermal_surge_30d"] = d["thermal_index"].diff(3).round(1)

    # F. Radar Moraine Dynamics
    vv_med = baseline_stats.get("vv_med", -10.0)
    vv_std = baseline_stats.get("vv_std", 1.0) or 1.0
    d["vv_dev"] = ((d["vv_db"] - vv_med) / vv_std).round(2)
    d["vv_diff_10d"] = d["vv_db"].diff(1).round(2)

    # G. Hanging Glacier Velocity & Dynamic Instability
    vel_mean = baseline_stats.get("vel_mean", 0.0)
    vel_std = baseline_stats.get("vel_std", 1.0) or 1.0
    d["vel_trend_z"] = ((d["velocity_trend"] - vel_mean) / vel_std).round(2)

    acc_mean = baseline_stats.get("acc_mean", 0.0)
    acc_std = baseline_stats.get("acc_std", 1.0) or 1.0
    d["vel_accel_z"] = ((d["velocity_accel"] - acc_mean) / acc_std).round(2)

    d.attrs["baseline_stats"] = baseline_stats
    return d
