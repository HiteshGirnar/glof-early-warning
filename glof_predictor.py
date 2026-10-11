"""
Predictive Engine for Glacial Lake Outburst Flood (GLOF) Early Warning and Forecasting.

Key Capabilities:
1. Pre-Breach Early Warning (15 to 45 days lead time before potential overtopping/breach)
2. Runaway Drainage Detection (Catastrophic sudden lake area/volume collapse)
3. Multi-Horizon Time-Series Forecasting (Next 10d, 20d, 30d area & capacity projections)
4. Calibrated Multi-Tier Alert Engine (Normal, Advisory, Watch, Alert) with Explainable Drivers
"""

import numpy as np
import pandas as pd


class GLOFPredictor:
    def __init__(self, watch_threshold=55.0, alert_threshold=80.0):
        self.watch_threshold = watch_threshold
        self.alert_threshold = alert_threshold

    def predict(self, prep_df):
        """
        Runs comprehensive GLOF hazard prediction and multi-horizon forecasting
        on a preprocessed lake dataframe.
        
        Returns:
            DataFrame containing calibrated threat scores, driver attribution, 
            lead-time estimates, and multi-horizon forecast projections.
        """
        d = prep_df.copy()
        n = len(d)
        
        # -------------------------------------------------------------
        # 1. Physical Pre-Breach Hazard Component (Overtopping & Failure)
        # -------------------------------------------------------------
        # Moraine dam freeboard failure requires high capacity fill.
        # Below 50% capacity, danger of overtopping wave is minimal.
        # Above 85% capacity, freeboard is critical. Above 95%, dam is at bursting pressure.
        cap = d["capacity_ratio"].fillna(0.5).to_numpy()
        cap_factor = np.clip((cap - 0.50) / 0.50, 0.0, 1.0) ** 1.4 * 100.0

        # Expansion Surge Factor: Rapid area growth in the last 30 days
        exp_pct = d["expansion_pct_30d"].fillna(0.0).to_numpy()
        exp_factor = np.clip(exp_pct / 15.0, 0.0, 1.0) * 100.0

        # Cumulative Thermal Melt Factor: heatwave causing ablation & permafrost thaw
        th_cum = d["thermal_cum_60d"].fillna(0.0).to_numpy()
        th_factor = np.clip(th_cum / 120.0, 0.0, 1.0) * 100.0

        # Hanging Glacier Dynamic Acceleration Anomaly (Trigger for ice avalanches)
        vel_z = np.maximum(0.0, d["vel_accel_z"].fillna(0.0).to_numpy())
        vel_factor = np.clip(vel_z / 2.0, 0.0, 1.0) * 100.0

        # Non-linear Compound Interaction:
        # Pre-breach hazard occurs when high capacity is combined with rapid expansion and heat
        cap_weight = cap_factor / 100.0
        pre_breach = (
            0.45 * cap_factor + 
            0.25 * exp_factor * cap_weight + 
            0.20 * th_factor * cap_weight + 
            0.10 * vel_factor
        )

        # Critical surge override: if lake is at or above historical maximum capacity (> 95%)
        # with strong thermal melt forcing, trigger pre-breach Alert!
        critical_condition = (cap >= 0.95) & (th_factor >= 50.0)
        pre_breach = np.where(critical_condition, np.maximum(pre_breach, 85.0), pre_breach)
        pre_breach = np.clip(pre_breach, 0.0, 100.0).round(1)

        # -------------------------------------------------------------
        # 2. Catastrophic Drainage Shock Component (Breach Detection)
        # -------------------------------------------------------------
        drainage = np.zeros(n)
        drain_shock = d["drainage_shock"].fillna(False).to_numpy()
        for i in range(n):
            if drain_shock[i]:
                drainage[i] = 98.0
            elif i > 0 and drain_shock[i-1]:
                drainage[i] = 85.0  # post-breach aftermath
            elif i > 1 and drain_shock[i-2]:
                drainage[i] = 70.0
        drainage = drainage.round(1)

        # -------------------------------------------------------------
        # 3. Multi-Horizon Time-Series Forecasting (t+10d, t+20d, t+30d)
        # -------------------------------------------------------------
        # Project forward using recent 30-day velocity trend & seasonal baseline
        area_curr = d["area_state_filled"].to_numpy()
        rate_10d = d["expansion_rate_10d"].fillna(0.0).to_numpy()
        rate_30d = d["expansion_rate_30d"].fillna(0.0).to_numpy()
        
        # Smoothed momentum
        momentum = 0.6 * (rate_30d / 3.0) + 0.4 * rate_10d
        
        # Forecasts
        fc_10d = np.maximum(0.01, area_curr + momentum).round(4)
        fc_20d = np.maximum(0.01, area_curr + 2.0 * momentum * 0.9).round(4)
        fc_30d = np.maximum(0.01, area_curr + 3.0 * momentum * 0.8).round(4)
        
        # Uncertainty envelopes (+/- 1.645 std based on historical observation residuals)
        obs_res = np.nanstd(d["lake_area_km2"].diff()) if d["lake_area_km2"].notna().sum() > 10 else 0.05
        obs_res = max(obs_res, 0.03)
        fc_30d_low = np.maximum(0.01, fc_30d - 1.645 * obs_res * np.sqrt(3)).round(4)
        fc_30d_high = (fc_30d + 1.645 * obs_res * np.sqrt(3)).round(4)

        # Projected 30-day capacity ratio
        area_max = np.nanquantile(d["lake_area_km2"].dropna(), 0.98) if d["lake_area_km2"].notna().sum() >= 15 else 1.0
        area_min = np.nanquantile(d["lake_area_km2"].dropna(), 0.05) if d["lake_area_km2"].notna().sum() >= 15 else 0.0
        denom = max(area_max - area_min, 0.05)
        proj_cap_30d = np.clip((fc_30d - area_min) / denom, 0.0, 1.5).round(3)
        forecast_threat = np.clip((proj_cap_30d - 0.70) / 0.30, 0.0, 1.0) * 80.0

        # -------------------------------------------------------------
        # 4. Composite Calibrated GLOF Threat Score ($P_{GLOF}$)
        # -------------------------------------------------------------
        glof_threat = np.maximum(pre_breach, drainage)
        glof_threat = np.maximum(glof_threat, forecast_threat * 0.75).round(1)

        # Operational Alert Levels:
        # Normal (< 30), Advisory (30-55), Watch (55-80), Alert (>= 80)
        alert_levels = []
        for s in glof_threat:
            if s >= self.alert_threshold:
                alert_levels.append("Alert")
            elif s >= self.watch_threshold:
                alert_levels.append("Watch")
            elif s >= 30.0:
                alert_levels.append("Advisory")
            else:
                alert_levels.append("Normal")

        # -------------------------------------------------------------
        # 5. Primary Physical Driver Attribution
        # -------------------------------------------------------------
        primary_drivers = []
        lead_time_windows = []
        for i in range(n):
            if drainage[i] >= 80.0:
                primary_drivers.append("Catastrophic Breach Drainage")
                lead_time_windows.append("Immediate Breach / Outburst")
            elif cap[i] >= 0.95 and pre_breach[i] >= 70.0:
                primary_drivers.append(f"Critical Capacity Overfill ({cap[i]*100:.0f}%)")
                lead_time_windows.append("15-45d Early Warning Window")
            elif exp_factor[i] >= 50.0 and cap[i] >= 0.70:
                primary_drivers.append(f"Rapid Expansion Surge (+{exp_pct[i]:.0f}% in 30d)")
                lead_time_windows.append("15-30d Early Warning Window")
            elif th_factor[i] >= 60.0 and cap[i] >= 0.70:
                primary_drivers.append(f"Intense Thermal Meltwave ({th_cum[i]:.0f} units)")
                lead_time_windows.append("30-60d Early Warning Window")
            elif vel_factor[i] >= 60.0:
                primary_drivers.append("Glacier Velocity Instability Shock")
                lead_time_windows.append("15-45d Early Warning Window")
            elif glof_threat[i] >= 30.0:
                primary_drivers.append("Moderate Seasonal Expansion / Melt")
                lead_time_windows.append("Elevated Monitoring Window")
            else:
                primary_drivers.append("Stable Lake Dynamics")
                lead_time_windows.append("Stable Baseline")

        # Compile final outputs
        out = pd.DataFrame({
            "window": d["window"],
            "glof_threat_pct": glof_threat,
            "pre_breach_hazard": pre_breach,
            "drainage_score": drainage,
            "alert_level": alert_levels,
            "primary_driver": primary_drivers,
            "lead_time_window": lead_time_windows,
            "capacity_ratio": cap,
            "proj_cap_30d": proj_cap_30d,
            "area_forecast_10d": fc_10d,
            "area_forecast_20d": fc_20d,
            "area_forecast_30d": fc_30d,
            "area_forecast_lower_30d": fc_30d_low,
            "area_forecast_upper_30d": fc_30d_high,
            "data_confidence": d["data_confidence"]
        })
        return out


def evaluate_lead_time(res_df, event_date="2023-09-28"):
    """
    Evaluates detection lead time in days prior to a documented outburst event.
    """
    event_ts = pd.Timestamp(event_date)
    ev_idx = res_df[res_df["window"] == event_ts].index
    if len(ev_idx) == 0:
        return None
    idx = ev_idx[0]
    
    # Check prior windows
    lead_time_days = 0
    flagged_prior = []
    for step in range(1, 10):
        if idx - step < 0:
            break
        prev_row = res_df.iloc[idx - step]
        days_before = (event_ts - prev_row["window"]).days
        lvl = str(prev_row.get("glof_alert_level", prev_row.get("alert_level", prev_row.get("risk_level", "Low"))))
        threat = float(prev_row.get("glof_threat_pct", prev_row.get("risk_pct", 0.0)))
        if lvl in ("Watch", "Alert"):
            flagged_prior.append((days_before, lvl, threat))
            lead_time_days = max(lead_time_days, days_before)
            
    ev_row = res_df.loc[idx]
    ev_threat = float(ev_row.get("glof_threat_pct", ev_row.get("risk_pct", 0.0)))
    ev_alert = str(ev_row.get("glof_alert_level", ev_row.get("alert_level", ev_row.get("risk_level", "Low"))))

    return {
        "event_window": event_date,
        "lead_time_days": lead_time_days,
        "flagged_windows_prior": flagged_prior,
        "event_day_threat": ev_threat,
        "event_day_alert": ev_alert
    }
