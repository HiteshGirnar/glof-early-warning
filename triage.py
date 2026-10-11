"""
Fleet triage: turns per-lake risk scores into a short, explainable list of lakes to look at.

Why it exists: a per-lake risk % alone raises too many flags (about 45% of the high scores in this data fall in windows where
many lakes spike together, mostly April-June snowmelt). A lake is only flagged when it ALSO stands out from its peers.

    ALERT     risk >= 95  and  at least +40 points above the fleet median      -> look at the imagery now
    WATCH     risk >= 80  and  at least +25 points above the fleet median      -> look this window
    REGIONAL  risk >= 80  but many lakes are high at once (>=40% of lakes)     -> likely seasonal / sensor, not flagged
    CLEAR     assessed, nothing unusual
    NO DATA   no usable signal: cannot assess (never shown as clear)

With fewer than MIN_PEERS lakes the peer test is skipped and plain thresholds are used.
These thresholds were set to cap workload, not tuned for detection: with one recorded event nothing can be tuned honestly.
"""
import os
import numpy as np, pandas as pd
import config as C
import model as M

ALERT_RISK, ALERT_EXCESS = 95, 40
WATCH_RISK, WATCH_EXCESS = 80, 25
REGIONAL_SHARE = 0.40
MIN_PEERS = 8
TIERS = ["NO DATA", "CLEAR", "REGIONAL", "WATCH", "ALERT"]          # index = code used by the GUI
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ------------------------------------------------------------------ load
def load_fleet(risk_files=None):
    """dict lake_id -> risk dataframe indexed by window (production scores)."""
    out = {}
    for lake in C.lakes_with_data():
        p = (risk_files or {}).get(lake) or os.path.join(C.BASE, "data", lake, "risk.csv")
        if os.path.exists(p):
            out[lake] = pd.read_csv(p, parse_dates=["window"]).set_index("window")
    return out


def last_window_with_data(frames):
    R = pd.DataFrame({l: f.risk_pct for l, f in frames.items()})
    ok = R.notna().any(axis=1)
    return R.index[ok][-1]


# ------------------------------------------------------------------ core rule (vectorised over all windows)
def tiers(R):
    """R: windows x lakes risk. Returns (tier codes DataFrame, fleet median, excess, share of lakes >= Watch, regional flag)."""
    n = R.notna().sum(axis=1)
    med = R.median(axis=1).where(n >= MIN_PEERS)
    share = (R >= C.WATCH_PCT).sum(axis=1) / n.replace(0, np.nan)
    regional = (share >= REGIONAL_SHARE) & (n >= MIN_PEERS)
    E = R.sub(med, axis=0)
    have_peers = med.notna()
    alert = (R >= ALERT_RISK) & ((E >= ALERT_EXCESS) | (~have_peers.to_numpy()[:, None] & R.notna()))
    watch = (R >= WATCH_RISK) & ((E >= WATCH_EXCESS) | (~have_peers.to_numpy()[:, None] & R.notna())) & ~alert
    reg = (R >= WATCH_RISK) & regional.to_numpy()[:, None] & ~alert & ~watch
    code = pd.DataFrame(1, index=R.index, columns=R.columns)
    code[reg] = 2; code[watch] = 3; code[alert] = 4
    code[R.isna()] = 0
    return code, med, E, share, regional


def streaks(code):
    """Consecutive windows (up to and including each one) at WATCH or ALERT."""
    flagged = (code >= 3).astype(int).to_numpy()
    out = np.zeros_like(flagged)
    for i in range(len(flagged)):
        out[i] = np.where(flagged[i] == 1, (out[i - 1] if i else 0) + 1, 0)
    return pd.DataFrame(out, index=code.index, columns=code.columns)


# ------------------------------------------------------------------ plain-language reasons
def _recent_median(s, i, n=6, min_obs=3):
    prev = s.iloc[:i].dropna().tail(n)
    return float(prev.median()) if len(prev) >= min_obs else np.nan


def reasons(f, i, med_i, excess_i):
    """Short human-readable reasons for lake frame f at integer position i (only signals that are actually unusual)."""
    row, out = f.iloc[i], []

    # Check for dedicated glaciological GLOF hazards
    drainage = row.get("drainage_score", 0) if hasattr(row, "get") else (row["drainage_score"] if "drainage_score" in row else 0)
    pre_breach = row.get("pre_breach_hazard", 0) if hasattr(row, "get") else (row["pre_breach_hazard"] if "pre_breach_hazard" in row else 0)
    cap = row.get("capacity_ratio", np.nan) if hasattr(row, "get") else (row["capacity_ratio"] if "capacity_ratio" in row else np.nan)
    lead_time = row.get("lead_time_window", "") if hasattr(row, "get") else (row["lead_time_window"] if "lead_time_window" in row else "")

    if pd.notna(drainage) and drainage >= 80:
        out.append("Catastrophic lake drainage detected (potential moraine breach / outburst)")
    elif pd.notna(pre_breach) and pre_breach >= 70:
        cap_str = f"lake at {cap * 100:.0f}% historical capacity" if pd.notna(cap) else "critical lake capacity"
        out.append(f"Pre-breach warning: {cap_str} ({lead_time})")

    sig = [("area_pct", "area"), ("radar_pct", "radar"), ("thermal_pct", "thermal")]
    order = sorted([(row[c], k) for c, k in sig if c in row and pd.notna(row[c]) and row[c] >= 80], reverse=True)
    for _, k in order:
        if k == "area":
            a, m = row.lake_area_km2, _recent_median(f.lake_area_km2, i)
            if pd.notna(a) and pd.notna(m) and m > 0:
                out.append(f"Lake area {a:.2f} km² vs recent median {m:.2f} ({(a - m) / m * 100:+.0f}%)")
            else:
                out.append("Lake area change is unusual")
        elif k == "radar":
            v, m = row.vv_db, _recent_median(f.vv_db, i)
            out.append(f"Radar backscatter {v:.1f} dB, {v - m:+.1f} dB vs recent median" if pd.notna(v) and pd.notna(m) else "Radar backscatter is unusual")
        else:
            tr = f[f.index <= pd.Timestamp(C.TRAIN_END)]
            same = tr[tr.index.month == f.index[i].month].thermal_index.dropna()
            t = row.thermal_index
            if pd.notna(t) and len(same) >= 3:
                out.append(f"Temperature index {t:.0f} vs {same.median():.0f} typical for {MONTHS[f.index[i].month - 1]} (context only)")
    if pd.notna(med_i):
        out.append(f"{row.risk_pct:.0f}% vs fleet median {med_i:.0f}% ({excess_i:+.0f} points vs peers)")
    if pd.isna(row.lake_area_km2):
        out.append("Lake area not observed this window (cloud/snow or lake not found)")
    elif small_lake(f):
        out.append("Small lake: area readings are close to the noise level")
    return out[:4]


SMALL_LAKE_KM2 = 0.10          # below this the area readings are close to the noise level (a few dozen pixels)


def small_lake(f):
    tr = f[f.index <= pd.Timestamp(C.TRAIN_END)].lake_area_km2.dropna()
    return bool(len(tr) < 25 or tr.median() < SMALL_LAKE_KM2)


def confidence(f, i):
    r = f.iloc[i]
    a, v = pd.notna(r.lake_area_km2), pd.notna(r.vv_db)
    c = "high" if (a and v) else ("medium" if (a or v) else "low")
    return "medium" if (c == "high" and small_lake(f)) else c       # small lakes never get "high"


# ------------------------------------------------------------------ snapshot for one window (CSV / report / GUI)
def snapshot(frames, when=None):
    R = pd.DataFrame({l: f.risk_pct for l, f in frames.items()})
    when = pd.Timestamp(when) if when is not None else last_window_with_data(frames)
    R = R[R.index <= when]
    code, med, E, share, regional = tiers(R)
    st = streaks(code)
    i = len(R) - 1
    reg = C.registry()
    rows = []
    for lake, f in frames.items():
        f = f.reindex(R.index)
        c = int(code[lake].iloc[-1]); r = R[lake].iloc[-1]
        lat, lon = reg.loc[lake, "seed_lat"] or reg.loc[lake, "lat"], reg.loc[lake, "seed_lon"] or reg.loc[lake, "lon"]
        try:
            lat, lon = float(lat), float(lon)
            link = f"https://apps.sentinel-hub.com/eo-browser/?zoom=15&lat={lat}&lng={lon}&themeId=DEFAULT-THEME&datasetId=S2L2A"
        except (TypeError, ValueError):
            lat = lon = np.nan; link = ""
        
        last_row = f.iloc[-1]
        pre_b = float(last_row["pre_breach_hazard"]) if "pre_breach_hazard" in last_row and pd.notna(last_row["pre_breach_hazard"]) else 0.0
        drain_s = float(last_row["drainage_score"]) if "drainage_score" in last_row and pd.notna(last_row["drainage_score"]) else 0.0
        cap_val = float(last_row["capacity_ratio"]) if "capacity_ratio" in last_row and pd.notna(last_row["capacity_ratio"]) else np.nan
        lead_val = str(last_row["lead_time_window"]) if "lead_time_window" in last_row and pd.notna(last_row["lead_time_window"]) else ""

        rows.append(dict(
            lake_id=lake, name=reg.loc[lake, "name"] or lake, region=reg.loc[lake, "region"], window=R.index[-1].date(),
            tier=TIERS[c], risk_pct=r, fleet_median=med.iloc[-1], excess=E[lake].iloc[-1], streak=int(st[lake].iloc[-1]),
            confidence=confidence(f, i) if c else "none", why=" | ".join(reasons(f, i, med.iloc[-1], E[lake].iloc[-1])) if c else "No usable signal: cannot assess",
            pre_breach_hazard=pre_b, drainage_score=drain_s, capacity_ratio=cap_val, lead_time=lead_val,
            lat=lat, lon=lon, imagery_link=link))
    out = pd.DataFrame(rows)
    out["_o"] = out.tier.map({t: k for k, t in enumerate(["ALERT", "WATCH", "REGIONAL", "CLEAR", "NO DATA"])})
    out = out.sort_values(["_o", "excess"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
    out.attrs.update(window=R.index[-1], regional=bool(regional.iloc[-1]), share=float(share.iloc[-1]), fleet_median=float(med.iloc[-1]) if pd.notna(med.iloc[-1]) else None)
    return out


# ------------------------------------------------------------------ workload + event checks
def workload(frames):
    """How many flags the rule produces. Training years vs live years, per lake-year and per 10-day window."""
    R = pd.DataFrame({l: f.risk_pct for l, f in frames.items()})
    R = R[R.index <= last_window_with_data(frames)]
    code, med, E, share, regional = tiers(R)
    out = {}
    for name, m in (("2017-2024 (learned)", R.index <= pd.Timestamp(C.TRAIN_END)), ("2025+ (live)", R.index > pd.Timestamp(C.TRAIN_END))):
        c = code[m]
        if len(c) == 0:
            continue
        lake_years = len(c) / 36 * c.shape[1]
        out[name] = dict(windows=len(c), alerts_per_lake_year=round(float((c == 4).sum().sum() / lake_years), 2),
                         watch_or_alert_per_lake_year=round(float((c >= 3).sum().sum() / lake_years), 2),
                         flags_per_window_alert=round(float((c == 4).sum(axis=1).mean()), 2),
                         flags_per_window_watch_or_alert=round(float((c >= 3).sum(axis=1).mean()), 2),
                         regional_windows_pct=round(float(regional[m].mean() * 100)),
                         raw_high_in_regional_pct=round(float(((R[m] >= 95).sum(axis=1)[regional[m]].sum()) / max((R[m] >= 95).sum().sum(), 1) * 100)))
    return out


def event_checks(frames):
    """Recorded outbursts (lakes.csv event_windows): what did the triage say at and before the event?"""
    R = pd.DataFrame({l: f.risk_pct for l, f in frames.items()})
    R = R[R.index <= last_window_with_data(frames)]
    code, med, E, share, regional = tiers(R)
    reg, res = C.registry(), []
    flagged = (code >= 3).astype(int)
    base90 = float(np.mean([(flagged.iloc[max(0, j - 9):j].sum() > 0).mean() for j in range(9, len(R), 3)]))   # chance a random lake has a flag in a 90-day span
    for lake in R.columns:
        for w in [x.strip() for x in reg.loc[lake, "event_windows"].split(";") if x.strip()]:
            w = pd.Timestamp(w)
            if w not in R.index:
                continue
            i = R.index.get_loc(w)
            before3, before9 = code[lake].iloc[max(0, i - 3):i], code[lake].iloc[max(0, i - 9):i]
            res.append(dict(lake_id=lake, name=reg.loc[lake, "name"], event=reg.loc[lake, "event_label"] or str(w.date()), window=w.date(),
                            tier_at_event=TIERS[int(code[lake].iloc[i])], risk=float(R[lake].iloc[i]),
                            peer_rank=int(R.loc[w].rank(ascending=False)[lake]), n_lakes=int(R.loc[w].notna().sum()),
                            excess=float(E[lake].iloc[i]), flagged_30d_before=bool((before3 >= 3).any()), flagged_90d_before=bool((before9 >= 3).any()),
                            base_rate_90d=round(base90 * 100)))
    return res


# ------------------------------------------------------------------ honest out-of-sample test of the whole triage
def fleet_backtest(train_end="2021-12-31", save=True):
    """Every lake gets a model trained ONLY on windows <= train_end, scored afterwards; then the same triage rule runs on those scores."""
    scores = {}
    for lake in C.lakes_with_data():
        C.use(lake)
        h = M.load_windows(C.HISTORY_CSV)
        h = h[h.window.map(lambda s: (pd.Timestamp(s.year + 1, 1, 1) if s.dayofyear > 350 else s + pd.Timedelta(days=10))) <= pd.Timestamp.today().normalize()]
        art = M.fit(h[(h.window <= train_end) & ~h.window.isin(pd.to_datetime(C.KNOWN_EVENT_WINDOWS))])
        scores[lake] = M.score(h, art).set_index("window")
    C.use(C.DEFAULT_LAKE)
    R = pd.DataFrame({l: f.risk_pct for l, f in scores.items()})
    R = R[R.index > pd.Timestamp(train_end)]
    if save:
        R.to_csv(os.path.join(C.BASE, "data", "fleet_backtest.csv"))
    return R, scores


def backtest_summary(R):
    code, med, E, share, regional = tiers(R)
    reg, ev = C.registry(), []
    lake_years = len(R) / 36 * R.shape[1]
    out = dict(train_end="2021", windows=len(R), alerts_per_lake_year=round(float((code == 4).sum().sum() / lake_years), 2),
               watch_or_alert_per_lake_year=round(float((code >= 3).sum().sum() / lake_years), 2),
               raw_95_per_lake_year=round(float((R >= 95).sum().sum() / lake_years), 2), events=[])
    for lake in R.columns:
        for w in [x.strip() for x in reg.loc[lake, "event_windows"].split(";") if x.strip()]:
            w = pd.Timestamp(w)
            if w in R.index:
                i = R.index.get_loc(w)
                out["events"].append(dict(lake_id=lake, window=str(w.date()), tier=TIERS[int(code[lake].iloc[i])], risk=float(R[lake].iloc[i]),
                                          peer_rank=int(R.loc[w].rank(ascending=False)[lake]), n_lakes=int(R.loc[w].notna().sum()),
                                          flagged_30d_before=bool((code[lake].iloc[max(0, i - 3):i] >= 3).any())))
    return out


# ------------------------------------------------------------------ reports
def write_report(frames=None, out_dir=None):
    frames = frames or load_fleet()
    snap = snapshot(frames)
    out_dir = out_dir or os.path.join(C.BASE, "data")
    snap.to_csv(os.path.join(out_dir, "attention.csv"), index=False)
    w, a = snap.attrs["window"], snap
    lines = [f"# Lake attention report: window starting {w.date()}", "",
             f"Fleet: {int((a.tier != 'NO DATA').sum())}/{len(a)} lakes assessed. Fleet median risk "
             f"{snap.attrs['fleet_median']:.0f}%." if snap.attrs["fleet_median"] is not None else "", ""]
    if snap.attrs["regional"]:
        lines += [f"**Regional signal:** {snap.attrs['share'] * 100:.0f}% of lakes are high at once. Scores are likely seasonal or sensor-driven, so peer-based flags are the ones to trust.", ""]
    for tier in ("ALERT", "WATCH", "REGIONAL", "NO DATA"):
        sub = a[a.tier == tier]
        if len(sub):
            lines.append(f"## {tier} ({len(sub)})")
            for _, r in sub.iterrows():
                ex = f"{r.excess:+.0f} vs peers" if pd.notna(r.excess) else "no peer comparison"
                lines.append(f"- **{r['name']}** ({r.region or 'region n/a'}): {r.risk_pct:.0f}% ({ex}), streak {r.streak}, confidence {r.confidence}. {r.why}"
                             + (f" [imagery]({r.imagery_link})" if r.imagery_link else ""))
            lines.append("")
    lines += ["_Decision support only. A score says how unusual a window is, not that a flood will happen._"]
    open(os.path.join(out_dir, "attention.md"), "w", encoding="utf-8").write("\n".join(lines))
    return snap


def notify(snap, webhook=None, dry_run=True):
    """Post ALERT/WATCH lakes to a Slack/Teams/Discord-style incoming webhook ({"text": ...}). Dry run prints only."""
    hot = snap[snap.tier.isin(["ALERT", "WATCH"])]
    if hot.empty:
        return "No ALERT/WATCH lakes this window."
    msg = f"GLOF attention, window {snap.attrs['window'].date()}:\n" + "\n".join(
        f"- {r.tier}: {r['name']} {r.risk_pct:.0f}% ({r.excess:+.0f} vs peers, streak {r.streak}). {r.why.split(' | ')[0]}" for _, r in hot.iterrows())
    if dry_run or not webhook:
        return msg
    import json, urllib.request
    req = urllib.request.Request(webhook, data=json.dumps({"text": msg}).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=20)
    return msg
