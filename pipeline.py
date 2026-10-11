"""
python pipeline.py train  --lake south_lhonak     # learn 'normal' from 2017-2024, score every window (run once)
python pipeline.py train  --all                   # same for every lake that has data/<lake>/windows.csv + portfolio table
python pipeline.py update --lake south_lhonak     # fetch finished 10-day windows not yet stored, append, re-score
python pipeline.py update --all                   # same for every lake with a verified position (see setup_lake.py)
python pipeline.py report                         # fleet triage for the latest window -> data/attention.csv + attention.md
python pipeline.py report --send                  # also post ALERT/WATCH lakes to the webhook in env ALERT_WEBHOOK
"""
import argparse, os, sys
import numpy as np, pandas as pd, joblib
import config as C, model as M

LIVE_COLS = ["window", "lake_area_km2", "lake_area_min_km2", "vv_db",
             "sar_backscatter_change", "thermal_index", "observability"]
PORTFOLIO_CSV = os.path.join(C.BASE, "data", "portfolio.csv")


def completed_windows(last_done, today):
    """Your grid: 36 windows/year (start = 1 Jan + 10k days); the last one runs to 1 Jan.
    Returns (start, end) of every window after last_done whose end <= today."""
    out = []
    for y in range(last_done.year, today.year + 1):
        for k in range(36):
            s = pd.Timestamp(y, 1, 1) + pd.Timedelta(days=10 * k)
            e = s + pd.Timedelta(days=10) if k < 35 else pd.Timestamp(y + 1, 1, 1)
            if s > last_done and e <= today:
                out.append((s, e))
    return out


def window_end(s):
    return pd.Timestamp(s.year + 1, 1, 1) if s.dayofyear > 350 else s + pd.Timedelta(days=10)


def load_all(today=None):
    today = pd.Timestamp(today) if today else pd.Timestamp.today().normalize()
    hist = M.load_windows(C.HISTORY_CSV)
    hist = hist[hist["window"].map(window_end) <= today].reset_index(drop=True)   # drop future (not-yet-happened) windows
    empty = hist[M.BASE].isna().all(axis=1)
    while len(hist) and empty.iloc[-1]:        # trailing empty placeholder rows (notebook ran before they happened)
        hist, empty = hist.iloc[:-1], empty.iloc[:-1]     # -> not 'stored', so update will fetch them
    if os.path.exists(C.LIVE_CSV) and os.path.getsize(C.LIVE_CSV) > 0:
        live = M.load_windows(C.LIVE_CSV)
    else:
        live = pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "window" else "float64") for c in LIVE_COLS})
    return hist, live


def combined(hist, live):
    return pd.concat([hist, live]).drop_duplicates("window", keep="first").sort_values("window").reset_index(drop=True)


def train():
    hist, live = load_all()
    df = combined(hist, live)
    art = M.train_production(df)                      # learns ONLY from windows <= TRAIN_END (2017-24)
    scores = M.score(df, art)
    scores.to_csv(C.RISK_CSV, index=False)            # 2025+ windows are scored, never learned from
    bt, _ = M.backtest(hist[hist.window <= pd.Timestamp(C.TRAIN_END)])
    if bt is not None:
        bt.to_csv(C.BACKTEST_CSV, index=False)
    elif os.path.exists(C.BACKTEST_CSV):
        os.remove(C.BACKTEST_CSV)
    return scores


def run_update(today=None, fetch=None, progress=None):
    """Backfill + live. progress(i, n, start) is an optional callback (used by the app)."""
    if fetch is None:
        if C.BBOX is None or C.TILE is None:
            raise RuntimeError(f"{C.LAKE}: position/tile not set. Add lon/lat in lakes/lakes.csv and run "
                               f"`python setup_lake.py {C.LAKE}` first.")
        from fetch import fetch_window as fetch
    today = pd.Timestamp(today) if today else pd.Timestamp.today().normalize()
    hist, live = load_all(today)
    last = max(hist.window.max(), live.window.max() if len(live) else hist.window.max())
    todo = completed_windows(last, today)
    rows = []
    for i, (s, e) in enumerate(todo, 1):
        vals = fetch(s, e) or {}
        rows.append({"window": s, **{c: vals.get(c, np.nan) for c in LIVE_COLS[1:]}})
        if progress:
            progress(i, len(todo), s)
        pd.DataFrame(rows).pipe(lambda d: pd.concat([live, d])).drop_duplicates("window", keep="last") \
            .sort_values("window").to_csv(C.LIVE_CSV, index=False)      # saved after EVERY window: safe to resume
    stale = (os.path.exists(C.LIVE_CSV) and (not os.path.exists(C.RISK_CSV)
             or os.path.getmtime(C.RISK_CSV) < os.path.getmtime(C.LIVE_CSV)))
    if not rows and not stale:
        return 0                                  # nothing new and scores already up to date
    live = M.load_windows(C.LIVE_CSV)               # (also re-scores if a previous run stopped before scoring)
    df = combined(hist, live)
    live_full = df[df.window.isin(live.window)].copy()
    live_full["sar_backscatter_change"] = df["vv_db"].diff()[df.window.isin(live.window)].values
    live_full[LIVE_COLS].to_csv(C.LIVE_CSV, index=False)
    art = joblib.load(C.MODEL_FILE) if os.path.exists(C.MODEL_FILE) else M.train_production(df)   # frozen 2017-24 model
    M.score(df, art).to_csv(C.RISK_CSV, index=False)
    return len(rows)


def portfolio():
    """One row per lake: latest scored window, risk, and how live scoring compares with the training years."""
    rows = []
    for lake in C.lakes_with_data():
        C.use(lake)
        if not os.path.exists(C.RISK_CSV):
            continue
        r = pd.read_csv(C.RISK_CSV, parse_dates=["window"])
        sc = r.dropna(subset=["risk_pct"])
        if sc.empty:
            continue
        last, live = sc.iloc[-1], sc[sc.period == "live scoring"]
        recent = sc[sc.window > sc.window.max() - pd.Timedelta(days=60)]
        rows.append(dict(
            lake_id=lake, name=C.LAKE_NAME, latest_window=last.window.date(),
            risk_pct=last.risk_pct, risk_level=last.risk_level,
            glof_threat_pct=last.get("glof_threat_pct", last.risk_pct),
            pre_breach_hazard=last.get("pre_breach_hazard", np.nan),
            drainage_score=last.get("drainage_score", np.nan),
            capacity_ratio=last.get("capacity_ratio", np.nan),
            lead_time_window=last.get("lead_time_window", ""),
            area_fc_30d=last.get("area_forecast_30d", np.nan),
            main_driver=last.main_driver, signals_used=int(last.signals_used),
            max_risk_last_60d=recent.risk_pct.max(), live_windows=len(live),
            live_share_above_watch=round((live.risk_pct > C.WATCH_PCT).mean() * 100) if len(live) else np.nan,
            has_area_signal=bool(sc.area_pct.notna().sum() >= 25), fetch_ready=bool(C.BBOX is not None and C.TILE is not None)))
    out = pd.DataFrame(rows).sort_values(["risk_pct", "max_risk_last_60d"], ascending=False)
    out.to_csv(PORTFOLIO_CSV, index=False)
    return out


def predict(lakes_list=None):
    """Generates and displays operational GLOF forecasts and multi-horizon projections."""
    lakes_list = lakes_list or C.lakes_with_data()
    rows = []
    for lake in lakes_list:
        C.use(lake)
        if not os.path.exists(C.RISK_CSV):
            continue
        r = pd.read_csv(C.RISK_CSV, parse_dates=["window"])
        if r.empty:
            continue
        last = r.iloc[-1]
        rows.append(dict(
            lake_id=lake,
            name=C.LAKE_NAME,
            latest_window=last.window.date(),
            threat_pct=last.get("glof_threat_pct", last.risk_pct),
            alert_level=last.get("glof_alert_level", last.risk_level),
            pre_breach_hz=last.get("pre_breach_hazard", 0.0),
            drainage_hz=last.get("drainage_score", 0.0),
            capacity_fill_pct=round(last.get("capacity_ratio", 0.0) * 100, 1) if pd.notna(last.get("capacity_ratio")) else np.nan,
            area_now=last.get("lake_area_km2", np.nan),
            area_fc_30d=last.get("area_forecast_30d", np.nan),
            lead_time_status=last.get("lead_time_window", "Stable"),
            primary_driver=last.get("primary_driver", last.main_driver)
        ))
    df_pred = pd.DataFrame(rows).sort_values("threat_pct", ascending=False).reset_index(drop=True)
    return df_pred


def evaluate():
    """Honest evaluation of detection lead time on documented GLOF events."""
    import glof_predictor as GP
    print("=" * 70)
    print("  GLOF EARLY WARNING SYSTEM: HISTORICAL EVENT LEAD-TIME EVALUATION")
    print("=" * 70)
    
    # 1. South Lhonak Outburst Evaluation
    C.use("south_lhonak")
    if os.path.exists(C.RISK_CSV):
        r = pd.read_csv(C.RISK_CSV, parse_dates=["window"])
        ev_eval = GP.evaluate_lead_time(r, "2023-09-28")
        if ev_eval:
            print("\n[Event Check: South Lhonak Outburst (October 3-4, 2023)]")
            print(f"  Event Window:              {ev_eval['event_window']}")
            print(f"  Earliest Early Warning:    {ev_eval['lead_time_days']} DAYS PRIOR TO EVENT")
            print(f"  Event Day Detected Threat: {ev_eval['event_day_threat']}% ({ev_eval['event_day_alert']})")
            print("  Prior Warning Windows:")
            for days_prior, lvl, score in ev_eval["flagged_windows_prior"]:
                print(f"    - {days_prior} days before: {lvl} ({score}%)")
    
    # 2. Portfolio Workload / False Alarm Rate
    port = portfolio()
    print("\n[Fleet Risk Distribution - Latest Window]")
    if "alert_level" in port.columns:
        print(port["alert_level"].value_counts().to_string())
    elif "risk_level" in port.columns:
        print(port["risk_level"].value_counts().to_string())
    print("=" * 70)


def _lakes(args):
    return C.lakes_with_data() if args.all else [args.lake or C.LAKE]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "update", "portfolio", "report", "predict", "evaluate"])
    ap.add_argument("--send", action="store_true", help="report: really post to ALERT_WEBHOOK (default is a dry run)")
    ap.add_argument("--lake")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "portfolio":
        print(portfolio().to_string(index=False)); sys.exit()
    if a.cmd == "predict":
        print(predict(_lakes(a)).to_string(index=False)); sys.exit()
    if a.cmd == "evaluate":
        evaluate(); sys.exit()
    if a.cmd == "report":
        import triage
        snap = triage.write_report()
        print(triage.notify(snap, os.environ.get("ALERT_WEBHOOK"), dry_run=not a.send))
        print("wrote data/attention.csv and data/attention.md"); sys.exit()
    for lake in _lakes(a):
        C.use(lake)
        try:
            if a.cmd == "train":
                train(); print(f"[{lake}] trained and scored")
            else:
                n = run_update(progress=lambda i, n, s: print(f"[{lake}] [{i}/{n}] fetched window {s.date()}"))
                print(f"[{lake}] {n} new window(s) added." if n else f"[{lake}] nothing new.")
        except Exception as e:                      # one lake failing must not stop the others
            print(f"[{lake}] FAILED: {type(e).__name__}: {e}")
    if a.all:
        print(portfolio().to_string(index=False))
        if a.cmd == "update":                       # fresh data -> fresh triage
            import triage
            print(triage.notify(triage.write_report(), None, dry_run=True))
