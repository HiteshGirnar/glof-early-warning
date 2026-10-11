"""Builds the data for the 'Watch Console' view (all lakes, one self-contained HTML page)."""
import json, os
import numpy as np, pandas as pd
import config as C

SIGNALS = [("lake_area_km2", "lake area km²"), ("lake_area_min_km2", "lake area min km²"), ("vv_db", "radar VV dB"),
           ("sar_backscatter_change", "radar change"), ("thermal_index", "thermal index"),
           ("velocity_trend", "velocity trend (yearly)"), ("velocity_accel", "velocity accel (yearly)")]


def _arr(s, nd=3):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def _read_hist(path):
    h = pd.read_csv(path)
    h = h.rename(columns={h.columns[0]: "window"})
    h["window"] = pd.to_datetime(h["window"])
    return h


def _test_text(lake_name):
    """Honest backtest summary for lakes with a recorded event (None otherwise)."""
    if not C.KNOWN_EVENT_WINDOWS or not os.path.exists(C.BACKTEST_CSV):
        return None
    bt = pd.read_csv(C.BACKTEST_CSV, parse_dates=["window"])
    ev = pd.Timestamp(C.KNOWN_EVENT_WINDOWS[0])
    row = bt[bt.window == ev]
    if row.empty or pd.isna(row.iloc[0].risk_pct):
        return None
    row = row.iloc[0]
    before = bt[bt.window < ev].dropna(subset=["risk_pct"])
    last3, run27 = before.tail(3), before.tail(27)
    held = bt[bt.period == "held-out normal"].dropna(subset=["risk_pct"])
    fa = float((held.risk_pct > C.WATCH_PCT).mean() * 100) if len(held) else float("nan")
    label = C.EVENT_LABEL or f"event {ev.date()}"
    w = C.WATCH_PCT
    hit27 = int((run27.risk_pct >= w).sum())
    chance27 = len(run27) * fa / 100
    if len(last3) and (last3.risk_pct >= w).any():
        first = last3[last3.risk_pct >= w].iloc[0]
        head, neg = "A WATCH PRECEDED THE EVENT, BUT WATCH IS COMMON", False
        lead = f"Watch ({w}%) was reached on {first.window:%d %b %Y}, within the 30 days before the outburst window."
    else:
        mx = last3.loc[last3.risk_pct.idxmax()] if len(last3) else None
        head, neg = "DETECTED, NOT PREDICTED", True
        lead = (f"In the 3 windows (30 days) just before it the maximum was {mx.risk_pct:.0f}% ({mx.window:%d %b %Y}), "
                f"so Watch ({w}%) was not reached: no advance warning.") if mx is not None else "There was no scored window before it."
    body = (f"With the model trained on 2017-2021 only, the outburst window ({label}) scored {row.risk_pct:.0f}% "
            f"({row.risk_level}), driven by {row.main_driver}. {lead} Over the 27-window run-up Watch was reached in {hit27} windows; "
            f"chance alone gives about {chance27:.0f}, because normal unseen windows exceed Watch {fa:.0f}% of the time. "
            "One event is an anecdote, not validation.")
    return {"head": head, "body": body, "negative": neg, "event_risk": float(row.risk_pct)}


def _layout(lakes):
    have = [l for l in lakes if l["lon"] is not None and l["lat"] is not None]
    if len(have) >= 0.8 * len(lakes):
        lo, la = np.array([l["lon"] for l in have]), np.array([l["lat"] for l in have])
        for l in lakes:
            if l["lon"] is None:
                l["x"] = l["y"] = 0.5
            else:
                l["x"] = float(0.08 + 0.84 * (l["lon"] - lo.min()) / max(np.ptp(lo), 1e-9))
                l["y"] = float(0.08 + 0.84 * (1 - (l["lat"] - la.min()) / max(np.ptp(la), 1e-9)))
        return "geographic"
    groups = {}
    for l in lakes:
        groups.setdefault(l["region"] or "Other", []).append(l)
    order = sorted(groups, key=lambda g: (-len(groups[g]) if g != "Other" else 0, g))
    cols = 3                                   # clusters laid out left to right, 3 per row
    rows_n = int(np.ceil(len(order) / cols))
    for gi, g in enumerate(order):
        gx, gy = gi % cols, gi // cols
        members = groups[g]
        k = int(np.ceil(np.sqrt(len(members))))
        for mi, l in enumerate(members):
            cx = (gx + 0.5) / cols + (((mi % k) - (k - 1) / 2) * 0.045)
            cy = (gy + 0.5) / rows_n + (((mi // k) - (k - 1) / 2) * 0.075)
            l["x"], l["y"] = float(np.clip(cx, 0.05, 0.95)), float(np.clip(cy, 0.08, 0.92))
    return "schematic"


def build_payload(default_lake=None):
    reg = C.registry()
    raw = []
    for lake in C.lakes_with_data():
        C.use(lake)
        if not os.path.exists(C.RISK_CSV):
            continue
        r = pd.read_csv(C.RISK_CSV, parse_dates=["window"])
        h = _read_hist(C.HISTORY_CSV)
        vel = h[["window"] + [c for c in ("velocity_trend", "velocity_accel") if c in h.columns]]
        r = r.merge(vel, on="window", how="left")
        raw.append((lake, r, reg.loc[lake], dict(
            name=C.LAKE_NAME, event=(C.KNOWN_EVENT_WINDOWS[0] if C.KNOWN_EVENT_WINDOWS else None), event_label=C.EVENT_LABEL,
            seed=C.SEED, tile=C.TILE, track=C.S1_TRACK, min_area=C.MIN_AREA, test=_test_text(C.LAKE_NAME))))
    if not raw:
        return None
    dates = sorted(set().union(*[set(r.window) for _, r, _, _ in raw]))
    any_assessed = pd.DataFrame({l: r.set_index("window").risk_pct.reindex(dates) for l, r, _, _ in raw}).notna().any(axis=1)
    last = max(i for i, v in enumerate(any_assessed.tolist()) if v)
    dates = dates[:last + 1]
    lakes = []
    for lake, r, row, meta in raw:
        r = r.set_index("window").reindex(dates)
        risk = r.risk_pct
        opt = r.lake_area_km2.notna()
        mons = r.index.month.isin([6, 7, 8, 9])
        lakes.append(dict(
            id=lake, name=meta["name"], region=row["region"], lon=meta["seed"][0] if meta["seed"] else None,
            lat=meta["seed"][1] if meta["seed"] else None,
            event=meta["event"], event_label=meta["event_label"], test=meta["test"],
            has_area=bool(r.area_pct.notna().sum() >= 25),
            tile=meta["tile"], track=("%s %s" % meta["track"]) if meta["track"] else None, min_area=meta["min_area"],
            status=row["setup_status"],
            risk=_arr(risk, 1), driver=[None if pd.isna(v) else v for v in r.main_driver],
            signals_used=[None if pd.isna(v) else int(v) for v in r.signals_used], obs=_arr(r.observability, 2),
            pct={"area": _arr(r.area_pct, 0), "radar": _arr(r.radar_pct, 0), "thermal": _arr(r.thermal_pct, 0)},
            series={k: _arr(r[k], 3) for k, _ in SIGNALS if k in r.columns},
            stats=dict(mean=None if risk.notna().sum() == 0 else round(float(risk.mean()), 1),
                       p95=None if risk.notna().sum() == 0 else round(float(risk.quantile(0.95)), 1),
                       abstain=round(float(risk.isna().mean() * 100), 1),
                       mons_all=round(float(risk[mons].notna().mean() * 100), 1) if mons.any() else None,
                       mons_opt=round(float(opt[mons].mean() * 100), 1) if mons.any() else None)))
    layout = _layout(lakes)
    allr = pd.DataFrame({l["id"]: l["risk"] for l in lakes}, dtype=float)
    allopt = pd.DataFrame({l["id"]: l["series"].get("lake_area_km2", [None] * len(dates)) for l in lakes}, dtype=float)
    mons = pd.DatetimeIndex(dates).month.isin([6, 7, 8, 9])
    today = pd.Timestamp.today().normalize()
    end_of_last = pd.Timestamp(dates[-1]) + pd.Timedelta(days=10)
    return dict(
        default=default_lake or lakes[0]["id"], dates=[d.strftime("%Y-%m-%d") for d in dates], lakes=lakes, layout=layout,
        watch=C.WATCH_PCT, alert=C.ALERT_PCT, train_end=C.TRAIN_END, today=today.strftime("%Y-%m-%d"),
        record_end=end_of_last.strftime("%Y-%m-%d"), behind_days=int((today - end_of_last).days),
        n_windows=len(dates), n_lake_windows=int(allr.size),
        mons_all=round(float(allr[mons].notna().to_numpy().mean() * 100), 1),
        mons_opt=round(float(allopt[mons].notna().to_numpy().mean() * 100), 1),
        abstain_all=round(float(allr.isna().to_numpy().mean() * 100), 1))


def render_html(payload):
    tpl = open(os.path.join(C.BASE, "console_template.html"), encoding="utf-8").read()
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return tpl.replace("/*__DATA__*/null", data)


if __name__ == "__main__":      # python console.py [--lake south_lhonak] [--out data/console.html]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default=C.DEFAULT_LAKE)
    ap.add_argument("--out", default=os.path.join(C.BASE, "data", "console.html"))
    a = ap.parse_args()
    html = render_html(build_payload(a.lake))
    open(a.out, "w", encoding="utf-8").write(html)
    print("wrote", a.out, f"({len(html) // 1024} KB). Open it in any browser; it works offline.")
