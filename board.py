"""Builds the data + HTML for the Operations Board (fleet triage view)."""
import json, os
import numpy as np, pandas as pd
import config as C
import triage as T


def _arr(s, nd=1):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def build_payload(default_lake=None):
    frames = T.load_fleet()
    if not frames:
        return None
    last = T.last_window_with_data(frames)
    R = pd.DataFrame({l: f.risk_pct for l, f in frames.items()})
    R = R[R.index <= last]
    code, med, E, share, regional = T.tiers(R)
    reg = C.registry()
    lakes = []
    for lake, f in frames.items():
        f = f.reindex(R.index)
        why = {}
        for i in np.flatnonzero((code[lake].to_numpy() >= 2) | (np.arange(len(R)) == len(R) - 1)):
            if code[lake].iloc[i] > 0:
                why[int(i)] = T.reasons(f, int(i), med.iloc[i], E[lake].iloc[i])
        row = reg.loc[lake]
        lat = row["seed_lat"] or row["lat"]; lon = row["seed_lon"] or row["lon"]
        try:
            lat, lon = float(lat), float(lon)
            link = f"https://apps.sentinel-hub.com/eo-browser/?zoom=15&lat={lat}&lng={lon}&themeId=DEFAULT-THEME&datasetId=S2L2A"
        except (TypeError, ValueError):
            lat = lon = None; link = ""
        ev = [x.strip() for x in row["event_windows"].split(";") if x.strip()]
        lakes.append(dict(
            id=lake, name=row["name"] or lake, region=row["region"] or "Other", lat=lat, lon=lon, link=link,
            small=T.small_lake(f), has_area=bool(f.area_pct.notna().sum() >= 25), status=row["setup_status"],
            events=[dict(i=R.index.get_loc(pd.Timestamp(w)), label=row["event_label"] or w) for w in ev if pd.Timestamp(w) in R.index],
            risk=_arr(R[lake]), excess=_arr(E[lake]), code=[int(v) for v in code[lake]],
            series={
                "lake area km²": _arr(f.lake_area_km2, 3),
                "radar VV dB": _arr(f.vv_db, 2),
                "thermal index": _arr(f.thermal_index, 1),
                "GLOF Threat %": _arr(f["glof_threat_pct"] if "glof_threat_pct" in f.columns else R[lake], 1),
                "Pre-Breach Hazard %": _arr(f["pre_breach_hazard"] if "pre_breach_hazard" in f.columns else [np.nan]*len(f), 1),
                "Dam Capacity %": _arr(f["capacity_ratio"] * 100 if "capacity_ratio" in f.columns else [np.nan]*len(f), 1)
            },
            why=why))
    bt_path = os.path.join(C.BASE, "data", "fleet_backtest.csv")
    bt = None
    if os.path.exists(bt_path):
        Rb = pd.read_csv(bt_path, index_col=0, parse_dates=True)
        bt = T.backtest_summary(Rb)
    today = pd.Timestamp.today().normalize()
    end = R.index[-1] + pd.Timedelta(days=10)
    return dict(
        default=default_lake or lakes[0]["id"], dates=[d.strftime("%Y-%m-%d") for d in R.index], lakes=lakes,
        med=_arr(med), share=[None if pd.isna(v) else round(float(v), 3) for v in share], regional=[bool(v) for v in regional],
        rules=dict(alert_risk=T.ALERT_RISK, alert_excess=T.ALERT_EXCESS, watch_risk=T.WATCH_RISK, watch_excess=T.WATCH_EXCESS,
                   regional=T.REGIONAL_SHARE, small=T.SMALL_LAKE_KM2, watch_pct=C.WATCH_PCT),
        train_end=C.TRAIN_END, today=today.strftime("%Y-%m-%d"), record_end=end.strftime("%Y-%m-%d"), behind_days=int((today - end).days),
        workload=T.workload(frames), events=T.event_checks(frames), backtest=bt)


def render_html(payload):
    tpl = open(os.path.join(C.BASE, "board_template.html"), encoding="utf-8").read()
    return tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), default=str).replace("</", "<\\/"))


if __name__ == "__main__":      # python board.py [--lake id] [--out data/board.html]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default=C.DEFAULT_LAKE)
    ap.add_argument("--out", default=os.path.join(C.BASE, "data", "board.html"))
    a = ap.parse_args()
    html = render_html(build_payload(a.lake))
    open(a.out, "w", encoding="utf-8").write(html)
    print("wrote", a.out, f"({len(html) // 1024} KB)")
