"""
Set up (and check) a lake automatically, so you never have to guess a tile or radar track by hand.

    python setup_lake.py south_lhonak            # one lake
    python setup_lake.py --all                   # every lake that has lon/lat (or bbox) in lakes/lakes.csv
    python setup_lake.py my_lake --no-verify     # detect only

You provide per lake in lakes/lakes.csv:  lon, lat  (centre of the lake)   [optional: half_km, bbox, seed_lon, seed_lat]
This script then:
  1. detects the Sentinel-2 tile that fully covers the box,
  2. detects the Sentinel-1 track (most common one, as in your notebook) - then frozen,
  3. derives MIN_AREA from the lake's own history CSV,
  4. VERIFIES by re-fetching a few windows that are already in the lake's CSV and comparing the numbers,
  5. writes the result and a status to lakes/lakes.csv (status starts with 'verified' only if the numbers match).
"""
import argparse, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
import config as C
import fetch as F
import model as M


def detect_tile(sample="2023-07-01/2023-09-30"):
    items = list(F.catalog().search(collections=["sentinel-2-l2a"], bbox=C.BBOX, datetime=sample).items())
    if not items:
        raise RuntimeError("no Sentinel-2 scenes found for this box: check lon/lat (lon first, then lat)")
    by = defaultdict(list)
    for i in items:
        by[i.properties["s2:mgrs_tile"]].append(i)
    best, report = None, {}
    for tile, its in by.items():
        its.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
        valid = float((F.read_band(its[0], "B03", C.BBOX) > 0).mean())          # does this tile cover the whole box?
        report[tile] = (round(valid, 2), len(its))
        key = (valid >= 0.9, len(its))
        if best is None or key > best[0]:
            best = (key, tile)
    return best[1], report


def detect_track():
    items = list(F.catalog().search(collections=["sentinel-1-rtc"], bbox=C.BBOX, datetime="2019-01-01/2026-12-31").items())
    if not items:
        return None, {}
    cnt = Counter((i.properties.get("sat:orbit_state"), i.properties.get("sat:relative_orbit")) for i in items)
    return cnt.most_common(1)[0][0], dict(cnt.most_common(3))


def derive_min_area(hist):
    """Smallest lake-area scene that was KEPT historically (so history is reproduced exactly)."""
    m = hist["lake_area_min_km2"].dropna()
    return round(float(m.min()), 4) if len(m) else None


def pick_windows(hist, n=4):
    """Windows already in the CSV: two with area+radar, one radar-only, one with nothing optical (should stay empty)."""
    h = hist[hist.window < "2026-09-01"]
    full = h[h.lake_area_km2.notna() & h.vv_db.notna()]
    rad = h[h.lake_area_km2.isna() & h.vv_db.notna() & h.thermal_index.notna()]
    pick = []
    if len(full):
        pick += list(full.window.iloc[np.linspace(0, len(full) - 1, 3).astype(int)[:2]])
    if len(rad):
        pick.append(rad.window.iloc[len(rad) // 2])
    return pick[:n]


def compare(got, ref):
    """Return list of problems (empty list = this window matches)."""
    bad = []
    a, b = got.get("lake_area_km2"), ref["lake_area_km2"]
    if pd.notna(b) and not (pd.notna(a) and abs(a - b) <= max(0.003, 0.03 * b)):
        bad.append(f"area fetched {a} vs csv {b}")
    if pd.isna(b) and pd.notna(a):
        bad.append(f"area fetched {a} but csv is empty")
    if pd.notna(ref["vv_db"]) and not (pd.notna(got.get("vv_db")) and abs(got["vv_db"] - ref["vv_db"]) <= 0.02):
        bad.append(f"vv fetched {got.get('vv_db')} vs csv {ref['vv_db']}")
    if pd.notna(ref["thermal_index"]) and not (pd.notna(got.get("thermal_index")) and abs(got["thermal_index"] - ref["thermal_index"]) <= 0.5):
        bad.append(f"thermal fetched {got.get('thermal_index')} vs csv {ref['thermal_index']}")
    return bad


def verify(hist, fetch=None):
    fetch = fetch or F.fetch_window
    results = []
    for s in pick_windows(hist):
        got = fetch(s, pd.Timestamp(s.year + 1, 1, 1) if s.dayofyear > 350 else s + pd.Timedelta(days=10))
        ref = hist.set_index("window").loc[s]
        results.append((s.date(), compare(got, ref)))
    return results


def setup(lake, do_verify=True):
    C.use(lake)
    reg = C.registry()
    if C.BBOX is None:
        print(f"[{lake}] skipped: no lon/lat or bbox in lakes/lakes.csv"); return
    print(f"[{lake}] box {C.BBOX} seed {C.SEED}")
    hist = M.load_windows(C.HISTORY_CSV)
    tile, rep = detect_tile()
    print(f"  tiles (valid coverage, scenes): {rep} -> {tile}")
    track, cnt = detect_track()
    print(f"  radar tracks: {cnt} -> {track}")
    min_area = derive_min_area(hist)
    print(f"  MIN_AREA from history: {min_area}")
    reg.loc[lake, "tile"] = tile
    if track:
        reg.loc[lake, "s1_orbit_state"], reg.loc[lake, "s1_rel_orbit"] = track[0], str(track[1])
    if min_area is not None:
        reg.loc[lake, "min_area"] = str(min_area)
    reg.loc[lake, "setup_status"] = "detected, not verified"
    C.save_registry(reg); C.use(lake)
    if do_verify:
        res = verify(hist)
        for d, bad in res:
            print(f"  verify {d}: {'OK' if not bad else 'MISMATCH ' + '; '.join(bad)}")
        ok = bool(res) and all(not b for _, b in res)
        reg = C.registry()
        reg.loc[lake, "setup_status"] = ("verified: fetch matches CSV on " + ", ".join(str(d) for d, _ in res)) if ok else \
            "NOT verified: " + "; ".join(f"{d}: {', '.join(b)}" for d, b in res if b)[:200]
        C.save_registry(reg)
        print(f"  => {reg.loc[lake, 'setup_status']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lake", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    lakes = [l for l in C.lakes_with_data() if (C.registry().loc[l, "lon"] or C.registry().loc[l, "bbox"])] if a.all else [a.lake]
    for l in lakes:
        try:
            setup(l, not a.no_verify)
        except Exception as e:
            print(f"[{l}] FAILED: {type(e).__name__}: {e}")
