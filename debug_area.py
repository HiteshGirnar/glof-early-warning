"""
Find out WHY lake area comes back empty for one window.
    python debug_area.py 2023-01-21
Prints every Sentinel-2 scene in the box and, for each, what the lake detection saw.
"""
import sys
import numpy as np, pandas as pd
import config as C
import fetch as F

if len(sys.argv) > 2:
    C.use(sys.argv[2])          # python debug_area.py 2023-01-21 [lake_id]
start = F.window_start(sys.argv[1])
end = pd.Timestamp(start.year + 1, 1, 1) if start.dayofyear > 350 else start + pd.Timedelta(days=10)
rng = f"{start:%Y-%m-%d}/{(end - pd.Timedelta(days=1)):%Y-%m-%d}"
print(f"window {start.date()} -> {end.date()} | config: TILE={C.TILE} BBOX={C.BBOX} SEED={C.SEED} radius={C.SEED_RADIUS_M} m "
      f"MIN_AREA={C.MIN_AREA}")
items = [i for i in F.catalog().search(collections=["sentinel-2-l2a"], bbox=C.BBOX, datetime=rng).items()
         if F.window_start(F._item_date(i)) == start]
print(f"\n{len(items)} Sentinel-2 scene(s) in the box for this window")
if not items:
    print("-> nothing to measure. Not published yet, or no overpass in this window.")
for it in items:
    p = it.properties
    tile = p.get("s2:mgrs_tile")
    print(f"\n{it.id}\n  date={F._item_date(it)} tile={tile} {'(matches TILE)' if tile == C.TILE else '(!! DIFFERENT from TILE)'} "
          f"cloud_cover(metadata)={p.get('eo:cloud_cover')} baseline={p.get('s2:processing_baseline')}")
    try:
        green = F.read_band(it, "B03", C.BBOX); shp = green.shape
        nir = F.read_band(it, "B08", C.BBOX, out_shape=shp); swir = F.read_band(it, "B11", C.BBOX, out_shape=shp)
        scl = F.read_band(it, "SCL", C.BBOX, out_shape=shp)
        valid = float((green > 0).mean())
        base = float(p.get("s2:processing_baseline", "0") or 0)
        if base >= 4.0:
            green, nir, swir = green - 1000.0, nir - 1000.0, swir - 1000.0
        water = (F._safe_div(green - nir, green + nir) > 0.30) & (F._safe_div(green - swir, green + swir) > 0.30)
        H, W = water.shape
        c = int((C.SEED[0] - C.BBOX[0]) / (C.BBOX[2] - C.BBOX[0]) * W); r = int((C.BBOX[3] - C.SEED[1]) / (C.BBOX[3] - C.BBOX[1]) * H)
        lake = F._lake_component(water, C.BBOX)
        area = lake.sum() * F.PIXEL_KM2
        cloud = float(np.isin(scl, F.SCL_CLOUD).mean())
        print(f"  pixels read: {H}x{W} | valid_frac={valid:.2f} (need >={C.MIN_VALID}) | cloud_frac={cloud:.2f} (need <={C.MAX_CLOUD})")
        print(f"  water pixels in box: {int(water.sum())} (= {water.sum() * F.PIXEL_KM2:.2f} km2) | seed pixel (row {r}, col {c}) is water: "
              f"{bool(water[min(max(r, 0), H - 1), min(max(c, 0), W - 1)])}")
        print(f"  lake picked around seed: {area:.3f} km2 (need >={C.MIN_AREA})")
        verdict = []
        if valid < C.MIN_VALID: verdict.append("DROPPED: partial coverage (box not fully inside this tile -> wrong TILE or scene edge)")
        if cloud > C.MAX_CLOUD: verdict.append("DROPPED: too cloudy")
        if not area >= C.MIN_AREA: verdict.append("DROPPED: lake not found at SEED or smaller than MIN_AREA (check SEED / snow-ice cover)")
        print("  ->", "; ".join(verdict) if verdict else "USABLE")
    except Exception as e:
        print(f"  !! could not read this scene: {type(e).__name__}: {e}")
