"""
Fetch ONE 10-day window of satellite data (Planetary Computer) and turn it into one CSV row.
Logic is ported from your Colab notebook: same masks, filters, aggregation, observability formula.
Heavy libraries are imported lazily, so the app works without them until you press "Update".
"""
import os, warnings
import numpy as np, pandas as pd
import config as C

SCL_CLOUD = [3, 8, 9, 10]
PIXEL_KM2 = 0.0001
_cat = None


def _libs():
    global rasterio, pystac_client, pc, ndimage, Window, from_bounds, transform_bounds
    import rasterio, pystac_client, planetary_computer as pc
    from scipy import ndimage
    from rasterio.windows import Window, from_bounds
    from rasterio.warp import transform_bounds
    if C.PC_KEY:
        pc.settings.set_subscription_key(C.PC_KEY)
    for k, v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
                 "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "2", "VSI_CACHE": "TRUE"}.items():
        os.environ[k] = v


def catalog():
    global _cat
    if _cat is None:
        _libs()
        _cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    return _cat


# ---------------- measuring (from the notebook) ----------------
def read_band(item, band, bbox, out_shape=None):
    href = pc.sign(item.assets[band].href)
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        w = from_bounds(*b, transform=src.transform)
        win = Window(int(round(w.col_off)), int(round(w.row_off)),
                     max(1, int(round(w.width))), max(1, int(round(w.height))))
        inside = (win.col_off >= 0 and win.row_off >= 0 and
                  win.col_off + win.width <= src.width and win.row_off + win.height <= src.height)
        arr = src.read(1, window=win, out_shape=out_shape, boundless=not inside,
                       fill_value=0 if not inside else None)
    return arr.astype("float32")


def _safe_div(a, b):
    out = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a, b, out=out, where=b > 0)
    return out


def _nanmean(a):
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else np.nan


def _to_db(p):
    db = np.full(p.shape, np.nan, dtype="float32")
    np.log10(p, out=db, where=p > 0)
    return db * 10


def _item_date(item):
    if item.datetime is not None:
        return item.datetime.date()
    p = item.properties
    return pd.Timestamp(p.get("start_datetime") or p.get("end_datetime")).date()


def _lake_component(mask, bbox):
    """Keep only the connected water body touching the seed (same as the notebook's lake_component).
    A small closing (2 pixels) first bridges thin cracks/ice/shadow so one lake is not split into pieces."""
    H, W = mask.shape
    c = (C.SEED[0] - bbox[0]) / (bbox[2] - bbox[0]) * W
    r = (bbox[3] - C.SEED[1]) / (bbox[3] - bbox[1]) * H
    closed = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    lab, n = ndimage.label(closed, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    yy, xx = np.ogrid[:H, :W]
    disk = (yy - r) ** 2 + (xx - c) ** 2 <= (C.SEED_RADIUS_M / 10.0) ** 2
    ids = lab[disk & (lab > 0)]
    if ids.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return (lab == np.bincount(ids).argmax()) & mask      # AND with the ORIGINAL mask: closing adds no fake water


def measure_s2(item, bbox):
    green = read_band(item, "B03", bbox); shape = green.shape
    nir = read_band(item, "B08", bbox, out_shape=shape)
    swir = read_band(item, "B11", bbox, out_shape=shape)
    scl = read_band(item, "SCL", bbox, out_shape=shape)
    valid = float((green > 0).mean())
    try:
        base = float(item.properties.get("s2:processing_baseline", "0"))
    except (TypeError, ValueError):
        base = 0.0
    if base >= 4.0:                                    # reflectance offset after 25 Jan 2022
        green, nir, swir = green - 1000.0, nir - 1000.0, swir - 1000.0
    water = _lake_component((_safe_div(green - nir, green + nir) > 0.30) &
                            (_safe_div(green - swir, green + swir) > 0.30), bbox)
    area = float(water.sum()) * PIXEL_KM2
    return {"date": str(_item_date(item)), "area_km2": area if area > 0 else np.nan,
            "valid_frac": valid, "cloud_frac": float(np.isin(scl, SCL_CLOUD).mean())}


def measure_s1(item, bbox):
    return {"date": str(_item_date(item)), "vv_db": _nanmean(_to_db(read_band(item, "vv", bbox)))}


def measure_lst(item, bbox):
    raw = read_band(item, "LST_Day_1km", bbox)
    raw[raw == 0] = np.nan
    c = raw * 0.02 - 273.15
    return {"date": str(_item_date(item)), "lst_c": _nanmean(c), "valid_frac": float(np.isfinite(c).mean())}


# ---------------- aggregation (pure function, same as the notebook) ----------------
def aggregate_window(s2, s1, lst):
    out = dict(lake_area_km2=np.nan, lake_area_min_km2=np.nan, vv_db=np.nan, thermal_index=np.nan)
    n_s2 = n_s1 = n_lst = 0
    if s2:
        d = pd.DataFrame(s2)
        d = d[(d.cloud_frac <= C.MAX_CLOUD) & (d.valid_frac >= C.MIN_VALID) & (d.area_km2 >= C.MIN_AREA)]
        n_s2 = len(d)
        if n_s2:
            out["lake_area_km2"], out["lake_area_min_km2"] = d.area_km2.median(), d.area_km2.min()
    if s1:
        n_s1 = len(s1)
        out["vv_db"] = pd.DataFrame(s1).vv_db.median()
    if lst:
        d = pd.DataFrame(lst)
        d = d[d.valid_frac >= C.MIN_LST_VALID]
        n_lst = len(d)
        if n_lst:
            out["thermal_index"] = d.lst_c.clip(lower=0).mean() * 10
    # same weights as the notebook: 0.40 optical + 0.30 radar + 0.15 thermal + 0.15 glacier velocity.
    # Velocity is a yearly value that is not fetched live, so live rows can reach at most 0.85.
    out["observability"] = 0.40 * (n_s2 > 0) + 0.30 * (n_s1 > 0) + 0.15 * (n_lst > 0)
    return out


def window_start(d):
    d = pd.Timestamp(d)
    return pd.Timestamp(year=d.year, month=1, day=1) + pd.Timedelta(days=10 * min((d.dayofyear - 1) // 10, 35))


def fetch_window(start, end):
    """Everything with date in [start, end) -> one row of values (NaN where nothing usable)."""
    if C.BBOX is None or C.SEED is None or C.TILE is None:
        raise RuntimeError(f"{C.LAKE}: position or Sentinel-2 tile missing. Add lon/lat in lakes/lakes.csv, then run "
                           f"`python setup_lake.py {C.LAKE}`.")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rng = f"{start:%Y-%m-%d}/{(end - pd.Timedelta(days=1)):%Y-%m-%d}"
    inwin = lambda it: window_start(_item_date(it)) == start

    def measure(fn, items):
        rows = []
        for it in items:
            try:
                rows.append(fn(it, C.BBOX))
            except Exception as e:
                warnings.warn(f"skipped {it.id}: {type(e).__name__}: {e}")
        return rows

    cat = catalog()
    s2_all = [i for i in cat.search(collections=["sentinel-2-l2a"], bbox=C.BBOX, datetime=rng).items() if inwin(i)]
    s2 = [i for i in s2_all if i.properties["s2:mgrs_tile"] == C.TILE]
    if s2_all and not s2:
        warnings.warn(f"Sentinel-2: {len(s2_all)} scene(s) found in the box but NONE on tile {C.TILE}; "
                      f"tiles seen: {sorted({i.properties['s2:mgrs_tile'] for i in s2_all})}. Check TILE in config.py.")
    elif not s2_all:
        warnings.warn(f"Sentinel-2: no scenes at all for {rng} (not published yet, or none in this window).")
    s1 = [i for i in cat.search(collections=["sentinel-1-rtc"], bbox=C.BBOX, datetime=rng).items()
          if inwin(i) and (i.properties.get("sat:orbit_state"), i.properties.get("sat:relative_orbit")) == tuple(C.S1_TRACK)]
    lst = [i for i in cat.search(collections=["modis-11A1-061"], bbox=C.BBOX, datetime=rng).items() if inwin(i)]
    s2_rows = measure(measure_s2, s2)
    if s2_rows:
        d = pd.DataFrame(s2_rows)
        bad = {"cloudy": int((d.cloud_frac > C.MAX_CLOUD).sum()), "partial coverage": int((d.valid_frac < C.MIN_VALID).sum()),
               "lake not found / too small": int((~(d.area_km2 >= C.MIN_AREA)).sum())}
        ok = (d.cloud_frac <= C.MAX_CLOUD) & (d.valid_frac >= C.MIN_VALID) & (d.area_km2 >= C.MIN_AREA)
        if not ok.any():
            warnings.warn(f"Sentinel-2: {len(d)} scene(s) measured but none usable -> area stays empty. Reasons (can overlap): {bad}. "
                          "Run python debug_area.py <date> for details.")
    return aggregate_window(s2_rows, measure(measure_s1, s1), measure(measure_lst, lst))


if __name__ == "__main__":     # python fetch.py 2024-09-17 [lake_id]  -> compare a re-fetch with your CSV row
    import sys
    if len(sys.argv) > 2:
        C.use(sys.argv[2])
    s = window_start(sys.argv[1])
    e = pd.Timestamp(year=s.year + 1, month=1, day=1) if s.dayofyear > 350 else s + pd.Timedelta(days=10)
    got = fetch_window(s, e)
    ref = pd.read_csv(C.HISTORY_CSV, index_col=0, parse_dates=True).loc[s]
    print(pd.DataFrame({"fetched_now": pd.Series(got), "in_csv": ref}))
