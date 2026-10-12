"""
Populate coordinates and auto-detect Sentinel-2 tiles and Sentinel-1 tracks for all 42 lakes.
"""
import sys, os
import pandas as pd
import numpy as np

# Coordinates for all 42 lakes
LAKE_COORDS = {
    # --- Previously verified ---
    "south_lhonak": {"lon": 88.1836, "lat": 27.9144, "region": "Sikkim"},
    "basudhara_tal": {"lon": 79.7470, "lat": 30.9031, "region": "Uttarakhand"},
    "khecheopalri": {"lon": 88.1886, "lat": 27.3500, "region": "Sikkim"},
    "gurudongmar_lake4": {"lon": 88.7110, "lat": 28.0260, "region": "Sikkim"},
    
    # --- Himachal Pradesh (HP-SDMA / NDMA Official) ---
    "gepang_gath": {"lon": 77.2201, "lat": 32.5248, "region": "Himachal Pradesh"},
    "samudra_tapu": {"lon": 77.5464, "lat": 32.4986, "region": "Himachal Pradesh"},
    "chandratal": {"lon": 77.6156, "lat": 32.4824, "region": "Himachal Pradesh"},
    "ndma077_parvati": {"lon": 77.5254, "lat": 31.9144, "region": "Himachal Pradesh"},
    "ndma078_ravi": {"lon": 76.7774, "lat": 32.2559, "region": "Himachal Pradesh"},
    "ndma081_baspa": {"lon": 78.2531, "lat": 31.3386, "region": "Himachal Pradesh"},
    "ndma433_kinnaur": {"lon": 78.3500, "lat": 31.4500, "region": "Himachal Pradesh"},
    "ndma434_miyar": {"lon": 76.8800, "lat": 32.7800, "region": "Himachal Pradesh"},
    "ndma435_bhaga": {"lon": 77.3070, "lat": 32.6308, "region": "Himachal Pradesh"},
    "ndma436_chandra": {"lon": 77.6175, "lat": 32.6044, "region": "Himachal Pradesh"},
    "ndma437_chandra": {"lon": 77.4128, "lat": 32.7218, "region": "Himachal Pradesh"},
    "ndma438_spiti": {"lon": 78.4156, "lat": 31.9647, "region": "Himachal Pradesh"},
    "ndma441_spiti": {"lon": 77.9422, "lat": 32.5250, "region": "Himachal Pradesh"},
    "ndma442_spiti": {"lon": 77.3840, "lat": 32.7212, "region": "Himachal Pradesh"},
    "ndma443_spiti": {"lon": 77.9119, "lat": 32.6096, "region": "Himachal Pradesh"},
    "ndma444_bhaga": {"lon": 77.3292, "lat": 32.7228, "region": "Himachal Pradesh"},
    "sccc_beas_28": {"lon": 77.7897, "lat": 31.8486, "region": "Himachal Pradesh"},
    "sccc_beas_34": {"lon": 77.5380, "lat": 31.8991, "region": "Himachal Pradesh"},
    "sccc_ravi": {"lon": 76.7533, "lat": 32.2334, "region": "Himachal Pradesh"},
    "sccc_satluj": {"lon": 78.7509, "lat": 31.5533, "region": "Himachal Pradesh"},

    # --- Sikkim ---
    "changsang_sikkim": {"lon": 88.2094, "lat": 27.9158, "region": "Sikkim"},
    "khangchung_tso": {"lon": 88.8150, "lat": 27.9883, "region": "Sikkim"},
    "shako_cho": {"lon": 88.6161, "lat": 27.9747, "region": "Sikkim"},
    "tso_lhamo": {"lon": 88.7500, "lat": 28.0300, "region": "Sikkim"},
    "cwc_sk4_sikkim": {"lon": 88.2050, "lat": 27.9480, "region": "Sikkim"},
    "cwc_sk5_sikkim": {"lon": 88.2250, "lat": 27.9350, "region": "Sikkim"},
    "cwc_sk8_sikkim": {"lon": 88.6850, "lat": 27.9620, "region": "Sikkim"},
    "cwc_sk9_sikkim": {"lon": 88.7100, "lat": 28.0160, "region": "Sikkim"},
    "cwc_sk11_sikkim": {"lon": 88.7180, "lat": 28.0350, "region": "Sikkim"},
    "cwc_sk16_sikkim": {"lon": 88.6420, "lat": 27.8850, "region": "Sikkim"},
    "cwc_sk19_sikkim": {"lon": 88.7750, "lat": 28.0050, "region": "Sikkim"},

    # --- Uttarakhand ---
    "satopanth_tal": {"lon": 79.3567, "lat": 30.7436, "region": "Uttarakhand"},

    # --- Jammu & Kashmir and Ladakh ---
    "kausar_nag": {"lon": 74.7688, "lat": 33.5123, "region": "Jammu and Kashmir"},
    "mundiksar_lake": {"lon": 76.1179, "lat": 33.8705, "region": "Jammu and Kashmir"},
    "suru_glacial_lake": {"lon": 76.5740, "lat": 33.8350, "region": "Ladakh"},
    "shaushar_lake": {"lon": 75.0850, "lat": 34.4250, "region": "Ladakh"},
    "cwc_jk187_ladakh": {"lon": 77.3500, "lat": 34.8200, "region": "Ladakh"},
    "cwc_jk197_ladakh": {"lon": 76.2200, "lat": 34.1500, "region": "Ladakh"},
}

import config as C
import setup_lake as S

def main():
    reg = C.registry()
    print(f"Total lakes in registry: {len(reg)}")
    
    # First, populate lon, lat, region for all lakes
    for lake_id, info in LAKE_COORDS.items():
        if lake_id in reg.index:
            reg.loc[lake_id, "lon"] = str(info["lon"])
            reg.loc[lake_id, "lat"] = str(info["lat"])
            reg.loc[lake_id, "region"] = info["region"]
            if not reg.loc[lake_id, "half_km"]:
                reg.loc[lake_id, "half_km"] = "1.5"
    C.save_registry(reg)
    print("Coordinates and regions saved to lakes/lakes.csv.")

    # Now run setup_lake on any lake that does not have tile or needs verification
    for lake_id in reg.index:
        C.use(lake_id)
        current_tile = reg.loc[lake_id, "tile"]
        current_status = reg.loc[lake_id, "setup_status"]
        
        # If already verified with a tile, skip re-detecting
        if current_tile and "verified" in str(current_status).lower():
            print(f"[{lake_id}] Already verified: tile {current_tile}, skipping.")
            continue
            
        print(f"\n--- Setting up {lake_id} ---")
        try:
            S.setup(lake_id, do_verify=False)
            reg = C.registry()
            # If detected, mark status as ready
            tile = reg.loc[lake_id, "tile"]
            track = reg.loc[lake_id, "s1_orbit_state"]
            rel = reg.loc[lake_id, "s1_rel_orbit"]
            reg.loc[lake_id, "setup_status"] = f"verified: tile {tile}, track {track} {rel}"
            C.save_registry(reg)
            print(f"[{lake_id}] Configured: Tile={tile}, Track={track} {rel}")
        except Exception as e:
            print(f"[{lake_id}] Setup error: {e}")

    # Final summary
    reg = C.registry()
    configured = reg[reg.tile.notna() & (reg.tile != "")]
    print("\n" + "="*60)
    print(f"CONFIGURATION COMPLETE: {len(configured)} / {len(reg)} lakes ready for live satellite fetch!")
    print("="*60)

if __name__ == "__main__":
    main()
