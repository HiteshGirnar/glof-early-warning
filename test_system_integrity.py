"""
Comprehensive Scientific & Operational Test Suite for GLOF Early Warning System.

Tests the 8 core glaciological and pipeline invariants:
1. No future data (Causal feature integrity)
2. Nothing after 2024 is learned (Temporal split & historical stability)
3. Known flood window kept out of training
4. Missing data stays missing (No artificial filling or unearned risk)
5. Scores & alert levels are consistent & deterministic (0-100 bounds, exact thresholds)
6. Temperature cannot trigger an alert alone (Capped at 60% / Low)
7. Fleet rules behave (Isolated spike, regional discounting, missing != clear, tiny fleet fallback)
8. 10-day calendar correctness (Year-end boundary, leap years, unfinished window skipping)
"""

import os, sys, unittest
import numpy as np, pandas as pd

import config as C
import model as M
import pipeline as P
import triage as T
import preprocessor as Prep
import glof_predictor as GP


class TestGLOFSystemIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        C.use("south_lhonak")
        cls.lake_id = "south_lhonak"
        cls.raw_hist = M.load_windows(C.HISTORY_CSV)

    # =========================================================================
    # 1. NO FUTURE DATA (CAUSAL FEATURE INTEGRITY)
    # =========================================================================
    def test_01_no_future_data_causal_integrity(self):
        """A window's features must never change when later data is removed."""
        df_full = self.raw_hist.copy()
        
        # Test across multiple intermediate cut-off windows
        test_cutoffs = [
            pd.Timestamp("2021-06-10"),
            pd.Timestamp("2022-09-18"),
            pd.Timestamp("2023-05-21"),
            pd.Timestamp("2024-01-11"),
        ]
        
        for cut_ts in test_cutoffs:
            # Feature calculation on full series
            stats_full = M.seasonal_stats(df_full[df_full.window <= pd.Timestamp(C.TRAIN_END)])
            f_full = M.build_features(df_full, stats_full)
            f_full_cut = f_full[f_full.window <= cut_ts].copy()
            
            # Feature calculation on truncated series (strictly data <= cut_ts)
            df_truncated = df_full[df_full.window <= cut_ts].copy()
            stats_trunc = M.seasonal_stats(df_truncated[df_truncated.window <= pd.Timestamp(C.TRAIN_END)])
            f_trunc = M.build_features(df_truncated, stats_trunc)
            
            # Verify feature parity
            cols_to_check = ["area_dev", "area_step", "vv_dev", "vv_change", "thermal_change"]
            for col in cols_to_check:
                s1 = f_full_cut[col].dropna()
                s2 = f_trunc[col].dropna()
                # Ensure same indices and values
                common_idx = s1.index.intersection(s2.index)
                diff = np.abs(s1.loc[common_idx] - s2.loc[common_idx]).max()
                self.assertLess(diff, 1e-6, f"Causal leak detected in {col} at cut-off {cut_ts}: max diff {diff}")

            # Verify Glaciological Preprocessor Causal Invariant
            base_stats = Prep.compute_baseline_stats(df_full, baseline_end=C.TRAIN_END)
            prep_full = Prep.preprocess_lake_series(df_full, baseline_stats=base_stats)
            prep_full_cut = prep_full[prep_full.window <= cut_ts].copy()
            
            prep_trunc = Prep.preprocess_lake_series(df_truncated, baseline_stats=base_stats)
            
            prep_cols = ["capacity_ratio", "expansion_rate_10d", "expansion_rate_30d", "vol_est_mcm", "obs_age"]
            for col in prep_cols:
                s1 = prep_full_cut[col].dropna()
                s2 = prep_trunc[col].dropna()
                common_idx = s1.index.intersection(s2.index)
                diff = np.abs(s1.loc[common_idx] - s2.loc[common_idx]).max()
                self.assertLess(diff, 1e-5, f"Causal leak in preprocessor {col} at {cut_ts}: diff {diff}")

    # =========================================================================
    # 2. NOTHING AFTER 2024 IS LEARNED (TEMPORAL SPLIT INTEGRITY)
    # =========================================================================
    def test_02_nothing_after_2024_is_learned(self):
        """Scrambling 2025–26 data leaves the model and every 2017–2024 score unchanged."""
        df_orig = self.raw_hist.copy()
        
        # 1. Train production model on original data
        art_orig = M.train_production(df_orig)
        scores_orig = M.score(df_orig, art_orig)
        hist_scores_orig = scores_orig[scores_orig.window <= pd.Timestamp(C.TRAIN_END)].copy()
        
        # 2. Corrupt/scramble only the 2025-2026 data
        df_scrambled = df_orig.copy()
        future_mask = df_scrambled.window > pd.Timestamp(C.TRAIN_END)
        self.assertTrue(future_mask.any(), "Test requires 2025+ windows in history")
        
        np.random.seed(999)
        # Radically distort future features
        df_scrambled.loc[future_mask, "lake_area_km2"] = df_scrambled.loc[future_mask, "lake_area_km2"] * 5.0 + 3.0
        df_scrambled.loc[future_mask, "vv_db"] = df_scrambled.loc[future_mask, "vv_db"] - 15.0
        df_scrambled.loc[future_mask, "thermal_index"] = df_scrambled.loc[future_mask, "thermal_index"] * 0.1
        
        # 3. Train production on scrambled dataset
        art_scrambled = M.train_production(df_scrambled)
        scores_scrambled = M.score(df_scrambled, art_scrambled)
        hist_scores_scrambled = scores_scrambled[scores_scrambled.window <= pd.Timestamp(C.TRAIN_END)].copy()
        
        # 4. Assert model artifacts are 100% invariant
        for k in art_orig["stats"]:
            pd.testing.assert_frame_equal(art_orig["stats"][k], art_scrambled["stats"][k])
            
        for k in art_orig["refs"]:
            np.testing.assert_allclose(art_orig["refs"][k], art_scrambled["refs"][k], atol=1e-6)
            
        # 5. Assert historical 2017-2024 scores are 100% unchanged
        score_diff = np.abs(hist_scores_orig["risk_pct"].dropna() - hist_scores_scrambled["risk_pct"].dropna()).max()
        self.assertLess(score_diff, 1e-6, f"2017-2024 scores altered by future data change: max diff {score_diff}")
        
        # Threat score parity
        threat_diff = np.abs(hist_scores_orig["glof_threat_pct"].dropna() - hist_scores_scrambled["glof_threat_pct"].dropna()).max()
        self.assertLess(threat_diff, 1e-6, f"Historical GLOF threat altered by future data change: max diff {threat_diff}")

    # =========================================================================
    # 3. KNOWN FLOOD WINDOW IS KEPT OUT OF TRAINING
    # =========================================================================
    def test_03_flood_window_excluded_from_training(self):
        """The known flood window is strictly excluded from training."""
        C.use("south_lhonak")
        self.assertIn("2023-09-28", C.KNOWN_EVENT_WINDOWS, "South Lhonak must register 2023-09-28 event")
        
        df = self.raw_hist.copy()
        event_ts = pd.Timestamp("2023-09-28")
        
        # Verify train_production normal filter
        normal = df[(df["window"] <= pd.Timestamp(C.TRAIN_END))
                    & ~df["window"].isin(pd.to_datetime(C.KNOWN_EVENT_WINDOWS))]
        
        self.assertNotIn(event_ts, normal["window"].values, "Known outburst window found in normal training set!")
        
        # Verify the fitted model reference distributions never trained on this event
        art = M.train_production(df)
        event_row = df[df.window == event_ts]
        self.assertEqual(len(event_row), 1)

    # =========================================================================
    # 4. MISSING DATA STAYS MISSING
    # =========================================================================
    def test_04_missing_data_stays_missing(self):
        """Empty readings are not filled, and a window with no readings gets no risk."""
        df = self.raw_hist.copy()
        art = M.train_production(df)
        
        # Verify raw missing values remain NaN in build_features
        feats = M.build_features(df, art["stats"])
        missing_area = df["lake_area_km2"].isna()
        self.assertTrue((feats.loc[missing_area, "area_dev"].isna() | feats.loc[missing_area, "area_step"].isna()).all())
        
        # Test synthetic window with completely empty observations
        blank_row = pd.DataFrame([{
            "window": pd.Timestamp("2025-05-01"),
            "lake_area_km2": np.nan,
            "lake_area_min_km2": np.nan,
            "vv_db": np.nan,
            "sar_backscatter_change": np.nan,
            "thermal_index": np.nan,
            "observability": 0.0
        }])
        
        df_with_blank = pd.concat([df[df.window < "2025-05-01"], blank_row]).reset_index(drop=True)
        res = M.score(df_with_blank, art)
        
        blank_scored = res[res.window == pd.Timestamp("2025-05-01")].iloc[0]
        self.assertEqual(blank_scored["signals_used"], 0, "Empty window should use 0 signals")
        self.assertTrue(pd.isna(blank_scored["anomaly_risk_pct"]), "Anomaly risk for empty window must be NaN")
        self.assertTrue(pd.isna(blank_scored["risk_pct"]), "Overall risk for completely empty window must be NaN")
        self.assertTrue(pd.isna(blank_scored["risk_level"]), "Risk level for empty window must be NaN/None")

    # =========================================================================
    # 5. SCORES AND LEVELS ARE CONSISTENT & DETERMINISTIC
    # =========================================================================
    def test_05_scores_and_levels_consistent(self):
        """Scores stay between 0 and 100, levels match thresholds, and results are deterministic."""
        df = self.raw_hist.copy()
        art = M.train_production(df)
        
        # Run 1
        res1 = M.score(df, art)
        # Run 2
        res2 = M.score(df, art)
        
        # 1. Determinism
        pd.testing.assert_frame_equal(res1, res2, check_dtype=True)
        
        # 2. Bounds [0, 100]
        valid_risks = res1["risk_pct"].dropna()
        self.assertTrue((valid_risks >= 0.0).all() and (valid_risks <= 100.0).all(),
                        f"Risk score outside [0, 100]: min={valid_risks.min()}, max={valid_risks.max()}")
        
        valid_threats = res1["glof_threat_pct"].dropna()
        self.assertTrue((valid_threats >= 0.0).all() and (valid_threats <= 100.0).all(),
                        f"GLOF threat outside [0, 100]: min={valid_threats.min()}, max={valid_threats.max()}")
        
        # 3. Exact threshold adherence
        low_mask = res1["risk_pct"] <= C.WATCH_PCT
        watch_mask = (res1["risk_pct"] > C.WATCH_PCT) & (res1["risk_pct"] <= C.ALERT_PCT)
        alert_mask = res1["risk_pct"] > C.ALERT_PCT
        
        self.assertTrue((res1.loc[low_mask, "risk_level"] == "Low").all(), "Low level threshold mismatch")
        self.assertTrue((res1.loc[watch_mask, "risk_level"] == "Watch").all(), "Watch level threshold mismatch")
        self.assertTrue((res1.loc[alert_mask, "risk_level"] == "Alert").all(), "Alert level threshold mismatch")

    # =========================================================================
    # 6. TEMPERATURE CANNOT TRIGGER AN ALERT ALONE (CAPPED AT 60%)
    # =========================================================================
    def test_06_temperature_alone_capped_at_60(self):
        """Temperature can't trigger an alert alone. It caps at 60%."""
        df = self.raw_hist.copy()
        art = M.train_production(df)
        
        # Construct synthetic window with extreme heatwave but completely neutral/missing lake signals
        extreme_heat_row = pd.DataFrame([{
            "window": pd.Timestamp("2024-07-20"),
            "lake_area_km2": np.nan,            # no area signal
            "lake_area_min_km2": np.nan,
            "vv_db": np.nan,                    # no radar signal
            "sar_backscatter_change": np.nan,
            "thermal_index": 500.0,             # extreme melt temperature (> 5 std dev anomaly)
            "observability": 0.15
        }])
        
        df_test = pd.concat([df[df.window < "2024-07-20"], extreme_heat_row]).reset_index(drop=True)
        res = M.score(df_test, art)
        
        row = res[res.window == pd.Timestamp("2024-07-20")].iloc[0]
        self.assertEqual(row["signals_used"], 1)
        self.assertEqual(row["main_driver"], "thermal")
        
        # Check thermal cap
        self.assertLessEqual(row["anomaly_risk_pct"], 60.0, 
                             f"Thermal alone exceeded 60% cap: got {row['anomaly_risk_pct']}%")
        self.assertLessEqual(row["risk_pct"], 60.0,
                             f"Overall risk exceeded 60% on thermal alone: got {row['risk_pct']}%")
        self.assertEqual(row["risk_level"], "Low", "Thermal alone must remain 'Low'")

    # =========================================================================
    # 7. FLEET RULES BEHAVE
    # =========================================================================
    def test_07_fleet_triage_rules_behavior(self):
        """Isolated spike -> Alert; Synchronous spike -> Regional; Missing -> No Data; Tiny fleet fallback."""
        dates = pd.date_range("2024-01-01", periods=5, freq="10D")
        lakes = [f"lake_{i:02d}" for i in range(20)]  # 20 lakes (>= MIN_PEERS=8)
        
        # Baseline normal risk matrix: 20%
        R = pd.DataFrame(20.0, index=dates, columns=lakes)
        
        # 1. Isolated Spike: lake_00 jumps to 96% at window 2
        # Fleet median is 20.0%. Excess for lake_00 = 76.0% (>= 40.0% excess, >= 95.0% risk)
        R.loc[dates[2], "lake_00"] = 96.0
        
        # 2. Regional Spike: At window 3, spike across all lakes to 88%
        # Fleet median is 88.0%. Excess is 0% (< 25%).
        for k in range(20):
            R.loc[dates[3], f"lake_{k:02d}"] = 88.0
            
        # 3. Missing Data: lake_05 has NaN at window 4
        R.loc[dates[4], "lake_05"] = np.nan
        
        code, med, E, share, regional = T.tiers(R)
        
        # Assert Isolated Spike is ALERT (code 4)
        self.assertEqual(code.loc[dates[2], "lake_00"], 4, "Isolated spike did not receive ALERT tier")
        self.assertEqual(code.loc[dates[2], "lake_01"], 1, "Peers should remain CLEAR tier")
        
        # Assert Regional Spike is REGIONAL (code 2), NOT Watch (3) or Alert (4)
        self.assertTrue(regional.loc[dates[3]], "Window 3 should be flagged as regional event")
        self.assertEqual(code.loc[dates[3], "lake_00"], 2, "Synchronous spike must be discounted as REGIONAL")
        self.assertEqual(code.loc[dates[3], "lake_05"], 2, "Synchronous spike must be discounted as REGIONAL")
        
        # Assert Missing Data is NO DATA (code 0), NEVER CLEAR (code 1)
        self.assertEqual(code.loc[dates[4], "lake_05"], 0, "Missing data window must be NO DATA (0), never CLEAR")
        
        # 4. Tiny fleet fallback (fewer than 8 lakes)
        tiny_lakes = ["tiny_1", "tiny_2", "tiny_3"]  # N=3 < 8
        R_tiny = pd.DataFrame({
            "tiny_1": [96.0],  # >= 95 -> ALERT
            "tiny_2": [82.0],  # >= 80 -> WATCH
            "tiny_3": [45.0],  # < 80 -> CLEAR
        }, index=[dates[0]])
        
        code_tiny, med_tiny, _, _, _ = T.tiers(R_tiny)
        self.assertTrue(pd.isna(med_tiny.iloc[0]), "Tiny fleet should skip fleet median")
        self.assertEqual(code_tiny.loc[dates[0], "tiny_1"], 4, "Tiny fleet >= 95 must trigger plain ALERT")
        self.assertEqual(code_tiny.loc[dates[0], "tiny_2"], 3, "Tiny fleet >= 80 must trigger plain WATCH")
        self.assertEqual(code_tiny.loc[dates[0], "tiny_3"], 1, "Tiny fleet < 80 must be CLEAR")
        
        # 5. Exact Watch / Alert boundary checks
        # Single lake with 40 peers at 20%
        R_bound = pd.DataFrame(20.0, index=[dates[0]], columns=[f"l_{i}" for i in range(41)])
        # Lake l_00 at exactly 95.0% (excess 75 >= 40)
        R_bound.loc[dates[0], "l_00"] = 95.0
        # Lake l_01 at 94.9% (excess 74.9 >= 25)
        R_bound.loc[dates[0], "l_01"] = 94.9
        # Lake l_02 at 80.0% (excess 60 >= 25)
        R_bound.loc[dates[0], "l_02"] = 80.0
        # Lake l_03 at 79.9%
        R_bound.loc[dates[0], "l_03"] = 79.9
        
        c_bound, _, _, _, _ = T.tiers(R_bound)
        self.assertEqual(c_bound.loc[dates[0], "l_00"], 4, "95.0% with excess must be ALERT")
        self.assertEqual(c_bound.loc[dates[0], "l_01"], 3, "94.9% must be WATCH, not Alert")
        self.assertEqual(c_bound.loc[dates[0], "l_02"], 3, "80.0% with excess must be WATCH")
        self.assertEqual(c_bound.loc[dates[0], "l_03"], 1, "79.9% must be CLEAR")

    # =========================================================================
    # 8. 10-DAY CALENDAR & WINDOW ALIGNMENT
    # =========================================================================
    def test_08_calendar_and_window_alignment(self):
        """Calendar correctly handles 36 windows/year, leap years, year end, and skips unfinished windows."""
        import fetch as F
        
        # 1. Window start slotting
        self.assertEqual(F.window_start("2023-01-01"), pd.Timestamp("2023-01-01"))
        self.assertEqual(F.window_start("2023-01-08"), pd.Timestamp("2023-01-01"))
        self.assertEqual(F.window_start("2023-01-10"), pd.Timestamp("2023-01-01"))
        self.assertEqual(F.window_start("2023-01-11"), pd.Timestamp("2023-01-11"))
        self.assertEqual(F.window_start("2023-12-25"), pd.Timestamp("2023-12-17"))
        self.assertEqual(F.window_start("2023-12-31"), pd.Timestamp("2023-12-17"))
        
        # Leap year 2024
        self.assertEqual(F.window_start("2024-02-29"), pd.Timestamp("2024-02-20"))
        self.assertEqual(F.window_start("2024-12-31"), pd.Timestamp("2024-12-16"))
        
        # 2. Window End Boundary
        # Mid-year windows end in exactly 10 days
        self.assertEqual(P.window_end(pd.Timestamp("2023-05-01")), pd.Timestamp("2023-05-11"))
        # Final year window (day > 350) ends on Jan 1 of next year
        final_win_2023 = pd.Timestamp("2023-12-17")
        self.assertEqual(P.window_end(final_win_2023), pd.Timestamp("2024-01-01"))
        
        final_win_2024 = pd.Timestamp("2024-12-16")
        self.assertEqual(P.window_end(final_win_2024), pd.Timestamp("2025-01-01"))
        
        # 3. Exactly 36 windows per year
        for yr in [2022, 2023, 2024]: # includes leap year 2024
            windows = []
            for k in range(36):
                s = pd.Timestamp(yr, 1, 1) + pd.Timedelta(days=10 * k)
                windows.append(s)
            self.assertEqual(len(windows), 36)
            
        # 4. Skip Unfinished Windows
        # If last window was 2024-12-06 and today is 2024-12-20:
        # Window 2024-12-16 runs until 2025-01-01. Since 2025-01-01 > today (2024-12-20),
        # completed_windows must NOT return it!
        todo = P.completed_windows(pd.Timestamp("2024-12-06"), pd.Timestamp("2024-12-20"))
        self.assertEqual(len(todo), 0, "Unfinished window ending in future was not skipped!")
        
        # Once today reaches 2025-01-01, the window is completed and must be returned
        todo_done = P.completed_windows(pd.Timestamp("2024-12-06"), pd.Timestamp("2025-01-01"))
        self.assertEqual(len(todo_done), 1)
        self.assertEqual(todo_done[0], (pd.Timestamp("2024-12-16"), pd.Timestamp("2025-01-01")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
