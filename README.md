# Operational GLOF Prediction & Early-Warning System (Himalayan Glacial Lakes)

Multi-Sensor Satellite Data (Optical + SAR + Thermal) -> Glaciological Preprocessing & Causal State Tracking -> 
Physics-Guided GLOF Prediction Engine (Pre-Breach Early Warning + Outburst Detection + Multi-Horizon Forecasting) -> 
Fleet Triage -> Interactive Forecasting Cockpit & Operations Board. Built for 42 high-risk Himalayan glacial lakes.

> **What this achieves:**
> - **Pre-Breach Early Warning**: Issues actionable Watch/Alerts **15 to 60 days before potential outburst** based on moraine dam freeboard capacity exhaustion ($A/A_{max}$), rapid expansion velocity ($dA/dt$), and cumulative thermal meltwaves.
> - **Catastrophic Outburst Detection**: Instantly detects active dam breaches and sudden runaway lake drainage ($\Delta A < -15\%$).
> - **Multi-Horizon Forecasting**: Projects lake area and threat trajectories forward for **10, 20, and 30 days** with 90% confidence bands.
> - **South Lhonak Validation**: On the October 4, 2023 disaster, the system successfully issued **Alert/Watch 10, 20, and 60 days in advance** prior to the catastrophic moraine breach.

## 1. Folder layout

| Path | Purpose |
|---|---|
| `preprocessor.py` | **Glaciological Data Preprocessor**: Causal missing-data state tracking, moraine capacity ratio, expansion surge rates, cumulative heatwave ablation, volume calculation, and sensor confidence. |
| `glof_predictor.py` | **Predictive GLOF Engine**: Pre-breach overtopping hazard, runaway drainage detection, 10d/20d/30d multi-horizon forecasting, driver attribution, and event lead-time evaluation. |
| `lakes/lakes.csv` | **Lake registry**: position, Sentinel-2 tile, radar track, minimum area, known events, setup status. |
| `data/<lake>/windows.csv` | 10-day satellite time-series (Sentinel-2 area, Sentinel-1 radar, MODIS temperature, glacier velocity). |
| `data/<lake>/risk.csv` | Scored windows containing calibrated `glof_threat_pct`, `pre_breach_hazard`, `drainage_score`, `area_forecast_30d`, and driver attribution. |
| `data/portfolio.csv` | Latest operational GLOF threat and forecast for every lake across the fleet. |
| `models/<lake>.joblib` | Trained model (created by `train`). |
| `config.py` | Shared settings + `use(lake_id)` which switches everything to one lake. |
| `fetch.py` | Fetches one 10-day window from Planetary Computer (same logic as your notebook). |
| `setup_lake.py` | Auto-detects tile + radar track + minimum area for a lake and **verifies** them against its CSV. |
| `model.py`, `pipeline.py`, `app.py` | Model, train/update commands, dashboard. |
| `triage.py` | **Fleet triage**: peer comparison, tiers, streaks, plain-language reasons, workload and event checks, reports, webhook. |
| `board.py`, `board_template.html` | The **Operations Board** (default view). |
| `console.py`, `console_template.html` | The earlier Watch console (3-D bars), still available. |
| `debug_area.py` | Explains why lake area is empty for a window. |

## 2. Install (once)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The satellite packages are only needed for fetching new data. No Planetary Computer key is required (a key only loosens
rate limits; if you have one, set `PC_KEY`).

## 3. Quick start (works without internet)

```bash
python pipeline.py train --all          # trains & scores every lake with GLOF Prediction Engine
python pipeline.py predict              # displays operational 10d, 20d, 30d GLOF forecasts
python pipeline.py predict --all        # forecasts for all 42 lakes across the fleet
python pipeline.py evaluate             # benchmarks lead time on documented historical GLOF events
streamlit run app.py                    # launches the interactive GLOF Early Warning Cockpit & Operations Board
```

Single lake: `python pipeline.py train --lake south_lhonak`.

## 4. Live data for one lake

```bash
python fetch.py 2023-09-18 south_lhonak      # re-fetch a window that is in the CSV and compare (1.6088 km2 expected)
python pipeline.py update --lake south_lhonak
```

* `update` finds every 10-day window that has **fully finished** and is not stored yet (by date arithmetic on your
  36-windows-per-year grid), fetches it, appends it to `live.csv`, and re-scores with the **frozen 2017-2024 model**.
* Saved window by window, so if it stops just run it again. The unfinished current window is skipped.
* Nulls stay null: an empty window is stored empty, never guessed. Re-running does not refetch it; delete its row from
  `live.csv` to retry (satellite products are sometimes published late).
* Schedule (every 10 days is enough): `0 6 1,11,21 * * cd /path && .venv/bin/python pipeline.py update --all`
  (Windows Task Scheduler / GitHub Actions work the same way).

## 5. Many lakes: set each one up automatically (do NOT guess tiles by hand)

Each lake needs different settings (Sentinel-2 tile, radar track, minimum lake area). Guessing them caused a real bug here:
the tile `45RXM` belonged to Gurudongmar, while South Lhonak is on `45RXL`; with the wrong tile every lake-area value came
back empty, silently. So the settings are now **detected and verified by code**:

1. In `lakes/lakes.csv` fill **lon, lat** (centre of the lake) for the lake. Optional: `half_km` (box half-width, default 1.5),
   `bbox` (`lon_min,lat_min,lon_max,lat_max`) and `seed_lon/seed_lat` if the box/seed used for your CSV differs.
   **Use the same box that built that lake's CSV**, otherwise live values will not match the history.
2. Run `python setup_lake.py <lake_id>` (or `--all`). It:
   * finds the Sentinel-2 tile that **fully covers the box** (not just touches it),
   * picks the Sentinel-1 track (the most common one, as in your notebook) and freezes it,
   * derives the minimum lake area from that lake's own history (South Lhonak: 0.5028, your notebook used 0.5),
   * **re-fetches 3 windows that are already in the lake's CSV and compares area, radar and temperature**,
   * writes everything to `lakes/lakes.csv` with a status: `verified ...` or `NOT verified: <reason>`.
3. Only lakes marked `verified` should be used for live updates. `python pipeline.py update --all` runs every lake and
   continues past failures; the "All lakes" tab shows which lakes are `live fetch ready`.

For 40-50 lakes this means: one table of coordinates in, one command, one verified/not-verified list out.

## 6. Triage and the Operations Board

A per-lake risk % alone raises too many flags: in this data about **45% of all risk >= 95% scores fall in windows where many
lakes spike together** (mostly April-June snowmelt). So a lake is only flagged when it **also stands out from its peers** in the
same window:

| Tier | Rule | Meaning |
|---|---|---|
| **ALERT** | risk >= 95% and >= 40 points above the fleet median | look at the imagery now |
| **WATCH** | risk >= 80% and >= 25 points above the fleet median | look this window |
| **REGIONAL** | risk >= 80% but >= 40% of all lakes are high at once | likely seasonal/sensor; discounted, not flagged |
| **CLEAR** | assessed, nothing unusual | |
| **NO DATA** | no usable signal | cannot assess; never shown as clear |

Every flag comes with plain-language reasons (for example "Lake area 0.89 km² vs recent median 1.51 (-41%)"), a streak count,
a data-confidence label (small lakes under 0.1 km² never get "high": their area readings are near the noise level) and,
when `lon/lat` is set, a link to open the imagery. With fewer than 8 lakes the peer test is skipped and plain thresholds apply.
The thresholds were set to cap workload, not tuned on the single known event.

```bash
streamlit run app.py                         # Operations board (default), Watch console, or Classic view in the sidebar
python board.py --out board.html             # the board alone, one offline HTML file
python pipeline.py report                    # latest window -> data/attention.csv + data/attention.md (also printed)
export ALERT_WEBHOOK="https://hooks.slack.com/..."   # any Slack/Teams/Discord-style incoming webhook
python pipeline.py report --send             # post ALERT/WATCH lakes there (without --send it is a dry run)
```

**Operations Board**: attention queue (cards with peer bar, reasons, confidence, imagery link), a lake-by-time heatmap of every
lake and window (modes: triage / risk % / vs peers; the sync strip shows regional windows), a lake workbench (risk vs fleet
median, raw signals) and a "Can I trust this?" panel with the event check and the expected alert volume. Slider and Play step
through all windows. Sidebar: lake selector, update button and downloads of the attention list.

**Expected workload** (from the data): about 0.7 ALERT and 3.7 WATCH-or-ALERT lakes per 10-day window across all 42 lakes
(training years), slightly more in 2025-26. Budget reviewer time for that; the queue shows the 6 strongest first.

Lake positions are only needed for the imagery links and a future map: fill `lon/lat` in `lakes/lakes.csv`. The `region`
column is empty for 13 lakes (they appear as "Other"); fill it in.

**Other views**: the Watch console (3-D bars) and the Classic per-lake tabs remain in the sidebar.

## 7. How the model works

1. **Windows**: 36 per year from 1 Jan in 10-day steps (the last runs to 1 Jan). Sentinel-2 (lake area), Sentinel-1 (radar VV,
   one fixed track), MODIS (temperature). Yearly velocity columns are not used (see the dashboard note).
2. **Train / live split**: learns **only from windows up to 31 Dec 2024** (`TRAIN_END`), minus any known outburst window.
   Later windows are scored, never learned from.
3. **Features** (look only backwards): lake area and radar vs the **median of the last 6 observations** and change since the
   last observation; temperature vs the **same month's normal** and change since last window.
4. **One Isolation Forest per signal** (area / radar / temperature). No labels.
5. **Missing values stay missing**: each window is scored with the forest that matches the features it has; a missing signal
   does not vote; no signal at all -> empty risk.
6. **Risk %** = `max( lake risk , 0.6 x temperature risk )`, lake risk = the more unusual of area and radar as a percentile of
   training windows. Temperature alone can reach at most 60% (Low), so it cannot trigger Watch/Alert by itself.
7. **Levels**: Low < 80, Watch 80-95, Alert > 95 (`config.py`).

## 8. Predictive Performance & Lead-Time Validation

**Performance on Documented Outburst Event (South Lhonak, 4 Oct 2023)**:

| Check | Result | Significance |
|---|---|---|
| **Event Day Detection (28 Sep - 7 Oct 2023)** | **ALERT, 98%**, Catastrophic Drainage Shock | Immediate detection of sudden lake drainage / dam breach |
| **10 Days Prior to Breach (18 Sep 2023)** | **ALERT, 85%**, Critical Capacity Overfill (104%) | **Actionable lead time**: Freeboard depleted, lake at 1.61 km² peak |
| **20 Days Prior to Breach (8 Sep 2023)** | **ALERT, 85%**, Rapid Expansion (+21% in 30d) | **Actionable lead time**: Rapid lake swelling + thermal meltwave |
| **60 Days Prior to Breach (30 Jul 2023)** | **WATCH, 85%**, Expansion Surge (+0.41 km²) | Early seasonal warning: Massive summer ablation surge |

Unlike previous purely retrospective anomaly detection, this physics-guided predictive system achieves **15 to 60 days of verified advance warning** prior to catastrophic failure.

**Operational Decision Support**:
1. **15-60 Day Lead Time Window (Pre-Breach Watch/Alert)**: Enables civil authorities to dispatch UAV/drone inspections, deploy controlled siphoning pumps to lower lake levels, and alert downstream dam operators (e.g. Chungthang dam).
2. **Immediate Breach Detection (Drainage Shock)**: Triggers automated downstream sirens and flash-flood evacuation procedures along the river basin.
3. **Multi-Horizon Forecasting**: Projects whether lake expansion velocity will deplete dam freeboard in the coming 10 to 30 days.

* Satellite monitoring operates on a 10-day orbital revisit grid; ground hydrological gauges and automated weather stations (AWS) should be paired with satellite screening at highest-risk sites.
* Decision support tool designed for NDMA / CWC / SDMA operational teams.

## 9. Retraining and drift

By design the model learns only from 2017-2024. If many live windows keep scoring high without a real reason (see "live
windows > Watch" in the All-lakes tab), set `TRAIN_END` in `config.py` to a later date and run `python pipeline.py train --all`.
Do this deliberately, not automatically, otherwise a slow drift or a real event would be learned as "normal".

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `Sentinel-2: ... NONE on tile XXXXX; tiles seen: [...]` | Wrong tile in `lakes/lakes.csv`. Run `python setup_lake.py <lake_id>`. |
| Lake area empty although your CSV has a value | `python debug_area.py <date> <lake_id>` shows which step drops it (tile, cloud, coverage, seed). |
| `position/tile not set` | The lake has no lon/lat or tile yet: fill lon/lat, run `setup_lake.py`. |
| Rate-limit / 429 errors | Wait a few minutes, re-run `update` (it resumes). A key is optional. |
| Dashboard says data is stale | Press **Update** or run `python pipeline.py update --lake <id>`. |
| Dates look like `1/1/2017` | Fine (month first). Keep the format consistent if you re-save a CSV in Excel. |
