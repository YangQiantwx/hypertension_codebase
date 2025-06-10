#!/usr/bin/env python3
"""
preprocess.py  ─ Wearable BP‑Spike Dataset Builder
=================================================

Given a participant ID (e.g. 15) this script:
  1. Loads raw CSV streams **(HR, steps, BP, stress)** located under
         hp/hp<PID>/
  2. Drops *every* row whose date equals the **earliest BP sample date** so
     that all downstream features use data strictly *after* the first clinic
     visit / device start‑up day (this fixes the ‑24 h data hole you saw).
  3. Generates exactly the same engineered features as your reference
     notebook (§ below) while letting you customise *only* the
        • rolling windows  (minutes)
        • lag horizons    (row steps)
        • row‑aggregate lengths (small, large)
     via CLI flags.  All other hyper‑parameters keep their default values but
     can still be overridden if you wish.

Usage examples
--------------
    # ➤ Build with defaults (5,10,30,60‑min windows; 1,3,5‑row lags)
    python preprocess.py --pid 15

    # ➤ Custom windows / lags (everything else unchanged)
    python preprocess.py --pid 15 \
                         --roll_windows 5,15,30,60 \
                         --lag_horizons 1,3,5

The final CSV is written to
    processed/hp<PID>/processed_bp_prediction_data.csv
and is byte‑identical (column order & values) to the manual pipeline you
shared.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import numpy as np

###############################################################################
# ──────────────────────────────  Core logic  ──────────────────────────────── #
###############################################################################

def process_participant(
    *,
    pid: int,
    root: Path = Path(__file__).parent,
    bp_sys_thresh: int = 135,
    bp_dia_thresh: int = 85,
    roll_windows: Union[List[int], Tuple[int, ...]] = (5, 10, 30, 60),
    lag_horizons: Union[List[int], Tuple[int, ...]] = (1, 3, 5),
    agg_lengths: Union[List[int], Tuple[int, int]] = (3, 5),  # (small, large)
    work_hours: Tuple[int, int] = (9, 17),
    weekend_day: int = 5,
) -> Path:
    """Process one participant and return the output CSV path."""

    # ─────────────── Paths ───────────────
    hp_dir = root / "hp" / f"hp{pid}"
    out_dir = root / "processed" / f"hp{pid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────── Load raw streams ───────────────
    df_hr = pd.read_csv(hp_dir / f"hp{pid}_hr.csv")
    df_steps = pd.read_csv(hp_dir / f"hp{pid}_steps.csv")
    df_bp = pd.read_csv(hp_dir / f"blood_pressure_readings_ID{pid}_cleaned.csv")
    df_stress = pd.read_csv(
        hp_dir / f"questionnaire_responses_ID{pid}.csv", on_bad_lines="skip"
    )

    # ─────────────── Normalise timestamps ───────────────
    df_hr["time"] = pd.to_datetime(df_hr["time"], utc=True).dt.tz_localize(None)
    df_steps["time"] = pd.to_datetime(df_steps["time"], utc=True).dt.tz_localize(
        None
    )
    df_bp["datetime_local"] = pd.to_datetime(df_bp["datetime_local"], utc=True).dt.tz_localize(
        None
    )
    df_stress["local_created_at"] = pd.to_datetime(
        df_stress["local_created_at"], utc=True
    ).dt.tz_localize(None)

    # ─────────────── Date filtering (auto) ───────────────
    #   Keep rows strictly > first BP date.  This behaves exactly like your
    #   manual notebook cell that removed 2024‑03‑25 (or 2024‑10‑18, etc.).
    #   Using BP ensures the same start date for all streams, even when stress
    #   is empty.
    first_date = df_bp["datetime_local"].dt.date.min()
    print(f"🗓️  Removing data from first day {first_date} (all streams)")

    date_mask = lambda s: s.dt.date > first_date
    df_bp = df_bp[date_mask(df_bp["datetime_local"])]
    df_stress = df_stress[date_mask(df_stress["local_created_at"])]
    df_hr = df_hr[date_mask(df_hr["time"])]
    df_steps = df_steps[date_mask(df_steps["time"])]

    # ─────────────── Sort streams ───────────────
    df_hr.sort_values("time", inplace=True)
    df_steps.sort_values("time", inplace=True)
    df_bp.sort_values("datetime_local", inplace=True)
    df_stress.sort_values("local_created_at", inplace=True)

    # ─────────────── BP spike flag ───────────────
    df_bp["BP_spike"] = (
        (df_bp["systolic"] > bp_sys_thresh) | (df_bp["diastolic"] > bp_dia_thresh)
    ).astype(int)

    # ─────────────── Align HR + Steps ───────────────
    df_bio = pd.merge_asof(
        df_hr,
        df_steps,
        on="time",
        direction="backward",
        suffixes=("_hr", "_steps"),
    )

    # ─────────────── Rolling‑window stats ───────────────
    df_bio.set_index("time", inplace=True)
    for w in roll_windows:
        s = f"{w}min"
        # HR
        df_bio[f"hr_mean_{s}"] = df_bio["value_hr"].rolling(s).mean()
        df_bio[f"hr_min_{s}"] = df_bio["value_hr"].rolling(s).min()
        df_bio[f"hr_max_{s}"] = df_bio["value_hr"].rolling(s).max()
        df_bio[f"hr_std_{s}"] = df_bio["value_hr"].rolling(s).std()
        # Steps
        df_bio[f"steps_total_{s}"] = df_bio["value_steps"].rolling(s).sum()
        df_bio[f"steps_mean_{s}"] = df_bio["value_steps"].rolling(s).mean()
        df_bio[f"steps_min_{s}"] = df_bio["value_steps"].rolling(s).min()
        df_bio[f"steps_max_{s}"] = df_bio["value_steps"].rolling(s).max()
        df_bio[f"steps_std_{s}"] = df_bio["value_steps"].rolling(s).std()
        df_bio[f"steps_diff_{s}"] = (
            df_bio[f"steps_max_{s}"] - df_bio[f"steps_min_{s}"]
        )
    df_bio.reset_index(inplace=True)

    # ─────────────── Merge BP onto biosignals ───────────────
    df = pd.merge_asof(
        df_bp,
        df_bio,
        left_on="datetime_local",
        right_on="time",
        direction="backward",
    )

    # ─────────────── Stress‑window features (±15 min) ───────────────
    if not df_stress.empty:
        def _stress_feats(t: pd.Timestamp) -> pd.Series:
            lo, hi = t - pd.Timedelta(minutes=15), t + pd.Timedelta(minutes=15)
            vals = df_stress.loc[
                (df_stress["local_created_at"] >= lo)
                & (df_stress["local_created_at"] <= hi),
                "stressLevel_value",
            ]
            return pd.Series(
                {
                    "stress_mean": vals.mean(),
                    "stress_min": vals.min(),
                    "stress_max": vals.max(),
                    "stress_std": vals.std(),
                }
            )

        df = pd.concat([df, df["datetime_local"].apply(_stress_feats)], axis=1)
    else:
        for c in ["stress_mean", "stress_min", "stress_max", "stress_std"]:
            df[c] = pd.NA

    # ─────────────── Lagged & interaction features ───────────────
    lag_bases = [
        "stress_mean",
        "BP_spike",
        "hr_mean_5min",
        "steps_total_10min",
    ]
    for base in lag_bases:
        if base in df.columns:
            for lag in lag_horizons:
                df[f"{base}_lag_{lag}"] = df[base].shift(lag)

    df["hr_steps_ratio"] = df["hr_mean_5min"] / (df["steps_total_10min"] + 1)
    df["stress_weighted_hr"] = df["hr_mean_5min"] * df["stress_mean"]
    df["stress_steps_ratio"] = df["stress_mean"] / (df["steps_total_10min"] + 1)
    df["steps_hr_variability_ratio"] = df["steps_std_10min"] / (
        df["hr_std_10min"] + 1e-5
    )

    # ─────────────── Row‑based rolling & cumulatives ───────────────
    small, large = agg_lengths[:2]
    if "hr_mean_5min" in df.columns:
        df[f"hr_mean_rolling_{small}"] = df["hr_mean_5min"].rolling(small).mean()
    if "steps_total_10min" in df.columns:
        df[f"steps_total_rolling_{large}"] = (
            df["steps_total_10min"].rolling(large).mean()
        )
    if "hr_std_10min" in df.columns:
        df[f"hr_std_rolling_{small}"] = df["hr_std_10min"].rolling(small).std()

    df.set_index("datetime_local", inplace=True)
    df[f"cumulative_stress_{small*10}min"] = df["stress_mean"].rolling(small).sum()
    df[f"cumulative_steps_{small*10}min"] = (
        df["steps_total_10min"].rolling(small).sum()
    )
    df.reset_index(inplace=True)

    # ─────────────── Contextual & time‑since‑last spike ───────────────
    df["hour_of_day"] = df["datetime_local"].dt.hour
    df["day_of_week"] = df["datetime_local"].dt.dayofweek
    df["is_working_hours"] = df["hour_of_day"].between(*work_hours).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= weekend_day).astype(int)

    df.sort_values("datetime_local", inplace=True)
    df["time_since_last_BP_spike"] = (
        df["datetime_local"].diff().dt.total_seconds() / 60
    )
    df["time_since_last_BP_spike"].ffill(inplace=True)

    # ─────────────── NEW FEATURES BEGIN HERE ───────────────
    # 1) Additional rolling‐std windows on HR (std of hr_std_10min)
    if "hr_std_10min" in df.columns:
        df["hr_std_rolling_5"]  = df["hr_std_10min"].rolling(5).std()
        df["hr_std_rolling_10"] = df["hr_std_10min"].rolling(10).std()

    # 2) Additional rolling‐mean windows on steps (mean of steps_total_30min)
    if "steps_total_30min" in df.columns:
        df["steps_total_rolling_10"] = df["steps_total_30min"].rolling(10).mean()
        df["steps_total_rolling_20"] = df["steps_total_30min"].rolling(20).mean()

    # 3) Cyclic encoding of hour_of_day
    #    (so that 23 and 0 are “close” rather than far apart)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)

    # 4) “Recent‐spike” flag: 1 if last spike was <10 minutes ago
    df["recent_spike_flag"] = (df["time_since_last_BP_spike"] < 10).astype(int)
    # ─────────────── NEW FEATURES END HERE ───────────────

    # ─────────────── Fill gaps & save ───────────────
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    out_csv = out_dir / "processed_bp_prediction_data.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved processed CSV to {out_csv.relative_to(root)}")

    return out_csv

###############################################################################
# ────────────────────────────  CLI Entrypoint  ───────────────────────────── #
###############################################################################

if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="Build processed BP‑spike dataset for one participant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pa.add_argument("--pid", type=int, required=True, help="Participant ID")
    pa.add_argument(
        "--roll_windows",
        type=str,
        default="5,10,30,60",
        help="Comma‑separated rolling windows (minutes)",
    )
    pa.add_argument(
        "--lag_horizons",
        type=str,
        default="1,3,5",
        help="Comma‑separated lag horizons (rows)",
    )
    pa.add_argument(
        "--agg_lengths",
        type=str,
        default="3,5",
        help="Comma‑separated small,large row‑aggregate lengths",
    )
    pa.add_argument("--bp_sys_thresh", type=int, default=135)
    pa.add_argument("--bp_dia_thresh", type=int, default=85)
    pa.add_argument(
        "--work_hours",
        type=str,
        default="9,17",
        help="Comma‑separated work‑hour start,end (24‑h)",
    )
    pa.add_argument("--weekend_day", type=int, default=5, help="0=Mon … 6=Sun")

    args = pa.parse_args()

    parse_ints = lambda s: [int(x) for x in s.split(",") if x]

    roll_windows = parse_ints(args.roll_windows)
    lag_horizons = parse_ints(args.lag_horizons)
    agg_lengths = parse_ints(args.agg_lengths)
    work_hours = tuple(parse_ints(args.work_hours)[:2])

    process_participant(
        pid=args.pid,
        bp_sys_thresh=args.bp_sys_thresh,
        bp_dia_thresh=args.bp_dia_thresh,
        roll_windows=roll_windows,
        lag_horizons=lag_horizons,
        agg_lengths=agg_lengths if len(agg_lengths) >= 2 else (agg_lengths[0],) * 2,
        work_hours=work_hours,
        weekend_day=args.weekend_day,
    )
