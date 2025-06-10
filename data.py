# data.py
"""
Data loading, preprocessing, scaling, and resampling utilities.
"""
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

TARGET = "BP_spike"

# ––– default feature list (edit here once – every other file just imports it) –––
# We have added the following new features at the end:
#   'hr_std_rolling_5', 'hr_std_rolling_10',         # extra rolling‐std on HR
#   'steps_total_rolling_10', 'steps_total_rolling_20', # extra rolling‐mean on steps
#   'sin_hour', 'cos_hour',                           # embed hour_of_day cyclically
#   'recent_spike_flag'                               # binary if last spike < 10 min ago
DEFAULT_FEATURES: Tuple[str, ...] = (
    'hr_mean_5min', 'hr_min_5min', 'hr_max_5min', 'hr_std_5min',
    'steps_total_5min', 'steps_mean_5min', 'steps_min_5min', 'steps_max_5min',
    'steps_std_5min', 'steps_diff_5min',
    'hr_mean_10min', 'hr_min_10min', 'hr_max_10min', 'hr_std_10min',
    'steps_total_10min', 'steps_mean_10min', 'steps_min_10min',
    'steps_max_10min', 'steps_std_10min', 'steps_diff_10min',
    'hr_mean_30min', 'hr_min_30min', 'hr_max_30min', 'hr_std_30min',
    'steps_total_30min', 'steps_mean_30min', 'steps_min_30min',
    'steps_max_30min', 'steps_std_30min', 'steps_diff_30min',
    'hr_mean_60min', 'hr_min_60min', 'hr_max_60min', 'hr_std_60min',
    'steps_total_60min', 'steps_mean_60min', 'steps_min_60min',
    'steps_max_60min', 'steps_std_60min', 'steps_diff_60min',
    'stress_mean', 'stress_min', 'stress_max', 'stress_std',
    'stress_mean_lag_1', 'stress_mean_lag_3', 'stress_mean_lag_5',
    'BP_spike_lag_1', 'BP_spike_lag_3', 'BP_spike_lag_5',
    'hr_mean_5min_lag_1', 'hr_mean_5min_lag_3', 'hr_mean_5min_lag_5',
    'steps_total_10min_lag_1', 'steps_total_10min_lag_3',
    'steps_total_10min_lag_5',
    'hr_steps_ratio', 'stress_weighted_hr', 'stress_steps_ratio',
    'steps_hr_variability_ratio',
    'hr_mean_rolling_3', 'steps_total_rolling_5', 'hr_std_rolling_3',
    'cumulative_stress_30min', 'cumulative_steps_30min',
    'hour_of_day', 'day_of_week', 'is_working_hours', 'is_weekend',
    'time_since_last_BP_spike',
    # ─────────── NEW FEATURES ───────────
    'hr_std_rolling_5',   # std of hr_std_10min over 5 rows
    'hr_std_rolling_10',  # std of hr_std_10min over 10 rows
    'steps_total_rolling_10',  # mean of steps_total_30min over 10 rows
    'steps_total_rolling_20',  # mean of steps_total_30min over 20 rows
    'sin_hour', 'cos_hour',       # cyclic encoding of hour_of_day
    'recent_spike_flag'           # 1 if time_since_last_BP_spike < 10 minutes
)


def load_data(
    csv_path: Path,
    keep_features: Optional[List[str]] = None,
    train_days: int = 20,
    sampling_strategy: float = 0.50,
    random_state: int = 42,
    use_adasyn: bool = True,
):
    """
    Returns
    -------
    X_train, y_train               – raw feature frames
    X_test,  y_test
    X_train_res, y_train_res       – **scaled + (optional) ADASYN-balanced** arrays
    X_test_scaled
    scaler
    features (list of column names actually used)
    """
    df = pd.read_csv(csv_path)

    # choose features
    features = keep_features or list(DEFAULT_FEATURES)

    # numeric sanitise
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    # split chronologically
    df["datetime_local"] = pd.to_datetime(df["datetime_local"])
    cutoff = df["datetime_local"].min() + pd.Timedelta(days=train_days)
    train_df = df[df["datetime_local"] < cutoff]
    test_df = df[df["datetime_local"] >= cutoff]

    X_train, y_train = train_df[features], train_df[TARGET]
    X_test, y_test = test_df[features], test_df[TARGET]

    # scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # optionally apply ADASYN
    if use_adasyn:
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        X_train_res, y_train_res = adasyn.fit_resample(X_train_s, y_train)
    else:
        # no oversampling → use the scaled data “as is”
        X_train_res, y_train_res = X_train_s, y_train

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_res,
        y_train_res,
        X_test_s,
        scaler,
        features,
    )
