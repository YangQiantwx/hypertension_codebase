# BP‑Spike Prediction Pipeline

This repository provides end‑to‑end tools to build a blood‑pressure spike prediction dataset from raw wearable CSV streams and train both XGBoost and attention‑based models with optional ensembling.

## Directory Structure

```
hypertension/
├── data.py                  # data‑loading & preprocessing utilities
├── evaluation.py            # ensemble search & threshold sweep helpers
├── model.py                 # XGBoost pipelines & attention model definitions
├── preprocess.py            # raw→processed CSV builder (per PID)
├── train.py                 # master CLI for training & plotting
├── hp/                      # raw data folders by participant
│   ├── hp10/
│   │   ├── blood_pressure_readings_ID10_cleaned.csv
│   │   ├── hp10_hr.csv
│   │   ├── hp10_steps.csv
│   │   └── questionnaire_responses_ID10.csv
│   ├── hp15/
│   │   └── ...
│   └── ...
└── processed/               # processed outputs by participant
    ├── hp10/
    │   └── processed_bp_prediction_data.csv
    ├── hp15/
    │   └── ...
    └── ...
```

*(Only a few files are shown above — each `hpXX/` contains the raw HR, steps, BP, and stress CSVs.)*

## Usage

### 1. Preprocessing

Generate the processed dataset for a given participant ID (`PID`), e.g. `10`:

```bash
python preprocess.py --pid 10
```

**Available flags:**

* `--roll_windows`: comma‑separated rolling window sizes in minutes (default: `5,10,30,60`)
* `--lag_horizons`: comma‑separated lag horizons in rows (default: `1,3,5`)
* `--agg_lengths`: small,large row‑aggregate lengths (default: `3,5`)
* `--bp_sys_thresh` / `--bp_dia_thresh`: systolic/diastolic thresholds (default: `135`, `85`)
* `--work_hours`: work‑hour start,end in 24‑h (default: `9,17`)
* `--weekend_day`: integer for weekend start (0=Mon…6=Sun, default `5`)

The processed CSV will be written to:

```
processed/hp10/processed_bp_prediction_data.csv
```

### 2. Training & Plotting

Train XGBoost and/or attention models, perform ensembling, and save performance plots:

```bash
python train.py \
  --csv processed/hp10/processed_bp_prediction_data.csv \
  --trials 5 \
  --epochs 50
```

**Key arguments:**

* `--models`: comma‑separated list of model tags (`xgb`, `xgb_fixed`, `attn`), default `xgb,attn`
* `--drop`: comma‑separated feature names to exclude from training
* `--train_days`: number of days for the train/test split (default `20`)
* `--batch`, `--verbose`, `--cpu` (force CPU), etc.

After training, two PNGs will be saved in the same folder as the CSV (e.g. `processed/hp10/`):

* `sens_spec_plot.png` — Sensitivity & Specificity vs. threshold
* `shap_summary.png` — SHAP summary plot for the XGBoost model
