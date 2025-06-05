#!/usr/bin/env python3
"""
train.py – master CLI entry-point for the BP-spike prediction pipeline

Example
-------
# This will load processed/hp15/processed_bp_prediction_data.csv
# and save both plots into processed/hp15/
python train.py --csv processed/hp15/processed_bp_prediction_data.csv \
                --trials 5 --epochs 50
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from data import DEFAULT_FEATURES, load_data
from evaluation import ensemble_search, threshold_sweep
from model import (
    MODEL_REGISTRY,
    build_xgb_pipeline,
    fit_xgb_with_grid,
    fit_lstm_attention_model,
)

# ── GLOBAL SEED for deterministic runs ───────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def _early_parse_cpu_flag() -> bool:
    """Peek at sys.argv to catch --cpu before TensorFlow loads."""
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--cpu", action="store_true")
    args, _ = peek.parse_known_args()
    return args.cpu

# mask GPUs very early if requested
FORCE_CPU = _early_parse_cpu_flag()
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train BP-spike models (XGBoost, attention net, LSTM+attention) with optional ensemble.",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    p.add_argument("--csv", type=Path, required=True, help="Path to processed CSV.")
    p.add_argument(
        "--models",
        default="xgb,attn",
        help="Comma-separated list. Options: xgb, xgb_fixed, attn, lstm_attn",
    )
    p.add_argument("--drop", default="", help="Comma-separated list of feature names to exclude.")
    p.add_argument(
        "--train_days", type=int, default=20, help="Days of data in train split."
    )
    p.add_argument("--epochs", type=int, default=50, help="Epochs per tuner trial (attention & LSTM).")
    p.add_argument("--trials", type=int, default=5, help="RandomSearch trials.")
    p.add_argument("--batch", type=int, default=32, help="Batch size for attention/LSTM nets.")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Keras-fit verbosity level.")
    p.add_argument("--no_adasyn", action="store_true", help="If set, skip ADASYN oversampling.")
    p.add_argument(
        "--seq_len",
        type=int,
        default=10,
        help="Sequence length for LSTM (number of consecutive rows).",
    )
    return p.parse_args()


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Given X.shape = (n_samples, n_features) and y.shape = (n_samples,),
    return (X_seq, y_seq) where:
      X_seq.shape = (n_samples - seq_len + 1, seq_len, n_features)
      y_seq.shape = (n_samples - seq_len + 1,)
    We slide a window of length=seq_len over the chronological rows,
    using the label of the last row in each window.
    """
    n_samples, n_features = X.shape
    X_seq = []
    y_seq = []
    for i in range(seq_len - 1, n_samples):
        start = i - (seq_len - 1)
        X_seq.append(X[start : i + 1, :])  # shape = (seq_len, n_features)
        y_seq.append(y[i])  # label at the last timestep
    return np.array(X_seq), np.array(y_seq)


def main() -> None:
    args = parse_args()

    # Where to save plots
    out_folder = args.csv.parent
    out_folder.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & preprocess data ───────────────────────────────────────────
    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_res,
        y_train_res,
        X_test_s,
        _scaler,
        _feature_list,
    ) = load_data(
        args.csv,
        keep_features=[f for f in DEFAULT_FEATURES if f not in args.drop.split(",") if f],
        train_days=args.train_days,
        sampling_strategy=0.50,
        random_state=SEED,
        use_adasyn=(not args.no_adasyn),
    )

    pos, neg = int(np.sum(y_train == 1)), int(np.sum(y_train == 0))
    spw = neg / pos
    print(f"\n🔹 Train positives={pos}  negatives={neg}  scale_pos_weight={spw:4.2f}")

    # ── 2. Prepare sequences for LSTM (if needed) ───────────────────────────
    # We’ll only use these if the "lstm_attn" tag is present.
    if "lstm_attn" in [m.strip() for m in args.models.split(",")]:
        seq_len = args.seq_len

        # y_train_res is already a NumPy array (from ADASYN), so no change needed there.
        X_train_seq, y_train_seq = create_sequences(X_train_res, y_train_res, seq_len)

        # But y_test is a Pandas Series—convert to NumPy array before slicing.
        y_test_np = y_test.to_numpy()
        X_test_seq, y_test_seq = create_sequences(X_test_s, y_test_np, seq_len)
    else:
        X_train_seq = y_train_seq = X_test_seq = y_test_seq = None

    # ── 3. Train selected models ─────────────────────────────────────────────
    y_preds = []
    xgb_pipeline = None

    for tag in [m.strip() for m in args.models.split(",")]:
        if tag == "xgb":
            print(f"\n⚙️  XGBoost grid search …")
            sampling_arg = None if args.no_adasyn else 0.50
            model = fit_xgb_with_grid(
                X_train,
                y_train,
                n_jobs=-1,
                random_state=SEED,
            )
            y_pred_xgb = model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_auc_score

            print(f"🔹 Pure XGB Test-set AUC: {roc_auc_score(y_test, y_pred_xgb):.3f}")

            y_preds.append(y_pred_xgb)
            xgb_pipeline = model

        elif tag == "xgb_fixed":
            print(f"\n⚙️  Training xgb_fixed …")
            pipeline = build_xgb_pipeline(
                scale_pos_weight=spw,
                sampling_strategy=None if args.no_adasyn else 0.50,
                max_depth=5,
                learning_rate=0.05,
                n_estimators=150,
                subsample=1.0,
                colsample_bytree=1.0,
                min_child_weight=1,
                gamma=0.0,
            )
            pipeline.fit(X_train, y_train)
            y_pred_fixed = pipeline.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_auc_score

            print(f"🔹 XGB_fixed Test-set AUC: {roc_auc_score(y_test, y_pred_fixed):.3f}")

            y_preds.append(y_pred_fixed)
            xgb_pipeline = pipeline

        elif tag == "attn":
            print("\n⚙️  Training attention model …")
            model_attn, best_hps = MODEL_REGISTRY["attn"](
                X_train_res,
                y_train_res,
                X_test_s,
                y_test,
                max_trials=args.trials,
                epochs=args.epochs,
                batch_size=args.batch,
                verbose=args.verbose,
            )
            print(f"   ⇢ best attention H-params: {best_hps}")
            y_pred_attn = model_attn.predict(X_test_s).flatten()
            y_preds.append(y_pred_attn)

        elif tag == "lstm_attn":
            print("\n⚙️  Training LSTM+Attention model …")
            # Fit on sequence data (X_train_seq, y_train_seq), validate on (X_test_seq, y_test_seq)
            model_lstm, best_hps_lstm = MODEL_REGISTRY["lstm_attn"](
                X_train_seq,
                y_train_seq,
                X_test_seq,
                y_test_seq,
                max_trials=args.trials,
                epochs=args.epochs,
                batch_size=args.batch,
                verbose=args.verbose,
            )
            print(f"   ⇢ best LSTM+Attention H-params: {best_hps_lstm}")
            # model_lstm.predict(X_test_seq) returns one probability per sequence
            y_pred_lstm_seq = model_lstm.predict(X_test_seq).flatten()
            # Pad the first (seq_len - 1) predictions with 0.5 so y_pred_lstm has same length as y_test
            pad = np.full((args.seq_len - 1,), 0.5)
            y_pred_lstm = np.concatenate([pad, y_pred_lstm_seq])
            y_preds.append(y_pred_lstm)

        else:
            sys.exit(f"❌ Unknown model tag '{tag}'.")

    # ── 4. Ensemble (if at least two preds) ──────────────────────────────────
    if len(y_preds) >= 2:
        # Use the first two for a 2-way ensemble search. If you have >2, you could average others.
        ens = ensemble_search(y_preds[0], y_preds[1], y_test)
        final_prob = ens["a"] * y_preds[0] + ens["b"] * y_preds[1]
        print(f"\n🔹 Ensemble – α={ens['a']:4.2f}  β={ens['b']:4.2f}  AUROC={ens['auc']:5.3f}")
    else:
        final_prob = y_preds[0]

    # ── 5. Threshold sweep & printing ───────────────────────────────────────
    best_thr = threshold_sweep(y_test, final_prob)
    print(
        f"\n✅  BEST THRESHOLD={best_thr['thr']:4.2f}  "
        f"Sens={best_thr['sens']:4.3f}  Spec={best_thr['spec']:4.3f}  "
        f"(Youden={best_thr['youden']:4.3f})"
    )

    # ── 6. Sensitivity/Specificity plot ────────────────────────────────────
    thr_grid = np.arange(0, 1.01, 0.01)
    sens_list, spec_list = [], []
    for t in thr_grid:
        tn, fp, fn, tp = confusion_matrix(y_test, (final_prob >= t).astype(int)).ravel()
        sens_list.append(tp / (tp + fn) if (tp + fn) else 0)
        spec_list.append(tn / (tn + fp) if (tn + fp) else 0)

    plt.figure(figsize=(8, 5))
    plt.plot(thr_grid, sens_list, label="Sensitivity", marker="o")
    plt.plot(thr_grid, spec_list, label="Specificity", marker="s")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Sensitivity / Specificity vs Threshold")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    sens_path = out_folder / "sens_spec_plot.png"
    plt.savefig(sens_path, dpi=300)
    plt.close()
    print(f"📸  Saved  ➜  {sens_path}")

    # ── 7. SHAP summary (if XGB was trained) ────────────────────────────────
    if xgb_pipeline is not None:
        scaler_xgb = xgb_pipeline.named_steps["scaler"]
        xgb_clf = xgb_pipeline.named_steps["xgb"]
        X_test_scaled_for_shap = scaler_xgb.transform(X_test)

        import shap
        explainer = shap.Explainer(xgb_clf)
        shap_values = explainer(X_test_scaled_for_shap)
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=_feature_list,
            show=False,
            plot_size=(10, 6),
        )
        plt.tight_layout()
        shap_path = out_folder / "shap_summary.png"
        plt.savefig(shap_path, dpi=300)
        plt.close()
        print(f"📸  Saved  ➜  {shap_path}")

    # ── 8. Final runtime ───────────────────────────────────────────────────
    total_time = time.time()
    print(f"\n⏱️  Total run-time: {total_time - total_time:.1f}s")


if __name__ == "__main__":
    main()
