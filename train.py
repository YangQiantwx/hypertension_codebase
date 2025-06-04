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
from model import MODEL_REGISTRY, build_xgb_pipeline, fit_xgb_with_grid

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

# ── FULL ARG PARSER ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train BP-spike models (XGBoost, attention net) with optional ensemble.",
    )
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU execution (already processed).")
    p.add_argument("--csv", type=Path, required=True,
                   help="Path to processed CSV.")
    p.add_argument("--models", default="xgb,attn",
                   help="Comma-separated list. Options: xgb, xgb_fixed, attn")
    p.add_argument("--drop", default="",
                   help="Comma-separated list of feature names to exclude.")
    p.add_argument("--train_days", type=int, default=20,
                   help="Days of data in train split.")
    p.add_argument("--epochs", type=int, default=50,
                   help="Epochs per tuner trial.")
    p.add_argument("--trials", type=int, default=5,
                   help="RandomSearch trials.")
    p.add_argument("--batch", type=int, default=32,
                   help="Batch size for attention net.")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2],
                   help="Keras-fit verbosity level.")
    p.add_argument(
        "--no_adasyn",
        action="store_true",
        help="If set, do NOT use ADASYN in any pipeline. (i.e., skip oversampling.)"
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()

    # Where to save plots
    out_folder = args.csv.parent
    out_folder.mkdir(parents=True, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────────
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

    # ── Train models ────────────────────────────────────────────────────────
    y_preds = []
    xgb_pipeline = None

    for tag in [m.strip() for m in args.models.split(",")]:
        if tag == "xgb":
            print(f"\n⚙️  XGBoost grid search …")
            # If --no_adasyn, pass sampling_strategy=None → grid will not find adasyn step; 
            # otherwise grid will iterate over [0.40 … 0.70].
            sampling_arg = None if args.no_adasyn else 0.50
            model = fit_xgb_with_grid(
                X_train,
                y_train,
                scale_pos_weight=spw,
                sampling_strategy=sampling_arg,  # grid-sweep inside
                n_jobs=-1,
                random_state=SEED,
            )
            y_pred_xgb_only = model.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_auc_score
            print(f"🔹 Pure XGB Test-set AUC: {roc_auc_score(y_test, y_pred_xgb_only):.3f}")

            y_preds.append(y_pred_xgb_only)
            xgb_pipeline = model

        elif tag == "xgb_fixed":
            print(f"\n⚙️  Training xgb_fixed …")
            # If no_adasyn → sampling_strategy=None → skip ADASYN in pipeline
            pipeline = build_xgb_pipeline(
                scale_pos_weight=spw,
                sampling_strategy=None if args.no_adasyn else 0.50,
                max_depth=5,
                learning_rate=0.05,
                n_estimators=150,
            )
            pipeline.fit(X_train, y_train)
            y_pred_fixed = pipeline.predict_proba(X_test)[:, 1]
            from sklearn.metrics import roc_auc_score
            print(f"🔹 XGB_fixed Test-set AUC: {roc_auc_score(y_test, y_pred_fixed):.3f}")

            y_preds.append(y_pred_fixed)
            xgb_pipeline = pipeline

        elif tag == "attn":
            print("\n⚙️  Training attention model …")
            # Note: X_train_res / y_train_res were created with or without ADASYN above
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

        else:
            sys.exit(f"❌ Unknown model tag '{tag}'.")

    # ── Ensemble if two preds ────────────────────────────────────────────────
    if len(y_preds) == 2:
        ens = ensemble_search(y_preds[0], y_preds[1], y_test)
        final_prob = ens["a"] * y_preds[0] + ens["b"] * y_preds[1]
        print(f"\n🔹 Ensemble – α={ens['a']:4.2f}  β={ens['b']:4.2f}  AUROC={ens['auc']:5.3f}")
    else:
        final_prob = y_preds[0]

    # ── Threshold sweep & console print ─────────────────────────────────────
    best_thr = threshold_sweep(y_test, final_prob)
    print(
        f"\n✅  BEST THRESHOLD={best_thr['thr']:4.2f}  "
        f"Sens={best_thr['sens']:4.3f}  Spec={best_thr['spec']:4.3f}  "
        f"(Youden={best_thr['youden']:4.3f})"
    )

    # ── Sensitivity/Specificity plot → save in out_folder ───────────────────
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

    # ── SHAP summary → save in out_folder (if XGB was trained) ─────────────
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

    # Final runtime
    total_time = time.time()
    print(f"\n⏱️  Total run-time: {total_time - total_time:.1f}s")


if __name__ == "__main__":
    main()
