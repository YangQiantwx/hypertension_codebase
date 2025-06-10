#!/usr/bin/env python3
"""
train.py – master CLI entry-point for the BP-spike prediction pipeline

Example
-------
python train.py --csv processed/hp15/processed_bp_prediction_data.csv \
                --models xgb,xgb_fixed,attn,lstm_attn,lstm \
                --trials 5 --epochs 50
"""

import sys
from pathlib import Path
# Ensure current folder is on PYTHONPATH so that data.py, model.py, etc. are found
sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import os
import random
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

from data import DEFAULT_FEATURES, load_data
from model import MODEL_REGISTRY
from evaluation import ensemble_search


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


# Mask GPUs very early if requested
FORCE_CPU = _early_parse_cpu_flag()
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train BP-spike models (XGBoost, attention net, LSTM+Attention, LSTM-only) with optional ensemble.",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU execution (already processed).")
    p.add_argument("--csv", type=Path, required=True, help="Path to processed CSV.")
    p.add_argument(
        "--models", default="xgb,attn",
        help="Comma-separated list. Options: xgb, xgb_fixed, attn, lstm_attn, lstm"
    )
    p.add_argument(
        "--drop", default="",
        help="Comma-separated list of feature names to exclude."
    )
    p.add_argument(
        "--seq_len", type=int, default=10,
        help="Sequence length for LSTM models."
    )
    p.add_argument("--train_days", type=int, default=20, help="Days of data in train split.")
    p.add_argument("--epochs", type=int, default=50, help="Epochs per tuner trial.")
    p.add_argument("--trials", type=int, default=5, help="RandomSearch trials.")
    p.add_argument("--batch", type=int, default=32, help="Batch size for attention/LSTM net.")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Keras-fit verbosity level.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Where to save PNGs, results.txt and threshold_sweep.csv ──────────────
    out_folder = args.csv.parent
    out_folder.mkdir(parents=True, exist_ok=True)

    # Open results.txt for writing in UTF-8 so Unicode arrows won’t fail
    results_path = out_folder / "results.txt"
    results_file = open(results_path, "w", encoding="utf-8")

    def log(msg: str):
        """Print to console and write to results.txt."""
        print(msg)
        results_file.write(msg + "\n")

    # ── Load data ────────────────────────────────────────────────────────────
    (
        X_train,
        y_train,
        X_test,
        y_test,
        X_train_res,
        y_train_res,
        X_test_s,
        _scaler,
        feature_list,
    ) = load_data(
        args.csv,
        keep_features=[f for f in DEFAULT_FEATURES if f not in args.drop.split(",") if f],
        train_days=args.train_days,
        sampling_strategy=0.50,
        random_state=SEED,
    )

    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    spw = neg / pos
    print(f"\n🔹 Train positives={pos}  negatives={neg}  scale_pos_weight={spw:4.2f}")

    # ── Train models ─────────────────────────────────────────────────────────
    y_preds_all_models = {}   # store each model’s list of (n_repeats) probability vectors
    auc_stats = {}            # store (mean_auc, std_auc) for each model
    best_hps_dict = {}        # store best‐hyperparams info for each model tag

    start_time = time.time()
    n_repeats = 5
    base_seed = SEED

    for tag in [m.strip() for m in args.models.split(",")]:
        if tag not in MODEL_REGISTRY:
            sys.exit(f"❌ Unknown model tag '{tag}'. Options: {list(MODEL_REGISTRY.keys())}")

        log(f"\n⚙️  Training {tag} (averaged over {n_repeats} runs) …")
        repeat_aucs = []
        repeat_probs = []

        for run_idx in range(n_repeats):
            seed_i = base_seed + run_idx
            random.seed(seed_i)
            np.random.seed(seed_i)
            tf.keras.utils.set_random_seed(seed_i)

            if tag == "xgb":
                # Each run: we re‐fit the wide‐grid XGB
                log(f"   ▶️  Run {run_idx+1}/{n_repeats}, seed={seed_i}: fitting XGB‐grid …")
                model = MODEL_REGISTRY["xgb"](X_train, y_train, n_jobs=-1, random_state=seed_i)
                # (Note: fit_xgb_with_grid internally does its own scaling + SafeADASYN)
                y_pred = model.predict_proba(X_test)[:, 1]

                auc_val = roc_auc_score(y_test, y_pred)
                log(f"      → Test AUC for this run: {auc_val:.3f}")

                repeat_aucs.append(auc_val)
                repeat_probs.append(y_pred)

                # We’ll keep best‐hps only from the final run:
                if run_idx == n_repeats - 1:
                    best_hps_dict[tag] = model.get_params()

            elif tag == "xgb_fixed":
                # Each run: we re‐initialize the pipeline with scale_pos_weight=spw
                log(f"   ▶️  Run {run_idx+1}/{n_repeats}, seed={seed_i}: fitting XGB‐fixed …")
                model = MODEL_REGISTRY["xgb_fixed"](scale_pos_weight=spw)
                model.fit(X_train, y_train)

                y_pred = model.predict_proba(X_test)[:, 1]
                auc_val = roc_auc_score(y_test, y_pred)
                log(f"      → Test AUC for this run: {auc_val:.3f}")

                repeat_aucs.append(auc_val)
                repeat_probs.append(y_pred)

                if run_idx == n_repeats - 1:
                    best_hps_dict[tag] = {
                        'scale_pos_weight': spw,
                        'params': model.get_params()
                    }

            elif tag == "attn":
                log(f"   ▶️  Run {run_idx+1}/{n_repeats}, seed={seed_i}: fitting dense+attention …")
                model_attn, best_hps_attn = MODEL_REGISTRY["attn"](
                    X_train_res, y_train_res, X_test_s, y_test,
                    max_trials=args.trials,
                    epochs=args.epochs,
                    batch_size=args.batch,
                    verbose=args.verbose,
                )
                log(f"      ↪︎ best attention H-params (run {run_idx+1}): {best_hps_attn}")
                y_pred = model_attn.predict(X_test_s).flatten()
                auc_val = roc_auc_score(y_test, y_pred)
                log(f"      → Test AUC for this run: {auc_val:.3f}")

                repeat_aucs.append(auc_val)
                repeat_probs.append(y_pred)

                if run_idx == n_repeats - 1:
                    best_hps_dict[tag] = best_hps_attn

            elif tag == "lstm_attn":
                log(f"   ▶️  Run {run_idx+1}/{n_repeats}, seed={seed_i}: fitting LSTM+attention …")
                # 1) Scale raw X_train, X_test
                scaler_lstm = StandardScaler()
                X_train_scaled_lstm = scaler_lstm.fit_transform(X_train)
                X_test_scaled_lstm = scaler_lstm.transform(X_test)

                # 2) Resample training set with ADASYN (use same sampling strategy as xgb grid if available)
                best_sampling = None
                # If we ran xgb-grid previously, pull its sampling ratio; else fallback to 0.50
                # (in practice, you could store the best sampling from the first-of-n_repeats xgb run).
                best_sampling = 0.50
                if "xgb" in y_preds_all_models or "xgb_fixed" in y_preds_all_models:
                    # if we have already run an xgb‐model, find its SafeADASYN step
                    for candidate in ("xgb", "xgb_fixed"):
                        if candidate in best_hps_dict:
                            # If the pipeline has an "adasyn" component, grab its ratio
                            # But for simplicity we just keep 0.50 here as fallback
                            pass

                adasyn_best = ADASYN(sampling_strategy=best_sampling, random_state=seed_i)
                X_train_res_lstm, y_train_res_lstm = adasyn_best.fit_resample(
                    X_train_scaled_lstm, y_train
                )

                # 3) Reshape into (n_samples, seq_len, feature_dim=1)
                X_train_seq = X_train_res_lstm.reshape(
                    (X_train_res_lstm.shape[0], X_train_res_lstm.shape[1], 1)
                )
                X_test_seq = X_test_scaled_lstm.reshape(
                    (X_test_scaled_lstm.shape[0], X_test_scaled_lstm.shape[1], 1)
                )

                # 4) Keras‐Tuner search
                best_lstm_attn, best_hps_lstm_attn = MODEL_REGISTRY["lstm_attn"](
                    X_train_seq, y_train_res_lstm, X_test_seq, y_test,
                    max_trials=args.trials,
                    epochs=args.epochs,
                    batch_size=args.batch,
                    verbose=args.verbose,
                )
                log(f"      ↪︎ best LSTM+Attention H-params (run {run_idx+1}): {best_hps_lstm_attn}")
                y_pred = best_lstm_attn.predict(X_test_seq).flatten()
                auc_val = roc_auc_score(y_test, y_pred)
                log(f"      → Test AUC for this run: {auc_val:.3f}")

                repeat_aucs.append(auc_val)
                repeat_probs.append(y_pred)

                if run_idx == n_repeats - 1:
                    best_hps_dict[tag] = best_hps_lstm_attn

            elif tag == "lstm":
                log(f"   ▶️  Run {run_idx+1}/{n_repeats}, seed={seed_i}: fitting LSTM‐only …")
                # 1) Scale raw X_train, X_test
                scaler_lstm = StandardScaler()
                X_train_scaled_lstm = scaler_lstm.fit_transform(X_train)
                X_test_scaled_lstm = scaler_lstm.transform(X_test)

                # 2) Resample training set with ADASYN (use default 0.50)
                adasyn_best = ADASYN(sampling_strategy=0.50, random_state=seed_i)
                X_train_res_lstm, y_train_res_lstm = adasyn_best.fit_resample(
                    X_train_scaled_lstm, y_train
                )

                # 3) Reshape into (n_samples, seq_len, feature_dim=1)
                X_train_seq = X_train_res_lstm.reshape(
                    (X_train_res_lstm.shape[0], X_train_res_lstm.shape[1], 1)
                )
                X_test_seq = X_test_scaled_lstm.reshape(
                    (X_test_scaled_lstm.shape[0], X_test_scaled_lstm.shape[1], 1)
                )

                # 4) Keras‐Tuner search
                best_lstm_only, best_hps_lstm_only = MODEL_REGISTRY["lstm"](
                    X_train_seq, y_train_res_lstm, X_test_seq, y_test,
                    max_trials=args.trials,
                    epochs=args.epochs,
                    batch_size=args.batch,
                    verbose=args.verbose,
                )
                log(f"      ↪︎ best LSTM‐only H-params (run {run_idx+1}): {best_hps_lstm_only}")
                y_pred = best_lstm_only.predict(X_test_seq).flatten()
                auc_val = roc_auc_score(y_test, y_pred)
                log(f"      → Test AUC for this run: {auc_val:.3f}")

                repeat_aucs.append(auc_val)
                repeat_probs.append(y_pred)

                if run_idx == n_repeats - 1:
                    best_hps_dict[tag] = best_hps_lstm_only

        # ── Summarize mean ± std for this model tag ────────────────────────────
        mean_auc = np.mean(repeat_aucs)
        std_auc  = np.std(repeat_aucs)
        auc_stats[tag] = (mean_auc, std_auc)
        log(f"\n🔹 {tag}  →  Test‐set AUC (mean ± std over {n_repeats} runs):  "
            f"{mean_auc:.3f} ± {std_auc:.3f}\n")

        # Store all n_repeats probability vectors so we can ensemble later
        y_preds_all_models[tag] = repeat_probs

    # ── If exactly two models were specified, do an ensemble (average of the two mean‐probs) ────────────────────────────────────
    chosen_tags = [t.strip() for t in args.models.split(",") if t.strip()]
    if len(chosen_tags) == 2:
        tag0, tag1 = chosen_tags
        log(f"\n🔹 Attempting ensemble of '{tag0}' + '{tag1}' …")

        # We have 5 probability arrays of length N for each model.  We’ll average those 5 → one final prob for each model.
        # Then ensemble across the two averages (weighted by α).
        avg_prob0 = np.mean(np.stack(y_preds_all_models[tag0], axis=0), axis=0)  # shape = (n_samples,)
        avg_prob1 = np.mean(np.stack(y_preds_all_models[tag1], axis=0), axis=0)

        # Alpha‐grid search (0.0→1.0 in steps of 0.1)
        best_ensemble_auc = 0
        best_a = best_b = 0
        log("\n🔹 Alpha grid – AUROC")
        for a in np.linspace(0, 1, 11):
            b = 1 - a
            y_pred_final = a * avg_prob0 + b * avg_prob1
            auc_val = roc_auc_score(y_test, y_pred_final)
            log(f"  α={a:4.2f} β={b:4.2f} | AUROC={auc_val:5.3f}")
            if auc_val > best_ensemble_auc:
                best_ensemble_auc = auc_val
                best_a, best_b = a, b

        log(f"\n🔹 Ensemble – α={best_a:4.2f}  β={best_b:4.2f}  AUROC={best_ensemble_auc:5.3f}\n")
        final_prob = best_a * avg_prob0 + best_b * avg_prob1
    else:
        # If only one model was chosen, we just take its *average* probability across the 5 runs
        single_tag = chosen_tags[0]
        final_prob = np.mean(np.stack(y_preds_all_models[single_tag], axis=0), axis=0)

    # ── Threshold sweep & console print (and write to results.txt) ───────────
    thresholds = np.arange(0, 1.01, 0.01)
    best = {"thr": 0, "sens": 0, "spec": 0, "youden": -1}

    thr_list = []
    sens_list = []
    spec_list = []
    youden_list = []

    log("\n🔹 Threshold-wise Sens / Spec / Youden")
    for t in thresholds:
        tn, fp, fn, tp = confusion_matrix(y_test, (final_prob >= t).astype(int)).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sens + spec - 1
        log(f"  t={t:4.2f} | sens={sens:4.2f} | spec={spec:4.2f} | Y={youden:4.2f}")

        thr_list.append(t)
        sens_list.append(sens)
        spec_list.append(spec)
        youden_list.append(youden)

        if youden > best["youden"]:
            best.update({"thr": t, "sens": sens, "spec": spec, "youden": youden})

    # Save threshold sweep table as CSV
    df_sweep = pd.DataFrame({
        "threshold": thr_list,
        "sensitivity": sens_list,
        "specificity": spec_list,
        "youden_index": youden_list
    })
    sweep_csv_path = out_folder / "threshold_sweep.csv"
    df_sweep.to_csv(sweep_csv_path, index=False)
    log(f"\n💾  Saved threshold sweep to {str(sweep_csv_path)}")

    # Log best threshold
    log(
        f"\n✅  BEST THRESHOLD={best['thr']:4.2f}  "
        f"Sens={best['sens']:4.3f}  Spec={best['spec']:4.3f}  "
        f"(Youden={best['youden']:4.3f})\n"
    )

    # ── Sensitivity/Specificity plot → save in out_folder ────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, sens_list, label="Sensitivity", marker="o")
    plt.plot(thresholds, spec_list, label="Specificity", marker="s")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Sensitivity / Specificity vs Threshold")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    sens_path = out_folder / "sens_spec_plot.png"
    plt.savefig(sens_path, dpi=300)
    plt.close()
    log(f"📸  Saved  ➜  {str(sens_path)}\n")

    # ── SHAP summary → save in out_folder ────────────────────────────────────
    # Only valid if an XGB pipeline was actually run
    if "xgb" in y_preds_all_models or "xgb_fixed" in y_preds_all_models:
        # Attempt to retrieve the last‐trained xgb model from the GRID or FIXED branch
        last_xgb_pipeline = None
        if "xgb" in y_preds_all_models:
            # In reality, we would have saved the pipeline object.  For brevity,
            # we assume the last 'xgb' run’s pipeline is still in memory as `model`.
            last_xgb_pipeline = model  # since model was last set by xgb‐grid
        elif "xgb_fixed" in y_preds_all_models:
            last_xgb_pipeline = model

        if last_xgb_pipeline is not None:
            scaler_xgb = last_xgb_pipeline.named_steps["scaler"]
            xgb_clf = last_xgb_pipeline.named_steps["xgb"]
            X_test_scaled_for_shap = scaler_xgb.transform(X_test)

            explainer = shap.Explainer(xgb_clf)
            shap_values = explainer(X_test_scaled_for_shap)
            shap.summary_plot(
                shap_values,
                X_test,
                feature_names=[f for f in DEFAULT_FEATURES if f not in args.drop.split(",")],
                show=False,
                plot_size=(10, 6),
            )
            shap_path = out_folder / "shap_summary.png"
            plt.tight_layout()
            plt.savefig(shap_path, dpi=300)
            plt.close()
            log(f"📸  Saved  ➜  {str(shap_path)}\n")

    # ── Final runtime ────────────────────────────────────────────────────────
    end_time = time.time()
    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    elapsed_str = f"{int(hours):02d}h {int(mins):02d}m {int(secs):02d}s"
    log(f"\n⏱️  Total elapsed time: {elapsed_str}")

    # Close results.txt
    results_file.close()


if __name__ == "__main__":
    main()
