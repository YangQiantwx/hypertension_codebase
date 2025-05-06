# evaluation.py
"""
Shared evaluation helpers – ROC‑AUC, threshold sweep, nice console prints.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def threshold_sweep(y_true, y_prob, step: float = 0.01):
    thresholds = np.arange(0, 1.0 + step, step)
    best = {"thr": 0, "sens": 0, "spec": 0, "youden": -1}

    print("\n🔹 Threshold‑wise Sens / Spec / Youden")
    for t in thresholds:
        tn, fp, fn, tp = confusion_matrix(
            y_true, (y_prob >= t).astype(int)
        ).ravel()
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        youden = sens + spec - 1
        print(f"  t={t:4.2f} | sens={sens:4.2f} | spec={spec:4.2f} | Y={youden:4.2f}")

        if youden > best["youden"]:
            best.update({"thr": t, "sens": sens, "spec": spec, "youden": youden})
    return best


def ensemble_search(y_xgb, y_attn, y_true, alphas=None):
    alphas = alphas or np.linspace(0, 1, 11)
    best = {"a": 0, "b": 1, "auc": 0}

    print("\n🔹 Alpha grid – AUROC")
    for a in alphas:
        b = 1 - a
        y_prob = a * y_xgb + b * y_attn
        auc = roc_auc_score(y_true, y_prob)
        print(f"  α={a:4.2f} β={b:4.2f} | AUROC={auc:5.3f}")
        if auc > best["auc"]:
            best.update({"a": a, "b": b, "auc": auc})

    return best
