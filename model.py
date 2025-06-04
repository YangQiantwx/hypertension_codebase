from typing import Dict, Any, Tuple, Optional
import os
import random

import keras_tuner as kt
import numpy as np
import tensorflow as tf
import xgboost as xgb
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# --------------------------------------------------------------------------- #
#  GLOBAL SEED – deterministic runs across NumPy / TF / Python / XGB
# --------------------------------------------------------------------------- #
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# XGBoost ≥ 2.0 no longer has “random_state” in set_config; keep only verbosity
xgb.set_config(verbosity=0)

# --------------------------------------------------------------------------- #
# 1.  XGBoost (fixed-hyper-param, fast baseline)
# --------------------------------------------------------------------------- #
def build_xgb_pipeline(
    scale_pos_weight: float,
    sampling_strategy: Optional[float] = 0.60,  # if None → skip ADASYN
    max_depth: int = 5,
    learning_rate: float = 0.05,
    n_estimators: int = 150,
    n_jobs: int = -1,
    random_state: int = SEED,
):
    """
    One-shot XGB classifier wrapped in (StandardScaler → [optional ADASYN] → XGBClassifier).
    If sampling_strategy is None, the ADASYN step is omitted.
    """
    steps = [("scaler", StandardScaler())]
    if sampling_strategy is not None:
        steps.append(("adasyn", ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)))
    steps.append(
        (
            "xgb",
            xgb.XGBClassifier(
                random_state=random_state,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                scale_pos_weight=scale_pos_weight,
                n_jobs=n_jobs,
            ),
        )
    )
    return ImbPipeline(steps)


# --------------------------------------------------------------------------- #
# 2.  XGBoost + GridSearchCV  (search over ADASYN sampling + XGB hyperparams)
# --------------------------------------------------------------------------- #
def fit_xgb_with_grid(
    X_train,
    y_train,
    scale_pos_weight: float,
    sampling_strategy: float = 0.50,  # serves as a default, but grid overrides
    n_jobs: int = -1,
    random_state: int = SEED,
):
    """
    Recreates the original 3-fold grid search over:
      - adasyn__sampling_strategy ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
      - xgb__max_depth ∈ {3, 5, 7}
      - xgb__learning_rate ∈ {0.01, 0.05, 0.1}
      - xgb__n_estimators ∈ {100, 150, 200}
    Returns the best estimator already fitted.
    """
    print("\n⚙️  XGBoost grid search …")

    base = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("adasyn", ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)),
            ("xgb", xgb.XGBClassifier(
                random_state=random_state,
                scale_pos_weight=scale_pos_weight,
                n_jobs=n_jobs,
            )),
        ]
    )

    param_grid = {
        "adasyn__sampling_strategy": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "xgb__max_depth": [3, 5, 7],
        "xgb__learning_rate": [0.01, 0.05, 0.1],
        "xgb__n_estimators": [100, 150, 200],
    }

    gs = GridSearchCV(
        base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=n_jobs,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    print("   ⇢ best params:", gs.best_params_)
    print(f"   ⇢ best AUROC : {gs.best_score_:.3f}")
    return gs.best_estimator_


# --------------------------------------------------------------------------- #
# 3.  Feature-attention network (tabular Transformer-lite)
# --------------------------------------------------------------------------- #
class FeatureAttentionHyperModel(kt.HyperModel):
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def build(self, hp):
        inputs = Input(shape=(self.input_dim,))

        # ── First dense block ─────────────────────────────────────────
        x = Dense(
            hp.Choice("dense1", [32, 64, 128, 256]),
            activation="relu"
        )(inputs)
        x = Dropout(hp.Float("drop1", 0.1, 0.5, step=0.1))(x)

        # ── Second dense block ────────────────────────────────────────
        x = Dense(
            hp.Choice("dense2", [16, 32, 64, 128]),
            activation="relu"
        )(x)
        x = Dropout(hp.Float("drop2", 0.1, 0.5, step=0.1))(x)

        # ── Attention hyperparameters: num_heads & key_dim ─────────────
        num_heads = hp.Choice("num_heads", [1, 2, 4, 8])
        key_dim = hp.Choice("key_dim", [4, 8, 16, 32])
        x_exp = tf.expand_dims(x, axis=1)
        x_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim
        )(x_exp, x_exp)
        x_pool = tf.reduce_mean(x_attn, axis=1)

        # ── Final dense + regularization ───────────────────────────────
        x = Dense(
            hp.Choice("final", [16, 32, 64, 128]),
            activation="relu",
            kernel_regularizer=l2(hp.Choice("reg", [0.0, 1e-4, 1e-3, 1e-2]))
        )(x_pool)
        x = Dropout(hp.Float("drop3", 0.1, 0.5, step=0.1))(x)

        # ── Output & compile ──────────────────────────────────────────
        lr = hp.Choice("lr", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5])
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(lr),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")]
        )
        return model


def fit_attention_model(
    X_train_res,
    y_train_res,
    X_val,
    y_val,
    *,
    max_trials: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 1,
) -> Tuple[Model, Dict[str, Any]]:
    """Hyper-parameter search wrapper with deterministic seeding."""
    gpus = tf.config.list_physical_devices("GPU")
    print(f"🖥️  TensorFlow sees {len(gpus)} GPU(s): {[g.name for g in gpus]}")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train_res)
    cw_dict = {0: cw[0], 1: cw[1]}

    tuner = kt.RandomSearch(
        FeatureAttentionHyperModel(input_dim=X_train_res.shape[1]),
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=max_trials,
        executions_per_trial=1,
        directory="feature_attention_tuner",
        project_name="bp_attention",
        overwrite=True,
    )

    tuner.search(
        X_train_res,
        y_train_res,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw_dict,
        verbose=verbose,
    )

    return tuner.get_best_models(1)[0], tuner.get_best_hyperparameters(1)[0].values


# --------------------------------------------------------------------------- #
# 4.  Registry – lets train.py pick models by tag
# --------------------------------------------------------------------------- #
MODEL_REGISTRY: Dict[str, Any] = {
    "xgb": fit_xgb_with_grid,     # grid-search variant
    "xgb_fixed": build_xgb_pipeline,
    "attn": fit_attention_model,
}
