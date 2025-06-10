# model.py

from typing import Dict, Any, Tuple, Optional
import os
import random

import numpy as np
import xgboost as xgb
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LSTM,
    Bidirectional,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN

# ── GLOBAL SEED FOR REPRODUCIBILITY ─────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
xgb.set_config(verbosity=0)


# ── 1. SafeADASYN ──────────────────────────────────────────────────────────
class SafeADASYN(ADASYN):
    """
    Subclass of ADASYN that catches ValueErrors when the requested sampling
    ratio is impossible. If ADASYN fails, returns (X, y) unchanged.
    """
    def __init__(self, sampling_strategy=0.5, random_state=None):
        super().__init__(sampling_strategy=sampling_strategy, random_state=random_state)

    def fit_resample(self, X, y):
        try:
            return super().fit_resample(X, y)
        except ValueError as e:
            msg = str(e)
            if ("No samples will be generated" in msg
                    or "remove samples from the minority class" in msg):
                return X, y
            raise


# ── 2. XGBoost (fixed‐hyperparam) ───────────────────────────────────────────
def build_xgb_pipeline(
    scale_pos_weight: float,
    sampling_strategy: Optional[float] = 0.60,  # if None → skip SafeADASYN
    max_depth: int = 5,
    learning_rate: float = 0.05,
    n_estimators: int = 150,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    min_child_weight: int = 1,
    gamma: float = 0.0,
    n_jobs: int = -1,
    random_state: int = SEED,
) -> ImbPipeline:
    """
    Build a pipeline: (StandardScaler → [optional SafeADASYN] → XGBClassifier).
    If sampling_strategy is None, ADASYN is omitted entirely.
    """
    steps = [("scaler", StandardScaler())]
    if sampling_strategy is not None:
        steps.append(
            ("adasyn", SafeADASYN(sampling_strategy=sampling_strategy, random_state=random_state))
        )
    steps.append(
        (
            "xgb",
            xgb.XGBClassifier(
                random_state=random_state,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                scale_pos_weight=scale_pos_weight,
                n_jobs=n_jobs,
                use_label_encoder=False,
                eval_metric="logloss",
            ),
        )
    )
    return ImbPipeline(steps)


# ── 3. XGBoost + ADASYN GridSearchCV (wider grid) ──────────────────────────
def fit_xgb_with_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_jobs: int = -1,
    random_state: int = SEED,
) -> ImbPipeline:
    """
    Perform 3-fold grid search over:
      - adasyn__sampling_strategy ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
      - xgb__max_depth ∈ {3, 5, 7, 9}
      - xgb__learning_rate ∈ {0.005, 0.01, 0.05, 0.1, 0.2}
      - xgb__n_estimators ∈ {50, 100, 150, 200, 300}
      - xgb__subsample ∈ {0.6, 0.8, 1.0}
      - xgb__colsample_bytree ∈ {0.6, 0.8, 1.0}
      - xgb__min_child_weight ∈ {1, 5, 10}
      - xgb__gamma ∈ {0.0, 0.1, 0.2}
      - xgb__scale_pos_weight = neg/pos  (computed internally)
    Returns the best‐fitted ImbPipeline.
    """
    # Compute scale_pos_weight = (#neg)/(#pos)
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    scale_pos_weight = neg / pos
    print(f"🔹 Computed scale_pos_weight: {scale_pos_weight:.2f}")

    # Build the base pipeline
    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("adasyn", SafeADASYN(random_state=random_state)),
            (
                "xgb",
                xgb.XGBClassifier(
                    random_state=random_state,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    # Define the grid (wider than before)
    param_grid = {
        "adasyn__sampling_strategy": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "xgb__max_depth": [3, 5, 7, 9],
        "xgb__learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
        "xgb__n_estimators": [50, 100, 150, 200, 300],
        "xgb__subsample": [0.6, 0.8, 1.0],
        "xgb__colsample_bytree": [0.6, 0.8, 1.0],
        "xgb__min_child_weight": [1, 5, 10],
        "xgb__gamma": [0.0, 0.1, 0.2],
        "xgb__scale_pos_weight": [scale_pos_weight],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=n_jobs,
        verbose=2,
        error_score=np.nan,
    )
    grid_search.fit(X_train, y_train)

    print("🔹 Best parameters from GridSearchCV (XGBoost pipeline):")
    print(grid_search.best_params_)

    return grid_search.best_estimator_  # This is an ImbPipeline


# ── 4. Dense + MultiHeadAttention HyperModel ───────────────────────────────
class FeatureAttentionHyperModel(kt.HyperModel):
    """
    Dense → MultiHeadAttention → Dense architecture with hyperparameter search.
    """
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def build(self, hp):
        inputs = Input(shape=(self.input_dim,))

        # First dense block
        x = Dense(hp.Choice("dense1", [32, 64, 128]), activation="relu")(inputs)
        x = Dropout(hp.Float("drop1", 0.1, 0.5, step=0.1))(x)

        # Second dense block
        x = Dense(hp.Choice("dense2", [16, 32, 64]), activation="relu")(x)
        x = Dropout(hp.Float("drop2", 0.1, 0.5, step=0.1))(x)

        # MultiHead Attention
        num_heads = hp.Choice("num_heads", [1, 2, 4])
        key_dim = hp.Choice("key_dim", [8, 16, 32])
        x_exp = tf.expand_dims(x, axis=1)  # (batch, 1, features)
        x_attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_exp, x_exp)
        x_pool = tf.reduce_mean(x_attn, axis=1)  # (batch, features)

        # Final dense + regularization
        x = Dense(
            hp.Choice("final", [16, 32, 64]),
            activation="relu",
            kernel_regularizer=l2(hp.Choice("reg", [0.0, 1e-4, 1e-3])),
        )(x_pool)
        x = Dropout(hp.Float("drop3", 0.1, 0.5, step=0.1))(x)

        # Output & compile
        lr = hp.Choice("lr", [1e-2, 5e-3, 1e-3, 5e-4])
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(lr),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model


def fit_attention_model(
    X_train_res: np.ndarray,
    y_train_res: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_trials: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 1,
) -> Tuple[Model, Dict[str, Any]]:
    """
    Hyperparameter search for the dense + MultiHeadAttention model.
    X_train_res / y_train_res should be scaled & resampled (e.g., via ADASYN).
    X_val / y_val should be scaled but not resampled.
    """
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

    best_model = tuner.get_best_models(1)[0]
    best_hps = tuner.get_best_hyperparameters(1)[0].values
    return best_model, best_hps


# ── 5. FOUR Attention‐Variant Layers ────────────────────────────────────────
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        # input_shape = (batch, time_steps, features)
        self.W = self.add_weight(
            name="W_attn",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform"
        )
        self.U = self.add_weight(
            name="U_attn",
            shape=(input_shape[-1],),
            initializer="zeros"
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time, features)
        e = tf.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.U)
        alpha = tf.nn.softmax(tf.reduce_sum(e, axis=2, keepdims=True), axis=1)  # (batch, time, 1)
        context = tf.reduce_sum(alpha * inputs, axis=1)  # weighted sum → (batch, features)
        return context


class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.Wq = self.add_weight(
            name="Wq", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform"
        )
        self.Wk = self.add_weight(
            name="Wk", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform"
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, time, features)
        Q = tf.tensordot(inputs, self.Wq, axes=[[2], [0]])  # (batch, time, features)
        K = tf.tensordot(inputs, self.Wk, axes=[[2], [0]])  # (batch, time, features)
        scores = tf.matmul(Q, K, transpose_b=True)  # (batch, time, time)
        weights = tf.nn.softmax(scores, axis=-1)    # (batch, time, time)
        attended = tf.matmul(weights, inputs)       # (batch, time, features)
        context = tf.reduce_mean(attended, axis=1)  # pool along time → (batch, features)
        return context


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, key_dim: int, ff_dim: int, rate: float = 0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(key_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)  # (batch, time, key_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # (batch, time, key_dim)
        ffn_output = self.ffn(out1)                   # (batch, time, key_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)     # (batch, time, key_dim)


# ── 6. LSTM + (Choice of 4) Attention HyperModel ────────────────────────────
class LSTMAttentionHyperModel(kt.HyperModel):
    """
    Sequence-based model that lets KerasTuner pick among four attention‐variants:
      1) "custom"        → AttentionLayer (Bahdanau‐style)
      2) "multihead"     → Keras’s MultiHeadAttention
      3) "selfattention"→ SelfAttentionLayer (single‐head dot-product)
      4) "transformer"   → TransformerBlock (multihead + feed‐forward)

    Input shape: (seq_len, feature_dim).
    """
    def __init__(self, seq_len: int, feature_dim: int):
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def build(self, hp):
        inputs = Input(shape=(self.seq_len, self.feature_dim))

        # ── 1st Bi-LSTM block
        x = Bidirectional(
            LSTM(hp.Choice("lstm1_units", [64, 128, 256]), return_sequences=True),
            merge_mode="concat",
        )(inputs)
        x = Dropout(hp.Float("drop1", 0.2, 0.5, step=0.1))(x)

        # ── 2nd LSTM block (always return_sequences=True)
        x = LSTM(hp.Choice("lstm2_units", [32, 64, 128]), return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(hp.Float("drop2", 0.2, 0.5, step=0.1))(x)

        # ── Choose attention variant
        att_variant = hp.Choice(
            "attention_variant",
            ["custom", "multihead", "selfattention", "transformer"],
        )

        if att_variant == "custom":
            x_att = AttentionLayer()(x)          # (batch, features)
        elif att_variant == "multihead":
            mh_heads = hp.Choice("mh_num_heads", [1, 2, 4])
            mh_key   = hp.Choice("mh_key_dim", [16, 32, 64])
            x_mh     = MultiHeadAttention(num_heads=mh_heads, key_dim=mh_key)(x, x)
            x_att    = tf.reduce_mean(x_mh, axis=1)  # pool over time → (batch, features)
        elif att_variant == "selfattention":
            x_att = SelfAttentionLayer()(x)      # (batch, features)
        else:  # "transformer"
            t_heads = hp.Choice("t_num_heads", [1, 2, 4])
            t_key   = hp.Choice("t_key_dim", [16, 32, 64])
            ff_dim  = hp.Choice("t_ff_dim", [32, 64, 128])
            rate    = hp.Float("t_dropout", 0.1, 0.5, step=0.1)
            x_trans = TransformerBlock(
                num_heads=t_heads,
                key_dim=t_key,
                ff_dim=ff_dim,
                rate=rate,
            )(x)                           # (batch, time, key_dim)
            x_att  = tf.reduce_mean(x_trans, axis=1)  # (batch, key_dim)

        # ── Final dense block
        dense_units = hp.Choice("dense_units", [16, 32, 64])
        dense_reg   = hp.Choice("dense_reg", [0.0, 1e-4, 1e-3])
        x_out = Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=l2(dense_reg),
        )(x_att)
        x_out = Dropout(hp.Float("drop3", 0.2, 0.5, step=0.1))(x_out)

        # ── Output & compile
        lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])
        outputs = Dense(1, activation="sigmoid")(x_out)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(lr),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model


def fit_lstm_attention_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    *,
    max_trials: int = 20,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 1,
) -> Tuple[Model, Dict[str, Any]]:
    """
    Hyperparameter search for the LSTM+AttentionHyperModel (all four variants).
    X_train_seq : shape (n_samples, seq_len, feature_dim)
    y_train_seq : shape (n_samples,)
    X_val_seq, y_val_seq similarly for validation.
    """
    gpus = tf.config.list_physical_devices("GPU")
    print(f"🖥️  TensorFlow sees {len(gpus)} GPU(s): {[g.name for g in gpus]}")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train_seq)
    cw_dict = {0: cw[0], 1: cw[1]}

    seq_len     = X_train_seq.shape[1]
    feature_dim = X_train_seq.shape[2]

    tuner = kt.RandomSearch(
        LSTMAttentionHyperModel(seq_len=seq_len, feature_dim=feature_dim),
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=max_trials,
        executions_per_trial=1,
        directory="lstm_attention_tuner",
        project_name="bp_lstm_attn",
        overwrite=True,
    )

    tuner.search(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw_dict,
        verbose=verbose,
    )

    best_model = tuner.get_best_models(1)[0]
    best_hps   = tuner.get_best_hyperparameters(1)[0].values
    return best_model, best_hps


# ── 6. LSTM‐Only HyperModel (no attention) ──────────────────────────────────
class LSTMOnlyHyperModel(kt.HyperModel):
    """
    A “pure” LSTM model: two stacked Bi-LSTM layers + a small Dense block at the end.
    Input shape: (seq_len, feature_dim).
    """
    def __init__(self, seq_len: int, feature_dim: int):
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def build(self, hp):
        inputs = Input(shape=(self.seq_len, self.feature_dim))

        # 1st Bi-LSTM
        x = Bidirectional(
            LSTM(hp.Choice("lstm1_units", [64, 128, 256]), return_sequences=True),
            merge_mode="concat",
        )(inputs)
        x = Dropout(hp.Float("drop1", 0.2, 0.5, step=0.1))(x)

        # 2nd Bi-LSTM (return_sequences=False → last hidden only)
        x = Bidirectional(
            LSTM(hp.Choice("lstm2_units", [32, 64, 128]), return_sequences=False),
            merge_mode="concat",
        )(x)
        x = LayerNormalization()(x)
        x = Dropout(hp.Float("drop2", 0.2, 0.5, step=0.1))(x)

        # Final Dense + reg
        dense_units = hp.Choice("dense_units", [16, 32, 64, 128])
        dense_reg = hp.Choice("dense_reg", [0.0, 1e-4, 1e-3])
        x = Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=l2(dense_reg),
        )(x)
        x = Dropout(hp.Float("drop3", 0.2, 0.5, step=0.1))(x)

        # Output & compile
        lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])
        outputs = Dense(1, activation="sigmoid")(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(lr),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model


def fit_lstm_only_model(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    *,
    max_trials: int = 20,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 1,
) -> Tuple[Model, Dict[str, Any]]:
    """
    Hyperparameter search for the “pure” LSTM (no attention).
    X_train_seq : shape (n_samples, seq_len, feature_dim)
    y_train_seq : shape (n_samples,)
    X_val_seq, y_val_seq similarly for validation.
    """
    gpus = tf.config.list_physical_devices("GPU")
    print(f"🖥️  TensorFlow sees {len(gpus)} GPU(s): {[g.name for g in gpus]}")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train_seq)
    cw_dict = {0: cw[0], 1: cw[1]}

    seq_len     = X_train_seq.shape[1]
    feature_dim = X_train_seq.shape[2]

    tuner = kt.RandomSearch(
        LSTMOnlyHyperModel(seq_len=seq_len, feature_dim=feature_dim),
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=max_trials,
        executions_per_trial=1,
        directory="lstm_only_tuner",
        project_name="bp_lstm_only",
        overwrite=True,
    )

    tuner.search(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw_dict,
        verbose=verbose,
    )

    best_model = tuner.get_best_models(1)[0]
    best_hps = tuner.get_best_hyperparameters(1)[0].values
    return best_model, best_hps


# ── 7. MODEL REGISTRY ──────────────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, Any] = {
    "xgb": fit_xgb_with_grid,             # ADASYN + XGB GridSearchCV (wide grid)
    "xgb_fixed": build_xgb_pipeline,      # fixed-parameter XGB pipeline
    "attn": fit_attention_model,          # dense + MultiHeadAttention
    "lstm_attn": fit_lstm_attention_model,  # LSTM + (choice of four) Attention
    "lstm": fit_lstm_only_model,          # Pure LSTM (no attention)
}
