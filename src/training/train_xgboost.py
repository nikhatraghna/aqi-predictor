"""XGBoost — anti-overfit config, leakage-free CV, early stopping, R²-gap overfitting verdict."""

import os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_engineering.feature_selector import select_for_tree_models

FEATURE_PATH = "data/processed/islamabad_features.parquet"
MODEL_DIR    = "models/saved_models"
TARGET       = "pm25"
RANDOM_STATE = 42
K_FEATURES   = 10
N_SPLITS     = 3

XGB_PARAMS = dict(
    n_estimators          = 4000,
    learning_rate         = 0.02,
    max_depth             = 2,
    subsample             = 0.6,
    colsample_bytree      = 0.5,
    min_child_weight      = 20,
    reg_alpha             = 3.0,
    reg_lambda            = 15.0,
    gamma                 = 1.0,
    early_stopping_rounds = 50,
    objective             = "reg:squarederror",
    random_state          = RANDOM_STATE,
    n_jobs                = -1,
)


def rmse(a, b): return float(mean_squared_error(a, b) ** 0.5)

def get_metrics(y, p):
    return {"mae": round(float(mean_absolute_error(y, p)), 4),
            "rmse": round(rmse(y, p), 4),
            "r2": round(float(r2_score(y, p)), 4)}

def classify_gap(g):
    """Overfitting verdict based on train-val R² gap (the robust metric)."""
    if g <= 0.03: return "EXCELLENT ✅ (no meaningful overfitting)"
    if g <= 0.05: return "VERY GOOD ✅"
    if g <= 0.10: return "ACCEPTABLE ⚠️"
    return "OVERFITTING ❌"

def save_metrics(m):
    p = Path("models/metrics"); p.mkdir(parents=True, exist_ok=True)
    with open(p / "xgboost.json", "w") as f: json.dump(m, f, indent=4)
    print(f"[SUCCESS] Metrics saved → {p}/xgboost.json")


def main():
    print("\n==============================\n XGBOOST (ANTI-OVERFIT)\n==============================")

    # ── Load + sort ──────────────────────────────────────────────────────
    df = pd.read_parquet(FEATURE_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    assert df["datetime"].is_monotonic_increasing, "Data not sorted chronologically"

    drop_cols = ["datetime", TARGET]
    all_cols = [c for c in df.columns if c not in drop_cols]
    df = df.dropna(subset=all_cols + [TARGET]).reset_index(drop=True)
    X_all, y_all = df[all_cols], df[TARGET]

    # ── Holdout test ─────────────────────────────────────────────────────
    TEST_SIZE = max(int(0.15 * len(X_all)), 24)
    dev_end = len(X_all) - TEST_SIZE
    X_dev, y_dev = X_all.iloc[:dev_end], y_all.iloc[:dev_end]
    X_test, y_test = X_all.iloc[dev_end:], y_all.iloc[dev_end:]
    print(f"[INFO] Development : {len(X_dev)} rows  |  Final test : {len(X_test)} rows")

    # ── Leakage-free CV (nested feature selection + per-fold early stop) ──
    print(f"\n[INFO] {N_SPLITS}-fold TimeSeriesSplit CV (nested feature selection)...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    folds = []
    for fold, (tr, va) in enumerate(tscv.split(X_dev), 1):
        Xtr, ytr = X_dev.iloc[tr], y_dev.iloc[tr]
        Xva, yva = X_dev.iloc[va], y_dev.iloc[va]
        feats = select_for_tree_models(Xtr, ytr, k=K_FEATURES)
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(Xtr[feats], ytr, eval_set=[(Xva[feats], yva)], verbose=False)
        tr_pred, va_pred = m.predict(Xtr[feats]), m.predict(Xva[feats])
        tr_r, va_r = rmse(ytr, tr_pred), rmse(yva, va_pred)
        folds.append({
            "fold": fold,
            "train_rmse": round(tr_r, 4), "val_rmse": round(va_r, 4),
            "train_r2": round(float(r2_score(ytr, tr_pred)), 4),
            "val_r2":   round(float(r2_score(yva, va_pred)), 4),
            "ratio": round(va_r / tr_r, 4),
            "best_iter": int(m.best_iteration),
        })
        print(f"  Fold {fold}: train R²={folds[-1]['train_r2']:.3f}  "
              f"val R²={folds[-1]['val_r2']:.3f}  "
              f"val RMSE={va_r:.3f}  ratio={va_r/tr_r:.2f}x  best_iter={m.best_iteration}")

    # ── Overfitting verdict via R² gap (robust); RMSE ratio informational ─
    mean_train_r2 = float(np.mean([f["train_r2"] for f in folds]))
    mean_val_r2   = float(np.mean([f["val_r2"]   for f in folds]))
    r2_gap        = mean_train_r2 - mean_val_r2
    median_ratio  = float(np.median([f["ratio"] for f in folds]))

    print("\n------------------------------")
    print(f"  CV mean train R²   : {mean_train_r2:.4f}")
    print(f"  CV mean val   R²   : {mean_val_r2:.4f}")
    print(f"  R² gap (train-val) : {r2_gap:.4f}  →  {classify_gap(r2_gap)}")
    print(f"  RMSE ratio (median): {median_ratio:.2f}x  (informational only — "
          f"inflated by very low train RMSE)")

    # ── Final production model (early stop on dev tail) ──────────────────
    print("\n[INFO] Final model on full development set...")
    FEATURE_COLS = select_for_tree_models(X_dev, y_dev, k=K_FEATURES)
    print(f"[INFO] Final features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    inner = max(int(0.15 * len(X_dev)), 24)
    Xtr_f, Xva_f = X_dev[FEATURE_COLS].iloc[:-inner], X_dev[FEATURE_COLS].iloc[-inner:]
    ytr_f, yva_f = y_dev.iloc[:-inner], y_dev.iloc[-inner:]
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], verbose=False)
    print(f"[INFO] Best iteration: {model.best_iteration} (ceiling {XGB_PARAMS['n_estimators']})")
    print("[SUCCESS] Early stopping triggered ✅" if model.best_iteration < XGB_PARAMS["n_estimators"] - 1
          else "[WARNING] Early stopping never triggered")

    train_m = get_metrics(y_dev,  model.predict(X_dev[FEATURE_COLS]))
    test_m  = get_metrics(y_test, model.predict(X_test[FEATURE_COLS]))
    final_gap = train_m["r2"] - test_m["r2"]
    print(f"\n  Final TRAIN : MAE={train_m['mae']}  RMSE={train_m['rmse']}  R²={train_m['r2']}")
    print(f"  Final TEST  : MAE={test_m['mae']}  RMSE={test_m['rmse']}  R²={test_m['r2']}")
    print(f"  Test R² gap (train-test): {final_gap:.4f}  →  {classify_gap(final_gap)}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature Importance:")
    for feat, imp in importances.items():
        print(f"  {feat:<22} {imp:.4f}  {'█' * int(imp * 100)}")

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/xgboost_model.pkl")
    print(f"\n[SUCCESS] Model saved → {MODEL_DIR}/xgboost_model.pkl")

    save_metrics({
        "cv": {
            "mean_train_r2": round(mean_train_r2, 4),
            "mean_val_r2":   round(mean_val_r2, 4),
            "r2_gap":        round(r2_gap, 4),
            "verdict":       classify_gap(r2_gap),
            "rmse_ratio_median": round(median_ratio, 4),
            "n_splits":      N_SPLITS,
            "folds":         folds,
        },
        "train": train_m,
        "test":  test_m,
        "test_r2_gap": round(final_gap, 4),
        "best_iteration": int(model.best_iteration),
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
    })
    print("\n[SUCCESS] XGBoost pipeline complete.")


if __name__ == "__main__":
    main()
