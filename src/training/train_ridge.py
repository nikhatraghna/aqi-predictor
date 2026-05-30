"""Ridge — TimeSeriesSplit CV (no fixed val split), nested feature selection, R²-gap verdict."""

import os, json, joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from src.feature_engineering.feature_selector import select_for_ridge

FEATURE_PATH = "data/processed/islamabad_features.parquet"
MODEL_DIR    = "models/saved_models"
MODEL_PATH   = f"{MODEL_DIR}/ridge_model.pkl"
SCALER_PATH  = f"{MODEL_DIR}/ridge_scaler.pkl"
TARGET       = "pm25"
K_FEATURES   = 12
N_SPLITS     = 3
ALPHA_GRID   = {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0]}


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

def tune_alpha(X_sc, y):
    """Pick best alpha via inner TimeSeriesSplit grid search."""
    grid = GridSearchCV(Ridge(), ALPHA_GRID, cv=TimeSeriesSplit(n_splits=3),
                        scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_sc, y)
    return grid.best_params_["alpha"]

def save_metrics(m):
    p = Path("models/metrics"); p.mkdir(parents=True, exist_ok=True)
    with open(p / "ridge.json", "w") as f: json.dump(m, f, indent=4)
    print(f"[SUCCESS] Metrics saved → {p}/ridge.json")


def main():
    print("\n==============================\n RIDGE TRAINING PIPELINE\n==============================")

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

    # ── Leakage-free CV: nested feature selection + scaling + alpha tuning ─
    print(f"\n[INFO] {N_SPLITS}-fold TimeSeriesSplit CV (nested feature selection)...")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    folds = []
    for fold, (tr, va) in enumerate(tscv.split(X_dev), 1):
        Xtr, ytr = X_dev.iloc[tr], y_dev.iloc[tr]
        Xva, yva = X_dev.iloc[va], y_dev.iloc[va]

        feats = select_for_ridge(Xtr, ytr, k=K_FEATURES)          # train portion only
        sc = StandardScaler().fit(Xtr[feats])                     # scaler fit on train only
        Xtr_sc, Xva_sc = sc.transform(Xtr[feats]), sc.transform(Xva[feats])

        alpha = tune_alpha(Xtr_sc, ytr)                           # tuned on train only
        m = Ridge(alpha=alpha).fit(Xtr_sc, ytr)
        tr_pred, va_pred = m.predict(Xtr_sc), m.predict(Xva_sc)
        tr_r, va_r = rmse(ytr, tr_pred), rmse(yva, va_pred)

        folds.append({
            "fold": fold, "alpha": alpha,
            "train_rmse": round(tr_r, 4), "val_rmse": round(va_r, 4),
            "train_r2": round(float(r2_score(ytr, tr_pred)), 4),
            "val_r2":   round(float(r2_score(yva, va_pred)), 4),
            "ratio": round(va_r / tr_r, 4),
        })
        print(f"  Fold {fold}: train R²={folds[-1]['train_r2']:.3f}  "
              f"val R²={folds[-1]['val_r2']:.3f}  val RMSE={va_r:.3f}  "
              f"ratio={va_r/tr_r:.2f}x  alpha={alpha}")

    # ── Overfitting verdict via R² gap (robust); RMSE ratio informational ─
    mean_train_r2 = float(np.mean([f["train_r2"] for f in folds]))
    mean_val_r2   = float(np.mean([f["val_r2"]   for f in folds]))
    r2_gap        = mean_train_r2 - mean_val_r2
    median_ratio  = float(np.median([f["ratio"] for f in folds]))

    print("\n------------------------------")
    print(f"  CV mean train R²   : {mean_train_r2:.4f}")
    print(f"  CV mean val   R²   : {mean_val_r2:.4f}")
    print(f"  R² gap (train-val) : {r2_gap:.4f}  →  {classify_gap(r2_gap)}")
    print(f"  RMSE ratio (median): {median_ratio:.2f}x  (informational only)")

    # ── Final production model: select + scale + tune on full dev ────────
    print("\n[INFO] Final model on full development set...")
    FEATURE_COLS = select_for_ridge(X_dev, y_dev, k=K_FEATURES)
    print(f"[INFO] Final features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    scaler = StandardScaler().fit(X_dev[FEATURE_COLS])
    X_dev_sc  = scaler.transform(X_dev[FEATURE_COLS])
    X_test_sc = scaler.transform(X_test[FEATURE_COLS])

    best_alpha = tune_alpha(X_dev_sc, y_dev)
    print(f"[INFO] Best alpha (full dev): {best_alpha}")
    model = Ridge(alpha=best_alpha).fit(X_dev_sc, y_dev)

    train_m = get_metrics(y_dev,  model.predict(X_dev_sc))
    test_m  = get_metrics(y_test, model.predict(X_test_sc))
    final_gap = train_m["r2"] - test_m["r2"]
    print(f"\n  Final TRAIN : MAE={train_m['mae']}  RMSE={train_m['rmse']}  R²={train_m['r2']}")
    print(f"  Final TEST  : MAE={test_m['mae']}  RMSE={test_m['rmse']}  R²={test_m['r2']}")
    print(f"  Test R² gap (train-test): {final_gap:.4f}  →  {classify_gap(final_gap)}")

    coefs = pd.Series(model.coef_, index=FEATURE_COLS).reindex(
        pd.Series(np.abs(model.coef_), index=FEATURE_COLS).sort_values(ascending=False).index)
    print("\nCoefficients (by magnitude):")
    for feat, c in coefs.items():
        print(f"  {feat:<22} {c:+.4f}")

    # ── Save model + scaler + metrics ────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[SUCCESS] Model  saved → {MODEL_PATH}")
    print(f"[SUCCESS] Scaler saved → {SCALER_PATH}")

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
        "best_alpha": best_alpha,
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
    })

    print("\nPrediction Preview (test set, first 10 rows):")
    preview = pd.DataFrame({"Actual": y_test.values[:10],
                            "Predicted": model.predict(X_test_sc)[:10]})
    print(preview.round(2).to_string(index=False))
    print("\n[SUCCESS] Ridge training pipeline complete.")
    return model, scaler, test_m


if __name__ == "__main__":
    main()
