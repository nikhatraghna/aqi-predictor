import pandas as pd

import json
from pathlib import Path
# ─────────────────────────────────────────
# MODEL RESULTS
# ─────────────────────────────────────────

results = pd.DataFrame({

    "Model": [

        "Ridge",

        "Random Forest",

        "XGBoost",

        "Prophet",
    ],

    "MAE": [

        1.88,

        2.74,

        2.43,

        14.85,
    ],

    "RMSE": [

        2.93,

        4.02,

        3.80,

        17.43,
    ],

    "R2": [

        0.9737,

        0.9507,

        0.9560,

        0.0736,
    ],
})


# ─────────────────────────────────────────
# SORT RESULTS
# ─────────────────────────────────────────

results = results.sort_values(
    by="RMSE"
)

results = results.reset_index(drop=True)


# ─────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────

print("\n==============================")

print(" MODEL COMPARISON")

print("==============================\n")

print(results)


# ─────────────────────────────────────────
# BEST MODEL
# ─────────────────────────────────────────

best_model = results.iloc[0]

print("\n==============================")

print(" BEST MODEL")

print("==============================\n")

print(f"Model : {best_model['Model']}")

print(f"MAE   : {best_model['MAE']}")

print(f"RMSE  : {best_model['RMSE']}")

print(f"R²    : {best_model['R2']}")

# ---------------------------------------------------
# Save metrics
# ---------------------------------------------------

import json
from pathlib import Path

Path("models").mkdir(exist_ok=True)

metrics = {}

for _, row in results.iterrows():

    model_key = (
        row["Model"]
        .lower()
        .replace(" ", "_")
    )

    metrics[model_key] = {
        "mae": float(row["MAE"]),
        "rmse": float(row["RMSE"]),
        "r2": float(row["R2"]),
    }

with open("models/model_metrics.json", "w") as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )

print("\n[SUCCESS] Metrics saved.")
