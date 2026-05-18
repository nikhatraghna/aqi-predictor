
import pandas as pd


# ─────────────────────────────────────────
# VALIDATE DATA
# ─────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[INFO] Validating dataset...\n")

    # ─────────────────────────────────────
    # NULL CHECK
    # ─────────────────────────────────────

    null_counts = df.isnull().sum()

    print("[INFO] Null values:\n")

    print(null_counts)

    # ─────────────────────────────────────
    # REMOVE ALL-NULL COLUMNS
    # ─────────────────────────────────────

    all_null_cols = [
        col for col in df.columns
        if df[col].isnull().all()
    ]

    if all_null_cols:

        print("\n[WARNING] Removing ALL-NULL columns:")

        for col in all_null_cols:
            print(f"   - {col}")

        df = df.drop(columns=all_null_cols)

    # ─────────────────────────────────────
    # REMOVE HIGH-NULL COLUMNS
    # (>40%)
    # ─────────────────────────────────────

    threshold = 0.80

    high_null_cols = []

    for col in df.columns:

        ratio = df[col].isnull().mean()

        if ratio > threshold:

            high_null_cols.append(col)

    if high_null_cols:

        print("\n[WARNING] Removing HIGH-NULL columns:")

        for col in high_null_cols:
            print(f"   - {col}")

        df = df.drop(columns=high_null_cols)

    # ─────────────────────────────────────
    # NEGATIVE POLLUTANT CHECK
    # ─────────────────────────────────────

    pollutant_cols = [
        "pm25",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3"
    ]

    for col in pollutant_cols:

        if col in df.columns:

            negatives = (df[col] < 0).sum()

            if negatives > 0:

                print(f"[WARNING] {negatives} negative values found in {col}")

                df = df[df[col] >= 0]

    # ─────────────────────────────────────
    # FINAL CLEAN
    # ─────────────────────────────────────

    before = len(df)

    df = df.dropna().reset_index(drop=True)

    after = len(df)

    print(f"\n[INFO] Removed rows with NaNs: {before - after}")

    print(f"[INFO] Final shape: {df.shape}")

    print("\n[SUCCESS] Validation complete.")

    return df
