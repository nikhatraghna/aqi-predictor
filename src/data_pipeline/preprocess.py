
import pandas as pd


# ─────────────────────────────────────────
# PREPROCESS DATA
# ─────────────────────────────────────────

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[INFO] Starting preprocessing...")

    # ─────────────────────────────────────
    # REMOVE DUPLICATES
    # ─────────────────────────────────────

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"[INFO] Removed duplicates: {before - after}")

    # ─────────────────────────────────────
    # SORT DATETIME
    # ─────────────────────────────────────

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("datetime")

    # ─────────────────────────────────────
    # RESET INDEX
    # ─────────────────────────────────────

    df = df.reset_index(drop=True)

    print("[INFO] Datetime standardized.")

    # ─────────────────────────────────────
    # LOWERCASE COLUMNS
    # ─────────────────────────────────────

    df.columns = [col.lower() for col in df.columns]

    print("[INFO] Column names standardized.")

    print("[SUCCESS] Preprocessing complete.")

    return df
