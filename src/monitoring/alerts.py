"""Aggregate drift reports into a single health alert (console now; webhook/email later)."""

import json
from pathlib import Path

REPORT_DIR       = Path("reports/drift")
DATA_DRIFT_PATH  = REPORT_DIR / "data_drift_report.json"
MODEL_DRIFT_PATH = REPORT_DIR / "model_drift_report.json"
ALERTS_PATH      = REPORT_DIR / "alerts_report.json"

SEVERITY = {"NORMAL": 0, "WARNING": 1, "DRIFT": 2}
EMOJI    = {"NORMAL": "🟢", "WARNING": "🟡", "DRIFT": "🔴"}

# Escalate data drift by how many features flagged (one noisy feature ≠ alarm)
DATA_WARN_FRAC  = 0.20
DATA_DRIFT_FRAC = 0.50


def _normalize(status: str) -> str:
    """Map any '🟡 WARNING'-style string to NORMAL / WARNING / DRIFT."""
    s = (status or "").upper()
    if "DRIFT" in s:
        return "DRIFT"
    if "WARNING" in s:
        return "WARNING"
    return "NORMAL"


def _load(path: Path):
    if not path.exists():
        print(f"[WARNING] Missing report: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def assess_data_drift(report) -> dict:
    if not report:
        return {"severity": "NORMAL", "drifted": 0, "total": 0, "fraction": 0.0}
    total   = len(report)
    drifted = sum(1 for r in report if _normalize(r.get("status")) == "DRIFT")
    frac    = drifted / total if total else 0.0
    if   frac >= DATA_DRIFT_FRAC: sev = "DRIFT"
    elif frac >= DATA_WARN_FRAC:  sev = "WARNING"
    else:                         sev = "NORMAL"
    return {"severity": sev, "drifted": drifted, "total": total, "fraction": round(frac, 2)}


def assess_model_drift(report) -> dict:
    if not report:
        return {"severity": "NORMAL"}
    return {
        "severity":        _normalize(report.get("status")),
        "degradation_pct": report.get("rmse_degradation_pct"),
        "current_rmse":    report.get("current_metrics", {}).get("rmse"),
        "baseline_rmse":   report.get("baseline_rmse"),
    }


def send_alert(message: str, severity: str) -> None:
    """Emit the alert. Console for now — extend to Slack/email/webhook here."""
    print(message)
    # TODO: if severity in ("WARNING", "DRIFT"): post to Slack webhook / send email


def main():
    print("\n================================================")
    print(" AQI MONITORING ALERTS ")
    print("================================================")

    data  = assess_data_drift(_load(DATA_DRIFT_PATH))
    model = assess_model_drift(_load(MODEL_DRIFT_PATH))

    overall = max(data["severity"], model["severity"], key=lambda s: SEVERITY[s])
    retrain_recommended = (model["severity"] == "DRIFT")

    lines = [
        "",
        f"{EMOJI[overall]} OVERALL STATUS: {overall}",
        "------------------------------------------------",
        f"  Data drift  : {EMOJI[data['severity']]} {data['severity']} "
        f"({data['drifted']}/{data['total']} features drifted)",
        f"  Model drift : {EMOJI[model['severity']]} {model['severity']} "
        f"(RMSE degraded {model.get('degradation_pct')}%)",
    ]
    if retrain_recommended:
        lines.append("  ⚠️  ACTION: model drift = DRIFT → retraining recommended.")
    elif overall != "NORMAL":
        lines.append("  ACTION: monitor closely; no retraining yet.")
    else:
        lines.append("  ACTION: none — system healthy.")
    send_alert("\n".join(lines), overall)

    report = {
        "overall_status":      overall,
        "retrain_recommended": retrain_recommended,
        "data_drift":          data,
        "model_drift":         model,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_PATH, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n[SUCCESS] Alert summary saved → {ALERTS_PATH}")
    print("================================================")
    return report


if __name__ == "__main__":
    main()
