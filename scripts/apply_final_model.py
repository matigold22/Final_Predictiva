"""Aplica el modelo final al mismo conjunto de test reproducible.

Uso desde la raiz del repositorio:
    python scripts/apply_final_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges_was_missing"] = data["TotalCharges"].isna().astype(int)
    data["TotalCharges"] = data["TotalCharges"].fillna(0)
    tenure_safe = data["tenure"].replace(0, np.nan)
    data["avg_charge_per_tenure"] = (
        data["TotalCharges"] / tenure_safe
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    data["charges_gap"] = data["MonthlyCharges"] - data["avg_charge_per_tenure"]
    data["tenure_is_zero"] = (data["tenure"] == 0).astype(int)
    data["tenure_group"] = pd.cut(
        data["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6", "7-12", "13-24", "25-48", "49+"],
    ).astype("object")
    data["is_month_to_month"] = (data["Contract"] == "Month-to-month").astype(int)
    data["has_fiber_optic"] = (data["InternetService"] == "Fiber optic").astype(int)
    data["is_electronic_check"] = (data["PaymentMethod"] == "Electronic check").astype(int)
    data["paperless_billing_flag"] = (data["PaperlessBilling"] == "Yes").astype(int)
    data["digital_payment_risk"] = (
        data["paperless_billing_flag"] & data["is_electronic_check"]
    ).astype(int)
    support_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    streaming_cols = ["StreamingTV", "StreamingMovies"]
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    data["support_services_count"] = (data[support_cols] == "Yes").sum(axis=1)
    data["streaming_services_count"] = (data[streaming_cols] == "Yes").sum(axis=1)
    data["phone_or_addon_services_count"] = (data[service_cols] == "Yes").sum(axis=1)
    data["has_internet_service"] = (data["InternetService"] != "No").astype(int)
    data["total_services_count"] = (
        data["phone_or_addon_services_count"] + data["has_internet_service"]
    )
    data["has_support_services"] = (data["support_services_count"] > 0).astype(int)
    return data


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "datos/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--model", type=Path, default=root / "modelos/final_model.joblib")
    parser.add_argument("--metadata", type=Path, default=root / "modelos/final_model_metadata.json")
    parser.add_argument("--output", type=Path, default=root / "modelos/resultados/test_predictions_from_saved_model.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.data)
    model = joblib.load(args.model)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    data = build_features(raw)
    y = (data["Churn"] == "Yes").astype(int)
    X = data.drop(columns=["Churn", "customerID"])
    _, X_test, _, y_test, _, customer_test = train_test_split(
        X,
        y,
        data["customerID"],
        test_size=float(metadata["test_size"]),
        stratify=y,
        random_state=int(metadata["random_state"]),
    )

    probabilities = model.predict_proba(X_test)[:, 1]
    threshold = float(metadata["decision_threshold"])
    predictions = (probabilities >= threshold).astype(int)
    output = pd.DataFrame({
        "customerID": customer_test.values,
        "y_true": y_test.values,
        "churn_probability": probabilities,
        "decision_threshold": threshold,
        "churn_prediction": predictions,
    })
    output["y_true_label"] = output["y_true"].map({0: "No", 1: "Yes"})
    output["churn_prediction_label"] = output["churn_prediction"].map({0: "No", 1: "Yes"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Predicciones guardadas en: {args.output}")
    print(f"Filas: {len(output)}")


if __name__ == "__main__":
    main()
