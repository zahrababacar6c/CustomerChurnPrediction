
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "customer_churn_pipeline.joblib"
)


def load_model(model_path=MODEL_PATH):
    """Load the trusted, trained churn prediction pipeline."""
    return joblib.load(model_path)


def validate_customer(customer):
    """Validate customer service combinations."""
    if customer["PhoneService"] == "No":
        if customer["MultipleLines"] != "No phone service":
            raise ValueError("Invalid phone service combination.")
    elif customer["MultipleLines"] == "No phone service":
        raise ValueError("Invalid multiple lines selection.")

    internet_features = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    if customer["InternetService"] == "No":
        if any(
            customer[field] != "No internet service"
            for field in internet_features
        ):
            raise ValueError("Invalid internet service combination.")
    elif any(
        customer[field] == "No internet service"
        for field in internet_features
    ):
        raise ValueError("Invalid internet add-on selection.")


def predict_churn(customer, model):
    """Return the estimated churn probability."""
    validate_customer(customer)

    features = pd.DataFrame([customer])
    probability = float(model.predict_proba(features)[0, 1])

    if not 0 <= probability <= 1:
        raise ValueError("Model returned an invalid probability.")

    return probability
