
"""Validation and inference utilities for customer churn prediction."""

from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
import math

import joblib
import pandas as pd


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "customer_churn_pipeline.joblib"
)


CATEGORICAL_FEATURES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

NUMERIC_FEATURES = {
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalCharges_missing",
}

EXPECTED_FEATURES = (
    set(CATEGORICAL_FEATURES)
    | NUMERIC_FEATURES
)

INTERNET_ADDONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)


def load_model(model_path=MODEL_PATH):
    """Load a trusted, serialized scikit-learn pipeline."""
    return joblib.load(model_path)


def validate_customer(customer):
    """Validate customer features before model inference."""

    if not isinstance(customer, Mapping):
        raise ValueError(
            "Customer input must be a dictionary-like mapping."
        )

    provided = set(customer)

    missing = EXPECTED_FEATURES - provided
    unexpected = provided - EXPECTED_FEATURES

    if missing:
        raise ValueError(
            f"Missing required features: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected features: {sorted(unexpected)}"
        )

    # Validate categorical values.
    for feature, allowed in CATEGORICAL_FEATURES.items():
        value = customer[feature]

        if not isinstance(value, str) or value not in allowed:
            raise ValueError(
                f"Invalid value for {feature}: {value!r}. "
                f"Allowed values: {sorted(allowed)}"
            )

    # Validate numeric values.
    for feature in NUMERIC_FEATURES:
        value = customer[feature]

        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{feature} must be a numeric value."
            )

        if not math.isfinite(float(value)):
            raise ValueError(
                f"{feature} must be finite."
            )

    # Validate binary indicators.
    for feature in ("SeniorCitizen", "TotalCharges_missing"):
        value = customer[feature]

        if not isinstance(value, Integral) or value not in (0, 1):
            raise ValueError(
                f"{feature} must be either 0 or 1."
            )

    # Validate tenure.
    tenure = customer["tenure"]

    if not isinstance(tenure, Integral) or tenure < 0:
        raise ValueError(
            "tenure must be a non-negative integer."
        )

    # Validate billing amounts.
    for feature in ("MonthlyCharges", "TotalCharges"):
        if customer[feature] < 0:
            raise ValueError(
                f"{feature} cannot be negative."
            )

    # Validate phone service consistency.
    if customer["PhoneService"] == "No":
        if customer["MultipleLines"] != "No phone service":
            raise ValueError(
                "MultipleLines must be 'No phone service' "
                "when PhoneService is 'No'."
            )

    elif customer["MultipleLines"] == "No phone service":
        raise ValueError(
            "MultipleLines must be 'Yes' or 'No' "
            "when PhoneService is active."
        )

    # Validate internet service consistency.
    if customer["InternetService"] == "No":
        if any(
            customer[field] != "No internet service"
            for field in INTERNET_ADDONS
        ):
            raise ValueError(
                "Internet add-ons must be 'No internet service' "
                "when InternetService is 'No'."
            )

    elif any(
        customer[field] == "No internet service"
        for field in INTERNET_ADDONS
    ):
        raise ValueError(
            "Internet add-ons must be 'Yes' or 'No' "
            "when InternetService is active."
        )


def predict_churn(customer, model):
    """Return the estimated probability of customer churn."""

    validate_customer(customer)

    features = pd.DataFrame([customer])

    probability = float(
        model.predict_proba(features)[0, 1]
    )

    if not math.isfinite(probability):
        raise ValueError(
            "Model returned a non-finite probability."
        )

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "Model returned a probability outside [0, 1]."
        )

    return probability
