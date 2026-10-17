
"""Automated tests for customer churn prediction."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from app.predict import (
    EXPECTED_FEATURES,
    load_model,
    predict_churn,
    validate_customer,
)


@pytest.fixture
def customer():
    """Return a valid IBM Telco customer profile."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
        "TotalCharges_missing": 0,
    }


@pytest.fixture(scope="module")
def trained_model():
    """Load the trusted trained pipeline once."""
    return load_model()


def test_model_loading(trained_model):
    assert hasattr(trained_model, "predict_proba")


def test_expected_feature_count(customer):
    assert len(EXPECTED_FEATURES) == 20
    assert set(customer) == EXPECTED_FEATURES


def test_valid_customer(customer):
    assert validate_customer(customer) is None


def test_prediction_probability(customer, trained_model):
    probability = predict_churn(customer, trained_model)

    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


@pytest.mark.parametrize(
    "feature",
    sorted(EXPECTED_FEATURES),
)
def test_missing_features(customer, feature):
    invalid = deepcopy(customer)
    del invalid[feature]

    with pytest.raises(ValueError, match="Missing required features"):
        validate_customer(invalid)


def test_unexpected_feature(customer):
    customer["UnknownFeature"] = "unexpected"

    with pytest.raises(ValueError, match="Unexpected features"):
        validate_customer(customer)


@pytest.mark.parametrize(
    "feature,value",
    [
        ("gender", "Unknown"),
        ("Contract", "Three year"),
        ("InternetService", "Satellite"),
        ("PaymentMethod", "Cash"),
        ("Partner", None),
    ],
)
def test_invalid_categories(customer, feature, value):
    customer[feature] = value

    with pytest.raises(ValueError, match="Invalid value"):
        validate_customer(customer)


@pytest.mark.parametrize(
    "feature,value",
    [
        ("tenure", -1),
        ("tenure", 1.5),
        ("MonthlyCharges", -20.0),
        ("TotalCharges", -100.0),
        ("MonthlyCharges", float("nan")),
        ("MonthlyCharges", float("inf")),
        ("TotalCharges", None),
        ("SeniorCitizen", 2),
        ("SeniorCitizen", True),
        ("TotalCharges_missing", -1),
    ],
)
def test_invalid_numeric_values(customer, feature, value):
    customer[feature] = value

    with pytest.raises(ValueError):
        validate_customer(customer)


@pytest.mark.parametrize(
    "phone,multiple",
    [
        ("No", "Yes"),
        ("No", "No"),
        ("Yes", "No phone service"),
    ],
)
def test_invalid_phone_combinations(customer, phone, multiple):
    customer["PhoneService"] = phone
    customer["MultipleLines"] = multiple

    with pytest.raises(ValueError):
        validate_customer(customer)


@pytest.mark.parametrize(
    "internet,addon",
    [
        ("No", "Yes"),
        ("No", "No"),
        ("DSL", "No internet service"),
        ("Fiber optic", "No internet service"),
    ],
)
def test_invalid_internet_combinations(customer, internet, addon):
    customer["InternetService"] = internet
    customer["TechSupport"] = addon

    with pytest.raises(ValueError):
        validate_customer(customer)


def test_valid_customer_without_phone(customer):
    customer["PhoneService"] = "No"
    customer["MultipleLines"] = "No phone service"

    assert validate_customer(customer) is None


def test_valid_customer_without_internet(customer):
    customer["InternetService"] = "No"

    for feature in (
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ):
        customer[feature] = "No internet service"

    assert validate_customer(customer) is None


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_invalid_model_probability(customer, probability):
    mock_model = Mock()
    mock_model.predict_proba.return_value = [
        [1.0 - probability, probability]
    ]

    with pytest.raises(ValueError):
        predict_churn(customer, mock_model)


def test_prediction_rejects_invalid_customer(customer, trained_model):
    customer["Contract"] = "Invalid contract"

    with pytest.raises(ValueError):
        predict_churn(customer, trained_model)
