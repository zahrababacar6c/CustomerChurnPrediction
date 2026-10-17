
import pytest

from app.predict import (
    load_model,
    predict_churn,
    validate_customer,
)


@pytest.fixture
def customer():
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


def test_model_loading():
    model = load_model()
    assert hasattr(model, "predict_proba")


def test_prediction_probability(customer):
    model = load_model()
    probability = predict_churn(customer, model)

    assert isinstance(probability, float)
    assert 0 <= probability <= 1


def test_invalid_phone_service(customer):
    customer["PhoneService"] = "No"
    customer["MultipleLines"] = "Yes"

    with pytest.raises(ValueError):
        validate_customer(customer)


def test_invalid_internet_service(customer):
    customer["InternetService"] = "No"
    customer["TechSupport"] = "Yes"

    with pytest.raises(ValueError):
        validate_customer(customer)
