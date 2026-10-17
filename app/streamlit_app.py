
"""Interactive Customer Churn Prediction application."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure imports work both locally and on Streamlit Community Cloud.
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.predict import load_model as load_pipeline
from app.predict import predict_churn, validate_customer


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn | ML Demo",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

@st.cache_resource
def get_model():
    """Load and cache the trained scikit-learn pipeline."""
    return load_pipeline()


st.title("Customer Churn Prediction")

st.caption(
    "IBM Telco Customer Churn · "
    "scikit-learn · Logistic Regression"
)

st.write(
    "Explore a customer's estimated churn risk using "
    "a trained machine-learning pipeline. "
    "This is an educational demonstration, not a "
    "production retention decision system."
)


try:
    model = get_model()

except FileNotFoundError:
    st.warning(
        "Trained model not found. Please ensure "
        "`customer_churn_pipeline.joblib` exists "
        "in the repository's `models/` directory."
    )
    st.stop()

except Exception as exc:
    st.error("Unable to load the trained model.")
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------
# Customer input form
# ---------------------------------------------------------

with st.form("customer_profile"):

    st.subheader("Customer profile")

    col1, col2, col3 = st.columns(3)

    # Customer information
    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior = st.selectbox(
            "Senior citizen",
            ["No", "Yes"],
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"],
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"],
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=120,
            value=12,
            step=1,
        )

        phone = st.selectbox(
            "Phone service",
            ["Yes", "No"],
        )

        multiple = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"],
        )

    # Internet services
    with col2:

        internet = st.selectbox(
            "Internet service",
            ["Fiber optic", "DSL", "No"],
        )

        online_security = st.selectbox(
            "Online security",
            ["No", "Yes", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online backup",
            ["No", "Yes", "No internet service"],
        )

        device_protection = st.selectbox(
            "Device protection",
            ["No", "Yes", "No internet service"],
        )

        tech_support = st.selectbox(
            "Tech support",
            ["No", "Yes", "No internet service"],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming movies",
            ["No", "Yes", "No internet service"],
        )

    # Contract and billing
    with col3:

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless = st.selectbox(
            "Paperless billing",
            ["Yes", "No"],
        )

        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly = st.number_input(
            "Monthly charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=75.0,
            step=1.0,
        )

        total = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=900.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Estimate churn probability",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if submitted:

    # Build the customer feature dictionary.
    customer = {
        "gender": gender,
        "SeniorCitizen": int(senior == "Yes"),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total),
        "TotalCharges_missing": 0,
    }

    # Validate inputs and generate prediction.
    try:
        validate_customer(customer)

        probability = predict_churn(
            customer=customer,
            model=model,
        )

    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    except Exception as exc:
        st.error(
            "Prediction failed. Please verify "
            "the trained model and feature schema."
        )
        st.exception(exc)
        st.stop()

    # Prepare input data for display.
    features = pd.DataFrame([customer])

    # -----------------------------------------------------
    # Prediction results
    # -----------------------------------------------------

    st.divider()

    st.subheader("Prediction result")

    left, right = st.columns([1, 2])

    threshold = 0.50

    with left:

        st.metric(
            label="Estimated churn probability",
            value=f"{probability:.1%}",
        )

        if probability >= threshold:
            st.write("Above 50% decision threshold")
        else:
            st.write("Below 50% decision threshold")

    with right:

        st.progress(probability)

        st.info(
            "The 50% threshold matches the notebook's "
            "baseline evaluation. A production threshold "
            "should be selected on validation data "
            "according to business costs."
        )

    st.caption(
        "This score is a model estimate, not a causal "
        "explanation or a calibrated guarantee of churn."
    )

    # -----------------------------------------------------
    # Submitted features
    # -----------------------------------------------------

    with st.expander("Show submitted features"):

        st.dataframe(
            features.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.markdown(
    "**Project:** "
    "[GitHub repository]"
    "(https://github.com/zahrababacar6c/CustomerChurnPrediction)"
    " · "
    "**Dataset:** "
    "[IBM Telco Customer Churn]"
    "(https://github.com/IBM/telco-customer-churn-on-icp4d)"
)
