"""Premium Streamlit interface for the IBM Telco churn prediction demo."""

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.predict import load_model, predict_churn  # noqa: E402

st.set_page_config(
    page_title="Churn Intelligence | ML Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#102E2D; --navy:#101C29; --sage:#90AE94; --ivory:#F7F5EF; }
      html, body, [class*="css"], [data-testid="stApp"] { font-family:'DM Sans',sans-serif; }
      [data-testid="stAppViewContainer"] { background:#F7F5EF; color:#101C29; }
      .block-container { max-width:1180px; padding-top:1.6rem; padding-bottom:3rem; }
      h1,h2,h3,h4 { font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
      [data-testid="stHeader"] { background:transparent; }
      [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:17px; }
      div[data-testid="stMetric"] { background:white; border:1px solid #E2E8E4; padding:1.1rem; border-radius:16px; }
      div[data-testid="stMetricValue"] { color:#102E2D; font-weight:800; }
      .brandbar { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:1.4rem; }
      .brand { font-family:'Manrope',sans-serif; font-weight:800; color:#102E2D; letter-spacing:-.05em; font-size:1.1rem; }
      .brand span { color:#52806A; }
      .brandtag { color:#5B6D6C; font-size:.76rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
      .hero { border-radius:24px; padding:2.25rem 2.5rem; background:linear-gradient(115deg,#102E2D 5%,#173D4B 75%,#20384B); color:#F7F5EF; margin-bottom:1.6rem; }
      .hero .kicker { color:#B6D6C2; text-transform:uppercase; font-size:.75rem; letter-spacing:.18em; font-weight:800; }
      .hero h1 { color:#FFF; font-size:clamp(2rem,4.2vw,3.25rem); margin:.7rem 0; line-height:1.15; }
      .hero p { color:#DBE6E0; max-width:690px; line-height:1.65; margin-bottom:1.15rem; }
      .hero .pill { display:inline-block; border:1px solid #66877E; color:#E3F0E7; padding:.4rem .75rem; border-radius:30px; font-size:.75rem; margin-right:.45rem; margin-top:.4rem; }
      .sectionlabel { font-size:.73rem; letter-spacing:.14em; text-transform:uppercase; font-weight:800; color:#567B6B; margin-bottom:.35rem; }
      .sectionheading { font-family:'Manrope',sans-serif; font-size:1.45rem; font-weight:800; color:#101C29; letter-spacing:-.04em; margin-bottom:.4rem; }
      .sectionnote { color:#60716E; font-size:.91rem; margin-bottom:1rem; }
      .micro { color:#61716C; font-size:.82rem; }
      .resultpanel { background:white; border:1px solid #E1E8E3; border-radius:18px; padding:1.3rem 1.5rem; }
      .resultpanel .eyebrow { color:#64816E; font-size:.73rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
      .resultpanel .bigscore { font-family:'Manrope',sans-serif; color:#102E2D; font-size:clamp(2.6rem,5vw,4rem); font-weight:800; letter-spacing:-.06em; line-height:1.25; }
      .resultpanel .sub { color:#60716E; font-size:.85rem; }
      .risk { border-radius:10px; display:inline-block; font-size:.8rem; font-weight:800; padding:.4rem .7rem; }
      .risk.high { background:#FFF0E8; color:#9A4529; }
      .risk.low { background:#E7F3EB; color:#25664A; }
      div.stButton > button[kind="primary"] { background:#102E2D; border:1px solid #102E2D; color:white; border-radius:12px; min-height:3.3rem; font-weight:800; }
      div.stButton > button[kind="primary"]:hover { background:#22504A; border-color:#22504A; }
      div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input { border-radius:10px; }
      .footer { border-top:1px solid #DEE6DF; padding-top:1.1rem; color:#63716D; font-size:.83rem; }
      @media(max-width:640px) { .hero { padding:1.5rem; } .brandtag { display:none; } .block-container { padding-top:.8rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    """Load the trusted scikit-learn pipeline once per Streamlit process."""
    return load_model()


st.markdown(
    '<div class="brandbar"><div class="brand">CHURN<span> / </span>INTELLIGENCE</div>'
    '<div class="brandtag">Applied ML · Portfolio Project</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <div class="kicker">Customer analytics / Machine learning</div>
      <h1>Understand customer<br>churn risk.</h1>
      <p>Explore a telecom customer's estimated churn probability using a trained
      scikit-learn pipeline. Configure the customer profile and generate a model-based score.</p>
      <span class="pill">IBM Telco dataset</span>
      <span class="pill">Logistic regression</span>
      <span class="pill">Interactive ML demo</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = get_model()
except FileNotFoundError:
    st.error("Model artifact missing: `models/customer_churn_pipeline.joblib`.")
    st.stop()
except Exception:
    st.error("Unable to load the model. Verify the artifact and dependencies.")
    st.stop()

st.markdown('<div class="sectionlabel">01 / Inputs</div><div class="sectionheading">Customer profile</div>'
            '<div class="sectionnote">Choose the customer attributes below. Service options update automatically.</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    with st.container(border=True):
        st.markdown("#### Customer & phone")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No phone service"] if phone == "No" else ["No", "Yes"], key=f"multiple_{phone}")

with col2:
    with st.container(border=True):
        st.markdown("#### Internet & add-ons")
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        internet_options = ["No internet service"] if internet == "No" else ["No", "Yes"]
        internet_key = "none" if internet == "No" else "active"
        online_security = st.selectbox("Online security", internet_options, key=f"security_{internet_key}")
        online_backup = st.selectbox("Online backup", internet_options, key=f"backup_{internet_key}")
        device_protection = st.selectbox("Device protection", internet_options, key=f"device_{internet_key}")
        tech_support = st.selectbox("Tech support", internet_options, key=f"tech_{internet_key}")
        streaming_tv = st.selectbox("Streaming TV", internet_options, key=f"tv_{internet_key}")
        streaming_movies = st.selectbox("Streaming movies", internet_options, key=f"movies_{internet_key}")

with col3:
    with st.container(border=True):
        st.markdown("#### Contract & billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=1000.0, value=75.0, step=1.0)
        total = st.number_input("Total charges ($)", min_value=0.0, max_value=100000.0, value=900.0, step=10.0)
        st.caption("Amounts follow the source dataset's USD units.")

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

st.write("")
left_action, right_action = st.columns([2, 1])
with left_action:
    st.caption("The score is an estimate, not an explanation of why a customer may churn.")
with right_action:
    submitted = st.button("Generate prediction →", type="primary", use_container_width=True)

if submitted:
    try:
        probability = predict_churn(customer, model)
    except ValueError as exc:
        st.error(f"Please review the profile: {exc}")
        st.stop()
    except Exception:
        st.error("Prediction failed. Please check the model and feature schema.")
        st.stop()
    st.session_state["prediction"] = {"customer": customer.copy(), "probability": probability}

last_prediction = st.session_state.get("prediction")
if last_prediction is not None and last_prediction["customer"] == customer:
    probability = last_prediction["probability"]
    threshold = 0.50
    above = probability >= threshold
    st.divider()
    st.markdown('<div class="sectionlabel">02 / Model output</div><div class="sectionheading">Prediction overview</div>'
                '<div class="sectionnote">A transparent view of the estimated score and the baseline decision rule.</div>',
                unsafe_allow_html=True)
    result_left, result_right = st.columns([1, 1.4], gap="medium")
    with result_left:
        risk_class = "high" if above else "low"
        risk_label = "Above baseline threshold" if above else "Below baseline threshold"
        st.markdown(
            '<div class="resultpanel"><div class="eyebrow">Estimated churn probability</div>'
            f'<div class="bigscore">{probability:.1%}</div>'
            f'<span class="risk {risk_class}">{html.escape(risk_label)}</span>'
            '<p class="sub">Model-estimated probability of customer churn.</p></div>',
            unsafe_allow_html=True,
        )
    with result_right:
        with st.container(border=True):
            st.markdown("#### Risk score")
            st.progress(probability)
            st.metric("Baseline decision threshold", "50%")
            st.write("The estimated score is **above** the baseline threshold." if above else
                     "The estimated score is **below** the baseline threshold.")
            st.caption("This 50% threshold follows the notebook baseline; production decisions require business-cost analysis and validation.")

    with st.expander("Explore the submitted model features"):
        st.dataframe(pd.DataFrame([customer]).T.rename(columns={0: "Value"}), use_container_width=True)
    st.info("Interpretation note: This score is not a causal explanation or a calibrated guarantee. Do not use it as an automated retention decision.")
else:
    st.markdown('<p class="micro">Prediction results will appear here after you submit a profile.</p>', unsafe_allow_html=True)

st.write("")
st.markdown(
    '<div class="footer">Portfolio demonstration · IBM Telco Customer Churn · scikit-learn · '
    '<a href="https://github.com/zahrababacar6c/CustomerChurnPrediction">GitHub repository</a> · '
    '<a href="https://github.com/IBM/telco-customer-churn-on-icp4d">Dataset source</a></div>',
    unsafe_allow_html=True,
)
