"""Churn Intelligence | premium customer decision workspace."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.predict import load_model, predict_churn  # noqa: E402

st.set_page_config(
    page_title="Churn Intelligence | Decision Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root { --navy:#071f43; --blue:#0759db; --ink:#14243b;
        --muted:#687b96; --line:#e3eaf3; --paper:#f4f7fc; }
[data-testid="stAppViewContainer"] {background:#f4f7fc;}
.block-container {max-width:1480px; padding:1.15rem 2.5rem 3rem;}
header[data-testid="stHeader"] {background:transparent;}
#MainMenu, footer {visibility:hidden;}
h1,h2,h3 {letter-spacing:-.035em;}
.topbar {display:flex; justify-content:space-between; align-items:center;
         padding:13px 20px; background:white; border:1px solid #e2e9f2;
         border-radius:16px 16px 0 0; font-size:12px; color:#6d7e95;}
.brand {font-weight:850; color:#092a60; font-size:15px; letter-spacing:-.04em;}
.brand span {color:#1672eb;}
.nav-pill {display:inline-block; padding:6px 13px; border-radius:20px;
           background:#e9f1ff; color:#1358b7; font-weight:700;}
.hero {padding:30px 34px 32px; background:linear-gradient(115deg,#071b40 0%,#083f8f 72%,#0867c6 100%);
       color:white; border-radius:0 0 20px 20px; margin-bottom:24px; position:relative;}
.eyebrow {font-size:10px; font-weight:800; letter-spacing:.17em; color:#a8d5ff;}
.hero h1 {color:white; font-size:clamp(29px,3vw,42px); margin:10px 0 8px;}
.hero p {color:#d3e4ff; max-width:670px; font-size:14px; line-height:1.7; margin:0;}
.hero-meta {margin-top:20px; display:flex; gap:9px; flex-wrap:wrap;}
.hero-meta span {border:1px solid #7da8df75; background:#ffffff12; color:#e4efff;
                 padding:6px 11px; border-radius:99px; font-size:11px;}
.section-label {color:#1965c7; letter-spacing:.16em; font-weight:800;
                font-size:10px; margin:0 0 5px;}
.section-heading {font-size:21px; font-weight:800; color:#10294b; margin-bottom:4px;}
.section-copy {color:#6c7e98; font-size:12px; margin-bottom:13px;}
.panel-title {color:#122b4b; font-weight:800; font-size:15px; margin:0;}
.panel-sub {color:#7a8aa0; font-size:11px; margin-bottom:10px;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#e0e8f2 !important;
    border-radius:17px !important; background:#fff !important;}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] > div {
    background:#f8faff !important;}
[data-testid="stWidgetLabel"] p {font-size:12px; font-weight:650; color:#344962;}
[data-baseweb="select"] > div, [data-testid="stNumberInput"] input {
    background:#f7f9fd !important; border-color:#e1e8f1 !important;
    border-radius:9px !important; min-height:40px; color:#19314f;}
[data-testid="stNumberInput"] button {background:#f0f4fa !important;}
div.stButton > button[kind="primary"] {
    background:#0862d8; color:white; border:0; border-radius:11px;
    min-height:47px; font-weight:750; box-shadow:0 5px 14px #0c59c72b;}
div.stButton > button[kind="primary"]:hover {background:#074eb0;}
.kpi-label {color:#647a95; font-size:11px; font-weight:750; letter-spacing:.09em;}
.kpi-value {font-size:clamp(32px,4vw,52px); font-weight:850; letter-spacing:-.06em;
            color:#0d2a50; margin:5px 0;}
.status {display:inline-block; padding:6px 11px; border-radius:99px;
         font-weight:750; font-size:11px;}
.status-high {background:#fff0df; color:#9a520e;}
.status-low {background:#e5f8ef; color:#166c4b;}
.muted-note {font-size:11px; color:#71839b; line-height:1.6;}
.footerline {border-top:1px solid #dce5f0; padding-top:14px; margin-top:22px;
             font-size:11px; color:#71839b;}
.footerline a {color:#165dbb; text-decoration:none;}
@media (max-width:850px) {
 .block-container {padding:1rem 1rem 2rem;}
 .hero {padding:25px 20px;}
 .topbar {padding:12px 14px;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Cache the trusted serialized model."""
    return load_model()


def section(title, subtitle):
    st.markdown(f'<div class="panel-title">{title}</div>'
                f'<div class="panel-sub">{subtitle}</div>',
                unsafe_allow_html=True)


def choice(label, options, key, help_text=None):
    return st.selectbox(label, options, key=key, help=help_text)


st.markdown("""
<div class="topbar">
  <div class="brand">◈ CHURN<span> / </span>INTELLIGENCE</div>
  <div><span class="nav-pill">● Decision workspace</span> &nbsp; APPLIED ML · PORTFOLIO</div>
</div>
<div class="hero">
  <div class="eyebrow">CUSTOMER ANALYTICS / MACHINE LEARNING</div>
  <h1>Understand churn.<br>Make informed decisions.</h1>
  <p>Explore a telecom customer's estimated churn probability using a
  trained machine-learning pipeline. Configure a profile and inspect the
  model's score in one streamlined workspace.</p>
  <div class="hero-meta">
    <span>IBM Telco dataset</span><span>Logistic regression</span>
    <span>20 input features</span><span>Interactive ML demo</span>
  </div>
</div>
""", unsafe_allow_html=True)

try:
    model = get_model()
except Exception:
    st.error("The trained model could not be loaded. Verify the model artifact and dependencies.")
    st.stop()

st.markdown('<div class="section-label">01 / CUSTOMER CONFIGURATION</div>'
            '<div class="section-heading">Build a customer profile</div>'
            '<div class="section-copy">Service-dependent fields update immediately when you change an option.</div>',
            unsafe_allow_html=True)

# Native Streamlit widgets stay outside st.form to enable instant reruns.
profile_col, services_col, billing_col = st.columns([1, 1, 1], gap="medium")

with profile_col:
    with st.container(border=True):
        section("01 · Customer & phone", "Demographics and telephone service")
        gender = choice("Gender", ["Female", "Male"], "gender")
        senior = choice("Senior citizen", ["No", "Yes"], "senior")
        partner = choice("Partner", ["No", "Yes"], "partner")
        dependents = choice("Dependents", ["No", "Yes"], "dependents")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120,
                                 value=12, step=1, key="tenure")
        phone = choice("Phone service", ["Yes", "No"], "phone")
        multiple_options = ["No phone service"] if phone == "No" else ["No", "Yes"]
        multiple = choice("Multiple lines", multiple_options, f"multiple_{phone}")

with services_col:
    with st.container(border=True):
        section("02 · Internet & services", "Connectivity and value-added services")
        internet = choice("Internet service", ["Fiber optic", "DSL", "No"], "internet")
        addons = ["No internet service"] if internet == "No" else ["No", "Yes"]
        suffix = "off" if internet == "No" else "on"
        online_security = choice("Online security", addons, f"security_{suffix}")
        online_backup = choice("Online backup", addons, f"backup_{suffix}")
        device_protection = choice("Device protection", addons, f"protection_{suffix}")
        tech_support = choice("Tech support", addons, f"support_{suffix}")
        streaming_tv = choice("Streaming TV", addons, f"tv_{suffix}")
        streaming_movies = choice("Streaming movies", addons, f"movies_{suffix}")

with billing_col:
    with st.container(border=True):
        section("03 · Contract & billing", "Commercial terms and account value")
        contract = choice("Contract", ["Month-to-month", "One year", "Two year"], "contract")
        paperless = choice("Paperless billing", ["Yes", "No"], "paperless")
        payment = choice(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"],
            "payment",
        )
        monthly = st.number_input("Monthly charges (USD)", min_value=0.0,
                                  max_value=1000.0, value=75.0, step=1.0, key="monthly")
        total = st.number_input("Total charges (USD)", min_value=0.0,
                                max_value=100000.0, value=900.0, step=10.0, key="total")
        st.caption("Amounts use the source dataset's USD units.")
        st.info("Prediction uses the exact customer values shown in this workspace.",
                icon="ℹ️")

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

action_left, action_right = st.columns([2, 1], gap="medium")
with action_left:
    st.caption("Educational demonstration · No automated customer retention decision.")
with action_right:
    submitted = st.button("Generate prediction  →", type="primary",
                          use_container_width=True)

if submitted:
    try:
        probability = predict_churn(customer, model)
        st.session_state["churn_result"] = {
            "profile": customer.copy(),
            "probability": probability,
        }
    except ValueError as exc:
        st.session_state.pop("churn_result", None)
        st.error(f"Please review the input profile: {exc}")
    except Exception:
        st.session_state.pop("churn_result", None)
        st.error("Prediction failed. Verify the trained model and feature schema.")

result = st.session_state.get("churn_result")
st.markdown('<div class="section-label" style="margin-top:26px">02 / MODEL OUTPUT</div>'
            '<div class="section-heading">Prediction insights</div>',
            unsafe_allow_html=True)

if result is not None and result["profile"] == customer:
    probability = result["probability"]
    high = probability >= 0.5
    left, middle, right = st.columns([1.1, 1.2, 1], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="kpi-label">ESTIMATED CHURN PROBABILITY</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-value">{probability:.1%}</div>',
                        unsafe_allow_html=True)
            state = "Above baseline threshold" if high else "Below baseline threshold"
            style = "status-high" if high else "status-low"
            st.markdown(f'<span class="status {style}">{state}</span>',
                        unsafe_allow_html=True)
            st.caption("Model-estimated probability, not a guarantee.")
    with middle:
        with st.container(border=True):
            section("Probability indicator", "Position relative to the 50% baseline")
            st.progress(probability)
            st.metric("Baseline threshold", "50%")
            st.caption(f"Difference: {(probability - 0.5) * 100:+.1f} percentage points")
    with right:
        with st.container(border=True):
            section("Profile snapshot", "Values supplied for this prediction")
            st.write(f"**Contract:** {customer['Contract']}")
            st.write(f"**Internet:** {customer['InternetService']}")
            st.write(f"**Tenure:** {customer['tenure']} months")
            st.write(f"**Monthly charge:** ${customer['MonthlyCharges']:,.2f}")
            st.caption("These are inputs, not identified causes of churn.")
    with st.expander("View the 20 submitted model features"):
        st.dataframe(pd.DataFrame(
            {"Feature": list(customer.keys()), "Value": list(customer.values())}
        ), hide_index=True, use_container_width=True)
    st.info("Interpretation: this score is not a causal explanation or a calibrated "
            "guarantee. The 50% threshold is a baseline, not a validated business policy.",
            icon="ℹ️")
else:
    with st.container(border=True):
        st.markdown("**Your prediction workspace is ready.**")
        st.caption("Complete or adjust the customer profile, then select "
                   "Generate prediction to display model insights.")

st.markdown("""
<div class="footerline">
  Portfolio ML demonstration · IBM Telco Customer Churn · scikit-learn ·
  <a href="https://github.com/zahrababacar6c/CustomerChurnPrediction">GitHub repository</a> ·
  <a href="https://github.com/IBM/telco-customer-churn-on-icp4d">Dataset source</a>
</div>
""", unsafe_allow_html=True)
