"""Interactive demo for the IBM Telco Customer Churn scikit-learn pipeline."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn | ML Demo", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
h1,h2,h3 {letter-spacing: -.03em;}
[data-testid="stMetricValue"] {font-size: 2.5rem;}
</style>""", unsafe_allow_html=True)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "customer_churn_pipeline.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)  # Load only a model file you created and trust.

st.title("Customer Churn Prediction")
st.caption("IBM Telco Customer Churn · scikit-learn · Logistic Regression")
st.write("Explore a customer's estimated churn risk using a trained machine-learning pipeline. This is an educational demonstration, not a production retention decision system.")

model = load_model()
if model is None:
    st.warning("Trained model not found. Run the scikit-learn notebook and place `customer_churn_pipeline.joblib` in the repository's `models/` folder to enable predictions.")
    st.stop()

with st.form("customer_profile"):
    st.subheader("Customer profile")
    a,b,c=st.columns(3)
    with a:
        gender=st.selectbox("Gender", ["Female","Male"])
        senior=st.selectbox("Senior citizen", ["No","Yes"])
        partner=st.selectbox("Partner", ["No","Yes"])
        dependents=st.selectbox("Dependents", ["No","Yes"])
        tenure=st.number_input("Tenure (months)", min_value=0,max_value=120,value=12,step=1)
        phone=st.selectbox("Phone service", ["Yes","No"])
        multiple=st.selectbox("Multiple lines", ["No","Yes","No phone service"])
    with b:
        internet=st.selectbox("Internet service", ["Fiber optic","DSL","No"])
        online_security=st.selectbox("Online security", ["No","Yes","No internet service"])
        online_backup=st.selectbox("Online backup", ["No","Yes","No internet service"])
        device_protection=st.selectbox("Device protection", ["No","Yes","No internet service"])
        tech_support=st.selectbox("Tech support", ["No","Yes","No internet service"])
        streaming_tv=st.selectbox("Streaming TV", ["No","Yes","No internet service"])
        streaming_movies=st.selectbox("Streaming movies", ["No","Yes","No internet service"])
    with c:
        contract=st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        paperless=st.selectbox("Paperless billing", ["Yes","No"])
        payment=st.selectbox("Payment method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        monthly=st.number_input("Monthly charges ($)",min_value=0.0,max_value=1000.0,value=75.0,step=1.0)
        total=st.number_input("Total charges ($)",min_value=0.0,max_value=100000.0,value=900.0,step=10.0)
    submitted=st.form_submit_button("Estimate churn probability",type="primary",use_container_width=True)

if submitted:
    warnings=[]
    if phone == "No" and multiple != "No phone service":
        warnings.append("Multiple lines should be 'No phone service' when phone service is 'No'.")
    if phone == "Yes" and multiple == "No phone service":
        warnings.append("Multiple lines should be 'Yes' or 'No' when phone service is 'Yes'.")
    internet_fields=[online_security,online_backup,device_protection,tech_support,streaming_tv,streaming_movies]
    if internet == "No" and any(v != "No internet service" for v in internet_fields):
        warnings.append("Internet add-ons should be 'No internet service' when internet service is 'No'.")
    if internet != "No" and any(v == "No internet service" for v in internet_fields):
        warnings.append("Internet add-ons should be 'Yes' or 'No' when internet service is active.")
    if warnings:
        for msg in warnings: st.error(msg)
        st.stop()
    row={
        "gender":gender,"SeniorCitizen":int(senior=="Yes"),"Partner":partner,"Dependents":dependents,
        "tenure":int(tenure),"PhoneService":phone,"MultipleLines":multiple,"InternetService":internet,
        "OnlineSecurity":online_security,"OnlineBackup":online_backup,"DeviceProtection":device_protection,
        "TechSupport":tech_support,"StreamingTV":streaming_tv,"StreamingMovies":streaming_movies,
        "Contract":contract,"PaperlessBilling":paperless,"PaymentMethod":payment,
        "MonthlyCharges":float(monthly),"TotalCharges":float(total),"TotalCharges_missing":0
    }
    X=pd.DataFrame([row])
    try:
        p=float(model.predict_proba(X)[0,1])
    except Exception as exc:
        st.error("Prediction failed. Check that this model was trained with the matching scikit-learn notebook and feature schema.")
        st.exception(exc)
        st.stop()
    st.divider()
    st.subheader("Prediction result")
    left,right=st.columns([1,2])
    with left:
        st.metric("Estimated churn probability",f"{p:.1%}")
        st.write("Above 50% decision threshold" if p>=0.5 else "Below 50% decision threshold")
    with right:
        st.progress(min(max(p,0.0),1.0))
        st.info("The 50% threshold matches the notebook's baseline evaluation. A production threshold should be selected on validation data according to business costs.")
    st.caption("This score is a model estimate, not a causal explanation or a calibrated guarantee of churn.")
    with st.expander("Show submitted features"):
        st.dataframe(X.T.rename(columns={0:"Value"}),use_container_width=True)

st.divider()
st.markdown("**Project:** [GitHub repository](https://github.com/zahrababacar6c/CustomerChurnPrediction) · **Dataset:** [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)")
