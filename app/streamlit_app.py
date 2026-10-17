        border-radius: 14px; padding: 1.1rem;
      }
      div[data-testid="stMetricValue"] {font-size: 2.25rem; color: #102E2D;}
      .eyebrow {font-size: .78rem; font-weight: 750; letter-spacing: .15em;
        color: #277D72; text-transform: uppercase; margin-bottom: .5rem;}
      .hero {padding: 1.7rem 2rem; background: linear-gradient(115deg,#10283D,#174A4A);
        color: white; border-radius: 20px; margin-bottom: 1.5rem;}
      .hero h1 {color: white; margin: .15rem 0 .5rem; font-size: clamp(2rem,4vw,3rem);}
      .hero p {color: #D9E8EA; margin: 0; max-width: 680px; line-height: 1.6;}
      .hero .eyebrow {color: #A7E0CE;}
      .section-intro {color: #617186; margin-top: -.5rem; margin-bottom: 1rem;}
      .small-note {font-size: .86rem; color: #68798C;}
      div.stButton > button[kind="primary"] {
        background: #11695E; border-color: #11695E; border-radius: 10px;
        min-height: 3rem; font-weight: 700;
      }
      div.stButton > button[kind="primary"]:hover {
        background: #0B554B; border-color: #0B554B;
      }
      @media (max-width: 640px) {
        .hero {padding: 1.35rem;}
        .block-container {padding-top: 1rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    """Load the trusted trained scikit-learn pipeline once."""
    return load_model()


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Applied machine learning · Interactive demo</div>
      <h1>Customer Churn Prediction</h1>
      <p>Explore the estimated probability that a telecom customer will leave,
      using a trained scikit-learn pipeline and a structured customer profile.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = get_model()
except FileNotFoundError:
    st.error("Model artifact not found. Check `models/customer_churn_pipeline.joblib`.")
    st.stop()
except Exception:
    st.error("The prediction model could not be loaded. Check the deployed artifact and dependencies.")
    st.stop()

st.subheader("Customer profile")
st.markdown(
    '<p class="section-intro">Update the customer information below. Dependent service fields adapt automatically.</p>',
    unsafe_allow_html=True,
)

# The widgets intentionally live outside st.form: Streamlit must rerun immediately
# when PhoneService or InternetService changes.
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.markdown("#### 01 · Customer")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple lines",
            ["No phone service"] if phone == "No" else ["No", "Yes"],
            key=f"multiple_lines_{phone}",
        )

with col2:
    with st.container(border=True):
        st.markdown("#### 02 · Internet & add-ons")
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        internet_options = ["No internet service"] if internet == "No" else ["No", "Yes"]
        internet_key = "no_internet" if internet == "No" else "internet_active"
        online_security = st.selectbox("Online security", internet_options, key=f"security_{internet_key}")
        online_backup = st.selectbox("Online backup", internet_options, key=f"backup_{internet_key}")
        device_protection = st.selectbox("Device protection", internet_options, key=f"protection_{internet_key}")
        tech_support = st.selectbox("Tech support", internet_options, key=f"support_{internet_key}")
        streaming_tv = st.selectbox("Streaming TV", internet_options, key=f"tv_{internet_key}")
        streaming_movies = st.selectbox("Streaming movies", internet_options, key=f"movies_{internet_key}")

with col3:
    with st.container(border=True):
        st.markdown("#### 03 · Contract & billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
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
            "Monthly charges ($)", min_value=0.0, max_value=1000.0, value=75.0, step=1.0
        )
        total = st.number_input(
            "Total charges ($)", min_value=0.0, max_value=100000.0, value=900.0, step=10.0
        )
        st.caption("Amounts are in the dataset's original currency (USD).")

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
_, action_col = st.columns([2, 1])
with action_col:
    submitted = st.button("Estimate churn probability →", type="primary", use_container_width=True)

# Clear outdated predictions when any input changes; show only a result for the
# exact profile that was submitted.
if submitted:
    try:
        probability = predict_churn(customer, model)
    except ValueError as exc:
        st.error(f"Please review the customer profile: {exc}")
        st.stop()
    except Exception:
        st.error("Prediction failed. Please check the model and input feature schema.")
        st.stop()
    st.session_state["last_prediction"] = {
        "customer": customer.copy(),
        "probability": probability,
    }
