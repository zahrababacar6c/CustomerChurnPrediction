# Customer Churn Prediction

**End-to-End Machine Learning | scikit-learn · PySpark · Streamlit**

An end-to-end machine learning project that explores customer churn in the telecommunications industry, develops classification pipelines, evaluates predictive performance, and deploys an interactive churn prediction application.

**[Live Streamlit App](https://fama-cisse-churn-prediction.streamlit.app)** | **[scikit-learn Notebook](notebooks/01_customer_churn_sklearn.ipynb)** | **[PySpark Notebook](notebooks/02_customer_churn_pyspark.ipynb)**

## 1. Business Problem

Customer churn represents a challenge for telecommunications companies because customer departures can reduce recurring revenue and increase the need for new customer acquisition.

The objective of this project is to estimate the likelihood of customer churn using historical customer attributes, helping a hypothetical retention team identify customers who may benefit from proactive outreach.

**Key question:** Can machine learning help identify customers at risk of leaving before retention resources are allocated?

This project demonstrates predictive modeling rather than a validated commercial retention strategy.

## 2. Dataset

The project uses the publicly available [IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d).

- **Records:** 7,043 customers
- **Original columns:** 21
- **Target:** `Churn` (`Yes` or `No`)
- **Features:** customer tenure, contract type, monthly and total charges, subscribed services, payment method, and customer attributes

During data quality checks, 11 blank `TotalCharges` values were identified, all associated with customers whose tenure was zero. These values were replaced with zero under a documented assumption, and a missingness indicator was retained.

The customer identifier was excluded from model training. No private company data was used.

## 3. Exploratory Data Analysis

The analysis identified several associations between customer characteristics and churn.

| Contract type | Observed churn rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

Customers on month-to-month contracts exhibited substantially higher observed churn rates than customers with longer-term contracts.

Other variables explored include tenure, monthly charges, internet service, payment method, and technical support.

These patterns describe statistical associations, not causal relationships.

## 4. Machine Learning Approach

### scikit-learn

The scikit-learn workflow includes:

1. Data validation and exploratory analysis.
2. Feature preparation and missing-value handling.
3. Stratified training and test split.
4. Preprocessing pipelines with encoding, imputation, and scaling.
5. Five-fold stratified cross-validation.
6. Logistic Regression and Random Forest comparison.
7. Final evaluation on a held-out test set.
8. Pipeline serialization using Joblib.

**Selected model:** Logistic Regression, based on cross-validation ROC-AUC.

### PySpark MLlib

The PySpark workflow includes:

1. Data cleaning and feature engineering.
2. Stratified training, validation, and test splits.
3. Spark ML feature transformation pipelines.
4. Logistic Regression and Random Forest training.
5. Model selection on validation data.
6. Decision-threshold selection on validation data.
7. Final evaluation on an untouched test set.

**Selected model:** Random Forest.

PySpark was included to demonstrate a distributed machine learning workflow. The small dataset does not establish a performance or scalability advantage from distributed processing.

## 5. Model Performance

Results obtained from the executed notebooks:

| Metric | scikit-learn | PySpark |
|---|---:|---:|
| Selected model | Logistic Regression | Random Forest |
| ROC-AUC | 0.8418 | 0.8386 |
| Average precision | 0.6334 | 0.6508 |
| Accuracy | 0.74 | 0.79 |
| Churn precision | 0.51 | 0.59 |
| Churn recall | 0.78 | 0.63 |
| Churn F1-score | 0.61 | 0.61 |
| Decision threshold | 0.50 | 0.40 |

**Evaluation note:** These results are not a controlled head-to-head comparison. The two workflows use different data splits and threshold-selection strategies.

For customer retention, recall is particularly relevant because false negatives represent customers who churn but are not identified by the model. Precision must also be considered to avoid unnecessary outreach.

## 6. Application Architecture

The deployed application uses the trained scikit-learn pipeline.

**Training workflow:**

IBM Telco Dataset → Data Cleaning → Feature Engineering → Preprocessing Pipeline → Logistic Regression → Model Evaluation → Joblib Serialization

**Inference workflow:**

Customer Input → Streamlit Form → Input Validation → Saved scikit-learn Pipeline → Churn Probability → Prediction Display

The serialized pipeline includes the fitted preprocessing transformations and classification model, ensuring consistent feature processing during inference.

## 7. Streamlit Application

**[Launch the Customer Churn Prediction App](https://fama-cisse-churn-prediction.streamlit.app)**

The application allows users to enter customer information and obtain an estimated churn probability.

Features include:

- Interactive customer profile form
- Input consistency checks
- Real-time inference using the trained pipeline
- Estimated churn probability
- Classification relative to a 50% baseline threshold
- Display of submitted model features

### Functional Testing

Three illustrative customer profiles were tested through the deployed application:

| Test profile | Estimated churn score |
|---|---:|
| Recent customer, month-to-month contract | 85.6% |
| Long-tenure customer, two-year contract | 2.0% |
| Intermediate-tenure customer, one-year contract | 21.8% |

All three tests returned predictions successfully. The observed ordering was plausible, but these manual tests do not establish model calibration, robustness, or generalization.

The application is an educational demonstration, not a production decision system.

## 8. Repository Structure

```text
CustomerChurnPrediction/
├── app/
│   └── streamlit_app.py
├── notebooks/
│   ├── 01_customer_churn_sklearn.ipynb
│   └── 02_customer_churn_pyspark.ipynb
├── models/
│   ├── customer_churn_pipeline.joblib
│   └── README.md
├── data/
│   └── README.md
├── archive/
│   └── CustomerChurn.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## 9. Run Locally

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/zahrababacar6c/CustomerChurnPrediction.git
cd CustomerChurnPrediction

python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app/streamlit_app.py
```

The application loads the trusted, serialized scikit-learn model stored in the `models/` directory. Compatible Python and library versions are required.

The notebooks can also be executed separately in Google Colab. Local PySpark execution requires a compatible Java environment.

## 10. Limitations and Future Improvements

- The dataset is a public demonstration dataset and may not represent current telecom customer behavior.
- Churn probabilities have not been independently calibrated.
- The classification threshold should ultimately reflect retention costs and intervention value.
- Model performance should be monitored for changes in customer behavior and data distributions.
- Feature importance does not establish causality.
- A controlled scikit-learn versus PySpark benchmark requires identical data splits and evaluation protocols.
- Automated inference tests, model monitoring, and a model card are future improvements.

## 11. Technologies

**Data Analysis:** Python, pandas, NumPy, Matplotlib, Seaborn

**Machine Learning:** scikit-learn, Logistic Regression, Random Forest, cross-validation

**Distributed ML:** PySpark MLlib

**Deployment:** Streamlit, Joblib, GitHub, Streamlit Community Cloud

## Author

**Fama Cissé**  
Data Science Engineer | Machine Learning & Applied AI

[LinkedIn](https://linkedin.com/in/fz6c) · [GitHub](https://github.com/zahrababacar6c) · [Live Application](https://fama-cisse-churn-prediction.streamlit.app)
