# Customer Churn Prediction | End-to-End Machine Learning

**An end-to-end machine learning project for customer churn prediction, combining exploratory data analysis, scikit-learn, PySpark MLlib, automated testing, and interactive deployment with Streamlit.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://fama-cisse-churn-prediction.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![CI](https://github.com/zahrababacar6c/CustomerChurnPrediction/actions/workflows/tests.yml/badge.svg)](https://github.com/zahrababacar6c/CustomerChurnPrediction/actions/workflows/tests.yml)

**[Live Application](https://fama-cisse-churn-prediction.streamlit.app) · [Source Code](https://github.com/zahrababacar6c/CustomerChurnPrediction) · [Training Notebooks](notebooks/)**

---

## 1. Project Overview

Customer churn is an important challenge for subscription-based businesses. Identifying customers who may discontinue their services can help organizations prioritize retention efforts and better understand customer behavior.

This project develops and evaluates machine learning approaches to estimate customer churn risk using the public **IBM Telco Customer Churn dataset**.

It demonstrates a practical machine learning workflow, from data preparation and model evaluation to reusable inference code, automated tests, and interactive deployment.

### Project Objectives

- Explore customer characteristics associated with churn.
- Prepare and transform customer data for machine learning.
- Train and evaluate classification models using scikit-learn.
- Implement an additional distributed machine learning workflow using PySpark MLlib.
- Evaluate classification performance with metrics appropriate for an imbalanced target.
- Deploy a trained scikit-learn pipeline through an interactive Streamlit application.
- Improve code reliability through automated testing and GitHub Actions.

**Scope:** This is an educational and portfolio project using publicly available data. It is not a production customer retention system.

---

## 2. Business Problem

### Problem Statement

A telecommunications provider wants to identify customers who may cancel their subscriptions.

Rather than treating every customer as equally likely to leave, a predictive model can estimate churn risk from available customer characteristics.

Potential business applications include:

- Prioritizing customers for retention outreach.
- Supporting customer segmentation and risk analysis.
- Exploring relationships between contract types, service subscriptions, billing characteristics, and churn.
- Informing the design of future retention strategies.

### Machine Learning Task

The problem is formulated as a **binary classification task**.

| Target | Meaning |
|---|---|
| `Churn = Yes` | Customer discontinued service |
| `Churn = No` | Customer retained service |

The deployed model produces an estimated probability of churn.

A decision threshold of **0.50** is used for the scikit-learn demonstration.

This threshold is a baseline and has not been optimized against real business costs.

---

## 3. Dataset

**Dataset:** IBM Telco Customer Churn

**Source:** [IBM Telco Customer Churn repository](https://github.com/IBM/telco-customer-churn-on-icp4d)

| Property | Description |
|---|---|
| Records | 7,043 customers |
| Original columns | 21 |
| Target variable | `Churn` |
| Problem type | Binary classification |
| Data type | Customer demographics, services, contracts, and billing |

The dataset includes features such as:

- Customer tenure and demographic characteristics.
- Phone and internet service subscriptions.
- Online security, technical support, and streaming services.
- Contract duration.
- Payment method and paperless billing.
- Monthly and total charges.

### Data Quality

The original `TotalCharges` field contains **11 blank values**, all associated with customers whose `tenure` equals zero.

The notebooks document the treatment of these records during preprocessing.

The dataset is retrieved from its public source by the notebooks. No confidential employer or customer data is used.

---

## 4. Exploratory Data Analysis

Exploratory analysis investigates how customer characteristics relate to observed churn.

One notable finding is the variation in churn rates across contract types.

| Contract Type | Observed Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

In this dataset, customers with month-to-month contracts exhibit substantially higher observed churn rates than customers with longer contracts.

**Interpretation:** These are descriptive associations, not evidence that contract duration itself causes churn.

Additional exploratory analysis and visualizations are available in the scikit-learn notebook.

---

## 5. Machine Learning Methodology

Two machine learning workflows were implemented.

### 5.1 scikit-learn Workflow

**Notebook:** [01_customer_churn_sklearn.ipynb](notebooks/01_customer_churn_sklearn.ipynb)

The scikit-learn workflow includes:

1. Data loading and quality checks.
2. Exploratory data analysis.
3. Feature preprocessing.
4. Stratified train/test splitting.
5. Pipeline-based transformations and model training.
6. Five-fold cross-validation for model selection.
7. Evaluation on a held-out test set.

Models evaluated:

- Logistic Regression
- Random Forest Classifier

**Selected model: Logistic Regression**

The selected fitted preprocessing and classification pipeline is serialized using Joblib and used by the Streamlit application.

### 5.2 PySpark MLlib Workflow

**Notebook:** [02_customer_churn_pyspark.ipynb](notebooks/02_customer_churn_pyspark.ipynb)

The PySpark workflow demonstrates:

1. Data preparation with PySpark.
2. Feature transformation for MLlib.
3. Train/validation/test splitting.
4. Logistic Regression and Random Forest training.
5. Validation-based model selection.
6. Decision threshold selection.
7. Held-out test evaluation.

**Selected model: Random Forest**

The PySpark workflow is included to demonstrate machine learning with Spark. It is not the model deployed in Streamlit.

---

## 6. Model Evaluation

Results reported from the executed notebooks:

| Metric | scikit-learn | PySpark MLlib |
|---|---:|---:|
| Selected model | Logistic Regression | Random Forest |
| ROC-AUC | 0.8418 | 0.8386 |
| Average precision | 0.6334 | 0.6508 |
| Accuracy | 0.74 | 0.79 |
| Churn precision | 0.51 | 0.59 |
| Churn recall | 0.78 | 0.63 |
| Churn F1-score | 0.61 | 0.61 |
| Decision threshold | 0.50 | 0.40 |

### Evaluation Interpretation

**ROC-AUC** measures how well the model ranks churned customers above non-churned customers across classification thresholds.

**Average precision** summarizes precision-recall performance and is particularly relevant when the target classes are imbalanced.

**Recall** measures the proportion of actual churn cases identified by the classifier at the selected threshold.

**Precision** measures the proportion of predicted churn cases that actually belong to the churn class.

The scikit-learn model achieves a churn recall of **0.78** at the baseline threshold, while the PySpark model achieves **0.63** at its selected threshold.

### Important Comparison Limitation

These results are **not a controlled head-to-head benchmark**.

The notebooks use different data splitting strategies and decision thresholds. Therefore, differences in performance cannot be attributed solely to scikit-learn versus PySpark or to the selected algorithms.

A controlled comparison would require identical evaluation datasets and a consistent model-selection and threshold-selection protocol.

---

## 7. Application Architecture

The deployed application uses the trained scikit-learn pipeline.

**Inference workflow:**

```text
Customer Input
      |
      v
Streamlit Interface
(app/streamlit_app.py)
      |
      v
Customer Input Validation
(app/predict.py)
      |
      v
Feature DataFrame
      |
      v
Serialized scikit-learn Pipeline
(models/customer_churn_pipeline.joblib)
      |
      v
Estimated Churn Probability
      |
      v
Streamlit Prediction Results
```

### Main Components

**Streamlit interface — `app/streamlit_app.py`**

Provides the customer profile form, displays estimated churn probability, applies the baseline decision threshold, and presents the submitted features.

**Prediction module — `app/predict.py`**

Centralizes model loading, customer service combination validation, and probability prediction.

**Trained model — `models/customer_churn_pipeline.joblib`**

Contains the fitted preprocessing and Logistic Regression pipeline.

The model artifact is intentionally version-controlled to support deployment without retraining.

**Automated tests — `tests/test_prediction.py`**

Verify model loading, probability output, and selected invalid customer service combinations.

---

## 8. Interactive Streamlit Application

**[Launch the Customer Churn Prediction App](https://fama-cisse-churn-prediction.streamlit.app)**

The application allows users to enter a customer profile and estimate the probability of churn.

### Features

- Interactive customer information form.
- Input validation for phone and internet service combinations.
- Loading of the pre-trained scikit-learn pipeline.
- Estimated churn probability.
- Baseline classification threshold of 50%.
- Display of submitted model features.

### Functional Testing

During manual application testing, three customer profiles produced the following estimated probabilities:

| Test Profile | Estimated Churn Probability |
|---|---:|
| High-risk example | 85.6% |
| Low-risk example | 2.0% |
| Intermediate-risk example | 21.8% |

These values are example model outputs from manual testing, not validated customer outcomes or guarantees of future churn.

The application was also checked after integrating the reusable prediction module, with a successful prediction and passing GitHub Actions workflow.

---

## 9. Repository Structure

```text
CustomerChurnPrediction/
|
|-- .github/
|   |-- workflows/
|       |-- tests.yml
|
|-- app/
|   |-- predict.py
|   |-- streamlit_app.py
|
|-- archive/
|
|-- data/
|   |-- README.md
|
|-- models/
|   |-- README.md
|   |-- customer_churn_pipeline.joblib
|
|-- notebooks/
|   |-- 01_customer_churn_sklearn.ipynb
|   |-- 02_customer_churn_pyspark.ipynb
|
|-- tests/
|   |-- test_prediction.py
|
|-- .devcontainer/
|-- .gitignore
|-- README.md
|-- requirements.txt
```

---

## 10. Getting Started

### Prerequisites

- Python 3.11
- Git
- A Python virtual environment (recommended)

A compatible Java runtime is also required for running PySpark locally.

### Clone the Repository

```bash
git clone https://github.com/zahrababacar6c/CustomerChurnPrediction.git

cd CustomerChurnPrediction
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```powershell
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

The current requirements include dependencies for the notebooks, Streamlit application, and automated tests.

The serialized model was created with scikit-learn 1.6.1, which is pinned in `requirements.txt` for compatibility.

### Run the Streamlit Application

```bash
python -m streamlit run app/streamlit_app.py
```

The application loads the committed model artifact from the `models/` directory.

Retraining is not required to launch the interactive demonstration.

### Run the Notebooks

Open the notebooks using Jupyter or Google Colab.

Run the cells in order to reproduce the documented analysis and training workflows.

The notebooks retrieve the public dataset from the IBM repository, so internet access is required.

Reproducing the saved artifact exactly may also depend on compatible dependency versions, data availability, and training configuration.

---

## 11. Automated Testing

The project uses **pytest** to validate key components of the prediction module.

The existing tests cover:

- Successful loading of the trained model.
- Generation of a probability between 0 and 1.
- Rejection of an invalid phone service combination.
- Rejection of an invalid internet service combination.

### Run Tests Locally

From the repository root:

```bash
python -m pytest -v
```

The current test suite provides baseline inference checks. It is not an exhaustive validation of the input schema or all possible failure cases.

---

## 12. Continuous Integration with GitHub Actions

The project includes a GitHub Actions workflow:

**[ML Pipeline Tests](.github/workflows/tests.yml)**

The workflow runs on:

- Pushes to the `main` branch.
- Pull requests targeting `main`.
- Manual workflow dispatch.

### CI Workflow

```text
Push / Pull Request
        |
        v
GitHub Actions
        |
        v
Set Up Python 3.11
        |
        v
Install Dependencies
        |
        v
Run pytest
        |
        v
Test Results
```

This workflow helps detect regressions in the tested prediction functionality.

**CI scope:** The workflow runs automated tests. It does not retrain both machine learning models, execute all notebook cells, or perform a full end-to-end deployment test.

---

## 13. Model Reproducibility and Artifact Management

The deployed scikit-learn pipeline is stored at:

`models/customer_churn_pipeline.joblib`

The artifact is intentionally included in version control because the Streamlit application loads it during inference.

Other generated model artifacts are ignored by default through `.gitignore`.

The trained artifact was serialized with scikit-learn 1.6.1.

**Security note:** Joblib uses Python object serialization. Only load model artifacts from trusted sources because deserialization can execute arbitrary code.

For additional information, see [Model Artifacts Documentation](models/README.md).

---

## 14. Limitations

This project has several important limitations.

**Dataset representativeness**

The model is trained on a public demonstration dataset. Its performance may not generalize to other telecommunications providers, markets, or customer populations.

**Probability calibration**

The estimated churn probabilities have not been independently established as calibrated probabilities.

**Decision threshold**

The deployed 50% threshold is a baseline. A business deployment would require threshold selection based on retention costs, intervention capacity, and expected value.

**Framework comparison**

The scikit-learn and PySpark workflows use different evaluation configurations. Their reported results should not be interpreted as a controlled comparison.

**Interpretability**

The scikit-learn notebook includes post-hoc permutation importance analysis on the test set. This should not be used to tune the model or interpreted as causal evidence.

**Production readiness**

The application does not implement a complete production monitoring, model governance, or retraining system.

**Automated test coverage**

The current test suite covers selected prediction behaviors but does not yet exhaustively test schema validation, unexpected values, or deployment integration.

---

## 15. Future Improvements

Potential improvements include:

- Expanding input validation and automated test coverage.
- Separating application, development, and notebook dependencies.
- Creating a controlled scikit-learn versus PySpark comparison using consistent splits.
- Evaluating probability calibration.
- Selecting decision thresholds using explicit business costs.
- Developing a model card documenting intended use, evaluation, and limitations.
- Adding model monitoring and data drift analysis.
- Improving experiment tracking and training reproducibility.

These are future improvements, not features currently implemented.

---

## 16. Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data manipulation | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine learning | scikit-learn |
| Distributed machine learning | PySpark MLlib |
| Model serialization | Joblib |
| Interactive application | Streamlit |
| Automated testing | pytest |
| Continuous integration | GitHub Actions |
| Version control | Git, GitHub |

---

## 17. Author

**Fama Cissé**

*Data Science Engineer | Machine Learning & Applied AI*

[GitHub](https://github.com/zahrababacar6c) · [LinkedIn](https://linkedin.com/in/fz6c)

---

*This repository demonstrates applied machine learning, reproducible analytical workflows, reusable inference code, automated testing, and interactive model deployment using publicly available data.*
