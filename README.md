# Customer Churn Prediction | scikit-learn & PySpark

An end-to-end machine learning case study using the public IBM Telco Customer Churn dataset. The project explores customer retention, builds classification pipelines, and compares a local scikit-learn workflow with a PySpark MLlib workflow.

## Business problem

Identify customers who may discontinue their service, enabling a hypothetical retention team to prioritize outreach. Predictions are exploratory; no production or causal claims are made.

## Dataset

- Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
- 7,043 customer records, 21 original columns
- Target: `Churn` (`Yes` / `No`)
- Original `TotalCharges` contains 11 blank entries, all with `tenure = 0`. The notebooks document the handling of these records.
- No private employer data is used.

## Project structure

```text
notebooks/
  01_customer_churn_sklearn.ipynb
  02_customer_churn_pyspark.ipynb
data/README.md
models/README.md
requirements.txt
.gitignore
README.md
```

## Methods

**scikit-learn:** exploratory analysis, preprocessing pipeline, stratified train/test split, cross-validation, Logistic Regression and Random Forest. Model selection uses training cross-validation.

**PySpark:** feature preparation, train/validation/test split, MLlib Logistic Regression and Random Forest, validation-based model and threshold selection.

## Results from executed notebooks

| Metric | scikit-learn | PySpark |
|---|---:|---:|
| Selected model | Logistic Regression | Random Forest |
| Test ROC-AUC | 0.8418 | 0.8386 |
| Test average precision | 0.6334 | 0.6508 |
| Test accuracy | 0.74 | 0.79 |
| Churn recall | 0.78 | 0.63 |
| Churn F1-score | 0.61 | 0.61 |
| Decision threshold | 0.50 | 0.40 |

**Important:** these are not controlled head-to-head benchmarks. The notebooks use different split strategies and decision thresholds, so performance differences cannot be attributed solely to the framework.

## Run locally or in Google Colab

1. Open either notebook in [Google Colab](https://colab.research.google.com/), or create a Python environment.
2. Install dependencies with `pip install -r requirements.txt` (local PySpark also requires a compatible Java runtime).
3. Run the cells in order. The notebooks retrieve the public CSV from the IBM repository; internet access is required.
4. Model artifacts are generated locally in `models/` and intentionally excluded from Git.

## Limitations and next steps

- Threshold choice should be tied to outreach cost and retention value.
- The scikit-learn notebook uses permutation importance on the test set for post-hoc interpretation; do not use it to tune the model.
- Reproducibility depends on compatible package versions and availability of the public dataset.
- Planned: consistent splits for a controlled comparison, model card, Streamlit demo, and automated tests.

## Author

Fama Cissé · Data Science Engineer  
[GitHub](https://github.com/zahrababacar6c) · [LinkedIn](https://linkedin.com/in/fz6c)
