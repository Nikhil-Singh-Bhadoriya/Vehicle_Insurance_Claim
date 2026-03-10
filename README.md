# 🚗 Vehicle Insurance Claim Fraud Detection using Classification Algorithms

A machine-learning project focused on detecting potentially fraudulent vehicle insurance claims using classification models, feature engineering and analytical insights.

---

## 📌 Project Overview

This project develops a full workflow: collecting insurance claims data (claim amount, claimant demographics, vehicle details, previous claims, police report data etc.), performing exploratory analysis, engineering features, training classification models to detect fraud, evaluating performance, and deriving business-actionable insights. The goal is to build a system to flag high-risk claims for further investigation and reduce fraud losses.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **Environment:** Jupyter Notebook / Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection

Dataset includes features such as: claim amount, claim type, vehicle age, driver age, number of previous claims, police report filed (yes/no), witness statements, days until claim reported, region, and target variable indicating whether the claim was flagged as fraudulent.

### 2. Exploratory Data Analysis (EDA)

* Distribution of fraudulent vs non-fraudulent claims by features (claim amount, driver age, vehicle age)
* Visualisations: boxplots, histograms, and heatmap of feature correlations
* Check for class imbalance (fraud cases often are much fewer)
* Outlier detection for unusually high claim amounts

### 3. Feature Engineering

* Encode categorical variables (claim type, region, vehicle type) via one-hot or label encoding
* Create derived features such as:

  * claim_amount / vehicle_age
  * days_until_reporting flag (early vs late)
  * previous_claims_count per driver
  * witness_count or police_report_flag as binary feature
* Scale numerical features where required
* Split into training/test sets (often stratified by fraud label) and possibly oversample fraud cases (SMOTE) or apply class weights

### 4. Modeling

Classification algorithms used include:

* **Logistic Regression** (baseline)
* **Random Forest Classifier** (strong performer)
* **Gradient Boosting (XGBoost/LightGBM)** for improved performance
* Hyper-parameter tuning (e.g., max_depth, n_estimators, learning_rate) via cross-validation

### 5. Evaluation

Metrics for fraud detection:

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix (particularly focus on false negatives: missed frauds)
* ROC-AUC
  **Result:** The best model achieved high recall and acceptable precision, helping catch fraud cases while limiting false alarms.

### 6. Insights & Business Application

* Key fraud indicators identified: unusually high claim amounts, late reporting, vehicle older than threshold, multiple prior claims, missing witness statements.
* Operational recommendations: flag claims with high risk score for manual investigation; deploy automated alerts; prioritise resources for drivers with previous claims or late-reporting behaviour.
* Model helps reduce investigation workload by focusing on high-probability fraudulent claims.

---

## 📁 Project Structure

```
Vehicle-Insurance-Claim-Fraud-Detection/
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│   └── fraud_detection_analysis.ipynb
│── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── evaluate.py
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* Late claim reporting and high claim amount relative to vehicle age were among the most influential predictors of fraud.
* Tree-based models (Random Forest, Gradient Boosting) outperformed logistic regression by capturing non-linear interactions and complex fraud patterns.
* Handling class imbalance via oversampling or class weights significantly improved recall of the fraud class.
* Feature importance analysis revealed that driver’s prior claim history and police report flag had strong predictive power.

---

## 🚀 Future Improvements

* Incorporate additional features like vehicle usage telemetry, driver behaviour data (telematics), damage image analysis for richer fraud detection.
* Expand to multi-modal data: images of damage, videos of incident, voice calls transcripts—use deep-learning for cross-modal fraud analysis.
* Deploy model via a dashboard or API for claims team: real-time score, explanation (SHAP values), investigation workflow integration.
* Use ensemble stacking (e.g., blend Random Forest + XGBoost + Neural Network) to improve detection performance.
* Monitor model drift over time as fraud tactics evolve; retrain and update features periodically.

