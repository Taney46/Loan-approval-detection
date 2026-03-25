# Loan Approval Prediction Using Supervised Machine Learning

**Group 4** — Juanita Chepchumba · Timothy Ng'etich · Hassan Kuloba  
University of Eastern Africa, Baraton | COSC440: Artificial Intelligence

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest** ✓ | 77.24% | 84.34% | 82.35% | **83.33%** | **80.84%** |
| XGBoost | 77.24% | 86.08% | 80.00% | 82.93% | 78.61% |
 
Random Forest was selected as the final model based on highest F1-score and AUC-ROC.

---

## Dataset

> **The dataset is not included in this repository.**  
> Download it from Kaggle before running the scripts:
>
> 📥 [Analytics Vidhya — Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
>
> Place the downloaded files in the `data/` folder as `train.csv` and `test.csv`.

Originally provided by Analytics Vidhya (2014) as part of a data science hackathon. The training set contains 614 records with labels; the test set contains 367 records without labels (competition format).

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Numerical | Number of dependents (0–3+) |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Monthly income |
| CoapplicantIncome | Numerical | Co-applicant monthly income |
| LoanAmount | Numerical | Loan amount in thousands |
| Loan_Amount_Term | Numerical | Term in months |
| Credit_History | Binary | 1 = good history, 0 = bad |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| **Loan_Status** | **Target** | **Y = Approved, N = Rejected** |

---

## Repository Structure

```
loan-approval-detection/
│
├── data/                        #download from Kaggle (link above)
│   ├── train.csv
│   └── test.csv
│
├── src/                         # Source scripts — run in order
│   ├── EDA.py                   ← Step 1: EDA & preprocessing
│   └── trainingmodel.py         ← Step 2: Model training & evaluation
│
├── outputs/
│   └── figures/                 # All 11 EDA & results figures
│       ├── fig_01_target_distribution.png
│       ├── fig_02_numerical_univariate.png
│       ├── fig_03_categorical_univariate.png
│       ├── fig_04_categorical_bivariate.png
│       ├── fig_05_numerical_bivariate.png
│       ├── fig_06_correlation_heatmap.png
│       ├── fig_07_log_transform.png
│       ├── fig_08_metrics_comparison.png
│       ├── fig_09_confusion_matrices.png
│       ├── fig_10_roc_curves.png
│       └── fig_11_feature_importance.png
│
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install the following:
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
```

### 2. Download the dataset
Download from the [Kaggle link above](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) and place `train.csv` and `test.csv` in the `data/` folder.

### 3. Run Step 1 — EDA & Preprocessing
```bash
python src/EDA.py
```
Outputs: 7 EDA figures in `outputs/figures/` + preprocessed CSVs in `outputs/`

### 4. Run Step 2 — Model Training & Evaluation
```bash
python src/trainingmodel.py
```
Outputs: 4 results figures in `outputs/figures/` 

---

## Methodology Summary

1. **EDA** — Distribution analysis, bivariate plots, correlation heatmap, log-transform comparison
2. **Preprocessing** — Mode/median imputation, IQR outlier capping, log transformation, label + one-hot encoding, StandardScaler
3. **Class imbalance** — SMOTE applied to 80% training partition only (balanced to 337/337)
4. **Models** — Random Forest (bagging) vs XGBoost (boosting), tuned via Grid Search + 5-fold stratified CV
5. **Evaluation** — Accuracy, Precision, Recall, F1-score, AUC-ROC, Confusion Matrix, Feature Importance

imbalanced-learn>=0.10.0
```
