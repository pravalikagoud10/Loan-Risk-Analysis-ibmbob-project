# Loan Default Risk Prediction Using Machine Learning

**Author:** Pravalika Katam  
**Dataset:** Kaggle Loan Risk Analysis (148,670 rows, 34 columns)  
**Target:** `Status` — 0 = Non-Default, 1 = Default

---

## Project Structure

```
Loan_Risk_Analysis_Project/
├── Loan_Default.csv                          # Raw dataset
├── PRAVALIKA_KATAM_LoanRiskPrediction.ipynb  # Full notebook (Steps 1–5)
├── PRAVALIKA_KATAM_ProjectReport.docx        # Written project report
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
├── step1_data_inspection.py                  # Step 1 script
└── step2_cleaning_eda.py                     # Step 2 script
```

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook PRAVALIKA_KATAM_LoanRiskPrediction.ipynb
```

---

## Pipeline Overview

### Step 1 — Data Loading and Inspection
- Loaded CSV: 148,670 rows × 34 columns
- 14 columns had missing values (up to 26.7%)
- Zero duplicate rows
- Target confirmed: `Status` (0/1 binary)

### Step 2 — Data Cleaning
| Action | Detail |
|---|---|
| Dropped `ID`, `year` | Identifier and constant column |
| Dropped `rate_of_interest` | Missing for ALL 36,439 default rows — structural leakage |
| Dropped `Interest_rate_spread` | Derived from rate_of_interest — same leakage |
| Dropped `Upfront_charges` | Set post-risk-assessment — leakage |
| Dropped `credit_type` | EQUI category: 99.99% default rate — structural anomaly |
| Capped LTV > 200 → NaN | 17 rows with physically impossible LTV |
| Median imputation | `term`, `property_value`, `income`, `LTV`, `dtir1` |
| Mode imputation | `loan_limit`, `approv_in_adv`, `submission_of_application`, `age`, `loan_purpose`, `Neg_ammortization` |
| Rows deleted | **0** — all 148,670 preserved |

### Step 3 — EDA
- Target imbalance: 75.4% Non-Default / 24.6% Default (3:1 ratio)
- `loan_type = type2` has 34.5% default rate (above 24.6% average)
- `age > 74` has 30.0% default rate
- `Region = North-East` has 30.4% default rate
- LTV, property_value, dtir1 most correlated with default

### Step 4 — Preprocessing
- Ordinal encoding for `age` (7 ordered bins → 0–6)
- One-hot encoding for 19 remaining categorical columns
- Final feature matrix: **37 features**
- Train/test split: 80/20, stratified on `Status`
- StandardScaler applied for Logistic Regression only

### Step 5 — Model Training and Evaluation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7069 | 0.4363 | 0.6478 | 0.5214 | 0.7480 |
| Decision Tree | 0.8593 | 0.7377 | 0.6659 | 0.7000 | 0.8578 |
| Random Forest | 0.8740 | 0.7823 | 0.6773 | 0.7260 | 0.8808 |
| **Gradient Boosting** | **0.8953** | **0.9364** | **0.6169** | **0.7438** | **0.8925** |

**Best model:** Gradient Boosting — ROC-AUC = 0.8925, F1 = 0.7438

**Top predictive features (Random Forest importances):**
1. LTV (0.2475)
2. property_value (0.2211)
3. dtir1 (0.1900)
4. income (0.0558)
5. co-applicant_credit_type_EXP (0.0477)

---

## Notes on Data Leakage

This dataset contains columns that are **recorded after the loan outcome is known**, not before. Using them would let the model "cheat":

- **`rate_of_interest`**: Missing for 100% of default rows, 0% of non-default rows. Imputing median creates a perfect artificial separator.
- **`Interest_rate_spread`**: Directly derived from `rate_of_interest`.
- **`Upfront_charges`**: Set based on assessed loan risk.
- **`credit_type = EQUI`**: 15,297 of 15,298 rows are defaults — this is a structural data anomaly, not a predictive pattern.

All four were removed before training. The reported metrics reflect a **leak-free model**.

---

## Evaluation Strategy

Plain accuracy was **not used as the primary metric** because the dataset is imbalanced (3:1). A model that predicts "Non-Default" for every row would achieve 75.4% accuracy — misleadingly high. Instead:
- **ROC-AUC**: measures the model's ability to rank defaults above non-defaults across all thresholds
- **F1-score**: balances precision and recall for the minority (default) class
- `class_weight='balanced'` was used in all applicable models

---

## Environment

Tested on Python 3.9+. See `requirements.txt` for package versions.
