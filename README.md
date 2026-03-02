# Lead Scoring AI

Lead Scoring AI is an end-to-end machine learning project that helps a sales team decide which leads to follow first.

The system takes lead information (company, source, purchase history, deal value, response behavior) and predicts:
- conversion probability
- final class decision (`Convert` or `No Convert`)
- priority band (`High`, `Medium`, `Low`)

This repository includes:
- model training pipeline
- model comparison and evaluation
- API for prediction
- Streamlit dashboard for business users

---

## 1. Why this project exists
In real sales flow, teams get many leads but time is limited. Without a scoring system, sales teams may spend time on low-potential leads and miss high-potential ones.

This project solves that by:
- giving each lead a probability score
- converting that score into an action-friendly decision
- showing results in a dashboard and API

---

## 2. Problem type
This is a **binary classification** problem.

- Target column: `converted`
- `1` means lead converted
- `0` means lead did not convert

---

## 3. Dataset details
Dataset file:
- `data/lead_scoring_logistics_Dataset.csv`

Columns:
- `lead_code` (ID only; removed before training)
- `company_size`
- `industry`
- `region`
- `prev_purchases`
- `response_time`
- `last_contact`
- `source`
- `deal_value`
- `converted` (target)

Current dataset size (from metrics artifact):
- Rows: `500`
- Train: `300`
- Validation: `100`
- Test: `100`

---

## 4. Technology stack used
Language:
- Python

Data and ML:
- Pandas (`2.0.3`)
- Scikit-learn (`1.3.2`)
- Joblib (`1.3.2`)

Backend:
- FastAPI (`0.110.3`)
- Pydantic (`2.10.6`)
- Uvicorn (`0.30.6`)

Frontend:
- Streamlit (`1.33.0`)

Key dependency files:
- `requirements.txt`
- `pyproject.toml`

---

## 5. Project architecture
Core files:
- `src/lead_scoring/config.py` -> constants, file paths, feature lists
- `src/lead_scoring/train.py` -> full V3 training pipeline
- `src/lead_scoring/inference.py` -> load model + run prediction
- `api/main.py` -> FastAPI prediction service
- `dashboard/app.py` -> Streamlit UI
- `scripts/train.py` -> train command
- `scripts/smoke_test.py` -> basic runtime checks

Generated artifacts:
- `artifacts/best_model.joblib` -> final trained model
- `artifacts/metrics.json` -> full metrics + selected model metadata
- `artifacts/model_comparison.md` -> multi-model comparison table

---

## 6. How this project was built (step by step)

### Step 1: Load and prepare data
- Read CSV from `data/lead_scoring_logistics_Dataset.csv`
- Drop `lead_code` because it is an identifier, not business signal
- Split data into features (`X`) and target (`y`)

### Step 2: Create train/validation/test splits
- First split into train_val and test
- Then split train_val into train and validation
- Stratified splits are used to preserve class ratio

### Step 3: Feature engineering (V3)
Custom engineered features are created in pipeline, including:
- log transforms (`deal_value_log`, `company_size_log`)
- ratio features (`purchase_intensity`, `contact_freshness`, `deal_per_employee`)
- interaction features (`response_contact_product`, `deal_purchase_product`)
- combined categorical features (`industry_source`, `region_source`)

### Step 4: Preprocessing
Numeric pipeline:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`

Categorical pipeline:
- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

### Step 5: Multi-model training and tuning
Models compared:
- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting

Each model is tuned with `GridSearchCV` using ROC-AUC.

### Step 6: Calibration
After selecting best candidate model, calibration methods are compared:
- none
- sigmoid
- isotonic

Best calibration is selected by **lowest Brier score** on validation.

### Step 7: Threshold optimization
The model predicts probabilities. To make a binary decision, threshold is tuned from `0.20` to `0.80`.

Selection rule:
- maximize F1
- tie-break with higher precision

### Step 8: Final model and artifacts
- Refit final model on train+validation
- Evaluate on test set
- Save model and metrics artifacts

### Step 9: Serve predictions via API
FastAPI endpoints:
- `GET /health`
- `GET /model-info`
- `POST /predict`

`/predict` uses tuned threshold from `metrics.json`.

### Step 10: Dashboard usage
Streamlit dashboard allows:
- entering lead details
- loading scenario presets
- seeing probability, class, priority, and recommendation
- viewing model governance table
- exporting decision report JSON

---

## 7. Model comparison (current)
Comparison file:
- `artifacts/model_comparison.md`

Important:
- Section 7 numbers are for **model selection stage** (cross-validation + validation split).
- These are used to choose model/calibration before final holdout testing.

Current table summary:
- Extra Trees: CV ROC-AUC `0.6377`, Validation ROC-AUC `0.5546`, Brier `0.2530`
- Logistic Regression: CV ROC-AUC `0.6157`, Validation ROC-AUC `0.5353`, Brier `0.2543`
- Random Forest: CV ROC-AUC `0.6322`, Validation ROC-AUC `0.5177`, Brier `0.2545`
- Gradient Boosting: CV ROC-AUC `0.6529`, Validation ROC-AUC `0.4873`, Brier `0.2614`

Selected model in current run:
- `extra_trees`
- calibration: `sigmoid`

---

## 8. Current performance (honest)
From `artifacts/metrics.json`:
- ROC-AUC: `0.5677`
- F1: `0.5821`
- Precision: `0.4239`
- Recall: `0.9286`
- Accuracy: `0.44`
- Decision threshold: `0.36`

Important:
- Section 8 numbers are from the **final test set** (holdout evaluation after model selection).
- So values in Section 8 can differ from Section 7, and this is expected.

Important interpretation:
- This setup is recall-heavy (find more true converters)
- That increases false positives, so precision and accuracy are lower
- ROC-AUC is moderate, not state-of-the-art

This is an honest engineering project with a complete ML workflow, but model quality can improve with richer data.

---

## 9. How to run this project

### 9.1 Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### 9.2 Train
```bash
python3 scripts/train.py
```

### 9.3 Run API
```bash
uvicorn api.main:app --reload --port 8000
```

### 9.4 Run dashboard
```bash
streamlit run dashboard/app.py
```

### 9.5 Run smoke test
```bash
python3 scripts/smoke_test.py
```

Smoke test checks:
- valid payload works
- unknown categories are handled
- invalid payload is rejected

---

## 10. API request example
`POST /predict`

```json
{
  "company_size": 120,
  "industry": "Pharma",
  "region": "Mumbai",
  "prev_purchases": 3,
  "response_time": 14,
  "last_contact": 21,
  "source": "Website",
  "deal_value": 15000
}
```

---

## 11. Dashboard screenshots

### Executive Demo
![Executive Demo](assets/screenshots/executive_demo.png)

### Growth Inbound
Screenshot omitted due capture artifact in this environment.

### Cold Prospect
![Cold Prospect](assets/screenshots/cold_prospect.png)

### Large Slow Account
![Large Slow Account](assets/screenshots/large_slow_account.png)

---

## 12. Limitations and next improvements
Current limitations:
- small dataset size
- moderate ROC-AUC
- one global threshold for all business scenarios

Next improvements:
- more behavioral features from CRM activity
- threshold modes (`precision-first` and `recall-first`)
- periodic retraining and drift monitoring
- experiment tracking (MLflow/W&B)

---

## 13. Repository structure
```text
lead-scoring-ai/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── scripts/
│   ├── train.py
│   ├── smoke_test.py
│   └── capture_screenshots_mac.sh
├── src/lead_scoring/
│   ├── __init__.py
│   ├── config.py
│   ├── train.py
│   └── inference.py
├── data/
│   └── lead_scoring_logistics_Dataset.csv
├── artifacts/
│   ├── best_model.joblib
│   ├── metrics.json
│   └── model_comparison.md
├── assets/screenshots/
│   ├── executive_demo.png
│   ├── cold_prospect.png
│   └── large_slow_account.png
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```

---

## 14. License
This project is licensed under the MIT License.
