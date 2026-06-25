# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to churn, served through a deployed REST API and an interactive web app.

## Live demo
- **Web app:** [add your Streamlit URL after deployment]
- **API docs:** https://telco-churn-prediction-lhdg.onrender.com/docs

_Both are hosted on free tiers — the first request after inactivity may take ~30–60 seconds to wake up._

## What it does
Given a customer's details (contract type, tenure, charges, services, etc.), the model estimates their probability of churning. It's deployed two ways: a FastAPI service for programmatic access, and a Streamlit app for a point-and-click interface.

## Tech stack
Python · pandas · scikit-learn · FastAPI · Streamlit · pytest · Render

## Approach
1. **Data exploration** — found that `TotalCharges` was stored as text with 11 hidden blank values, all belonging to brand-new (tenure-0) customers.
2. **Cleaning** — a reusable function converts `TotalCharges` to numeric, handles those blanks, and encodes the target. Used by training and serving alike.
3. **Modelling** — compared logistic regression, gradient boosting, hyperparameter tuning, feature engineering and threshold adjustment in a modular experiment framework (results in `results/comparison.csv`).
4. **Serving** — the chosen model is wrapped in a scikit-learn pipeline (so preprocessing and prediction stay consistent) and exposed via API and web app.

## Results
All logistic-regression variants plateaued at ROC AUC ≈ 0.842 — the dataset's ceiling. Tuning and feature engineering gave no meaningful gain, and gradient boosting was slightly behind.

The final model is a logistic regression used at a decision threshold of **0.3**, deliberately prioritising recall: it catches ~82% of churners (recall 0.82) at precision 0.52. This favours identifying at-risk customers, on the reasoning that retention outreach is cheap relative to losing a customer.

## Testing
API tested with pytest (`tests/test_api.py`): health check, prediction response shape, and rejection of malformed input.

## Project structure
- `data/` — dataset
- `notebooks/` — exploratory analysis
- `src/` — cleaning, experiments, and the API
- `tests/` — API tests
- `models/` — saved model artifact
- `results/` — model comparison table
- `app.py` — Streamlit web app

## Running locally
```
pip install -r requirements.txt
uvicorn src.main:app --reload      # API at http://127.0.0.1:8000/docs
streamlit run app.py               # web app at http://localhost:8501
```