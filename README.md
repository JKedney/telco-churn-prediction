# Telco Customer Churn Prediction

Predicting which telecom customers are likely to leave, served via a deployed API.

## Overview
_One or two sentences on the problem and what this project does._

## Dataset
_Source, size, and the target variable. (Fill in after exploration.)_

## Tech stack
Python · pandas · scikit-learn · FastAPI · pytest

## Project structure
- `data/` — dataset
- `notebooks/` — exploratory analysis
- `src/` — pipeline and API code
- `tests/` — unit tests

## Live demo
- API: https://telco-churn-prediction-lhdg.onrender.com
- Interactive docs: https://telco-churn-prediction-lhdg.onrender.com/docs

_Hosted on Render's free tier — the first request after a period of inactivity may take ~30–60 seconds while the service wakes up._

## How to run locally
```pip install -r requirements.txt```
_More steps to follow once there's something to run._

## Results
Several models were compared (full table in `results/comparison.csv`). All logistic-regression variants plateaued at ROC AUC ≈ 0.842 — the dataset's ceiling. Gradient boosting was slightly behind, and hyperparameter tuning and feature engineering gave no meaningful gain.

The chosen model is a logistic regression used at a decision threshold of **0.3**, prioritising recall: it catches ~82% of churners (recall 0.82) at precision 0.52. This trade favours catching at-risk customers, on the basis that retention outreach is cheap relative to losing a customer.

## Next steps
_What you'd improve with more time._