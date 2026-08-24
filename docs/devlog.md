# Development Log

## Session 3 — 2026-06-07 — Data exploration
**Done:** Loaded the Telco dataset; checked shape (7,043 × 21), dtypes, missing values, target balance.
**Key findings / decisions:** Churn ~73/27 (mild imbalance) → accuracy is a poor metric, plan to use precision/recall. TotalCharges stored as text with 11 hidden blanks, all tenure-0 new customers. Decision: convert to numeric and fill the 11 with 0 in Session 5.
## Session 4 — 2026-06-08 — API skeleton + deployment
**Done:** Built a minimal FastAPI app (/health + dummy /predict), verified locally via /docs, deployed to Render's free tier from main. App is live.
**Key findings / decisions:** Render appended a suffix to the URL because the plain name was taken — real URL is telco-churn-prediction-lhdg.onrender.com. Free tier cold-starts after inactivity (noted in README).
**Next:** Session 5 — clean the data (convert TotalCharges to numeric, fill the 11 tenure-0 blanks with 0) and build the preprocessing pipeline in src/.
## Session 6 — 2026-06-10 — Baseline model
**Done:** Built src/baseline_model.py — loads/cleans the data, does an 80/20 stratified train/test split, and trains a LogisticRegression inside a sklearn pipeline (StandardScaler + OneHotEncoder via ColumnTransformer). Also re-added preprocessing.py, which had gone missing from the branch.
**Baseline metrics (test set):** churn-class precision 0.66, recall 0.56, F1 0.60; ROC AUC 0.842; accuracy 0.81 (inflated by class imbalance). Weak spot: catches only 56% of churners.
## Session 7 — 2026-06-18 — Model comparison and selection
**Done:** Built a modular experiment framework (experiment_utils.py + experiments.py): a shared split/preprocessor/evaluator plus one function per experiment. Compared baseline LogReg, class_weight, gradient boosting, tuned LogReg (GridSearchCV), feature engineering, and threshold variants. Saved results to results/comparison.csv and the chosen model to models/churn_model.joblib.
**Key findings:** AUC capped ~0.842 across all logistic variants (dataset ceiling). Tuning and feature engineering gave no gain; gradient boosting slightly worse. Only the precision/recall operating point (via class_weight or threshold) moved meaningfully. Chose baseline LogReg at threshold 0.3 (recall 0.82, precision 0.52) — prioritising catching churners.
## Session 8 — 2026-06-25 — Wire model into API + tests
**Done:** Replaced the dummy /predict with the real model: loads churn_model.joblib at startup, accepts all 19 features (Pydantic validation), returns Yes/No at threshold 0.3 plus probability. Verified locally (high-risk customer 0.61 → Yes, loyal customer 0.03 → No). Wrote pytest tests in tests/test_api.py: health check, valid-prediction shape, and malformed-input rejection (422). All 3 pass.
**Key findings / decisions:** TestClient needs httpx (added to requirements). Tests assert response *shape*, not exact labels, to avoid brittleness.
## Session 9 — 2026-06-25 — Streamlit app + README
**Done:** Built app.py, a Streamlit front-end with a form for all 19 features that loads the model directly and shows a colour-coded churn prediction at threshold 0.3. Verified high-risk vs loyal customers predict differently. Wrote the full README: live links, approach, results, testing, structure, run instructions.
**Key findings / decisions:** Streamlit app loads the model directly rather than calling the API — simpler and more robust for a demo, and the API remains a standalone deliverable.
**Next:** Session 10 — deploy the Streamlit app, fill in its README link, polish the repo, final CV write-up.

## Session 11 — Review fixes
**Done:** Reconciled README metrics against results (README had overstated recall). Confirmed the feature-engineering null result was genuine, not a bug — the two derived features reached the model (47 vs 45 coefficients) but were assigned near-zero weight. Replaced ad-hoc threshold selection with a proper 60/20/20 split: threshold chosen on validation (F1 peaked at 0.35), then evaluated once on the untouched test set. Fixed function-name typos, made the threshold comparison consistent between evaluation and API, made the model path file-relative, trimmed requirements to direct dependencies, renamed response keys to snake_case, added a known-prediction test and three example payloads.
**Key findings / decisions:** The original 0.3 threshold was selected on test data — mild leakage. Validated selection gives 0.35, with honest test metrics: precision 0.545, recall 0.711, F1 0.617, AUC 0.843.