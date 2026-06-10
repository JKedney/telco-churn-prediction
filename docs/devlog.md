# Development Log

## Session 3 — 2026-06-07 — Data exploration
**Done:** Loaded the Telco dataset; checked shape (7,043 × 21), dtypes, missing values, target balance.
**Key findings / decisions:** Churn ~73/27 (mild imbalance) → accuracy is a poor metric, plan to use precision/recall. TotalCharges stored as text with 11 hidden blanks, all tenure-0 new customers. Decision: convert to numeric and fill the 11 with 0 in Session 5.
## Session 4 — 2026-06-08 — API skeleton + deployment
**Done:** Built a minimal FastAPI app (/health + dummy /predict), verified locally via /docs, deployed to Render's free tier from main. App is live.
**Key findings / decisions:** Render appended a suffix to the URL because the plain name was taken — real URL is telco-churn-prediction-lhdg.onrender.com. Free tier cold-starts after inactivity (noted in README).
## Session 5 — 2026-06-08 — Data cleaning function
**Done:** Built src/preprocessing.py with load_data() and clean_data(). clean_data converts TotalCharges to numeric and fills the 11 tenure-0 blanks with 0, drops customerID, and maps Churn to 1/0. Verified: 7043 × 20, no missing TotalCharges, Churn as 0/1.
**Key findings / decisions:** Kept encoding/scaling out of this function — those go in the model pipeline so the API transforms new data identically to training.
**Next:** Session 6 — train/test split and a baseline model.