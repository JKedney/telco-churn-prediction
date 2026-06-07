# Development Log

## Session 3 — 2026-06-07 — Data exploration
**Done:** Loaded the Telco dataset; checked shape (7,043 × 21), dtypes, missing values, target balance.
**Key findings / decisions:** Churn ~73/27 (mild imbalance) → accuracy is a poor metric, plan to use precision/recall. TotalCharges stored as text with 11 hidden blanks, all tenure-0 new customers. Decision: convert to numeric and fill the 11 with 0 in Session 5.
**Next:** Session 4 — build and deploy a minimal API skeleton (the de-risk step).