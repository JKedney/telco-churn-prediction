from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Telco Churn Prediction API")


# Health-check: visit this to confirm the service is alive
@app.get("/health")
def health():
    return {"status": "ok"}


# Describes the input the /predict endpoint expects
class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str


# Dummy prediction for now — the real model gets wired in at Session 8
@app.post("/predict")
def predict(customer: Customer):
    return {"churn_prediction": "No", "probability": 0.0}