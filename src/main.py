import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title = "Telco Churn Prediction")

model = joblib.load("models/churn_model.joblib")
Threshold = 0.35

@app.get("/health")
def health():
    return {"status": "ok"}

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    proba = model.predict_proba(data)[0, 1]
    prediction = "Yes" if proba >= Threshold else "No"
    return {"churn": prediction, "Churn probability": round(float(proba), 3), "Threshold": Threshold,}
            
            
