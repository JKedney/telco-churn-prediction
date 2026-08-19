from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_valid_response():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert body["churn"] in ["Yes", "No"]
    assert 0.0 <= body["Churn probability"] <= 1.0

def test_predict_rejects_invalid_data():
    bad_customer = {"gender": "Female", "tenure": "not a number"}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422

def test_predict_high_risk_customer_returns_yes():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    assert response.json()["churn"] == "Yes"