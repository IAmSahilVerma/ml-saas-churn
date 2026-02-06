from fastapi import FastAPI
from api.schemas import CustomerInput
from api.model_loader import load_model

app = FastAPI(title="Churn Prediction API")

model = load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "1.0"}

@app.post("/predict")
def predict(data: CustomerInput):
    # Placeholder until preprocessing is wired
    return {
        "churn_probability": 0.5,
        "risk_band": "MEDIUM"
    }