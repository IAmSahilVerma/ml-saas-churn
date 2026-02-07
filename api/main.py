from fastapi import FastAPI
import torch
import numpy as np
from api.schemas import ChurnInput, ChurnOutput
from api.model_loader import load_preprocessor, load_model
from api.metrics_loader import get_latest_metrics

app = FastAPI(title="Churn Prediction API")

preprocessor = load_preprocessor()
model_type = "xgb"
model = load_model(model_type=model_type)

@app.get("/health")
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnOutput)
def predict(input_data: ChurnInput):
    # Convert Pydantic to DataFrame
    import pandas as pd
    df = pd.DataFrame([input_data.dict()])
    
    # Preprocess
    X = preprocessor.transform(df)
    
    # Predict
    if model_type in ["logreg", "xgb"]:
        prob = model.predict_proba(X)[:, 1][0]
    else:
        X_tensor = torch.FloatTensor(X.values)
        prob = model(X_tensor).detach().numpy()[0, 0]
        
    # Risk band
    if prob < 0.33:
        risk = "Low"
    elif prob < 0.66:
        risk = "Medium"
    else:
        risk = "High"
        
    return ChurnOutput(churn_probability=float(prob), risk_band=risk)

@app.get("/metrics")
def metrics():
    return get_latest_metrics()