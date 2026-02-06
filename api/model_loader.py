import joblib

MODEL_PATH = "models/churn_model_v1.pkl"

def load_model():
    return joblib.load(MODEL_PATH)