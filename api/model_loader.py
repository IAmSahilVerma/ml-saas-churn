import joblib
import torch
import os

from training.features import Preprocessor
from training.train import MLP

MODELS_DIR = "models"

def load_preprocessor():
    return Preprocessor.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))

def load_model(model_type="xgb"):
    files = [f for f in os.listdir(MODELS_DIR) if f.startswith(model_type)]
    model = os.path.join(MODELS_DIR, files[-1])
    
    if model_type == "logreg" or model_type == "xgb":
        return joblib.load(model)
    elif model_type == "mlp":
        preprocessor = load_preprocessor()
        input_dim = len(preprocessor.ohe.get_feature_names_out(preprocessor.cat_features)) + len(preprocessor.num_features)
        mlp_model = MLP(input_dim)
        mlp_model.load_state_dict(torch.load(model))
        mlp_model.eval()
        return mlp_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")