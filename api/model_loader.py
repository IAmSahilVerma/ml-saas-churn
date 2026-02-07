import joblib
import torch
import os
import json
from training.features import Preprocessor
from training.train import MLP

MODELS_DIR = "models"

def load_registry():
    with open(os.path.join(MODELS_DIR, "registry.json")) as f:
        return json.load(f)

def load_preprocessor():
    return Preprocessor.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))

def load_model(model_name: str | None = None):
    registry = load_registry()
    
    if model_name is None:
        model_name = registry["default"]
        
    if model_name not in registry["models"]:
        raise ValueError(f"Model '{model_name}' no found")
    
    model_info = registry["models"][model_name]
    model_path = os.path.join(MODELS_DIR, model_info["path"])
    
    if model_info["type"] in ["logreg", "xgb"]:
        return joblib.load(model_path)
    elif model_info["type"] == "mlp":
        preprocessor = load_preprocessor()
        input_dim = len(preprocessor.ohe.get_feature_names_out(preprocessor.cat_features)) + len(preprocessor.num_features)
        mlp_model = MLP(input_dim)
        mlp_model.load_state_dict(torch.load(model_path))
        mlp_model.eval()
        return mlp_model
    else:
        raise ValueError(f"Unsupported model type")