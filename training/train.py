import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from training.features import Preprocessor
import json
import os
from datetime import datetime

REGISTRY_PATH = "models/registry.json"

# ----------------------
# Helper: Registry
# ----------------------
def add_model_to_registry(model_name, model_type, file_path, metrics):
    """Add trained model entry to registry.json"""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
    else:
        registry = []

    existing_versions = [m["version"] for m in registry if m["name"] == model_name]
    version = max(existing_versions, default=0) + 1

    entry = {
        "name": model_name,
        "type": model_type,
        "version": version,
        "file_path": file_path,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat()
    }

    registry.append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

    print(f"Model {model_name} v{version} added to registry.")

# ----------------------
# Helper: Evaluation
# ----------------------
def evaluate_model(model, X, y, model_type="sklearn"):
    if model_type in ["sklearn", "xgboost"]:
        y_pred_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
    elif model_type == "pytorch":
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            y_pred_prob = model(X_tensor).numpy().flatten()
            y_pred = (y_pred_prob >= 0.5).astype(int)
    else:
        raise ValueError("Unknown model_type")
    
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    roc_acu = roc_auc_score(y, y_pred_prob)
    return {"precision": precision, "recall": recall, "roc_auc": roc_acu}

# ----------------------
# PyTorch MLP
# ----------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# ----------------------
# Main Training
# ----------------------
def main():
    # Set experiment
    experiment_name = "ChurnProject"
    mlflow.set_experiment(experiment_name)

    # Load raw data
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df_raw.columns = df_raw.columns.str.strip()
    df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce").fillna(0)

    # Load preprocessor
    preprocessor = Preprocessor.load("models/preprocessor.pkl")
    X = preprocessor.transform(df_raw)
    y = df_raw["Churn"].apply(lambda x: 1 if x in ["Yes", 1, "Y", "y"] else 0).values

    # ----------------------
    # Logistic Regression
    # ----------------------
    mlflow.start_run(run_name="logistic_regression")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X, y)
    metrics = evaluate_model(logreg, X, y)
    mlflow.log_params({"model": "LogisticRegression", "max_iter": 1000})
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    logreg_file = "models/logreg_model.pkl"
    joblib.dump(logreg, logreg_file)
    add_model_to_registry("logistic_regression", "sklearn", logreg_file, metrics)
    mlflow.end_run()
    print("Logistic Regression done:", metrics)

    # ----------------------
    # XGBoost
    # ----------------------
    mlflow.start_run(run_name="xgboost")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    xgb_model.fit(X, y)
    metrics = evaluate_model(xgb_model, X, y, model_type="xgboost")
    mlflow.log_params(xgb_model.get_params())
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    xgb_file = "models/xgb_model.pkl"
    joblib.dump(xgb_model, xgb_file)
    add_model_to_registry("xgboost", "xgboost", xgb_file, metrics)
    mlflow.end_run()
    print("XGBoost done:", metrics)

    # ----------------------
    # PyTorch MLP
    # ----------------------
    mlflow.start_run(run_name="pytorch_mlp")

    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y.reshape(-1, 1))
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    mlp_model = MLP(input_dim=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

    for epoch in range(10):
        mlp_model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            y_pred = mlp_model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.4f}")

    metrics = evaluate_model(mlp_model, X, y, model_type="pytorch")
    mlflow.log_params({"model": "MLP", "epochs": 10, "batch_size": 64, "lr": 0.001})
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    mlp_file = "models/mlp_model.pt"
    torch.save(mlp_model.state_dict(), mlp_file)
    add_model_to_registry("mlp", "pytorch", mlp_file, metrics)
    mlflow.end_run()
    print("PyTorch MLP done:", metrics)

if __name__ == "__main__":
    main()