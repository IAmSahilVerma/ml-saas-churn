
import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from training.features import Preprocessor
import json
import os
from datetime import datetime
 
REGISTRY_PATH = "models/training_history.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42
 
# ----------------------
# Helper: Registry
# ----------------------
def add_model_to_registry(model_name, model_type, file_path, metrics):
    """Append a trained model entry to the training history file."""
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
 
    print(f"Model {model_name} v{version} added to training history.")
 
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
        raise ValueError(f"Unknown model_type: {model_type}")
 
    return {
        "precision": round(float(precision_score(y, y_pred)), 4),
        "recall": round(float(recall_score(y, y_pred)), 4),
        "f1": round(float(f1_score(y, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, y_pred_prob)), 4),
    }
 
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
    mlflow.set_experiment("ChurnProject")
 
    # Load raw data
    df_raw = pd.read_csv("data/raw/telco_churn.csv")
    df_raw.columns = df_raw.columns.str.strip()
    df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce").fillna(0)
 
    # Load preprocessor
    preprocessor = Preprocessor.load("models/preprocessor.pkl")
    X = preprocessor.transform(df_raw)
    y = df_raw["Churn"].apply(lambda x: 1 if x in ["Yes", 1, "Y", "y"] else 0).values
 
    # Hold out a stratified test set. All models train on the training split
    # and are scored only on the test split they have never seen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, test size: {len(X_test)}")
 
    # ----------------------
    # Logistic Regression
    # ----------------------
    with mlflow.start_run(run_name="logistic_regression"):
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)
        metrics = evaluate_model(logreg, X_test, y_test)
        mlflow.log_params({"model": "LogisticRegression", "max_iter": 1000, "test_size": TEST_SIZE})
        mlflow.log_metrics(metrics)
        logreg_file = "models/logreg_model_v1.pkl"
        joblib.dump(logreg, logreg_file)
        add_model_to_registry("logreg", "sklearn", logreg_file, metrics)
    print("Logistic Regression (test set):", metrics)
 
    # ----------------------
    # XGBoost
    # ----------------------
    with mlflow.start_run(run_name="xgboost"):
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            eval_metric="logloss"
        )
        xgb_model.fit(X_train, y_train)
        metrics = evaluate_model(xgb_model, X_test, y_test, model_type="xgboost")
        mlflow.log_params({"model": "XGBClassifier", "n_estimators": 100,
                           "learning_rate": 0.1, "max_depth": 5, "test_size": TEST_SIZE})
        mlflow.log_metrics(metrics)
        xgb_file = "models/xgb_model_v1.pkl"
        joblib.dump(xgb_model, xgb_file)
        add_model_to_registry("xgb", "xgboost", xgb_file, metrics)
    print("XGBoost (test set):", metrics)
 
    # ----------------------
    # PyTorch MLP
    # ----------------------
    with mlflow.start_run(run_name="pytorch_mlp"):
        torch.manual_seed(RANDOM_STATE)
        X_tensor = torch.FloatTensor(X_train.values)
        y_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)
 
        mlp_model = MLP(input_dim=X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
 
        for epoch in range(10):
            mlp_model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(mlp_model(xb), yb)
                loss.backward()
                optimizer.step()
            if epoch % 2 == 0:
                print(f"Epoch {epoch}, loss: {loss.item():.4f}")
 
        metrics = evaluate_model(mlp_model, X_test, y_test, model_type="pytorch")
        mlflow.log_params({"model": "MLP", "epochs": 10, "batch_size": 64,
                           "lr": 0.001, "test_size": TEST_SIZE})
        mlflow.log_metrics(metrics)
        mlp_file = "models/mlp_model_v1.pt"
        torch.save(mlp_model.state_dict(), mlp_file)
        add_model_to_registry("mlp", "pytorch", mlp_file, metrics)
    print("PyTorch MLP (test set):", metrics)
 
 
if __name__ == "__main__":
    main()
 
