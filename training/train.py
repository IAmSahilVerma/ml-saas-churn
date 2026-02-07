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

# Helper function: evaluate
def evaluate_model(model, X, y, model_type="sklearn"):
    if model_type == "sklearn" or model_type == "xgboost":
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
    return {"precision": precision,
            "recall": recall,
            "roc_auc": roc_acu}

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
    
    # Model 1: Logistic Regression
    mlflow.start_run(run_name="logistic_regression")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X, y)
    metrics = evaluate_model(logreg, X, y)
    mlflow.log_params({"model": "LogisticRegression", "max_iter": 1000})
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    joblib.dump(logreg, "models/logreg_model_v1.pkl")
    mlflow.end_run()
    print("Logistic Regression done:", metrics)

    # Model 2: XGBoost
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
    joblib.dump(xgb_model, "models/xgb_model_v1.pkl")
    mlflow.end_run()
    print("XGBoost done:", metrics)

    # Model 3: PyTorch MLP (Multi-Layer-Perceptron)
    mlflow.start_run(run_name="pytorch_mlp")

    X_tensor = torch.FloatTensor(X.values)
    y_tensor = torch.FloatTensor(y.reshape(-1, 1))
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    mlp_model = MLP(input_dim=X.shape[1])
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

    # Training Loop
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
    mlflow.log_params({
        "model": "MLP",
        "epochs": 10,
        "batch_size": 64,
        "lr": 0.001
    })
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
    torch.save(mlp_model.state_dict(), "models/mlp_model_v1.pt")
    mlflow.end_run()
    print("PyTroch MLP done:", metrics)