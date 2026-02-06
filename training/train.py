import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

from features import preprocess

def train():
    df = pd.read_csv("data/processed/churn.csv")
    X = preprocess(df.drop("churn", axis=1))
    y = df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        
        mlflow.log_metric("roc_auc", auc)
        
        joblib.dump(model, "model/churn_model_v1.pkl")
        
    if __name__ == "__main__":
        train()