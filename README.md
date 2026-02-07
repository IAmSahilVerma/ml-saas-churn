# ML SaaS Churn Prediction

An **end-to-end SaaS-style ML project** for customer churn prediction.  
This repository demonstrates **real-world ML system thinking**, from raw data to model deployment, monitoring, and API serving.

---

## 🏗 Project Overview

This project is not just a notebook. It includes:

- **Data preprocessing:** Handles missing values, numeric scaling, and categorical encoding using `pandas` and `scikit-learn`.
- **Machine Learning models:**
  - Logistic Regression
  - XGBoost
  - PyTorch MLP (Multi-Layer Perceptron)
- **Evaluation metrics:** Precision, Recall, ROC-AUC, feature importance, and model explainability (via SHAP/feature importance placeholders).
- **API serving:** FastAPI endpoints for prediction, metrics, and health checks.
- **MLOps-lite:**
  - Experiment tracking with `MLflow`
  - Model versioning and logging
  - Dockerized for reproducibility
- **SaaS framing:** API exposes `/predict`, `/metrics`, and `/health` endpoints for easy integration.

---

## 🗂 Directory Structure

```text
ml-saas-churn/
│
├─ api/
│   ├─ main.py           # FastAPI app
│   ├─ model_loader.py   # Load latest models & preprocessor
│   └─ schemas.py        # Pydantic request/response models
│
├─ data/
│   ├─ raw/telco_churn.csv  # Raw dataset
│
├─ models/
│   ├─ preprocessor.pkl
│   ├─ logreg_model_v1.pkl
│   ├─ xgb_model_v1.pkl
│   └─ mlp_model_v1.pt
│
├─ training/
│   ├─ preprocess_data.py  # Preprocessing script
│   ├─ features.py         # Preprocessor class
│   └─ train.py            # Train & save models
│
├─ requirements.txt        # Python dependencies
└─ Dockerfile              # Docker configuration
```

# Quick Start
## 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ml-saas-churn.git
cd ml-saas-churn
```

## 2. Create a virtual environment
```bash
conda create -n ml-saas-churn python=3.10
conda activate ml-saas-churn
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Preprocess data
```bash
python training/preprocess_data.py
```

## 5. Train models
```bash
python training/train.py
```

## 6. Run API locally
```bash
uvicorn api.main:app --reload
```
Open [http://localhost:8000/docs](http://localhost:8000/docs) to access Swagger UI