# ML SaaS Churn Prediction

![CI](https://github.com/IAmSahilVerma/ml-saas-churn/actions/workflows/ci.yml/badge.svg)

Customer churn prediction on the Telco customer churn dataset, packaged the way a small production service would be: a reproducible training pipeline, a FastAPI service, a Docker image, and CI that rebuilds everything from raw data on every push.

## Results

All models are evaluated on a stratified 20% held-out test set (1,409 customers) that they never see during training.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (served)** | 0.645 | 0.554 | 0.596 | 0.841 |
| XGBoost | 0.660 | 0.524 | 0.584 | 0.839 |

A small PyTorch MLP is also trained for comparison.

Logistic regression matched XGBoost on unseen data, so the API serves the simpler, more interpretable model.

## How it works

- **Preprocessing** (`training/features.py`, `training/preprocess_data.py`): fills missing values, scales numeric features and one-hot encodes categorical ones, then saves the fitted preprocessor.
- **Training** (`training/train.py`): splits the data 80/20 with stratification, trains logistic regression, XGBoost and a PyTorch MLP, logs parameters and test metrics to MLflow, and records each run in a training history file.
- **Serving** (`api/`): a FastAPI service that loads the preprocessor and the model named in `models/registry.json`, the serving config that records which model is live and its test metrics.
- **CI** (`.github/workflows/ci.yml`): on every push, GitHub Actions installs dependencies, checks for syntax errors, preprocesses and retrains from the raw data, runs an API smoke test, and builds the Docker image with the freshly trained models.

### API endpoints

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/health` | Service status |
| POST | `/predict` | Churn probability and a Low / Medium / High risk band |
| GET | `/metrics` | Held-out test metrics for the model being served |

## Project structure

```text
ml-saas-churn/
├─ .github/workflows/ci.yml   # CI: retrain, test, build Docker image
├─ api/
│   ├─ main.py                # FastAPI app
│   ├─ model_loader.py        # Loads the preprocessor and served model
│   ├─ metrics_loader.py      # Metrics for the served model
│   └─ schemas.py             # Pydantic request and response models
├─ data/raw/telco_churn.csv   # Raw dataset
├─ models/
│   └─ registry.json          # Serving config: default model and test metrics
├─ training/
│   ├─ preprocess_data.py     # Fits and saves the preprocessor
│   ├─ features.py            # Preprocessor class
│   └─ train.py               # Train/test split, training, MLflow logging
├─ tests/test_api.py          # API smoke test
├─ requirements.txt
└─ Dockerfile
```

The preprocessor and model files are created in `models/` when you run the training steps below.

## Quick start

```bash
# 1. Clone
git clone https://github.com/IAmSahilVerma/ml-saas-churn.git
cd ml-saas-churn

# 2. Create an environment and install dependencies
conda create -n ml-saas-churn python=3.10
conda activate ml-saas-churn
pip install -r requirements.txt

# 3. Preprocess and train
python training/preprocess_data.py
python training/train.py

# 4. Run the tests
pip install pytest httpx
python -m pytest -v

# 5. Run the API
uvicorn api.main:app --reload
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API docs.

### Run with Docker

Train the models first (step 3), so the image includes them:

```bash
docker build -t ml-saas-churn .
docker run -p 8000:8000 ml-saas-churn
```

## What I learned

- My first version scored every model on the same data it was trained on. XGBoost looked far ahead, at 0.907 ROC-AUC. Once I added a proper held-out test set, that lead disappeared and the simpler model matched it.
- Setting up CI exposed several bugs that never showed up on my own machine: an unpinned scikit-learn version, a mismatch between the registry format training wrote and the format the API read, and filenames that had drifted apart between training and serving. Rebuilding from scratch on a clean machine turned out to be the quickest way to find them.

## Next steps

- Tune the decision threshold for recall. At the default 0.5 threshold, the model catches about 55% of churners, and missing a churner usually costs a business more than a false alarm.
- Fit the preprocessor on the training split only. It is currently fitted on the full dataset before splitting, a small source of leakage.
- Pin dependency versions in `requirements.txt`.