import json
import os

REGISTRY_PATH = os.path.join("models", "registry.json")
METRIC_KEYS = ("precision", "recall", "f1", "roc_auc")


def get_latest_metrics():
    """Return held-out test metrics for the model currently being served."""
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    model_name = registry["default"]
    model_info = registry["models"][model_name]

    return {
        "model": model_name,
        "metrics": {k: model_info[k] for k in METRIC_KEYS if k in model_info},
        "evaluated_on": "stratified 20% held-out test set",
    }