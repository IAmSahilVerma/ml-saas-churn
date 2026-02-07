import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "ChurnProject"

def get_latest_metrics():
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return {"error": "No MLflow experiment found"}
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1
    )
    
    if len(runs) == 0:
        return {"error": "No runs found"}
    
    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "model_type": run.data.params.get("model", "unknown"),
        "metrics": run.data.metrics
    }