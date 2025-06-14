# Directory: mlflow_tracking.py
import mlflow
import os
import yaml

def init_mlflow(config_path="config/config.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tracking_uri = config["mlflow"].get("tracking_uri", "./mlruns")
    experiment_name = config["mlflow"].get("experiment_name", "Default")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_step_metrics(step_name, metrics: dict):
    with mlflow.start_run(run_name=step_name):
        mlflow.log_params({"step": step_name})
        mlflow.log_metrics(metrics)


def log_step_artifact(file_path):
    if os.path.exists(file_path):
        mlflow.log_artifact(file_path)
    else:
        print(f"[MLflow] Artifact not found: {file_path}")
