# Directory: mlflow_tracking.py
import mlflow
import os
import yaml
from urllib.parse import urlparse

def init_mlflow(config_path="config/config.yaml"):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        tracking_uri = config["mlflow"].get("tracking_uri", "./mlruns")
        experiment_name = config["mlflow"].get("experiment_name", "Default")

        # Validate tracking URI
        if tracking_uri.startswith(("http://", "https://")):
            parsed = urlparse(tracking_uri)
            if not parsed.netloc:
                raise ValueError(f"Invalid tracking URI: {tracking_uri}")
        else:
            # For local paths, ensure directory exists
            os.makedirs(os.path.dirname(tracking_uri), exist_ok=True)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(f"[MLflow] Tracking initialized: {tracking_uri}")
        print(f"[MLflow] Experiment: {experiment_name}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize MLflow: {e}")
        raise


def log_step_metrics(step_name, metrics: dict):
    try:
        with mlflow.start_run(run_name=step_name):
            mlflow.log_params({"step": step_name})
            mlflow.log_metrics(metrics)
    except Exception as e:
        print(f"[ERROR] Failed to log metrics for {step_name}: {e}")
        raise


def log_step_artifact(file_path):
    try:
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path)
        else:
            print(f"[WARNING] Artifact not found: {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to log artifact {file_path}: {e}")
        raise
