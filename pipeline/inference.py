import zarr
import numpy as np
import yaml
import os
from functions import inference, utils
from mlflow_tracking import init_mlflow, log_step_metrics, log_step_artifact

def main(config, upstream_params=None):
    params = config["inference"]
    lag = params["lag"]
    input_path = params["input_path"]
    conn_path = params["conn_matrix"]
    save_output = params.get("save_output", True)

    hash_id = utils.config_hash(params)
    output_name = os.path.basename(input_path).replace(".zarr", f"-inf-lag{lag}-hash{hash_id}.npy")
    output_path = os.path.join("data", output_name)

    signals = zarr.open(input_path, mode="r")
    conn_matrix = np.load(conn_path)

    corr, A = inference.conn_inf_LR(conn_matrix, signals, lag)
    np.save(output_path, A)

    if not save_output:
        os.remove(output_path)

    log_step_artifact(output_path)
    log_step_metrics("inference", {
        "lag": lag,
        "correlation": corr,
        "hash": hash_id,
        "output_path": output_path,
        "saved": save_output,
        **(upstream_params or {})
    })

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    init_mlflow()
    main(config)
