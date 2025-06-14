import zarr
import dask.array as da
import numpy as np
import os
import yaml
from functions import preprocessing, utils, slurm_dask
from mlflow_tracking import init_mlflow, log_step_metrics, log_step_artifact

def main(config, upstream_params=None):
    params = config["preprocessing"]
    sg_win = params["sg_win"]
    win_len = params["win_len"]
    input_path = params["input_path"]
    spikes_path = params["spikes_path"]
    save_output = params.get("save_output", True)

    hash_id = utils.config_hash(params)
    output_name = utils.nameit(input_path, "preprocessed", sg_win=sg_win, win_len=win_len, hash=hash_id)
    output_path = os.path.join("data", output_name)

    calcium = da.from_zarr(input_path)
    spikes = da.from_zarr(spikes_path)

    feed = da.map_blocks(
        preprocessing.dask_preprocess,
        calcium,
        spikes,
        sg_win,
        win_len,
        dtype=np.float64
    )
    utils.handle_output(feed, output_path, save_output=save_output)

    log_step_artifact(output_path)
    log_step_metrics("preprocessing", {
        "sg_win": sg_win,
        "win_len": win_len,
        "hash": hash_id,
        "output_path": output_path,
        "saved": save_output,
        **(upstream_params or {})
    })

if __name__ == "__main__":
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    init_mlflow()
    cluster, client = slurm_dask.setup_cluster(**config["resources"]["cluster"])
    try:
        main(config)
    finally:
        client.close()
