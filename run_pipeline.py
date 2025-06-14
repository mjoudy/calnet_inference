# Directory: run_pipeline.py
import yaml
import argparse
from mlflow_tracking import init_mlflow
from functions.pipeline import calcium, preprocessing, inference


def load_config(path="config/config.yml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main(config, steps):
    init_mlflow(config_path="config/config.yml")

    if "calcium" in steps:
        print("\n[RUNNING] Step 1: Calcium Simulation")
        calcium.main(config)

    if "preprocessing" in steps:
        print("\n[RUNNING] Step 2: Preprocessing")
        preprocessing.main(config)

    if "inference" in steps:
        print("\n[RUNNING] Step 3: Connection Inference")
        inference.main(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", default=["calcium", "preprocessing", "inference"],
                        help="Which steps to run: calcium preprocessing inference")
    parser.add_argument("--config", default="config/config.yml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config, args.steps)
