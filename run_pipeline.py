# Directory: run_pipeline.py
import yaml
import argparse
import os
from mlflow_tracking import init_mlflow
from functions import simulation as calcium, preprocessing, inference


def load_config(path="config/config.yaml"):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at {path}")
        with open(path) as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a YAML dictionary")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")


def main(config, steps):
    try:
        init_mlflow(config_path="config/config.yaml")

        if "calcium" in steps:
            print("\n[RUNNING] Step 1: Calcium Simulation")
            calcium.main(config)

        if "preprocessing" in steps:
            print("\n[RUNNING] Step 2: Preprocessing")
            preprocessing.main(config)

        if "inference" in steps:
            print("\n[RUNNING] Step 3: Connection Inference")
            inference.main(config)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", default=["calcium", "preprocessing", "inference"],
                        help="Which steps to run: calcium preprocessing inference")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config, args.steps)
