# 🧠 Neural Connectivity Pipeline (HPC + Dask + MLflow)

This repository implements a modular and reproducible pipeline for:
- Simulating calcium activity from spike trains
- Preprocessing the calcium signal
- Inferring a neural connectivity matrix

It is optimized for large datasets using Dask and runs on SLURM-based HPC clusters. MLflow is used for complete experiment tracking, including config parameters, input/output files, and derived metrics.

---

## 📁 Repository Structure
```
├── config/
│   └── config.yaml            # All pipeline parameters & paths
│
├── functions/
│   ├── __init__.py           # Package initialization
│   ├── simulation.py         # Step 1: Calcium simulation logic
│   ├── preprocessing.py      # Step 2: Feed signal estimation
│   ├── inference.py          # Step 3: Connection inference
│   ├── util.py              # Shared tools: file naming, config hash
│   └── slurm_dask.py         # Dask cluster setup for SLURM
│
├── logs/                     # Logs written by SLURM
│
├── run_pipeline.py           # Main entry point for chaining all steps
├── mlflow_tracking.py        # Centralized MLflow logging logic
├── job.slurm                 # Example SLURM job submission file
├── environment.yml           # Conda environment definition
│
├── tests/
│   ├── dry_run_setup.py      # Creates dummy spike & conn_matrix data
│   ├── test_pipeline.sh      # Script to run the whole pipeline locally
│   └── test_pipeline.ipynb   # Notebook interface for testing & visualization
│
└── README.md
```

---

## 🔧 Configuration

The `config/config.yaml` file should contain the following structure:

```yaml
# MLflow configuration
mlflow:
  tracking_uri: "./mlruns"  # Local path or remote URI
  experiment_name: "Default"

# Pipeline parameters
pipeline:
  save_output: true  # Whether to save intermediate outputs
  cleanup: false     # Whether to delete intermediate files

# Step-specific parameters
calcium:
  # Parameters for calcium simulation
  tau: 0.5
  dt: 0.01

preprocessing:
  # Parameters for signal preprocessing
  window_size: 100
  threshold: 0.1

inference:
  # Parameters for connectivity inference
  method: "correlation"
  threshold: 0.05
```

---

## 🔁 Pipeline Overview

```text
spikes.zarr → step1_calcium → calcium.zarr
                  ↓
               step2_preprocessing → preprocessed.zarr
                                          ↓
                                 step3_inference → est_matrix.npy
```

Each step:
- Uses parameter-based versioning for output files
- Can optionally save/delete outputs via `save_output` flag
- Tracks all parameters and data paths using MLflow

---

## ✅ Usage Instructions

### 1. Set up your environment
```bash
conda env create -f environment.yml
conda activate calnet
```

### 2. Generate test data
```bash
python tests/dry_run_setup.py
```

### 3. Run pipeline (locally or via SLURM)
```bash
python run_pipeline.py --steps calcium preprocessing inference
```

Or use SLURM:
```bash
sbatch job.slurm
```

### 4. View MLflow UI
```bash
mlflow ui --port 5000
```

---

## 🧪 Features
- Modular steps (can run individually or chained)
- Parameter-aware file naming with hashes
- Optional output cleanup
- Reproducible experiment tracking via MLflow
- Lightweight versioning built-in
- Comprehensive error handling
- Automatic log directory creation

---

## ✍️ Author
Mohammad Joudy

Please feel free to reach out or open an issue to collaborate or suggest improvements.
