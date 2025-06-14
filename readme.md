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
│   ├── simulation.py         # Step 1: Calcium simulation logic
│   ├── preprocessing.py      # Step 2: Feed signal estimation
│   ├── inference.py          # Step 3: Connection inference
│   ├── utils.py              # Shared tools: file naming, config hash
│   └── slurm_dask.py         # Dask cluster setup for SLURM
│
├── pipeline/
│   ├── step1_calcium.py      # SLURM-ready script for calcium simulation
│   ├── step2_preprocessing.py# SLURM-ready script for preprocessing
│   └── step3_inference.py    # SLURM-ready script for inference
│
├── run_pipeline.py           # Main entry point for chaining all steps
├── mlflow_tracking.py        # Centralized MLflow logging logic
├── job.slurm                 # Example SLURM job submission file
├── environment.yml           # Conda environment definition
├── logs/                     # Logs written by SLURM
│
├── tests/
│   ├── dry_run_setup.py      # Creates dummy spike & conn_matrix data
│   ├── test_pipeline.sh      # Script to run the whole pipeline locally
│   └── test_pipeline.ipynb   # Notebook interface for testing & visualization
│
└── README.md
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
conda activate neural-pipeline
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

---

## ✍️ Author
Mohammad Joudy

Please feel free to reach out or open an issue to collaborate or suggest improvements.
