#!/bin/bash

source ~/.bashrc
conda activate neural-pipeline

echo "🔧 Creating test data..."
python tests/dry_run_setup.py

echo "🚀 Running pipeline..."
python run_pipeline.py --steps calcium preprocessing inference

echo "✅ Test complete. Check 'data/' and 'mlruns/' folders."
