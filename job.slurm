#!/bin/bash
#SBATCH --job-name=neural_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=dev_single

# Load modules or activate environment
source ~/.bashrc
conda activate neural-pipeline

echo "🚀 Starting pipeline at $(date)"
python run_pipeline.py --steps calcium preprocessing inference
echo "Pipeline finished at $(date)"
