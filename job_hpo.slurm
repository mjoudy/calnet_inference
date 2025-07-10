#!/bin/bash
#SBATCH --job-name=hpo_job
#SBATCH --output=logs/hpo_%j.out
#SBATCH --error=logs/hpo_%j.err
#SBATCH --partition=your_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Activate your environment
source ~/.bashrc
conda activate your_env_name

# Run HPO script (edit as needed)
python functions/hpo_sklearn.py --distributed --model_type lasso --data_source simulate
# python functions/hpo_optuna.py --distributed --model_type ridge --data_source preprocess
# python functions/hpo_bayes.py --distributed --model_type elasticnet --data_source dummy 