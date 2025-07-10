import numpy as np
import optuna
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.model_selection import cross_val_score
import argparse
from functions.inference import conn_inf_regression, evaluate_inference_performance
from functions.slurm_dask import setup_cluster
from mlflow_tracking import init_mlflow, log_step_metrics
import sys
from functions.preprocessing import preprocess_data  # assuming this exists
from functions.simulation import simulate_data  # assuming this exists


def objective(trial, X, y, model_type, eval_metric, G=None, lag=10, multi_class_labels=None):
    if model_type == 'lasso' or model_type == 'ridge':
        alpha = trial.suggest_loguniform('alpha', 1e-4, 10)
        params = {'alpha': alpha}
    elif model_type == 'elasticnet':
        alpha = trial.suggest_loguniform('alpha', 1e-4, 10)
        l1_ratio = trial.suggest_uniform('l1_ratio', 0.1, 0.9)
        params = {'alpha': alpha, 'l1_ratio': l1_ratio}
    else:
        params = {}
    model = None
    if G is not None:
        _, A = conn_inf_regression(G, X, lag=lag, model_type=model_type, **params)
        score = evaluate_inference_performance(G, A, metric=eval_metric, multi_class_labels=multi_class_labels)
        # For Optuna, maximize unless metric is 'mse'
        return -score if eval_metric == 'mse' else score
    else:
        # fallback to CV
        if model_type == 'lasso':
            model = Lasso(alpha=params.get('alpha', 1.0), max_iter=10000)
        elif model_type == 'ridge':
            model = Ridge(alpha=params.get('alpha', 1.0), max_iter=10000)
        elif model_type == 'elasticnet':
            model = ElasticNet(alpha=params.get('alpha', 1.0), l1_ratio=params.get('l1_ratio', 0.5), max_iter=10000)
        elif model_type == 'linear':
            model = LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
        return score.mean()


def main():
    parser = argparse.ArgumentParser(description='Regression HPO with Optuna (local or Dask)')
    parser.add_argument('--distributed', action='store_true', help='Use Dask/SLURM for distributed HPO')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--mlflow_config', type=str, default='config/config.yaml', help='MLflow config path')
    parser.add_argument('--model_type', type=str, default='lasso', choices=['lasso', 'ridge', 'elasticnet', 'linear'], help='Regression type')
    parser.add_argument('--data_source', type=str, default='dummy', choices=['dummy', 'simulate', 'preprocess'], help='Data source')
    parser.add_argument('--eval_metric', type=str, default='correlation', choices=['correlation', 'mse', 'auc', 'multi-class-tp-rate'], help='Evaluation metric for HPO')
    parser.add_argument('--ground_truth_path', type=str, default=None, help='Path to ground truth connectivity matrix (optional)')
    parser.add_argument('--lag', type=int, default=10, help='Lag parameter for inference')
    args = parser.parse_args()

    # Data loading
    if args.data_source == 'simulate':
        X, y = simulate_data()
    elif args.data_source == 'preprocess':
        X, y = preprocess_data()
    else:
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)

    # MLflow setup
    init_mlflow(args.mlflow_config)

    # Dask setup
    dask_client = None
    if args.distributed:
        _, dask_client = setup_cluster()
        import joblib
        with joblib.parallel_backend('dask'):
            G = None
            if args.ground_truth_path:
                G = np.load(args.ground_truth_path)
            study = optuna.create_study(direction='maximize' if args.eval_metric != 'mse' else 'minimize')
            study.optimize(lambda trial: objective(trial, X, y, args.model_type, args.eval_metric, G=G, lag=args.lag), n_trials=args.n_trials, n_jobs=-1)
            best_trial = study.best_trial
            log_step_metrics(f'optuna_{args.model_type}_hpo', {**best_trial.params, f'best_{args.eval_metric}': best_trial.value})
            print(f'Best params: {best_trial.params}, Best {args.eval_metric}: {best_trial.value}')
    else:
        G = None
        if args.ground_truth_path:
            G = np.load(args.ground_truth_path)
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X, y, args.model_type, args.eval_metric), n_trials=args.n_trials, n_jobs=-1)
        best_trial = study.best_trial
        log_step_metrics(f'optuna_{args.model_type}_hpo', {**best_trial.params, 'best_score': best_trial.value})
        print(f'Best params: {best_trial.params}, Best score: {best_trial.value}')

    if dask_client is not None:
        dask_client.close()

if __name__ == '__main__':
    main() 