import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid, ParameterSampler
from sklearn.metrics import mean_squared_error
import argparse
import dask
from dask.distributed import Client
from functions.inference import conn_inf_regression, evaluate_inference_performance
from functions.slurm_dask import setup_cluster
from mlflow_tracking import init_mlflow, log_step_metrics
from functions.preprocessing import preprocess_data  # assuming this exists
from functions.simulation import simulate_data  # assuming this exists


def run_regression_cv(X, y, model_type, param_grid, search_type='grid', n_iter=10, distributed=False, dask_client=None):
    if model_type == 'lasso':
        model = Lasso(max_iter=10000)
    elif model_type == 'ridge':
        model = Ridge(max_iter=10000)
    elif model_type == 'elasticnet':
        model = ElasticNet(max_iter=10000)
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    if search_type == 'grid':
        search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    else:
        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    if distributed and dask_client is not None:
        with dask.distributed.Client():
            search.fit(X, y)
    else:
        search.fit(X, y)
    return search


def run_regression_eval(X, y, model_type, param_grid, eval_metric, G=None, lag=10, search_type='grid', n_iter=10, distributed=False, dask_client=None, multi_class_labels=None):
    if search_type == 'grid':
        param_iter = ParameterGrid(param_grid)
    else:
        param_iter = ParameterSampler(param_grid, n_iter=n_iter, random_state=42)
    best_score = None
    best_params = None
    for params in param_iter:
        # Fit model
        _, A = conn_inf_regression(G, X, lag=lag, model_type=model_type, **params)
        score = evaluate_inference_performance(G, A, metric=eval_metric, multi_class_labels=multi_class_labels) if G is not None else float('nan')
        if (best_score is None) or (eval_metric == 'mse' and score < best_score) or (eval_metric != 'mse' and score > best_score):
            best_score = score
            best_params = params
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(description='Regression HPO with sklearn (local or Dask)')
    parser.add_argument('--distributed', action='store_true', help='Use Dask/SLURM for distributed HPO')
    parser.add_argument('--search_type', type=str, default='grid', choices=['grid', 'random'], help='Type of search')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of iterations for random search')
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

    # Parameter grid
    if args.model_type == 'lasso' or args.model_type == 'ridge':
        param_grid = {'alpha': np.logspace(-4, 1, 20)}
    elif args.model_type == 'elasticnet':
        param_grid = {'alpha': np.logspace(-4, 1, 10), 'l1_ratio': np.linspace(0.1, 0.9, 5)}
    else:
        param_grid = {}

    # MLflow setup
    init_mlflow(args.mlflow_config)

    # Dask setup
    dask_client = None
    if args.distributed:
        _, dask_client = setup_cluster()

    G = None
    if args.ground_truth_path:
        G = np.load(args.ground_truth_path)

    if G is not None:
        best_params, best_score = run_regression_eval(X, y, args.model_type, param_grid, args.eval_metric, G=G, lag=args.lag)
        log_step_metrics(f'sklearn_{args.model_type}_hpo', {**best_params, f'best_{args.eval_metric}': best_score})
        print(f'Best params: {best_params}, Best {args.eval_metric}: {best_score}')
    else:
        # fallback to CV
        search = run_regression_cv(X, y, args.model_type, param_grid, search_type=args.search_type, n_iter=args.n_iter, distributed=args.distributed, dask_client=dask_client)
        best_params = search.best_params_
        best_score = search.best_score_
        log_step_metrics(f'sklearn_{args.model_type}_hpo', {**best_params, 'best_score': best_score})
        print(f'Best params: {best_params}, Best score: {best_score}')

    if dask_client is not None:
        dask_client.close()

if __name__ == '__main__':
    main() 