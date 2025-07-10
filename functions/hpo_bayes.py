import numpy as np
from skopt import BayesSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
import argparse
from functions.inference import conn_inf_regression
from functions.slurm_dask import setup_cluster
from mlflow_tracking import init_mlflow, log_step_metrics
from functions.preprocessing import preprocess_data  # assuming this exists
from functions.simulation import simulate_data  # assuming this exists


def run_bayes_cv(X, y, model_type, search_space, n_iter=20, distributed=False, dask_client=None):
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
    search = BayesSearchCV(model, search_space, n_iter=n_iter, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    if distributed and dask_client is not None:
        import joblib
        with joblib.parallel_backend('dask'):
            search.fit(X, y)
    else:
        search.fit(X, y)
    return search


def main():
    parser = argparse.ArgumentParser(description='Regression HPO with Bayesian Optimization (skopt)')
    parser.add_argument('--distributed', action='store_true', help='Use Dask/SLURM for distributed HPO')
    parser.add_argument('--n_iter', type=int, default=20, help='Number of Bayesian optimization iterations')
    parser.add_argument('--mlflow_config', type=str, default='config/config.yaml', help='MLflow config path')
    parser.add_argument('--model_type', type=str, default='lasso', choices=['lasso', 'ridge', 'elasticnet', 'linear'], help='Regression type')
    parser.add_argument('--data_source', type=str, default='dummy', choices=['dummy', 'simulate', 'preprocess'], help='Data source')
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

    # Search space
    if args.model_type == 'lasso' or args.model_type == 'ridge':
        search_space = {'alpha': (1e-4, 10.0, 'log-uniform')}
    elif args.model_type == 'elasticnet':
        search_space = {'alpha': (1e-4, 10.0, 'log-uniform'), 'l1_ratio': (0.1, 0.9, 'uniform')}
    else:
        search_space = {}

    # MLflow setup
    init_mlflow(args.mlflow_config)

    # Dask setup
    dask_client = None
    if args.distributed:
        _, dask_client = setup_cluster()

    # Run HPO
    search = run_bayes_cv(X, y, args.model_type, search_space, n_iter=args.n_iter, distributed=args.distributed, dask_client=dask_client)

    # Log best result
    best_params = search.best_params_
    best_score = search.best_score_
    log_step_metrics(f'bayes_{args.model_type}_hpo', {**best_params, 'best_score': best_score})
    print(f'Best params: {best_params}, Best score: {best_score}')

    if dask_client is not None:
        dask_client.close()

if __name__ == '__main__':
    main() 