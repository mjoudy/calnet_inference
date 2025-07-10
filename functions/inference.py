import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, roc_auc_score

def conn_inf_LR(conn_matrix, signals, lag=10):
    G = conn_matrix
    G = G - np.diag(np.diag(G))  # zero the diagonal

    Y = signals[:, lag:]
    Y_prime = signals[:, :-lag]

    yk = Y.T
    y_k = Y_prime.T

    reg = LinearRegression(n_jobs=-1).fit(y_k, yk)
    A = reg.coef_
    A = A - np.diag(np.diag(A))

    corr_G_A = np.corrcoef(G.flatten(), A.flatten())[0, 1]
    return corr_G_A, A

def conn_inf_Lasso(conn_matrix, signals, lag=10, alpha=1.0):
    G = conn_matrix
    G = G - np.diag(np.diag(G))  # zero the diagonal

    Y = signals[:, lag:]
    Y_prime = signals[:, :-lag]

    yk = Y.T
    y_k = Y_prime.T

    reg = Lasso(alpha=alpha, max_iter=10000).fit(y_k, yk)
    A = reg.coef_
    if A.ndim == 1:
        A = A.reshape(1, -1)
    A = A - np.diag(np.diag(A))

    corr_G_A = np.corrcoef(G.flatten(), A.flatten())[0, 1]
    return corr_G_A, A

def conn_inf_Ridge(conn_matrix, signals, lag=10, alpha=1.0):
    G = conn_matrix
    G = G - np.diag(np.diag(G))
    Y = signals[:, lag:]
    Y_prime = signals[:, :-lag]
    yk = Y.T
    y_k = Y_prime.T
    reg = Ridge(alpha=alpha, max_iter=10000).fit(y_k, yk)
    A = reg.coef_
    if A.ndim == 1:
        A = A.reshape(1, -1)
    A = A - np.diag(np.diag(A))
    corr_G_A = np.corrcoef(G.flatten(), A.flatten())[0, 1]
    return corr_G_A, A

def conn_inf_ElasticNet(conn_matrix, signals, lag=10, alpha=1.0, l1_ratio=0.5):
    G = conn_matrix
    G = G - np.diag(np.diag(G))
    Y = signals[:, lag:]
    Y_prime = signals[:, :-lag]
    yk = Y.T
    y_k = Y_prime.T
    reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000).fit(y_k, yk)
    A = reg.coef_
    if A.ndim == 1:
        A = A.reshape(1, -1)
    A = A - np.diag(np.diag(A))
    corr_G_A = np.corrcoef(G.flatten(), A.flatten())[0, 1]
    return corr_G_A, A

def conn_inf_regression(conn_matrix, signals, lag=10, model_type='lasso', **kwargs):
    if model_type == 'lasso':
        return conn_inf_Lasso(conn_matrix, signals, lag=lag, alpha=kwargs.get('alpha', 1.0))
    elif model_type == 'ridge':
        return conn_inf_Ridge(conn_matrix, signals, lag=lag, alpha=kwargs.get('alpha', 1.0))
    elif model_type == 'elasticnet':
        return conn_inf_ElasticNet(conn_matrix, signals, lag=lag, alpha=kwargs.get('alpha', 1.0), l1_ratio=kwargs.get('l1_ratio', 0.5))
    elif model_type == 'linear':
        return conn_inf_LR(conn_matrix, signals, lag=lag)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def evaluate_inference_performance(G, A, metric='correlation', multi_class_labels=None):
    """
    Evaluate the performance of inferred connectivity matrix A against ground truth G.
    Supported metrics: 'correlation', 'mse', 'auc', 'multi-class-tp-rate'
    - 'correlation': Pearson correlation between flattened G and A
    - 'mse': Mean squared error between flattened G and A
    - 'auc': ROC AUC score (binary, flattens both)
    - 'multi-class-tp-rate': True positive rate for multi-class (requires multi_class_labels)
    """
    G_flat = G.flatten()
    A_flat = A.flatten()
    if metric == 'correlation':
        return np.corrcoef(G_flat, A_flat)[0, 1]
    elif metric == 'mse':
        return mean_squared_error(G_flat, A_flat)
    elif metric == 'auc':
        # Binarize G and A for AUC
        G_bin = (G_flat > 0).astype(int)
        A_bin = (A_flat > 0).astype(int)
        try:
            return roc_auc_score(G_bin, A_flat)
        except Exception:
            return float('nan')
    elif metric == 'multi-class-tp-rate':
        # multi_class_labels should be a 2D array of same shape as G, with integer class labels
        if multi_class_labels is None:
            raise ValueError('multi_class_labels must be provided for multi-class-tp-rate')
        # For each class, compute TP rate
        from sklearn.metrics import confusion_matrix
        G_labels = multi_class_labels.flatten()
        A_labels = A.flatten().round().astype(int)
        cm = confusion_matrix(G_labels, A_labels)
        # True positive rate per class: TP / (TP + FN)
        tp_rate = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            denom = tp + fn
            tp_rate.append(tp / denom if denom > 0 else 0)
        return np.mean(tp_rate)
    else:
        raise ValueError(f"Unknown metric: {metric}")
