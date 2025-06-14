import numpy as np
from sklearn.linear_model import LinearRegression

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
