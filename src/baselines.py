"""baselines.py — SVD matrix factorization baseline."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_svd(train_df, n_users, n_movies, k=50):
    """Train SVD on user-item matrix. Returns predicted rating matrix."""
    R = np.zeros((n_users, n_movies))
    for _, row in train_df.iterrows():
        R[int(row["user_idx"]), int(row["movie_idx"])] = row["rating"]
    U, sigma, Vt = svds(csr_matrix(R), k=k)
    return np.dot(np.dot(U, np.diag(sigma)), Vt)


def evaluate_svd(predicted, test_df):
    preds, actuals = [], []
    for _, row in test_df.iterrows():
        preds.append(predicted[int(row["user_idx"]), int(row["movie_idx"])])
        actuals.append(row["rating"])
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    print(f"  SVD  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return rmse, mae
