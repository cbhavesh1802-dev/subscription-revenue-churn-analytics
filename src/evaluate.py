"""evaluate.py — RMSE, MAE, Precision@K, NDCG@K metrics."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json, os


def compute_rmse_mae(y_true, y_pred, scale=4.0, shift=1.0):
    """Convert normalized predictions back to 1-5 scale and compute metrics."""
    y_true_s = np.array(y_true) * scale + shift
    y_pred_s = np.array(y_pred) * scale + shift
    return (
        round(float(np.sqrt(mean_squared_error(y_true_s, y_pred_s))), 4),
        round(float(mean_absolute_error(y_true_s, y_pred_s)), 4)
    )


def ndcg_at_k(relevances, k=10):
    """Compute NDCG@K for a single user's ranked list."""
    relevances = np.array(relevances[:k], dtype=float)
    if relevances.sum() == 0:
        return 0.0
    ideal = np.sort(relevances)[::-1]
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    dcg  = (relevances / discounts).sum()
    idcg = (ideal / discounts).sum()
    return dcg / idcg if idcg > 0 else 0.0


def plot_training_history(history, save_path="results/training_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EA-HNCF Training History", fontsize=14, fontweight="bold")
    for ax, metric, title in zip(axes, ["loss", "mae"], ["Loss", "MAE"]):
        ax.plot(history.history[metric],     label="Train")
        ax.plot(history.history[f"val_{metric}"], label="Validation")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved → {save_path}")


def save_results(metrics, path="results/metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Results saved → {path}")
