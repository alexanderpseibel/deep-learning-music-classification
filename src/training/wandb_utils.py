# src/training/wandb_utils.py
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def init_wandb(config: dict, project_name: str = "nlp-mini-project"):
    wandb.init(
        project=project_name,
        config=config
    )
    return wandb.config


def log_metrics(metrics: dict, step=None):
    wandb.log(metrics, step=step)


# ------------------------------------------------
# CONFUSION HEATMAP
# ------------------------------------------------
def compute_confusion_cooccurrence(y_true, y_pred):
    num = y_true.shape[1]
    M = np.zeros((num, num))
    for i in range(num):
        idx = np.where(y_true[:, i] == 1)[0]
        if len(idx) > 0:
            M[i] = np.mean(y_pred[idx], axis=0)
    return M


def log_confusion_heatmap(y_true, y_pred, class_names, step):
    M = compute_confusion_cooccurrence(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(M, cmap="viridis",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    plt.tight_layout()

    wandb.log({"confusion/cooc": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ------------------------------------------------
# ERROR HEATMAP
# ------------------------------------------------
def compute_error_cooccurrence(y_true, y_pred):
    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)
    errs = fp | fn

    num = y_true.shape[1]
    M = np.zeros((num, num))
    for i in range(num):
        idx = np.where(errs[:, i])[0]
        if len(idx) > 0:
            M[i] = np.mean(errs[idx], axis=0)
    return M


def log_error_heatmap(y_true, y_pred, class_names, step):
    M = compute_error_cooccurrence(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(M, cmap="magma",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    plt.tight_layout()

    wandb.log({"confusion/error": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ------------------------------------------------
# PRECISION–RECALL CURVES
# ------------------------------------------------
def log_precision_recall(y_true, y_probs, class_names, step):

    num_classes = y_true.shape[1]

    for i in range(num_classes):
        if y_true[:, i].sum() == 0:
            continue

        prec, rec, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])

        table = wandb.Table(columns=["recall", "precision"])
        for r, p in zip(rec, prec):
            table.add_data(r, p)

        wandb.log({
            f"pr/{class_names[i]}": wandb.plot.line(
                table, "recall", "precision", title=f"PR: {class_names[i]}"
            )
        }, step=step)
