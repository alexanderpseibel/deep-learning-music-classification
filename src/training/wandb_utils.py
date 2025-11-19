# src/training/wandb_utils.py
import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix


def init_wandb(config: dict, project_name: str = "nlp-mini-project"):
    wandb.init(project=project_name, config=config)
    return wandb.config


def log_metrics(metrics: dict, step=None):
    wandb.log(metrics, step=step)


# ------------------------------------------------
# CO-OCCURRENCE CONFUSION
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
                xticklabels=class_names, yticklabels=class_names, ax=ax)
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
                xticklabels=class_names, yticklabels=class_names, ax=ax)
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


# ------------------------------------------------
# PER-CLASS BINARY CONFUSION MATRICES
# ------------------------------------------------
def log_binary_confusion_matrices(y_true, y_pred, class_names, step):
    for i, cls in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"],
                    ax=ax)
        ax.set_title(f"Confusion: {cls}")
        plt.tight_layout()
        wandb.log({f"confusion/{cls}": wandb.Image(fig)}, step=step)
        plt.close(fig)
