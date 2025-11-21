# src/training/wandb_utils.py

import wandb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    average_precision_score
)

# ---------------------------------------------------------
# Compute mAP + per-class AP
# ---------------------------------------------------------
def compute_map_and_per_class_ap(y_true, y_pred_probs):
    per_class_ap = []
    for i in range(y_true.shape[1]):
        try:
            ap = average_precision_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            ap = np.nan
        per_class_ap.append(ap)

    mAP = float(np.nanmean(per_class_ap))
    return mAP, per_class_ap


# ---------------------------------------------------------
# Recall@K
# ---------------------------------------------------------
def recall_at_k(y_true, y_probs, k=3):
    topk_idx = np.argsort(-y_probs, axis=1)[:, :k]

    recalls = []
    for true_row, topk in zip(y_true, topk_idx):
        true_labels = np.where(true_row == 1)[0]
        if len(true_labels) == 0:
            continue
        hits = sum(1 for t in true_labels if t in topk)
        recalls.append(hits / len(true_labels))

    return float(np.mean(recalls)) if recalls else 0.0


# ---------------------------------------------------------
# W&B basic logging
# ---------------------------------------------------------
def init_wandb(config: dict, project_name: str = "nlp-mini-project"):
    wandb.init(
        project=project_name,
        config=config,
        # Increase W&B timeouts to avoid ReadTimeout crashes
        settings=wandb.Settings(
            _service_timeout=300
        )
    )
    return wandb.config


# ---------------------------------------------------------
# CO-OCCURRENCE CONFUSION
# ---------------------------------------------------------
def compute_confusion_cooccurrence(y_true, y_pred):
    num = y_true.shape[1]
    M = np.zeros((num, num))
    for i in range(num):
        idx = np.where(y_true[:, i] == 1)[0]
        if len(idx) > 0:
            M[i] = np.mean(y_pred[idx], axis=0)
    return M


def log_confusion_heatmap(y_true, y_pred, class_names):
    M = compute_confusion_cooccurrence(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        M, cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    plt.tight_layout()
    wandb.log({"confusion/cooc": wandb.Image(fig)})
    plt.close(fig)


# ---------------------------------------------------------
# ERROR HEATMAP
# ---------------------------------------------------------
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


def log_error_heatmap(y_true, y_pred, class_names):
    M = compute_error_cooccurrence(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        M, cmap="magma",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    plt.tight_layout()
    wandb.log({"confusion/error": wandb.Image(fig)})
    plt.close(fig)


# ---------------------------------------------------------
# PRECISION–RECALL CURVES
# ---------------------------------------------------------
def log_precision_recall(y_true, y_probs, class_names):
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        if y_true[:, i].sum() == 0:
            continue

        prec, rec, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])

        # Skip very large curves to avoid heavy W&B uploads
        if len(prec) > 2000:
            print(f"[WARN] PR curve for {class_names[i]} skipped (too many points: {len(prec)})")
            continue

        try:
            table = wandb.Table(columns=["recall", "precision"])
            for r, p in zip(rec, prec):
                table.add_data(r, p)

            wandb.log({
                f"pr/{class_names[i]}": wandb.plot.line(
                    table, "recall", "precision",
                    title=f"PR: {class_names[i]}"
                )
            })
        except Exception as e:
            print(f"[WARN] WANDB PR logging failed for {class_names[i]}: {e}")



# ---------------------------------------------------------
# PER-CLASS BINARY CONF MATRICES
# ---------------------------------------------------------
def log_binary_confusion_matrices(y_true, y_pred, class_names):
    for i, cls in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax
        )
        ax.set_title(f"Confusion: {cls}")
        plt.tight_layout()
        wandb.log({f"confusion/{cls}": wandb.Image(fig)})
        plt.close(fig)
