import wandb
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


# ----------------------------------------------------------
#                      INIT
# ----------------------------------------------------------
def init_wandb(config: dict, project_name: str = "nlp-mini-project"):
    """Initialize a W&B run."""
    wandb.init(
        project=project_name,
        config=config,
        settings=wandb.Settings(code_dir=os.getcwd())
    )
    return wandb.config


# ----------------------------------------------------------
#                   BASIC LOGGING
# ----------------------------------------------------------
def log_metrics(metrics: dict, step: int = None):
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def log_example_audio(audio_array, sample_rate, name="audio_example", step=None):
    wandb.log({name: wandb.Audio(audio_array, sample_rate=sample_rate)}, step=step)


def log_example_image(image, name="image_example", step=None):
    wandb.log({name: wandb.Image(image)}, step=step)


# ----------------------------------------------------------
#         MISCLASSIFIED EXAMPLES (MULTI-LABEL)
# ----------------------------------------------------------
def log_misclassified_examples(misclassified_examples, step, max_items=3):
    """Log spectrograms with FP/FN tags."""
    for idx, ex in enumerate(misclassified_examples[:max_items]):
        wandb.log({
            f"misclassified/{idx}": wandb.Image(
                ex["mel"],
                caption=f"FP: {ex['fps']} | FN: {ex['fns']}"
            )
        }, step=step)


# ----------------------------------------------------------
#              CO-OCCURRENCE CONFUSION HEATMAP
# ----------------------------------------------------------
def compute_confusion_cooccurrence(y_true, y_pred):
    """
    matrix[i, j] = when true label i = 1, how often was j predicted as 1.
    """
    num_classes = y_true.shape[1]
    matrix = np.zeros((num_classes, num_classes), dtype=float)

    for i in range(num_classes):
        idx = np.where(y_true[:, i] == 1)[0]
        if len(idx) == 0:
            continue
        matrix[i] = np.mean(y_pred[idx], axis=0)

    return matrix


def log_confusion_heatmap(y_true, y_pred, class_names, step):
    matrix = compute_confusion_cooccurrence(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="viridis",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Label Prediction Co-occurrence Heatmap")
    plt.tight_layout()
    wandb.log({"confusion/cooccurrence": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ----------------------------------------------------------
#             FP/FN ERROR CO-OCCURRENCE HEATMAP
# ----------------------------------------------------------
def compute_error_cooccurrence(y_true, y_pred):
    num_classes = y_true.shape[1]
    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)
    errs = fp | fn

    matrix = np.zeros((num_classes, num_classes), dtype=float)

    for i in range(num_classes):
        idx = np.where(errs[:, i])[0]
        if len(idx) > 0:
            matrix[i] = np.mean(errs[idx], axis=0)

    return matrix


def log_error_heatmap(y_true, y_pred, class_names, step):
    matrix = compute_error_cooccurrence(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="magma",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("FP/FN Error Co-occurrence Heatmap")
    plt.tight_layout()
    wandb.log({"confusion/error_matrix": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ----------------------------------------------------------
#              PRECISION–RECALL CURVES
# ----------------------------------------------------------
def log_precision_recall(y_true, y_probs, class_names, step):
    """Log PR curves per class."""
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
        wandb.log({
            f"pr_curves/{class_names[i]}": wandb.plot.curve(
                xs=recall.tolist(),
                ys=precision.tolist(),
                title=f"PR Curve: {class_names[i]}",
                xname="Recall",
                yname="Precision"
            )
        }, step=step)
