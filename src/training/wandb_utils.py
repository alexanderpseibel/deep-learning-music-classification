import wandb
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


# ----------------------------------------------------------
#                      INIT
# ----------------------------------------------------------
def init_wandb(config: dict, project_name: str = "nlp-mini-project"):
    wandb.init(
        project=project_name,
        config=config,
        settings=wandb.Settings(code_dir=os.getcwd())
    )
    return wandb.config


# ----------------------------------------------------------
#         MEL → COLORED IMAGE (REAL SPECTROGRAM)
# ----------------------------------------------------------
def mel_to_image(mel, cmap="magma"):
    """
    Convert a raw mel-spectrogram (128×T float32 array)
    into a colored RGB uint8 image suitable for W&B.
    """
    mel = np.array(mel)

    # Already in dB → normalize to 0–1
    mel_norm = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    # Apply colormap
    colormap = plt.get_cmap(cmap)
    mel_rgb = colormap(mel_norm)[:, :, :3]

    # Convert to uint8 0–255
    mel_uint8 = (mel_rgb * 255).astype(np.uint8)
    return mel_uint8


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


def log_example_image(mel, name="image_example", step=None):
    img = mel_to_image(mel)
    wandb.log({name: wandb.Image(img)}, step=step)


# ----------------------------------------------------------
#         MISCLASSIFIED EXAMPLES (MULTI-LABEL)
# ----------------------------------------------------------
def log_misclassified_examples(misclassified_examples, step, max_items=3):
    for idx, ex in enumerate(misclassified_examples[:max_items]):
        img = mel_to_image(ex["mel"])
        wandb.log({
            f"misclassified/{idx}": wandb.Image(
                img,
                caption=f"FP: {ex['fps']} | FN: {ex['fns']}"
            )
        }, step=step)


# ----------------------------------------------------------
#              CO-OCCURRENCE CONFUSION HEATMAP
# ----------------------------------------------------------
def compute_confusion_cooccurrence(y_true, y_pred):
    num_classes = y_true.shape[1]
    matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        idx = np.where(y_true[:, i] == 1)[0]
        if len(idx) > 0:
            matrix[i] = np.mean(y_pred[idx], axis=0)
    return matrix


def log_confusion_heatmap(y_true, y_pred, class_names, step):
    matrix = compute_confusion_cooccurrence(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis",
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
    error_mask = fp | fn

    matrix = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        idx = np.where(error_mask[:, i])[0]
        if len(idx) > 0:
            matrix[i] = np.mean(error_mask[idx], axis=0)

    return matrix


def log_error_heatmap(y_true, y_pred, class_names, step):
    matrix = compute_error_cooccurrence(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, cmap="magma",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("FP/FN Error Co-occurrence Heatmap")
    plt.tight_layout()

    wandb.log({"confusion/error_matrix": wandb.Image(fig)}, step=step)
    plt.close(fig)


# ----------------------------------------------------------
#              PRECISION–RECALL CURVES
#         (SUPPORTED FORMAT FOR W&B 0.23.x)
# ----------------------------------------------------------
def log_precision_recall(y_true, y_probs, class_names, step):
    num_classes = y_true.shape[1]

    for i in range(num_classes):
        if y_true[:, i].sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])

        table = wandb.Table(columns=["recall", "precision"])
        for r, p in zip(recall, precision):
            table.add_data(r, p)

        wandb.log({
            f"pr_curves/{class_names[i]}":
                wandb.plot.line(table, "recall", "precision",
                                title=f"PR Curve: {class_names[i]}")
        }, step=step)
