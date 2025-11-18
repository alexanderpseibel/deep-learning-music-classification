import torch
from sklearn.metrics import f1_score, average_precision_score
import numpy as np
import wandb

from src.training.wandb_utils import (
    log_metrics,
    log_example_image,
    log_confusion_heatmap,
    log_error_heatmap,
    log_precision_recall,
    log_misclassified_examples,
)


CLASS_NAMES = [
    "Rock", "Electronic", "Pop", "Folk", "Instrumental", "HipHop",
    "International", "Classical", "Jazz", "Country", "Blues", "SoulRnB"
]


def compute_map(y_true, y_pred_probs):
    aps = []
    for i in range(y_true.shape[1]):
        try:
            aps.append(average_precision_score(y_true[:, i], y_pred_probs[:, i]))
        except ValueError:
            pass
    return float(np.mean(aps)) if aps else 0.0


def train_model(model, train_loader, valid_loader, device, epochs, lr):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        # ------------------------------
        # TRAIN
        # ------------------------------
        model.train()
        total_loss = 0

        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ------------------------------
        # VALIDATION
        # ------------------------------
        model.eval()
        val_loss = 0
        all_preds = []
        all_probs = []
        all_targets = []
        misclassified_examples = []

        with torch.no_grad():
            for mel, labels in valid_loader:
                mel, labels = mel.to(device), labels.to(device)

                logits = model(mel)
                val_loss += criterion(logits, labels).item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()

                np_probs = probs.cpu().numpy()
                np_preds = np_probs > 0.5
                np_labels = labels.cpu().numpy()

                all_targets.append(np_labels)
                all_probs.append(np_probs)
                all_preds.append(np_preds)

                # Misclassified examples
                for i in range(len(labels)):
                    t = np_labels[i]
                    p = np_preds[i]

                    fp = np.where((p == 1) & (t == 0))[0]
                    fn = np.where((p == 0) & (t == 1))[0]

                    if len(fp) > 0 or len(fn) > 0:

                        # extract mel (shape: (1, 128, T))
                        mel_i = mel[i].detach().cpu().numpy()

                        # FIX: squeeze channel dimension → (128, T)
                        if mel_i.ndim == 3:
                            mel_i = mel_i.squeeze()

                        misclassified_examples.append({
                            "mel": mel_i,
                            "true": t,
                            "pred": p,
                            "fps": fp.tolist(),
                            "fns": fn.tolist(),
                        })

        # Stack arrays
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        all_probs = np.vstack(all_probs)

        # Metrics
        f1_micro = f1_score(all_targets, all_preds, average="micro")
        f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        mAP = compute_map(all_targets, all_probs)
        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
        avg_val_loss = val_loss / len(valid_loader)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"F1 (micro): {f1_micro:.4f} | "
            f"mAP: {mAP:.4f}"
        )

        # ------------------------------
        # W&B LOGGING
        # ------------------------------
        log_metrics({
            "loss/train": avg_train_loss,
            "loss/val": avg_val_loss,
            "f1/micro": f1_micro,
            "f1/macro": f1_macro,
            "mAP": mAP,
        }, step=epoch)

        wandb.log({f"f1/class_{i}": v for i, v in enumerate(per_class_f1)}, step=epoch)

        # Example mel spectrogram (colored)
        example_mel = mel[0].detach().cpu().numpy()

        # squeeze channel dimension (1,128,T → 128,T)
        if example_mel.ndim == 3:
            example_mel = example_mel.squeeze()

        log_example_image(example_mel, name="example_mel", step=epoch)


        # Misclassified examples
        log_misclassified_examples(misclassified_examples, step=epoch)

        # Heatmaps
        log_confusion_heatmap(all_targets, all_preds, CLASS_NAMES, step=epoch)
        log_error_heatmap(all_targets, all_preds, CLASS_NAMES, step=epoch)

        # Precision–Recall curves
        log_precision_recall(all_targets, all_probs, CLASS_NAMES, step=epoch)

    return model
