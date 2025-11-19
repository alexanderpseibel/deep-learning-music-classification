import os
import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score
import wandb

from src.training.wandb_utils import (
    log_metrics,
    log_example_image,
    log_confusion_heatmap,
    log_error_heatmap,
    log_precision_recall,
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


def train_model(model, train_loader, valid_loader, device, epochs, lr, run_folder):
    """
    The full training loop:
    - training step
    - validation step
    - compute metrics
    - extract misclassified examples
    - log to W&B
    - save misclassified CSV each epoch
    """

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        # -------------------------------
        # TRAIN
        # -------------------------------
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

        # -------------------------------
        # VALIDATION
        # -------------------------------
        model.eval()
        val_loss = 0
        all_targets, all_preds, all_probs = [], [], []
        misclassified = []

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

                    if not np.array_equal(t, p):
                        mel_i = mel[i].detach().cpu().numpy()
                        if mel_i.ndim == 3:
                            mel_i = mel_i.squeeze()

                        misclassified.append({
                            "mel": mel_i,
                            "true": t.tolist(),
                            "pred": p.tolist(),
                        })

        # Stack arrays
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        all_probs = np.vstack(all_probs)

        # Metrics
        f1_micro = f1_score(all_targets, all_preds, average="micro")
        f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        mAP = compute_map(all_targets, all_probs)
        avg_val_loss = val_loss / len(valid_loader)
        per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"F1micro: {f1_micro:.4f} | mAP: {mAP:.4f}")

        # -------------------------------
        # W&B LOGGING
        # -------------------------------
        log_metrics({
            "loss/train": avg_train_loss,
            "loss/val": avg_val_loss,
            "f1/micro": f1_micro,
            "f1/macro": f1_macro,
            "mAP": mAP,
        }, step=epoch)

        for i, v in enumerate(per_class_f1):
            wandb.log({f"f1/class_{i}": v}, step=epoch)

        # Log one example mel
        log_example_image(misclassified[0]["mel"], name="example_misclassified", step=epoch)

        # Confusion / error heatmaps
        log_confusion_heatmap(all_targets, all_preds, CLASS_NAMES, step=epoch)
        log_error_heatmap(all_targets, all_preds, CLASS_NAMES, step=epoch)
        log_precision_recall(all_targets, all_probs, CLASS_NAMES, step=epoch)

        # -------------------------------
        # SAVE MISCLASSIFIED CSV
        # -------------------------------
        import csv
        csv_path = os.path.join(run_folder, f"misclassified_epoch_{epoch}.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true_labels", "pred_labels"])
            for ex in misclassified:
                writer.writerow([ex["true"], ex["pred"]])

        print("Saved misclassified to:", csv_path)

    return model
