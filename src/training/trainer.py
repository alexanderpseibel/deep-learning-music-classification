# src/training/trainer.py
import os
import numpy as np
import torch
from sklearn.metrics import f1_score, average_precision_score
import wandb
import csv

from src.training.wandb_utils import (
    log_metrics,
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

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        # ============================
        # TRAIN
        # ============================
        model.train()
        total_loss = 0

        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(mel), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ============================
        # VALIDATION
        # ============================
        model.eval()
        val_loss = 0
        all_targets = []
        all_preds = []
        all_probs = []
        misclassified_rows = []

        with torch.no_grad():
            for batch_idx, (mel, labels) in enumerate(valid_loader):

                mel, labels = mel.to(device), labels.to(device)
                logits = model(mel)

                val_loss += criterion(logits, labels).item()

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                labels_np = labels.cpu().numpy()

                all_targets.append(labels_np)
                all_probs.append(probs)
                all_preds.append(preds)

                # gather metadata from dataset
                batch_df = valid_loader.dataset.df.iloc[
                    batch_idx * valid_loader.batch_size:
                    batch_idx * valid_loader.batch_size + len(labels)
                ]

                # store paths + labels for misclassified ONLY
                for i in range(len(labels)):
                    true = labels_np[i]
                    pred = preds[i]

                    if not np.array_equal(true, pred):
                        row = batch_df.iloc[i]

                        misclassified_rows.append([
                            row["track_id"],
                            row["audio_path"],
                            row["mel_path"],
                            true.tolist(),
                            pred.tolist()
                        ])

        # ============================
        # METRICS
        # ============================
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        all_probs = np.vstack(all_probs)

        f1_micro = f1_score(all_targets, all_preds, average="micro")
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        per_class_f1 = f1_score(all_targets, all_preds, average=None)
        mAP = compute_map(all_targets, all_probs)
        avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"TrainLoss={avg_train_loss:.4f} | "
              f"ValLoss={avg_val_loss:.4f} | "
              f"F1micro={f1_micro:.4f} | mAP={mAP:.4f}")

        # ============================
        # W&B LOGGING
        # ============================
        log_metrics({
            "loss/train": avg_train_loss,
            "loss/val": avg_val_loss,
            "f1/micro": f1_micro,
            "f1/macro": f1_macro,
            "mAP": mAP,
        }, step=epoch)

        for i, v in enumerate(per_class_f1):
            wandb.log({f"f1/class_{i}": v}, step=epoch)

        log_confusion_heatmap(all_targets, all_preds, CLASS_NAMES, step=epoch)
        log_error_heatmap(all_targets, all_preds, CLASS_NAMES, step=epoch)
        log_precision_recall(all_targets, all_probs, CLASS_NAMES, step=epoch)

        # ============================
        # SAVE MISCLASSIFIED CSV
        # ============================
        csv_path = os.path.join(run_folder, f"misclassified_epoch_{epoch}.csv")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "audio_path", "mel_path",
                             "true_labels", "pred_labels"])
            writer.writerows(misclassified_rows)

        print("Saved misclassified CSV to:", csv_path)

    return model
