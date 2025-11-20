# src/training/trainer.py
import os
import numpy as np
import torch
import csv

from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
    label_ranking_loss
)

import wandb

from src.training.wandb_utils import (
    log_metrics,
    log_confusion_heatmap,
    log_error_heatmap,
    log_precision_recall,
    log_binary_confusion_matrices
)

CLASS_NAMES = [
    "Rock", "Electronic", "Pop", "Folk", "Instrumental", "HipHop",
    "International", "Classical", "Jazz", "Country", "Blues", "SoulRnB"
]


# ---------------------------------------------------------
# Compute mAP and AP per class
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
        hits = sum(t in topk for t in true_labels)
        recalls.append(hits / len(true_labels))

    return float(np.mean(recalls)) if recalls else 0.0


# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
def train_model(model, train_loader, valid_loader, device, epochs, lr, run_folder):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):

        # =====================================================
        #                     TRAIN
        # =====================================================
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

        # =====================================================
        #                     VALID
        # =====================================================
        model.eval()
        val_loss = 0
        all_targets, all_preds, all_probs = [], [], []
        misclassified_rows = []

        with torch.no_grad():
            for batch_idx, (mel, labels) in enumerate(valid_loader):

                mel, labels = mel.to(device), labels.to(device)
                logits = model(mel)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                labels_np = labels.cpu().numpy()

                val_loss += criterion(logits, labels).item()

                all_targets.append(labels_np)
                all_probs.append(probs)
                all_preds.append(preds)

                # Misclassified items
                batch_df = valid_loader.dataset.df.iloc[
                    batch_idx * valid_loader.batch_size:
                    batch_idx * valid_loader.batch_size + len(labels)
                ]

                for i in range(len(labels)):
                    if not np.array_equal(labels_np[i], preds[i]):
                        row = batch_df.iloc[i]
                        misclassified_rows.append([
                            row["track_id"],
                            row["audio_path"],
                            row["mel_path"],
                            labels_np[i].tolist(),
                            preds[i].tolist()
                        ])

        # stack arrays
        all_targets = np.vstack(all_targets)
        all_preds = np.vstack(all_preds)
        all_probs = np.vstack(all_probs)
        avg_val_loss = val_loss / len(valid_loader)

        # =====================================================
        #                METRICS (multilabel)
        # =====================================================
        f1_micro = f1_score(all_targets, all_preds, average="micro")
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        per_class_f1 = f1_score(all_targets, all_preds, average=None)

        mAP, per_class_ap = compute_map_and_per_class_ap(all_targets, all_probs)

        try:
            auc_macro = roc_auc_score(all_targets, all_probs, average="macro")
        except ValueError:
            auc_macro = 0.0

        try:
            auc_micro = roc_auc_score(all_targets, all_probs, average="micro")
        except ValueError:
            auc_micro = 0.0

        lrl = label_ranking_loss(all_targets, all_probs)
        r3 = recall_at_k(all_targets, all_probs, k=3)

        print(
            f"Epoch {epoch+1}/{epochs} | Train={avg_train_loss:.4f} | "
            f"Val={avg_val_loss:.4f} | F1micro={f1_micro:.4f} | mAP={mAP:.4f} | R@3={r3:.4f}"
        )

        # =====================================================
        #                  W&B LOGGING
        # =====================================================
        log_dict = {
            "epoch": epoch,
            "loss/train": avg_train_loss,
            "loss/val": avg_val_loss,
            "f1/micro": f1_micro,
            "f1/macro": f1_macro,
            "mAP": mAP,
            "auc/macro": auc_macro,
            "auc/micro": auc_micro,
            "ranking/loss": lrl,
            "recall@3": r3
        }

        # per-class f1 + AP
        for i, v in enumerate(per_class_f1):
            log_dict[f"f1/{CLASS_NAMES[i]}"] = v

        for i, ap in enumerate(per_class_ap):
            log_dict[f"AP/{CLASS_NAMES[i]}"] = ap

        # Single log call
        wandb.log(log_dict)

        # =====================================================
        # Diagnostic visualizations (NO step argument!)
        # =====================================================
        log_confusion_heatmap(all_targets, all_preds, CLASS_NAMES)
        log_error_heatmap(all_targets, all_preds, CLASS_NAMES)
        log_precision_recall(all_targets, all_probs, CLASS_NAMES)
        log_binary_confusion_matrices(all_targets, all_preds, CLASS_NAMES)

        # Save misclassified examples
        csv_path = os.path.join(run_folder, f"misclassified_epoch_{epoch}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "audio_path", "mel_path", "true_labels", "pred_labels"])
            writer.writerows(misclassified_rows)

        print("Saved misclassified CSV:", csv_path)

    return model
