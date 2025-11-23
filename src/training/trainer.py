# src/training/trainer.py
import os
import numpy as np
import torch
import csv

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    label_ranking_loss
)

import wandb

from src.training.wandb_utils import (
    compute_map_and_per_class_ap,
    recall_at_k,
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
# TRAINING LOOP
# ---------------------------------------------------------
def train_model(model, train_loader, valid_loader, device, epochs, lr, weight_decay, run_folder, cfg):

    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    # -----------------------------------------
    # Optimizer
    # -----------------------------------------
    opt_name = cfg.get("optimizer", "adam").lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("max_lr", lr),
            weight_decay=weight_decay
        )
        print(">>> Using AdamW optimizer")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        print(">>> Using Adam optimizer")

    # -----------------------------------------
    # Scheduler
    # -----------------------------------------
    scheduler_name = cfg.get("scheduler", "none").lower()

    if scheduler_name == "onecycle":
        max_lr = cfg.get("max_lr", lr * 10)
        final_lr = cfg.get("final_lr", max_lr / 25)
        pct_start = cfg.get("pct_start", 0.3)

        final_div_factor = max_lr / final_lr

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=pct_start,
            anneal_strategy="cos",
            final_div_factor=final_div_factor
        )

        print(f">>> Using OneCycleLR (max_lr={max_lr}, final_lr={final_lr}, pct_start={pct_start})")

    else:
        scheduler = None
        print(">>> No LR scheduler")

    # ---------------------------------------------------------
    # MAIN TRAINING LOOP
    # ---------------------------------------------------------
    for epoch in range(epochs):

        # =====================================================
        # TRAIN
        # =====================================================
        model.train()
        total_loss = 0

        for mel, labels in train_loader:
            mel, labels = mel.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, labels)
            loss.backward()

            #GRADIENT LOGGING
            from src.training.wandb_utils import log_gradients_norm
            log_gradients_norm(model)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # =====================================================
        # VALIDATION
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

                # Misclassified tracking
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

        # Stack arrays
        all_targets = np.vstack(all_targets)
        all_probs = np.vstack(all_probs)
        all_preds = np.vstack(all_preds)
        avg_val_loss = val_loss / len(valid_loader)

        # =====================================================
        # METRICS
        # =====================================================
        f1_micro = f1_score(all_targets, all_preds, average="micro")
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        per_class_f1 = f1_score(all_targets, all_preds, average=None)

        mAP, per_class_ap = compute_map_and_per_class_ap(all_targets, all_probs)

        try:
            auc_macro = roc_auc_score(all_targets, all_probs, average="macro")
        except:
            auc_macro = 0.0

        try:
            auc_micro = roc_auc_score(all_targets, all_probs, average="micro")
        except:
            auc_micro = 0.0

        lrl = label_ranking_loss(all_targets, all_probs)
        r3 = recall_at_k(all_targets, all_probs, k=3)

        print(
            f"Epoch {epoch+1}/{epochs} | Train={avg_train_loss:.4f} | "
            f"Val={avg_val_loss:.4f} | F1micro={f1_micro:.4f} | mAP={mAP:.4f} | R@3={r3:.4f}"
        )

        # =====================================================
        # W&B LOGGING
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

        # Per-class metrics
        for i, v in enumerate(per_class_f1):
            log_dict[f"f1/{CLASS_NAMES[i]}"] = v
        for i, ap in enumerate(per_class_ap):
            log_dict[f"AP/{CLASS_NAMES[i]}"] = ap

        try:
            wandb.log(log_dict)
        except Exception as e:
            print(f"[WARN] WANDB log failed: {e}")


        # =====================================================
        # VISUALIZATIONS
        # =====================================================
        VIS_EVERY = 5   # log every 5 epochs (or set 10)

        if (epoch % VIS_EVERY == 0) or (epoch == epochs - 1):
            try:
                log_confusion_heatmap(all_targets, all_preds, CLASS_NAMES)
                log_error_heatmap(all_targets, all_preds, CLASS_NAMES)
                log_precision_recall(all_targets, all_probs, CLASS_NAMES)
                log_binary_confusion_matrices(all_targets, all_preds, CLASS_NAMES)
            except Exception as e:
                print(f"[WARN] WANDB visualization failed: {e}")


        # Save misclassified samples
        csv_path = os.path.join(run_folder, f"misclassified_epoch_{epoch}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "audio_path", "mel_path", "true_labels", "pred_labels"])
            writer.writerows(misclassified_rows)

    return model
