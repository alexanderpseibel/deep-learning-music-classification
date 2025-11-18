import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np
from src.training.wandb_utils import log_metrics

def train_model(model, train_loader, valid_loader, device, epochs, lr):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        # ---- Training ----
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

        # ---- Validation ----
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for mel, labels in valid_loader:
                mel, labels = mel.to(device), labels.to(device)

                logits = model(mel)
                val_loss += criterion(logits, labels).item()

                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        f1 = f1_score(all_targets, all_preds, average="micro")
        avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | F1: {f1:.4f}")

        # ---- WandB Logging ----
        log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_f1": f1
        }, step=epoch)
