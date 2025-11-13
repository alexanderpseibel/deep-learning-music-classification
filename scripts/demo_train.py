import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os

from src.wandb_utils import init_wandb, log_metrics


# -------------------------------------------------------------
# 1. CONFIG LOADER
# -------------------------------------------------------------
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------
# 2. SIMPLE EXAMPLE CNN (you will replace with your real model)
# -------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.classifier(x)


# -------------------------------------------------------------
# 3. DUMMY DATASET (replace this with your real dataset later)
# -------------------------------------------------------------
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=256):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        x = torch.randn(1, 128, 128)     # fake mel-spectrogram shape
        y = torch.randint(0, 10, (1,))   # fake label
        return x, y.item()


# -------------------------------------------------------------
# 4. TRAINING FUNCTION
# -------------------------------------------------------------
def train():
    # -----------------------------
    # Load config
    # -----------------------------
    config_path = "configs/cnn.yaml"
    config = load_config(config_path)

    # -----------------------------
    # Init W&B
    # -----------------------------
    config = init_wandb(config)  # logs hyperparameters into W&B

    # -----------------------------
    # Prepare model, data, optimizer
    # -----------------------------
    model = SimpleCNN(num_classes=config["num_classes"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Replace DummyDataset with your real dataset later
    train_set = DummyDataset()
    val_set = DummyDataset()

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(config["epochs"]):
        
        # ---- TRAIN ----
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), torch.tensor(y).to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), torch.tensor(y).to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()

        # ---- LOG TO W&B ----
        log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss},
            step=epoch
        )

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")


# -------------------------------------------------------------
# 5. RUN SCRIPT
# -------------------------------------------------------------
if __name__ == "__main__":
    train()
