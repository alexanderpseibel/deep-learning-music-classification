# scripts/train_simple_cnn.py
import yaml
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.fma_dataset import FMAAudioDataset
from src.models.simple_cnn import SimpleCNN
from src.training.trainer import train_model
from src.training.wandb_utils import init_wandb


# -------- Load config --------
cfg_paths = yaml.safe_load(open("configs/dataset_paths.yaml"))
cfg_model = yaml.safe_load(open("configs/simple_cnn.yaml"))
cfg = {**cfg_paths, **cfg_model}     # merge dictionaries


# -------- Initialize WandB --------
init_wandb(config=cfg, project_name="NLP-mini-project")


# -------- Load metadata --------
df = pd.read_csv(cfg["metadata_csv"])
print("Loaded metadata rows:", len(df))

if "limit_samples" in cfg:
    df = df.sample(n=cfg["limit_samples"], random_state=42).reset_index(drop=True)

# -------- Train/Validation Split --------
# Multi-label stratification not possible → random shuffle split
train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)


# -------- Create Datasets --------
train_ds = FMAAudioDataset(train_df)
valid_ds = FMAAudioDataset(valid_df)


# -------- Create DataLoaders --------
train_loader = DataLoader(
    train_ds,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=cfg["num_workers"],
    pin_memory=True
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=cfg["num_workers"],
    pin_memory=True
)


# -------- Create Model --------
num_classes = len([c for c in df.columns if c.startswith("label_")])
model = SimpleCNN(num_classes=num_classes)


# -------- Select Device --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)


# -------- Train Model --------
train_model(
    model,
    train_loader,
    valid_loader,
    device=device,
    epochs=cfg["epochs"],
    lr=cfg["lr"]
)

