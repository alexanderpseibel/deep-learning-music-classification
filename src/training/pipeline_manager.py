# src/training/pipeline_manager.py
import os
import random
import yaml
import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.data.fma_dataset import FMAAudioDataset
from src.training.wandb_utils import init_wandb
from src.training.trainer import train_model
from src.models.model_factory import build_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except ValueError:
            pass
    path = os.path.join(base_dir, f"run_{max(nums) + 1 if nums else 1:03d}")
    os.makedirs(path)
    return path


def run_training_pipeline(model_config_path, project_name="NLP-mini-project"):
    cfg_paths = yaml.safe_load(open("configs/dataset_paths.yaml"))
    cfg_model = yaml.safe_load(open(model_config_path))
    cfg = {**cfg_paths, **cfg_model}

    seed = cfg.get("seed", 42)
    set_seed(seed)
    print(f"Seed: {seed}")

    df = pd.read_csv(cfg["metadata_csv"])
    print(f"Loaded metadata: {len(df)} rows")

    label_cols = [c for c in df.columns if c.startswith("label_")]

    if cfg.get("subset_size") is not None:
        subset_size = cfg["subset_size"]
        if subset_size < len(df):
            print(f"Selecting stratified subset of {subset_size}...")
            y_full = df[label_cols].values
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                train_size=subset_size / len(df),
                test_size=1 - subset_size / len(df),
                random_state=seed,
            )
            subset_idx, _ = next(splitter.split(df, y_full))
            df = df.iloc[subset_idx].reset_index(drop=True)
            print(f"Subset size: {len(df)}")
        else:
            print("Subset size >= dataset, using full dataset.")

    y = df[label_cols].values
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    train_idx, valid_idx = next(splitter.split(df, y))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(valid_df)}")

    if cfg.get("use_normalization", False):
        mean, std = cfg["mel_mean"], cfg["mel_std"]
        print(f">>> Normalization on (mean={mean}, std={std})")
    else:
        mean, std = None, None
        print(">>> Normalization off")

    train_ds = FMAAudioDataset(train_df, mean=mean, std=std)
    valid_ds = FMAAudioDataset(valid_df, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)

    num_classes = len(label_cols)
    model = build_model(cfg, num_classes)
    print(f"Model: {cfg['model']['type']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    init_wandb(config=cfg, project_name=project_name)
    wandb.define_metric("epoch")
    wandb.define_metric("loss/*", step_metric="epoch")

    run_folder = create_run_folder(cfg["misclassified_examples"])
    print(f"Run folder: {run_folder}")
    yaml.safe_dump(cfg, open(os.path.join(run_folder, "config_used.yaml"), "w"))

    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        run_folder=run_folder,
        cfg=cfg,
    )

    print("Training complete.")
