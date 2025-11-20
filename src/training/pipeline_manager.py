import os
import yaml
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.data.fma_dataset import FMAAudioDataset
from src.training.wandb_utils import init_wandb
from src.training.trainer import train_model


# ------------------------------------------------------------
# Create incrementing run folder: run_001, run_002, ...
# ------------------------------------------------------------
def create_run_folder(base_dir):
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
    ]

    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[1]))
        except:
            pass

    next_num = max(nums) + 1 if nums else 1
    folder_name = f"run_{next_num:03d}"

    run_path = os.path.join(base_dir, folder_name)
    os.makedirs(run_path)

    return run_path


# ------------------------------------------------------------
# General Training Pipeline
# ------------------------------------------------------------
def run_training_pipeline(model_class, model_config_path, project_name="NLP-mini-project"):

    # --------------------------------------------------------
    # Load YAML configs
    # --------------------------------------------------------
    cfg_paths = yaml.safe_load(open("configs/dataset_paths.yaml"))
    cfg_model = yaml.safe_load(open(model_config_path))
    cfg = {**cfg_paths, **cfg_model}

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------
    df = pd.read_csv(cfg["metadata_csv"])
    print("Loaded metadata rows:", len(df))

    # --------------------------------------------------------
    # Balanced subset selection (if subset_size is set)
    # --------------------------------------------------------
    if "subset_size" in cfg and cfg["subset_size"] is not None:
        subset_size = cfg["subset_size"]
        print(f"Selecting balanced subset of size {subset_size}...")

        label_cols = [c for c in df.columns if c.startswith("label_")]
        y_full = df[label_cols].values

        # fraction of dataset to select
        train_fraction = subset_size / len(df)

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            train_size=train_fraction,
            test_size=1 - train_fraction,
            random_state=42
        )

        subset_idx, _ = next(splitter.split(df, y_full))
        df = df.iloc[subset_idx].reset_index(drop=True)

        print(f"Balanced subset selected: {len(df)} samples.")

    # --------------------------------------------------------
    # Train/valid split
    # --------------------------------------------------------
    label_cols = [c for c in df.columns if c.startswith("label_")]
    y = df[label_cols].values

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.1,
        random_state=42
    )

    for train_idx, valid_idx in splitter.split(df, y):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)

    print("TRAIN size:", len(train_df))
    print("VALID size:", len(valid_df))
    print("Positive counts per class (TRAIN):\n", train_df[label_cols].sum())
    print("Positive counts per class (VALID):\n", valid_df[label_cols].sum())

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------
    train_ds = FMAAudioDataset(train_df)
    valid_ds = FMAAudioDataset(valid_df)

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    num_classes = len([c for c in df.columns if c.startswith("label_")])
    model = model_class(num_classes=num_classes)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    # --------------------------------------------------------
    # W&B Init
    # --------------------------------------------------------
    run_config = init_wandb(config=cfg, project_name=project_name)
    print("W&B run initialized.")

    # Tell W&B to use epoch as x-axis for all loss/* metrics
    wandb.define_metric("epoch")
    wandb.define_metric("loss/*", step_metric="epoch")


    # --------------------------------------------------------
    # Create run folder
    # --------------------------------------------------------
    log_root = cfg["misclassified_examples"]
    run_folder = create_run_folder(log_root)

    print("Saving logs to:", run_folder)
    yaml.safe_dump(cfg, open(os.path.join(run_folder, "config_used.yaml"), "w"))

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        run_folder=run_folder
    )

    print("Training complete.")
