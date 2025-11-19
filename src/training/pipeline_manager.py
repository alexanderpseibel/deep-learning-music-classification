import os
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.fma_dataset import FMAAudioDataset
from src.training.wandb_utils import init_wandb
from src.training.trainer import train_model


# ------------------------------------------------------------
# Create incrementing run folder: run_001, run_002, ...
# ------------------------------------------------------------
def create_run_folder(base_dir):
    """
    Creates an incrementing run folder inside base_dir:
    run_001, run_002, ...
    """
    os.makedirs(base_dir, exist_ok=True)

    # list existing
    existing = [
        d for d in os.listdir(base_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_dir, d))
    ]

    # extract run numbers
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
#     General Training Pipeline (works for any model class)
# ------------------------------------------------------------
def run_training_pipeline(model_class, model_config_path, project_name="NLP-mini-project"):
    """
    General training pipeline:
    - loads dataset configs
    - loads model configs
    - creates dataloaders
    - instantiates model_class
    - initializes W&B
    - creates run folder (loggings/run_XXX)
    - calls the trainer
    """

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

    if "limit_samples" in cfg:
        df = df.sample(cfg["limit_samples"], random_state=42).reset_index(drop=True)

    # --------------------------------------------------------
    # Train/valid split
    # --------------------------------------------------------
    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=42)

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

    # --------------------------------------------------------
    # Create run folder
    # --------------------------------------------------------
    log_root = cfg["misclassified_examples"]
    run_folder = create_run_folder(log_root)

    print("Saving logs to:", run_folder)

    # Save config to run folder
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
