import os
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.fma_dataset import FMAAudioDataset
from src.training.wandb_utils import init_wandb
from src.training.trainer import train_model


def run_training_pipeline(model_class, model_config_path, project_name="NLP-mini-project"):
    """
    General training pipeline:
    - loads dataset configs
    - loads model configs
    - creates dataloaders
    - instantiates model_class
    - initializes W&B
    - creates run folder (loggings/<run_id>)
    - calls the trainer
    """

    # --------------------------------------------------------
    # Load YAML configs
    # --------------------------------------------------------
    cfg_paths = yaml.safe_load(open("configs/dataset_paths.yaml"))
    cfg_model = yaml.safe_load(open(model_config_path))
    cfg = {**cfg_paths, **cfg_model}   # merge

    # --------------------------------------------------------
    # Load dataset metadata
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
    # Create datasets + dataloaders
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
    # Instantiate model
    # --------------------------------------------------------
    num_classes = len([c for c in df.columns if c.startswith("label_")])
    model = model_class(num_classes=num_classes)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    # --------------------------------------------------------
    # Initialize W&B
    # --------------------------------------------------------
    run_config = init_wandb(config=cfg, project_name=project_name)
    run_id = run_config.get("_wandb", {}).get("run_id", "run")
    print("W&B Run ID:", run_id)

    # --------------------------------------------------------
    # Create run folder for logs/misclassified/configs
    # --------------------------------------------------------
    log_root = cfg["misclassified_examples"]
    full_run_path = os.path.join(log_root, f"run_{run_id}")
    os.makedirs(full_run_path, exist_ok=True)

    # Save the model config to the run folder
    yaml.safe_dump(cfg, open(os.path.join(full_run_path, "config_used.yaml"), "w"))

    print("Saving logs to:", full_run_path)

    # --------------------------------------------------------
    # Train loop (delegates to trainer)
    # --------------------------------------------------------
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        run_folder=full_run_path
    )

    print("Training complete.")
