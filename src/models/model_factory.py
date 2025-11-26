#src/models/model_factory.py
from src.models.cnn_generic import GenericCNN

def build_model(cfg, num_classes):
    """Constructs the correct model based on cfg['model']['type']"""

    mcfg = cfg["model"]
    model_type = mcfg["type"].lower()

    # ------------------------------------------------------
    # K2C2 Generic
    # ------------------------------------------------------
    if model_type == "genericcnn":
        return GenericCNN(
            num_classes=num_classes,
            channels=mcfg["channels"],
            kernels=mcfg["kernels"],
            pooling=mcfg["pooling"],
            dropout=mcfg.get("dropout", 0.2),
            activation=mcfg.get("activation", "elu")
        )

    # ------------------------------------------------------
    # Future models go here
    # ------------------------------------------------------

    raise ValueError(f"Unknown model type: {model_type}")
