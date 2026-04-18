# src/models/model_factory.py
from src.models.cnn_generic import GenericCNN


def build_model(cfg, num_classes):
    """Instantiates the model specified by cfg['model']['type']."""
    mcfg = cfg["model"]
    model_type = mcfg["type"].lower()

    if model_type == "genericcnn":
        return GenericCNN(
            num_classes=num_classes,
            channels=mcfg["channels"],
            kernels=mcfg["kernels"],
            pooling=mcfg["pooling"],
            dropout=mcfg.get("dropout", 0.2),
            activation=mcfg.get("activation", "elu"),
        )

    raise ValueError(f"Unknown model type: {model_type}")
