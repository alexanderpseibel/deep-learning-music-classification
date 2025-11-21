from src.models.improved_cnn_k2c2 import ImprovedK2C2CNN

run_training_pipeline(
    model_class=ImprovedK2C2CNN,
    model_config_path="configs/improved_k2c2.yaml",
)
