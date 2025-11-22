#scripts/train_k2c2_cnn.py
from src.training.pipeline_manager import run_training_pipeline
from src.models.k2c2_cnn import ImprovedK2C2CNN

if __name__ == "__main__":
    run_training_pipeline(
        model_class=ImprovedK2C2CNN,
        model_config_path="configs/k2c2_cnn.yaml",
        project_name="NLP-mini-project"
    )
