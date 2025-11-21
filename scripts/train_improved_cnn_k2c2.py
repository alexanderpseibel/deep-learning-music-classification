from src.training.pipeline_manager import run_training_pipeline
from src.models.improved_cnn_k2c2 import ImprovedK2C2CNN

if __name__ == "__main__":
    run_training_pipeline(
        model_class=ImprovedK2C2CNN,
        model_config_path="configs/improved_k2c2.yaml",
        project_name="NLP-mini-project"
    )
