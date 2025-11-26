#scripts/train_simple_cnn.py
from src.training.pipeline_manager import run_training_pipeline
from src.models.old_models.simple_cnn import SimpleCNN

if __name__ == "__main__":
    run_training_pipeline(
        model_class=SimpleCNN,
        model_config_path="configs/old_configs/simple_cnn.yaml",
        project_name="NLP-mini-project"
    )

