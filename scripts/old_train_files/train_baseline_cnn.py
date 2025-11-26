# scripts/train_baseline_cnn.py
import yaml
from src.models.old_models.baseline_cnn import BaselineCNN
from src.training.pipeline_manager import run_training_pipeline

if __name__ == "__main__":
    model_config_path = "configs/old_configs/baseline_cnn.yaml"
    run_training_pipeline(
        model_class=BaselineCNN,
        model_config_path=model_config_path,
        project_name="NLP-mini-project"
    )
