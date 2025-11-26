# scripts/train_crnn.py

from src.training.pipeline_manager import run_training_pipeline
from src.models.old_models.crnn import CRNN

if __name__ == "__main__":
    run_training_pipeline(
        model_class=CRNN,
        model_config_path="configs/old_configs/crnn.yaml",
        project_name="NLP-mini-project"
    )
