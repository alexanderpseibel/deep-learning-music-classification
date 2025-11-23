# scripts/train_crnn.py

from src.training.pipeline_manager import run_training_pipeline
from src.models.crnn import CRNNFMA

if __name__ == "__main__":
    run_training_pipeline(
        model_class=CRNNFMA,
        model_config_path="configs/crnn.yaml",
        project_name="NLP-mini-project"
    )
