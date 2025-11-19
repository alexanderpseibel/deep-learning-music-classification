from src.training.pipeline_manager import run_training_pipeline
from src.models.simple_cnn import SimpleCNN

if __name__ == "__main__":
    run_training_pipeline(
        model_class=SimpleCNN,
        model_config_path="configs/simple_cnn.yaml",
        project_name="NLP-mini-project"
    )
