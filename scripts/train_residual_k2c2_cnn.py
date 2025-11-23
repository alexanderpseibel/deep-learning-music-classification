from src.training.pipeline_manager import run_training_pipeline
from src.models.residual_k2c2_cnn import ResidualK2C2CNN

if __name__ == "__main__":
    run_training_pipeline(
        model_class=ResidualK2C2CNN,
        model_config_path="configs/residual_k2c2_cnn.yaml",
        project_name="NLP-mini-project"
    )
