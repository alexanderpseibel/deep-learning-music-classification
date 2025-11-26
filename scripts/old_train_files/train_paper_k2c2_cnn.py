from src.training.pipeline_manager import run_training_pipeline
from src.models.old_models.paper_k2c2_cnn import PaperK2C2CNN

if __name__ == "__main__":
    run_training_pipeline(
        model_class=PaperK2C2CNN,
        model_config_path="configs/old_configs/paper_k2c2_cnn.yaml",
        project_name="NLP-mini-project"
    )
