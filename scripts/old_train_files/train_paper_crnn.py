from src.training.pipeline_manager import run_training_pipeline
from src.models.old_models.paper_crnn import PaperCRNN

if __name__ == "__main__":
    run_training_pipeline(
        model_class=PaperCRNN,
        model_config_path="configs/old_configs/paper_crnn.yaml",
        project_name="NLP-mini-project"
    )
