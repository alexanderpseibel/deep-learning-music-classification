from src.training.pipeline_manager import run_training_pipeline
from src.models.k2c2_silu_depthwise_cnn import K2C2_SiLU_DepthwiseCNN

if __name__ == "__main__":
    run_training_pipeline(
        model_class=K2C2_SiLU_DepthwiseCNN,
        model_config_path="configs/k2c2_silu_depthwise.yaml",
        project_name="NLP-mini-project"
    )
