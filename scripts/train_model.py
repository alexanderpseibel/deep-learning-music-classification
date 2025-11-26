# scripts/train_model.py
from src.training.pipeline_manager import run_training_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project", default="NLP-mini-project")
    args = parser.parse_args()

    run_training_pipeline(
        model_config_path=args.config,
        project_name=args.project
    )
