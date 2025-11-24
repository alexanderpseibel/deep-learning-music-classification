import time
from src.training.pipeline_manager import run_training_pipeline
from src.models.paper_crnn_v2_uni import PaperCRNNv2Uni
from src.models.paper_crnn_v2_bi import PaperCRNNv2Bi

# --------------- CONFIG PATH ------------------
CONFIG_PATH = "configs/paper_crnn_v2.yaml"
PROJECT = "NLP-mini-project"

# --------------- TRAIN UNI-GRU ----------------
print("\n===============================")
print("   Training CRNNv2Uni")
print("===============================\n")

run_training_pipeline(
    model_class=PaperCRNNv2Uni,
    model_config_path=CONFIG_PATH,
    project_name=PROJECT
)

print("\nFinished CRNNv2Uni. Sleeping 20 seconds...\n")
time.sleep(20)

# --------------- TRAIN BI-GRU -----------------
print("\n===============================")
print("   Training CRNNv2Bi")
print("===============================\n")

run_training_pipeline(
    model_class=PaperCRNNv2Bi,
    model_config_path=CONFIG_PATH,
    project_name=PROJECT
)

print("\nAll CRNN training completed.\n")
