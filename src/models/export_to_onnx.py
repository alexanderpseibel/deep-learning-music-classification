import sys
import os

# Add the project root directory to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import torch
from src.models.baseline_cnn import BaselineCNN

model = BaselineCNN(num_classes=8)
model.eval()

dummy_input = torch.randn(1, 1, 128, 3000)

torch.onnx.export(
    model,
    dummy_input,
    "baseline_cnn.onnx",
    input_names=["mel"],
    output_names=["logits"],
    opset_version=17
)

print("ONNX export completed: baseline_cnn.onnx")
