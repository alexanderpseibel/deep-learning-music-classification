# src/interpretability/visualizer.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb


# ----------------------------------------------------------
# Load mel spectrogram from disk
# ----------------------------------------------------------
def load_mel(path):
    mel = np.load(path).astype(np.float32)
    mel = torch.tensor(mel)[None, None, :, :]  # (1,1,128,T)
    return mel


# ----------------------------------------------------------
# Grad-CAM for the LAST conv block
# ----------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_forward)
        target_layer.register_full_backward_hook(self._save_backward)

    def _save_forward(self, module, input, output):
        self.activations = output.detach()

    def _save_backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        loss = logits[:, class_idx].sum()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)

        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        return cam[0, 0].cpu().numpy()


# ----------------------------------------------------------
# SmoothGrad
# ----------------------------------------------------------
def smoothgrad(model, input_tensor, class_idx, noise_level=0.1, samples=20):

    saliency_total = 0

    for _ in range(samples):
        noisy_input = input_tensor + torch.randn_like(input_tensor) * noise_level
        noisy_input = noisy_input.clone().detach().requires_grad_(True)

        logits = model(noisy_input)
        loss = logits[:, class_idx].sum()

        model.zero_grad()
        loss.backward()

        saliency = noisy_input.grad.abs()
        saliency_total += saliency

    saliency_avg = saliency_total / samples
    s = saliency_avg[0, 0].cpu().numpy()

    s -= s.min()
    s /= (s.max() + 1e-6)
    return s


# ----------------------------------------------------------
# W&B logging helper
# ----------------------------------------------------------
def log_interpretability_images(model, samples, device):

    model.eval()

    target_layer = model.blocks[-1][-3]  # final conv BEFORE Dropout+Pool
    cam_generator = GradCAM(model, target_layer)

    for sample in samples:

        mel = load_mel(sample["path"]).to(device)
        mel = mel.clone().detach().requires_grad_(True)

        class_idx = sample["label"]

        cam = cam_generator.generate(mel, class_idx)
        sg  = smoothgrad(model, mel, class_idx)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        # Mel spectrogram
        ax[0].imshow(mel[0, 0].detach().cpu().numpy(), aspect="auto", origin="lower")
        ax[0].set_title(f"Mel {sample['track_id']}")

        # Grad-CAM
        ax[1].imshow(cam, aspect="auto", origin="lower", cmap="magma")
        ax[1].set_title("Grad-CAM")

        # SmoothGrad
        ax[2].imshow(sg, aspect="auto", origin="lower", cmap="inferno")
        ax[2].set_title("SmoothGrad")

        plt.tight_layout()
        wandb.log({f"interpretability/{sample['track_id']}": wandb.Image(fig)})
        plt.close(fig)
