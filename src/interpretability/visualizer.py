# src/interpretability/visualizer.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb


def load_mel(path):
    mel = np.load(path).astype(np.float32)
    return torch.tensor(mel)[None, None, :, :]  # (1, 1, 128, T)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
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
        logits[:, class_idx].sum().backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        return cam[0, 0].cpu().numpy()


def smoothgrad(model, input_tensor, class_idx, noise_level=0.1, samples=20):
    saliency_total = 0
    for _ in range(samples):
        noisy = (input_tensor + torch.randn_like(input_tensor) * noise_level).clone().detach().requires_grad_(True)
        logits = model(noisy)
        model.zero_grad()
        logits[:, class_idx].sum().backward()
        saliency_total += noisy.grad.abs()
    s = (saliency_total / samples)[0, 0].cpu().numpy()
    s -= s.min()
    s /= (s.max() + 1e-6)
    return s


def log_interpretability_images(model, samples, device):
    model.eval()
    target_layer = model.blocks[-1][-3]  # last conv before Dropout+Pool
    cam_generator = GradCAM(model, target_layer)

    for sample in samples:
        mel = load_mel(sample["path"]).to(device)
        mel = mel.clone().detach().requires_grad_(True)
        class_idx = sample["label"]

        cam = cam_generator.generate(mel, class_idx)
        sg = smoothgrad(model, mel, class_idx)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(mel[0, 0].detach().cpu().numpy(), aspect="auto", origin="lower")
        ax[0].set_title(f"Mel {sample['track_id']}")
        ax[1].imshow(cam, aspect="auto", origin="lower", cmap="magma")
        ax[1].set_title("Grad-CAM")
        ax[2].imshow(sg, aspect="auto", origin="lower", cmap="inferno")
        ax[2].set_title("SmoothGrad")
        plt.tight_layout()
        wandb.log({f"interpretability/{sample['track_id']}": wandb.Image(fig)})
        plt.close(fig)
