"""
AgriAI — M8.1: Grad-CAM Explainability Engine for Adapted LS-HGNet

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) per
Selvaraju et al. (2017) [ICCV, DOI: 10.1109/ICCV.2017.74] adapted for
the LS-HGNet crop disease classification architecture.

Key Features:
- Seamless forward/backward hook registration and guaranteed cleanup.
- Target layer resolution by string path or direct module reference.
- Normalization safeguards against divide-by-zero on flat activations.
- High-fidelity 3-panel visualization generator (Original | Heatmap | Overlay).
- Comprehensive quantitative diagnostics (min, max, mean, gradient norms).
"""

from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for PyTorch CNN classification models.

    Args:
        model: PyTorch model in evaluation mode.
        target_layer: Either the nn.Module instance or a string path
            (e.g., 'stacked_hourglass.stacks.2.decoder_blocks.3').
    """
    def __init__(self, model: nn.Module, target_layer: Union[nn.Module, str]):
        self.model = model
        self.model.eval()

        if isinstance(target_layer, str):
            self.target_layer_name = target_layer
            self.target_layer = self._resolve_layer_by_name(target_layer)
        elif isinstance(target_layer, nn.Module):
            self.target_layer = target_layer
            self.target_layer_name = self._find_module_name(target_layer)
        else:
            raise TypeError(f"target_layer must be nn.Module or str, got {type(target_layer)}")

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks = []

    def _resolve_layer_by_name(self, name: str) -> nn.Module:
        """Finds a sub-module by dot-separated attribute path or named_modules lookup."""
        named_mods = dict(self.model.named_modules())
        if name in named_mods:
            return named_mods[name]

        # Alternative: traverse attributes directly
        curr = self.model
        for part in name.split("."):
            if hasattr(curr, part):
                curr = getattr(curr, part)
            elif part.isdigit() and isinstance(curr, (nn.Sequential, nn.ModuleList)):
                curr = curr[int(part)]
            else:
                raise AttributeError(f"Could not resolve layer component '{part}' in '{name}'.")
        return curr

    def _find_module_name(self, module: nn.Module) -> str:
        """Finds string name of an nn.Module inside model.named_modules()."""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name
        return module.__class__.__name__

    def _register_hooks(self):
        """Attaches forward and backward hooks."""
        self._clear_hooks()
        self.activations = None
        self.gradients = None

        def forward_hook(module, input_tensor, output_tensor):
            # Capture activations tensor
            if isinstance(output_tensor, (tuple, list)):
                self.activations = output_tensor[0].detach()
            else:
                self.activations = output_tensor.detach()

        def backward_hook(module, grad_input, grad_output):
            # Capture gradient of target score w.r.t layer output
            if isinstance(grad_output, (tuple, list)):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._hooks.extend([h_fwd, h_bwd])

    def _clear_hooks(self):
        """Removes registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, float, Dict[str, Any]]:
        """
        Computes Grad-CAM heatmap for a single input image.

        Args:
            input_tensor: Image tensor of shape [1, 3, H, W] or [3, H, W].
            target_class: Optional ground-truth or specific class index.
                If None, uses model's predicted class (argmax).

        Returns:
            cam_map: 2D Tensor of shape [H, W], normalized in [0.0, 1.0].
            pred_class: Integer predicted class index.
            confidence: Softmax probability of predicted class in [0.0, 1.0].
            diagnostics: Dictionary containing audit statistics.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(f"input_tensor must be [1, 3, H, W], got {input_tensor.shape}")

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        B, C, H, W = input_tensor.shape

        self._register_hooks()
        try:
            # 1. Forward pass
            self.model.zero_grad()
            logits = self.model(input_tensor)  # [1, num_classes]
            probabilities = F.softmax(logits, dim=-1)
            pred_class = int(logits.argmax(dim=-1).item())
            pred_confidence = float(probabilities[0, pred_class].item())

            # 2. Select target score
            chosen_class = pred_class if target_class is None else int(target_class)
            target_score = logits[0, chosen_class]
            target_confidence = float(probabilities[0, chosen_class].item())

            # 3. Backward pass
            target_score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

            activations = self.activations  # [1, C_feat, H_feat, W_feat]
            gradients = self.gradients      # [1, C_feat, H_feat, W_feat]

            # 4. Compute channel importance weights (Global Average Pooling over spatial dimensions)
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C_feat, 1, 1]

            # 5. Weighted combination of activation maps
            cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H_feat, W_feat]

            # 6. Apply ReLU (retain only positive features)
            cam = F.relu(cam)

            # 7. Bilinear interpolation to input image resolution (H, W)
            cam_interpolated = F.interpolate(
                cam,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze()  # [H, W]

            # 8. Min-max normalization into [0.0, 1.0]
            cam_min = float(cam_interpolated.min().item())
            cam_max = float(cam_interpolated.max().item())

            if cam_max > cam_min:
                cam_normalized = (cam_interpolated - cam_min) / (cam_max - cam_min)
            else:
                cam_normalized = torch.zeros_like(cam_interpolated)

            # Diagnostics for scientific audit
            diagnostics = {
                "target_layer": self.target_layer_name,
                "feature_map_shape": list(activations.shape),
                "gradient_norm": float(torch.norm(gradients).item()),
                "raw_cam_min": cam_min,
                "raw_cam_max": cam_max,
                "normalized_cam_min": float(cam_normalized.min().item()),
                "normalized_cam_max": float(cam_normalized.max().item()),
                "normalized_cam_mean": float(cam_normalized.mean().item()),
                "predicted_class": pred_class,
                "predicted_confidence": pred_confidence,
                "target_class": chosen_class,
                "target_confidence": target_confidence,
                "target_score": float(target_score.item()),
            }

            return cam_normalized.cpu(), pred_class, pred_confidence, diagnostics

        finally:
            self._clear_hooks()


def overlay_cam_on_image(
    image: Union[np.ndarray, Image.Image],
    cam: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> np.ndarray:
    """
    Overlays a normalized Grad-CAM heatmap onto an RGB image.

    Args:
        image: RGB image as PIL Image or numpy array in [0, 1] or [0, 255] of shape (H, W, 3).
        cam: 2D heatmap in [0.0, 1.0] of shape (H, W).
        alpha: Blending weight for original image (1 - alpha for heatmap).
        colormap_name: Matplotlib colormap name ('jet', 'viridis', 'turbo').

    Returns:
        Blended RGB image as numpy float array in [0.0, 1.0] with shape (H, W, 3).
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image).astype(np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        img_np = image.copy().astype(np.float32)
        if img_np.max() > 1.0:
            img_np /= 255.0
    else:
        raise TypeError("image must be PIL.Image or numpy.ndarray")

    if isinstance(cam, torch.Tensor):
        cam_np = cam.detach().cpu().numpy()
    else:
        cam_np = np.asarray(cam, dtype=np.float32)

    # Ensure 2D
    if cam_np.ndim != 2:
        raise ValueError(f"cam must be 2D array, got shape {cam_np.shape}")

    # Clip to [0, 1]
    cam_np = np.clip(cam_np, 0.0, 1.0)

    # Colorize heatmap
    try:
        cmap = plt.colormaps[colormap_name]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(colormap_name)
    heatmap_rgba = cmap(cam_np)  # [H, W, 4] in [0, 1]
    heatmap_rgb = heatmap_rgba[..., :3]  # [H, W, 3]

    # Resize image if dimensions differ
    if img_np.shape[:2] != cam_np.shape:
        from PIL import Image as PImg
        pil_img = PImg.fromarray((img_np * 255).astype(np.uint8))
        pil_img = pil_img.resize((cam_np.shape[1], cam_np.shape[0]), PImg.BILINEAR)
        img_np = np.array(pil_img).astype(np.float32) / 255.0

    # Alpha blending
    blended = (alpha * img_np) + ((1.0 - alpha) * heatmap_rgb)
    blended = np.clip(blended, 0.0, 1.0)
    return blended


def create_gradcam_3panel_figure(
    original_rgb: np.ndarray,
    cam_map: np.ndarray,
    overlay_rgb: np.ndarray,
    title_meta: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Creates a clean 3-panel figure: Original | Grad-CAM Heatmap | Overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=dpi)

    true_name = title_meta.get("true_class", "N/A")
    pred_name = title_meta.get("predicted_class", "N/A")
    conf = title_meta.get("confidence", 0.0)
    is_correct = title_meta.get("is_correct", True)
    status_str = "CORRECT" if is_correct else "MISCLASSIFIED"
    status_color = "#1b5e20" if is_correct else "#b71c1c"

    # Panel 1: Original
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Input Image", fontsize=12, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: Grad-CAM Heatmap
    im = axes[1].imshow(cam_map, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Activation Heatmap", fontsize=12, fontweight="bold", pad=8)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Blended Overlay
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Heatmap Overlay", fontsize=12, fontweight="bold", pad=8)
    axes[2].axis("off")

    # Suptitle
    fig.suptitle(
        f"True: {true_name}  |  Pred: {pred_name}  |  Conf: {conf * 100:.1f}%  [{status_str}]",
        fontsize=13,
        fontweight="bold",
        color=status_color,
        y=0.98,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    return fig
