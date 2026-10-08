"""
Unit and Integration Tests for M8.1 Grad-CAM Explainability Module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from src.models.ls_hgnet import LS_HGNet
from src.evaluation.gradcam import (
    GradCAM,
    overlay_cam_on_image,
    create_gradcam_3panel_figure,
)


@pytest.fixture
def small_lshgnet():
    """Builds a lightweight LS-HGNet instance for fast CPU unit testing."""
    torch.manual_seed(42)
    model = LS_HGNet(
        in_channels=3,
        num_classes=10,
        base_channels=16,
        num_stacks=2,
        encoder_levels=2,
        dilation_rates=(1, 2),
        dropout=0.1,
    )
    model.eval()
    return model


def test_gradcam_initialization_with_module_and_string(small_lshgnet):
    """Verifies target layer resolution via both module reference and string path."""
    target_mod = small_lshgnet.stacked_hourglass.stacks[1].decoder_blocks[1]
    gcam1 = GradCAM(small_lshgnet, target_mod)
    assert gcam1.target_layer is target_mod

    gcam2 = GradCAM(small_lshgnet, "stacked_hourglass.stacks.1.decoder_blocks.1")
    assert gcam2.target_layer is target_mod

    with pytest.raises(Exception):
        GradCAM(small_lshgnet, "non_existent_module_path")


def test_gradcam_output_shape_and_range(small_lshgnet):
    """Verifies CAM tensor shape [H, W], bounds [0, 1], and finite values."""
    target_mod = small_lshgnet.stacked_hourglass.stacks[1].decoder_blocks[1]
    gcam = GradCAM(small_lshgnet, target_mod)

    x = torch.randn(1, 3, 128, 128)
    cam, pred_class, conf, diag = gcam.generate_cam(x)

    assert cam.shape == (128, 128)
    assert not torch.isnan(cam).any()
    assert not torch.isinf(cam).any()
    assert cam.min().item() >= 0.0
    assert cam.max().item() <= 1.0 + 1e-6
    assert 0 <= pred_class < 10
    assert 0.0 <= conf <= 1.0

    assert diag["gradient_norm"] > 0.0
    assert diag["normalized_cam_min"] >= 0.0
    assert diag["normalized_cam_max"] <= 1.0


def test_gradcam_preserves_model_weights(small_lshgnet):
    """Ensures that backward pass inside Grad-CAM does not modify model weights."""
    target_mod = small_lshgnet.stacked_hourglass.stacks[1].decoder_blocks[1]
    gcam = GradCAM(small_lshgnet, target_mod)

    # Save copy of initial parameter values
    orig_params = [p.clone() for p in small_lshgnet.parameters()]

    x = torch.randn(1, 3, 64, 64)
    _ = gcam.generate_cam(x)

    for p_orig, p_now in zip(orig_params, small_lshgnet.parameters()):
        assert torch.equal(p_orig, p_now), "Model parameters were altered by Grad-CAM execution!"


def test_gradcam_hook_cleanup(small_lshgnet):
    """Ensures hooks are completely removed after generate_cam execution."""
    target_mod = small_lshgnet.stacked_hourglass.stacks[1].decoder_blocks[1]
    gcam = GradCAM(small_lshgnet, target_mod)

    x = torch.randn(1, 3, 64, 64)
    _ = gcam.generate_cam(x)

    assert len(gcam._hooks) == 0
    # Also verify target module hook dictionaries are empty
    assert len(target_mod._forward_hooks) == 0
    assert len(target_mod._backward_hooks) == 0


def test_overlay_cam_on_image():
    """Verifies image overlay dimensions, ranges, and types."""
    cam = np.random.uniform(0.0, 1.0, (100, 100)).astype(np.float32)
    img_arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    overlay = overlay_cam_on_image(img_arr, cam, alpha=0.5)
    assert overlay.shape == (100, 100, 3)
    assert overlay.min() >= 0.0
    assert overlay.max() <= 1.0

    # Test PIL input
    pil_img = Image.fromarray(img_arr)
    overlay_pil = overlay_cam_on_image(pil_img, cam, alpha=0.6)
    assert overlay_pil.shape == (100, 100, 3)


def test_create_gradcam_3panel_figure(tmp_path):
    """Verifies 3-panel figure creation and saving."""
    orig = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    cam = np.zeros((64, 64), dtype=np.float32)
    overlay = np.ones((64, 64, 3), dtype=np.float32) * 0.8

    meta = {
        "true_class": "Tomato___healthy",
        "predicted_class": "Tomato___healthy",
        "confidence": 0.95,
        "is_correct": True,
    }

    out_file = tmp_path / "test_panel.png"
    fig = create_gradcam_3panel_figure(orig, cam, overlay, meta, save_path=out_file)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
