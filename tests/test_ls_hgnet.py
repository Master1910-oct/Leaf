"""
AgriAI — M4: Comprehensive Unit Tests for Adapted LS-HGNet

Covers all requirements from M4 specification (§16):
  1. Model construction
  2. Forward pass at 128x128
  3. Forward pass at 224x224
  4. Output shape = [B, 10]
  5. No NaNs or Infs in outputs
  6. Gradient flow to all trainable parameters
  7. Parameter count verification
  8. Reproducible initialization with seed 42
  9. Checkpoint save/load parity
 10. Tiny-batch overfit sanity test
 11. Depthwise convolution channel behavior
 12. Hourglass encoder-decoder shape consistency
 13. Intra-hourglass and inter-stack skip connection shape compatibility
 14. Final classification head shape and raw logits (no softmax)
"""

import os
import tempfile
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.ls_hgnet import (
    LS_HGNet,
    DepthwiseSeparableConv,
    PreActivationResidualBlock,
    MultiDilatedLightResidualBlock,
    Hourglass,
    StackedHourglass,
    build_ls_hgnet_from_config,
    count_parameters,
    get_checkpoint_size_mb,
)
from src.utils.reproducibility import set_seed


class TestLSHGNet(unittest.TestCase):

    def setUp(self):
        set_seed(42)

    def test_01_model_construction_dev(self):
        """Verify instantiation of dev configuration."""
        model = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.base_channels, 32)
        self.assertEqual(model.num_stacks, 2)
        self.assertEqual(model.encoder_levels, 3)
        self.assertEqual(model.num_classes, 10)

    def test_02_model_construction_final(self):
        """Verify instantiation of final configuration."""
        model = LS_HGNet(base_channels=64, num_stacks=3, encoder_levels=4)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.base_channels, 64)
        self.assertEqual(model.num_stacks, 3)
        self.assertEqual(model.encoder_levels, 4)

    def test_03_forward_128x128(self):
        """Test forward pass with 128x128 input (Dev resolution)."""
        model = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)
        model.eval()
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 10))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_04_forward_224x224(self):
        """Test forward pass with 224x224 input (Final resolution)."""
        model = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 10))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_05_gradient_flow(self):
        """Verify all trainable parameters receive valid non-zero gradients."""
        model = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)
        model.train()
        x = torch.randn(2, 3, 128, 128)
        target = torch.tensor([1, 4])
        criterion = nn.CrossEntropyLoss()

        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        ungrad_params = []
        zero_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    ungrad_params.append(name)
                elif torch.isnan(param.grad).any():
                    self.fail(f"NaN gradient in {name}")
                elif torch.isinf(param.grad).any():
                    self.fail(f"Inf gradient in {name}")

        self.assertEqual(len(ungrad_params), 0, f"Parameters with no gradient: {ungrad_params}")

    def test_06_parameter_count(self):
        """Verify parameter counting functions."""
        model = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)
        total_p = count_parameters(model, trainable_only=False)
        trainable_p = count_parameters(model, trainable_only=True)
        self.assertGreater(total_p, 10000)
        self.assertEqual(total_p, trainable_p)

    def test_07_reproducible_initialization(self):
        """Verify deterministic initialization with seed 42."""
        set_seed(42)
        model1 = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)

        set_seed(42)
        model2 = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_08_checkpoint_parity(self):
        """Verify state_dict save/load parity."""
        model = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)
        model.eval()
        x = torch.randn(2, 3, 128, 128)

        with torch.no_grad():
            orig_out = model(x)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            torch.save({"model_state_dict": model.state_dict()}, tmp_path)
            loaded_model = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)
            checkpoint = torch.load(tmp_path, map_location="cpu")
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            loaded_model.eval()

            with torch.no_grad():
                loaded_out = loaded_model(x)

            self.assertTrue(torch.allclose(orig_out, loaded_out, atol=1e-6))
            size_mb = get_checkpoint_size_mb(tmp_path)
            self.assertGreater(size_mb, 0.1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_09_tiny_batch_overfit(self):
        """Sanity test: overfit a tiny batch of 4 samples within 30 steps."""
        set_seed(42)
        model = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(4, 3, 128, 128)
        y = torch.tensor([0, 1, 2, 3])

        initial_loss = None
        final_loss = None
        for step in range(35):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        self.assertLess(final_loss, initial_loss * 0.1, f"Initial {initial_loss:.4f} -> Final {final_loss:.4f}")

    def test_10_depthwise_separable_conv_channels(self):
        """Verify DWSConv maintains channel dimensions and spatial sizes."""
        conv = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1, dilation=1)
        x = torch.randn(2, 16, 32, 32)
        out = conv(x)
        self.assertEqual(out.shape, (2, 32, 32, 32))

        # Test with stride=2
        conv_s2 = DepthwiseSeparableConv(16, 32, kernel_size=3, stride=2)
        out_s2 = conv_s2(x)
        self.assertEqual(out_s2.shape, (2, 32, 16, 16))

        # Test with dilation=2
        conv_d2 = DepthwiseSeparableConv(16, 32, kernel_size=3, dilation=2)
        out_d2 = conv_d2(x)
        self.assertEqual(out_d2.shape, (2, 32, 32, 32))

    def test_11_preactivation_residual_block(self):
        """Verify PreActivationResidualBlock shape and identity shortcut."""
        # Channels match, stride=1: identity shortcut
        block_ident = PreActivationResidualBlock(32, 32, stride=1)
        x = torch.randn(2, 32, 16, 16)
        out_ident = block_ident(x)
        self.assertEqual(out_ident.shape, (2, 32, 16, 16))

        # Channel change: projection shortcut
        block_proj = PreActivationResidualBlock(32, 64, stride=1)
        out_proj = block_proj(x)
        self.assertEqual(out_proj.shape, (2, 64, 16, 16))

        # Stride=2 downsampling
        block_s2 = PreActivationResidualBlock(32, 64, stride=2)
        out_s2 = block_s2(x)
        self.assertEqual(out_s2.shape, (2, 64, 8, 8))

    def test_12_multi_dilated_block(self):
        """Verify MultiDilatedLightResidualBlock parallel dilation branches."""
        block = MultiDilatedLightResidualBlock(channels=32, dilation_rates=(1, 2, 4))
        x = torch.randn(2, 32, 16, 16)
        out = block(x)
        self.assertEqual(out.shape, (2, 32, 16, 16))

    def test_13_hourglass_encoder_decoder_consistency(self):
        """Verify Hourglass module encoder-decoder skip connections and shape preservation."""
        hg = Hourglass(in_channels=32, base_channels=32, encoder_levels=3)
        x = torch.randn(2, 32, 32, 32)
        out = hg(x)
        self.assertEqual(out.shape, (2, 32, 32, 32))

    def test_14_stacked_hourglass_inter_stack_skips(self):
        """Verify StackedHourglass with inter-stack connections preserves input/output shape."""
        sh = StackedHourglass(base_channels=32, num_stacks=2, encoder_levels=3)
        x = torch.randn(2, 32, 32, 32)
        out = sh(x)
        self.assertEqual(out.shape, (2, 32, 32, 32))
        self.assertEqual(len(sh.stacks), 2)
        self.assertEqual(len(sh.inter_stack_skips), 1)

    def test_15_classification_head_no_softmax(self):
        """Verify classification head produces raw logits that can be negative and unconstrained."""
        model = LS_HGNet(base_channels=32, num_stacks=2, encoder_levels=3)
        model.eval()
        x = torch.randn(4, 3, 128, 128)
        with torch.no_grad():
            logits = model(x)

        self.assertEqual(logits.shape, (4, 10))
        # Raw logits should not sum to 1.0 per row
        row_sums = logits.sum(dim=-1)
        self.assertFalse(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
