"""
AgriAI — M3 Baseline CNN Verification Suite
Tests required by M3 milestone specification:
1. Forward pass (shapes [B, 3, 128, 128] and [B, 3, 224, 224])
2. Parameter count verification (~500K lightweight target)
3. Gradient flow verification (non-zero gradients across all trainable layers)
4. Tiny-batch overfit test (loss reduction and convergence sanity check)
5. Checkpoint save, load, and inference parity test
6. Reproducibility test across identical seeds
"""

import unittest
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.baseline_cnn import BaselineCNN, count_parameters
from src.utils.reproducibility import set_seed


class TestBaselineCNN(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.num_classes = 10
        self.model = BaselineCNN(in_channels=3, num_classes=self.num_classes)

    def test_01_forward_pass_dimensions(self):
        """Test 1: Forward pass produces correct logit shapes for dev (128x128) and final (224x224)."""
        batch_size = 4
        
        # Development resolution: 128x128
        x_dev = torch.randn(batch_size, 3, 128, 128)
        out_dev = self.model(x_dev)
        self.assertEqual(out_dev.shape, (batch_size, self.num_classes),
                         f"Expected shape ({batch_size}, {self.num_classes}), got {out_dev.shape}")
        
        # Final resolution: 224x224
        x_final = torch.randn(batch_size, 3, 224, 224)
        out_final = self.model(x_final)
        self.assertEqual(out_final.shape, (batch_size, self.num_classes),
                         f"Expected shape ({batch_size}, {self.num_classes}), got {out_final.shape}")

    def test_02_parameter_count(self):
        """Test 2: Parameter count must be within ~400K-550K range (targeting ~500K)."""
        trainable_params = count_parameters(self.model, trainable_only=True)
        total_params = count_parameters(self.model, trainable_only=False)
        self.assertEqual(trainable_params, total_params)
        self.assertGreater(trainable_params, 400_000, f"Params too low: {trainable_params}")
        self.assertLess(trainable_params, 550_000, f"Params too high: {trainable_params}")
        print(f"\n[Test 2] Verified parameter count: {trainable_params:,} parameters")

    def test_03_gradient_flow(self):
        """Test 3: Gradients must propagate to all trainable parameters."""
        self.model.train()
        x = torch.randn(2, 3, 128, 128)
        targets = torch.tensor([1, 4])
        criterion = nn.CrossEntropyLoss()
        
        outputs = self.model(x)
        loss = criterion(outputs, targets)
        loss.backward()

        zero_grad_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient.")
                if param.grad.abs().sum().item() == 0:
                    zero_grad_layers.append(name)
        
        self.assertEqual(len(zero_grad_layers), 0, f"Zero gradient in layers: {zero_grad_layers}")

    def test_04_tiny_batch_overfit(self):
        """Test 4: Model can rapidly overfit a tiny batch (8 samples) to near-zero loss."""
        set_seed(42)
        model = BaselineCNN(in_channels=3, num_classes=self.num_classes)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        batch_size = 8
        x = torch.randn(batch_size, 3, 128, 128)
        y = torch.randint(0, self.num_classes, (batch_size,))

        initial_loss = criterion(model(x), y).item()
        
        # Train for 40 steps
        for _ in range(40):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        final_loss = criterion(model(x), y).item()
        preds = torch.argmax(model(x), dim=1)
        accuracy = (preds == y).float().mean().item()

        self.assertLess(final_loss, 0.05, f"Overfit loss {final_loss:.4f} not sufficiently low (<0.05)")
        self.assertEqual(accuracy, 1.0, f"Overfit accuracy {accuracy*100:.1f}% is not 100%")
        print(f"\n[Test 4] Tiny-batch overfit: Initial Loss {initial_loss:.4f} -> Final Loss {final_loss:.5f}, Acc {accuracy*100:.1f}%")

    def test_05_checkpoint_save_and_load(self):
        """Test 5: Saving and loading model weights reproduces exact predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_baseline.pt"
            
            # Save state
            torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)
            self.assertTrue(ckpt_path.exists())

            # Load into fresh instance
            loaded_model = BaselineCNN(in_channels=3, num_classes=self.num_classes)
            checkpoint = torch.load(ckpt_path)
            loaded_model.load_state_dict(checkpoint["model_state_dict"])
            
            self.model.eval()
            loaded_model.eval()

            test_input = torch.randn(4, 3, 128, 128)
            with torch.no_grad():
                out1 = self.model(test_input)
                out2 = loaded_model(test_input)

            self.assertTrue(torch.allclose(out1, out2, atol=1e-6), "Outputs differ after loading checkpoint")

    def test_06_reproducibility(self):
        """Test 6: Identical seed produces identical weight initialization and forward pass."""
        set_seed(100)
        m1 = BaselineCNN(in_channels=3, num_classes=self.num_classes)
        x1 = torch.randn(2, 3, 128, 128)

        set_seed(100)
        m2 = BaselineCNN(in_channels=3, num_classes=self.num_classes)
        x2 = torch.randn(2, 3, 128, 128)

        # Inputs match
        self.assertTrue(torch.equal(x1, x2), "Seeded inputs are not identical")

        # Weights match
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2), f"Parameter {n1} differed under identical seed")

        # Forward outputs match
        m1.eval()
        m2.eval()
        with torch.no_grad():
            y1 = m1(x1)
            y2 = m2(x2)
        self.assertTrue(torch.equal(y1, y2), "Outputs differed under identical seed")


if __name__ == "__main__":
    unittest.main()
