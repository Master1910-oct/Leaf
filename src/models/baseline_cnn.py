"""
AgriAI — M3: Baseline CNN Model Architecture
Implements a lightweight 4-block Convolutional Neural Network (~500K parameters)
to serve as an experimental control for subsequent LS-HGNet and STOA evaluations.

Architectural Specification:
    Input: [B, C, H, W]
      ↓
    Conv Block 1: Conv2D(3 -> 32, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
      ↓
    Conv Block 2: Conv2D(32 -> 64, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
      ↓
    Conv Block 3: Conv2D(64 -> 128, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
      ↓
    Conv Block 4: Conv2D(128 -> 256, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
      ↓
    Global Average Pooling: AdaptiveAvgPool2D(1x1)
      ↓
    Classifier: Flatten -> Linear(256 -> 256) -> ReLU -> Dropout(0.2) -> Linear(256 -> num_classes)
      ↓
    Output Logits: [B, num_classes]
"""

from typing import Tuple, Dict, Any
import os
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard convolutional block consisting of:
    Conv2D(kernel_size=3, padding=1) -> BatchNorm2D -> ReLU -> MaxPool2D(kernel_size=2, stride=2)
    """
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Baseline Convolutional Neural Network for PlantVillage Tomato disease classification.
    
    Target Parameter Count: ~500K parameters
    Design Principle: Lightweight, reproducible conventional CNN control.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channels: Tuple[int, ...] = (32, 64, 128, 256),
        fc_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.fc_dim = fc_dim
        self.dropout_rate = dropout

        # Feature extraction backbone (4 Conv blocks)
        conv_blocks = []
        curr_in = in_channels
        for c_out in channels:
            conv_blocks.append(ConvBlock(curr_in, c_out, pool=True))
            curr_in = c_out
        self.features = nn.Sequential(*conv_blocks)

        # Global average pooling reduces spatial dimensions to 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, num_classes)
        )

        # Deterministic weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Applies Kaiming normal initialization to conv/linear layers and constants to BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape [B, in_channels, H, W]
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        features = self.features(x)
        pooled = self.gap(features)
        logits = self.classifier(pooled)
        return logits

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Returns the total number of parameters (trainable by default)."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_model_summary(self) -> Dict[str, Any]:
        """Returns structured metadata about the model architecture and size."""
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_mb = (param_bytes + buffer_bytes) / (1024 * 1024)

        return {
            "architecture": "BaselineCNN",
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "channels": list(self.channels),
            "fc_dim": self.fc_dim,
            "dropout": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_model_size_mb": round(model_size_mb, 4)
        }


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Helper function to count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_checkpoint_size_mb(path: str) -> float:
    """Returns file size on disk in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)
