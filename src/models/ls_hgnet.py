"""
AgriAI — M4: Adapted Lightweight Stacked Hourglass Network (LS-HGNet)

This implementation adapts the LS-HGNet architecture for tomato leaf disease
classification. The original source work (Krishnan et al., 2024, IJERR, Vol.37(spl),
pp. 96-108) addressed UAV-based maize tassel detection/counting. This is an
independent adaptation for a 10-class foliar image classification task on the
PlantVillage tomato subset.

Research Boundaries & Specifications:
  - Source application: UAV maize tassel detection/counting (Krishnan et al., 2024)
  - AgriAI adaptation: Tomato foliar disease classification (10 classes)
  - Architectural lineage:
      * Newell et al. (2016), Stacked Hourglass Networks (ECCV) — hourglass topology
      * Howard et al. (2017), MobileNets — depthwise-separable convolution
      * Yu & Koltun (2016), Multi-scale context aggregation — dilated convolution
      * He et al. (2016), Identity Mappings — pre-activation residual blocks

Modular Architecture Components:
  1. PreActivationResidualBlock: ReLU -> BatchNorm -> Convolution with identity shortcut
  2. DepthwiseSeparableConv: Depthwise (k x k, groups=C) + Pointwise (1x1)
  3. MultiDilatedLightResidualBlock: Parallel DWSConv branches (rates [1,2,4]) + 1x1 fusion + residual
  4. Hourglass: Encoder (L levels) -> MultiDilated Bottleneck -> Decoder (L levels) + skip connections
  5. StackedHourglass: Sequential Hourglass stacks with additive inter-stack skip connections
  6. LS_HGNet: Stem -> StackedHourglass -> Global Average Pooling -> Dropout -> Linear(10) -> Logits
"""

from typing import List, Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building Block 1: Depthwise-Separable Convolution
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-Separable Convolution (DWSConv).

    Decomposes standard spatial convolution into:
      1. Depthwise convolution: C_in filters of spatial size (kernel_size x kernel_size),
         one per input channel (groups=in_channels).
      2. Pointwise convolution: 1x1 convolution projecting to C_out channels.

    Source basis: Howard et al. (2017), MobileNets (arXiv:1704.04861).
    Configuration per M0 §12.6.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_dw(self.depthwise(x)))
        x = self.relu(self.bn_pw(self.pointwise(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Building Block 2: Pre-Activation Lightweight Residual Block
# ─────────────────────────────────────────────────────────────────────────────

class PreActivationResidualBlock(nn.Module):
    """
    Pre-Activation Lightweight Residual Block using Depthwise-Separable Convolutions.

    Ordering per M4 specification:
        Input
          │
          ├─── [Projection Shortcut if in_ch != out_ch or stride != 1] ───┐
          │                                                               │
          ▼                                                               │
        ReLU                                                              │
          ▼                                                               │
        BatchNorm2d                                                       │
          ▼                                                               │
        DepthwiseConv (k x k, stride)                                     │
          ▼                                                               │
        PointwiseConv (1 x 1)                                             │
          ▼                                                               │
        ReLU                                                              │
          ▼                                                               │
        BatchNorm2d                                                       │
          ▼                                                               │
        DepthwiseConv (k x k, stride=1)                                   │
          ▼                                                               │
        PointwiseConv (1 x 1)                                             │
          │                                                               │
          ▼                                                               ▼
        (Add) ◄───────────────────────────────────────────────────────────┘
          ▼
        Output (Unimpeded identity gradient highway per He et al., 2016)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        mid_channels = max(in_channels, out_channels)

        # First pre-activation stage: ReLU -> BN -> Conv(3x3 depthwise) -> Conv(1x1 pointwise)
        self.relu1 = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dw1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pw1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

        # Second pre-activation stage: ReLU -> BN -> Conv(3x3 depthwise) -> Conv(1x1 pointwise)
        self.relu2 = nn.ReLU(inplace=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.dw2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=mid_channels,
            bias=False,
        )
        self.pw2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

        # Shortcut projection: Identity when channels match and stride is 1
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        # Stage 1: ReLU -> BN -> DWSConv
        out = self.relu1(x)
        out = self.bn1(out)
        out = self.dw1(out)
        out = self.pw1(out)

        # Stage 2: ReLU -> BN -> DWSConv
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.dw2(out)
        out = self.pw2(out)

        # Residual addition: pure additive shortcut (unimpeded identity highway)
        return out + residual


# Alias for backward compatibility
LightResidualBlock = PreActivationResidualBlock


# ─────────────────────────────────────────────────────────────────────────────
# Building Block 3: Multi-Dilated Lightweight Block (Bottleneck)
# ─────────────────────────────────────────────────────────────────────────────

class MultiDilatedLightResidualBlock(nn.Module):
    """
    Multi-Dilated Lightweight Residual Block (Hourglass Bottleneck).

    Engineering Decision (M0 §12.8):
      Parallel DWSConv branches with dilation rates [1, 2, 4].
      Captures multi-scale contextual features with large effective receptive field.
      NOT from the source smart-farming paper (Krishnan et al., 2024).

    Structure:
      Input (B, C, H, W)
        │
        ├─ [d=1 branch: DWSConv] ─┐
        ├─ [d=2 branch: DWSConv] ─┼─ Concatenation -> (B, 3C, H, W)
        └─ [d=4 branch: DWSConv] ─┘
                                  │
                                1x1 Conv + BN + ReLU -> (B, C, H, W)
                                  │
                                (Add) ◄── Input Residual
                                  │
                                Output (B, C, H, W)
    """
    def __init__(self, channels: int, dilation_rates: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.channels = channels
        self.dilation_rates = list(dilation_rates)

        self.branches = nn.ModuleList([
            DepthwiseSeparableConv(channels, channels, kernel_size=3, dilation=d)
            for d in self.dilation_rates
        ])

        n_branches = len(self.dilation_rates)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * n_branches, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.branches]
        cat = torch.cat(branch_outs, dim=1)
        fused = self.fusion(cat)
        return fused + x


# Alias for compatibility
MultiDilatedBlock = MultiDilatedLightResidualBlock


# ─────────────────────────────────────────────────────────────────────────────
# Core Module 4: Single Hourglass Module
# ─────────────────────────────────────────────────────────────────────────────

class Hourglass(nn.Module):
    """
    Single Hourglass Module (Encoder -> MultiDilated Bottleneck -> Decoder).

    Structure per M0 §12.5:
      Encoder:
        L levels (L=3 dev, L=4 final).
        Each level: PreActivationResidualBlock -> MaxPool(2x2).
        Features are preserved at each level for encoder-decoder skip connections.
      Bottleneck:
        MultiDilatedLightResidualBlock at lowest encoder resolution.
      Decoder:
        L levels matching encoder (in reverse order).
        Each level:
          1. Bilinear upsample (x2)
          2. 1x1 Conv projection
          3. Intra-hourglass skip connection: additive feature fusion
          4. PreActivationResidualBlock

    Encoder skip connections (intra-Hourglass) are distinct from
    inter-stack skip connections between successive Hourglass modules.
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        encoder_levels: int,
        dilation_rates: Tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.encoder_levels = encoder_levels
        self.dilation_rates = list(dilation_rates)

        # Encoder channel progression: base, base*2, base*4, ...
        enc_channels = [base_channels * (2 ** i) for i in range(encoder_levels)]
        self.enc_channels = enc_channels

        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        prev_ch = in_channels
        for ch in enc_channels:
            self.encoder_blocks.append(PreActivationResidualBlock(prev_ch, ch))
            self.encoder_pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_ch = ch

        # Multi-dilated bottleneck at lowest resolution
        self.bottleneck = MultiDilatedLightResidualBlock(
            enc_channels[-1],
            dilation_rates=tuple(dilation_rates),
        )

        # Decoder levels (in reverse order)
        dec_channels = list(reversed(enc_channels))
        self.decoder_upsample_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(encoder_levels):
            in_ch = dec_channels[i]
            skip_ch = dec_channels[i + 1] if (i + 1) < len(dec_channels) else dec_channels[-1]
            out_ch = skip_ch

            self.decoder_upsample_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            self.decoder_blocks.append(PreActivationResidualBlock(out_ch, out_ch))

        self.out_channels = dec_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder pass
        encoder_skips = []
        feat = x
        for enc_block, pool in zip(self.encoder_blocks, self.encoder_pools):
            feat = enc_block(feat)
            encoder_skips.append(feat)
            feat = pool(feat)

        # Bottleneck pass
        feat = self.bottleneck(feat)

        # Decoder pass with intra-hourglass skip connections
        rev_skips = list(reversed(encoder_skips))
        for i, (up_conv, dec_block) in enumerate(
            zip(self.decoder_upsample_convs, self.decoder_blocks)
        ):
            feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
            feat = up_conv(feat)
            skip = rev_skips[i + 1] if (i + 1) < len(rev_skips) else rev_skips[-1]
            if feat.shape[-2:] != skip.shape[-2:]:
                feat = F.interpolate(feat, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            feat = feat + skip
            feat = dec_block(feat)

        return feat


# Alias for backward compatibility
HourglassModule = Hourglass


# ─────────────────────────────────────────────────────────────────────────────
# Core Module 5: Stacked Hourglass with Inter-Stack Skip Connections
# ─────────────────────────────────────────────────────────────────────────────

class StackedHourglass(nn.Module):
    """
    Stacked Hourglass Module.

    Stacks multiple Hourglass modules sequentially with explicit inter-stack
    skip connections between successive stacks (Engineering Decision per M0 §12.4).

    Distinction:
      - Intra-hourglass skips: between encoder and decoder within a single hourglass.
      - Inter-stack skips: between the output of stack k and the input of stack k+1.
    """
    def __init__(
        self,
        base_channels: int,
        num_stacks: int = 2,
        encoder_levels: int = 3,
        dilation_rates: Tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()
        self.base_channels = base_channels
        self.num_stacks = num_stacks
        self.encoder_levels = encoder_levels
        self.dilation_rates = list(dilation_rates)

        self.stacks = nn.ModuleList([
            Hourglass(
                in_channels=base_channels,
                base_channels=base_channels,
                encoder_levels=encoder_levels,
                dilation_rates=dilation_rates,
            )
            for _ in range(num_stacks)
        ])

        # Inter-stack skip projection modules (between stack k and stack k+1)
        self.inter_stack_skips = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(base_channels),
            )
            for _ in range(num_stacks - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for i, stack in enumerate(self.stacks):
            out = stack(feat)
            if i < len(self.inter_stack_skips):
                skip = self.inter_stack_skips[i](out)
                feat = out + skip
            else:
                feat = out
        return feat


# ─────────────────────────────────────────────────────────────────────────────
# Complete Architecture: Adapted LS-HGNet Classification Model
# ─────────────────────────────────────────────────────────────────────────────

class LS_HGNet(nn.Module):
    """
    Adapted Lightweight Stacked Hourglass Network (LS-HGNet) for Crop Disease Classification.

    ADAPTATION STATEMENT:
      This implementation adapts the LS-HGNet conceptual architecture from UAV-based
      maize tassel detection/counting (Krishnan et al., 2024, IJERR) to 10-class tomato
      leaf disease classification on the PlantVillage dataset.

    Topological Structure:
      Input Image [B, 3, H, W]
        │
        ▼ Stem: Conv(3x3, stride=2) -> BN -> ReLU -> MaxPool(2x2)  [Downsamples 4x]
      [B, base_channels, H/4, W/4]
        │
        ▼ StackedHourglass (num_stacks Hourglass modules with inter-stack skips)
      [B, base_channels, H/4, W/4]
        │
        ▼ Global Average Pooling -> [B, base_channels]
        ▼ Dropout(p)
        ▼ Linear(base_channels, 10) -> Raw Logits [B, 10] (NO SOFTMAX INSIDE MODEL)

    No softmax inside the model: PyTorch CrossEntropyLoss applies log_softmax
    internally. Temperature scaling and Platt calibration operate directly on logits.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 32,
        num_stacks: int = 2,
        encoder_levels: int = 3,
        dilation_rates: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_stacks = num_stacks
        self.encoder_levels = encoder_levels
        self.dilation_rates = list(dilation_rates)
        self.dropout_rate = dropout

        # ── Stem Block ──
        # Reduces spatial dimensions 4x (e.g., 128 -> 32; 224 -> 56)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Stacked Hourglass Core ──
        self.stacked_hourglass = StackedHourglass(
            base_channels=base_channels,
            num_stacks=num_stacks,
            encoder_levels=encoder_levels,
            dilation_rates=dilation_rates,
        )

        # Expose sub-modules for inspectability
        self.stacks = self.stacked_hourglass.stacks
        self.inter_stack_skips = self.stacked_hourglass.inter_stack_skips

        # ── Classification Head ──
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(base_channels, num_classes)

        # Deterministic Kaiming Normal initialization per project protocol
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming normal for conv/linear, constant for BN (per M3 convention)."""
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
            x: Input tensor [B, 3, H, W]
        Returns:
            Raw logits [B, num_classes] (no softmax).
        """
        feat = self.stem(x)
        feat = self.stacked_hourglass(feat)
        feat = self.gap(feat)
        feat = feat.flatten(1)
        feat = self.dropout(feat)
        logits = self.classifier(feat)
        return logits

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Returns total or trainable parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_model_summary(self) -> Dict[str, Any]:
        """Returns structured metadata about the model."""
        total_params = self.count_parameters(trainable_only=False)
        trainable_params = self.count_parameters(trainable_only=True)
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_mb = (param_bytes + buffer_bytes) / (1024 * 1024)
        return {
            "architecture": "LS-HGNet (AgriAI adaptation)",
            "adaptation_note": (
                "Adapted from UAV maize tassel detection (Krishnan et al., 2024) "
                "to 10-class tomato leaf disease classification on PlantVillage."
            ),
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "num_stacks": self.num_stacks,
            "encoder_levels": self.encoder_levels,
            "dilation_rates": self.dilation_rates,
            "dropout": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_model_size_mb": round(model_size_mb, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory Helpers & Utilities
# ─────────────────────────────────────────────────────────────────────────────

def build_ls_hgnet_from_config(config: dict, num_classes: int = 10) -> LS_HGNet:
    """
    Constructs LS_HGNet from a loaded YAML configuration dictionary.
    Reads from config['model'] section.
    """
    model_cfg = config.get("model", {})
    return LS_HGNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=model_cfg.get("base_channels", 32),
        num_stacks=model_cfg.get("num_stacks", 2),
        encoder_levels=model_cfg.get("encoder_levels", 3),
        dilation_rates=tuple(model_cfg.get("dilation_rates", [1, 2, 4])),
        dropout=model_cfg.get("dropout", 0.3),
    )


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Utility function to count parameters across models."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_checkpoint_size_mb(checkpoint_path: str) -> float:
    """Returns actual size of checkpoint file in megabytes."""
    if not os.path.isfile(checkpoint_path):
        return 0.0
    return os.path.getsize(checkpoint_path) / (1024 * 1024)
