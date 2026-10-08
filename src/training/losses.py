"""
AgriAI Training Losses Module
Implements standard Cross-Entropy Loss for the Baseline CNN control.

As mandated by M0/M1 research specifications:
- Baseline v1 MUST use unweighted CrossEntropyLoss().
- Class imbalance is to be measured and observed under standard loss.
- Any future class-weighted loss must be treated as a distinct ablation/experiment.
"""

from typing import Optional
import torch
import torch.nn as nn


def get_loss_function(
    loss_type: str = "cross_entropy",
    weights: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Returns the loss criterion.
    
    Args:
        loss_type: Loss identifier ("cross_entropy" default).
        weights: Optional tensor of class weights. None for Baseline v1.
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weights)
    else:
        raise ValueError(f"Unsupported loss function: '{loss_type}'. Expected 'cross_entropy'.")
