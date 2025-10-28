"""
Loss functions for Cascade R-CNN
Path: ultralytics/utils/loss_cascade.py
"""

from __future__ import annotations
from typing import Dict
import torch
from torch import Tensor, nn


class RPNLoss(nn.Module):
    """RPN objectness and bbox regression loss."""
    
    def __init__(
        self,
        obj_weight: float = 1.0,
        box_weight: float = 1.0,
        beta: float = 1.0
    ):
        super().__init__()
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.beta = beta

    def forward(
        self,
        obj_logits: Tensor,
        obj_targets: Tensor,
        box_preds: Tensor,
        box_targets: Tensor
    ) -> Tensor:
        """
        Compute RPN loss.
        
        Args:
            obj_logits: (N,) objectness logits
            obj_targets: (N,) objectness targets {0, 1}
            box_preds: (M, 4) bbox predictions for positive anchors
            box_targets: (M, 4) bbox targets for positive anchors
        
        Returns:
            loss: scalar loss
        """
        # Objectness loss (binary cross entropy)
        obj_loss = nn.functional.binary_cross_entropy_with_logits(
            obj_logits,
            obj_targets.float(),
            reduction='mean'
        )
        
        # Box regression loss (smooth L1) - only on positives
        if box_preds.numel() > 0 and box_targets.numel() > 0:
            box_loss = nn.functional.smooth_l1_loss(
                box_preds,
                box_targets,
                beta=self.beta,
                reduction='mean'
            )
        else:
            box_loss = obj_logits.sum() * 0.0
        
        total_loss = self.obj_weight * obj_loss + self.box_weight * box_loss
        
        return total_loss


class CascadeRCNNLoss(nn.Module):
    """Multi-stage Cascade R-CNN loss."""
    
    def __init__(
        self,
        num_stages: int = 3,
        stage_weights: Tuple[float, ...] = (1.0, 1.0, 1.0),
        cls_weight: float = 1.0,
        reg_weight: float = 1.0
    ):
        super().__init__()
        self.num_stages = num_stages
        self.stage_weights = stage_weights[:num_stages]
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(self, stage_losses: Dict[str, Tensor]) -> Tensor:
        """
        Compute total cascade loss.
        
        Args:
            stage_losses: Dict with keys 's{i}_cls' and 's{i}_reg'
        
        Returns:
            total_loss: scalar loss
        """
        total = 0.0
        
        for i, weight in enumerate(self.stage_weights, start=1):
            cls_key = f's{i}_cls'
            reg_key = f's{i}_reg'
            
            if cls_key in stage_losses and reg_key in stage_losses:
                stage_loss = (
                    self.cls_weight * stage_losses[cls_key] +
                    self.reg_weight * stage_losses[reg_key]
                )
                total = total + weight * stage_loss
        
        return total


class MaskRCNNLoss(nn.Module):
    """Mask prediction loss."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        mask_logits: Tensor,
        mask_targets: Tensor
    ) -> Tensor:
        """
        Compute mask loss.
        
        Args:
            mask_logits: (N, 1, H, W) predicted mask logits
            mask_targets: (N, H, W) binary mask targets
        
        Returns:
            loss: scalar loss
        """
        if mask_logits.numel() == 0:
            return mask_logits.sum() * 0.0
        
        # Binary cross entropy
        mask_logits = mask_logits.squeeze(1)  # (N, H, W)
        
        loss = nn.functional.binary_cross_entropy_with_logits(
            mask_logits,
            mask_targets.float(),
            reduction='mean'
        )
        
        return self.weight * loss


class CompositeCascadeLoss(nn.Module):
    """Combined loss for Cascade R-CNN with masks."""
    
    def __init__(
        self,
        num_stages: int = 3,
        stage_weights: Tuple[float, ...] = (1.0, 1.0, 1.0),
        rpn_obj_weight: float = 1.0,
        rpn_box_weight: float = 1.0,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        mask_weight: float = 1.0
    ):
        super().__init__()
        
        self.rpn_loss = RPNLoss(rpn_obj_weight, rpn_box_weight)
        self.cascade_loss = CascadeRCNNLoss(
            num_stages, stage_weights, cls_weight, reg_weight
        )
        self.mask_loss = MaskRCNNLoss(mask_weight)

    def forward(
        self,
        rpn_obj_logits: Tensor,
        rpn_obj_targets: Tensor,
        rpn_box_preds: Tensor,
        rpn_box_targets: Tensor,
        stage_losses: Dict[str, Tensor],
        mask_logits: Tensor = None,
        mask_targets: Tensor = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute all losses.
        
        Returns:
            total_loss: scalar total loss
            loss_dict: Dict of individual losses for logging
        """
        # RPN loss
        rpn_loss_val = self.rpn_loss(
            rpn_obj_logits,
            rpn_obj_targets,
            rpn_box_preds,
            rpn_box_targets
        )
        
        # Cascade loss
        cascade_loss_val = self.cascade_loss(stage_losses)
        
        # Total
        total = rpn_loss_val + cascade_loss_val
        
        loss_dict = {
            'rpn': rpn_loss_val,
            'cascade': cascade_loss_val,
            **stage_losses
        }
        
        # Mask loss (if applicable)
        if mask_logits is not None and mask_targets is not None:
            mask_loss_val = self.mask_loss(mask_logits, mask_targets)
            total = total + mask_loss_val
            loss_dict['mask'] = mask_loss_val
        
        return total, loss_dict