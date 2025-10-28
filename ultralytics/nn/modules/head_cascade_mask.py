"""
Cascade Mask R-CNN Head
Path: ultralytics/nn/modules/head_cascade_mask.py
"""

from __future__ import annotations
from typing import List, Tuple
import torch
from torch import Tensor, nn
from .roi import assign_fpn_levels, roi_align_multilevel
from .head_cascade import CascadeRCNNHead


class MaskHead(nn.Module):
    """Mask prediction head with FCN architecture."""
    
    def __init__(
        self,
        in_channels: int,
        dim: int = 256,
        num_convs: int = 4,
        out_res: int = 28
    ):
        super().__init__()
        self.out_res = out_res
        
        # Convolutional layers
        convs = []
        for i in range(num_convs):
            convs.append(nn.Conv2d(
                in_channels if i == 0 else dim,
                dim,
                3,
                padding=1
            ))
        self.convs = nn.ModuleList(convs)
        
        # Deconvolution for upsampling
        self.deconv = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        
        # Final prediction layer (class-agnostic)
        self.mask_logits = nn.Conv2d(dim, 1, 1)
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: (N, C, H, W) RoI-aligned features
        
        Returns:
            mask_logits: (N, 1, out_res, out_res) mask predictions
        """
        for conv in self.convs:
            x = torch.relu(conv(x))
        
        x = torch.relu(self.deconv(x))
        mask_logits = self.mask_logits(x)
        
        return mask_logits


class CascadeRCNNMaskHead(CascadeRCNNHead):
    """Cascade R-CNN head with mask prediction branch."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_stages: int = 3,
        stage_stds: Tuple[Tuple[float, ...], ...] = (
            (0.1, 0.1, 0.2, 0.2),
            (0.05, 0.05, 0.1, 0.1),
            (0.033, 0.033, 0.067, 0.067)
        ),
        hidden_dim: int = 1024,
        mask_dim: int = 256,
        mask_resolution: int = 28
    ):
        super().__init__(in_channels, num_classes, num_stages, stage_stds, hidden_dim)
        
        # Add mask head
        self.mask_head = MaskHead(
            in_channels,
            dim=mask_dim,
            out_res=mask_resolution
        )
        self.mask_resolution = mask_resolution

    def forward_mask_train(
        self,
        feats: List[Tensor],
        pos_rois: Tensor,
        pos_labels: Tensor
    ) -> Tensor:
        """
        Forward through mask head for training.
        
        Args:
            feats: List of FPN features [P2, P3, P4, P5]
            pos_rois: (N, 5) positive RoIs [batch_idx, x1, y1, x2, y2]
            pos_labels: (N,) class labels for positive RoIs (1..num_classes)
        
        Returns:
            mask_logits: (N, 1, mask_resolution, mask_resolution)
        """
        if pos_rois.numel() == 0:
            device = feats[0].device
            return torch.zeros(
                (0, 1, self.mask_resolution, self.mask_resolution),
                device=device
            )
        
        # Assign FPN levels
        levels = assign_fpn_levels(pos_rois[:, 1:5])
        
        # RoI align at higher resolution (14x14 input for 28x28 output)
        x = roi_align_multilevel(feats, pos_rois, levels, output_size=14)
        
        # Forward through mask head
        mask_logits = self.mask_head(x)
        
        return mask_logits

    @torch.no_grad()
    def infer_with_masks(
        self,
        feats: List[Tensor],
        proposals: Tensor,
        score_thresh: float = 0.05,
        nms_iou: float = 0.5,
        max_dets: int = 300
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Inference with mask predictions.
        
        Args:
            feats: List of FPN features [P2, P3, P4, P5]
            proposals: (N, 5) [batch_idx, x1, y1, x2, y2]
            score_thresh: score threshold
            nms_iou: IoU threshold for NMS
            max_dets: maximum detections
        
        Returns:
            boxes: (M, 4) final boxes
            scores: (M,) confidence scores
            labels: (M,) class labels
            masks: (M, mask_resolution, mask_resolution) mask predictions
        """
        # Get box predictions first
        boxes, scores, labels = self.infer(
            feats, proposals, score_thresh, nms_iou, max_dets
        )
        
        if boxes.numel() == 0:
            device = feats[0].device
            return (
                boxes,
                scores,
                labels,
                torch.zeros(
                    (0, self.mask_resolution, self.mask_resolution),
                    device=device
                )
            )
        
        # Create RoIs for mask prediction
        batch_idx = proposals[0, 0].expand(boxes.shape[0])
        mask_rois = torch.cat([batch_idx.view(-1, 1), boxes], dim=1)
        
        # Forward through mask head
        levels = assign_fpn_levels(boxes)
        x = roi_align_multilevel(feats, mask_rois, levels, output_size=14)
        mask_logits = self.mask_head(x)
        
        # Sigmoid to get probabilities
        masks = mask_logits.squeeze(1).sigmoid()
        
        return boxes, scores, labels, masks