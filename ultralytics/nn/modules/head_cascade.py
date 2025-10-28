"""
Cascade R-CNN Detection Head
Path: ultralytics/nn/modules/head_cascade.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict
import torch
from torch import Tensor, nn
from .roi import assign_fpn_levels, roi_align_multilevel, BoxCoder, batched_nms


class TwoFCHead(nn.Module):
    """Two fully-connected layer head for RCNN."""
    
    def __init__(self, in_channels: int, hidden_dim: int = 1024):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Initialize weights
        for layer in [self.fc1, self.fc2]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: (N, C, H, W) RoI-aligned features
        
        Returns:
            features: (N, hidden_dim) feature vectors
        """
        x = self.avgpool(x).flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class BBoxHead(nn.Module):
    """Single stage bbox classification and regression head."""
    
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 1024):
        super().__init__()
        self.tower = TwoFCHead(in_channels, hidden_dim)
        
        # Classification: num_classes + 1 (background)
        self.cls_score = nn.Linear(hidden_dim, num_classes + 1)
        
        # Regression: class-agnostic (4 values)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        # Initialize
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            x: (N, C, H, W) RoI-aligned features
        
        Returns:
            cls_logits: (N, num_classes + 1) classification logits
            bbox_deltas: (N, 4) bbox regression deltas
        """
        feat = self.tower(x)
        cls_logits = self.cls_score(feat)
        bbox_deltas = self.bbox_pred(feat)
        return cls_logits, bbox_deltas


class CascadeRCNNHead(nn.Module):
    """
    Cascade R-CNN detection head with multiple refinement stages.
    
    Label convention: 0 = background, 1..num_classes = foreground classes
    """
    
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
        hidden_dim: int = 1024
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = num_stages
        
        # Stage heads
        self.stage_heads = nn.ModuleList([
            BBoxHead(in_channels, num_classes, hidden_dim)
            for _ in range(num_stages)
        ])
        
        # Box coders with different stds per stage
        self.box_coders = [
            BoxCoder(stds=stage_stds[i])
            for i in range(num_stages)
        ]

    def forward_train(
        self,
        feats: List[Tensor],
        rois: Tensor,
        roi_levels: Tensor,
        targets: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Training forward pass through all cascade stages.
        
        Args:
            feats: List of FPN features [P2, P3, P4, P5]
            rois: (N, 5) [batch_idx, x1, y1, x2, y2]
            roi_levels: (N,) FPN level assignment
            targets: Dict with keys 's{i}_labels' and 's{i}_boxes' for each stage
        
        Returns:
            losses: Dict with cls and reg losses per stage
        """
        losses = {}
        
        # Initial RoI pooling
        x = roi_align_multilevel(feats, rois, roi_levels, output_size=7)
        boxes = rois[:, 1:5].clone()
        
        for stage_idx in range(self.num_stages):
            stage_num = stage_idx + 1
            
            # Forward through stage head
            cls_logits, bbox_deltas = self.stage_heads[stage_idx](x)
            
            # Classification loss
            labels = targets[f's{stage_num}_labels']
            cls_loss = nn.functional.cross_entropy(cls_logits, labels)
            losses[f's{stage_num}_cls'] = cls_loss
            
            # Regression loss (only on positives)
            pos_mask = labels > 0
            
            if pos_mask.sum() > 0:
                pos_boxes = boxes[pos_mask]
                pos_deltas = bbox_deltas[pos_mask]
                pos_targets = targets[f's{stage_num}_boxes'][pos_mask]
                
                # Decode predictions
                pred_boxes = self.box_coders[stage_idx].decode(pos_boxes, pos_deltas)
                
                # Smooth L1 loss
                reg_loss = nn.functional.smooth_l1_loss(
                    pred_boxes,
                    pos_targets,
                    beta=1.0,
                    reduction='mean'
                )
                losses[f's{stage_num}_reg'] = reg_loss
            else:
                # No positive samples
                losses[f's{stage_num}_reg'] = cls_logits.sum() * 0.0
            
            # Refine boxes for next stage
            if stage_idx < self.num_stages - 1:
                with torch.no_grad():
                    boxes = self.box_coders[stage_idx].decode(boxes, bbox_deltas)
                    
                    # Re-assign FPN levels
                    roi_levels = assign_fpn_levels(boxes)
                    
                    # Create new rois tensor
                    rois_new = torch.cat([rois[:, :1], boxes], dim=1)
                    
                    # Re-pool features
                    x = roi_align_multilevel(feats, rois_new, roi_levels, output_size=7)
        
        return losses

    @torch.no_grad()
    def infer(
        self,
        feats: List[Tensor],
        proposals: Tensor,
        score_thresh: float = 0.05,
        nms_iou: float = 0.5,
        max_dets: int = 300
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Inference through all cascade stages.
        
        Args:
            feats: List of FPN features [P2, P3, P4, P5]
            proposals: (N, 5) [batch_idx, x1, y1, x2, y2]
            score_thresh: score threshold for filtering
            nms_iou: IoU threshold for NMS
            max_dets: maximum detections to return
        
        Returns:
            boxes: (M, 4) final boxes
            scores: (M,) confidence scores
            labels: (M,) class labels (0-indexed, no background)
        """
        if proposals.numel() == 0:
            device = feats[0].device
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), dtype=torch.long, device=device)
            )
        
        boxes = proposals[:, 1:5].clone()
        levels = assign_fpn_levels(boxes)
        x = roi_align_multilevel(feats, proposals, levels, output_size=7)
        
        # Cascade through all stages
        for stage_idx, head in enumerate(self.stage_heads):
            cls_logits, bbox_deltas = head(x)
            
            # Decode boxes
            boxes = self.box_coders[stage_idx].decode(boxes, bbox_deltas)
            
            # Get scores (softmax over classes)
            scores = cls_logits.softmax(dim=1)  # (N, num_classes + 1)
            
            # Re-pool for next stage
            if stage_idx < self.num_stages - 1:
                levels = assign_fpn_levels(boxes)
                rois_new = torch.cat([proposals[:, :1], boxes], dim=1)
                x = roi_align_multilevel(feats, rois_new, levels, output_size=7)
        
        # Post-processing: remove background class
        scores = scores[:, 1:]  # (N, num_classes)
        
        # Expand boxes and scores for all classes
        N, C = scores.shape
        boxes_expanded = boxes[:, None, :].expand(N, C, 4).reshape(-1, 4)
        scores_flat = scores.reshape(-1)
        labels_flat = torch.arange(C, device=boxes.device).view(1, -1).expand(N, C).reshape(-1)
        
        # Filter by score threshold
        keep = scores_flat > score_thresh
        boxes_kept = boxes_expanded[keep]
        scores_kept = scores_flat[keep]
        labels_kept = labels_flat[keep]
        
        if boxes_kept.numel() == 0:
            device = boxes.device
            return (
                torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), dtype=torch.long, device=device)
            )
        
        # Apply class-aware NMS
        keep_nms = batched_nms(boxes_kept, scores_kept, labels_kept, nms_iou)
        keep_nms = keep_nms[:max_dets]
        
        return boxes_kept[keep_nms], scores_kept[keep_nms], labels_kept[keep_nms]