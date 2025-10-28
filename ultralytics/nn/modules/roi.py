"""
ROI utilities for Cascade R-CNN
Path: ultralytics/nn/modules/roi.py
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Tuple
import torch
from torch import Tensor, nn
from torchvision.ops import roi_align, nms


def assign_fpn_levels(
    boxes: Tensor,
    canonical_scale: int = 224,
    k0: int = 4,
    min_level: int = 2,
    max_level: int = 5
) -> Tensor:
    """
    Map each box to an FPN level using sqrt(area) heuristic.
    
    Args:
        boxes: (N, 4) tensor in xyxy format (pixel coordinates)
        canonical_scale: reference scale for level assignment
        k0: base level offset
        min_level: minimum FPN level
        max_level: maximum FPN level
    
    Returns:
        levels: (N,) tensor with values in [min_level, max_level]
    """
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    area = (wh[:, 0].clamp(min=1e-6) * wh[:, 1].clamp(min=1e-6)).float()
    s = torch.sqrt(area)
    level = torch.floor(k0 + torch.log2(s / float(canonical_scale) + 1e-6))
    return level.clamp(min=min_level, max=max_level).to(torch.long)


@dataclass
class BoxCoder:
    """Encode/decode boxes using deltas relative to anchors."""
    stds: Tuple[float, float, float, float] = (0.1, 0.1, 0.2, 0.2)

    def encode(self, anchors: Tensor, gt: Tensor) -> Tensor:
        """
        Encode ground truth boxes relative to anchors.
        
        Args:
            anchors: (N, 4) reference boxes in xyxy format
            gt: (N, 4) ground truth boxes in xyxy format
        
        Returns:
            deltas: (N, 4) encoded offsets
        """
        wa = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
        ha = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
        xa = anchors[:, 0] + 0.5 * wa
        ya = anchors[:, 1] + 0.5 * ha
        
        wg = (gt[:, 2] - gt[:, 0]).clamp(min=1e-6)
        hg = (gt[:, 3] - gt[:, 1]).clamp(min=1e-6)
        xg = gt[:, 0] + 0.5 * wg
        yg = gt[:, 1] + 0.5 * hg
        
        tx = (xg - xa) / wa
        ty = (yg - ya) / ha
        tw = torch.log(wg / wa)
        th = torch.log(hg / ha)
        
        deltas = torch.stack([tx, ty, tw, th], dim=1)
        std = torch.tensor(self.stds, device=deltas.device, dtype=deltas.dtype)
        return deltas / std

    def decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        """
        Decode deltas to produce predicted boxes.
        
        Args:
            anchors: (N, 4) reference boxes in xyxy format
            deltas: (N, 4) encoded offsets
        
        Returns:
            boxes: (N, 4) predicted boxes in xyxy format
        """
        std = torch.tensor(self.stds, device=deltas.device, dtype=deltas.dtype)
        d = deltas * std
        
        wa = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
        ha = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
        xa = anchors[:, 0] + 0.5 * wa
        ya = anchors[:, 1] + 0.5 * ha
        
        # Clamp for numerical stability
        dx = d[:, 0].clamp(min=-1000, max=1000)
        dy = d[:, 1].clamp(min=-1000, max=1000)
        dw = d[:, 2].clamp(min=-1000, max=1000)
        dh = d[:, 3].clamp(min=-1000, max=1000)
        
        x = dx * wa + xa
        y = dy * ha + ya
        w = wa * torch.exp(dw)
        h = ha * torch.exp(dh)
        
        x1 = x - 0.5 * w
        y1 = y - 0.5 * h
        x2 = x + 0.5 * w
        y2 = y + 0.5 * h
        
        return torch.stack([x1, y1, x2, y2], dim=1)


def roi_align_multilevel(
    feats: List[Tensor],
    rois: Tensor,
    levels: Tensor,
    output_size: int = 7,
    sampling_ratio: int = 2,
    aligned: bool = True
) -> Tensor:
    """
    Apply RoI Align on multi-level features.
    
    Args:
        feats: List of [P2, P3, P4, P5], each (B, C, H, W)
        rois: (M, 5) [batch_idx, x1, y1, x2, y2]
        levels: (M,) values in {2, 3, 4, 5}
        output_size: output spatial size
        sampling_ratio: number of sampling points
        aligned: use aligned RoI pooling
    
    Returns:
        pooled: (M, C, output_size, output_size)
    """
    if rois.numel() == 0:
        C = feats[0].shape[1]
        return rois.new_zeros((0, C, output_size, output_size))
    
    pooled = []
    device = rois.device
    
    for lvl in range(2, 6):  # P2, P3, P4, P5
        if lvl - 2 >= len(feats):
            continue
        
        feat = feats[lvl - 2]
        idx = torch.where(levels == lvl)[0]
        
        if idx.numel() == 0:
            continue
        
        # Scale ROIs to feature map coordinates
        stride = 2 ** lvl
        rois_lvl = rois[idx].clone()
        rois_lvl[:, 1:5] = rois_lvl[:, 1:5] / stride
        
        pooled_lvl = roi_align(
            feat,
            rois_lvl,
            output_size=output_size,
            spatial_scale=1.0,  # already scaled above
            sampling_ratio=sampling_ratio,
            aligned=aligned
        )
        pooled.append(pooled_lvl)
    
    if not pooled:
        C = feats[0].shape[1]
        return rois.new_zeros((0, C, output_size, output_size))
    
    return torch.cat(pooled, dim=0)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    iou_threshold: float
) -> Tensor:
    """
    Batched NMS with class-aware offsets.
    
    Args:
        boxes: (N, 4) boxes in xyxy format
        scores: (N,) confidence scores
        labels: (N,) class labels
        iou_threshold: IoU threshold for NMS
    
    Returns:
        keep: indices of kept boxes
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    max_coord = boxes.max()
    offsets = labels.to(boxes) * (max_coord + 1)
    boxes_for_nms = boxes + offsets.view(-1, 1)
    
    return nms(boxes_for_nms, scores, iou_threshold)


def clip_boxes_to_image(boxes: Tensor, height: int, width: int) -> Tensor:
    """
    Clip boxes to image boundaries.
    
    Args:
        boxes: (N, 4) boxes in xyxy format
        height: image height
        width: image width
    
    Returns:
        clipped_boxes: (N, 4) clipped boxes
    """
    boxes = boxes.clone()
    boxes[:, 0].clamp_(min=0, max=width - 1)
    boxes[:, 1].clamp_(min=0, max=height - 1)
    boxes[:, 2].clamp_(min=0, max=width - 1)
    boxes[:, 3].clamp_(min=0, max=height - 1)
    return boxes


def filter_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    Filter out boxes smaller than min_size.
    
    Args:
        boxes: (N, 4) boxes in xyxy format
        min_size: minimum box dimension
    
    Returns:
        keep: indices of boxes to keep
    """
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return torch.where(keep)[0]