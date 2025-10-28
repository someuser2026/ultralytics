"""
Target assignment for Cascade R-CNN
Path: ultralytics/models/cascade_rcnn/targets.py
"""

from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) boxes in xyxy format
        boxes2: (M, 4) boxes in xyxy format
    
    Returns:
        iou: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area1[:, None] + area2 - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    return iou


def build_rpn_targets(
    anchors: Tensor,
    gt_boxes: Tensor,
    pos_thresh: float = 0.7,
    neg_thresh: float = 0.3,
    num_samples: int = 256,
    pos_fraction: float = 0.5
) -> Tuple[Tensor, Tensor]:
    """
    Build RPN targets with balanced sampling.
    
    Args:
        anchors: (N, 4) anchor boxes in xyxy format
        gt_boxes: (M, 4) ground truth boxes in xyxy format
        pos_thresh: IoU threshold for positive anchors
        neg_thresh: IoU threshold for negative anchors
        num_samples: total number of anchors to sample
        pos_fraction: fraction of positive samples
    
    Returns:
        labels: (N,) with values {-1: ignore, 0: negative, 1: positive}
        box_targets: (N, 4) regression targets for positive anchors
    """
    num_anchors = anchors.shape[0]
    device = anchors.device
    
    # Initialize labels as "ignore"
    labels = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    
    if gt_boxes.numel() == 0:
        # No ground truth: all negatives
        labels[:] = 0
        box_targets = torch.zeros((num_anchors, 4), device=device)
        return labels, box_targets
    
    # Compute IoU
    iou = box_iou(anchors, gt_boxes)  # (N, M)
    max_iou, gt_idx = iou.max(dim=1)
    
    # Assign labels based on IoU
    labels[max_iou >= pos_thresh] = 1  # Positive
    labels[max_iou < neg_thresh] = 0   # Negative
    
    # For each GT, assign at least one anchor (highest IoU)
    if gt_boxes.shape[0] > 0:
        gt_max_iou, gt_max_idx = iou.max(dim=0)
        labels[gt_max_idx] = 1
    
    # Subsample positive anchors
    pos_inds = torch.where(labels == 1)[0]
    num_pos = int(num_samples * pos_fraction)
    
    if pos_inds.numel() > num_pos:
        # Randomly disable excess positives
        perm = torch.randperm(pos_inds.numel(), device=device)
        disable_inds = pos_inds[perm[num_pos:]]
        labels[disable_inds] = -1
    
    # Subsample negative anchors
    neg_inds = torch.where(labels == 0)[0]
    num_neg = num_samples - (labels == 1).sum().item()
    
    if neg_inds.numel() > num_neg:
        # Randomly disable excess negatives
        perm = torch.randperm(neg_inds.numel(), device=device)
        disable_inds = neg_inds[perm[num_neg:]]
        labels[disable_inds] = -1
    
    # Box targets for positive anchors
    box_targets = torch.zeros((num_anchors, 4), device=device)
    pos_mask = labels == 1
    
    if pos_mask.sum() > 0:
        assigned_gt = gt_idx[pos_mask]
        box_targets[pos_mask] = gt_boxes[assigned_gt]
    
    return labels, box_targets


def build_stage_targets(
    rois: Tensor,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    iou_thresh: float,
    num_samples: int = 512,
    pos_fraction: float = 0.25
) -> Dict[str, Tensor]:
    """
    Build targets for one cascade stage with sampling.
    
    Args:
        rois: (N, 4) region proposals in xyxy format
        gt_boxes: (M, 4) ground truth boxes in xyxy format
        gt_labels: (M,) ground truth labels (1..num_classes)
        iou_thresh: IoU threshold for positive assignment
        num_samples: number of RoIs to sample
        pos_fraction: fraction of positive samples
    
    Returns:
        Dict with:
            - labels: (N,) class labels (0=background, 1..C=classes)
            - boxes: (N, 4) matched GT boxes
            - sampled_inds: (K,) indices of sampled RoIs
    """
    device = rois.device
    num_rois = rois.shape[0]
    
    if gt_boxes.numel() == 0:
        # No ground truth: all background
        return {
            'labels': torch.zeros(num_rois, dtype=torch.long, device=device),
            'boxes': torch.zeros((num_rois, 4), device=device),
            'sampled_inds': torch.arange(min(num_samples, num_rois), device=device)
        }
    
    # Compute IoU
    iou = box_iou(rois, gt_boxes)  # (N, M)
    max_iou, gt_idx = iou.max(dim=1)
    
    # Assign labels
    labels = gt_labels[gt_idx].clone()  # Start with matched GT labels
    labels[max_iou < iou_thresh] = 0    # Below threshold = background
    
    # Matched GT boxes
    matched_boxes = gt_boxes[gt_idx]
    
    # Sample RoIs
    pos_inds = torch.where(labels > 0)[0]
    neg_inds = torch.where(labels == 0)[0]
    
    num_pos = int(num_samples * pos_fraction)
    num_pos = min(num_pos, pos_inds.numel())
    num_neg = num_samples - num_pos
    num_neg = min(num_neg, neg_inds.numel())
    
    # Random sampling
    if pos_inds.numel() > num_pos:
        perm = torch.randperm(pos_inds.numel(), device=device)
        pos_inds = pos_inds[perm[:num_pos]]
    
    if neg_inds.numel() > num_neg:
        perm = torch.randperm(neg_inds.numel(), device=device)
        neg_inds = neg_inds[perm[:num_neg]]
    
    sampled_inds = torch.cat([pos_inds, neg_inds], dim=0)
    
    return {
        'labels': labels,
        'boxes': matched_boxes,
        'sampled_inds': sampled_inds
    }


def build_mask_targets(
    gt_masks: Tensor,
    gt_boxes: Tensor,
    pos_boxes: Tensor,
    mask_size: int = 28
) -> Tensor:
    """
    Build mask targets by cropping and resizing GT masks.
    
    Args:
        gt_masks: (M, H, W) binary ground truth masks
        gt_boxes: (M, 4) GT boxes in xyxy format
        pos_boxes: (N, 4) positive RoI boxes in xyxy format
        mask_size: output mask resolution
    
    Returns:
        mask_targets: (N, mask_size, mask_size) binary masks
    """
    device = pos_boxes.device
    num_pos = pos_boxes.shape[0]
    
    if num_pos == 0 or gt_masks.numel() == 0:
        return torch.zeros((0, mask_size, mask_size), device=device)
    
    # Match each positive RoI to a GT box
    iou = box_iou(pos_boxes, gt_boxes)
    _, matched_gt_idx = iou.max(dim=1)
    
    # Extract matched masks
    matched_masks = gt_masks[matched_gt_idx]  # (N, H, W)
    matched_gt_boxes = gt_boxes[matched_gt_idx]  # (N, 4)
    
    # Crop masks to matched boxes and resize to mask_size
    mask_targets = []
    
    for i in range(num_pos):
        mask = matched_masks[i]
        box = matched_gt_boxes[i]
        
        # Convert box to integer coordinates
        x1, y1, x2, y2 = box.int().tolist()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[1], x2)
        y2 = min(mask.shape[0], y2)
        
        # Crop mask
        if x2 > x1 and y2 > y1:
            cropped = mask[y1:y2, x1:x2].float()
            # Resize to mask_size
            resized = F.interpolate(
                cropped.unsqueeze(0).unsqueeze(0),
                size=(mask_size, mask_size),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        else:
            resized = torch.zeros((mask_size, mask_size), device=device)
        
        mask_targets.append(resized)
    
    return torch.stack(mask_targets, dim=0)


def subsample_rois(
    rois: Tensor,
    labels: Tensor,
    num_samples: int = 512,
    pos_fraction: float = 0.25
) -> Tuple[Tensor, Tensor]:
    """
    Subsample RoIs for training.
    
    Args:
        rois: (N, 5) [batch_idx, x1, y1, x2, y2]
        labels: (N,) class labels (0=background)
        num_samples: number of samples to keep
        pos_fraction: target fraction of positive samples
    
    Returns:
        sampled_rois: (K, 5) subsampled RoIs
        sampled_labels: (K,) corresponding labels
    """
    pos_inds = torch.where(labels > 0)[0]
    neg_inds = torch.where(labels == 0)[0]
    
    num_pos = int(num_samples * pos_fraction)
    num_pos = min(num_pos, pos_inds.numel())
    num_neg = num_samples - num_pos
    num_neg = min(num_neg, neg_inds.numel())
    
    # Sample
    if pos_inds.numel() > num_pos:
        perm = torch.randperm(pos_inds.numel(), device=rois.device)
        pos_inds = pos_inds[perm[:num_pos]]
    
    if neg_inds.numel() > num_neg:
        perm = torch.randperm(neg_inds.numel(), device=rois.device)
        neg_inds = neg_inds[perm[:num_neg]]
    
    sampled_inds = torch.cat([pos_inds, neg_inds], dim=0)
    
    return rois[sampled_inds], labels[sampled_inds]