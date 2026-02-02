# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model validation metrics."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER, DataExportMixin, SimpleClass, TryExcept, checks, plt_settings

OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)


def calculate_fitness(metrics_dict: dict, fitness_weights: dict = None) -> float:
    """
    Calculate a weighted fitness score from multiple metrics.
    
    Args:
        metrics_dict (dict): Dictionary containing metric values with keys like:
            - 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
            - 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)'
            - 'metrics/f2(M)', 'metrics/dice(M)', 'metrics/mIoU(M)', 'metrics/boundaryF1(M)'
        fitness_weights (dict, optional): Dictionary of weights for each metric. If None, uses default weights.
        
    Returns:
        (float): Weighted fitness score
        
    Example:
        >>> metrics = {'metrics/mAP50-95(B)': 0.5, 'metrics/f2(M)': 0.7}
        >>> weights = {'mAP50_95': 0.6, 'f2': 0.4}
        >>> fitness = calculate_fitness(metrics, weights)
    """
    if fitness_weights is None:
        # Default weights: focus on mAP50-95 for detection
        fitness_weights = {
            'precision': 0.0, 'recall': 0.0, 'mAP50': 0.0, 'mAP50_95': 1.0,
            'mask_precision': 0.0, 'mask_recall': 0.0, 'mask_mAP50': 0.0, 'mask_mAP50_95': 0.0,
            'f1': 0.0, 'f2': 0.0, 'dice': 0.0, 'miou': 0.0, 'boundary_f1': 0.0
        }
    
    # Mapping from config keys to metric keys in results
    metric_mapping = {
        'precision': 'metrics/precision(B)',
        'recall': 'metrics/recall(B)', 
        'mAP50': 'metrics/mAP50(B)',
        'mAP50_95': 'metrics/mAP50-95(B)',
        'mask_precision': 'metrics/precision(M)',
        'mask_recall': 'metrics/recall(M)',
        'mask_mAP50': 'metrics/mAP50(M)', 
        'mask_mAP50_95': 'metrics/mAP50-95(M)',
        'f1': 'metrics/f1(M)',
        'f2': 'metrics/f2(M)',
        'dice': 'metrics/dice(M)',
        'miou': 'metrics/mIoU(M)',
        'boundary_f1': 'metrics/boundaryF1(M)'
    }
    
    fitness = 0.0
    total_weight = 0.0
    
    for weight_key, weight_value in fitness_weights.items():
        if weight_value > 0:  # Only include metrics with non-zero weights
            metric_key = metric_mapping.get(weight_key)
            if metric_key and metric_key in metrics_dict:
                metric_value = metrics_dict[metric_key]
                if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                    fitness += weight_value * metric_value
                    total_weight += weight_value
    
    # Normalize by total weight if any weights were applied
    if total_weight > 0:
        fitness = fitness / total_weight
    else:
        # Fallback: use mAP50-95 if no weights are configured
        fallback_key = 'metrics/mAP50-95(B)'
        if fallback_key in metrics_dict:
            fitness = metrics_dict[fallback_key]
        else:
            fitness = 0.0
    
    return float(fitness)


def bbox_ioa(box1: np.ndarray, box2: np.ndarray, iou: bool = False, eps: float = 1e-7) -> np.ndarray:
    """
    Calculate the intersection over box2 area given box1 and box2.

    Args:
        box1 (np.ndarray): A numpy array of shape (N, 4) representing N bounding boxes in x1y1x2y2 format.
        box2 (np.ndarray): A numpy array of shape (M, 4) representing M bounding boxes in x1y1x2y2 format.
        iou (bool, optional): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (np.ndarray): A numpy array of shape (N, M) representing the intersection over box2 area.
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate intersection-over-union (IoU) of boxes.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes in (x1, y1, x2, y2) format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.

    References:
        https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def dice_score(mask1: torch.Tensor, mask2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute Dice score between two binary masks.

    Args:
        mask1 (torch.Tensor): Tensor of shape (N, H, W) or (H, W).
        mask2 (torch.Tensor): Tensor of shape (N, H, W) or (H, W).
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        (torch.Tensor): Dice score per-pair (N,).
    """
    if mask1.dim() == 2:
        mask1 = mask1.unsqueeze(0)
    if mask2.dim() == 2:
        mask2 = mask2.unsqueeze(0)
    inter = (mask1 & mask2).sum(dim=(1, 2)).float()
    denom = mask1.sum(dim=(1, 2)).float() + mask2.sum(dim=(1, 2)).float()
    return (2.0 * inter) / (denom + eps)


def _extract_boundary(mask: torch.Tensor) -> torch.Tensor:
    """Extract a 1-pixel boundary map from a binary mask using morphological gradient."""
    original_shape = mask.shape
    original_device = mask.device
    original_dtype = mask.dtype
    
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    
    # Convert to float for processing
    x = mask.float()
    
    # Add channel dimension for max_pool2d
    x = x.unsqueeze(1) if x.dim() == 3 else x
    
    # Dilation: max pool on the mask
    dil = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    # Erosion: 1 - max_pool(1 - mask)
    er = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
    
    # Remove channel dimension
    dil = dil.squeeze(1)
    er = er.squeeze(1)
    
    # Gradient (dilation - erosion)
    boundary = (dil - er).clamp_(0, 1)
    result = (boundary > 0.5).to(original_dtype)
    
    # Restore original shape
    if len(original_shape) == 2:
        result = result.squeeze(0)
    
    return result.to(original_device)


def boundary_f1(
    mask_gt: torch.Tensor,
    mask_pred: torch.Tensor,
    tolerance: int = 1,
    eps: float = 1e-7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute boundary precision, recall, and F1 score with pixel tolerance.

    This metric evaluates the quality of predicted object boundaries by comparing extracted boundaries from
    ground truth and predicted masks. A tolerance parameter allows for slight misalignments, making the
    metric more robust to minor prediction errors.

    Args:
        mask_gt (torch.Tensor): Ground truth binary mask of shape (H, W) or (N, H, W).
        mask_pred (torch.Tensor): Predicted binary mask of shape (H, W) or (N, H, W).
        tolerance (int): Dilation radius in pixels used for boundary matching. Default is 1.
        eps (float): Small constant for numerical stability. Default is 1e-7.

    Returns:
        (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing:
            - precision (torch.Tensor): Boundary precision for each mask, shape (N,).
            - recall (torch.Tensor): Boundary recall for each mask, shape (N,).
            - f1 (torch.Tensor): Boundary F1 score for each mask, shape (N,).

    Notes:
        - Boundaries are extracted using morphological gradient (dilation - erosion).
        - Tolerance creates a band around boundaries where matches are allowed.
        - Useful for instance segmentation where exact pixel alignment is less critical.

    Example:
        >>> gt = torch.zeros(100, 100)
        >>> gt[20:80, 20:80] = 1
        >>> pred = torch.zeros(100, 100)
        >>> pred[22:78, 22:78] = 1  # Slightly smaller prediction
        >>> precision, recall, f1 = boundary_f1(gt, pred, tolerance=2)
        >>> print(f"Boundary F1: {f1.item():.3f}")
    """
    if mask_gt.dim() == 2:
        mask_gt = mask_gt.unsqueeze(0)
    if mask_pred.dim() == 2:
        mask_pred = mask_pred.unsqueeze(0)
    
    device = mask_gt.device
    gt_b = _extract_boundary(mask_gt).bool()
    pr_b = _extract_boundary(mask_pred).bool()
    
    if tolerance > 0:
        k = 2 * tolerance + 1
        # Dilate boundaries for tolerance matching
        gt_d = F.max_pool2d(
            gt_b.unsqueeze(1).float(), 
            kernel_size=k, 
            stride=1, 
            padding=tolerance
        ).squeeze(1).bool()
        pr_d = F.max_pool2d(
            pr_b.unsqueeze(1).float(), 
            kernel_size=k, 
            stride=1, 
            padding=tolerance
        ).squeeze(1).bool()
    else:
        gt_d, pr_d = gt_b, pr_b
    
    # Compute TP, FP, FN
    tp = (pr_b & gt_d).sum(dim=(1, 2)).float()
    fp = (pr_b & (~gt_d)).sum(dim=(1, 2)).float()
    fn = (gt_b & (~pr_d)).sum(dim=(1, 2)).float()
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    return precision, recall, f1

def boundary_iou(
    mask_gt: torch.Tensor,
    mask_pred: torch.Tensor,
    tolerance: int = 1,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute boundary IoU between ground truth and predicted masks with pixel tolerance.
    
    This metric evaluates boundary quality by computing IoU on extracted boundary pixels,
    with optional tolerance for slight misalignments. Useful for tasks where precise
    boundary localization is critical.
    
    Args:
        mask_gt (torch.Tensor): Ground truth binary mask of shape (H, W) or (N, H, W).
        mask_pred (torch.Tensor): Predicted binary mask of shape (H, W) or (N, H, W).
        tolerance (int): Dilation radius in pixels for boundary matching. Default is 1.
        eps (float): Small constant for numerical stability. Default is 1e-7.
        
    Returns:
        (torch.Tensor): Boundary IoU for each mask, shape (N,).
        
    Notes:
        - Boundaries are extracted using morphological gradient (dilation - erosion).
        - Tolerance creates a band around boundaries where matches are allowed.
        - Returns per-pair IoU values, unlike boundary_f1 which returns precision/recall/f1.
        
    Example:
        >>> gt = torch.zeros(100, 100)
        >>> gt[20:80, 20:80] = 1
        >>> pred = torch.zeros(100, 100)
        >>> pred[22:78, 22:78] = 1
        >>> biou = boundary_iou(gt, pred, tolerance=2)
        >>> print(f"Boundary IoU: {biou.item():.3f}")
    """
    if mask_gt.dim() == 2:
        mask_gt = mask_gt.unsqueeze(0)
    if mask_pred.dim() == 2:
        mask_pred = mask_pred.unsqueeze(0)

    gt_b = _extract_boundary(mask_gt).bool()
    pr_b = _extract_boundary(mask_pred).bool()
    
    # ADD: Check if boundaries exist
    gt_b_count = gt_b.sum(dim=(1, 2)).float()
    pr_b_count = pr_b.sum(dim=(1, 2)).float()
    has_boundary = (gt_b_count > 0) | (pr_b_count > 0)

    if tolerance > 0:
        k = 2 * tolerance + 1
        gt_b = F.max_pool2d(
            gt_b.unsqueeze(1).float(), 
            kernel_size=k, 
            stride=1, 
            padding=tolerance
        ).squeeze(1).bool()
        pr_b = F.max_pool2d(
            pr_b.unsqueeze(1).float(), 
            kernel_size=k, 
            stride=1, 
            padding=tolerance
        ).squeeze(1).bool()

    inter = (gt_b & pr_b).sum(dim=(1, 2)).float()
    union = (gt_b | pr_b).sum(dim=(1, 2)).float()
    biou = inter / (union + eps)
    
    # CHANGE: Set to 0 where no boundaries exist
    biou = torch.where(has_boundary, biou, torch.zeros_like(biou))
    
    return biou



def kpt_iou(
    kpt1: torch.Tensor, kpt2: torch.Tensor, area: torch.Tensor, sigma: list[float], eps: float = 1e-7
) -> torch.Tensor:
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate covariance matrix from oriented bounding boxes.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance matrices corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1: torch.Tensor, obb2: torch.Tensor, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 5), format xywhr.
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 5), format xywhr.
        CIoU (bool, optional): If True, calculate CIoU.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        (torch.Tensor): OBB similarities, shape (N,).

    Notes:
        OBB format: [center_x, center_y, width, height, rotation_angle].

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """
    Calculate the probabilistic IoU between oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.

    References:
        https://arxiv.org/pdf/2106.06072v1.pdf
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_bce(eps: float = 0.1) -> tuple[float, float]:
    """
    Compute smoothed positive and negative Binary Cross-Entropy targets.

    Args:
        eps (float, optional): The epsilon value for label smoothing.

    Returns:
        pos (float): Positive label smoothing BCE target.
        neg (float): Negative label smoothing BCE target.

    References:
        https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix(DataExportMixin):
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of category.
        names (list[str]): The names of the classes, used as labels on the plot.
        matches (dict): Contains the indices of ground truths and predictions categorized into TP, FP and FN.
    """

    def __init__(self, names: dict[int, str] = [], task: str = "detect", save_matches: bool = False):
        """
        Initialize a ConfusionMatrix instance.

        Args:
            names (dict[int, str], optional): Names of classes, used as labels on the plot.
            task (str, optional): Type of task, either 'detect' or 'classify'.
            save_matches (bool, optional): Save the indices of GTs, TPs, FPs, FNs for visualization.
        """
        self.task = task
        self.nc = len(names)  # number of classes
        self.matrix = np.zeros((self.nc, self.nc)) if self.task == "classify" else np.zeros((self.nc + 1, self.nc + 1))
        self.names = names  # name of classes
        self.matches = {} if save_matches else None

    def _append_matches(self, mtype: str, batch: dict[str, Any], idx: int) -> None:
        """
        Append the matches to TP, FP, FN or GT list for the last batch.

        This method updates the matches dictionary by appending specific batch data
        to the appropriate match type (True Positive, False Positive, or False Negative).

        Args:
            mtype (str): Match type identifier ('TP', 'FP', 'FN' or 'GT').
            batch (dict[str, Any]): Batch data containing detection results with keys
                like 'bboxes', 'cls', 'conf', 'keypoints', 'masks'.
            idx (int): Index of the specific detection to append from the batch.

        Note:
            For masks, handles both overlap and non-overlap cases. When masks.max() > 1.0,
            it indicates overlap_mask=True with shape (1, H, W), otherwise uses direct indexing.
        """
        if self.matches is None:
            return
        for k, v in batch.items():
            if k in {"bboxes", "cls", "conf", "keypoints"}:
                self.matches[mtype][k] += v[[idx]]
            elif k == "masks":
                # NOTE: masks.max() > 1.0 means overlap_mask=True with (1, H, W) shape
                self.matches[mtype][k] += [v[0] == idx + 1] if v.max() > 1.0 else [v[idx]]

    def process_cls_preds(self, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        """
        Update confusion matrix for classification task.

        Args:
            preds (list[N, min(nc,5)]): Predicted class labels.
            targets (list[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(
        self,
        detections: dict[str, torch.Tensor],
        batch: dict[str, Any],
        conf: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        """
        Update confusion matrix for object detection task.

        Args:
            detections (dict[str, torch.Tensor]): Dictionary containing detected bounding boxes and their associated information.
                                       Should contain 'cls', 'conf', and 'bboxes' keys, where 'bboxes' can be
                                       Array[N, 4] for regular boxes or Array[N, 5] for OBB with angle.
            batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' (Array[M, 4]| Array[M, 5]) and
                'cls' (Array[M]) keys, where M is the number of ground truth objects.
            conf (float, optional): Confidence threshold for detections.
            iou_thres (float, optional): IoU threshold for matching detections to ground truth.
        """
        gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]
        if self.matches is not None:  # only if visualization is enabled
            self.matches = {k: defaultdict(list) for k in {"TP", "FP", "FN", "GT"}}
            for i in range(gt_cls.shape[0]):
                self._append_matches("GT", batch, i)  # store GT
        is_obb = gt_bboxes.shape[1] == 5  # check if boxes contains angle for OBB
        conf = 0.25 if conf in {None, 0.01 if is_obb else 0.001} else conf  # apply 0.25 if default val conf is passed
        no_pred = detections["cls"].shape[0] == 0
        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if not no_pred:
                detections = {k: detections[k][detections["conf"] > conf] for k in detections}
                detection_classes = detections["cls"].int().tolist()
                for i, dc in enumerate(detection_classes):
                    self.matrix[dc, self.nc] += 1  # FP
                    self._append_matches("FP", detections, i)
            return
        if no_pred:
            gt_classes = gt_cls.int().tolist()
            for i, gc in enumerate(gt_classes):
                self.matrix[self.nc, gc] += 1  # FN
                self._append_matches("FN", batch, i)
            return

        detections = {k: detections[k][detections["conf"] > conf] for k in detections}
        gt_classes = gt_cls.int().tolist()
        detection_classes = detections["cls"].int().tolist()
        bboxes = detections["bboxes"]
        iou = batch_probiou(gt_bboxes, bboxes) if is_obb else box_iou(gt_bboxes, bboxes)

        x = torch.where(iou > iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                dc = detection_classes[m1[j].item()]
                self.matrix[dc, gc] += 1  # TP if class is correct else both an FP and an FN
                if dc == gc:
                    self._append_matches("TP", detections, m1[j].item())
                else:
                    self._append_matches("FP", detections, m1[j].item())
                    self._append_matches("FN", batch, i)
            else:
                self.matrix[self.nc, gc] += 1  # FN
                self._append_matches("FN", batch, i)

        for i, dc in enumerate(detection_classes):
            if not any(m1 == i):
                self.matrix[dc, self.nc] += 1  # FP
                self._append_matches("FP", detections, i)

    def matrix(self):
        """Return the confusion matrix."""
        return self.matrix

    def tp_fp(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return true positives and false positives.

        Returns:
            tp (np.ndarray): True positives.
            fp (np.ndarray): False positives.
        """
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp, fp) if self.task == "classify" else (tp[:-1], fp[:-1])  # remove background class if task=detect

    def plot_matches(self, img: torch.Tensor, im_file: str, save_dir: Path) -> None:
        """
        Plot grid of GT, TP, FP, FN for each image.

        Args:
            img (torch.Tensor): Image to plot onto.
            im_file (str): Image filename to save visualizations.
            save_dir (Path): Location to save the visualizations to.
        """
        if not self.matches:
            return
        from .ops import xyxy2xywh
        from .plotting import plot_images

        # Create batch of 4 (GT, TP, FP, FN)
        labels = defaultdict(list)
        for i, mtype in enumerate(["GT", "FP", "TP", "FN"]):
            mbatch = self.matches[mtype]
            if "conf" not in mbatch:
                mbatch["conf"] = torch.tensor([1.0] * len(mbatch["bboxes"]), device=img.device)
            mbatch["batch_idx"] = torch.ones(len(mbatch["bboxes"]), device=img.device) * i
            for k in mbatch.keys():
                labels[k] += mbatch[k]

        labels = {k: torch.stack(v, 0) if len(v) else torch.empty(0) for k, v in labels.items()}
        if self.task != "obb" and labels["bboxes"].shape[0]:
            labels["bboxes"] = xyxy2xywh(labels["bboxes"])
        (save_dir / "visualizations").mkdir(parents=True, exist_ok=True)
        plot_images(
            labels,
            img.repeat(4, 1, 1, 1),
            paths=["Ground Truth", "False Positives", "True Positives", "False Negatives"],
            fname=save_dir / "visualizations" / Path(im_file).name,
            names=self.names,
            max_subplots=4,
            conf_thres=0.001,
        )

    @TryExcept(msg="ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize: bool = True, save_dir: str = "", on_plot=None):
        """
        Plot the confusion matrix using matplotlib and save it to a file.

        Args:
            normalize (bool, optional): Whether to normalize the confusion matrix.
            save_dir (str, optional): Directory where the plot will be saved.
            on_plot (callable, optional): An optional callback to pass plots path and data when they are rendered.
        """
        import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        names, n = list(self.names.values()), self.nc
        if self.nc >= 100:  # downsample for large class count
            k = max(2, self.nc // 60)  # step size for downsampling, always > 1
            keep_idx = slice(None, None, k)  # create slice instead of array
            names = names[keep_idx]  # slice class names
            array = array[keep_idx, :][:, keep_idx]  # slice matrix rows and cols
            n = (self.nc + k - 1) // k  # number of retained classes
        nc = nn = n if self.task == "classify" else n + 1  # adjust for background if needed
        ticklabels = (names + ["background"]) if (0 < nn < 99) and (nn == nc) else "auto"
        xy_ticks = np.arange(len(ticklabels))
        tick_fontsize = max(6, 15 - 0.1 * nc)  # Minimum size is 6
        label_fontsize = max(6, 12 - 0.1 * nc)
        title_fontsize = max(6, 12 - 0.1 * nc)
        btm = max(0.1, 0.25 - 0.001 * nc)  # Minimum value is 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            im = ax.imshow(array, cmap="Blues", vmin=0.0, interpolation="none")
            ax.xaxis.set_label_position("bottom")
            if nc < 30:  # Add score for each cell of confusion matrix
                color_threshold = 0.45 * (1 if normalize else np.nanmax(array))  # text color threshold
                for i, row in enumerate(array[:nc]):
                    for j, val in enumerate(row[:nc]):
                        val = array[i, j]
                        if np.isnan(val):
                            continue
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}" if normalize else f"{int(val)}",
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="white" if val > color_threshold else "black",
                        )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True", fontsize=label_fontsize, labelpad=10)
        ax.set_ylabel("Predicted", fontsize=label_fontsize, labelpad=10)
        ax.set_title(title, fontsize=title_fontsize, pad=20)
        ax.set_xticks(xy_ticks)
        ax.set_yticks(xy_ticks)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
        if ticklabels != "auto":
            ax.set_xticklabels(ticklabels, fontsize=tick_fontsize, rotation=90, ha="center")
            ax.set_yticklabels(ticklabels, fontsize=tick_fontsize)
        for s in {"left", "right", "bottom", "top", "outline"}:
            if s != "outline":
                ax.spines[s].set_visible(False)  # Confusion matrix plot don't have outline
            cbar.ax.spines[s].set_visible(False)
        fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=btm)  # Adjust layout to ensure equal margins
        plot_fname = Path(save_dir) / f"{title.lower().replace(' ', '_')}.png"
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """Print the confusion matrix to the console."""
        for i in range(self.matrix.shape[0]):
            LOGGER.info(" ".join(map(str, self.matrix[i])))

    def summary(self, normalize: bool = False, decimals: int = 5) -> list[dict[str, float]]:
        """
        Generate a summarized representation of the confusion matrix as a list of dictionaries, with optional
        normalization. This is useful for exporting the matrix to various formats such as CSV, XML, HTML, JSON, or SQL.

        Args:
            normalize (bool): Whether to normalize the confusion matrix values.
            decimals (int): Number of decimal places to round the output values to.

        Returns:
            (list[dict[str, float]]): A list of dictionaries, each representing one predicted class with corresponding values for all actual classes.

        Examples:
            >>> results = model.val(data="coco8.yaml", plots=True)
            >>> cm_dict = results.confusion_matrix.summary(normalize=True, decimals=5)
            >>> print(cm_dict)
        """
        import re

        names = list(self.names.values()) if self.task == "classify" else list(self.names.values()) + ["background"]
        clean_names, seen = [], set()
        for name in names:
            clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
            original_clean = clean_name
            counter = 1
            while clean_name.lower() in seen:
                clean_name = f"{original_clean}_{counter}"
                counter += 1
            seen.add(clean_name.lower())
            clean_names.append(clean_name)
        array = (self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)).round(decimals)
        return [
            dict({"Predicted": clean_names[i]}, **{clean_names[j]: array[i, j] for j in range(len(clean_names))})
            for i in range(len(clean_names))
        ]


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


@plt_settings()
def plot_pr_curve(
    px: np.ndarray,
    py: np.ndarray,
    ap: np.ndarray,
    save_dir: Path = Path("pr_curve.png"),
    names: dict[int, str] = {},
    on_plot=None,
):
    """
    Plot precision-recall curve.

    Args:
        px (np.ndarray): X values for the PR curve.
        py (np.ndarray): Y values for the PR curve.
        ap (np.ndarray): Average precision values.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


@plt_settings()
def plot_mc_curve(
    px: np.ndarray,
    py: np.ndarray,
    save_dir: Path = Path("mc_curve.png"),
    names: dict[int, str] = {},
    xlabel: str = "Confidence",
    ylabel: str = "Metric",
    on_plot=None,
):
    """
    Plot metric-confidence curve.

    Args:
        px (np.ndarray): X values for the metric-confidence curve.
        py (np.ndarray): Y values for the metric-confidence curve.
        save_dir (Path, optional): Path to save the plot.
        names (dict[int, str], optional): Dictionary mapping class indices to class names.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        on_plot (callable, optional): Function to call after plot is saved.
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def compute_ap(recall: list[float], precision: list[float]) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        ap (float): Average precision.
        mpre (np.ndarray): Precision envelope curve.
        mrec (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz  # np.trapz deprecated
        ap = func(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def compute_f2(precision: np.ndarray, recall: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    """
    Compute F2 score from precision and recall arrays.
    
    F2 score weights recall twice as much as precision, making it useful when
    recall is more important than precision (e.g., medical diagnosis, security).
    
    Args:
        precision (np.ndarray): Precision values.
        recall (np.ndarray): Recall values.
        eps (float, optional): Small value to avoid division by zero.
        
    Returns:
        (np.ndarray): F2 scores.
        
    Formula:
        F2 = (1 + 2²) * (precision * recall) / (2² * precision + recall)
        F2 = 5 * (precision * recall) / (4 * precision + recall)
    """
    return 5 * (precision * recall) / (4 * precision + recall + eps)

def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    plot: bool = False,
    on_plot=None,
    save_dir: Path = Path(),
    names: dict[int, str] = {},
    eps: float = 1e-16,
    prefix: str = "",
    target_areas: np.ndarray = None,  # ADD
    matched_gt_idx: np.ndarray = None,  # ADD
) -> tuple:
    """
    Compute AP per class with size-based breakdown.
    
    Size categories (COCO standard):
    - Small: area < 15^2  (225 pixels²)
    - Medium: 15^2 ≤ area < 32^2 (225-1024 pixels²)
    - Large: area ≥ 32^2 (≥1024 pixels²)
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    if matched_gt_idx is not None:
        matched_gt_idx = matched_gt_idx[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    # Create Precision-Recall curve
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    
    # Size-based AP
    SMALL_THRESHOLD = 15 * 15
    MEDIUM_THRESHOLD = 32 * 32
    ap_small = np.zeros(nc)
    ap_medium = np.zeros(nc)
    ap_large = np.zeros(nc)
    
    for ci, c in enumerate(unique_classes):
        i_pred = pred_cls == c
        i_gt = target_cls == c
        n_l = nt[ci]
        n_p = i_pred.sum()
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i_pred]).cumsum(0)
        tpc = tp[i_pred].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i_pred], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i_pred], precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))

        # Size-based AP (only at first IoU threshold for efficiency)
        if target_areas is not None and matched_gt_idx is not None:
            # SMALL_THRESHOLD = 32 * 32
            # MEDIUM_THRESHOLD = 96 * 96
            
            # Get indices for this class
            gt_indices_this_class = np.where(i_gt)[0]
            class_areas = target_areas[gt_indices_this_class]
            
            # Define size categories
            size_categories = [
                ('small', ap_small, 0, SMALL_THRESHOLD),
                ('medium', ap_medium, SMALL_THRESHOLD, MEDIUM_THRESHOLD),
                ('large', ap_large, MEDIUM_THRESHOLD, float('inf'))
            ]
            
            for size_name, size_ap_array, area_min, area_max in size_categories:
                # Find GT instances in this size range
                size_mask = (class_areas >= area_min) & (class_areas < area_max)
                size_gt_global_idx = gt_indices_this_class[size_mask]
                n_size = len(size_gt_global_idx)
                
                if n_size == 0:
                    # No ground truth objects in this size category for this class
                    continue
                
                # Get all predictions for this class
                class_preds_idx = np.where(i_pred)[0]
                class_matched_gt = matched_gt_idx[class_preds_idx]
                
                # Determine which predictions are TP/FP for this size category:
                # - TP: prediction matched a GT in this size category
                # - FP: prediction either matched a GT outside this size category or didn't match any GT
                
                # Create TP array: True if matched GT is in size category
                tp_size = np.isin(class_matched_gt, size_gt_global_idx).astype(float)
                
                # Get confidence scores for all predictions
                conf_size = conf[class_preds_idx]
                
                # Sort by confidence (descending)
                sort_idx = np.argsort(-conf_size)
                tp_size_sorted = tp_size[sort_idx]
                conf_size_sorted = conf_size[sort_idx]
                
                # Compute cumulative TP and FP
                tp_cumsum = np.cumsum(tp_size_sorted)
                fp_cumsum = np.cumsum(1 - tp_size_sorted)
                
                # Compute precision and recall
                # Recall: TP / total number of GTs in this size category
                recall_size = tp_cumsum / n_size
                
                # Precision: TP / (TP + FP)
                precision_size = tp_cumsum / (tp_cumsum + fp_cumsum)
                
                # Compute AP using the standard method
                ap_size, _, _ = compute_ap(recall_size, precision_size)
                
                # Store in appropriate array
                if size_name == 'small':
                    ap_small[ci] = ap_size
                elif size_name == 'medium':
                    ap_medium[ci] = ap_size
                else:  # large
                    ap_large[ci] = ap_size

    prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))

    # Compute F1 and F2
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    f2_curve = compute_f2(p_curve, r_curve, eps)
    
    names = {i: names[k] for i, k in enumerate(unique_classes) if k in names}  # dict: only classes that have data
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, f2_curve, save_dir / f"{prefix}F2_curve.png", names, ylabel="F2", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1, f2 = p_curve[:, i], r_curve[:, i], f1_curve[:, i], f2_curve[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    
    return (tp, fp, p, r, f1, f2, ap, unique_classes.astype(int), 
            p_curve, r_curve, f1_curve, f2_curve, x, prec_values,
            ap_small, ap_medium, ap_large)

class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for Ultralytics YOLO models.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        f2 (list): F2 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.
        fitness_weights (dict): Weights for computing fitness score.

    Methods:
        ap50: AP at IoU threshold of 0.5 for all classes.
        ap: AP at IoU thresholds from 0.5 to 0.95 for all classes.
        mp: Mean precision of all classes.
        mr: Mean recall of all classes.
        map50: Mean AP at IoU threshold of 0.5 for all classes.
        map75: Mean AP at IoU threshold of 0.75 for all classes.
        map: Mean AP at IoU thresholds from 0.5 to 0.95 for all classes.
        mf1: Mean F1 score of all classes.
        mf2: Mean F2 score of all classes.
        mean_results: Mean of results, returns mp, mr, map50, map.
        class_result: Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps: mAP of each class.
        fitness: Model fitness as a weighted combination of metrics.
        update: Update metric attributes with new evaluation results.
        curves: Provides a list of curves for accessing specific metrics like precision, recall, F1, etc.
        curves_results: Provide a list of results for accessing specific metrics like precision, recall, F1, etc.

    Examples:
        >>> metric = Metric()
        >>> metric.fitness_weights = {'mAP50_95': 0.6, 'f2': 0.4}
        >>> fitness_score = metric.fitness()
    """
    
    def __init__(self, fitness_weights: dict = None) -> None:
        """
        Initialize a Metric instance for computing evaluation metrics for the YOLO model.
        
        Args:
            fitness_weights (dict, optional): Custom weights for fitness calculation. If None, uses default weights
                focusing on mAP50-95. Keys can include: 'precision', 'recall', 'mAP50', 'mAP50_95', 'f1', 'f2'.

        Examples:
            >>> metric = Metric()
            >>> metric = Metric(fitness_weights={'mAP50_95': 0.7, 'f2': 0.3})
        """
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.f2 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

        # self.r_at_low_iou = []  # ADD: (nc, ) - Recall at IoU=0.15
        self.ap_small = []  # ADD: (nc, ) - AP for small objects
        self.ap_medium = []  # ADD: (nc, ) - AP for medium objects
        self.ap_large = []  # ADD: (nc, ) - AP for large objects
        
        # Default weights: focus on mAP50-95
        self.fitness_weights = fitness_weights or {
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'mAP50_95': 1.0,
            'f1': 0.0,
            'f2': 0.0
        }

    @property
    def ap50(self) -> np.ndarray | list:
        """
        Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray | list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self) -> np.ndarray | list:
        """
        Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray | list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self) -> float:
        """
        Return the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self) -> float:
        """
        Return the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self) -> float:
        """
        Return the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self) -> float:
        """
        Return the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self) -> float:
        """
        Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0
    
    @property
    def mf1(self) -> float:
        """
        Return the Mean F1 score of all classes.

        Returns:
            (float): The mean F1 score of all classes.
        """
        return self.f1.mean() if len(self.f1) else 0.0

    @property
    def mf2(self) -> float:
        """
        Return the Mean F2 score of all classes.

        Returns:
            (float): The mean F2 score of all classes.
        """
        return self.f2.mean() if len(self.f2) else 0.0

    def mean_results(self) -> List[float]:
        """Return mean of results, mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i: int) -> tuple[float, ...]:
        """Return class-aware result: p[i], r[i], ap50[i], ap[i], f1[i], f2[i], ap_small[i], ap_medium[i], ap_large[i]."""
        def _at(arr, idx: int) -> float:
            if hasattr(arr, "__len__") and len(arr) > idx:
                val = arr[idx]
                return float(val) if hasattr(val, "item") else val
            return 0.0

        return (
            _at(self.p, i),
            _at(self.r, i),
            _at(self.ap50, i),
            _at(self.ap, i),
            _at(self.f1, i),
            _at(self.f2, i),
            _at(self.ap_small, i),
            _at(self.ap_medium, i),
            _at(self.ap_large, i),
        )

    @property
    def maps(self) -> np.ndarray:
        """Return mAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps
    
    # @property
    # def mr_low_iou(self) -> float:
    #     """Mean Recall at IoU=0.15."""
    #     return self.r_at_low_iou.mean() if len(self.r_at_low_iou) else 0.0
    
    @property
    def map_small(self) -> float:
        """Mean AP for small objects (area < 32²)."""
        return self.ap_small.mean() if len(self.ap_small) else 0.0
    
    @property
    def map_medium(self) -> float:
        """Mean AP for medium objects (32² ≤ area < 96²)."""
        return self.ap_medium.mean() if len(self.ap_medium) else 0.0
    
    @property
    def map_large(self) -> float:
        """Mean AP for large objects (area ≥ 96²)."""
        return self.ap_large.mean() if len(self.ap_large) else 0.0

    def fitness(self) -> float:
        """
        Return model fitness as a weighted combination of metrics.
        
        Computes a weighted sum of available metrics (precision, recall, mAP50, mAP50-95, F1, F2) based on
        the configured fitness_weights. The result is normalized by the sum of weights. If no weights are
        configured or all weights are zero, falls back to mAP50-95.
        
        Returns:
            (float): Weighted fitness score in range [0.0, 1.0].

        Examples:
            >>> metric = Metric(fitness_weights={'mAP50': 0.3, 'mAP50_95': 0.7})
            >>> metric.map50 = 0.85
            >>> metric.map = 0.75
            >>> fitness = metric.fitness()
            >>> print(f"Fitness: {fitness:.3f}")
        """
        # Map weight keys to actual metric values
        metrics_map = {
            'precision': self.mp,
            'recall': self.mr,
            'mAP50': self.map50,
            'mAP50_95': self.map,
            'f1': self.mf1,
            'f2': self.mf2
        }
        
        fitness = 0.0
        total_weight = 0.0
        
        for key, weight in self.fitness_weights.items():
            if weight > 0 and key in metrics_map:
                metric_value = metrics_map[key]
                if not np.isnan(metric_value):
                    fitness += weight * metric_value
                    total_weight += weight
        
        # Normalize by total weight if any weights were applied
        if total_weight > 0:
            return fitness / total_weight
        
        # Fallback to mAP50-95 if no weights configured
        return self.map

    def update(self, results: tuple):
        """
        Update the evaluation metrics with a new set of results.

        Args:
            results (tuple): A tuple containing evaluation metrics:
                - p (list): Precision for each class.
                - r (list): Recall for each class.
                - f1 (list): F1 score for each class.
                - f2 (list): F2 score for each class.
                - all_ap (list): AP scores for all classes and all IoU thresholds.
                - ap_class_index (list): Index of class for each AP score.
                - p_curve (list): Precision curve for each class.
                - r_curve (list): Recall curve for each class.
                - f1_curve (list): F1 curve for each class.
                - f2_curve (list): F2 curve for each class.
                - px (list): X values for the curves.
                - prec_values (list): Precision values for each class.
        """
        (
        self.p,
        self.r,
        self.f1,
        self.f2,           # ADD this line
        self.all_ap,
        self.ap_class_index,
        self.p_curve,
        self.r_curve,
        self.f1_curve,
        self.f2_curve,     # ADD this line
        self.px,
        self.prec_values,
        # self.r_at_low_iou,  # ADD
        self.ap_small,      # ADD
        self.ap_medium,     # ADD
        self.ap_large,      # ADD
    ) = results

    @property
    def curves(self) -> list:
        """Return a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self) -> list[list]:
        """Return a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.f2_curve, "Confidence", "F2"],  # ADD this line
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class DetMetrics(SimpleClass, DataExportMixin):
    """
    Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).

    Attributes:
        names (dict[int, str]): A dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'detect'.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes, target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.
        fitness_weights (dict): Weights for computing fitness score.

    Methods:
        update_stats: Update statistics by appending new values to existing stat collections.
        process: Process predicted results for object detection and update metrics.
        clear_stats: Clear the stored statistics.
        keys: Return a list of keys for accessing specific metrics.
        mean_results: Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95.
        class_result: Return the result of evaluating the performance of an object detection model on a specific class.
        maps: Return mean Average Precision (mAP) scores per class.
        fitness: Return the fitness of box object.
        ap_class_index: Return the average precision index per class.
        results_dict: Return dictionary of computed performance metrics and statistics.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Return a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class detection metrics as a list of dictionaries.

    Examples:
        >>> metrics = DetMetrics(names={0: 'person', 1: 'car'})
        >>> metrics = DetMetrics(names={0: 'person'}, fitness_weights={'mAP50_95': 0.6, 'f2': 0.4})
    """
    
    def __init__(self, names: dict[int, str] = {}, fitness_weights: dict = None) -> None:
        """
        Initialize a DetMetrics instance with a save directory, plot flag, and class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
            fitness_weights (dict, optional): Custom weights for fitness calculation. If None, uses default weights
                focusing on mAP50-95. Keys can include: 'precision', 'recall', 'mAP50', 'mAP50_95', 'f1', 'f2'.

        Examples:
            >>> metrics = DetMetrics(names={0: 'person', 1: 'car'})
            >>> metrics = DetMetrics(names={0: 'person'}, fitness_weights={'mAP50_95': 0.7, 'recall': 0.3})
        """
        self.names = names
        
        # Default weights for detection: focus on mAP50-95
        default_weights = {
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'mAP50_95': 0.1,
            'f1': 0.0,
            'f2': 0.9
        }
        # self.fitness_weights = fitness_weights or default_weights
        self.fitness_weights = fitness_weights or default_weights
        
        self.box = Metric(fitness_weights=self.fitness_weights)
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"
        self.stats = dict(
            tp=[],
            matched_gt_idx = [],
            conf=[],
            pred_cls=[], target_cls=[],
            target_areas = [],
            target_img=[],
            target_img_names = [],
            pred_img = [],
        )
        self.per_image = {}
        self.per_image_conf_thr = None
        self.nt_per_class = None
        self.nt_per_image = None
    
    def _conf_for_per_image(self) -> float:
        """
        Get the confidence threshold to use for per-image metric computation.
        
        Returns the validator-provided threshold if set, otherwise defaults to 0.25
        (matching the ConfusionMatrix default threshold).
        """
        return 0.01
        return 0.25 if self.per_image_conf_thr is None else float(self.per_image_conf_thr)

    def update_stats(self, stat: dict[str, Any]) -> None:
        """
        Update statistics by appending new values to existing stat collections.

        Args:
            stat (dict[str, any]): Dictionary containing new statistical values to append.
                         Keys should match existing keys in self.stats.
        """
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """
        Process predicted results for object detection and update metrics.

        Args:
            save_dir (Path): Directory to save plots. Defaults to Path(".").
            plot (bool): Whether to plot precision-recall curves. Defaults to False.
            on_plot (callable, optional): Function to call after plots are generated. Defaults to None.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
        """
        stats = {k: np.concatenate(v, 0) for k, v in self.stats.items()}  # to numpy
        if not stats:
            return stats
        results = ap_per_class(
            stats["tp"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            save_dir=save_dir,
            names=self.names,
            on_plot=on_plot,
            prefix="Box",
            target_areas=stats.get("target_areas", None),  # ADD
            matched_gt_idx=stats.get("matched_gt_idx", None),  # ADD
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=len(self.names))
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=len(self.names))

        conf_thr = self._conf_for_per_image()
        self.per_image["box"] = self._compute_per_image_prf(stats, conf_thr, iou_index=0)
        return stats
    
    def _compute_per_image_prf(self, stats, conf_thr: float, iou_index: int = 0) -> dict[str, list]:
        """
        Compute per-image precision, recall, and F2 score for bounding boxes.
        
        Args:
            stats: Dictionary of concatenated statistics arrays
            conf_thr: Confidence threshold to filter predictions
            iou_index: Index into tp array (0 for IoU=0.50, 1 for IoU=0.75, etc.)
        
        Returns:
            Dictionary with lists of per-image metrics:
            - image_id: Image identifier
            - tp: True positives count
            - fp: False positives count
            - fn: False negatives count
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f2: F2 score (weighted F-score favoring recall: 5*P*R / (4*P + R))
        """
        # Filter predictions by confidence threshold
        pred_keep = stats["conf"] >= conf_thr
        
        # Get all unique image IDs that have either predictions or ground truth
        img_ids = np.union1d(stats["pred_img"][pred_keep], stats["target_img_names"])
        
        # Initialize output dictionary
        out = {"image_id": [], "tp": [], "fp": [], "fn": [], "precision": [], "recall": [], "f2": []}
        
        # Compute metrics for each image independently
        for gid in img_ids:
            # Filter predictions for this image
            pid = (stats["pred_img"] == gid) & pred_keep
            
            # Extract TP values for this image at the specified IoU threshold
            tp_vec = stats["tp"][pid, iou_index] if pid.any() else np.zeros((0,), dtype=np.float32)
            
            # Count true positives, false positives, and false negatives
            tp = int(tp_vec.sum())
            fp = int(pid.sum() - tp)  # Total predictions minus TPs
            fn = int((stats["target_img_names"] == gid).sum() - tp)  # Total GTs minus TPs
            
            # Compute precision, recall, and F2 score with zero-division handling
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) else 0.0
            
            # Store results
            out["image_id"].append(str(gid))
            out["tp"].append(tp)
            out["fp"].append(fp)
            out["fn"].append(fn)
            out["precision"].append(float(p))
            out["recall"].append(float(r))
            out["f2"].append(float(f2))
        return out

    def clear_stats(self):
        """Clear the stored statistics."""
        for v in self.stats.values():
            v.clear()

    @property
    def keys(self) -> list[str]:
        """Return a list of keys for accessing specific metrics."""
        return [
            "metrics/precision(B)", 
            "metrics/recall(B)",
            # "metrics/recall@0.15(B)",  # ADD 
            "metrics/mAP50(B)", 
            "metrics/mAP50-95(B)",
            "metrics/f1(B)",
            "metrics/f2(B)",
            "metrics/mAP_small(B)",    # ADD
            "metrics/mAP_medium(B)",   # ADD
            "metrics/mAP_large(B)",    # ADD
        ]

    def mean_results(self) -> List[float]:
        """Calculate mean of detected objects & return precision, recall, mAP50, mAP50-95, and mF2."""
        return self.box.mean_results() + [
            self.box.mf1, self.box.mf2,
            # self.box.mr_low_iou,  # ADD
            self.box.map_small,   # ADD
            self.box.map_medium,  # ADD
            self.box.map_large    # ADD
        ]

    def class_result(self, i: int) -> tuple[float, float, float, float]:
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """Return mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self) -> float:
        """
        Return the fitness score of the detection model.
        
        Computes fitness as a weighted combination of box detection metrics based on the configured fitness_weights.
        
        Returns:
            (float): Fitness score in range [0.0, 1.0].

        Examples:
            >>> metrics = DetMetrics(names={0: 'person', 1: 'car'})
            >>> # ... process validation results ...
            >>> fitness = metrics.fitness
            >>> print(f"Model fitness: {fitness:.3f}")
        """
        return self.box.fitness()

    @property
    def ap_class_index(self) -> list:
        """Return the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self) -> dict[str, float]:
        """Return dictionary of computed performance metrics and statistics."""
        keys = self.keys + ["fitness"]
        values = ((float(x) if hasattr(x, "item") else x) for x in (self.mean_results() + [self.fitness]))
        return dict(zip(keys, values))

    @property
    def curves(self) -> list[str]:
        """Return a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)", 
            "F1-Confidence(B)", 
            "F2-Confidence(B)",  # ADD this line
            "Precision-Confidence(B)", 
            "Recall-Confidence(B)"
        ]

    @property
    def curves_results(self) -> list[list]:
        """Return a list of computed performance metrics and statistics."""
        return self.box.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """
        Generate a summarized representation of per-class detection metrics as a list of dictionaries. Includes shared
        scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

        Args:
           normalize (bool): For Detect metrics, everything is normalized  by default [0-1].
           decimals (int): Number of decimal places to round the metrics values to.

        Returns:
           (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric values.

        Examples:
           >>> results = model.val(data="coco8.yaml")
           >>> detection_summary = results.summary()
           >>> print(detection_summary)
        """
        per_class = {
            "Box-P": self.box.p,
            "Box-R": self.box.r,
            "Box-F1": self.box.f1,
            "Box-F2": self.box.f2,  # ADD this line
        }
        return [
            {
                "Class": self.names[self.ap_class_index[i]],
                "Images": self.nt_per_image[self.ap_class_index[i]],
                "Instances": self.nt_per_class[self.ap_class_index[i]],
                **{k: round(v[i], decimals) for k, v in per_class.items()},
                "mAP50": round(self.class_result(i)[2], decimals),
                "mAP50-95": round(self.class_result(i)[3], decimals),
            }
            for i in range(len(per_class["Box-P"]))
        ]


class SegmentMetrics(DetMetrics):
    """
    Calculate and aggregate detection and segmentation metrics over a given set of classes.

    This class extends DetMetrics to include mask-based evaluation metrics. It computes standard segmentation
    metrics (precision, recall, mAP) as well as additional pixel-level metrics including Dice coefficient,
    mean IoU, and boundary F1 score. These metrics are aggregated per-class and averaged across the dataset.

    Attributes:
        seg (Metric): Metric object for mask-based precision-recall metrics.
        task (str): Task type, set to "segment".
        names (dict[int, str]): Dictionary mapping class indices to class names.
        stats (dict): Dictionary containing statistics including 'tp_m' for mask true positives.
        fitness_weights (dict): Weights for computing fitness score (supports both box and mask metrics).

    Methods:
        update_mask_aggregates: Update per-class Dice, mIoU, and boundary F1 metrics.
        reset_mask_aggregates: Clear per-class metric buffers.
        process: Process all accumulated statistics and compute final metrics.

    Properties:
        mdice (float): Mean Dice coefficient across all classes.
        miou (float): Mean Intersection over Union across all classes.
        mbf1 (float): Mean boundary F1 score across all classes.
        fitness (float): Combined fitness score from detection and segmentation metrics.

    Examples:
        >>> from ultralytics.utils.metrics import SegmentMetrics
        >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
        >>> custom_weights = {'box_mAP50_95': 0.3, 'mask_mAP50_95': 0.3, 'dice': 0.2, 'miou': 0.2}
        >>> metrics = SegmentMetrics(names={0: 'cat'}, fitness_weights=custom_weights)
    """
    
    def __init__(self, names: dict[int, str] = {}, fitness_weights: dict = None) -> None:
        """
        Initialize a SegmentMetrics instance with detection and segmentation metrics.

        Args:
            names (dict[int, str]): Dictionary mapping class indices to class names. Default is {}.
            fitness_weights (dict, optional): Custom weights for fitness calculation. If None, uses default weights
                that balance box and mask mAP50-95. Keys can include box metrics (prefixed with 'box_'), mask metrics
                (prefixed with 'mask_'), and additional segmentation metrics: 'dice', 'miou', 'boundary_f1'.

        Examples:
            >>> metrics = SegmentMetrics(names={0: 'person', 1: 'car', 2: 'dog'})
            >>> weights = {'box_mAP50_95': 0.3, 'mask_mAP50_95': 0.4, 'dice': 0.2, 'miou': 0.1}
            >>> metrics = SegmentMetrics(names={0: 'cat'}, fitness_weights=weights)
        """
        # Default weights for segmentation: balance box and mask metrics
        default_weights = {
            # Box metrics
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'mAP50_95': 0.0,
            'f1': 0.0,
            'f2': 0.0,
            # Mask metrics
            'mask_precision': 0.0,
            'mask_recall': 0.0,
            'mask_mAP50': 0.0,
            'mask_mAP50_95': 0.1,
            'mask_f1': 0.0,
            'mask_f2': 0.9,
            # Additional segmentation metrics
            'dice': 0.0,
            'miou': 0.0,
            'boundary_f1': 0.0,
            'boundary_iou': 0.0
        }
        
        # self.fitness_weights = fitness_weights or default_weights
        self.fitness_weights = default_weights
        
        # Initialize parent with box-specific weights
        # box_weights = {k: v for k, v in self.fitness_weights.items() if k.startswith('box_')}
        super().__init__(names, fitness_weights=fitness_weights)
        
        # Initialize mask metrics with mask-specific weights
        mask_weights = {k.replace('mask_', ''): v for k, v in self.fitness_weights.items() if k.startswith('mask_')}
        self.seg = Metric(fitness_weights=mask_weights)
        self.task = "segment"
        self.stats["tp_m"] = []

        # Aggregates for additional segmentation metrics
        self._init_done = False
        self._device = torch.device("cpu")
        self._dice_num = None
        self._dice_den = None
        self._iou_inter = None
        self._iou_union = None
        self._b_tp = None
        self._b_fp = None
        self._b_fn = None
        # Boundary IoU accumulators
        self._biou_inter = None
        self._biou_union = None

    def _ensure_init(self):
        """
        Initialize per-class accumulation tensors for Dice, IoU, and boundary metrics.

        This method creates zero-initialized tensors for each class to accumulate metric components
        across all predictions. Uses float64 precision on CPU for numerical stability with large datasets.

        Notes:
            - Called automatically before first metric update.
            - Creates tensors with length equal to number of classes.
            - All tensors stored on CPU to prevent GPU memory overflow.
        """
        if self._init_done:
            return
        nc = len(self.names)
        zeros = torch.zeros(nc, dtype=torch.float64, device=self._device)
        self._dice_num = zeros.clone()
        self._dice_den = zeros.clone()
        self._iou_inter = zeros.clone()
        self._iou_union = zeros.clone()
        self._b_tp = zeros.clone()
        self._b_fp = zeros.clone()
        self._b_fn = zeros.clone()
        # Boundary IoU accumulators
        self._biou_inter = zeros.clone()
        self._biou_union = zeros.clone()
        self._init_done = True

    def reset_mask_aggregates(self) -> None:
        """
        Clear per-class aggregation buffers for Dice, mIoU, and boundary F1 metrics.

        This method resets all accumulated metric values, preparing the instance for a new evaluation pass.
        Should be called at the start of each validation/test epoch.

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> # ... accumulate metrics during validation ...
            >>> metrics.reset_mask_aggregates()  # Reset for next epoch
        """
        self._init_done = False
        self._dice_num = self._dice_den = self._iou_inter = self._iou_union = None
        self._b_tp = self._b_fp = self._b_fn = None
        self._biou_inter = self._biou_union = None

    def update_mask_aggregates(
        self,
        cls_indices: torch.Tensor,
        gt_masks: torch.Tensor,
        pred_masks: torch.Tensor,
        boundary_tolerance: int = 1,
    ) -> None:
        """
        Update per-class aggregates for Dice, mIoU, and boundary F1 using matched mask pairs.

        This method processes matched ground truth and predicted mask pairs, computing intersection, union,
        and boundary statistics for each pair, then accumulating these values per class. Matching should be
        performed before calling this method (typically using IoU threshold-based assignment).

        Args:
            cls_indices (torch.Tensor): Ground truth class indices for each matched pair, shape (K,).
            gt_masks (torch.Tensor): Boolean ground truth masks, shape (K, H, W).
            pred_masks (torch.Tensor): Boolean predicted masks, shape (K, H, W).
            boundary_tolerance (int): Dilation radius in pixels for boundary matching. Default is 1.

        Notes:
            - Assumes 1:1 matching between gt_masks and pred_masks.
            - Uses scatter_add for efficient per-class accumulation.
            - All computations performed on input device, then moved to CPU for storage.
            - Empty input (K=0) is handled gracefully without errors.

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> cls = torch.tensor([0, 1, 0])  # 3 matched pairs
            >>> gt = torch.rand(3, 100, 100) > 0.5  # Random binary masks
            >>> pred = torch.rand(3, 100, 100) > 0.5
            >>> metrics.update_mask_aggregates(cls, gt, pred)
        """
        if gt_masks.numel() == 0:
            return
        
        self._ensure_init()
        
        device = gt_masks.device
        nc = len(self.names)
        cls_indices = cls_indices.to(torch.long)
        
        # FIXED: Filter out invalid class indices instead of clamping
        valid_mask = (cls_indices >= 0) & (cls_indices < nc)
        if not valid_mask.any():
            return  # No valid classes to process
        
        # Apply filter to all inputs
        cls_indices = cls_indices[valid_mask]
        gt_masks = gt_masks[valid_mask].bool()
        pred_masks = pred_masks[valid_mask].bool()
        
        # Compute intersection, areas, and union
        inter = (gt_masks & pred_masks).sum(dim=(1, 2)).float()
        area_gt = gt_masks.sum(dim=(1, 2)).float()
        area_pr = pred_masks.sum(dim=(1, 2)).float()
        union = (area_gt + area_pr - inter).clamp(min=0.0)
        
        # Dice components
        dice_n = 2.0 * inter
        dice_d = (area_gt + area_pr).clamp(min=1e-7)  # Avoid division by zero
        
        # Boundary extraction and matching
        gt_b = _extract_boundary(gt_masks).bool()
        pr_b = _extract_boundary(pred_masks).bool()
        
        if boundary_tolerance > 0:
            k = 2 * boundary_tolerance + 1
            # Dilate boundaries for tolerance matching
            gt_d = F.max_pool2d(
                gt_b.unsqueeze(1).float(), 
                kernel_size=k, 
                stride=1, 
                padding=boundary_tolerance
            ).squeeze(1).bool()
            pr_d = F.max_pool2d(
                pr_b.unsqueeze(1).float(), 
                kernel_size=k, 
                stride=1, 
                padding=boundary_tolerance
            ).squeeze(1).bool()
        else:
            gt_d, pr_d = gt_b, pr_b
        
        # Boundary metrics
        tp_b = (pr_b & gt_d).sum(dim=(1, 2)).float()
        fp_b = (pr_b & (~gt_d)).sum(dim=(1, 2)).float()
        fn_b = (gt_b & (~pr_d)).sum(dim=(1, 2)).float()
        b_inter = (gt_d & pr_d).sum(dim=(1, 2)).float()
        b_union = (gt_d | pr_d).sum(dim=(1, 2)).float()
        
        
        # Move to CPU for accumulation
        cls_indices_cpu = cls_indices.cpu()
        
        # Accumulate per-class (using double precision for numerical stability)
        self._dice_num.scatter_add_(0, cls_indices_cpu, dice_n.cpu().double())
        self._dice_den.scatter_add_(0, cls_indices_cpu, dice_d.cpu().double())
        self._iou_inter.scatter_add_(0, cls_indices_cpu, inter.cpu().double())
        self._iou_union.scatter_add_(0, cls_indices_cpu, union.cpu().double())
        self._b_tp.scatter_add_(0, cls_indices_cpu, tp_b.cpu().double())
        self._b_fp.scatter_add_(0, cls_indices_cpu, fp_b.cpu().double())
        self._b_fn.scatter_add_(0, cls_indices_cpu, fn_b.cpu().double())
        self._biou_inter.scatter_add_(0, cls_indices_cpu, b_inter.cpu().double())
        self._biou_union.scatter_add_(0, cls_indices_cpu, b_union.cpu().double())

    @property
    def mdice(self) -> float:
        """
        Calculate mean Dice coefficient across all classes with valid predictions.

        The Dice coefficient (also known as F1 score for binary segmentation) measures the overlap between
        predicted and ground truth masks: Dice = 2|X∩Y| / (|X| + |Y|). This property computes the per-class
        Dice from accumulated intersection and union statistics, then averages over classes with non-zero
        denominator.

        Returns:
            (float): Mean Dice coefficient in range [0, 1]. Returns 0.0 if no valid classes are present.

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> # ... process predictions ...
            >>> dice = metrics.mdice
            >>> print(f"Mean Dice: {dice:.3f}")
        """
        self._ensure_init()
        eps = 1e-7
        dice_c = self._dice_num / (self._dice_den + eps)
        valid = self._dice_den > eps
        if not valid.any():
            return 0.0
        return float(dice_c[valid].mean().item())

    @property
    def miou(self) -> float:
        """
        Calculate mean Intersection over Union (mIoU) across all classes with valid predictions.

        IoU measures the overlap between predicted and ground truth masks: IoU = |X∩Y| / |X∪Y|.
        This property computes per-class IoU from accumulated statistics, then averages over classes
        with non-zero union.

        Returns:
            (float): Mean IoU in range [0, 1]. Returns 0.0 if no valid classes are present.

        Notes:
            - Also known as Jaccard index.
            - Standard metric for semantic and instance segmentation.
            - More strict than Dice coefficient (IoU ≤ Dice).

        Example:
            >>> metrics = SegmentMetrics(names={0: 'person', 1: 'car'})
            >>> # ... accumulate predictions ...
            >>> iou = metrics.miou
            >>> print(f"Mean IoU: {iou:.3f}")
        """
        self._ensure_init()
        eps = 1e-7
        iou_c = self._iou_inter / (self._iou_union + eps)
        valid = self._iou_union > eps
        if not valid.any():
            return 0.0
        return float(iou_c[valid].mean().item())

    @property
    def mbf1(self) -> float:
        """
        Calculate mean boundary F1 score across all classes with detected boundaries.

        Boundary F1 evaluates the quality of predicted object boundaries by computing precision and recall
        on extracted boundary pixels (with optional tolerance). This metric is particularly useful for
        applications where boundary accuracy is critical, such as medical image segmentation.

        Returns:
            (float): Mean boundary F1 in range [0, 1]. Returns 0.0 if no boundaries are detected.

        Notes:
            - Boundary pixels are extracted using morphological gradient (dilation - erosion).
            - Tolerance parameter allows for slight misalignments (configured in update_mask_aggregates).
            - Only classes with detected boundaries (TP+FP+FN > 0) contribute to the mean.

        Example:
            >>> metrics = SegmentMetrics(names={0: 'tumor', 1: 'organ'})
            >>> # ... process medical image predictions ...
            >>> bf1 = metrics.mbf1
            >>> print(f"Mean Boundary F1: {bf1:.3f}")
        """
        self._ensure_init()
        eps = 1e-7
        prec = self._b_tp / (self._b_tp + self._b_fp + eps)
        rec = self._b_tp / (self._b_tp + self._b_fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        valid = (self._b_tp + self._b_fp + self._b_fn) > eps
        if not valid.any():
            return 0.0
        return float(f1[valid].mean().item())
    
    @property
    def mboundary_iou(self) -> float:
        """
        Calculate mean boundary IoU across all classes with detected boundaries.
        
        Boundary IoU measures the overlap of boundary pixels between predicted and ground truth
        masks, providing an alternative to boundary F1 that directly measures boundary overlap
        rather than precision/recall trade-offs.
        
        Returns:
            (float): Mean boundary IoU in range [0, 1]. Returns 0.0 if no boundaries are detected.
            
        Notes:
            - Computed using tolerance-dilated boundaries (same as boundary F1).
            - Only classes with non-zero boundary union contribute to the mean.
            - More lenient than boundary F1 for asymmetric boundary errors.
            
        Example:
            >>> metrics = SegmentMetrics(names={0: 'cell', 1: 'nucleus'})
            >>> # ... process microscopy predictions ...
            >>> biou = metrics.mboundary_iou
            >>> print(f"Mean Boundary IoU: {biou:.3f}")
        """
        self._ensure_init()
        eps = 1e-7
        valid = self._biou_union > eps
        if not valid.any():
            return 0.0
        biou_c = self._biou_inter / (self._biou_union + eps)
        return float(biou_c[valid].mean().item())

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """
        Process accumulated detection and segmentation statistics to compute final metrics.

        This method processes both bounding box (from DetMetrics) and mask statistics, computing precision,
        recall, mAP, and additional segmentation metrics (Dice, mIoU, boundary F1). It resets per-class
        aggregates at the start to ensure fresh computation for the current evaluation pass.

        Args:
            save_dir (Path): Directory to save metric plots and results. Default is Path(".").
            plot (bool): Whether to generate and save metric plots. Default is False.
            on_plot (callable | None): Optional callback function for plot generation. Default is None.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing processed statistics including:
                - 'tp_m': Mask true positives per IoU threshold.
                - 'conf': Confidence scores.
                - 'pred_cls': Predicted class indices.
                - 'target_cls': Ground truth class indices.

        Notes:
            - Automatically calls reset_mask_aggregates() before processing.
            - Mask aggregates should be updated during validation using update_mask_aggregates().
            - Returns empty dict if no valid statistics are available.

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> # ... during validation: metrics.update_mask_aggregates() ...
            >>> results = metrics.process(save_dir=Path('./runs/val'), plot=True)
            >>> print(f"Processed {len(results)} metric types")
        """
        # Start fresh aggregates for this evaluation
        # self.reset_mask_aggregates()

        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # box metrics
        if not stats:
            return stats

        results_mask = ap_per_class(
            stats["tp_m"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Mask",
            target_areas=None,        # ADD: No size-based AP for masks
            matched_gt_idx=None,
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)

        conf_thr = self._conf_for_per_image()
        self.per_image["mask"] = self._compute_per_image_prf_masks(stats, conf_thr, iou_index=0)
        return stats
    
    def _compute_per_image_prf_masks(self, stats, conf_thr: float, iou_index: int = 0) -> dict[str, list]:
        """
        Compute per-image precision, recall, and F2 score for segmentation masks.
        
        Args:
            stats: Dictionary of concatenated statistics arrays
            conf_thr: Confidence threshold to filter predictions
            iou_index: Index into tp_m array (0 for IoU=0.50, 1 for IoU=0.75, etc.)
        
        Returns:
            Dictionary with lists of per-image metrics (same structure as _compute_per_image_prf)
        
        Note: Uses stats["tp_m"] (mask TPs) instead of stats["tp"] (box TPs)
        """
        # Filter predictions by confidence threshold
        pred_keep = stats["conf"] >= conf_thr
        
        # Get all unique image IDs that have either predictions or ground truth
        img_ids = np.union1d(stats["pred_img"][pred_keep], stats["target_img_names"])
        
        # Initialize output dictionary
        out = {"image_id": [], "tp": [], "fp": [], "fn": [], "precision": [], "recall": [], "f2": []}
        
        # Compute metrics for each image independently
        for gid in img_ids:
            # Filter predictions for this image
            pid = (stats["pred_img"] == gid) & pred_keep
            
            # Extract mask TP values for this image at the specified IoU threshold
            tp_vec = stats["tp_m"][pid, iou_index] if pid.any() else np.zeros((0,), dtype=np.float32)
            
            # Count true positives, false positives, and false negatives
            tp = int(tp_vec.sum())
            fp = int(pid.sum() - tp)  # Total predictions minus TPs
            fn = int((stats["target_img_names"] == gid).sum() - tp)  # Total GTs minus TPs
            
            # Compute precision, recall, and F2 score with zero-division handling
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) else 0.0
            
            # Store results
            out["image_id"].append(str(gid))
            out["tp"].append(tp)
            out["fp"].append(fp)
            out["fn"].append(fn)
            out["precision"].append(float(p))
            out["recall"].append(float(r))
            out["f2"].append(float(f2))
        return out

    @property
    def keys(self) -> list[str]:
        """
        Return a list of all metric keys including detection and segmentation metrics.

        Returns:
            (list[str]): List of metric keys in the format 'metrics/<metric_name>' for logging and display.
                Keys include standard detection metrics plus mask-based metrics:
                - Detection: precision(B), recall(B), mAP50(B), mAP50-95(B)
                - Segmentation: precision(M), recall(M), mAP50(M), mAP50-95(M), f2(M), dice(M), mIoU(M), boundaryF1(M)

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> keys = metrics.keys
            >>> print(keys[-3:])  # Print last 3 keys
            ['metrics/dice(M)', 'metrics/mIoU(M)', 'metrics/boundaryF1(M)']
        """
        return DetMetrics.keys.fget(self) + [
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
            "metrics/f1(M)",
            "metrics/f2(M)",
            "metrics/dice(M)",
            "metrics/mIoU(M)",
            "metrics/boundaryF1(M)",
            "metrics/boundaryIoU(M)",
        ]

    def mean_results(self) -> list[float]:
        """
        Return mean results for all detection and segmentation metrics.

        Combines mean results from detection metrics (bounding boxes) with segmentation-specific metrics
        including mask precision/recall, F2, Dice, mIoU, and boundary F1. Used for logging and comparison.

        Returns:
            (list[float]): List of mean metric values in the order corresponding to self.keys property.
                Typically includes: [box_metrics..., mask_p, mask_r, mask_map50, mask_map, mask_f2, dice, miou, bf1].

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> # ... process validation results ...
            >>> results = metrics.mean_results()
            >>> print(f"Mean Dice: {results[-3]:.3f}, Mean IoU: {results[-2]:.3f}, Mean BF1: {results[-1]:.3f}")
        """
        return DetMetrics.mean_results(self) + self.seg.mean_results() + [
            self.seg.mf1,
            self.seg.mf2, 
            self.mdice, 
            self.miou, 
            self.mbf1,
            self.mboundary_iou
        ]

    def class_result(self, i: int) -> list[float]:
        """
        Return detection and segmentation metric results for a specific class.

        Args:
            i (int): Class index for which to retrieve results.

        Returns:
            (list[float]): List of metric values for the specified class, combining both detection (box)
                and segmentation (mask) results.

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> # ... process results ...
            >>> cat_metrics = metrics.class_result(0)
            >>> print(f"Cat AP: {cat_metrics[2]:.3f}")
        """
        return DetMetrics.class_result(self, i) + self.seg.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """
        Return mean Average Precision (mAP) values for both detection and segmentation.

        Returns:
            (np.ndarray): Array containing mAP values for bounding boxes and masks across different IoU thresholds.

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> maps = metrics.maps
            >>> print(f"Box mAP@50: {maps[0]:.3f}, Mask mAP@50: {maps[-1]:.3f}")
        """
        return np.concatenate([DetMetrics.maps.fget(self), self.seg.maps])

    @property
    def fitness(self) -> float:
        """
        Calculate overall fitness score combining detection and segmentation performance.
        
        Computes a weighted combination of box metrics (precision, recall, mAP), mask metrics (precision, recall, mAP),
        and additional segmentation metrics (Dice, mIoU, boundary F1) based on configured fitness_weights. The result
        is normalized by the sum of weights. If no weights are configured, falls back to averaging box and mask mAP50-95.

        Returns:
            (float): Combined fitness score in range [0.0, 1.0].

        Examples:
            >>> metrics = SegmentMetrics(names={0: 'person', 1: 'car'})
            >>> # ... process validation results ...
            >>> fitness = metrics.fitness
            >>> print(f"Segmentation fitness: {fitness:.3f}")
        """
        # Map weight keys to actual metric values
        metrics_map = {
            # Box metrics
            'precision': self.box.mp,
            'recall': self.box.mr,
            'mAP50': self.box.map50,
            'mAP50_95': self.box.map,
            'f1': self.box.mf1,
            'f2': self.box.mf2,
            # Mask metrics
            'mask_precision': self.seg.mp,
            'mask_recall': self.seg.mr,
            'mask_mAP50': self.seg.map50,
            'mask_mAP50_95': self.seg.map,
            'mask_f1': self.seg.mf1,
            'mask_f2': self.seg.mf2,
            # Additional segmentation metrics
            'dice': self.mdice,
            'miou': self.miou,
            'boundary_f1': self.mbf1,
            'boundary_iou': self.mboundary_iou
        }

        # print("-"*50)
        # print("Inside metrics.py 2290")
        # print(self.fitness_weights)
        # print("-"*50)
        
        fitness = 0.0
        total_weight = 0.0
        
        for key, weight in self.fitness_weights.items():
            if weight > 0 and key in metrics_map:
                metric_value = metrics_map[key]
                if not np.isnan(metric_value):
                    fitness += weight * metric_value
                    total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            return fitness / total_weight
        
        # Fallback: average of box and mask mAP50-95
        return (self.box.map + self.seg.map) / 2

    @property
    def curves(self) -> list[str]:
        """
        Return names of all available metric curves for plotting.

        Returns:
            (list[str]): List of curve names including both detection and segmentation curves:
                - Detection: Precision-Recall(B), F1-Confidence(B), etc.
                - Segmentation: Precision-Recall(M), F1-Confidence(M), F2-Confidence(M), etc.

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat'})
            >>> curves = metrics.curves
            >>> print(f"Available curves: {curves}")
        """
        return DetMetrics.curves.fget(self) + [
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "F2-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """
        Return curve data for all detection and segmentation metrics.

        Returns:
            (list[list]): Nested list containing curve data points for plotting precision-recall,
                F1-confidence, and other metric curves.

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> curve_data = metrics.curves_results
            >>> # Plot curves using curve_data
        """
        return DetMetrics.curves_results.fget(self) + self.seg.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """
        Generate a formatted summary of detection and segmentation metrics for each class.

        Args:
            normalize (bool): Whether to normalize metric values. Default is True.
            decimals (int): Number of decimal places for rounding metric values. Default is 5.

        Returns:
            (list[dict[str, Any]]): List of dictionaries, one per class, containing metric names and values.
                Each dictionary includes class name, detection metrics (Box-P, Box-R, etc.), and
                segmentation metrics (Mask-P, Mask-R, Mask-F1, Mask-F2).

        Example:
            >>> metrics = SegmentMetrics(names={0: 'cat', 1: 'dog'})
            >>> summary = metrics.summary()
            >>> for cls_metrics in summary:
            ...     print(f"{cls_metrics['Class']}: Mask-F1={cls_metrics['Mask-F1']:.3f}")
        """
        per_class = {"Mask-P": self.seg.p, "Mask-R": self.seg.r, "Mask-F1": self.seg.f1, "Mask-F2": self.seg.f2}
        summary = DetMetrics.summary(self, normalize, decimals)
        for i, s in enumerate(summary):
            s.update({k: round(v[i], decimals) for k, v in per_class.items()})
        return summary


class PoseMetrics(DetMetrics):
    """
    Calculate and aggregate detection and pose metrics over a given set of classes.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        pose (Metric): An instance of the Metric class to calculate pose metrics.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'pose'.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes, target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    Methods:
        process: Process the detection and pose metrics over the given set of predictions. R
        keys: Return a list of keys for accessing metrics.
        mean_results: Return the mean results of box and pose.
        class_result: Return the class-wise detection results for a specific class i.
        maps: Return the mean average precision (mAP) per class for both box and pose detections.
        fitness: Return combined fitness score for pose and box detection.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a summarized representation of per-class pose metrics as a list of dictionaries.
    """

    def __init__(self, names: dict[int, str] = {}) -> None:
        """
        Initialize the PoseMetrics class with directory path, class names, and plotting options.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        super().__init__(names)
        self.pose = Metric()
        self.task = "pose"
        self.stats["tp_p"] = []  # add additional stats for pose

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """
        Process the detection and pose metrics over the given set of predictions.

        Args:
            save_dir (Path): Directory to save plots. Defaults to Path(".").
            plot (bool): Whether to plot precision-recall curves. Defaults to False.
            on_plot (callable, optional): Function to call after plots are generated.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing concatenated statistics arrays.
        """
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)  # process box stats
        results_pose = ap_per_class(
            stats["tp_p"],
            stats["conf"],
            stats["pred_cls"],
            stats["target_cls"],
            plot=plot,
            on_plot=on_plot,
            save_dir=save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        return stats

    @property
    def keys(self) -> list[str]:
        """Return a list of evaluation metric keys."""
        return DetMetrics.keys.fget(self) + [
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self) -> list[float]:
        """Return the mean results of box and pose."""
        return DetMetrics.mean_results(self) + self.pose.mean_results()

    def class_result(self, i: int) -> list[float]:
        """Return the class-wise detection results for a specific class i."""
        return DetMetrics.class_result(self, i) + self.pose.class_result(i)

    @property
    def maps(self) -> np.ndarray:
        """Return the mean average precision (mAP) per class for both box and pose detections."""
        return DetMetrics.maps.fget(self) + self.pose.maps

    @property
    def fitness(self) -> float:
        """Return combined fitness score for pose and box detection."""
        return self.pose.fitness() + DetMetrics.fitness.fget(self)

    @property
    def curves(self) -> list[str]:
        """Return a list of curves for accessing specific metrics curves."""
        return DetMetrics.curves.fget(self) + [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self) -> list[list]:
        """Return a list of computed performance metrics and statistics."""
        return DetMetrics.curves_results.fget(self) + self.pose.curves_results

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, Any]]:
        """
        Generate a summarized representation of per-class pose metrics as a list of dictionaries. Includes both box and
        pose scalar metrics (mAP, mAP50, mAP75) alongside precision, recall, and F1-score for each class.

        Args:
            normalize (bool): For Pose metrics, everything is normalized  by default [0-1].
            decimals (int): Number of decimal places to round the metrics values to.

        Returns:
            (list[dict[str, Any]]): A list of dictionaries, each representing one class with corresponding metric values.

        Examples:
            >>> results = model.val(data="coco8-pose.yaml")
            >>> pose_summary = results.summary(decimals=4)
            >>> print(pose_summary)
        """
        per_class = {
            "Pose-P": self.pose.p,
            "Pose-R": self.pose.r,
            "Pose-F1": self.pose.f1,
        }
        summary = DetMetrics.summary(self, normalize, decimals)  # get box summary
        for i, s in enumerate(summary):
            s.update({**{k: round(v[i], decimals) for k, v in per_class.items()}})
        return summary


class ClassifyMetrics(SimpleClass, DataExportMixin):
    """
    Class for computing classification metrics including top-1 and top-5 accuracy.

    Attributes:
        top1 (float): The top-1 accuracy.
        top5 (float): The top-5 accuracy.
        speed (dict): A dictionary containing the time taken for each step in the pipeline.
        task (str): The task type, set to 'classify'.

    Methods:
        process: Process target classes and predicted classes to compute metrics.
        fitness: Return mean of top-1 and top-5 accuracies as fitness score.
        results_dict: Return a dictionary with model's performance metrics and fitness score.
        keys: Return a list of keys for the results_dict property.
        curves: Return a list of curves for accessing specific metrics curves.
        curves_results: Provide a list of computed performance metrics and statistics.
        summary: Generate a single-row summary of classification metrics (Top-1 and Top-5 accuracy).
    """

    def __init__(self) -> None:
        """Initialize a ClassifyMetrics instance."""
        self.top1 = 0
        self.top5 = 0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "classify"

    def process(self, targets: torch.Tensor, pred: torch.Tensor):
        """
        Process target classes and predicted classes to compute metrics.

        Args:
            targets (torch.Tensor): Target classes.
            pred (torch.Tensor): Predicted classes.
        """
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self) -> float:
        """Return mean of top-1 and top-5 accuracies as fitness score."""
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self) -> dict[str, float]:
        """Return a dictionary with model's performance metrics and fitness score."""
        return dict(zip(self.keys + ["fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self) -> list[str]:
        """Return a list of keys for the results_dict property."""
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self) -> list:
        """Return a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self) -> list:
        """Return a list of curves for accessing specific metrics curves."""
        return []

    def summary(self, normalize: bool = True, decimals: int = 5) -> list[dict[str, float]]:
        """
        Generate a single-row summary of classification metrics (Top-1 and Top-5 accuracy).

        Args:
            normalize (bool): For Classify metrics, everything is normalized  by default [0-1].
            decimals (int): Number of decimal places to round the metrics values to.

        Returns:
            (list[dict[str, float]]): A list with one dictionary containing Top-1 and Top-5 classification accuracy.

        Examples:
            >>> results = model.val(data="imagenet10")
            >>> classify_summary = results.summary(decimals=4)
            >>> print(classify_summary)
        """
        return [{"top1_acc": round(self.top1, decimals), "top5_acc": round(self.top5, decimals)}]


class OBBMetrics(DetMetrics):
    """
    Metrics for evaluating oriented bounding box (OBB) detection.

    Attributes:
        names (dict[int, str]): Dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict[str, float]): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'obb'.
        stats (dict[str, list]): A dictionary containing lists for true positives, confidence scores, predicted classes, target classes, and target images.
        nt_per_class: Number of targets per class.
        nt_per_image: Number of targets per image.

    References:
        https://arxiv.org/pdf/2106.06072.pdf
    """

    def __init__(self, names: dict[int, str] = {}, fitness_weights: dict = None) -> None:
        """
        Initialize an OBBMetrics instance with directory, plotting, and class names.

        Args:
            names (dict[int, str], optional): Dictionary of class names.
        """
        DetMetrics.__init__(self, names, fitness_weights = fitness_weights)
        # TODO: probably remove task as well
        self.task = "obb"
