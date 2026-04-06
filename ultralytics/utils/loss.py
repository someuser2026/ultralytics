# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, regularize_rboxes, xywhr2xyxyxyxy, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist
# --- soft-ignore helpers (ADD) -----------------------------------------------
from typing import Optional
import math

try:
    import numpy as np
    import cv2
    from scipy.ndimage import distance_transform_edt
    _HAS_SCI_CV = True
except Exception:
    _HAS_SCI_CV = False

def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float32, non_blocking=True)


@torch.no_grad()
def create_soft_ignore_weights_fast(gt_masks: torch.Tensor,
                                    ignore_width: float,
                                    transition_ratio: float = 0.5,
                                    device: torch.device | None = None) -> torch.Tensor:
    """
    Accurate version (EDT). Accepts (H,W) or (N,H,W). Returns (N,H,W).
    Creates soft ignore band OUTSIDE the mask boundary.
    """
    if device is None:
        device = gt_masks.device
    if ignore_width <= 0:
        return gt_masks.new_ones((*((1,) if gt_masks.ndim == 2 else ()), *gt_masks.shape[-2:])).expand_as(
            gt_masks if gt_masks.ndim == 3 else gt_masks.unsqueeze(0)
        ).to(dtype=torch.float32)

    x = (gt_masks > 0.5).to(torch.uint8)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    N, H, W = x.shape

    if not _HAS_SCI_CV:
        return create_soft_ignore_weights_torch(x.to(torch.float32), ignore_width, transition_ratio, device=device)

    import numpy as np, cv2
    from scipy.ndimage import distance_transform_edt

    masks_np = x.detach().to('cpu', torch.uint8).numpy()
    weight_maps = np.ones((N, H, W), dtype=np.float32)

    hard_w = int(np.ceil(ignore_width * max(0.0, min(1.0, transition_ratio))))
    trans_w = max(0, int(np.ceil(ignore_width - hard_w)))
    sigma = max(trans_w / 3.0, 0.5) if trans_w > 0 else 1.0

    for i in range(N):
        m = masks_np[i]
        if m.sum() == 0:
            continue
        
        # KEY FIX: Calculate distance from OUTSIDE the mask boundary
        # dist_inside: distance from edge going inward (positive inside mask)
        # dist_outside: distance from edge going outward (positive outside mask)
        dist_inside = distance_transform_edt(m).astype(np.float32)
        dist_outside = distance_transform_edt(1 - m).astype(np.float32)
        
        # Start with all weights = 1.0
        w = np.ones_like(m, dtype=np.float32)
        
        # Apply ignore band OUTSIDE the mask
        # Hard ignore zone: 0 to hard_w pixels outside
        if hard_w > 0:
            outside_hard = (m == 0) & (dist_outside <= hard_w)
            w[outside_hard] = 0.0
        
        # Soft transition zone: hard_w to (hard_w + trans_w) pixels outside
        if trans_w > 0:
            outside_soft = (m == 0) & (dist_outside > hard_w) & (dist_outside <= hard_w + trans_w)
            if outside_soft.any():
                d = dist_outside[outside_soft] - hard_w
                # Gaussian transition from 0 to 1
                w[outside_soft] = 1.0 - np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        
        # Optional: Also apply smaller ignore band INSIDE near boundary
        # (for uncertain boundary pixels)
        inner_margin = min(2, hard_w // 2) if hard_w > 0 else 0
        if inner_margin > 0:
            inside_band = (m == 1) & (dist_inside <= inner_margin)
            if inside_band.any():
                # Soft transition from edge to core
                d_in = dist_inside[inside_band]
                w[inside_band] = d_in / inner_margin  # Linear: 0 at edge, 1 at inner_margin
        
        weight_maps[i] = w

    return torch.from_numpy(weight_maps).to(device=device, dtype=torch.float32)

@torch.no_grad()
def create_soft_ignore_weights_torch(gt_masks: torch.Tensor,
                                     ignore_width: float,
                                     transition_ratio: float = 0.5,
                                     device: torch.device | None = None) -> torch.Tensor:
    """
    Pure-Torch approximation. Accepts (H,W) or (N,H,W). Returns (N,H,W).
    Creates soft ignore band OUTSIDE the mask boundary.
    """
    if device is None:
        device = gt_masks.device
    x = (gt_masks >= 0.5).to(torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    N, H, W = x.shape

    if ignore_width <= 0:
        return torch.ones((N, H, W), device=device, dtype=torch.float32)

    hard_w = max(1, int(math.ceil(ignore_width * max(0.0, min(1.0, transition_ratio)))))
    trans_w = max(0, int(math.ceil(ignore_width - hard_w)))

    def dilate(bin_mask: torch.Tensor, r: int) -> torch.Tensor:
        """Dilate binary mask by r pixels."""
        if r <= 0:
            return bin_mask
        k = 2 * r + 1
        # Use max pooling to dilate
        dilated = torch.nn.functional.max_pool2d(
            bin_mask.unsqueeze(1),  # (N,1,H,W)
            kernel_size=k,
            stride=1,
            padding=r
        )
        return dilated.squeeze(1)  # (N,H,W)

    # Start with all weights = 1.0
    weights = torch.ones((N, H, W), device=device, dtype=torch.float32)
    
    # Hard ignore zone: dilate mask by hard_w, then subtract original
    if hard_w > 0:
        expanded_hard = dilate(x, hard_w)
        hard_ignore_zone = (expanded_hard > 0.5) & (x < 0.5)  # Outside original, inside expanded
        weights[hard_ignore_zone] = 0.0
    
    # Soft transition zone: between hard_w and (hard_w + trans_w)
    if trans_w > 0:
        expanded_soft = dilate(x, hard_w + trans_w)
        soft_zone = (expanded_soft > 0.5) & (x < 0.5)
        if hard_w > 0:
            soft_zone = soft_zone & ~hard_ignore_zone  # Exclude hard ignore zone
        
        if soft_zone.any():
            # Approximate gradient: use multiple dilations
            steps = min(5, trans_w)
            step_size = max(1, trans_w // steps)
            
            for s in range(1, steps + 1):
                r = hard_w + s * step_size
                expanded_s = dilate(x, r)
                band_s = (expanded_s > 0.5) & (x < 0.5) & soft_zone
                # Linear gradient in transition zone
                weights[band_s] = torch.maximum(weights[band_s], torch.tensor(s / steps, device=device))
    
    # Optional: Small ignore band INSIDE near boundary (for uncertain pixels)
    inner_margin = min(2, hard_w // 2) if hard_w > 0 else 0
    if inner_margin > 0:
        # Erode mask to find core
        def erode(bin_mask: torch.Tensor, r: int) -> torch.Tensor:
            if r <= 0:
                return bin_mask
            k = 2 * r + 1
            inv = 1.0 - bin_mask.unsqueeze(1)
            mp = torch.nn.functional.max_pool2d(inv, kernel_size=k, stride=1, padding=r)
            return (mp == 0).to(torch.float32).squeeze(1)
        
        core = erode(x, inner_margin)
        inner_band = (x > 0.5) & (core < 0.5)
        if inner_band.any():
            # Linear transition from 0.5 at edge to 1.0 at core
            weights[inner_band] = 0.7  # Simple approximation

    return weights

# ----------------------------------------------------------------------------- 


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing
    on hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


def _cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute 2D cross products."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _points_in_rboxes(points: torch.Tensor, rboxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Check whether points lie inside rotated boxes of matching batch length."""
    ctr = rboxes[:, None, :2]
    wh = rboxes[:, None, 2:4].clamp_min(eps)
    angle = rboxes[:, None, 4:5]
    offset = points - ctr
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    local_x = offset[..., 0:1] * cos_a + offset[..., 1:2] * sin_a
    local_y = -offset[..., 0:1] * sin_a + offset[..., 1:2] * cos_a
    return (
        (local_x.abs() <= wh[..., 0:1] / 2 + eps) &
        (local_y.abs() <= wh[..., 1:2] / 2 + eps)
    ).squeeze(-1)


def _segment_intersections(poly1: torch.Tensor, poly2: torch.Tensor, eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all pairwise edge intersections for two quadrilaterals."""
    p1 = poly1
    p2 = torch.roll(poly1, shifts=-1, dims=1)
    q1 = poly2
    q2 = torch.roll(poly2, shifts=-1, dims=1)

    p1 = p1[:, :, None, :]
    p2 = p2[:, :, None, :]
    q1 = q1[:, None, :, :]
    q2 = q2[:, None, :, :]

    r = p2 - p1
    s = q2 - q1
    qp = q1 - p1
    denom = _cross2d(r, s)
    safe_denom = torch.where(denom.abs() < eps, torch.ones_like(denom), denom)
    t = _cross2d(qp, s) / safe_denom
    u = _cross2d(qp, r) / safe_denom
    valid = (denom.abs() > eps) & (t >= -eps) & (t <= 1 + eps) & (u >= -eps) & (u <= 1 + eps)
    intersections = p1 + t.unsqueeze(-1) * r
    return intersections.reshape(intersections.shape[0], -1, 2), valid.reshape(valid.shape[0], -1)


def rotated_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute differentiable IoU for elementwise rotated box pairs."""
    if boxes1.shape != boxes2.shape:
        raise ValueError(f"rotated_box_iou expects matching shapes, got {boxes1.shape} and {boxes2.shape}.")
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))

    corners1 = xywhr2xyxyxyxy(boxes1)
    corners2 = xywhr2xyxyxyxy(boxes2)
    inside1 = _points_in_rboxes(corners1, boxes2, eps=eps)
    inside2 = _points_in_rboxes(corners2, boxes1, eps=eps)
    intersections, inter_mask = _segment_intersections(corners1, corners2, eps=eps)

    candidates = torch.cat((corners1, corners2, intersections), dim=1)
    valid_mask = torch.cat((inside1, inside2, inter_mask), dim=1)
    valid_count = valid_mask.sum(dim=1)

    centroid = (candidates * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_count.clamp_min(1).unsqueeze(-1)
    angles = torch.atan2(candidates[..., 1] - centroid[:, None, 1], candidates[..., 0] - centroid[:, None, 0])
    angles = angles.masked_fill(~valid_mask, float("inf"))
    sort_idx = angles.argsort(dim=1)
    sorted_pts = candidates.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
    sorted_mask = valid_mask.gather(1, sort_idx)

    first_valid = sorted_pts[:, :1]
    sorted_pts = torch.where(sorted_mask.unsqueeze(-1), sorted_pts, first_valid.expand_as(sorted_pts))
    rolled = torch.roll(sorted_pts, shifts=-1, dims=1)
    inter_area = 0.5 * (_cross2d(sorted_pts, rolled).sum(dim=1)).abs()
    inter_area = torch.where(valid_count >= 3, inter_area, inter_area.new_zeros(inter_area.shape))

    area1 = boxes1[:, 2].clamp_min(0) * boxes1[:, 3].clamp_min(0)
    area2 = boxes2[:, 2].clamp_min(0) * boxes2[:, 3].clamp_min(0)
    union = (area1 + area2 - inter_area).clamp_min(eps)
    return (inter_area / union).clamp_(0.0, 1.0)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    @staticmethod
    def _get_batch_cls_probs(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-target class probabilities with fallback to hard labels."""
        cls = batch["cls"].view(-1, 1)
        return batch.get("cls_probs", torch.ones_like(cls)).view(-1, 1).to(device=cls.device, dtype=cls.dtype)

    @staticmethod
    def _gather_assigned_cls_probs(
        gt_probs: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gather per-anchor label probabilities for assigned ground truths."""
        if gt_probs.numel() == 0 or gt_probs.shape[1] == 0:
            return torch.zeros_like(target_gt_idx, dtype=gt_probs.dtype, device=target_gt_idx.device)
        bs, n_max_boxes, _ = gt_probs.shape
        batch_ind = torch.arange(bs, device=gt_probs.device)[:, None]
        matched_probs = gt_probs.squeeze(-1)[batch_ind, target_gt_idx.clamp(0, n_max_boxes - 1)]
        return torch.where(fg_mask.bool(), matched_probs, torch.zeros_like(matched_probs))

    @staticmethod
    def _scale_target_scores_with_probs(
        target_scores: torch.Tensor,
        gt_probs: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scale assigned target scores by ground-truth class probabilities."""
        matched_probs = v8DetectionLoss._gather_assigned_cls_probs(gt_probs, target_gt_idx, fg_mask)
        return target_scores * matched_probs.unsqueeze(-1).to(target_scores.dtype)

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        cls_probs = self._get_batch_cls_probs(batch)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"], cls_probs), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes, gt_probs = targets.split((1, 4, 1), 2)  # cls, xyxy, prob
        gt_probs = gt_probs.clamp_(0.0, 1.0)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores = self._scale_target_scores_with_probs(target_scores, gt_probs, target_gt_idx, fg_mask)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation (IoU/Jaccard surrogate).

    This module implements the binary Lovasz extension on *logits* as introduced in:
        Berman et al., "The Lovasz-Softmax loss: A tractable surrogate for the optimization of the IoU measure".

    It is well-suited for tasks where boundary exactness is ambiguous and IoU alignment is desired.

    Args:
        ignore_index (int, optional): Label value to ignore in the loss. Defaults to -100.
        per_image (bool, optional): If True, compute loss per instance/map then return a vector (N,).
                                    If False, expects flattened (P,) inputs. Defaults to True.
    Returns:
        torch.Tensor: If `per_image=True`, returns a tensor of shape (N,) with the loss per instance.
                      If `per_image=False`, returns a scalar tensor.

    Notes:
        * Input must be raw logits (no sigmoid). Targets must be {0,1} or {0,1,ignore_index}.
        * For multi-class segmentation prefer the Lovasz-Softmax variant; for binary masks per instance,
          Lovasz-Hinge is typically preferred and lighter-weight.
    """

    def __init__(self, ignore_index: int = -100, per_image: bool = True):
        super().__init__()
        self.ignore_index = ignore_index
        self.per_image = per_image

    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        """Compute gradient of the Lovasz extension w.r.t. sorted errors."""
        p = gt_sorted.sum()
        if p == 0:
            return gt_sorted.new_zeros(gt_sorted.numel())
        intersection = p - gt_sorted.cumsum(0)
        union = p + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if gt_sorted.numel() > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard

    def _flat(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple:
        """Flatten logits/targets and drop ignore_index."""
        logits = logits.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        if self.ignore_index is not None:
            valid = targets != self.ignore_index
            return logits[valid], targets[valid]
        return logits, targets

    def _lovasz_hinge_flat(self, logits_flat: torch.Tensor, targets_flat: torch.Tensor) -> torch.Tensor:
        """Binary Lovasz-Hinge on flattened logits/targets."""
        if targets_flat.numel() == 0:
            return logits_flat.new_tensor(0.0)
        # Map {0,1} -> {-1,+1} margins
        signs = 2.0 * targets_flat.float() - 1.0
        errors = 1.0 - logits_flat * signs  # margin errors
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = targets_flat[perm]
        grad = self._lovasz_grad(gt_sorted)
        return torch.dot(F.relu(errors_sorted), grad)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Lovasz-Hinge loss.

        Args:
            logits (Tensor): Raw logits. If `per_image=True`, shape (N, H, W).
                             If `per_image=False`, provide flattened (P,).
            targets (Tensor): Binary targets with same shape, values in {0,1} or ignore_index.

        Returns:
            Tensor: Loss per instance (N,) if `per_image=True`; else a scalar tensor.
        """
        if self.per_image:
            assert logits.dim() == 3 and targets.dim() == 3, "Expect (N,H,W) when per_image=True."
            N = logits.shape[0]
            out = logits.new_zeros(N)
            for i in range(N):
                l_flat, y_flat = self._flat(logits[i], targets[i])
                out[i] = self._lovasz_hinge_flat(l_flat, y_flat)
            return out
        # flattened mode
        l_flat, y_flat = self._flat(logits, targets)
        return self._lovasz_hinge_flat(l_flat, y_flat)


def dice_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Dice loss computed on probabilities derived from logits.

    Args:
        logits (Tensor): Raw logits of shape (N, H, W).
        targets (Tensor): Binary targets (N, H, W).
        eps (float): Epsilon for numerical stability.

    Returns:
        Tensor: Per-instance Dice loss of shape (N,).
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1, 2))
    den = (probs * probs).sum(dim=(1, 2)) + (targets * targets).sum(dim=(1, 2)) + eps
    return 1.0 - (num + eps) / den


class MixedMaskLoss(nn.Module):
    """
    Mixed segmentation loss for proto-head models:
    Lovasz-Hinge (IoU surrogate) + Dice (with logits) + BCE-with-logits.

    Designed to operate on proto-assembled, cropped per-instance logits.

    Args:
        w_lovasz (float): Weight for Lovasz-Hinge term. Default: 1.0
        w_dice (float): Weight for Dice term. Default: 0.3
        w_bce (float): Weight for BCE-with-logits term. Default: 0.2
        ignore_index (int): Ignore label for targets. Default: -100
        area_normalize (bool): If True, divide each instance loss by its area to prevent domination by large objects.
                               Default: True
        lovasz (LovaszHingeLoss | None): Optional external Lovasz module; if None, a default is constructed.

    Forward Args:
        logits (Tensor): Assembled instance logits, shape (N, H, W), raw (no sigmoid).
        targets (Tensor): Binary targets, shape (N, H, W).
        xyxy (Tensor): Instance boxes in mask-space for cropping, shape (N, 4).
        area (Tensor): Per-instance areas, shape (N,).
        crop_mask_fn (Callable): Function(tensor, xyxy) -> cropped tensor with shape (N, h, w).

    Returns:
        Tensor: Scalar total loss (sum over instances after weighting/normalisation).

    Notes:
        * Keep BCE/Dice weights modest: Lovasz drives IoU alignment; Dice/BCE stabilise early training.
        * Works seamlessly with Ultralytics' proto head where masks are built via einsum(coeffs, prototypes).
    """

    def __init__(
        self,
        w_lovasz: float = 1.0,
        w_dice: float = 0.3,
        w_bce: float = 0.2,
        ignore_index: int = -100,
        area_normalize: bool = True,
        lovasz: Optional[LovaszHingeLoss] = None,
    ):
        super().__init__()
        self.w_lovasz = float(w_lovasz)
        self.w_dice = float(w_dice)
        self.w_bce = float(w_bce)
        self.area_normalize = area_normalize
        self.ignore_index = ignore_index
        self.lovasz = lovasz if lovasz is not None else LovaszHingeLoss(ignore_index=ignore_index, per_image=True)

    def forward(
        self,
        logits: torch.Tensor,          # (N, H, W), raw logits
        targets: torch.Tensor,         # (N, H, W), {0,1}
        xyxy: torch.Tensor,            # (N, 4), in mask-space
        area: torch.Tensor,            # (N,)
        crop_mask_fn: callable,        # callable(tensor, xyxy) -> (N, h, w)
        weight_map: Optional[torch.Tensor] = None,  # (N, H, W) in [0,1], optional (NEW)
    ) -> torch.Tensor:

        if logits.numel() == 0:
            return logits.new_tensor(0.0)

        # Crop to instance supports
        logits_c = crop_mask_fn(logits, xyxy)          # (N, h, w)
        targets_c = crop_mask_fn(targets, xyxy)        # (N, h, w)

        # Optionally crop weights
        if weight_map is not None:
            weights_c = crop_mask_fn(weight_map, xyxy) # (N, h, w)
            # avoid zero division later
            valid_w = weights_c.sum(dim=(1,2)).clamp_min(1e-6)
        else:
            weights_c = None

        # (1) Lovasz-Hinge per instance (N,)
        # Approximation: evaluate Lovasz only on "core" (high-confidence) pixels if weight_map is provided
        if weight_map is not None:
            core_mask = (weights_c >= 0.90).to(logits_c.dtype)
            # mark non-core as ignore by setting targets to ignore_index
            t_for_lovasz = targets_c.clone()
            if core_mask.any():
                t_for_lovasz[core_mask < 0.5] = self.lovasz.ignore_index
            lovasz_vec = self.lovasz(logits_c, t_for_lovasz)  # per-instance
        else:
            lovasz_vec = self.lovasz(logits_c, targets_c)
        
        # boost positive contribution and clamp bg to reduce pos-neg imbalance
        # only valid for soft ignore boundary loss
        pos_boost = 2.0
        bg_cap = 0.5
        if weights_c is not None:
            w = weights_c.clone()
            pos = targets_c > 0.5
            w_pos = w * pos_boost
            w_bg = torch.clamp_max(w, bg_cap)
            w_eff = torch.where(pos, w_pos, w_bg)
            # print("-"*50)
            # print("Inisde utils/loss.py 627")
            # print("weff:", w_eff)
            # print("wc:", weights_c)
            # print("target:", (targets_c[0, :, :] > 0.5).to(w.dtype).mean())
            # print("-"*50)

        # (2) Weighted Dice per instance (N,)
        probs = torch.sigmoid(logits_c)
        if weights_c is not None:
            num = 2.0 * (w_eff * probs * targets_c).sum(dim=(1, 2))
            den = (w_eff * probs * probs).sum(dim=(1, 2)) + (w_eff * targets_c * targets_c).sum(dim=(1, 2)) + 1e-6
            dice_vec = 1.0 - (num + 1e-6) / den
        else:
            dice_vec = dice_loss_with_logits(logits_c, targets_c)

        # (3) Weighted BCE per instance (N,)
        if weights_c is not None:
            bce_map = torch.nn.functional.binary_cross_entropy_with_logits(logits_c, targets_c, reduction="none")
            bce_sum = (bce_map * w_eff).sum(dim=(1, 2))
            bce_vec = bce_sum / valid_w
        else:
            # existing per-instance BCE
            N = logits_c.shape[0]
            bce_vals = []
            for i in range(N):
                bce_vals.append(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits_c[i:i+1, ...], targets_c[i:i+1, ...], reduction="mean"
                    )
                )
            bce_vec = torch.stack(bce_vals) if bce_vals else logits_c.new_zeros(1)

        # Optional area normalization (unchanged)
        if self.area_normalize:
            norm = area.clamp_min(1e-6)
            lovasz_vec = lovasz_vec / norm
            dice_vec = dice_vec / norm
            bce_vec = bce_vec / norm

        per_instance = self.w_lovasz * lovasz_vec + self.w_dice * dice_vec + self.w_bce * bce_vec
        return per_instance.sum()


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

        self.use_mixed_loss = getattr(model.args, "seg_use_mixed_loss", False)
        
        if self.use_mixed_loss:
            self.mixed_mask_loss = MixedMaskLoss(
                w_lovasz=float(getattr(model.args, "seg_w_lovasz", 1.0)),
                w_dice=float(getattr(model.args, "seg_w_dice", 0.3)),
                w_bce=float(getattr(model.args, "seg_w_bce", 0.2)),
                ignore_index=int(getattr(model.args, "seg_ignore_index", -100)),
                area_normalize=bool(getattr(model.args, "seg_area_normalize", True)),
            )
        
        # --- soft ignore config (ADD) ---
        self.use_soft_ignore_band = bool(getattr(model.args, "use_soft_ignore_band", False))
        self.ignore_band_width = float(getattr(model.args, "ignore_band_width", 10.0))  # in pixels @ tile size
        self.soft_ignore_transition_ratio = float(getattr(model.args, "soft_ignore_transition_ratio", 0.5))
        self.tile_size = int(getattr(model.args, "imgsz", 224))  # reference
        self.use_ultrafast_ignore = bool(getattr(model.args, "use_ultrafast_ignore_band", False))

        # cache for weight maps within a forward pass
        # self._weight_map_cache: dict[int, torch.Tensor] = {}


    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        # self._weight_map_cache.clear()

        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[-2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            cls_probs = self._get_batch_cls_probs(batch)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"], cls_probs), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes, gt_probs = targets.split((1, 4, 1), 2)  # cls, xyxy, prob
            gt_probs = gt_probs.clamp_(0.0, 1.0)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores = self._scale_target_scores_with_probs(target_scores, gt_probs, target_gt_idx, fg_mask)
        assigned_probs = self._gather_assigned_cls_probs(gt_probs, target_gt_idx, fg_mask)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch_idx,
                proto,
                pred_masks,
                imgsz,
                self.overlap,
                fg_probs=assigned_probs,
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
        
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.mask_weight  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

    # @staticmethod
    def single_mask_loss(
        self,
        gt_mask: torch.Tensor,              # (n, H, W) float/binary
        pred: torch.Tensor,                 # (n, 32)
        proto: torch.Tensor,                # (32, H, W)
        xyxy: torch.Tensor,                 # (n, 4) in mask-space
        area: torch.Tensor,                 # (n,)
    ) -> torch.Tensor:

        # 1) assemble logits from prototypes
        # Assemble logits from prototypes
        C = proto.shape[0]
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto) / (C ** 0.5) # (Npos, H, W)
        Npos, H, W = pred_mask.shape

        # print("-"*50)
        # print("Inside single_mask_loss function in loss.py 857")
        # print("gt_mask.shape", gt_mask.shape)
        # print("pred.shape", pred.shape)
        # print("proto.shape", proto.shape)
        # print("xyxy.shape", xyxy.shape)
        # print("area.shape", area.shape)
        # print("gt sum", gt_mask.sum())
        # print("pred sum", pred.sum())
        # print("proto sum", proto.sum())
        # print("xyxy sum", xyxy.sum())
        # print("area sum", area.sum())
        # print("pred_mask.shape", pred_mask.shape)
        # print("pred_mask sum", pred_mask.sum())
        # print("-"*50)

        # Generate weight map on the same per-anchor gt_mask tensor
        weight_map = None
        if self.use_soft_ignore_band:
            proto_h = proto.shape[-2]
            scale = float(proto_h) / float(max(1, self.tile_size))
            scaled_ignore = max(0.0, self.ignore_band_width * scale)

            if _HAS_SCI_CV and not self.use_ultrafast_ignore:
                wm = create_soft_ignore_weights_fast(gt_mask, scaled_ignore, self.soft_ignore_transition_ratio, device=gt_mask.device)
            else:
                wm = create_soft_ignore_weights_torch(gt_mask, scaled_ignore, self.soft_ignore_transition_ratio, device=gt_mask.device)

            # Robust shape guard
            if wm.ndim != 3:
                wm = wm.view(Npos, H, W)
            if wm.shape[0] != Npos or wm.shape[-2:] != (H, W):
                # Fallback: per-instance generation
                wms = []
                for k in range(Npos):
                    gk = gt_mask[k]  # (H, W)
                    wmk = create_soft_ignore_weights_torch(gk, scaled_ignore, self.soft_ignore_transition_ratio, device=gt_mask.device)
                    wms.append(wmk.unsqueeze(0))
                wm = torch.cat(wms, dim=0)

            weight_map = wm.to(dtype=pred_mask.dtype)



        # 3) route to MixedMaskLoss if enabled; else legacy BCE path
        if getattr(self, "use_mixed_loss", False) and hasattr(self, "mixed_mask_loss"):
            return self.mixed_mask_loss(
                logits=pred_mask,
                targets=gt_mask.float(),
                xyxy=xyxy,
                area=area,
                crop_mask_fn=crop_mask,   # from ultralytics.utils.ops
                weight_map=weight_map,    # may be None
            )

        # 4) legacy BCE with optional weighting
        # … then proceed:
        loss_map = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        if weight_map is not None:
            loss_map = loss_map * weight_map
            cropped_loss = crop_mask(loss_map, xyxy)
            cropped_w    = crop_mask(weight_map, xyxy)
            valid_sum = cropped_w.sum(dim=(1, 2)).clamp_min(1e-6)
            return (cropped_loss.sum(dim=(1, 2)) / valid_sum / area.clamp_min(1e-6)).sum()
        else:
            return (crop_mask(loss_map, xyxy).mean(dim=(1, 2)) / area.clamp_min(1e-6)).sum()


    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
        fg_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.
            fg_probs (torch.Tensor | None): Per-anchor soft-label probabilities for positive anchors.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = proto.new_tensor(0.0)
        normalizer = proto.new_tensor(0.0)

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                img_loss = self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
                if fg_probs is not None:
                    img_weight = fg_probs[i][fg_mask_i].mean().clamp_min(0.0)
                else:
                    img_weight = img_loss.new_tensor(1.0)
                loss += img_loss * img_weight
                normalizer += fg_mask_i.sum() * img_weight

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / normalizer.clamp_min(1.0)


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        cls_probs = self._get_batch_cls_probs(batch)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"], cls_probs), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes, gt_probs = targets.split((1, 4, 1), 2)  # cls, xyxy, prob
        gt_probs = gt_probs.clamp_(0.0, 1.0)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores = self._scale_target_scores_with_probs(target_scores, gt_probs, target_gt_idx, fg_mask)
        assigned_probs = self._gather_assigned_cls_probs(gt_probs, target_gt_idx, fg_mask)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )
            kpt_weight = assigned_probs[fg_mask].mean().clamp_min(0.0)
            loss[1] *= kpt_weight
            loss[2] *= kpt_weight

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


def _get_cfg_value(cfg: Any, key: str, default: Any) -> Any:
    """Read a config value from either a dict-like object or a namespace."""
    if cfg is None:
        return default
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


def _points_to_grid(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert pixel coordinates to grid_sample coordinates with align_corners=False."""
    if width > 1:
        x = ((points[..., 0] + 0.5) / width) * 2 - 1
    else:
        x = torch.zeros_like(points[..., 0])
    if height > 1:
        y = ((points[..., 1] + 0.5) / height) * 2 - 1
    else:
        y = torch.zeros_like(points[..., 1])
    return torch.stack((x, y), dim=-1)


def _points_in_bounds(points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Return a boolean mask for points that fall inside the image plane."""
    return (
        (points[..., 0] >= 0)
        & (points[..., 0] <= max(width - 1, 0))
        & (points[..., 1] >= 0)
        & (points[..., 1] <= max(height - 1, 0))
    )


def _sample_map_at_points(map_tensor: torch.Tensor, points: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    """Sample a (B, 1, H, W) map at pixel coordinates shaped (B, N, 2)."""
    if map_tensor is None or points.numel() == 0:
        return None
    _, _, height, width = map_tensor.shape
    grid = _points_to_grid(points, height, width).unsqueeze(1)
    sampled = F.grid_sample(map_tensor.float(), grid, mode=mode, padding_mode="zeros", align_corners=False)
    return sampled[:, 0, 0, :]


def _compute_obb_spatial_prior_losses(
    pred_rboxes: torch.Tensor,
    conf_scores: torch.Tensor,
    negative_mask: torch.Tensor,
    land_water_mask: torch.Tensor | None,
    shoreline_distance_map: torch.Tensor | None,
    point_mode: str = "center",
    shoreline_prior_max_dist: float = 128.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute shoreline-distance and land-region penalties for unmatched OBB predictions."""
    zero = conf_scores.new_tensor(0.0)
    if conf_scores.ndim == 3 and conf_scores.shape[-1] == 1:
        conf_scores = conf_scores.squeeze(-1)
    if pred_rboxes.numel() == 0 or conf_scores.numel() == 0 or not negative_mask.any():
        return zero, zero
    if land_water_mask is None and shoreline_distance_map is None:
        return zero, zero

    negative_mask = negative_mask.bool()
    max_dist = max(float(shoreline_prior_max_dist), 1.0)

    land_penalty = zero
    shoreline_penalty = zero

    centers = pred_rboxes[..., :2]
    if land_water_mask is not None:
        _, _, height, width = land_water_mask.shape
        center_in_bounds = _points_in_bounds(centers, height, width)
        center_classes = _sample_map_at_points(land_water_mask.float(), centers, mode="nearest").round().long()
        land_indicator = ((center_classes == 1) & center_in_bounds).to(conf_scores.dtype)
        land_penalty = (conf_scores[negative_mask] * land_indicator[negative_mask]).mean()
    else:
        center_classes = None

    if shoreline_distance_map is None or land_water_mask is None:
        return shoreline_penalty, land_penalty

    _, _, height, width = shoreline_distance_map.shape
    if point_mode == "closest_corner":
        corners = xywhr2xyxyxyxy(pred_rboxes).view(pred_rboxes.shape[0], pred_rboxes.shape[1], 4, 2)
        flat_corners = corners.view(pred_rboxes.shape[0], -1, 2)
        corner_in_bounds = _points_in_bounds(flat_corners, height, width).view(pred_rboxes.shape[0], -1, 4)
        corner_classes = _sample_map_at_points(land_water_mask.float(), flat_corners, mode="nearest").round().long()
        corner_classes = corner_classes.view(pred_rboxes.shape[0], -1, 4)
        corner_distance = _sample_map_at_points(shoreline_distance_map.float(), flat_corners, mode="bilinear")
        corner_distance = corner_distance.view(pred_rboxes.shape[0], -1, 4)
        valid_corners = (corner_classes == 2) & corner_in_bounds
        corner_distance = torch.where(valid_corners, corner_distance, torch.full_like(corner_distance, float("inf")))
        min_corner_distance = corner_distance.min(dim=-1).values
        shoreline_weights = torch.where(
            valid_corners.any(dim=-1),
            min_corner_distance.clamp_(0.0, max_dist) / max_dist,
            torch.zeros_like(conf_scores),
        )
    else:
        if center_classes is None:
            center_classes = _sample_map_at_points(land_water_mask.float(), centers, mode="nearest").round().long()
        shoreline_distance = _sample_map_at_points(shoreline_distance_map.float(), centers, mode="bilinear")
        shoreline_weights = torch.where(
            (center_classes == 2) & _points_in_bounds(centers, height, width),
            shoreline_distance.clamp_(0.0, max_dist) / max_dist,
            torch.zeros_like(conf_scores),
        )

    shoreline_penalty = (conf_scores[negative_mask] * shoreline_weights[negative_mask]).mean()
    return shoreline_penalty, land_penalty


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)
        self.use_shoreline_prior_loss = bool(_get_cfg_value(model.args, "use_shoreline_prior_loss", False))
        self.use_land_water_prior_loss = bool(_get_cfg_value(model.args, "use_land_water_prior_loss", False))
        self.shoreline_prior_point_mode = _get_cfg_value(model.args, "shoreline_prior_point_mode", "center")
        self.shoreline_prior_weight = float(_get_cfg_value(model.args, "shoreline_prior_weight", 1.0))
        self.land_water_prior_weight = float(_get_cfg_value(model.args, "land_water_prior_weight", 1.0))
        self.shoreline_prior_max_dist = float(_get_cfg_value(model.args, "shoreline_prior_max_dist", 128.0))

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, targets.shape[1] - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), targets.shape[1] - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:].clone()
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, shoreline_prior, land_water_prior
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            cls_probs = self._get_batch_cls_probs(batch)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5), cls_probs), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes, gt_probs = targets.split((1, 5, 1), 2)  # cls, xywhr, prob
            gt_probs = gt_probs.clamp_(0.0, 1.0)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores = self._scale_target_scores_with_probs(target_scores, gt_probs, target_gt_idx, fg_mask)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        if self.use_shoreline_prior_loss or self.use_land_water_prior_loss:
            pred_bboxes_px = pred_bboxes.clone()
            pred_bboxes_px[..., :4] *= stride_tensor
            loss[3], loss[4] = _compute_obb_spatial_prior_losses(
                pred_bboxes_px,
                pred_scores.sigmoid().amax(-1),
                ~fg_mask,
                batch.get("land_water_mask"),
                batch.get("shoreline_distance_map"),
                point_mode=self.shoreline_prior_point_mode,
                shoreline_prior_max_dist=self.shoreline_prior_max_dist,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.shoreline_prior_weight
        loss[4] *= self.land_water_prior_weight

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, shoreline_prior, land_water_prior)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class RotatedFCOSLoss:
    """Loss for MMRotate-style Rotated FCOS heads."""

    def __init__(self, model):
        device = next(model.parameters()).device
        m = model.model[-1]
        self.device = device
        self.hyp = model.args
        self.nc = m.nc
        self.stride = m.stride
        self.regress_ranges = m.regress_ranges
        self.center_sampling = m.center_sampling
        self.center_sample_radius = m.center_sample_radius
        self.norm_on_bbox = m.norm_on_bbox
        self.bbox_loss_type = m.bbox_loss_type
        self.angle_mode = getattr(m, "angle_mode", getattr(model, "yaml", {}).get("angle_mode", "oc"))
        self.use_shoreline_prior_loss = bool(_get_cfg_value(model.args, "use_shoreline_prior_loss", False))
        self.use_land_water_prior_loss = bool(_get_cfg_value(model.args, "use_land_water_prior_loss", False))
        self.shoreline_prior_point_mode = _get_cfg_value(model.args, "shoreline_prior_point_mode", "center")
        self.shoreline_prior_weight = float(_get_cfg_value(model.args, "shoreline_prior_weight", 1.0))
        self.land_water_prior_weight = float(_get_cfg_value(model.args, "land_water_prior_weight", 1.0))
        self.shoreline_prior_max_dist = float(_get_cfg_value(model.args, "shoreline_prior_max_dist", 128.0))
        if self.bbox_loss_type not in {"probiou", "rotated_iou"}:
            raise ValueError(
                f"Unsupported RotatedFCOS bbox_loss_type={self.bbox_loss_type!r}. "
                "Expected 'probiou' or 'rotated_iou'."
            )

        self.loss_cls = FocalLoss(gamma=2.0, alpha=0.25)
        self.loss_centerness = nn.BCEWithLogitsLoss(reduction="sum")

    @staticmethod
    def _get_batch_cls_probs(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-target class probabilities with fallback to hard labels."""
        cls = batch["cls"].view(-1, 1)
        return batch.get("cls_probs", torch.ones_like(cls)).view(-1, 1).to(device=cls.device, dtype=cls.dtype)

    def preprocess(
        self, targets: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack flat targets into padded per-image tensors."""
        if targets.shape[0] == 0:
            zeros = torch.zeros(batch_size, 0, device=self.device)
            return zeros.long().unsqueeze(-1), zeros.new_zeros((batch_size, 0, 5)), zeros.unsqueeze(-1), zeros.bool().unsqueeze(-1)

        image_indices = targets[:, 0].long()
        _, counts = image_indices.unique(return_counts=True)
        max_count = int(counts.max())
        gt_labels = torch.zeros(batch_size, max_count, 1, device=self.device, dtype=torch.long)
        gt_bboxes = torch.zeros(batch_size, max_count, 5, device=self.device, dtype=targets.dtype)
        gt_probs = torch.zeros(batch_size, max_count, 1, device=self.device, dtype=targets.dtype)
        mask_gt = torch.zeros(batch_size, max_count, 1, device=self.device, dtype=torch.bool)

        for i in range(batch_size):
            matches = image_indices == i
            if n := int(matches.sum()):
                gt_labels[i, :n, 0] = targets[matches, 1].long()
                gt_bboxes[i, :n] = regularize_rboxes(targets[matches, 2:7], angle_mode=self.angle_mode)
                gt_probs[i, :n, 0] = targets[matches, 7]
                mask_gt[i, :n, 0] = True

        return gt_labels, gt_bboxes, gt_probs, mask_gt

    @staticmethod
    def centerness_target(pos_bbox_targets: torch.Tensor) -> torch.Tensor:
        """Compute FCOS centerness targets from ltrb distances."""
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if left_right.numel() == 0:
            return left_right.new_zeros((0,))
        centerness = (
            left_right.min(dim=-1).values / left_right.max(dim=-1).values.clamp_min(1e-7)
        ) * (
            top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values.clamp_min(1e-7)
        )
        return centerness.clamp_min(0).sqrt()

    def _get_target_single(
        self,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_probs: torch.Tensor,
        points: torch.Tensor,
        regress_ranges: torch.Tensor,
        point_strides: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign FCOS targets for a single image."""
        num_points = points.size(0)
        labels = gt_labels.new_full((num_points,), self.nc)
        bbox_targets = gt_bboxes.new_zeros((num_points, 4))
        angle_targets = gt_bboxes.new_zeros((num_points, 1))
        prob_targets = gt_bboxes.new_zeros((num_points, 1))
        num_gts = gt_labels.numel()
        if num_gts == 0:
            return labels, bbox_targets, angle_targets, prob_targets

        areas = (gt_bboxes[:, 2] * gt_bboxes[:, 3])[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)

        gt_ctr = gt_bboxes[..., :2]
        gt_wh = gt_bboxes[..., 2:4]
        gt_angle = gt_bboxes[..., 4:5]
        offset = points - gt_ctr
        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        offset_x = offset[..., 0:1] * cos_angle + offset[..., 1:2] * sin_angle
        offset_y = -offset[..., 0:1] * sin_angle + offset[..., 1:2] * cos_angle

        left = gt_wh[..., 0:1] / 2 + offset_x
        right = gt_wh[..., 0:1] / 2 - offset_x
        top = gt_wh[..., 1:2] / 2 + offset_y
        bottom = gt_wh[..., 1:2] / 2 - offset_y
        all_bbox_targets = torch.cat((left, top, right, bottom), dim=-1)

        inside_gt_bbox_mask = all_bbox_targets.min(dim=-1).values > 0
        if self.center_sampling:
            radius = point_strides[:, None, 0].expand(num_points, num_gts) * self.center_sample_radius
            inside_center = (offset_x.squeeze(-1).abs() < radius) & (offset_y.squeeze(-1).abs() < radius)
            inside_gt_bbox_mask &= inside_center

        max_regress_distance = all_bbox_targets.max(dim=-1).values
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0]) &
            (max_regress_distance <= regress_ranges[..., 1])
        )

        inf = torch.full_like(areas, float("inf"))
        areas = torch.where(inside_gt_bbox_mask & inside_regress_range, areas, inf)
        min_area, min_inds = areas.min(dim=1)
        pos_mask = min_area.isfinite()
        if not pos_mask.any():
            return labels, bbox_targets, angle_targets, prob_targets

        point_ids = torch.arange(num_points, device=points.device)
        bbox_targets = all_bbox_targets[point_ids, min_inds]
        angle_targets = gt_bboxes[point_ids, min_inds, 4:5]
        labels[pos_mask] = gt_labels[min_inds[pos_mask]]
        prob_targets[pos_mask, 0] = gt_probs[min_inds[pos_mask], 0]
        bbox_targets = torch.where(pos_mask[:, None], bbox_targets, bbox_targets.new_zeros(bbox_targets.shape))
        angle_targets = torch.where(pos_mask[:, None], angle_targets, angle_targets.new_zeros(angle_targets.shape))
        if self.norm_on_bbox:
            bbox_targets = bbox_targets / point_strides
        return labels, bbox_targets, angle_targets, prob_targets

    def get_targets(
        self,
        points: torch.Tensor,
        regress_ranges: torch.Tensor,
        point_strides: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_probs: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign FCOS targets for a batch."""
        bs = gt_labels.shape[0]
        labels, bbox_targets, angle_targets, prob_targets = [], [], [], []
        for i in range(bs):
            valid = mask_gt[i, :, 0]
            results = self._get_target_single(
                gt_bboxes[i, valid],
                gt_labels[i, valid, 0],
                gt_probs[i, valid],
                points,
                regress_ranges,
                point_strides,
            )
            labels.append(results[0])
            bbox_targets.append(results[1])
            angle_targets.append(results[2])
            prob_targets.append(results[3])
        return (
            torch.stack(labels, 0),
            torch.stack(bbox_targets, 0),
            torch.stack(angle_targets, 0),
            torch.stack(prob_targets, 0),
        )

    def _bbox_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute bbox regression loss using the selected rotated-box objective."""
        if self.bbox_loss_type == "probiou":
            loss = 1.0 - probiou(pred_boxes, target_boxes)
        else:
            loss = -torch.log(rotated_box_iou(pred_boxes, target_boxes).clamp_min(1e-7))
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Rotated FCOS classification, bbox, and centerness losses."""
        loss = torch.zeros(5, device=self.device)
        cls_scores, bbox_preds, angle_preds, centernesses = preds if len(preds) == 4 else preds[1]
        batch_size = cls_scores[0].shape[0]
        dtype = cls_scores[0].dtype

        flatten_cls_scores = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.nc) for x in cls_scores], dim=1
        )
        flatten_bbox_preds = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, 4) for x in bbox_preds], dim=1
        )
        flatten_angle_preds = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1, 1) for x in angle_preds], dim=1
        )
        flatten_centerness = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(batch_size, -1) for x in centernesses], dim=1
        )

        imgsz = torch.tensor(cls_scores[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(cls_scores, self.stride, 0.5)
        pixel_points = anchor_points * stride_tensor

        expanded_regress_ranges = [
            pixel_points.new_tensor(self.regress_ranges[i])[None].expand(x.shape[2] * x.shape[3], 2)
            for i, x in enumerate(cls_scores)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            cls_probs = self._get_batch_cls_probs(batch)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5), cls_probs), dim=1)
            if targets.numel():
                targets[:, 2:6] *= imgsz[[1, 0, 1, 0]]
                widths, heights = targets[:, 4], targets[:, 5]
                targets = targets[(widths >= 2) & (heights >= 2)]
            gt_labels, gt_bboxes, gt_probs, mask_gt = self.preprocess(targets, batch_size)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a Rotated FCOS OBB model on a detect dataset.\n"
                "Verify your dataset is a correctly formatted OBB dataset.\n"
                "See https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        target_labels, target_bboxes, target_angles, target_probs = self.get_targets(
            pixel_points,
            concat_regress_ranges,
            stride_tensor,
            gt_bboxes,
            gt_labels,
            gt_probs,
            mask_gt,
        )

        fg_mask = target_labels != self.nc
        num_pos = fg_mask.sum().clamp(min=1).float()

        target_scores = flatten_cls_scores.new_zeros((batch_size, flatten_cls_scores.shape[1], self.nc))
        if fg_mask.any():
            batch_inds, point_inds = fg_mask.nonzero(as_tuple=True)
            pos_target_probs = target_probs.squeeze(-1)[fg_mask].to(dtype)
            target_scores[batch_inds, point_inds, target_labels[fg_mask]] = pos_target_probs
        loss[1] = self.loss_cls(flatten_cls_scores.reshape(-1, self.nc), target_scores.reshape(-1, self.nc)) / num_pos

        if fg_mask.any():
            pos_bbox_preds = flatten_bbox_preds[fg_mask]
            pos_angle_preds = flatten_angle_preds[fg_mask]
            pos_centerness = flatten_centerness[fg_mask]
            pos_bbox_targets = target_bboxes[fg_mask]
            pos_angle_targets = target_angles[fg_mask]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets).to(dtype)

            pos_points = pixel_points.unsqueeze(0).expand(batch_size, -1, 2)[fg_mask]
            pos_strides = stride_tensor.unsqueeze(0).expand(batch_size, -1, 1)[fg_mask]
            if self.norm_on_bbox:
                pos_bbox_preds = pos_bbox_preds * pos_strides
                pos_bbox_targets = pos_bbox_targets * pos_strides

            pred_boxes = torch.cat((dist2rbox(pos_bbox_preds, pos_angle_preds, pos_points, dim=-1), pos_angle_preds), dim=-1)
            target_boxes = torch.cat(
                (dist2rbox(pos_bbox_targets, pos_angle_targets, pos_points, dim=-1), pos_angle_targets), dim=-1
            )
            pred_boxes = regularize_rboxes(pred_boxes, angle_mode=self.angle_mode)
            target_boxes = regularize_rboxes(target_boxes, angle_mode=self.angle_mode)

            centerness_weights = pos_centerness_targets.clamp_min(1e-7)
            loss[0] = self._bbox_loss(pred_boxes, target_boxes, centerness_weights)
            loss[2] = self.loss_centerness(pos_centerness, pos_centerness_targets) / num_pos
        else:
            loss[0] += (flatten_bbox_preds * 0).sum() + (flatten_angle_preds * 0).sum()
            loss[2] += (flatten_centerness * 0).sum()

        if self.use_shoreline_prior_loss or self.use_land_water_prior_loss:
            all_bbox_preds = flatten_bbox_preds
            all_points = pixel_points.unsqueeze(0).expand(batch_size, -1, 2)
            if self.norm_on_bbox:
                all_bbox_preds = all_bbox_preds * stride_tensor.unsqueeze(0)
            all_pred_boxes = torch.cat(
                (dist2rbox(all_bbox_preds, flatten_angle_preds, all_points, dim=-1), flatten_angle_preds), dim=-1
            )
            all_pred_boxes = regularize_rboxes(all_pred_boxes, angle_mode=self.angle_mode)
            confidence = flatten_cls_scores.sigmoid().amax(-1) * flatten_centerness.sigmoid()
            loss[3], loss[4] = _compute_obb_spatial_prior_losses(
                all_pred_boxes,
                confidence,
                ~fg_mask,
                batch.get("land_water_mask"),
                batch.get("shoreline_distance_map"),
                point_mode=self.shoreline_prior_point_mode,
                shoreline_prior_max_dist=self.shoreline_prior_max_dist,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[3] *= self.shoreline_prior_weight
        loss[4] *= self.land_water_prior_weight
        return loss * batch_size, loss.detach()


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]


# ==================== RT-DETR Losses ====================

class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class for calculating various loss components.

    This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the
    DETR object detection model.

    Attributes:
        nc (int): Number of classes.
        loss_gain (dict[str, float]): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Whether to use FocalLoss.
        use_vfl (bool): Whether to use VarifocalLoss.
        use_uni_match (bool): Whether to use a fixed layer for auxiliary branch label assignment.
        uni_match_ind (int): Index of fixed layer to use if use_uni_match is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss | None): Focal Loss object if use_fl is True, otherwise None.
        vfl (VarifocalLoss | None): Varifocal Loss object if use_vfl is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ):
        """
        Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary
        losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict[str, float], optional): Coefficients for different loss components.
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer.
            use_fl (bool): Whether to use FocalLoss.
            use_vfl (bool): Whether to use VarifocalLoss.
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
            gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
            alpha (float): The balancing factor used to address class imbalance.
        """
        super().__init__()
        from ultralytics.models.utils.ops import HungarianMatcher

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(
        self, pred_scores: torch.Tensor, targets: torch.Tensor, gt_scores: torch.Tensor, num_gts: int, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """
        Compute classification loss based on predictions, target values, and ground truth scores.

        Args:
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            targets (torch.Tensor): Target class indices with shape (B, N).
            gt_scores (torch.Tensor): Ground truth confidence scores with shape (B, N).
            num_gts (int): Number of ground truth objects.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing classification loss value.

        Notes:
            The function supports different classification loss types:
            - Varifocal Loss (if self.vfl is True and num_gts > 0)
            - Focal Loss (if self.fl is True)
            - BCE Loss (default fallback)
        """
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """
        Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4).
            postfix (str, optional): String to append to the loss names for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing:
                - loss_bbox{postfix}: L1 loss between predicted and ground truth boxes, scaled by the bbox loss gain.
                - loss_giou{postfix}: GIoU loss between predicted and ground truth boxes, scaled by the giou loss gain.

        Notes:
            If no ground truth boxes are provided (empty list), zero-valued tensors are returned for both losses.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        match_indices: list[tuple] | None = None,
        postfix: str = "",
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Get auxiliary losses for intermediate decoder layers.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes from auxiliary layers.
            pred_scores (torch.Tensor): Predicted scores from auxiliary layers.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            match_indices (list[tuple], optional): Pre-computed matching indices.
            postfix (str, optional): String to append to loss names.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of auxiliary losses.
        """
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices: list[tuple]) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Extract batch indices, source indices, and destination indices from match indices.

        Args:
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            batch_idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
            dst_idx (torch.Tensor): Destination indices.
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(match_indices)])
        src_idx = torch.cat([src for (src, _) in match_indices])
        dst_idx = torch.cat([dst for (_, dst) in match_indices])
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, match_indices: list[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            pred_assigned (torch.Tensor): Assigned predicted bounding boxes.
            gt_assigned (torch.Tensor): Assigned ground truth bounding boxes.
        """
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Calculate losses for a single prediction layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            pred_scores (torch.Tensor): Predicted class scores.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.
            postfix (str, optional): String to append to loss names.
            match_indices (list[tuple], optional): Pre-computed matching indices.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of losses.
        """
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
            # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
        }

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores, shape (L, B, N, C).
            batch (dict[str, Any]): Batch information containing cls, bboxes, and gt_groups.
            postfix (str, optional): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict[str, torch.Tensor]): Computed losses, including main and auxiliary (if enabled).

        Notes:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass to compute detection loss with optional denoising loss.

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): Tuple containing predicted bounding boxes and scores.
            batch (dict[str, Any]): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes.
            dn_scores (torch.Tensor, optional): Denoising scores.
            dn_meta (dict[str, Any], optional): Metadata for denoising.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing total loss and denoising loss if applicable.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: list[torch.Tensor], dn_num_group: int, gt_groups: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get match indices for denoising.

        Args:
            dn_pos_idx (list[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (list[int]): List of integers representing number of ground truths per image.

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), (
                    f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                )
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices


class RTDETRSegmentLoss(RTDETRDetectionLoss):
    """
    RT-DETR Segmentation Loss class that extends RTDETRDetectionLoss to add mask loss support.

    This class computes detection loss plus mask loss with optional soft ignore boundary and mixed loss (Lovasz + Dice + BCE).
    Reuses utilities from v8SegmentationLoss including MixedMaskLoss and soft ignore weight generation.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
        # Mask loss parameters (similar to v8SegmentationLoss)
        overlap_mask: bool = True,
        use_mixed_loss: bool = False,
        use_soft_ignore_band: bool = False,
        ignore_band_width: float = 10.0,
        soft_ignore_transition_ratio: float = 0.5,
        tile_size: int = 640,
        use_ultrafast_ignore: bool = False,
        seg_w_lovasz: float = 1.0,
        seg_w_dice: float = 0.3,
        seg_w_bce: float = 0.2,
        seg_ignore_index: int = -100,
        seg_area_normalize: bool = True,
    ):
        """
        Initialize RT-DETR segmentation loss.

        Args:
            nc (int): Number of classes.
            loss_gain (dict[str, float], optional): Coefficients for different loss components.
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer.
            use_fl (bool): Whether to use FocalLoss.
            use_vfl (bool): Whether to use VarifocalLoss.
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
            gamma (float): Focal loss gamma parameter.
            alpha (float): Focal loss alpha parameter.
            use_mixed_loss (bool): Whether to use MixedMaskLoss (Lovasz + Dice + BCE).
            use_soft_ignore_band (bool): Whether to apply soft ignore boundary weights.
            ignore_band_width (float): Width of ignore band in pixels at tile size.
            soft_ignore_transition_ratio (float): Ratio of hard to soft transition zone.
            tile_size (int): Reference tile size for scaling ignore width.
            use_ultrafast_ignore (bool): Whether to use fast torch-only implementation.
            seg_w_lovasz (float): Weight for Lovasz-Hinge term in MixedMaskLoss.
            seg_w_dice (float): Weight for Dice term in MixedMaskLoss.
            seg_w_bce (float): Weight for BCE term in MixedMaskLoss.
            seg_ignore_index (int): Ignore label for targets.
            seg_area_normalize (bool): Whether to normalize loss by area.
        """
        super().__init__(
            nc=nc,
            loss_gain=loss_gain,
            aux_loss=aux_loss,
            use_fl=use_fl,
            use_vfl=use_vfl,
            use_uni_match=use_uni_match,
            uni_match_ind=uni_match_ind,
            gamma=gamma,
            alpha=alpha,
        )

        self.overlap_mask = overlap_mask
        self.use_mixed_loss = use_mixed_loss
        self.use_soft_ignore_band = use_soft_ignore_band
        self.ignore_band_width = ignore_band_width
        self.soft_ignore_transition_ratio = soft_ignore_transition_ratio
        self.tile_size = tile_size
        self.use_ultrafast_ignore = use_ultrafast_ignore

        if self.use_mixed_loss:
            self.mixed_mask_loss = MixedMaskLoss(
                w_lovasz=seg_w_lovasz,
                w_dice=seg_w_dice,
                w_bce=seg_w_bce,
                ignore_index=seg_ignore_index,
                area_normalize=seg_area_normalize,
            )

    def _get_loss_mask(
        self,
        mask_coeffs: torch.Tensor,  # (bs, nq, nm) or (ndl, bs, nq, nm)
        protos: torch.Tensor,  # (bs, nm, H, W)
        gt_masks: torch.Tensor,  # (N, H, W) or list of (H, W) per image
        gt_bboxes: torch.Tensor,  # (N, 4) normalized xywh
        match_indices: list[tuple] | list[list[tuple]],
        gt_groups: list[int],
        batch_idx: torch.Tensor,
        imgsz: torch.Tensor,  # (2,) [h, w]
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        """
        Compute mask loss from mask coefficients and prototypes.

        Args:
            mask_coeffs (torch.Tensor): Mask coefficients, shape (bs, nq, nm) or (ndl, bs, nq, nm).
            protos (torch.Tensor): Prototypes, shape (bs, nm, H, W).
            gt_masks (torch.Tensor): Ground truth masks, shape (N, H, W) where N is total GTs across batch.
            match_indices (list[tuple] | list[list[tuple]]): Per-image matches for one layer or per-layer matches.
            imgsz (torch.Tensor): Image size [h, w].
            postfix (str): Postfix for loss names.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing mask loss components.
        """
        name_mask = f"loss_mask{postfix}"
        name_dice = f"loss_dice{postfix}"

        # Handle multi-layer mask_coeffs (from auxiliary losses)
        if mask_coeffs.dim() == 4:
            # For auxiliary losses, process each layer
            if match_indices and isinstance(match_indices[0], list):
                match_indices_per_layer = match_indices
            else:
                match_indices_per_layer = [match_indices] * mask_coeffs.shape[0]
            total_mask_loss = 0.0
            total_dice_loss = 0.0
            for layer_coeffs, layer_match_indices in zip(mask_coeffs, match_indices_per_layer):
                layer_loss = self._get_loss_mask_single_layer(
                    layer_coeffs, protos, gt_masks, gt_bboxes, layer_match_indices, gt_groups, batch_idx, imgsz
                )
                total_mask_loss += layer_loss["mask"]
                total_dice_loss += layer_loss.get("dice", 0.0)
            num_layers = mask_coeffs.shape[0]
            return {
                name_mask: (total_mask_loss / num_layers) * self.loss_gain.get("mask", 1.0),
                name_dice: (total_dice_loss / num_layers) * self.loss_gain.get("dice", 1.0),
            }
        else:
            # Single layer
            loss_dict = self._get_loss_mask_single_layer(
                mask_coeffs, protos, gt_masks, gt_bboxes, match_indices, gt_groups, batch_idx, imgsz
            )
            return {
                name_mask: loss_dict["mask"] * self.loss_gain.get("mask", 1.0),
                name_dice: loss_dict.get("dice", 0.0) * self.loss_gain.get("dice", 1.0),
            }

    def _get_loss_mask_single_layer(
        self,
        mask_coeffs: torch.Tensor,  # (bs, nq, nm)
        protos: torch.Tensor,       # (bs, nm, H, W)
        gt_masks: torch.Tensor,     # (N_total, H, W) flattened across batch
        gt_bboxes: torch.Tensor,    # (N_total, 4) normalized xywh
        match_indices: list[tuple],
        gt_groups: list[int],
        batch_idx: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute mask loss for a single layer using per-image matching."""
        bs, nq, nm = mask_coeffs.shape
        _, _, proto_h, proto_w = protos.shape
        if gt_masks.numel() == 0 or sum(len(src) for src, _ in match_indices) == 0:
            return {
                "mask": torch.tensor(0.0, device=self.device),
                "dice": torch.tensor(0.0, device=self.device),
            }

        gt_offsets = torch.as_tensor([0, *gt_groups[:-1]], device=mask_coeffs.device, dtype=torch.long).cumsum_(0)
        per_img_mask_losses: list[torch.Tensor] = []
        per_img_dice_losses: list[torch.Tensor] = []

        for b in range(bs):
            q_idx_b, gt_idx_b = match_indices[b]
            q_idx_b = q_idx_b.to(mask_coeffs.device)
            gt_idx_b = gt_idx_b.to(mask_coeffs.device)
            if q_idx_b.numel() == 0:
                continue

            # (n_b, nm), (nm, H, W)
            coeffs_b = mask_coeffs[b, q_idx_b]  # (n_b, nm)
            protos_b = protos[b]  # (nm, H, W)

            if self.overlap_mask:
                masks_i = gt_masks[b]
                if masks_i.ndim == 3 and masks_i.shape[0] == 1:
                    masks_i = masks_i[0]
                local_gt_idx = gt_idx_b - gt_offsets[b]
                gt_masks_b = (masks_i == (local_gt_idx + 1).view(-1, 1, 1)).float()
            else:
                gt_masks_b = gt_masks[gt_idx_b]
            gt_boxes_b = gt_bboxes[gt_idx_b]

            if coeffs_b.numel() == 0 or gt_masks_b.numel() == 0:
                continue

            if gt_masks_b.shape[-2:] != (proto_h, proto_w):
                gt_masks_b = F.interpolate(gt_masks_b.unsqueeze(1), (proto_h, proto_w), mode="nearest").squeeze(1)

            # Predicted masks for image b
            pred_masks_b = torch.einsum("in,nhw->ihw", coeffs_b, protos_b)  # (n_b, H, W)

            # Convert matched GT boxes from normalized xywh to mask-space xyxy for cropping-based losses.
            xyxy_b = xywh2xyxy(gt_boxes_b.clone())
            xyxy_b[:, [0, 2]] *= proto_w
            xyxy_b[:, [1, 3]] *= proto_h
            area_b = gt_boxes_b[:, 2:].prod(1).clamp_min(1e-6)

            # Optional soft-ignore weights (per image)
            weight_map_b = None
            if self.use_soft_ignore_band:
                scale = float(proto_h) / float(max(1, self.tile_size))
                scaled_ignore = max(0.0, self.ignore_band_width * scale)

                try:
                    # fast path if available
                    weight_map_b = create_soft_ignore_weights_fast(
                        gt_masks_b, scaled_ignore, self.soft_ignore_transition_ratio, device=gt_masks_b.device
                    )
                except Exception:
                    weight_map_b = create_soft_ignore_weights_torch(
                        gt_masks_b, scaled_ignore, self.soft_ignore_transition_ratio, device=gt_masks_b.device
                    )

            # --- MixedMaskLoss branch -------------------------------------------------
            if self.use_mixed_loss and hasattr(self, "mixed_mask_loss"):
                mask_loss_b = self.mixed_mask_loss(
                    logits=pred_masks_b,
                    targets=gt_masks_b.float(),
                    xyxy=xyxy_b,
                    area=area_b,
                    crop_mask_fn=crop_mask,
                    weight_map=weight_map_b,
                )
                # MixedMaskLoss usually already includes a Dice-like term; set dice=0 here
                dice_loss_b = torch.tensor(0.0, device=pred_masks_b.device)
            else:
                # --- Legacy BCE + Dice -------------------------------------------------
                loss_map = F.binary_cross_entropy_with_logits(
                    pred_masks_b, gt_masks_b.float(), reduction="none"
                )  # (n_b, H, W)

                if weight_map_b is not None:
                    # apply per-instance weight map and normalize per instance
                    loss_map = loss_map * weight_map_b
                    valid_sum = weight_map_b.sum(dim=(1, 2)).clamp_min(1e-6)  # (n_b,)
                    mask_loss_b = (loss_map.sum(dim=(1, 2)) / valid_sum).mean()
                else:
                    mask_loss_b = loss_map.mean()

                probs_b = torch.sigmoid(pred_masks_b)
                intersection = (probs_b * gt_masks_b.float()).sum(dim=(1, 2))
                union = (probs_b * probs_b).sum(dim=(1, 2)) + (gt_masks_b.float() * gt_masks_b.float()).sum(
                    dim=(1, 2)
                ) + 1e-6
                dice_b = 1.0 - (2.0 * intersection + 1e-6) / union
                dice_loss_b = dice_b.mean()

            per_img_mask_losses.append(mask_loss_b)
            per_img_dice_losses.append(dice_loss_b)

        if not per_img_mask_losses:
            return {
                "mask": torch.tensor(0.0, device=self.device),
                "dice": torch.tensor(0.0, device=self.device),
            }

        mask_loss = torch.stack(per_img_mask_losses).mean()
        dice_loss = torch.stack(per_img_dice_losses).mean()

        return {"mask": mask_loss, "dice": dice_loss}


    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_mask_coeffs: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass to compute detection and mask losses.

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): (pred_bboxes, pred_scores).
            batch (dict[str, Any]): Batch data with 'masks' key for GT masks.
            masks (tuple, optional): (dec_mask_coeffs, enc_mask_coeffs, protos).
            dn_bboxes (torch.Tensor, optional): Denoising bboxes.
            dn_scores (torch.Tensor, optional): Denoising scores.
            dn_mask_coeffs (torch.Tensor, optional): Denoising mask coefficients.
            dn_meta (dict, optional): Denoising metadata.

        Returns:
            (dict[str, torch.Tensor]): Loss dictionary including mask losses.
        """
        pred_bboxes, pred_scores = preds
        total_loss = RTDETRDetectionLoss.forward(
            self, preds, batch, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )

        # Compute mask loss if masks are provided
        if masks is not None and "masks" in batch:
            dec_mask_coeffs, enc_mask_coeffs, protos = masks
            gt_masks = batch["masks"]  # (N, H, W) or list
            imgsz = batch.get("imgsz", torch.tensor([640, 640], device=pred_bboxes.device))

            # Get match indices from the main loss computation
            # For encoder
            enc_mask_coeffs_expanded = enc_mask_coeffs.unsqueeze(0)  # (1, bs, nq, nm)
            main_match_indices = self._get_match_indices_for_layer(pred_bboxes[-1], pred_scores[-1], batch)

            # For decoder layers
            dec_mask_coeffs_all = torch.cat([enc_mask_coeffs_expanded, dec_mask_coeffs], dim=0)  # (ndl+1, bs, nq, nm)

            # Compute mask loss for main prediction (last layer)
            main_mask_loss = self._get_loss_mask(
                dec_mask_coeffs_all[-1:],
                protos,
                gt_masks,
                batch["bboxes"],
                main_match_indices,
                batch["gt_groups"],
                batch["batch_idx"],
                imgsz,
            )
            total_loss.update(main_mask_loss)

            # Compute auxiliary mask losses if enabled
            if self.aux_loss and dec_mask_coeffs.shape[0] > 0:
                aux_match_indices = [
                    self._get_match_indices_for_layer(aux_bboxes, aux_scores, batch)
                    for aux_bboxes, aux_scores in zip(pred_bboxes[:-1], pred_scores[:-1])
                ]
                aux_mask_loss = self._get_loss_mask(
                    dec_mask_coeffs_all[:-1],
                    protos,
                    gt_masks,
                    batch["bboxes"],
                    aux_match_indices,
                    batch["gt_groups"],
                    batch["batch_idx"],
                    imgsz,
                    postfix="_aux",
                )
                total_loss.update(aux_mask_loss)

        return total_loss

    def _get_match_indices_for_layer(
        self, pred_bboxes: torch.Tensor, pred_scores: torch.Tensor, batch: dict[str, Any]
    ) -> list[tuple]:
        """Get match indices for a specific layer."""
        gt_bboxes = batch["bboxes"]
        gt_cls = batch["cls"]
        gt_groups = batch["gt_groups"]
        return self.matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)


class RTDETROBBLoss(RTDETRDetectionLoss):
    """
    RT-DETR Oriented Bounding Box Loss class that extends RTDETRDetectionLoss to handle rotated bboxes.

    This class uses probiou instead of bbox_iou for GIoU computation on 5D rotated bounding boxes (x, y, w, h, angle).
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
        use_shoreline_prior_loss: bool = False,
        use_land_water_prior_loss: bool = False,
        shoreline_prior_point_mode: str = "center",
        shoreline_prior_weight: float = 1.0,
        land_water_prior_weight: float = 1.0,
        shoreline_prior_max_dist: float = 128.0,
    ):
        super().__init__(
            nc=nc,
            loss_gain=loss_gain,
            aux_loss=aux_loss,
            use_fl=use_fl,
            use_vfl=use_vfl,
            use_uni_match=use_uni_match,
            uni_match_ind=uni_match_ind,
            gamma=gamma,
            alpha=alpha,
        )
        self.use_shoreline_prior_loss = use_shoreline_prior_loss
        self.use_land_water_prior_loss = use_land_water_prior_loss
        self.shoreline_prior_point_mode = shoreline_prior_point_mode
        self.shoreline_prior_weight = shoreline_prior_weight
        self.land_water_prior_weight = land_water_prior_weight
        self.shoreline_prior_max_dist = shoreline_prior_max_dist

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate rotated-box losses for a single prediction layer."""
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = probiou(pred_bboxes.detach(), gt_bboxes).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
        }

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """
        Compute bounding box and GIoU losses for rotated bounding boxes using probiou.

        Args:
            pred_bboxes (torch.Tensor): Predicted rotated bounding boxes with shape (N, 5) or (N, 6).
            gt_bboxes (torch.Tensor): Ground truth rotated bounding boxes with shape (N, 5) or (N, 6).
            postfix (str, optional): String to append to the loss names.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing:
                - loss_bbox{postfix}: L1 loss between predicted and ground truth boxes (only for x, y, w, h).
                - loss_giou{postfix}: GIoU loss using probiou for rotated boxes, scaled by the giou loss gain.

        Notes:
            Assumes bboxes are in xywhr format (x, y, w, h, angle) or may have additional dimensions.
            Only first 4 coordinates are used for L1 loss, all 5 for GIoU.
        """
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        # Extract 5D boxes (x, y, w, h, angle) - handle both (N, 5) and (N, 6) cases
        pred_rbox = pred_bboxes[..., :5] if pred_bboxes.shape[-1] >= 5 else pred_bboxes[..., :4]
        gt_rbox = gt_bboxes[..., :5] if gt_bboxes.shape[-1] >= 5 else gt_bboxes[..., :4]

        # L1 loss only on first 4 coordinates (x, y, w, h)
        pred_xywh = pred_rbox[..., :4]
        gt_xywh = gt_rbox[..., :4]
        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_xywh, gt_xywh, reduction="sum") / len(gt_bboxes)

        # GIoU loss using probiou for rotated boxes
        if pred_rbox.shape[-1] == 5 and gt_rbox.shape[-1] == 5:
            # Both have angle - use probiou
            loss[name_giou] = 1.0 - probiou(pred_rbox, gt_rbox)
            loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        else:
            # Fallback to regular bbox_iou if no angle
            loss[name_giou] = 1.0 - bbox_iou(pred_rbox, gt_rbox, xywh=True, GIoU=True)
            loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)

        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute RT-DETR OBB losses plus optional shoreline and land/water prior penalties."""
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(preds, batch, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta)

        loss_shoreline_prior = pred_bboxes.new_tensor(0.0)
        loss_land_water_prior = pred_bboxes.new_tensor(0.0)
        land_water_map = batch.get("land_water_mask")
        if (self.use_shoreline_prior_loss or self.use_land_water_prior_loss) and land_water_map is not None:
            final_bboxes_norm = pred_bboxes[-1]
            final_bboxes = final_bboxes_norm.clone()
            final_scores = pred_scores[-1]
            _, _, height, width = land_water_map.shape
            final_bboxes[..., [0, 2]] *= width
            final_bboxes[..., [1, 3]] *= height

            match_indices = self.matcher(final_bboxes_norm, final_scores, batch["bboxes"], batch["cls"], batch["gt_groups"])
            negative_mask = torch.ones(final_scores.shape[:2], dtype=torch.bool, device=final_scores.device)
            for batch_idx, (src_idx, _) in enumerate(match_indices):
                negative_mask[batch_idx, src_idx] = False

            loss_shoreline_prior, loss_land_water_prior = _compute_obb_spatial_prior_losses(
                final_bboxes,
                final_scores.sigmoid().amax(-1),
                negative_mask,
                land_water_map,
                batch.get("shoreline_distance_map"),
                point_mode=self.shoreline_prior_point_mode,
                shoreline_prior_max_dist=self.shoreline_prior_max_dist,
            )
            loss_shoreline_prior *= self.shoreline_prior_weight
            loss_land_water_prior *= self.land_water_prior_weight

        total_loss["loss_shoreline_prior"] = loss_shoreline_prior
        total_loss["loss_land_water_prior"] = loss_land_water_prior
        return total_loss

# --- Mask2Former losses ------------------------------------------------------
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def _sigmoid_focal(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
                   reduction: str = "none"):
    """
    Compute sigmoid focal loss for addressing class imbalance.
    
    Args:
        inputs: Raw logits (before sigmoid)
        targets: Binary target labels
        alpha: Weighting factor for positive/negative examples (default: 0.25)
        gamma: Focusing parameter to down-weight easy examples (default: 2.0)
        reduction: Reduction method - 'none', 'mean', or 'sum'
        
    Returns:
        Focal loss value(s)
        
    Why focal loss: Down-weights easy examples and focuses training on hard negatives,
    particularly useful for segmentation where background pixels dominate.
    """
    # Apply sigmoid to get probabilities
    p = inputs.sigmoid()
    
    # Compute standard binary cross-entropy
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    
    # Compute probability of correct class (pt)
    pt = p * targets + (1 - p) * (1 - targets)
    
    # Apply modulating factor (1 - pt)^gamma to focus on hard examples
    loss = ce * ((1 - pt) ** gamma)
    
    # Apply alpha weighting if specified
    if alpha >= 0:
        at = alpha * targets + (1 - alpha) * (1 - targets)
        loss = at * loss
    
    # Apply reduction
    if reduction == "mean": return loss.mean()
    if reduction == "sum":  return loss.sum()
    return loss

def _point_sample(x: torch.Tensor, pts: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
    """
    Sample values from feature map at specified point coordinates.
    
    Args:
        x: Feature map of shape (B, C, H, W)
        pts: Normalized point coordinates in [0, 1] of shape (B, P, 2)
        align_corners: Whether to align corners in grid sampling
        
    Returns:
        Sampled values of shape (B, C, P)
        
    Why point sampling: More efficient than full mask comparison, focuses on uncertain
    regions, and provides better gradient signal for mask prediction.
    """
    # Convert from [0,1] to [-1,1] for grid_sample
    grid = pts * 2 - 1
    
    # Reshape to (B, P, 1, 2) for grid_sample compatibility
    grid = grid.view(x.size(0), -1, 1, 2)
    
    # Sample using bilinear interpolation
    out = F.grid_sample(x, grid, mode="bilinear", align_corners=align_corners)
    
    # Reshape to (B, C, P)
    return out.view(x.size(0), x.size(1), -1)

def _uncertain_points(logits: torch.Tensor, num_points: int, oversample: int = 3, importance: float = 0.75):
    """
    Sample points focusing on uncertain regions of predicted masks.
    
    Args:
        logits: Predicted mask logits of shape (B, 1, H, W)
        num_points: Number of points to sample (P)
        oversample: Oversampling factor for candidate selection (default: 3)
        importance: Fraction of points from uncertain regions vs random (default: 0.75)
        
    Returns:
        Sampled point coordinates of shape (B, P, 2) in [0, 1] range
        
    Why uncertain sampling: Focuses computation on decision boundaries where the model
    is most uncertain, leading to better training signal and faster convergence.
    """
    B = logits.shape[0]
    
    # Generate oversampled candidate points
    k = int(num_points * oversample)
    pts = torch.rand(B, k, 2, device=logits.device, dtype=logits.dtype)
    
    with torch.no_grad():
        # Sample logits at candidate points
        cand = _point_sample(logits, pts).squeeze(1)     # (B,k)
        
        # Compute uncertainty: logits near 0 are most uncertain (sigmoid ≈ 0.5)
        uncert = -torch.abs(cand)
        
        # Select top uncertain points
        topk = uncert.topk(max(int(num_points * importance), 1), dim=1).indices
        pick = pts[torch.arange(B, device=pts.device)[:, None], topk]
        
        # Fill remaining points with random samples if needed
        if pick.shape[1] < num_points:
            rnd = torch.rand(B, num_points - pick.shape[1], 2, device=pts.device, dtype=pts.dtype)
            pick = torch.cat([pick, rnd], 1)
    
    return pick  # (B,P,2)

class HungarianMatcherM2F(nn.Module):
    """
    Hungarian matcher with class focal + point BCE + point Dice costs.
    
    Performs bipartite matching between predicted instances and ground truth instances
    using the Hungarian algorithm. This is a key component of set prediction losses.
    
    Why Hungarian matching: Enables set-to-set prediction without predefined anchor boxes,
    allowing flexible number of instances and better handling of overlapping objects.
    """
    def __init__(self, cost_class=2.0, cost_mask=1.0, cost_dice=1.0, alpha=0.25, gamma=2.0, num_points=1024):
        """
        Args:
            cost_class: Weight for classification cost (default: 2.0)
            cost_mask: Weight for mask BCE cost (default: 1.0)
            cost_dice: Weight for Dice cost (default: 1.0)
            alpha: Focal loss alpha parameter (default: 0.25)
            gamma: Focal loss gamma parameter (default: 2.0)
            num_points: Number of points to sample for mask comparison (default: 1024)
        """
        super().__init__()
        self.cc, self.cm, self.cd = cost_class, cost_mask, cost_dice
        self.alpha, self.gamma, self.num_points = alpha, gamma, num_points

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        """
        Compute optimal assignment between predictions and targets.
        
        Args:
            outputs: Model predictions with 'pred_logits' (B,Q,C) and 'pred_masks' (B,Q,H,W)
            targets: List of target dicts with 'labels' and 'masks' for each image
            
        Returns:
            List of (query_indices, target_indices) tuples for each batch element
        """
        B, Q, C = outputs["pred_logits"].shape
        pm = outputs["pred_masks"]  # (B,Q,H,W)
        indices = []
        
        # Process each image in batch independently
        for b in range(B):
            labels = targets[b]["labels"]
            
            # Handle empty target case
            if labels.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            
            # Resize target masks to match prediction resolution
            H, W = pm.shape[-2:]
            tmask = targets[b]["masks"].float()
            tmask = F.interpolate(tmask.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)  # (N,H,W)

            # Sample uncertain points from predicted masks for efficient comparison
            pts = _uncertain_points(pm[b].unsqueeze(1), self.num_points).squeeze(0)  # (Q,P,2)
            pred_pts = _point_sample(pm[b].unsqueeze(1), pts.unsqueeze(0)).squeeze(0)  # (Q,P)

            # Sample same points from target masks
            tgt_pts = []
            for n in range(tmask.shape[0]):
                tgt_pts.append(_point_sample(tmask[n].unsqueeze(0).unsqueeze(0), pts.unsqueeze(0)).squeeze(0).squeeze(0))
            tgt_pts = torch.stack(tgt_pts, 0)  # (N,P)

            # Compute pairwise costs between all query-target pairs
            predP = pred_pts.unsqueeze(1).repeat(1, tmask.shape[0], 1)  # (Q,N,P)
            tgtP  = tgt_pts.unsqueeze(0).repeat(pred_pts.shape[0], 1, 1) # (Q,N,P)
            
            # Mask BCE cost
            bce   = F.binary_cross_entropy_with_logits(predP, tgtP, reduction="none").mean(-1)  # (Q,N)
            
            # Mask Dice cost
            predS = predP.sigmoid()
            inter = (predS * tgtP).sum(-1)
            union = predS.sum(-1) + tgtP.sum(-1)
            dice  = 1 - (2*inter + 1e-6) / (union + 1e-6)  # (Q,N)

            # Classification focal cost (convert to one-hot for each target)
            logits = outputs["pred_logits"][b]  # (Q,C)
            tgt_oh = torch.zeros((Q, labels.numel(), C), device=logits.device, dtype=logits.dtype)
            for n, cid in enumerate(labels.tolist()):
                tgt_oh[:, n, cid] = 1.0
            c_cost = _sigmoid_focal(logits.unsqueeze(1).repeat(1, labels.numel(), 1),
                                    tgt_oh, alpha=self.alpha, gamma=self.gamma, reduction="none").sum(-1)
            
            # Combine all costs into final cost matrix
            Cmat = self.cc * c_cost + self.cm * bce + self.cd * dice

            # Solve optimal assignment using Hungarian algorithm
            from scipy.optimize import linear_sum_assignment
            qi, ti = linear_sum_assignment(Cmat.detach().cpu())
            indices.append((torch.as_tensor(qi, dtype=torch.long), torch.as_tensor(ti, dtype=torch.long)))
        
        return indices


class Mask2FormerLoss(nn.Module):
    """
    Ultralytics-compatible set-prediction loss for Mask2Former-like heads.
    
    Implements the full Mask2Former loss with:
    - Hungarian matching for optimal query-target assignment
    - Focal loss for classification
    - BCE + Dice loss for masks (evaluated at sampled points)
    - Deep supervision via auxiliary losses
    
    Expects model outputs: dict {'pred_logits','pred_masks','aux_outputs'}
    Targets accepted from standard Ultralytics batch (tensor or per-image lists).
    """
    def __init__(self, num_classes: int, alpha: float = 0.25, gamma: float = 2.0, num_points: int = 1024, aux_weight: float = 1.0):
        """
        Args:
            num_classes: Number of segmentation classes
            alpha: Focal loss alpha parameter (default: 0.25)
            gamma: Focal loss gamma parameter (default: 2.0)
            num_points: Number of points to sample for mask loss (default: 1024)
            aux_weight: Weight for auxiliary (intermediate) layer losses (default: 1.0)
        """
        super().__init__()
        self.num_classes = num_classes
        self.aux_weight  = aux_weight
        self.matcher     = HungarianMatcherM2F(alpha=alpha, gamma=gamma, num_points=num_points)

    def _build_targets(self, batch: dict, H: int, W: int) -> List[Dict[str, torch.Tensor]]:
        """
        Convert batch dictionary to list of per-image target dictionaries.
        
        Args:
            batch: Batch dictionary from dataloader
            H: Target height for masks
            W: Target width for masks
            
        Returns:
            List of dicts with 'labels' and 'masks' for each image
            
        Supports two formats:
          (A) Pre-split per-image: batch['gt_labels'][i], batch['gt_masks'][i]
          (B) Concatenated tensors + batch_idx: batch['cls'], batch['masks'], batch['batch_idx']
        """
        # Format A: Pre-split per-image lists
        if "gt_labels" in batch and "gt_masks" in batch:
            return [{"labels": batch["gt_labels"][i].to(torch.long),
                     "masks":  batch["gt_masks"][i].float()} for i in range(len(batch["gt_labels"]))]

        # Format B: Concatenated format - determine batch size
        imgs = batch.get("img", batch.get("imgs"))
        bs = int(imgs.shape[0]) if imgs is not None else int(batch["batch_idx"].max().item() + 1)
        
        cls = batch["cls"]
        
        # Handle list format
        if isinstance(cls, list):
            return [{"labels": c.view(-1).to(torch.long), "masks": m.float()} for c, m in zip(cls, batch["masks"])]
        
        # Handle concatenated tensor format - split by batch_idx
        bidx = batch["batch_idx"].to(torch.long)
        cls = cls.view(-1).to(torch.long)
        masks = batch["masks"].float()
        
        out = []
        for i in range(bs):
            sel = (bidx == i)
            out.append({"labels": cls[sel], "masks": masks[sel]})
        return out

    def _loss_class(self, logits: torch.Tensor, targets, indices):
        """
        Compute classification loss using focal loss.
        
        Args:
            logits: Predicted class logits of shape (B, Q, C)
            targets: List of target dicts
            indices: Matched query-target pairs from Hungarian matcher
            
        Returns:
            Scalar classification loss
            
        Why focal loss: Handles class imbalance between matched queries (positive)
        and unmatched queries (negative/background).
        """
        B, Q, C = logits.shape
        
        # Initialize all queries as background (zeros)
        tgt = torch.zeros((B, Q, C), device=logits.device, dtype=torch.float32)
        
        # Set matched queries to their target class (one-hot)
        for b, (qi, ti) in enumerate(indices):
            if ti.numel():
                tgt[b, qi, targets[b]["labels"][ti]] = 1.0
        
        # Compute focal loss
        return _sigmoid_focal(logits.view(-1, C), tgt.view(-1, C), reduction="mean")

    def _loss_masks(self, pm: torch.Tensor, targets, indices):
        """
        Compute mask losses using BCE and Dice on sampled points.
        
        Args:
            pm: Predicted masks of shape (B, Q, H, W)
            targets: List of target dicts
            indices: Matched query-target pairs from Hungarian matcher
            
        Returns:
            Tuple of (bce_loss, dice_loss)
            
        Why point sampling: Significantly reduces memory and computation compared to
        full-resolution masks while maintaining good training signal.
        """
        B, Q, H, W = pm.shape
        
        # Handle case with no matched instances
        if sum(len(t["labels"]) for t in targets) == 0:
            z = pm.sum() * 0.0
            return z, z
        
        # Gather all matched predictions
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        pred = pm[batch_idx, src_idx]  # (M,H,W) where M is total matched instances
        
        # Gather and resize all matched target masks
        tcat = []
        for i, (_, ti) in enumerate(indices):
            if ti.numel():
                m = targets[i]["masks"][ti].float()
                # Resize to prediction resolution
                m = F.interpolate(m.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
                tcat.append(m)
        tgt = torch.cat(tcat, 0) if tcat else torch.zeros_like(pred)

        # Sample uncertain points for efficient mask comparison
        pts = _uncertain_points(pred.unsqueeze(1), num_points=1024).squeeze(1)  # (M,P,2)
        p = _point_sample(pred.unsqueeze(1), pts).squeeze(1)    # (M,P)
        t = _point_sample(tgt.unsqueeze(1),  pts).squeeze(1)    # (M,P)

        # Compute BCE loss at sampled points
        bce = F.binary_cross_entropy_with_logits(p, t, reduction="mean")
        
        # Compute Dice loss at sampled points
        ps = p.sigmoid()
        inter = (ps * t).sum(-1).mean()
        uni = (ps + t).sum(-1).mean()
        dice = 1 - (2*inter + 1e-6) / (uni + 1e-6)
        
        return bce, dice

    def forward(self, outputs: Dict[str, torch.Tensor], batch: dict):
        """
        Compute total loss with deep supervision.
        
        Args:
            outputs: Model predictions dict with 'pred_logits', 'pred_masks', and 'aux_outputs'
            batch: Batch dictionary with ground truth annotations
            
        Returns:
            Tuple of (total_loss, loss_components_tensor)
            
        Deep supervision: Auxiliary losses from intermediate decoder layers help stabilize
        training and improve feature learning throughout the decoder stack.
        """
        # Prepare targets in standard format
        H, W = outputs["pred_masks"].shape[-2:]
        targets = self._build_targets(batch, H, W)
        
        # Compute matching for final predictions
        indices = self.matcher(outputs, targets)

        # Compute losses for final predictions
        lcls = self._loss_class(outputs["pred_logits"], targets, indices)
        lbce, ldice = self._loss_masks(outputs["pred_masks"], targets, indices)

        # Add auxiliary losses from intermediate decoder layers (deep supervision)
        for aux in outputs.get("aux_outputs", []):
            # Compute new matching for each auxiliary output
            aidx = self.matcher(aux, targets)
            
            # Add weighted auxiliary losses
            lcls  = lcls  + self._loss_class(aux["pred_logits"], targets, aidx) * self.aux_weight
            b, d  = self._loss_masks(aux["pred_masks"], targets, aidx)
            lbce  = lbce  + b * self.aux_weight
            ldice = ldice + d * self.aux_weight

        # Combine losses: weighted classification + BCE + Dice
        # Why 2x weight on classification: Balances gradient magnitudes between
        # classification and mask prediction tasks
        loss = 2.0 * lcls + lbce + ldice
        
        # Return total loss and individual components for logging
        return loss, torch.tensor([lcls.detach(), lbce.detach(), ldice.detach()],
                                  device=outputs["pred_masks"].device)
