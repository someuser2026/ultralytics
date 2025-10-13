# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any, Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
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
def create_soft_ignore_weights_fast(
    gt_masks: torch.Tensor,
    ignore_width: float,
    transition_ratio: float = 0.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Accurate soft-ignore using distance to the mask edge.
    Zones:
      - [0, hard): weight=0
      - [hard, ignore): weight rises smoothly to 1 (Gaussian)
      - [ignore, +inf): weight=1
    """
    if ignore_width <= 0:
        return torch.ones_like(gt_masks, dtype=torch.float32)

    if device is None:
        device = gt_masks.device

    if not _HAS_SCI_CV:
        # Fallback if SciPy/OpenCV is not present
        return create_soft_ignore_weights_torch(gt_masks, ignore_width, transition_ratio, device=device)

    masks_np = gt_masks.detach().to('cpu', torch.uint8).numpy()
    N, H, W = masks_np.shape
    weight_maps = np.ones((N, H, W), dtype=np.float32)

    hard_w = int(math.ceil(ignore_width * max(0.0, min(1.0, transition_ratio))))
    trans_w = max(0, int(math.ceil(ignore_width - hard_w)))
    sigma = max(trans_w / 3.0, 0.5) if trans_w > 0 else 1.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for i in range(N):
        mask = masks_np[i]
        if mask.sum() == 0:
            continue
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        # distance from edges in pixels
        dist = distance_transform_edt(1 - edges).astype(np.float32)

        w = np.ones_like(dist, dtype=np.float32)

        if hard_w > 0:
            hard_zone = dist < hard_w
            w[hard_zone] = 0.0

        if trans_w > 0:
            band = (dist >= hard_w) & (dist < hard_w + trans_w)
            if band.any():
                d = dist[band] - hard_w
                w[band] = 1.0 - np.exp(-(d ** 2) / (2.0 * sigma ** 2))
        weight_maps[i] = w

    return torch.from_numpy(weight_maps).to(device=device, dtype=torch.float32)


@torch.no_grad()
def create_soft_ignore_weights_torch(
    gt_masks: torch.Tensor,
    ignore_width: float,
    transition_ratio: float = 0.5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Ultra-fast, pure PyTorch approximation (no SciPy/OpenCV).
    Uses iterative erosions to approximate a soft band.
    """
    if device is None:
        device = gt_masks.device

    if ignore_width <= 0:
        return torch.ones_like(gt_masks, dtype=torch.float32)

    masks = (gt_masks > 0.5).to(torch.float32)
    N, H, W = masks.shape
    weight = torch.ones((N, H, W), device=device, dtype=torch.float32)

    hard_w = max(1, int(math.ceil(ignore_width * max(0.0, min(1.0, transition_ratio)))))
    trans_w = max(0, int(math.ceil(ignore_width - hard_w)))

    # erosion utility (binary)
    def erode(bin_mask: torch.Tensor, r: int) -> torch.Tensor:
        if r <= 0:
            return bin_mask
        k = 2 * r + 1
        inv = 1.0 - bin_mask  # 1 where background
        # If any background in the window -> max_pool > 0 => eroded becomes 0
        mp = torch.nn.functional.max_pool2d(inv.unsqueeze(1), kernel_size=k, stride=1, padding=r)
        return (mp == 0).to(torch.float32).squeeze(1)

    core = erode(masks, hard_w) if hard_w > 0 else masks
    weight[core > 0.5] = 1.0  # core = full weight

    if trans_w > 0:
        steps = min(5, trans_w)
        step_size = max(1, trans_w // steps)
        for s in range(1, steps + 1):
            r = hard_w + s * step_size
            er = erode(masks, r)
            # band is the ring that disappears at this erosion step
            band = (masks > 0.5) & (core < 0.5) & (er < 0.5)
            weight[band] = s / steps

    if hard_w > 0:
        inner = erode(masks, max(1, hard_w - 1))
        hard_zone = (masks > 0.5) & (inner < 0.5)
        weight[hard_zone] = 0.0

    return weight
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
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

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
            core_mask = (weights_c >= 0.99).to(logits_c.dtype)
            # mark non-core as ignore by setting targets to ignore_index
            t_for_lovasz = targets_c.clone()
            t_for_lovasz[core_mask < 0.5] = self.lovasz.ignore_index
            lovasz_vec = self.lovasz(logits_c, t_for_lovasz)  # per-instance
        else:
            lovasz_vec = self.lovasz(logits_c, targets_c)

        # (2) Weighted Dice per instance (N,)
        probs = torch.sigmoid(logits_c)
        if weights_c is not None:
            num = 2.0 * (weights_c * probs * targets_c).sum(dim=(1, 2))
            den = (weights_c * probs * probs).sum(dim=(1, 2)) + (weights_c * targets_c * targets_c).sum(dim=(1, 2)) + 1e-6
            dice_vec = 1.0 - (num + 1e-6) / den
        else:
            dice_vec = dice_loss_with_logits(logits_c, targets_c)

        # (3) Weighted BCE per instance (N,)
        if weights_c is not None:
            bce_map = torch.nn.functional.binary_cross_entropy_with_logits(logits_c, targets_c, reduction="none")
            bce_sum = (bce_map * weights_c).sum(dim=(1, 2))
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
        self._weight_map_cache: dict[int, torch.Tensor] = {}


    def __call__(self, preds: Any, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""

        self._weight_map_cache.clear()

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
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
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
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
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
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n,H,W)

        # 2) optional soft-ignore weight map per-instance
        weight_map = None
        if self.use_soft_ignore_band:
            # scale width from tile size to proto resolution (H)
            proto_h = proto.shape[-2]
            scale = float(proto_h) / float(max(1, self.tile_size))
            scaled_ignore = max(0.0, self.ignore_band_width * scale)

            # cache key per tensor reference (local to this forward)
            key = id(gt_mask)
            if key not in self._weight_map_cache:
                # choose accurate or fast generator
                if _HAS_SCI_CV and not self.use_ultrafast_ignore:
                    wm = create_soft_ignore_weights_fast(
                        (gt_mask > 0.5).to(torch.float32),
                        ignore_width=scaled_ignore,
                        transition_ratio=self.soft_ignore_transition_ratio,
                        device=gt_mask.device,
                    )
                else:
                    wm = create_soft_ignore_weights_torch(
                        (gt_mask > 0.5).to(torch.float32),
                        ignore_width=scaled_ignore,
                        transition_ratio=self.soft_ignore_transition_ratio,
                        device=gt_mask.device,
                    )
                self._weight_map_cache[key] = wm
            weight_map = self._weight_map_cache[key]

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
        loss_map = torch.nn.functional.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")

        if weight_map is not None:
            # element-wise weighting then crop + normalize over valid weights
            loss_map = loss_map * weight_map
            cropped_loss = crop_mask(loss_map, xyxy)
            cropped_w    = crop_mask(weight_map, xyxy)
            valid_sum = cropped_w.sum(dim=(1, 2)).clamp_min(1e-6)
            return (cropped_loss.sum(dim=(1, 2)) / valid_sum / area.clamp_min(1e-6)).sum()
        else:
            # original mean over crop
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

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

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

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


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
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
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


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
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
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
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
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

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

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

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
