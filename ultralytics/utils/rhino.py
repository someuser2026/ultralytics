# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RHINO-specific matcher and loss utilities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from ultralytics.utils.loss import RTDETROBBLoss, _compute_obb_spatial_prior_losses
from ultralytics.utils.metrics import batch_probiou, probiou
from ultralytics.utils.ops import xywhr2xyxyxyxy


def rhino_boxes_to_physical(boxes: torch.Tensor, image_shape: tuple[int, int] | torch.Tensor) -> torch.Tensor:
    """Convert normalized RHINO boxes to pixel ``cxcywh`` and radian angles."""
    shape = torch.as_tensor(image_shape, dtype=boxes.dtype, device=boxes.device).flatten()
    if shape.numel() < 2:
        raise ValueError(f"RHINO image_shape must contain height and width, got {image_shape!r}.")
    height, width = shape[0], shape[1]
    factor = torch.stack((width, height, width, height, boxes.new_tensor(torch.pi)))
    return boxes[..., :5] * factor


def _rhino_boxes_to_hausdorff_space(boxes: torch.Tensor) -> torch.Tensor:
    """Keep RHINO ``cxcywh`` normalized and convert only ``angle/pi`` to radians."""
    converted = boxes[..., :5].clone()
    converted[..., 4] *= torch.pi
    return converted


def _distributed_mean(value: int | float | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Synchronize a RHINO loss normalizer across distributed workers."""
    if torch.is_tensor(value):
        normalizer = value.detach().to(device=device, dtype=torch.float32).reshape(1)
    else:
        normalizer = torch.tensor([float(value)], device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(normalizer)
        normalizer /= torch.distributed.get_world_size()
    return normalizer.clamp_min(1.0)


def xy_wh_r_2_xy_sigma(xywhr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert oriented boxes to 2D Gaussian parameters."""
    assert xywhr.shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    angle = xywhr[..., 4]
    cos_r = torch.cos(angle)
    sin_r = torch.sin(angle)
    rot = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    scale = 0.5 * torch.diag_embed(wh)
    sigma = rot.bmm(scale.square()).bmm(rot.permute(0, 2, 1)).reshape(xywhr.shape[:-1] + (2, 2))
    return xy, sigma


def postprocess_distance(distance: torch.Tensor, fun: str = "log1p", tau: float = 1.0) -> torch.Tensor:
    """Apply RHINO-style nonlinear postprocessing to distances."""
    if fun == "log1p":
        distance = torch.log1p(distance.clamp_min(-1 + 1e-7))
    elif fun == "sqrt":
        distance = torch.sqrt(distance.clamp_min(1e-7))
    elif fun == "none":
        pass
    else:
        raise ValueError(f"Unsupported distance transform {fun!r}.")

    return 1 - 1 / (tau + distance) if tau >= 1.0 else distance


def kld_loss(
    pred: tuple[torch.Tensor, torch.Tensor],
    target: tuple[torch.Tensor, torch.Tensor],
    fun: str = "log1p",
    tau: float = 1.0,
    alpha: float = 1.0,
    sqrt: bool = False,
) -> torch.Tensor:
    """Elementwise KLD loss for paired Gaussian boxes."""
    xy_p, sigma_p = pred
    xy_t, sigma_t = target

    shape = xy_p.shape[:-1]
    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)

    sigma_p_inv = torch.stack(
        (sigma_p[..., 1, 1], -sigma_p[..., 0, 1], -sigma_p[..., 1, 0], sigma_p[..., 0, 0]), dim=-1
    ).reshape(-1, 2, 2)
    sigma_p_inv = sigma_p_inv / sigma_p.det().unsqueeze(-1).unsqueeze(-1).clamp_min(1e-7)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(sigma_p_inv).bmm(dxy).view(-1)
    whr_distance = 0.5 * sigma_p_inv.bmm(sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + 0.5 * (sigma_p.det().clamp_min(1e-7).log() - sigma_t.det().clamp_min(1e-7).log())
    whr_distance = whr_distance - 1.0
    distance = xy_distance / (alpha * alpha) + whr_distance
    if sqrt:
        distance = distance.clamp_min(1e-7).sqrt()
    return postprocess_distance(distance.reshape(shape), fun=fun, tau=tau)


def pairwise_kld_loss(
    pred: tuple[torch.Tensor, torch.Tensor],
    target: tuple[torch.Tensor, torch.Tensor],
    fun: str = "log1p",
    tau: float = 1.0,
    alpha: float = 1.0,
    sqrt: bool = False,
) -> torch.Tensor:
    """Pairwise KLD cost matrix for Gaussian boxes."""
    xy_1, sigma_1 = pred
    xy_2, sigma_2 = target
    n = xy_1.shape[0]
    m = xy_2.shape[0]
    if n == 0 or m == 0:
        return xy_1.new_zeros((n, m))

    xy_1 = xy_1.unsqueeze(1).repeat(1, m, 1).view(-1, 2)
    sigma_1 = sigma_1.unsqueeze(1).repeat(1, m, 1, 1).view(-1, 2, 2)
    xy_2 = xy_2.unsqueeze(0).repeat(n, 1, 1).view(-1, 2)
    sigma_2 = sigma_2.unsqueeze(0).repeat(n, 1, 1, 1).view(-1, 2, 2)
    return kld_loss((xy_1, sigma_1), (xy_2, sigma_2), fun=fun, tau=tau, alpha=alpha, sqrt=sqrt).view(n, m)


def box2multiple_corners(boxes: torch.Tensor, num_points: int) -> torch.Tensor:
    """Sample evenly spaced perimeter points from oriented boxes."""
    corners = xywhr2xyxyxyxy(boxes)
    if num_points == 4:
        return corners
    if num_points % 4 != 0:
        raise ValueError("Hausdorff num_points must be divisible by 4.")

    samples_per_edge = num_points // 4
    t = torch.linspace(0, 1, samples_per_edge + 1, device=boxes.device, dtype=boxes.dtype)[:-1]
    edge_points = []
    for edge_idx in range(4):
        start = corners[:, edge_idx]
        end = corners[:, (edge_idx + 1) % 4]
        edge = start[:, None, :] * (1 - t)[None, :, None] + end[:, None, :] * t[None, :, None]
        edge_points.append(edge)
    return torch.cat(edge_points, dim=1)


def hausdorff_distance(boxes1: torch.Tensor, boxes2: torch.Tensor, num_points: int = 4) -> torch.Tensor:
    """Elementwise Hausdorff distance between paired oriented boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((0,))
    points1 = box2multiple_corners(boxes1, num_points)
    points2 = box2multiple_corners(boxes2, num_points)
    pairwise = torch.norm(points1[:, :, None, :] - points2[:, None, :, :], dim=-1)
    d1 = pairwise.min(dim=-1).values.max(dim=-1).values
    d2 = pairwise.min(dim=-2).values.max(dim=-1).values
    return torch.maximum(d1, d2)


def hausdorff_pairwise_cost(boxes1: torch.Tensor, boxes2: torch.Tensor, num_points: int = 4) -> torch.Tensor:
    """Pairwise Hausdorff cost matrix."""
    n, m = boxes1.shape[0], boxes2.shape[0]
    if n == 0 or m == 0:
        return boxes1.new_zeros((n, m))

    points1 = box2multiple_corners(boxes1, num_points)
    points2 = box2multiple_corners(boxes2, num_points)
    pairwise = torch.norm(points1[:, None, :, None, :] - points2[None, :, None, :, :], dim=-1)
    d1 = pairwise.min(dim=-1).values.max(dim=-1).values
    d2 = pairwise.min(dim=-2).values.max(dim=-1).values
    return torch.maximum(d1, d2)


class RHINOHungarianMatcher(nn.Module):
    """Per-image Hungarian matcher with RHINO-specific cost composition."""

    DEFAULT_COSTS = (
        {"type": "focal", "weight": 2.0},
        {"type": "hausdorff", "weight": 5.0, "num_points": 4},
        {
            "type": "gdcost",
            "loss_type": "kld",
            "fun": "log1p",
            "tau": 1.0,
            "alpha": 1.0,
            "sqrt": False,
            "weight": 5.0,
        },
    )

    def __init__(
        self,
        costs: list[dict[str, Any]] | None = None,
        use_fl: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.costs = list(deepcopy(costs) if costs is not None else deepcopy(self.DEFAULT_COSTS))
        self.use_fl = use_fl
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        image_shapes: list[tuple[int, int]] | torch.Tensor | None = None,
        **_: Any,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        bs = pred_scores.shape[0]
        device = pred_bboxes.device
        if sum(gt_groups) == 0:
            return [
                (
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                )
                for _ in range(bs)
            ]

        results = []
        gt_offset = 0
        for batch_idx, num_gt in enumerate(gt_groups):
            if num_gt == 0:
                results.append(
                    (
                        torch.zeros(0, dtype=torch.long, device=device),
                        torch.zeros(0, dtype=torch.long, device=device),
                    )
                )
                continue
            gt_slice = slice(gt_offset, gt_offset + num_gt)
            cost = self.cost_matrix(
                pred_bboxes[batch_idx],
                pred_scores[batch_idx],
                gt_bboxes[gt_slice],
                gt_cls[gt_slice],
                image_shape=image_shapes[batch_idx] if image_shapes is not None else (1, 1),
            )
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu())
            row = torch.as_tensor(row_ind, dtype=torch.long, device=pred_bboxes.device)
            col = torch.as_tensor(col_ind, dtype=torch.long, device=pred_bboxes.device)
            results.append((row, col + gt_offset))
            gt_offset += num_gt
        return results

    def cost_matrix(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        image_shape: tuple[int, int] | torch.Tensor = (1, 1),
    ) -> torch.Tensor:
        nq = pred_scores.shape[0]
        ng = gt_bboxes.shape[0]
        if ng == 0:
            return pred_scores.new_zeros((nq, 0))

        pred_probs = pred_scores.detach().sigmoid() if self.use_fl else pred_scores.detach().softmax(-1)
        total_cost = pred_scores.new_zeros((nq, ng))

        for cfg in self.costs:
            cfg = deepcopy(cfg)
            weight = float(cfg.pop("weight", 1.0))
            name = str(cfg.pop("type", "focal")).lower().replace("-", "").replace("_", "")

            if name in {"focal", "focallosscost", "class", "classification"}:
                cls_scores = pred_probs[:, gt_cls]
                if self.use_fl:
                    neg = (1 - self.alpha) * (cls_scores**self.gamma) * (-(1 - cls_scores + 1e-8).log())
                    pos = self.alpha * ((1 - cls_scores) ** self.gamma) * (-(cls_scores + 1e-8).log())
                    cost = pos - neg
                else:
                    cost = -cls_scores
            elif name in {"rboxl1", "bbox", "bboxl1", "xywha", "xywha"}:
                cost = torch.cdist(pred_bboxes, gt_bboxes, p=1)
            elif name in {"centerl1", "center"}:
                cost = torch.cdist(pred_bboxes[:, :2], gt_bboxes[:, :2], p=1)
            elif name in {"hausdorff", "hausdorffcost"}:
                cost = hausdorff_pairwise_cost(
                    _rhino_boxes_to_hausdorff_space(pred_bboxes),
                    _rhino_boxes_to_hausdorff_space(gt_bboxes),
                    num_points=int(cfg.get("num_points", 4)),
                )
            elif name in {"gdcost", "gd", "kld"}:
                pred_gaussian = xy_wh_r_2_xy_sigma(rhino_boxes_to_physical(pred_bboxes, image_shape))
                gt_gaussian = xy_wh_r_2_xy_sigma(rhino_boxes_to_physical(gt_bboxes, image_shape))
                cost = pairwise_kld_loss(
                    pred_gaussian,
                    gt_gaussian,
                    fun=str(cfg.get("fun", "log1p")),
                    tau=float(cfg.get("tau", 1.0)),
                    alpha=float(cfg.get("alpha", 1.0)),
                    sqrt=bool(cfg.get("sqrt", False)),
                )
            elif name in {"probiou", "rotatediou", "iou"}:
                physical_pred = rhino_boxes_to_physical(pred_bboxes, image_shape)
                physical_gt = rhino_boxes_to_physical(gt_bboxes, image_shape)
                cost = 1.0 - batch_probiou(physical_gt, physical_pred).transpose(0, 1)
            else:
                raise ValueError(f"Unsupported RHINO matcher cost type {cfg.get('type', name)!r}.")

            total_cost += weight * cost

        total_cost[total_cost.isnan() | total_cost.isinf()] = 0.0
        return total_cost


class DNGroupHungarianAssigner:
    """Positive-Hungarian assigner for RHINO denoising groups."""

    def __init__(
        self,
        costs: list[dict[str, Any]] | None = None,
        use_fl: bool = True,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        self.matcher = RHINOHungarianMatcher(costs=costs, use_fl=use_fl, alpha=alpha, gamma=gamma)

    def assign(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        dn_bboxes: torch.Tensor,
        dn_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        num_groups: int,
        image_shape: tuple[int, int] | torch.Tensor = (1, 1),
    ) -> torch.Tensor:
        num_gt = gt_bboxes.shape[0]
        if num_gt == 0 or dn_bboxes.numel() == 0:
            return gt_cls.new_full((dn_bboxes.shape[0],), -1)

        dn_cost = self.matcher.cost_matrix(
            dn_bboxes, dn_scores, gt_bboxes, gt_cls, image_shape=image_shape
        ).view(num_groups, num_gt, num_gt)
        main_cost = self.matcher.cost_matrix(
            pred_bboxes, pred_scores, gt_bboxes, gt_cls, image_shape=image_shape
        )
        assigned = gt_cls.new_full((num_groups * num_gt,), -1)

        for group_idx in range(num_groups):
            all_cost = torch.cat([dn_cost[group_idx], main_cost], dim=0)
            row_ind, col_ind = linear_sum_assignment(all_cost.detach().cpu())
            row = torch.as_tensor(row_ind, dtype=torch.long, device=gt_bboxes.device)
            col = torch.as_tensor(col_ind, dtype=torch.long, device=gt_bboxes.device)
            keep = row < num_gt
            assigned[group_idx * num_gt + row[keep]] = col[keep]
        return assigned


class RHINOOBBLoss(RTDETROBBLoss):
    """RHINO OBB loss with RHINO matcher and positive-Hungarian denoising."""

    DEFAULT_DN_COSTS = (
        {"type": "focal", "weight": 2.0},
        {"type": "hausdorff", "weight": 5.0, "num_points": 4},
        {
            "type": "gdcost",
            "loss_type": "kld",
            "fun": "log1p",
            "tau": 1.0,
            "alpha": 1.0,
            "sqrt": False,
            "weight": 5.0,
        },
    )

    def __init__(
        self,
        nc: int = 80,
        matcher_costs: list[dict[str, Any]] | None = None,
        dn_matcher_costs: list[dict[str, Any]] | None = None,
        loss_weights: dict[str, float] | None = None,
        loss_types: dict[str, Any] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
        use_shoreline_prior_loss: bool = False,
        use_land_water_prior_loss: bool = False,
        shoreline_prior_point_mode: str = "center",
        shoreline_prior_weight: float = 1.0,
        land_water_prior_weight: float = 1.0,
        shoreline_prior_max_dist: float = 128.0,
        land_water_prior_land_threshold: float = 0.05,
        land_water_prior_exp_beta: float = 4.0,
    ):
        if loss_weights is None:
            loss_weights = {
                "class": 1.0,
                "bbox": 5.0,
                "giou": 5.0,
                "no_object": 0.0,
                "mask": 1.0,
                "dice": 1.0,
            }
        super().__init__(
            nc=nc,
            loss_gain=loss_weights,
            aux_loss=aux_loss,
            use_fl=use_fl,
            use_vfl=False,
            gamma=gamma,
            alpha=alpha,
            use_shoreline_prior_loss=use_shoreline_prior_loss,
            use_land_water_prior_loss=use_land_water_prior_loss,
            shoreline_prior_point_mode=shoreline_prior_point_mode,
            shoreline_prior_weight=shoreline_prior_weight,
            land_water_prior_weight=land_water_prior_weight,
            shoreline_prior_max_dist=shoreline_prior_max_dist,
            land_water_prior_land_threshold=land_water_prior_land_threshold,
            land_water_prior_exp_beta=land_water_prior_exp_beta,
        )
        self.matcher = RHINOHungarianMatcher(costs=matcher_costs, use_fl=use_fl, alpha=alpha, gamma=gamma)
        self.dn_assigner = DNGroupHungarianAssigner(
            costs=list(deepcopy(dn_matcher_costs) if dn_matcher_costs is not None else deepcopy(self.DEFAULT_DN_COSTS)),
            use_fl=use_fl,
            alpha=alpha,
            gamma=gamma,
        )
        self.loss_types = {"bbox": "l1", "giou": "kld"}
        self.focal_gamma = float(gamma)
        self.focal_alpha = float(alpha)
        self.bg_cls_weight = 0.0
        if loss_types:
            self.loss_types.update(deepcopy(loss_types))

    def _get_loss_bbox(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        postfix: str = "",
        image_shapes: list[tuple[int, int]] | torch.Tensor | None = None,
        image_indices: torch.Tensor | None = None,
        normalizer: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute normalized 5D L1 and pixel/radian RHINO geometry loss."""
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        if len(gt_bboxes) == 0:
            zero = pred_bboxes.sum() * 0.0
            return {
                name_bbox: zero,
                name_giou: zero,
            }

        pred_boxes = torch.cat(
            (pred_bboxes[..., :2], pred_bboxes[..., 2:4].clamp_min(1e-6), pred_bboxes[..., 4:5]), dim=-1
        )
        gt_boxes = torch.cat(
            (gt_bboxes[..., :2], gt_bboxes[..., 2:4].clamp_min(1e-6), gt_bboxes[..., 4:5]), dim=-1
        )
        denominator = (
            _distributed_mean(len(gt_boxes), pred_boxes.device)
            if normalizer is None
            else torch.as_tensor(normalizer, dtype=pred_boxes.dtype, device=pred_boxes.device).clamp_min(1.0)
        )
        loss_bbox = self.loss_gain["bbox"] * F.l1_loss(
            pred_boxes, gt_boxes, reduction="sum"
        ) / denominator

        if image_shapes is None:
            image_shapes = [(1, 1)]
        if image_indices is None:
            image_indices = torch.zeros(len(pred_boxes), dtype=torch.long, device=pred_boxes.device)
        physical_pred, physical_gt = [], []
        for image_index in range(len(image_shapes)):
            selected = image_indices == image_index
            if selected.any():
                physical_pred.append(rhino_boxes_to_physical(pred_boxes[selected], image_shapes[image_index]))
                physical_gt.append(rhino_boxes_to_physical(gt_boxes[selected], image_shapes[image_index]))
        physical_pred = torch.cat(physical_pred)
        physical_gt = torch.cat(physical_gt)
        giou_type = str(self.loss_types.get("giou", "kld")).lower()
        if giou_type == "probiou":
            loss_giou = 1.0 - probiou(physical_pred, physical_gt)
        elif giou_type == "hausdorff":
            loss_giou = hausdorff_distance(physical_pred, physical_gt)
        else:
            loss_giou = kld_loss(
                xy_wh_r_2_xy_sigma(physical_pred),
                xy_wh_r_2_xy_sigma(physical_gt),
                fun="log1p",
                tau=1.0,
                alpha=1.0,
                sqrt=False,
            )
        loss_giou = self.loss_gain["giou"] * loss_giou.sum() / denominator
        return {name_bbox: loss_bbox.squeeze(), name_giou: loss_giou.squeeze()}

    def _focal_classification_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        normalizer: torch.Tensor | float,
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Reference sigmoid focal classification loss."""
        one_hot = F.one_hot(labels, self.nc + 1)[..., : self.nc].to(scores.dtype)
        probability = scores.sigmoid()
        cross_entropy = F.binary_cross_entropy_with_logits(scores, one_hot, reduction="none")
        probability_target = probability * one_hot + (1 - probability) * (1 - one_hot)
        alpha_target = self.focal_alpha * one_hot + (1 - self.focal_alpha) * (1 - one_hot)
        loss = cross_entropy * ((1 - probability_target) ** self.focal_gamma) * alpha_target
        denominator = torch.as_tensor(normalizer, dtype=scores.dtype, device=scores.device).clamp_min(1.0)
        return {f"loss_class{postfix}": self.loss_gain["class"] * loss.sum() / denominator}

    def _single_matching_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
    ) -> dict[str, torch.Tensor]:
        """Match and score one encoder/decoder layer using PHC-Haus costs."""
        gt_bboxes = batch["bboxes"]
        gt_cls = batch["cls"]
        gt_groups = batch["gt_groups"]
        image_shapes = batch.get("img_shapes", [(1, 1)] * pred_bboxes.shape[0])
        matches = self.matcher(
            pred_bboxes,
            pred_scores,
            gt_bboxes,
            gt_cls,
            gt_groups,
            image_shapes=image_shapes,
        )
        batch_indices = torch.cat(
            [torch.full_like(source, index) for index, (source, _) in enumerate(matches)]
        )
        source_indices = torch.cat([source for source, _ in matches])
        target_indices = torch.cat([target for _, target in matches])

        labels = torch.full(
            pred_scores.shape[:2], self.nc, dtype=gt_cls.dtype, device=pred_scores.device
        )
        if target_indices.numel():
            labels[batch_indices, source_indices] = gt_cls[target_indices]
        positive_normalizer = _distributed_mean(target_indices.numel(), pred_scores.device)
        losses = self._focal_classification_loss(pred_scores, labels, positive_normalizer, postfix)

        if target_indices.numel():
            assigned_pred = pred_bboxes[batch_indices, source_indices]
            assigned_gt = gt_bboxes[target_indices]
            losses.update(
                self._get_loss_bbox(
                    assigned_pred,
                    assigned_gt,
                    postfix,
                    image_shapes=image_shapes,
                    image_indices=batch_indices,
                    normalizer=positive_normalizer,
                )
            )
        else:
            zero = pred_bboxes.sum() * 0.0
            losses.update({f"loss_bbox{postfix}": zero, f"loss_giou{postfix}": zero})
        return losses

    def _compute_matching_losses(
        self, pred_bboxes: torch.Tensor, pred_scores: torch.Tensor, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Compute final plus encoder/intermediate auxiliary RHINO losses."""
        final_losses = self._single_matching_loss(pred_bboxes[-1], pred_scores[-1], batch)
        if self.aux_loss and len(pred_bboxes) > 1:
            auxiliary = [
                self._single_matching_loss(boxes, scores, batch)
                for boxes, scores in zip(pred_bboxes[:-1], pred_scores[:-1])
            ]
            final_losses.update(
                {
                    "loss_class_aux": torch.stack([loss["loss_class"] for loss in auxiliary]).sum(),
                    "loss_bbox_aux": torch.stack([loss["loss_bbox"] for loss in auxiliary]).sum(),
                    "loss_giou_aux": torch.stack([loss["loss_giou"] for loss in auxiliary]).sum(),
                }
            )
        else:
            zero = pred_bboxes.sum() * 0.0
            final_losses.update({"loss_class_aux": zero, "loss_bbox_aux": zero, "loss_giou_aux": zero})
        return final_losses

    def _dn_single(
        self,
        dn_bboxes: torch.Tensor,
        dn_scores: torch.Tensor,
        matching_bboxes: torch.Tensor,
        matching_scores: torch.Tensor,
        batch: dict[str, Any],
        dn_meta: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one PHC denoising layer."""
        device = dn_scores.device
        batch_size, num_dn = dn_scores.shape[:2]
        gt_cls = batch["cls"].to(device=device, dtype=torch.long).view(-1)
        gt_bboxes = batch["bboxes"].to(device=device)
        gt_groups = batch["gt_groups"]
        image_shapes = batch.get("img_shapes", [(1, 1)] * batch_size)

        num_groups = int(dn_meta["num_denoising_groups"])
        if num_groups <= 0 or num_dn == 0:
            zero = dn_scores.sum() * 0.0
            return zero, zero, zero
        queries_per_group = int(num_dn / num_groups)
        labels = torch.full((batch_size, num_dn), self.nc, device=device, dtype=gt_cls.dtype)
        pos_pred_boxes: list[torch.Tensor] = []
        pos_target_boxes: list[torch.Tensor] = []
        pos_image_indices: list[torch.Tensor] = []
        total_new_pos = 0
        total_new_neg = 0
        total_original_pos = 0
        gt_offset = 0

        for batch_index, num_gt in enumerate(gt_groups):
            if num_gt == 0:
                continue

            gt_slice = slice(gt_offset, gt_offset + num_gt)
            gt_boxes_img = gt_bboxes[gt_slice]
            gt_cls_img = gt_cls[gt_slice]

            base = torch.arange(num_groups, device=device)[:, None] * queries_per_group
            positive_indices = (base + torch.arange(num_gt, device=device)[None]).reshape(-1)
            negative_indices = positive_indices + queries_per_group // 2

            repeated_boxes = gt_boxes_img.repeat(num_groups, 1)
            pos_pred_boxes.append(dn_bboxes[batch_index, positive_indices])
            pos_target_boxes.append(repeated_boxes)
            pos_image_indices.append(
                torch.full((len(positive_indices),), batch_index, dtype=torch.long, device=device)
            )
            total_original_pos += len(positive_indices)

            assigned = self.dn_assigner.assign(
                pred_bboxes=matching_bboxes[batch_index],
                pred_scores=matching_scores[batch_index],
                dn_bboxes=dn_bboxes[batch_index, positive_indices],
                dn_scores=dn_scores[batch_index, positive_indices],
                gt_bboxes=gt_boxes_img,
                gt_cls=gt_cls_img,
                num_groups=num_groups,
                image_shape=image_shapes[batch_index],
            )
            expected = torch.arange(num_gt, device=device).repeat(num_groups)
            accepted = assigned == expected
            repeated_classes = gt_cls_img.repeat(num_groups)
            labels[batch_index, positive_indices[accepted]] = repeated_classes[accepted]
            total_new_pos += int(accepted.sum())
            total_new_neg += int((~accepted).sum()) + len(negative_indices)

            gt_offset += num_gt

        class_normalizer = _distributed_mean(
            total_new_pos + total_new_neg * self.bg_cls_weight, device
        )
        class_loss = self._focal_classification_loss(dn_scores, labels, class_normalizer)["loss_class"]
        if pos_pred_boxes:
            regression_normalizer = _distributed_mean(total_original_pos, device)
            bbox_losses = self._get_loss_bbox(
                torch.cat(pos_pred_boxes),
                torch.cat(pos_target_boxes),
                image_shapes=image_shapes,
                image_indices=torch.cat(pos_image_indices),
                normalizer=regression_normalizer,
            )
            bbox_loss = bbox_losses["loss_bbox"]
            giou_loss = bbox_losses["loss_giou"]
        else:
            bbox_loss = dn_bboxes.sum() * 0.0
            giou_loss = dn_bboxes.sum() * 0.0
        self.last_dn_targets = {
            "labels": labels.detach(),
            "new_positive_count": total_new_pos,
            "new_negative_count": total_new_neg,
            "original_positive_count": total_original_pos,
        }
        return class_loss, bbox_loss, giou_loss

    def _compute_dn_losses(
        self,
        dn_bboxes: torch.Tensor,
        dn_scores: torch.Tensor,
        matching_bboxes: torch.Tensor,
        matching_scores: torch.Tensor,
        batch: dict[str, Any],
        dn_meta: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if dn_bboxes.shape[2] == 0 or int(dn_meta.get("num_denoising_groups", 0)) <= 0:
            zero = dn_bboxes.sum() * 0.0 + dn_scores.sum() * 0.0
            return {
                "loss_class_dn": zero,
                "loss_bbox_dn": zero,
                "loss_giou_dn": zero,
                "loss_class_aux_dn": zero,
                "loss_bbox_aux_dn": zero,
                "loss_giou_aux_dn": zero,
            }
        layer_losses = [
            self._dn_single(dn_box, dn_score, mtc_box, mtc_score, batch, dn_meta)
            for dn_box, dn_score, mtc_box, mtc_score in zip(dn_bboxes, dn_scores, matching_bboxes, matching_scores)
        ]
        main_cls, main_bbox, main_giou = layer_losses[-1]
        if self.aux_loss and len(layer_losses) > 1:
            aux_cls = torch.stack([x[0] for x in layer_losses[:-1]]).sum()
            aux_bbox = torch.stack([x[1] for x in layer_losses[:-1]]).sum()
            aux_giou = torch.stack([x[2] for x in layer_losses[:-1]]).sum()
        else:
            aux_cls = main_cls * 0.0
            aux_bbox = main_bbox * 0.0
            aux_giou = main_giou * 0.0
        return {
            "loss_class_dn": main_cls,
            "loss_bbox_dn": main_bbox,
            "loss_giou_dn": main_giou,
            "loss_class_aux_dn": aux_cls,
            "loss_bbox_aux_dn": aux_bbox,
            "loss_giou_aux_dn": aux_giou,
        }

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_bboxes, pred_scores = preds
        self.device = pred_bboxes.device
        total_loss = self._compute_matching_losses(pred_bboxes, pred_scores, batch)

        if dn_meta is not None and dn_bboxes is not None and dn_scores is not None:
            total_loss.update(
                self._compute_dn_losses(
                    dn_bboxes,
                    dn_scores,
                    pred_bboxes[1:],
                    pred_scores[1:],
                    batch,
                    dn_meta,
                )
            )
        else:
            zero = pred_bboxes.sum() * 0.0
            total_loss.update(
                {
                    "loss_class_dn": zero,
                    "loss_bbox_dn": zero,
                    "loss_giou_dn": zero,
                    "loss_class_aux_dn": zero,
                    "loss_bbox_aux_dn": zero,
                    "loss_giou_aux_dn": zero,
                }
            )

        loss_shoreline_prior = pred_bboxes.new_tensor(0.0)
        loss_land_water_prior = pred_bboxes.new_tensor(0.0)
        land_water_map = batch.get("land_water_mask")
        if (self.use_shoreline_prior_loss or self.use_land_water_prior_loss) and land_water_map is not None:
            final_bboxes_norm = pred_bboxes[-1]
            final_bboxes = final_bboxes_norm.clone()
            final_scores = pred_scores[-1]
            image_shapes = batch.get("img_shapes")
            if image_shapes is None:
                _, _, height, width = land_water_map.shape
                image_shapes = final_bboxes.new_tensor([[height, width]]).expand(final_bboxes.shape[0], -1)
            else:
                image_shapes = torch.as_tensor(
                    image_shapes, device=final_bboxes.device, dtype=final_bboxes.dtype
                ).reshape(final_bboxes.shape[0], 2)
            heights = image_shapes[:, 0].view(-1, 1)
            widths = image_shapes[:, 1].view(-1, 1)
            final_bboxes[..., 0] *= widths
            final_bboxes[..., 2] *= widths
            final_bboxes[..., 1] *= heights
            final_bboxes[..., 3] *= heights
            final_bboxes[..., 4] *= torch.pi

            loss_shoreline_prior, loss_land_water_prior = _compute_obb_spatial_prior_losses(
                final_bboxes,
                final_scores.sigmoid().amax(-1),
                land_water_map,
                batch.get("shoreline_distance_map"),
                point_mode=self.shoreline_prior_point_mode,
                shoreline_prior_max_dist=self.shoreline_prior_max_dist,
                land_threshold=self.land_water_prior_land_threshold,
                land_beta=self.land_water_prior_exp_beta,
            )
            loss_shoreline_prior *= self.shoreline_prior_weight
            loss_land_water_prior *= self.land_water_prior_weight

        total_loss["loss_shoreline_prior"] = loss_shoreline_prior
        total_loss["loss_land_water_prior"] = loss_land_water_prior
        return total_loss
