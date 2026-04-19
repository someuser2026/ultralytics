# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RHINO-specific matcher and loss utilities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from ultralytics.utils.loss import DETRLoss, RTDETROBBLoss, _compute_obb_spatial_prior_losses
from ultralytics.utils.metrics import batch_probiou, probiou
from ultralytics.utils.ops import xywhr2xyxyxyxy


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
        distance = torch.log1p(distance)
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
        {"type": "gdcost", "loss_type": "kld", "fun": "log1p", "tau": 1.0, "alpha": 1.0, "sqrt": False, "weight": 2.0},
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
        **_: Any,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        bs = pred_scores.shape[0]
        if sum(gt_groups) == 0:
            return [(torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)) for _ in range(bs)]

        results = []
        gt_offset = 0
        for batch_idx, num_gt in enumerate(gt_groups):
            if num_gt == 0:
                results.append((torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)))
                continue
            gt_slice = slice(gt_offset, gt_offset + num_gt)
            cost = self.cost_matrix(
                pred_bboxes[batch_idx],
                pred_scores[batch_idx],
                gt_bboxes[gt_slice],
                gt_cls[gt_slice],
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
                cost = hausdorff_pairwise_cost(pred_bboxes, gt_bboxes, num_points=int(cfg.get("num_points", 4)))
            elif name in {"gdcost", "gd", "kld"}:
                pred_gaussian = xy_wh_r_2_xy_sigma(pred_bboxes)
                gt_gaussian = xy_wh_r_2_xy_sigma(gt_bboxes)
                cost = pairwise_kld_loss(
                    pred_gaussian,
                    gt_gaussian,
                    fun=str(cfg.get("fun", "log1p")),
                    tau=float(cfg.get("tau", 1.0)),
                    alpha=float(cfg.get("alpha", 1.0)),
                    sqrt=bool(cfg.get("sqrt", False)),
                )
            elif name in {"probiou", "rotatediou", "iou"}:
                cost = 1.0 - batch_probiou(gt_bboxes, pred_bboxes).transpose(0, 1)
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
    ) -> torch.Tensor:
        num_gt = gt_bboxes.shape[0]
        if num_gt == 0 or dn_bboxes.numel() == 0:
            return gt_cls.new_full((dn_bboxes.shape[0],), -1)

        dn_cost = self.matcher.cost_matrix(dn_bboxes, dn_scores, gt_bboxes, gt_cls).view(num_groups, num_gt, num_gt)
        main_cost = self.matcher.cost_matrix(pred_bboxes, pred_scores, gt_bboxes, gt_cls)
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
        {"type": "center_l1", "weight": 5.0},
        {"type": "gdcost", "loss_type": "kld", "fun": "log1p", "tau": 1.0, "alpha": 1.0, "sqrt": False, "weight": 2.0},
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
        )
        self.matcher = RHINOHungarianMatcher(costs=matcher_costs, use_fl=use_fl, alpha=alpha, gamma=gamma)
        self.dn_assigner = DNGroupHungarianAssigner(
            costs=list(deepcopy(dn_matcher_costs) if dn_matcher_costs is not None else deepcopy(self.DEFAULT_DN_COSTS)),
            use_fl=use_fl,
            alpha=alpha,
            gamma=gamma,
        )
        self.loss_types = {"bbox": "l1", "giou": "kld"}
        if loss_types:
            self.loss_types.update(deepcopy(loss_types))

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        if len(gt_bboxes) == 0:
            return {
                name_bbox: torch.tensor(0.0, device=self.device),
                name_giou: torch.tensor(0.0, device=self.device),
            }

        bbox_dim = 5 if str(self.loss_types.get("bbox", "l1")).lower() in {"l1", "rboxl1", "xywha"} else 4
        loss_bbox = self.loss_gain["bbox"] * F.l1_loss(
            pred_bboxes[..., :bbox_dim], gt_bboxes[..., :bbox_dim], reduction="sum"
        ) / len(gt_bboxes)

        giou_type = str(self.loss_types.get("giou", "kld")).lower()
        if giou_type == "probiou":
            loss_giou = 1.0 - probiou(pred_bboxes[..., :5], gt_bboxes[..., :5])
        elif giou_type == "hausdorff":
            loss_giou = hausdorff_distance(pred_bboxes[..., :5], gt_bboxes[..., :5])
        else:
            loss_giou = kld_loss(
                xy_wh_r_2_xy_sigma(pred_bboxes[..., :5]),
                xy_wh_r_2_xy_sigma(gt_bboxes[..., :5]),
                fun="log1p",
                tau=1.0,
                alpha=1.0,
                sqrt=False,
            )
        loss_giou = self.loss_gain["giou"] * (loss_giou.sum() / len(gt_bboxes))
        return {name_bbox: loss_bbox.squeeze(), name_giou: loss_giou.squeeze()}

    def _dn_single(
        self,
        dn_bboxes: torch.Tensor,
        dn_scores: torch.Tensor,
        matching_bboxes: torch.Tensor,
        matching_scores: torch.Tensor,
        batch: dict[str, Any],
        dn_meta: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = dn_scores.device
        bs, num_dn = dn_scores.shape[:2]
        gt_cls = batch["cls"].to(device=device, dtype=torch.long).view(-1)
        gt_bboxes = batch["bboxes"].to(device=device)
        gt_groups = batch["gt_groups"]

        num_groups = int(dn_meta["num_denoising_groups"])
        queries_per_group = int(num_dn / num_groups)
        labels = torch.full((bs, num_dn), self.nc, device=device, dtype=gt_cls.dtype)
        gt_scores = torch.zeros((bs, num_dn), device=device)
        pos_pred_boxes: list[torch.Tensor] = []
        pos_target_boxes: list[torch.Tensor] = []
        total_new_pos = 0
        gt_offset = 0

        for batch_idx, num_gt in enumerate(gt_groups):
            if num_gt == 0:
                continue

            gt_slice = slice(gt_offset, gt_offset + num_gt)
            gt_boxes_img = gt_bboxes[gt_slice]
            gt_cls_img = gt_cls[gt_slice]

            base = torch.arange(num_groups, device=device)[:, None] * queries_per_group
            pos_inds = (base + torch.arange(num_gt, device=device)[None, :]).reshape(-1)

            repeated_boxes = gt_boxes_img.repeat(num_groups, 1)
            pos_pred_boxes.append(dn_bboxes[batch_idx, pos_inds])
            pos_target_boxes.append(repeated_boxes)

            assigned = self.dn_assigner.assign(
                pred_bboxes=matching_bboxes[batch_idx],
                pred_scores=matching_scores[batch_idx],
                dn_bboxes=dn_bboxes[batch_idx, pos_inds],
                dn_scores=dn_scores[batch_idx, pos_inds],
                gt_bboxes=gt_boxes_img,
                gt_cls=gt_cls_img,
                num_groups=num_groups,
            )
            expected = torch.arange(num_gt, device=device).repeat(num_groups)
            matched = assigned == expected
            if matched.any():
                labels[batch_idx, pos_inds[matched]] = gt_cls_img.repeat(num_groups)[matched]
                total_new_pos += int(matched.sum())

            gt_offset += num_gt

        class_loss = self._get_loss_class(dn_scores, labels, gt_scores, max(total_new_pos, 1))["loss_class"]
        if pos_pred_boxes:
            bbox_losses = self._get_loss_bbox(torch.cat(pos_pred_boxes, dim=0), torch.cat(pos_target_boxes, dim=0))
            bbox_loss = bbox_losses["loss_bbox"]
            giou_loss = bbox_losses["loss_giou"]
        else:
            bbox_loss = torch.tensor(0.0, device=device)
            giou_loss = torch.tensor(0.0, device=device)
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
            aux_cls = main_cls.new_tensor(0.0)
            aux_bbox = main_bbox.new_tensor(0.0)
            aux_giou = main_giou.new_tensor(0.0)
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
        total_loss = DETRLoss.forward(self, pred_bboxes, pred_scores, batch)

        if dn_meta is not None and dn_bboxes is not None and dn_scores is not None:
            total_loss.update(self._compute_dn_losses(dn_bboxes, dn_scores, pred_bboxes[1:], pred_scores[1:], batch, dn_meta))
        else:
            zero = pred_bboxes.new_tensor(0.0)
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
