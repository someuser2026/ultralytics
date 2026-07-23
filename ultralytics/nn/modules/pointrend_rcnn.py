# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Detectron2-style PointRend RCNN head for native Ultralytics models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ultralytics.utils.nms import TorchNMS

from .pointrend import PointRendTrainConfig, paste_roi_probabilities, point_sample, roi_points_to_image_points
from .rcnn import (
    HorizontalBoxCoder,
    _AxisRCNNBase,
    _clip_boxes,
    _match_hboxes,
    _merge_dict,
    _roi_align_multilevel,
    _split_targets,
)
from .roi import torchvision_native_roi_align

__all__ = ("PointRendRCNNHead",)


@dataclass(frozen=True)
class _PointRendRCNNConfig:
    """Architecture configuration for the self-contained PointRend mask branch."""

    coarse_pool_resolution: int = 14
    coarse_conv_dim: int = 256
    coarse_fc_dim: int = 1024
    coarse_num_fcs: int = 2
    coarse_output_resolution: int = 7
    point_hidden_dim: int = 256
    point_num_fcs: int = 3
    coarse_pred_each_layer: bool = True
    train_num_points: int = 196
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    subdivision_steps: int = 5
    subdivision_num_points: int = 784
    scale_factor: int = 2

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "_PointRendRCNNConfig":
        """Build and validate a PointRend RCNN configuration."""

        value = dict(value or {})
        cfg = cls(
            coarse_pool_resolution=int(value.get("coarse_pool_resolution", 14)),
            coarse_conv_dim=int(value.get("coarse_conv_dim", 256)),
            coarse_fc_dim=int(value.get("coarse_fc_dim", 1024)),
            coarse_num_fcs=int(value.get("coarse_num_fcs", 2)),
            coarse_output_resolution=int(value.get("coarse_output_resolution", 7)),
            point_hidden_dim=int(value.get("point_hidden_dim", 256)),
            point_num_fcs=int(value.get("point_num_fcs", 3)),
            coarse_pred_each_layer=bool(value.get("coarse_pred_each_layer", True)),
            train_num_points=int(value.get("train_num_points", 196)),
            oversample_ratio=float(value.get("oversample_ratio", 3.0)),
            importance_sample_ratio=float(value.get("importance_sample_ratio", 0.75)),
            subdivision_steps=int(value.get("subdivision_steps", 5)),
            subdivision_num_points=int(value.get("subdivision_num_points", 784)),
            scale_factor=int(value.get("scale_factor", 2)),
        )
        positive = {
            "coarse_pool_resolution": cfg.coarse_pool_resolution,
            "coarse_conv_dim": cfg.coarse_conv_dim,
            "coarse_fc_dim": cfg.coarse_fc_dim,
            "coarse_num_fcs": cfg.coarse_num_fcs,
            "coarse_output_resolution": cfg.coarse_output_resolution,
            "point_hidden_dim": cfg.point_hidden_dim,
            "point_num_fcs": cfg.point_num_fcs,
            "train_num_points": cfg.train_num_points,
            "oversample_ratio": cfg.oversample_ratio,
            "subdivision_num_points": cfg.subdivision_num_points,
            "scale_factor": cfg.scale_factor,
        }
        for name, number in positive.items():
            if number <= 0:
                raise ValueError(f"PointRend RCNN {name} must be positive, got {number}.")
        if not 0.0 <= cfg.importance_sample_ratio <= 1.0:
            raise ValueError(
                "PointRend RCNN importance_sample_ratio must be in [0, 1], "
                f"got {cfg.importance_sample_ratio}."
            )
        if cfg.subdivision_steps < 0:
            raise ValueError(f"PointRend RCNN subdivision_steps must be non-negative, got {cfg.subdivision_steps}.")
        if cfg.coarse_pool_resolution % 2:
            raise ValueError("PointRend RCNN coarse_pool_resolution must be even for the stride-2 coarse convolution.")
        return cfg

    def signature(self) -> tuple[Any, ...]:
        """Return all fields that determine PointRend RCNN topology or refinement behavior."""

        return tuple(getattr(self, name) for name in self.__dataclass_fields__)


def _regular_roi_grid(n: int, side: int, reference: Tensor) -> Tensor:
    """Return an ``n × side² × 2`` ROI-relative pixel-center grid."""

    ys = (torch.arange(side, device=reference.device, dtype=reference.dtype) + 0.5) / side
    xs = (torch.arange(side, device=reference.device, dtype=reference.dtype) + 0.5) / side
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, side * side, 2).expand(n, -1, -1)


def _select_class_channels(logits: Tensor, classes: Tensor) -> Tensor:
    """Select one class channel per instance while retaining the channel dimension."""

    if logits.shape[0] == 0:
        return logits[:, :1]
    rows = torch.arange(logits.shape[0], device=logits.device)
    return logits[rows, classes.long()][:, None]


def _paste_binary_masks(
    roi_logits: Tensor,
    boxes: Tensor,
    image_shape: tuple[int, int],
    threshold: float,
    chunk_size: int = 4,
) -> Tensor:
    """Paste and threshold small instance chunks without retaining a full float image-mask batch."""

    masks = roi_logits.new_zeros(
        (roi_logits.shape[0], int(image_shape[0]), int(image_shape[1])),
        dtype=torch.bool,
    )
    for start in range(0, roi_logits.shape[0], chunk_size):
        stop = min(start + chunk_size, roi_logits.shape[0])
        probabilities = paste_roi_probabilities(
            roi_logits[start:stop],
            boxes[start:stop],
            image_shape,
            max_chunk_size=1,
        )
        masks[start:stop] = probabilities >= threshold
    return masks


class _FastRCNNBoxHead(nn.Module):
    """Detectron2-equivalent two-FC Fast R-CNN classifier and class-specific regressor."""

    def __init__(self, in_channels: int, num_classes: int, pool_size: int = 7, hidden_dim: int = 1024):
        super().__init__()
        self.num_classes = int(num_classes)
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, self.num_classes + 1)
        self.bbox_pred = nn.Linear(hidden_dim, self.num_classes * 4)

        for layer in (self.fc1, self.fc2):
            # Caffe2 XavierFill is equivalent to Kaiming-uniform with a=1.
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Predict class scores and class-specific box deltas."""

        features = F.relu(self.fc1(features.flatten(1)), inplace=True)
        features = F.relu(self.fc2(features), inplace=True)
        return self.cls_score(features), self.bbox_pred(features).reshape(-1, self.num_classes, 4)


class _PointRendCoarseHead(nn.Module):
    """P2-only stride-2 convolution and two-FC coarse mask predictor."""

    def __init__(self, in_channels: int, num_classes: int, cfg: _PointRendRCNNConfig):
        super().__init__()
        self.num_classes = int(num_classes)
        self.output_resolution = cfg.coarse_output_resolution
        self.reduce_channel_dim = (
            nn.Conv2d(in_channels, cfg.coarse_conv_dim, 1) if in_channels > cfg.coarse_conv_dim else nn.Identity()
        )
        if in_channels < cfg.coarse_conv_dim:
            raise ValueError(
                "Detectron2 PointRend coarse head requires P2 channels >= coarse_conv_dim, "
                f"got {in_channels} < {cfg.coarse_conv_dim}."
            )
        self.reduce_spatial_dim = nn.Conv2d(cfg.coarse_conv_dim, cfg.coarse_conv_dim, 2, stride=2)
        reduced_side = cfg.coarse_pool_resolution // 2
        input_dim = cfg.coarse_conv_dim * reduced_side * reduced_side
        self.fcs = nn.ModuleList()
        for _ in range(cfg.coarse_num_fcs):
            layer = nn.Linear(input_dim, cfg.coarse_fc_dim)
            self.fcs.append(layer)
            input_dim = cfg.coarse_fc_dim
        self.predictor = nn.Linear(
            input_dim,
            self.num_classes * cfg.coarse_output_resolution * cfg.coarse_output_resolution,
        )

        for layer in (self.reduce_channel_dim, self.reduce_spatial_dim):
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        for layer in self.fcs:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, roi_features: Tensor) -> Tensor:
        """Return class-specific coarse logits shaped ``N × C × 7 × 7``."""

        features = F.relu(self.reduce_channel_dim(roi_features), inplace=True)
        features = F.relu(self.reduce_spatial_dim(features), inplace=True).flatten(1)
        for layer in self.fcs:
            features = F.relu(layer(features), inplace=True)
        return self.predictor(features).reshape(
            roi_features.shape[0],
            self.num_classes,
            self.output_resolution,
            self.output_resolution,
        )


class _StandardPointHead(nn.Module):
    """Detectron2 StandardPointHead with class-specific coarse features and predictions."""

    def __init__(self, fine_channels: int, num_classes: int, cfg: _PointRendRCNNConfig):
        super().__init__()
        self.num_classes = int(num_classes)
        self.coarse_pred_each_layer = cfg.coarse_pred_each_layer
        input_channels = fine_channels + self.num_classes
        self.fcs = nn.ModuleList()
        for _ in range(cfg.point_num_fcs):
            layer = nn.Conv1d(input_channels, cfg.point_hidden_dim, 1)
            self.fcs.append(layer)
            input_channels = cfg.point_hidden_dim + (
                self.num_classes if self.coarse_pred_each_layer else 0
            )
        self.predictor = nn.Conv1d(input_channels, self.num_classes, 1)

        for layer in self.fcs:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.predictor.weight, std=0.001)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, fine_features: Tensor, coarse_features: Tensor) -> Tensor:
        """Predict class-specific logits at sampled points."""

        features = torch.cat((fine_features, coarse_features), dim=1)
        for layer in self.fcs:
            features = F.relu(layer(features), inplace=True)
            if self.coarse_pred_each_layer:
                features = torch.cat((features, coarse_features), dim=1)
        return self.predictor(features)


class _PointRendMaskBranch(nn.Module):
    """Self-contained Detectron2 PointRend coarse and point mask branches."""

    self_contained_pointrend = True

    def __init__(self, p2_channels: int, num_classes: int, cfg: _PointRendRCNNConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(num_classes)
        self.source_channels = (int(p2_channels),)
        self.coarse_head = _PointRendCoarseHead(p2_channels, self.num_classes, cfg)
        self.point_head = _StandardPointHead(p2_channels, self.num_classes, cfg)
        self.train_config = PointRendTrainConfig(
            mode="joint",
            train_num_points=cfg.train_num_points,
            oversample_ratio=cfg.oversample_ratio,
            importance_sample_ratio=cfg.importance_sample_ratio,
        )

    def architecture_signature(self) -> tuple[Any, ...]:
        """Return a checkpoint compatibility signature."""

        return ("detectron2_rcnn", self.source_channels, self.num_classes, self.cfg.signature())

    def effective_subdivision_schedule(self) -> tuple[int, int]:
        """Apply Detectron2's optimization to the configured 7/5 inference schedule."""

        resolution = self.cfg.coarse_output_resolution
        steps = self.cfg.subdivision_steps
        while 4 * resolution**2 <= self.cfg.subdivision_num_points:
            resolution *= 2
            steps -= 1
        if steps < 0:
            raise ValueError(
                "PointRend subdivision optimization consumed more steps than configured: "
                f"resolution={self.cfg.coarse_output_resolution}, steps={self.cfg.subdivision_steps}, "
                f"points={self.cfg.subdivision_num_points}."
            )
        return resolution, steps

    def final_resolution(self) -> int:
        """Return the final square ROI mask resolution."""

        resolution, steps = self.effective_subdivision_schedule()
        return resolution * self.cfg.scale_factor**steps

    @staticmethod
    def _sample_p2(
        p2: Tensor,
        boxes: Tensor,
        batch_indices: Tensor,
        point_coords: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        """Sample direct P2 features at ROI-relative coordinates."""

        image_points = roi_points_to_image_points(point_coords, boxes, image_shape)
        per_instance = p2.index_select(0, batch_indices.long())
        return point_sample(per_instance, image_points)

    def coarse_logits(
        self,
        p2: Tensor,
        boxes: Tensor,
        batch_indices: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        """Sample the complete 14×14 P2 grid and predict 7×7 coarse logits."""

        if boxes.shape[0] == 0:
            side = self.cfg.coarse_output_resolution
            return p2.new_zeros((0, self.num_classes, side, side))
        grid = _regular_roi_grid(boxes.shape[0], self.cfg.coarse_pool_resolution, p2)
        roi_features = self._sample_p2(p2, boxes, batch_indices, grid, image_shape).reshape(
            boxes.shape[0],
            p2.shape[1],
            self.cfg.coarse_pool_resolution,
            self.cfg.coarse_pool_resolution,
        )
        return self.coarse_head(roi_features)

    def _point_logits(
        self,
        p2: Tensor,
        boxes: Tensor,
        batch_indices: Tensor,
        point_coords: Tensor,
        coarse_logits: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        """Predict all class logits at the requested ROI points."""

        fine = self._sample_p2(p2, boxes, batch_indices, point_coords, image_shape)
        coarse = point_sample(coarse_logits, point_coords)
        return self.point_head(fine, coarse)

    def _sample_train_points(self, coarse_logits: Tensor, classes: Tensor) -> Tensor:
        """Sample uncertain and random points using the GT-class coarse channel."""

        n = coarse_logits.shape[0]
        count = self.train_config.train_num_points
        if n == 0:
            return coarse_logits.new_zeros((0, count, 2))
        candidate_count = max(int(count * self.train_config.oversample_ratio), count)
        candidates = torch.rand(
            (n, candidate_count, 2),
            device=coarse_logits.device,
            dtype=coarse_logits.dtype,
        )
        sampled = point_sample(coarse_logits, candidates)
        uncertainty = -_select_class_channels(sampled, classes)[:, 0].abs()
        uncertain_count = int(count * self.train_config.importance_sample_ratio)
        random_count = count - uncertain_count
        if uncertain_count:
            indices = uncertainty.topk(uncertain_count, dim=1).indices
            selected = candidates.gather(1, indices[..., None].expand(-1, -1, 2))
        else:
            selected = candidates[:, :0]
        if random_count:
            random_points = torch.rand(
                (n, random_count, 2),
                device=coarse_logits.device,
                dtype=coarse_logits.dtype,
            )
            selected = torch.cat((selected, random_points), dim=1)
        return selected

    @staticmethod
    def _selected_gt_masks(
        gt_masks: list[Tensor | None],
        batch_indices: Tensor,
        gt_indices: Tensor,
        reference: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        """Gather the full-resolution GT bitmask corresponding to each positive ROI."""

        selected = []
        for batch_index, gt_index in zip(batch_indices.tolist(), gt_indices.tolist()):
            masks = gt_masks[int(batch_index)]
            if masks is None or masks.numel() == 0 or gt_index < 0:
                selected.append(reference.new_zeros(image_shape))
            else:
                selected.append(masks[int(gt_index)].to(device=reference.device, dtype=reference.dtype))
        return (
            torch.stack(selected, dim=0)[:, None]
            if selected
            else reference.new_zeros((0, 1, image_shape[0], image_shape[1]))
        )

    @staticmethod
    def aligned_coarse_targets(gt_masks: Tensor, boxes: Tensor, output_resolution: int) -> Tensor:
        """Create Detectron2 BitMasks.crop_and_resize-equivalent binary targets."""

        if boxes.shape[0] == 0:
            return boxes.new_zeros((0, output_resolution, output_resolution), dtype=torch.bool)
        local_indices = torch.arange(boxes.shape[0], device=boxes.device, dtype=torch.float32)[:, None]
        rois = torch.cat((local_indices, boxes.float()), dim=1)
        targets = torchvision_native_roi_align(
            gt_masks.float(),
            rois,
            output_size=output_resolution,
            spatial_scale=1.0,
            sampling_ratio=0,
            aligned=True,
        )[:, 0]
        return targets >= 0.5

    def losses(
        self,
        p2: Tensor,
        boxes: Tensor,
        batch_indices: Tensor,
        classes: Tensor,
        gt_indices: Tensor,
        gt_masks: list[Tensor | None],
        image_shape: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        """Compute class-specific coarse and point losses."""

        if boxes.shape[0] == 0:
            coarse_zero = sum(
                (parameter.sum() * 0.0 for parameter in self.coarse_head.parameters()),
                p2.sum() * 0.0,
            )
            point_zero = sum(
                (parameter.sum() * 0.0 for parameter in self.point_head.parameters()),
                p2.sum() * 0.0,
            )
            return coarse_zero, point_zero

        coarse_logits = self.coarse_logits(p2, boxes, batch_indices, image_shape)
        full_gt = self._selected_gt_masks(gt_masks, batch_indices, gt_indices, p2, image_shape)
        coarse_targets = self.aligned_coarse_targets(
            full_gt,
            boxes,
            self.cfg.coarse_output_resolution,
        )
        selected_coarse = _select_class_channels(coarse_logits, classes)[:, 0]
        coarse_loss = F.binary_cross_entropy_with_logits(selected_coarse, coarse_targets.to(selected_coarse.dtype))

        point_coords = self._sample_train_points(coarse_logits, classes)
        point_logits = self._point_logits(
            p2,
            boxes,
            batch_indices,
            point_coords,
            coarse_logits,
            image_shape,
        )
        selected_points = _select_class_channels(point_logits, classes)[:, 0]
        image_points = roi_points_to_image_points(point_coords, boxes, image_shape)
        point_targets = point_sample(full_gt, image_points.to(dtype=full_gt.dtype))[:, 0]
        point_loss = F.binary_cross_entropy_with_logits(selected_points, point_targets)
        return coarse_loss, point_loss * self.train_config.loss_weight

    @staticmethod
    def _uncertain_grid_points(logits: Tensor, classes: Tensor, count: int) -> tuple[Tensor, Tensor]:
        """Select uncertain grid indices from each predicted-class mask channel."""

        n, _, height, width = logits.shape
        count = min(int(count), height * width)
        selected = _select_class_channels(logits, classes)
        indices = (-selected.abs()).flatten(1).topk(count, dim=1).indices
        xs = (indices.remainder(width).to(logits.dtype) + 0.5) / width
        ys = (indices.div(width, rounding_mode="floor").to(logits.dtype) + 0.5) / height
        return indices, torch.stack((xs, ys), dim=-1)

    def refine(
        self,
        p2: Tensor,
        boxes: Tensor,
        batch_indices: Tensor,
        classes: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        """Run Detectron2's dense initialization and adaptive PointRend subdivisions."""

        resolution, steps = self.effective_subdivision_schedule()
        if boxes.shape[0] == 0:
            final_side = resolution * self.cfg.scale_factor**steps
            return p2.new_zeros((0, self.num_classes, final_side, final_side))
        coarse_logits = self.coarse_logits(p2, boxes, batch_indices, image_shape)
        dense_grid = _regular_roi_grid(boxes.shape[0], resolution, p2)
        refined = self._point_logits(
            p2,
            boxes,
            batch_indices,
            dense_grid,
            coarse_logits,
            image_shape,
        ).reshape(boxes.shape[0], self.num_classes, resolution, resolution)

        for _ in range(steps):
            refined = F.interpolate(
                refined,
                scale_factor=self.cfg.scale_factor,
                mode="bilinear",
                align_corners=False,
            )
            indices, point_coords = self._uncertain_grid_points(
                refined,
                classes,
                self.cfg.subdivision_num_points,
            )
            point_logits = self._point_logits(
                p2,
                boxes,
                batch_indices,
                point_coords,
                coarse_logits,
                image_shape,
            )
            flat = refined.flatten(2)
            flat = flat.scatter(2, indices[:, None].expand(-1, self.num_classes, -1), point_logits)
            refined = flat.reshape_as(refined)
        return refined


class PointRendRCNNHead(_AxisRCNNBase):
    """Native R50-FPN RCNN head with a self-contained Detectron2-style PointRend mask branch."""

    default_cfg = _merge_dict(
        _AxisRCNNBase.default_cfg,
        {
            "rpn": {
                "pre_nms_topk_train": 2000,
                "post_nms_topk_train": 1000,
                "pre_nms_topk_test": 1000,
                "post_nms_topk_test": 1000,
                "beta": 0.0,
            },
            "roi": {"pool_size": 7, "sampling_ratio": 0},
            "test": {"score_thresh": 0.05, "nms_iou": 0.5, "max_dets": 100, "mask_threshold": 0.5},
            "bbox_head": {
                "hidden_dim": 1024,
                "loss": "smooth_l1",
                "beta": 0.0,
                "train_on_pred_boxes": True,
                "class_agnostic": False,
            },
            "pointrend": {},
        },
    )

    def __init__(self, in_channels: list[int], nc: int, cfg: dict | None = None):
        super().__init__(in_channels, nc, cfg=cfg, with_mask=False, cascade=False)
        if len(in_channels) < 5:
            raise ValueError(f"PointRendRCNNHead expects P2-P6 features, got {len(in_channels)} levels.")
        if len(set(in_channels)) != 1:
            raise ValueError(f"PointRendRCNNHead expects equal-width FPN features, got {in_channels}.")
        self.bbox_heads = nn.ModuleList(
            [
                _FastRCNNBoxHead(
                    in_channels[0],
                    nc,
                    pool_size=self.cfg["roi"]["pool_size"],
                    hidden_dim=self.cfg["bbox_head"]["hidden_dim"],
                )
            ]
        )
        point_cfg = _PointRendRCNNConfig.from_dict(self.cfg.get("pointrend"))
        self.point_rend = _PointRendMaskBranch(in_channels[0], nc, point_cfg)
        self.point_rend_enabled = True
        self.point_rend_self_contained = True
        self.loss_names = ["rpn_cls", "rpn_box", "rcnn_cls", "rcnn_box", "coarse_mask", "point"]

    @property
    def bbox_head(self) -> _FastRCNNBoxHead:
        """Return the Detectron2-style box head."""

        return self.bbox_heads[0]

    def _box_losses_and_mask_rois(
        self,
        feats: list[Tensor],
        proposals: list[Tensor],
        gt_boxes: list[Tensor],
        gt_labels: list[Tensor],
        image_shape: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute Fast R-CNN losses and return GT-class predicted positive boxes."""

        sampled_rois, labels_all, matched_boxes, matched_gt_indices = [], [], [], []
        for batch_index, props in enumerate(proposals):
            props = torch.cat((props, gt_boxes[batch_index]), dim=0) if gt_boxes[batch_index].numel() else props
            _, labels, boxes, targets, gt_indices = _match_hboxes(
                props,
                gt_boxes[batch_index],
                gt_labels[batch_index],
                self.cfg["train"]["pos_iou"],
                self.cfg["train"]["neg_iou"],
                self.cfg["train"]["samples_per_img"],
                self.cfg["train"]["pos_fraction"],
            )
            sampled_rois.append(
                torch.cat((boxes.new_full((boxes.shape[0], 1), batch_index), boxes), dim=1)
            )
            labels_all.append(labels)
            matched_boxes.append(targets)
            matched_gt_indices.append(gt_indices)

        rois = torch.cat(sampled_rois, dim=0) if sampled_rois else feats[0].new_zeros((0, 5))
        labels = torch.cat(labels_all, dim=0) if labels_all else feats[0].new_zeros((0,), dtype=torch.long)
        matched = torch.cat(matched_boxes, dim=0) if matched_boxes else feats[0].new_zeros((0, 4))
        gt_indices = (
            torch.cat(matched_gt_indices, dim=0)
            if matched_gt_indices
            else feats[0].new_zeros((0,), dtype=torch.long)
        )
        if rois.shape[0] == 0:
            zero = sum((parameter.sum() * 0.0 for parameter in self.bbox_head.parameters()), feats[0].sum() * 0.0)
            return (
                zero,
                zero,
                rois.new_zeros((0, 5)),
                gt_indices.new_zeros((0,), dtype=torch.long),
                labels.new_zeros((0,), dtype=torch.long),
            )

        pooled = _roi_align_multilevel(
            feats[:4],
            rois,
            self.cfg["roi"]["pool_size"],
            self.cfg["roi"]["sampling_ratio"],
            self.cfg["roi"]["featmap_strides"],
        )
        class_logits, box_deltas = self.bbox_head(pooled)
        class_loss = F.cross_entropy(class_logits, labels)
        positive = labels > 0
        if positive.any():
            classes = labels[positive] - 1
            selected_deltas = box_deltas[positive, classes]
            targets = self.stage_coders[0].encode(rois[positive, 1:5], matched[positive])
            box_loss = F.smooth_l1_loss(
                selected_deltas,
                targets,
                beta=float(self.cfg["bbox_head"]["beta"]),
                reduction="sum",
            ) / max(labels.numel(), 1)
            with torch.no_grad():
                predicted_boxes = _clip_boxes(
                    self.stage_coders[0].decode(rois[positive, 1:5], selected_deltas),
                    image_shape,
                )
            mask_rois = torch.cat((rois[positive, :1], predicted_boxes.detach()), dim=1)
            return class_loss, box_loss, mask_rois, gt_indices[positive], classes

        box_loss = box_deltas.sum() * 0.0
        return (
            class_loss,
            box_loss,
            rois.new_zeros((0, 5)),
            gt_indices.new_zeros((0,), dtype=torch.long),
            labels.new_zeros((0,), dtype=torch.long),
        )

    def loss(self, feats: list[Tensor], batch: dict) -> tuple[Tensor, Tensor]:
        """Compute RPN, Fast R-CNN, coarse-mask, and point losses."""

        gt_boxes, gt_labels, gt_masks = _split_targets(batch, "segment")
        image_shape = tuple(batch["img"].shape[2:])
        rpn_cls, rpn_box, proposals = self._rpn_loss_and_proposals(
            feats,
            gt_boxes,
            image_shape,
            train=True,
        )
        rcnn_cls, rcnn_box, mask_rois, gt_indices, classes = self._box_losses_and_mask_rois(
            feats,
            proposals,
            gt_boxes,
            gt_labels,
            image_shape,
        )
        coarse_mask, point = self.point_rend.losses(
            feats[0],
            mask_rois[:, 1:5],
            mask_rois[:, 0].long(),
            classes,
            gt_indices,
            gt_masks,
            image_shape,
        )
        loss_items = torch.stack((rpn_cls, rpn_box, rcnn_cls, rcnn_box, coarse_mask, point))
        if self.point_rend.train_config.mode == "frozen":
            return loss_items[-2:].sum(), loss_items.detach()
        return loss_items.sum(), loss_items.detach()

    def _box_predictions(
        self,
        feats: list[Tensor],
        proposals: list[Tensor],
        image_shape: tuple[int, int],
    ) -> list[tuple[Tensor, Tensor, Tensor]]:
        """Decode class-specific boxes and apply standard per-class NMS."""

        predictions = []
        for batch_index, props in enumerate(proposals):
            if props.shape[0] == 0:
                predictions.append(
                    (
                        props.new_zeros((0, 4)),
                        props.new_zeros((0,)),
                        props.new_zeros((0,), dtype=torch.long),
                    )
                )
                continue
            rois = torch.cat((props.new_full((props.shape[0], 1), batch_index), props), dim=1)
            pooled = _roi_align_multilevel(
                feats[:4],
                rois,
                self.cfg["roi"]["pool_size"],
                self.cfg["roi"]["sampling_ratio"],
                self.cfg["roi"]["featmap_strides"],
            )
            class_logits, box_deltas = self.bbox_head(pooled)
            scores = class_logits.softmax(dim=-1)[:, 1:]
            repeated_props = props[:, None].expand(-1, self.nc, -1).reshape(-1, 4)
            boxes = _clip_boxes(
                self.stage_coders[0].decode(repeated_props, box_deltas.reshape(-1, 4)),
                image_shape,
            )
            scores = scores.reshape(-1)
            classes = torch.arange(self.nc, device=props.device).expand(props.shape[0], -1).reshape(-1)
            keep = scores > float(self.cfg["test"]["score_thresh"])
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
            keep = TorchNMS.batched_nms(
                boxes,
                scores,
                classes,
                float(self.cfg["test"]["nms_iou"]),
            )[: int(self.cfg["test"]["max_dets"])]
            predictions.append((boxes[keep], scores[keep], classes[keep]))
        return predictions

    @torch.no_grad()
    def forward(self, feats: list[Tensor]) -> list[dict[str, Tensor]]:
        """Return Ultralytics-native detections plus refined ROI logits for final-resolution pasting."""

        image_shape = self._image_shape_from_feats(feats)
        empty_gt = [feats[0].new_zeros((0, 4)) for _ in range(feats[0].shape[0])]
        _, _, proposals = self._rpn_loss_and_proposals(feats, empty_gt, image_shape, train=False)
        box_predictions = self._box_predictions(feats, proposals, image_shape)
        outputs = []
        threshold = float(self.cfg["test"].get("mask_threshold", 0.5))
        final_side = self.point_rend.final_resolution()
        for batch_index, (boxes, scores, classes) in enumerate(box_predictions):
            if boxes.shape[0]:
                batch_indices = boxes.new_full((boxes.shape[0],), batch_index, dtype=torch.long)
                all_class_logits = self.point_rend.refine(
                    feats[0],
                    boxes,
                    batch_indices,
                    classes,
                    image_shape,
                )
                roi_logits = _select_class_channels(all_class_logits, classes)
                # Keep validation peak memory bounded when the optimizer and EMA are also resident.
                masks = _paste_binary_masks(
                    roi_logits,
                    boxes,
                    image_shape,
                    threshold,
                )
            else:
                roi_logits = boxes.new_zeros((0, 1, final_side, final_side))
                masks = boxes.new_zeros((0, image_shape[0], image_shape[1]), dtype=torch.bool)
            outputs.append(
                {
                    "bboxes": boxes,
                    "conf": scores,
                    "cls": classes.float(),
                    "masks": masks,
                    "mask_roi_logits": roi_logits,
                }
            )
        return outputs
