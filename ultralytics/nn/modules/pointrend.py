# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model-agnostic PointRend modules and segmentation-head adapters."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class PointRendConfig:
    """Normalized runtime configuration for a PointRend add-on."""

    enabled: bool = False
    mode: str = "joint"
    feature_levels: tuple[int, ...] = (0,)
    project_channels: int = 256
    hidden_channels: int = 256
    num_fcs: int = 3
    coarse_resolution: int = 28
    train_num_points: int = 196
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    train_max_instances: int = 100
    subdivision_steps: int = 3
    subdivision_num_points: int = 784
    scale_factor: int = 2
    loss_weight: float = 1.0

    @classmethod
    def from_args(cls, args: Any) -> "PointRendConfig":
        """Create and validate a configuration from a dict or namespace."""

        get = args.get if isinstance(args, dict) else lambda key, default=None: getattr(args, key, default)
        levels = get("pointrend_feature_levels", [0])
        if isinstance(levels, int):
            levels = [levels]
        cfg = cls(
            enabled=bool(get("pointrend", False)),
            mode=str(get("pointrend_mode", "joint")).lower(),
            feature_levels=tuple(int(x) for x in levels),
            project_channels=int(get("pointrend_project_channels", 256)),
            hidden_channels=int(get("pointrend_hidden_channels", 256)),
            num_fcs=int(get("pointrend_num_fcs", 3)),
            coarse_resolution=int(get("pointrend_coarse_resolution", 28)),
            train_num_points=int(get("pointrend_train_num_points", 196)),
            oversample_ratio=float(get("pointrend_oversample_ratio", 3.0)),
            importance_sample_ratio=float(get("pointrend_importance_sample_ratio", 0.75)),
            train_max_instances=int(get("pointrend_train_max_instances", 100)),
            subdivision_steps=int(get("pointrend_subdivision_steps", 3)),
            subdivision_num_points=int(get("pointrend_subdivision_num_points", 784)),
            scale_factor=int(get("pointrend_scale_factor", 2)),
            loss_weight=float(get("pointrend_loss_weight", 1.0)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Raise an actionable error for invalid PointRend configuration."""

        if self.mode not in {"joint", "frozen"}:
            raise ValueError(f"pointrend_mode must be 'joint' or 'frozen', got {self.mode!r}.")
        if not self.feature_levels or min(self.feature_levels) < 0:
            raise ValueError("pointrend_feature_levels must contain non-negative feature indices.")
        positive = {
            "pointrend_project_channels": self.project_channels,
            "pointrend_hidden_channels": self.hidden_channels,
            "pointrend_num_fcs": self.num_fcs,
            "pointrend_coarse_resolution": self.coarse_resolution,
            "pointrend_train_num_points": self.train_num_points,
            "pointrend_train_max_instances": self.train_max_instances,
            "pointrend_subdivision_num_points": self.subdivision_num_points,
            "pointrend_scale_factor": self.scale_factor,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}.")
        if self.subdivision_steps < 0:
            raise ValueError(f"pointrend_subdivision_steps must be >= 0, got {self.subdivision_steps}.")
        if self.oversample_ratio < 1.0:
            raise ValueError(f"pointrend_oversample_ratio must be >= 1, got {self.oversample_ratio}.")
        if not 0.0 <= self.importance_sample_ratio <= 1.0:
            raise ValueError(
                "pointrend_importance_sample_ratio must be in [0, 1], "
                f"got {self.importance_sample_ratio}."
            )
        if self.loss_weight < 0:
            raise ValueError(f"pointrend_loss_weight must be >= 0, got {self.loss_weight}.")


@dataclass
class PointRendInstances:
    """Head-independent per-instance tensors consumed by PointRend."""

    coarse_logits: Tensor
    boxes: Tensor
    batch_indices: Tensor
    fine_features: list[Tensor]
    image_shape: tuple[int, int]
    gt_masks: Tensor | None = None
    source_indices: Tensor | None = None

    def detached_base(self) -> "PointRendInstances":
        """Detach all base-model tensors while retaining PointRend trainability."""

        return PointRendInstances(
            coarse_logits=self.coarse_logits.detach(),
            boxes=self.boxes.detach(),
            batch_indices=self.batch_indices,
            fine_features=[x.detach() for x in self.fine_features],
            image_shape=self.image_shape,
            gt_masks=self.gt_masks,
            source_indices=self.source_indices,
        )


def point_sample(input: Tensor, point_coords: Tensor) -> Tensor:
    """Sample ``input`` at [0, 1] point coordinates using PointRend conventions."""

    if point_coords.ndim != 3 or point_coords.shape[-1] != 2:
        raise ValueError(f"Expected point coordinates shaped (N, P, 2), got {tuple(point_coords.shape)}.")
    if input.shape[0] != point_coords.shape[0]:
        raise ValueError(f"Point batch {point_coords.shape[0]} does not match input batch {input.shape[0]}.")
    grid = point_coords.mul(2.0).sub(1.0).unsqueeze(2)
    return F.grid_sample(input, grid, mode="bilinear", align_corners=False).squeeze(3)


def calculate_uncertainty(logits: Tensor) -> Tensor:
    """Return binary-mask uncertainty with logits nearest zero ranked highest."""

    if logits.shape[1] != 1:
        raise ValueError(f"PointRend expects one class-agnostic mask channel, got {logits.shape[1]}.")
    return -logits.abs()


@torch.no_grad()
def sample_uncertain_points_train(
    logits: Tensor,
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> Tensor:
    """Sample uncertain and random ROI-relative points for PointRend training."""

    n = logits.shape[0]
    if n == 0:
        return logits.new_zeros((0, num_points, 2))
    num_candidates = max(int(num_points * oversample_ratio), num_points)
    candidates = torch.rand((n, num_candidates, 2), device=logits.device, dtype=logits.dtype)
    uncertainty = calculate_uncertainty(point_sample(logits, candidates))[:, 0]
    num_uncertain = int(num_points * importance_sample_ratio)
    num_random = num_points - num_uncertain
    if num_uncertain:
        indices = uncertainty.topk(num_uncertain, dim=1).indices
        candidates = candidates.gather(1, indices[..., None].expand(-1, -1, 2))
    else:
        candidates = candidates[:, :0]
    if num_random:
        random_points = torch.rand((n, num_random, 2), device=logits.device, dtype=logits.dtype)
        candidates = torch.cat((candidates, random_points), dim=1)
    return candidates


@torch.no_grad()
def select_uncertain_points_test(logits: Tensor, num_points: int) -> tuple[Tensor, Tensor]:
    """Select the most uncertain pixels and return flat indices plus pixel-center coordinates."""

    n, _, h, w = logits.shape
    count = min(int(num_points), h * w)
    if n == 0:
        return (
            torch.zeros((0, count), device=logits.device, dtype=torch.long),
            logits.new_zeros((0, count, 2)),
        )
    indices = calculate_uncertainty(logits).flatten(1).topk(count, dim=1).indices
    xs = (indices.remainder(w).to(logits.dtype) + 0.5) / w
    ys = (indices.div(w, rounding_mode="floor").to(logits.dtype) + 0.5) / h
    return indices, torch.stack((xs, ys), dim=-1)


def roi_points_to_image_points(point_coords: Tensor, boxes: Tensor, image_shape: tuple[int, int]) -> Tensor:
    """Map ROI-relative [0, 1] coordinates to normalized input-image coordinates."""

    h, w = image_shape
    boxes = boxes.to(dtype=point_coords.dtype)
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_min(1e-6)
    xy = boxes[:, None, :2] + point_coords * wh[:, None]
    scale = point_coords.new_tensor((max(w, 1), max(h, 1)))
    return xy / scale


def _regular_roi_grid(n: int, resolution: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a pixel-center grid in ROI-relative coordinates."""

    ys = (torch.arange(resolution, device=device, dtype=dtype) + 0.5) / resolution
    xs = (torch.arange(resolution, device=device, dtype=dtype) + 0.5) / resolution
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(1, resolution * resolution, 2).expand(n, -1, -1)


def crop_logits_to_rois(
    full_logits: Tensor,
    boxes: Tensor,
    image_shape: tuple[int, int],
    resolution: int,
) -> Tensor:
    """Crop per-instance full-image logits to fixed-resolution ROI logits."""

    n = full_logits.shape[0]
    if n == 0:
        return full_logits.new_zeros((0, 1, resolution, resolution))
    roi_grid = _regular_roi_grid(n, resolution, full_logits.device, full_logits.dtype)
    image_grid = roi_points_to_image_points(roi_grid, boxes, image_shape)
    return point_sample(full_logits, image_grid).reshape(n, full_logits.shape[1], resolution, resolution)


def paste_roi_logits(roi_logits: Tensor, boxes: Tensor, image_shape: tuple[int, int]) -> Tensor:
    """Paste refined ROI logits into input-image canvases for inference."""

    n = roi_logits.shape[0]
    h, w = image_shape
    if n == 0:
        return roi_logits.new_zeros((0, h, w))
    canvases = []
    for mask, box in zip(roi_logits[:, 0], boxes):
        x1, y1, x2, y2 = box.round().long().tolist()
        x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
        x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
        canvas = mask.new_full((h, w), -20.0)
        if x2 > x1 and y2 > y1:
            resized = F.interpolate(
                mask[None, None], size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False
            )[0, 0]
            canvas[y1:y2, x1:x2] = resized
        canvases.append(canvas)
    return torch.stack(canvases)


class PointRendPointHead(nn.Module):
    """Shared MLP that predicts a binary logit at each sampled point."""

    def __init__(self, fine_channels: int, hidden_channels: int = 256, num_fcs: int = 3):
        super().__init__()
        self.fcs = nn.ModuleList()
        in_channels = fine_channels + 1
        for _ in range(num_fcs):
            self.fcs.append(nn.Conv1d(in_channels, hidden_channels, 1))
            in_channels = hidden_channels + 1
        self.fc_logits = nn.Conv1d(in_channels, 1, 1)
        for layer in [*self.fcs, self.fc_logits]:
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)

    def forward(self, fine_features: Tensor, coarse_features: Tensor) -> Tensor:
        """Predict point logits from sampled fine and coarse features."""

        x = torch.cat((fine_features, coarse_features), dim=1)
        for fc in self.fcs:
            x = F.relu(fc(x), inplace=True)
            x = torch.cat((x, coarse_features), dim=1)
        return self.fc_logits(x)


class PointRendRefiner(nn.Module):
    """Feature projection, point supervision, and iterative PointRend refinement."""

    def __init__(self, source_channels: list[int] | tuple[int, ...], config: PointRendConfig):
        super().__init__()
        self.config = config
        self.source_channels = tuple(int(x) for x in source_channels)
        self.projections = nn.ModuleList(
            [nn.Conv2d(ch, config.project_channels, 1) for ch in self.source_channels]
        )
        self.point_head = PointRendPointHead(
            fine_channels=len(self.source_channels) * config.project_channels,
            hidden_channels=config.hidden_channels,
            num_fcs=config.num_fcs,
        )

    def project_features(self, features: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
        """Project the configured fine feature maps to a common width."""

        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} PointRend features, got {len(features)}.")
        return [projection(feature) for projection, feature in zip(self.projections, features)]

    def _sample_fine_features(self, instances: PointRendInstances, point_coords: Tensor) -> Tensor:
        image_points = roi_points_to_image_points(point_coords, instances.boxes, instances.image_shape)
        sampled = []
        for feature in instances.fine_features:
            per_instance = feature.index_select(0, instances.batch_indices.long())
            sampled.append(point_sample(per_instance, image_points))
        return torch.cat(sampled, dim=1)

    def predict_points(self, instances: PointRendInstances, point_coords: Tensor) -> Tensor:
        """Predict binary logits at ROI-relative coordinates."""

        fine = self._sample_fine_features(instances, point_coords)
        coarse = point_sample(instances.coarse_logits, point_coords)
        return self.point_head(fine, coarse)

    def point_loss(self, instances: PointRendInstances) -> Tensor:
        """Compute BCE supervision at uncertainty-biased points."""

        if instances.coarse_logits.shape[0] == 0 or instances.gt_masks is None:
            return sum((parameter.sum() * 0.0 for parameter in self.parameters()), instances.coarse_logits.sum() * 0.0)
        point_coords = sample_uncertain_points_train(
            instances.coarse_logits,
            self.config.train_num_points,
            self.config.oversample_ratio,
            self.config.importance_sample_ratio,
        )
        point_logits = self.predict_points(instances, point_coords)[:, 0]
        image_points = roi_points_to_image_points(point_coords, instances.boxes, instances.image_shape)
        targets = point_sample(instances.gt_masks.float(), image_points)[:, 0]
        return F.binary_cross_entropy_with_logits(point_logits, targets)

    def refine(self, instances: PointRendInstances) -> Tensor:
        """Iteratively upsample masks and replace their most uncertain logits."""

        refined = instances.coarse_logits
        if refined.shape[0] == 0:
            scale = self.config.scale_factor**self.config.subdivision_steps
            return refined.new_zeros((0, 1, refined.shape[-2] * scale, refined.shape[-1] * scale))
        for _ in range(self.config.subdivision_steps):
            refined = F.interpolate(
                refined, scale_factor=self.config.scale_factor, mode="bilinear", align_corners=False
            )
            indices, point_coords = select_uncertain_points_test(refined, self.config.subdivision_num_points)
            point_logits = self.predict_points(instances, point_coords)
            flat = refined.flatten(2)
            flat = flat.scatter(2, indices[:, None].expand(-1, flat.shape[1], -1), point_logits)
            refined = flat.reshape_as(refined)
        return refined


class PointRendAdapter:
    """Base adapter that converts head-native masks into common PointRend instances."""

    def __init__(self, refiner: PointRendRefiner):
        self.refiner = refiner

    def from_full_logits(
        self,
        full_logits: Tensor,
        boxes: Tensor,
        batch_indices: Tensor,
        fine_features: list[Tensor],
        image_shape: tuple[int, int],
        gt_masks: Tensor | None = None,
        source_indices: Tensor | None = None,
    ) -> PointRendInstances:
        """Build common instances from per-instance full-image logits."""

        coarse = crop_logits_to_rois(
            full_logits, boxes, image_shape, self.refiner.config.coarse_resolution
        )
        instances = PointRendInstances(
            coarse_logits=coarse,
            boxes=boxes,
            batch_indices=batch_indices.long(),
            fine_features=fine_features,
            image_shape=image_shape,
            gt_masks=gt_masks,
            source_indices=source_indices,
        )
        return instances.detached_base() if self.refiner.config.mode == "frozen" else instances

    def refined_image_logits(self, instances: PointRendInstances) -> Tensor:
        """Refine ROI masks and paste logits into input-image coordinates."""

        return paste_roi_logits(self.refiner.refine(instances), instances.boxes, instances.image_shape)


class YOLOPrototypePointRendAdapter(PointRendAdapter):
    """Adapter for YOLO prototype-mask segmentation heads."""

    def from_coefficients(
        self,
        coefficients: Tensor,
        prototypes: Tensor,
        boxes: Tensor,
        batch_indices: Tensor,
        fine_features: list[Tensor],
        image_shape: tuple[int, int],
        gt_masks: Tensor | None = None,
        source_indices: Tensor | None = None,
        normalize: bool = True,
    ) -> PointRendInstances:
        selected_proto = prototypes.index_select(0, batch_indices.long())
        full_logits = torch.einsum("in,inhw->ihw", coefficients, selected_proto).unsqueeze(1)
        if normalize:
            full_logits = full_logits / sqrt(max(coefficients.shape[1], 1))
        return self.from_full_logits(
            full_logits, boxes, batch_indices, fine_features, image_shape, gt_masks, source_indices
        )


class RTDETRPrototypePointRendAdapter(YOLOPrototypePointRendAdapter):
    """Adapter for RT-DETR query coefficients and prototype masks."""

    def from_coefficients(self, *args, **kwargs) -> PointRendInstances:
        """Assemble RT-DETR masks without YOLO's prototype-channel normalization."""

        kwargs["normalize"] = False
        return super().from_coefficients(*args, **kwargs)


class Mask2FormerPointRendAdapter(PointRendAdapter):
    """Adapter for Mask2Former query mask logits."""


class RCNNPointRendAdapter(PointRendAdapter):
    """Adapter for native RCNN ROI mask logits."""

    def from_roi_logits(
        self,
        roi_logits: Tensor,
        boxes: Tensor,
        batch_indices: Tensor,
        fine_features: list[Tensor],
        image_shape: tuple[int, int],
        gt_masks: Tensor | None = None,
        source_indices: Tensor | None = None,
    ) -> PointRendInstances:
        if roi_logits.ndim == 3:
            roi_logits = roi_logits[:, None]
        if roi_logits.shape[-2:] != (self.refiner.config.coarse_resolution,) * 2:
            roi_logits = F.interpolate(
                roi_logits,
                size=(self.refiner.config.coarse_resolution,) * 2,
                mode="bilinear",
                align_corners=False,
            )
        instances = PointRendInstances(
            coarse_logits=roi_logits,
            boxes=boxes,
            batch_indices=batch_indices.long(),
            fine_features=fine_features,
            image_shape=image_shape,
            gt_masks=gt_masks,
            source_indices=source_indices,
        )
        return instances.detached_base() if self.refiner.config.mode == "frozen" else instances


_POINTREND_ADAPTERS: dict[str, type[PointRendAdapter]] = {
    "Segment": YOLOPrototypePointRendAdapter,
    "Segment26": YOLOPrototypePointRendAdapter,
    "SegmentShoreAux": YOLOPrototypePointRendAdapter,
    "YOLOESegment": YOLOPrototypePointRendAdapter,
    "RTDETRSegmentDecoder": RTDETRPrototypePointRendAdapter,
    "Mask2FormerHead": Mask2FormerPointRendAdapter,
    "MaskRCNNHead": RCNNPointRendAdapter,
    "CascadeMaskRCNNHead": RCNNPointRendAdapter,
}


def register_pointrend_adapter(head_name: str, adapter: type[PointRendAdapter]) -> None:
    """Register a PointRend adapter for an additional segmentation head class name."""

    if not head_name or not isinstance(head_name, str):
        raise TypeError("head_name must be a non-empty segmentation head class name.")
    if not isinstance(adapter, type) or not issubclass(adapter, PointRendAdapter):
        raise TypeError("adapter must be a PointRendAdapter subclass.")
    _POINTREND_ADAPTERS[head_name] = adapter


def _head_from_model(model: nn.Module) -> nn.Module:
    """Return the native final segmentation head from wrapped or unwrapped models."""

    model = getattr(model, "module", model)
    sequence = getattr(model, "model", None)
    if sequence is None or not len(sequence):
        raise TypeError("PointRend requires an Ultralytics model with a final segmentation head.")
    return sequence[-1]


def configure_pointrend(model: nn.Module, args: Any) -> PointRendRefiner | None:
    """Attach or enable PointRend on a supported native segmentation head."""

    config = PointRendConfig.from_args(args)
    head = _head_from_model(model)
    existing = getattr(head, "point_rend", None)
    head.point_rend_enabled = config.enabled
    if not config.enabled:
        return existing
    channels = tuple(getattr(head, "point_rend_source_channels", ()))
    if not channels:
        raise TypeError(
            f"{type(head).__name__} does not expose point_rend_source_channels and cannot use PointRend."
        )
    if max(config.feature_levels) >= len(channels):
        raise ValueError(
            f"pointrend_feature_levels={list(config.feature_levels)} exceeds the {len(channels)} features "
            f"exposed by {type(head).__name__}."
        )
    selected_channels = [channels[i] for i in config.feature_levels]
    if existing is None:
        head.point_rend = PointRendRefiner(selected_channels, config)
    else:
        if tuple(existing.source_channels) != tuple(selected_channels):
            raise ValueError(
                "Loaded PointRend feature channels do not match the requested pointrend_feature_levels."
            )
        existing.config = config
    head.point_rend_feature_levels = config.feature_levels
    if type(head).__name__ in {"MaskRCNNHead", "CascadeMaskRCNNHead"} and "point" not in head.loss_names:
        head.loss_names.append("point")
    return head.point_rend


def has_pointrend(model_or_head: nn.Module) -> bool:
    """Return whether PointRend is attached and enabled."""

    head = model_or_head
    if hasattr(model_or_head, "model"):
        head = _head_from_model(model_or_head)
    return bool(getattr(head, "point_rend_enabled", False) and getattr(head, "point_rend", None) is not None)


def get_pointrend_adapter(head: nn.Module) -> PointRendAdapter:
    """Resolve the adapter for a supported native segmentation head."""

    refiner = getattr(head, "point_rend", None)
    if refiner is None:
        raise RuntimeError(f"PointRend is not attached to {type(head).__name__}.")
    name = type(head).__name__
    adapter = _POINTREND_ADAPTERS.get(name)
    if adapter is None:
        raise TypeError(f"No PointRend adapter is registered for segmentation head {name}.")
    return adapter(refiner)
