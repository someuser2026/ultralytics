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
    """Model architecture and inference configuration for a PointRend add-on."""

    enabled: bool = False
    feature_levels: tuple[int, ...] = (0,)
    project_channels: int = 256
    hidden_channels: int = 256
    num_fcs: int = 3
    coarse_resolution: int = 28
    subdivision_steps: int = 3
    subdivision_num_points: int = 784
    scale_factor: int = 2

    @classmethod
    def from_yaml(cls, model_yaml: dict[str, Any]) -> "PointRendConfig":
        """Create and validate architecture configuration from a model YAML dictionary."""

        value = model_yaml.get("pointrend")
        if value is None or value is False:
            return cls(enabled=False)
        if value is True:
            value = {}
        if not isinstance(value, dict):
            raise TypeError("model YAML 'pointrend' must be a mapping, true, false, or null.")
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError(f"pointrend.enabled must be true or false, got {enabled!r}.")
        levels = value.get("feature_levels", [0])
        if isinstance(levels, int):
            levels = [levels]
        cfg = cls(
            enabled=enabled,
            feature_levels=tuple(int(x) for x in levels),
            project_channels=int(value.get("project_channels", 256)),
            hidden_channels=int(value.get("hidden_channels", 256)),
            num_fcs=int(value.get("num_fcs", 3)),
            coarse_resolution=int(value.get("coarse_resolution", 28)),
            subdivision_steps=int(value.get("subdivision_steps", 3)),
            subdivision_num_points=int(value.get("subdivision_num_points", 784)),
            scale_factor=int(value.get("scale_factor", 2)),
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize the normalized architecture into a model-YAML-compatible mapping."""

        return {
            "enabled": self.enabled,
            "feature_levels": list(self.feature_levels),
            "project_channels": self.project_channels,
            "hidden_channels": self.hidden_channels,
            "num_fcs": self.num_fcs,
            "coarse_resolution": self.coarse_resolution,
            "subdivision_steps": self.subdivision_steps,
            "subdivision_num_points": self.subdivision_num_points,
            "scale_factor": self.scale_factor,
        }

    def shape_signature(self) -> tuple[Any, ...]:
        """Return architecture fields that determine PointRend parameter shapes."""

        return (
            self.feature_levels,
            self.project_channels,
            self.hidden_channels,
            self.num_fcs,
            self.coarse_resolution,
        )

    def validate(self) -> None:
        """Raise an actionable error for invalid PointRend configuration."""

        if not self.enabled:
            return
        if not self.feature_levels or min(self.feature_levels) < 0:
            raise ValueError("pointrend.feature_levels must contain non-negative feature indices.")
        positive = {
            "pointrend.project_channels": self.project_channels,
            "pointrend.hidden_channels": self.hidden_channels,
            "pointrend.num_fcs": self.num_fcs,
            "pointrend.coarse_resolution": self.coarse_resolution,
            "pointrend.subdivision_num_points": self.subdivision_num_points,
            "pointrend.scale_factor": self.scale_factor,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}.")
        if self.subdivision_steps < 0:
            raise ValueError(f"pointrend.subdivision_steps must be >= 0, got {self.subdivision_steps}.")


@dataclass(frozen=True)
class PointRendTrainConfig:
    """Training-only PointRend configuration sourced from overall train arguments."""

    mode: str = "joint"
    train_num_points: int = 196
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    train_max_instances: int = 100
    loss_weight: float = 1.0

    @classmethod
    def from_args(cls, args: Any) -> "PointRendTrainConfig":
        """Create and validate training configuration from a dict or namespace."""

        get = args.get if isinstance(args, dict) else lambda key, default=None: getattr(args, key, default)
        cfg = cls(
            mode=str(get("pointrend_mode", "joint")).lower(),
            train_num_points=int(get("pointrend_train_num_points", 196)),
            oversample_ratio=float(get("pointrend_oversample_ratio", 3.0)),
            importance_sample_ratio=float(get("pointrend_importance_sample_ratio", 0.75)),
            train_max_instances=int(get("pointrend_train_max_instances", 100)),
            loss_weight=float(get("pointrend_loss_weight", 1.0)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate PointRend training policy and sampling values."""

        if self.mode not in {"joint", "frozen"}:
            raise ValueError(f"pointrend_mode must be 'joint' or 'frozen', got {self.mode!r}.")
        if self.train_num_points <= 0:
            raise ValueError(f"pointrend_train_num_points must be > 0, got {self.train_num_points}.")
        if self.train_max_instances <= 0:
            raise ValueError(f"pointrend_train_max_instances must be > 0, got {self.train_max_instances}.")
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
        """Detach base predictions while preserving gradients through projected PointRend features."""

        return PointRendInstances(
            coarse_logits=self.coarse_logits.detach(),
            boxes=self.boxes.detach(),
            batch_indices=self.batch_indices,
            fine_features=self.fine_features,
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


def _paste_roi_logits_legacy(roi_logits: Tensor, boxes: Tensor, image_shape: tuple[int, int]) -> Tensor:
    """Paste ROI logits with the original rounded-box implementation used by Mask2Former."""

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


def paste_roi_probabilities(
    roi_logits: Tensor,
    boxes: Tensor,
    image_shape: tuple[int, int],
    *,
    max_chunk_size: int | None = None,
) -> Tensor:
    """Paste sigmoid ROI probabilities at continuous box coordinates using image pixel centers."""

    if roi_logits.ndim != 4 or roi_logits.shape[1] != 1:
        raise ValueError(f"Expected ROI logits shaped (N, 1, H, W), got {tuple(roi_logits.shape)}.")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != roi_logits.shape[0]:
        raise ValueError(
            f"Expected one xyxy box per ROI logit, got boxes={tuple(boxes.shape)} "
            f"and logits={tuple(roi_logits.shape)}."
        )
    if max_chunk_size is not None and max_chunk_size <= 0:
        raise ValueError(f"max_chunk_size must be positive when provided, got {max_chunk_size}.")

    n = roi_logits.shape[0]
    h, w = (int(image_shape[0]), int(image_shape[1]))
    if h < 0 or w < 0:
        raise ValueError(f"image_shape must be non-negative, got {image_shape}.")
    if n == 0:
        return roi_logits.new_zeros((0, h, w))

    sample_dtype = (
        torch.float32
        if roi_logits.device.type == "cpu" and roi_logits.dtype in {torch.float16, torch.bfloat16}
        else roi_logits.dtype
    )
    probabilities = roi_logits.to(dtype=sample_dtype).sigmoid()
    sample_boxes = boxes.to(device=roi_logits.device, dtype=sample_dtype)
    valid = torch.isfinite(sample_boxes).all(dim=1)
    valid &= sample_boxes[:, 2] > sample_boxes[:, 0]
    valid &= sample_boxes[:, 3] > sample_boxes[:, 1]
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    canvases = probabilities.new_zeros((n, h, w))
    if valid_indices.numel() == 0 or h == 0 or w == 0:
        return canvases.to(dtype=roi_logits.dtype)

    if max_chunk_size is None:
        bytes_per_instance = max(h * w * probabilities.element_size() * 3, 1)
        max_chunk_size = max(1, min(64, (256 * 1024**2) // bytes_per_instance))

    image_y = torch.arange(h, device=roi_logits.device, dtype=sample_dtype) + 0.5
    image_x = torch.arange(w, device=roi_logits.device, dtype=sample_dtype) + 0.5
    for indices in valid_indices.split(max_chunk_size):
        chunk_boxes = sample_boxes.index_select(0, indices)
        x0, y0, x1, y1 = chunk_boxes.unbind(dim=1)
        grid_x = (image_x[None] - x0[:, None]) / (x1 - x0)[:, None] * 2.0 - 1.0
        grid_y = (image_y[None] - y0[:, None]) / (y1 - y0)[:, None] * 2.0 - 1.0
        grid = torch.stack(
            (
                grid_x[:, None].expand(-1, h, -1),
                grid_y[:, :, None].expand(-1, -1, w),
            ),
            dim=-1,
        )
        pasted = F.grid_sample(
            probabilities.index_select(0, indices),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0]
        canvases = canvases.index_copy(0, indices, pasted)
    return canvases.to(dtype=roi_logits.dtype)


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
        for layer in self.fcs:
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.fc_logits.weight, std=0.001)
        nn.init.zeros_(self.fc_logits.bias)

    def forward(self, fine_features: Tensor, coarse_features: Tensor) -> Tensor:
        """Predict point logits from sampled fine and coarse features."""

        x = torch.cat((fine_features, coarse_features), dim=1)
        for fc in self.fcs:
            x = F.relu(fc(x), inplace=True)
            x = torch.cat((x, coarse_features), dim=1)
        return self.fc_logits(x)


class PointRendRefiner(nn.Module):
    """Feature projection, point supervision, and iterative PointRend refinement."""

    def __init__(
        self,
        source_channels: list[int] | tuple[int, ...],
        model_config: PointRendConfig,
        train_config: PointRendTrainConfig | None = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config or PointRendTrainConfig()
        self.source_channels = tuple(int(x) for x in source_channels)
        self.projections = nn.ModuleList(
            [nn.Conv2d(ch, model_config.project_channels, 1) for ch in self.source_channels]
        )
        self.point_head = PointRendPointHead(
            fine_channels=len(self.source_channels) * model_config.project_channels,
            hidden_channels=model_config.hidden_channels,
            num_fcs=model_config.num_fcs,
        )

    def project_features(self, features: list[Tensor] | tuple[Tensor, ...]) -> list[Tensor]:
        """Project the configured fine feature maps to a common width."""

        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} PointRend features, got {len(features)}.")
        if self.train_config.mode == "frozen":
            features = [feature.detach() for feature in features]
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
            self.train_config.train_num_points,
            self.train_config.oversample_ratio,
            self.train_config.importance_sample_ratio,
        )
        point_logits = self.predict_points(instances, point_coords)[:, 0]
        image_points = roi_points_to_image_points(point_coords, instances.boxes, instances.image_shape)
        targets = point_sample(instances.gt_masks.float(), image_points)[:, 0]
        return F.binary_cross_entropy_with_logits(point_logits, targets)

    def _subdivide(self, instances: PointRendInstances, refined: Tensor) -> Tensor:
        """Upsample a mask grid and replace its most uncertain logits at each subdivision."""

        for _ in range(self.model_config.subdivision_steps):
            refined = F.interpolate(
                refined, scale_factor=self.model_config.scale_factor, mode="bilinear", align_corners=False
            )
            indices, point_coords = select_uncertain_points_test(
                refined, self.model_config.subdivision_num_points
            )
            point_logits = self.predict_points(instances, point_coords)
            flat = refined.flatten(2)
            flat = flat.scatter(2, indices[:, None].expand(-1, flat.shape[1], -1), point_logits)
            refined = flat.reshape_as(refined)
        return refined

    def refine(self, instances: PointRendInstances) -> Tensor:
        """Run instance PointRend from a dense point-head grid, then perform adaptive subdivisions."""

        n = instances.coarse_logits.shape[0]
        resolution = self.model_config.coarse_resolution
        if n == 0:
            scale = self.model_config.scale_factor**self.model_config.subdivision_steps
            return instances.coarse_logits.new_zeros((0, 1, resolution * scale, resolution * scale))
        point_coords = _regular_roi_grid(
            n,
            resolution,
            instances.coarse_logits.device,
            instances.coarse_logits.dtype,
        )
        refined = self.predict_points(instances, point_coords).reshape(n, 1, resolution, resolution)
        return self._subdivide(instances, refined)

    def refine_from_coarse(self, instances: PointRendInstances) -> Tensor:
        """Preserve the coarse-first refinement sequence used by the Mask2Former integration."""

        refined = instances.coarse_logits
        if refined.shape[0] == 0:
            scale = self.model_config.scale_factor**self.model_config.subdivision_steps
            return refined.new_zeros((0, 1, refined.shape[-2] * scale, refined.shape[-1] * scale))
        return self._subdivide(instances, refined)


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
            full_logits, boxes, image_shape, self.refiner.model_config.coarse_resolution
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
        return instances.detached_base() if self.refiner.train_config.mode == "frozen" else instances

    def refined_image_probabilities(self, instances: PointRendInstances) -> Tensor:
        """Refine ROI masks and paste sigmoid probabilities into input-image coordinates."""

        return paste_roi_probabilities(
            self.refiner.refine(instances),
            instances.boxes,
            instances.image_shape,
        )

    def refined_image_logits(self, instances: PointRendInstances) -> Tensor:
        """Return logits derived from the reference-aligned image probabilities."""

        probabilities = self.refined_image_probabilities(instances)
        eps = max(float(torch.finfo(probabilities.dtype).eps), 1e-6)
        return torch.logit(probabilities.clamp(min=eps, max=1.0 - eps))


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

    def refined_image_logits(self, instances: PointRendInstances) -> Tensor:
        """Preserve Mask2Former's coarse-first refinement and rounded-box logit pasting."""

        return _paste_roi_logits_legacy(
            self.refiner.refine_from_coarse(instances),
            instances.boxes,
            instances.image_shape,
        )


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
        coarse_resolution = self.refiner.model_config.coarse_resolution
        if roi_logits.shape[-2:] != (coarse_resolution,) * 2:
            roi_logits = F.interpolate(
                roi_logits,
                size=(coarse_resolution,) * 2,
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
        return instances.detached_base() if self.refiner.train_config.mode == "frozen" else instances


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


def configure_pointrend_from_yaml(model: nn.Module) -> PointRendRefiner | None:
    """Attach PointRend from the model YAML without consulting training arguments."""

    model = getattr(model, "module", model)
    model_yaml = getattr(model, "yaml", None)
    if not isinstance(model_yaml, dict):
        raise TypeError("PointRend configuration requires model.yaml to be a dictionary.")
    config = PointRendConfig.from_yaml(model_yaml)
    if not config.enabled:
        return None
    head = _head_from_model(model)
    head_name = type(head).__name__
    if head_name not in _POINTREND_ADAPTERS:
        supported = ", ".join(sorted(_POINTREND_ADAPTERS))
        raise TypeError(
            f"PointRend is enabled in the model YAML, but head {head_name} is unsupported. "
            f"Register an adapter with register_pointrend_adapter(); supported heads: {supported}."
        )
    channels = tuple(getattr(head, "point_rend_source_channels", ()))
    if not channels:
        raise TypeError(
            f"{head_name} does not expose point_rend_source_channels and cannot use PointRend."
        )
    if max(config.feature_levels) >= len(channels):
        raise ValueError(
            f"pointrend.feature_levels={list(config.feature_levels)} exceeds the {len(channels)} features "
            f"exposed by {head_name}."
        )
    selected_channels = [channels[i] for i in config.feature_levels]
    existing = getattr(head, "point_rend", None)
    if existing is None:
        head.point_rend = PointRendRefiner(selected_channels, config)
        reference = next(head.parameters(), None)
        if reference is not None:
            head.point_rend.to(device=reference.device, dtype=reference.dtype)
    else:
        if tuple(existing.source_channels) != tuple(selected_channels):
            raise ValueError(
                "Loaded PointRend feature channels do not match the requested pointrend.feature_levels."
            )
        existing_config = _model_config_from_refiner(existing)
        if existing_config.shape_signature() != config.shape_signature():
            raise ValueError(
                "An attached PointRend module is incompatible with model.yaml['pointrend']; "
                f"module={existing_config.to_dict()}, YAML={config.to_dict()}."
            )
        existing.model_config = config
        if not hasattr(existing, "train_config"):
            existing.train_config = PointRendTrainConfig()
    head.point_rend_enabled = True
    head.point_rend_feature_levels = config.feature_levels
    model_yaml["pointrend"] = config.to_dict()
    if head_name in {"MaskRCNNHead", "CascadeMaskRCNNHead"} and "point" not in head.loss_names:
        head.loss_names.append("point")
    return head.point_rend


def configure_pointrend_training(model: nn.Module, args: Any) -> PointRendTrainConfig | None:
    """Update only PointRend training policy; never create or resize model modules."""

    if not has_pointrend(model):
        return None
    train_config = PointRendTrainConfig.from_args(args)
    head = _head_from_model(model)
    head.point_rend.train_config = train_config
    return train_config


def _model_config_from_refiner(refiner: PointRendRefiner) -> PointRendConfig:
    """Read architecture config from current or legacy runtime-config PointRend refiners."""

    config = getattr(refiner, "model_config", None)
    if isinstance(config, PointRendConfig):
        return config
    legacy = getattr(refiner, "config", None)
    if legacy is None:
        raise TypeError("The incoming PointRend refiner has no recoverable architecture configuration.")
    config = PointRendConfig(
        enabled=True,
        feature_levels=tuple(int(x) for x in getattr(legacy, "feature_levels", (0,))),
        project_channels=int(getattr(legacy, "project_channels", 256)),
        hidden_channels=int(getattr(legacy, "hidden_channels", 256)),
        num_fcs=int(getattr(legacy, "num_fcs", 3)),
        coarse_resolution=int(getattr(legacy, "coarse_resolution", 28)),
        subdivision_steps=int(getattr(legacy, "subdivision_steps", 3)),
        subdivision_num_points=int(getattr(legacy, "subdivision_num_points", 784)),
        scale_factor=int(getattr(legacy, "scale_factor", 2)),
    )
    config.validate()
    return config


def _pointrend_refiner(model: nn.Module) -> PointRendRefiner | None:
    """Return an attached refiner without requiring its enabled flag."""

    try:
        return getattr(_head_from_model(model), "point_rend", None)
    except TypeError:
        return None


def prepare_pointrend_weight_transfer(target_model: nn.Module, incoming_model: nn.Module) -> None:
    """Make PointRend topology compatible before intersecting incoming checkpoint tensors."""

    target_refiner = _pointrend_refiner(target_model)
    incoming_refiner = _pointrend_refiner(incoming_model)
    if target_refiner is not None:
        target_config = _model_config_from_refiner(target_refiner)
        target_refiner.model_config = target_config
        if not hasattr(target_refiner, "train_config"):
            target_refiner.train_config = PointRendTrainConfig()
        target = getattr(target_model, "module", target_model)
        if isinstance(getattr(target, "yaml", None), dict):
            target.yaml["pointrend"] = target_config.to_dict()
    if incoming_refiner is None:
        return

    incoming_config = _model_config_from_refiner(incoming_refiner)
    if target_refiner is None:
        target = getattr(target_model, "module", target_model)
        target.yaml["pointrend"] = incoming_config.to_dict()
        target_refiner = configure_pointrend_from_yaml(target)
        if target_refiner is None:  # pragma: no cover - guarded by enabled=True above
            raise RuntimeError("Failed to reconstruct PointRend from the incoming checkpoint.")

    target_config = _model_config_from_refiner(target_refiner)
    if target_config.shape_signature() != incoming_config.shape_signature():
        raise ValueError(
            "PointRend checkpoint architecture is incompatible with the target model. "
            f"target={target_config.to_dict()}, checkpoint={incoming_config.to_dict()}."
        )
    if tuple(target_refiner.source_channels) != tuple(incoming_refiner.source_channels):
        raise ValueError(
            "PointRend checkpoint feature-channel shapes are incompatible with the target model: "
            f"target={target_refiner.source_channels}, checkpoint={incoming_refiner.source_channels}."
        )


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
