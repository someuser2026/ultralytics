# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Native RCNN heads used by the Ultralytics RCNN wrapper.

The implementation is intentionally self-contained: proposal generation, assignment,
RoI extraction, bbox coding, mask prediction, and rotated-box handling all run on
pure PyTorch / torchvision primitives without MMDetection or MMRotate runtime
dependencies.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import roi_align

from ultralytics.utils import ops
from ultralytics.utils.metrics import batch_probiou, box_iou
from ultralytics.utils.nms import TorchNMS

__all__ = (
    "MaskRCNNHead",
    "CascadeMaskRCNNHead",
    "RotatedFasterRCNNHead",
    "OrientedRCNNHead",
)


def _merge_dict(defaults: dict, override: dict | None) -> dict:
    cfg = {**defaults}
    if override:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = _merge_dict(cfg[k], v)
            else:
                cfg[k] = v
    return cfg


def _split_targets(batch: dict, task: str) -> tuple[list[Tensor], list[Tensor], list[Tensor | None]]:
    """Convert collated Ultralytics batches into per-image GT lists."""
    imgsz = batch["img"].shape[2:]
    device = batch["img"].device
    batch_size = batch["img"].shape[0]
    angle_mode = batch.get("angle_mode", "le90") if isinstance(batch, dict) else getattr(batch, "angle_mode", "le90")
    gt_boxes, gt_labels, gt_masks = [], [], []

    masks_all = batch.get("masks")
    for i in range(batch_size):
        idx = batch["batch_idx"].view(-1) == i
        labels = batch["cls"][idx].view(-1).long()
        if task == "segment":
            boxes = ops.xywh2xyxy(batch["bboxes"][idx].clone())
            if boxes.numel():
                scale = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], device=device, dtype=boxes.dtype)
                boxes = boxes * scale
            masks = None
            if masks_all is not None:
                if masks_all.ndim == 2:
                    masks = masks_all[None]
                elif masks_all.shape[0] == batch["batch_idx"].numel():
                    masks = masks_all[idx]
                elif masks_all.shape[0] == batch_size:
                    masks = masks_all[i : i + 1]
                else:
                    masks = masks_all[idx] if masks_all.shape[0] > i else None
                if masks is not None and masks.shape[0] == 1 and labels.numel() > 1 and masks.max() > 1:
                    ids = torch.arange(labels.numel(), device=device).view(-1, 1, 1) + 1
                    masks = (masks.repeat(labels.numel(), 1, 1) == ids).float()
                if masks is not None and masks.shape[-2:] != imgsz:
                    masks = F.interpolate(masks[:, None].float(), size=imgsz, mode="nearest").squeeze(1)
            gt_masks.append(masks)
        else:
            boxes = batch["bboxes"][idx].clone()
            if boxes.numel():
                boxes[:, [0, 2]] *= imgsz[1]
                boxes[:, [1, 3]] *= imgsz[0]
                boxes = ops.regularize_rboxes(boxes, angle_mode=angle_mode)
            gt_masks.append(None)
        gt_boxes.append(boxes)
        gt_labels.append(labels)
    return gt_boxes, gt_labels, gt_masks


def _rboxes_to_xyxy(rboxes: Tensor) -> Tensor:
    if rboxes.numel() == 0:
        return rboxes.new_zeros((0, 4))
    corners = ops.xywhr2xyxyxyxy(rboxes)
    x = corners[..., 0]
    y = corners[..., 1]
    return torch.stack((x.min(dim=-1).values, y.min(dim=-1).values, x.max(dim=-1).values, y.max(dim=-1).values), dim=-1)


def _hboxes_to_rboxes(boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 5))
    ctr = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_(min=1e-6)
    angle = boxes.new_zeros((boxes.shape[0], 1))
    return torch.cat((ctr, wh, angle), dim=-1)


def _clip_boxes(boxes: Tensor, img_shape: tuple[int, int]) -> Tensor:
    boxes = boxes.clone()
    boxes[:, 0::2].clamp_(0, img_shape[1])
    boxes[:, 1::2].clamp_(0, img_shape[0])
    return boxes


def _clip_rboxes(rboxes: Tensor, img_shape: tuple[int, int], angle_mode: str = "le90") -> Tensor:
    rboxes = rboxes.clone()
    rboxes[:, 0].clamp_(0, img_shape[1])
    rboxes[:, 1].clamp_(0, img_shape[0])
    rboxes[:, 2:4].clamp_(min=1e-3)
    return ops.regularize_rboxes(rboxes, angle_mode=angle_mode)


def _remove_small_boxes(boxes: Tensor, min_size: float, rotated: bool = False) -> Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    if rotated:
        keep = boxes[:, 2].ge(min_size) & boxes[:, 3].ge(min_size)
    else:
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp_(min=0)
        keep = wh[:, 0].ge(min_size) & wh[:, 1].ge(min_size)
    return torch.where(keep)[0]


def _assign_levels_from_hboxes(boxes: Tensor, min_level: int = 2, max_level: int = 5, canonical_scale: int = 224, k0: int = 4) -> Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp_(min=1e-6)
    scales = torch.sqrt(wh[:, 0] * wh[:, 1])
    levels = torch.floor(k0 + torch.log2(scales / float(canonical_scale) + 1e-6))
    return levels.clamp_(min_level, max_level).long()


def _assign_levels_from_rboxes(rboxes: Tensor, min_level: int = 2, max_level: int = 5, canonical_scale: int = 224, k0: int = 4) -> Tensor:
    if rboxes.numel() == 0:
        return rboxes.new_zeros((0,), dtype=torch.long)
    scales = torch.sqrt(rboxes[:, 2].clamp(min=1e-6) * rboxes[:, 3].clamp(min=1e-6))
    levels = torch.floor(k0 + torch.log2(scales / float(canonical_scale) + 1e-6))
    return levels.clamp_(min_level, max_level).long()


class HorizontalBoxCoder:
    def __init__(self, stds: Iterable[float] = (1.0, 1.0, 1.0, 1.0)):
        self.stds = tuple(stds)

    def encode(self, anchors: Tensor, gt: Tensor) -> Tensor:
        wa = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
        ha = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
        xa = anchors[:, 0] + 0.5 * wa
        ya = anchors[:, 1] + 0.5 * ha

        wg = (gt[:, 2] - gt[:, 0]).clamp(min=1e-6)
        hg = (gt[:, 3] - gt[:, 1]).clamp(min=1e-6)
        xg = gt[:, 0] + 0.5 * wg
        yg = gt[:, 1] + 0.5 * hg

        deltas = torch.stack(((xg - xa) / wa, (yg - ya) / ha, torch.log(wg / wa), torch.log(hg / ha)), dim=-1)
        return deltas / deltas.new_tensor(self.stds)

    def decode(self, anchors: Tensor, deltas: Tensor) -> Tensor:
        d = deltas * deltas.new_tensor(self.stds)
        wa = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
        ha = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
        xa = anchors[:, 0] + 0.5 * wa
        ya = anchors[:, 1] + 0.5 * ha

        x = d[:, 0] * wa + xa
        y = d[:, 1] * ha + ya
        w = wa * torch.exp(d[:, 2].clamp(min=-math.log(1000 / 16), max=math.log(1000 / 16)))
        h = ha * torch.exp(d[:, 3].clamp(min=-math.log(1000 / 16), max=math.log(1000 / 16)))
        return torch.stack((x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h), dim=-1)


class DeltaXYWHAHBBoxCoder:
    """Encode horizontal boxes to rotated xywhr boxes."""

    def __init__(self, stds=(1.0, 1.0, 1.0, 1.0, 1.0), angle_mode: str = "le90", norm_factor=None, edge_swap=True):
        self.stds = tuple(stds)
        self.angle_mode = angle_mode
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap

    def encode(self, boxes: Tensor, gt: Tensor) -> Tensor:
        px = (boxes[:, 0] + boxes[:, 2]) * 0.5
        py = (boxes[:, 1] + boxes[:, 3]) * 0.5
        pw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        ph = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        gx, gy, gw, gh, ga = gt.unbind(dim=-1)
        ga = ops.regularize_rboxes(gt, angle_mode=self.angle_mode)[:, 4]
        if self.edge_swap:
            alt = ops.regularize_rboxes(torch.stack((gx, gy, gh, gw, ga + math.pi / 2), dim=-1), angle_mode=self.angle_mode)
            keep_alt = alt[:, 4].abs() < ga.abs()
            gw = torch.where(keep_alt, alt[:, 2], gw)
            gh = torch.where(keep_alt, alt[:, 3], gh)
            ga = torch.where(keep_alt, alt[:, 4], ga)
        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        da = ga / (self.norm_factor * math.pi) if self.norm_factor else ga
        deltas = torch.stack((dx, dy, dw, dh, da), dim=-1)
        return deltas / deltas.new_tensor(self.stds)

    def decode(self, boxes: Tensor, deltas: Tensor) -> Tensor:
        d = deltas * deltas.new_tensor(self.stds)
        px = (boxes[:, 0] + boxes[:, 2]) * 0.5
        py = (boxes[:, 1] + boxes[:, 3]) * 0.5
        pw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        ph = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        dx, dy, dw, dh, da = d.unbind(dim=-1)
        da = da * self.norm_factor * math.pi if self.norm_factor else da
        gw = pw * torch.exp(dw.clamp(min=-math.log(1000 / 16), max=math.log(1000 / 16)))
        gh = ph * torch.exp(dh.clamp(min=-math.log(1000 / 16), max=math.log(1000 / 16)))
        gx = px + pw * dx
        gy = py + ph * dy
        rboxes = torch.stack((gx, gy, gw, gh, da), dim=-1)
        return ops.regularize_rboxes(rboxes, angle_mode=self.angle_mode)


class DeltaXYWHAOBBoxCoder:
    """Encode rotated boxes to rotated boxes."""

    def __init__(self, stds=(1.0, 1.0, 1.0, 1.0, 1.0), angle_mode="le90", norm_factor=None, edge_swap=True, proj_xy=True):
        self.stds = tuple(stds)
        self.angle_mode = angle_mode
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.proj_xy = proj_xy

    def encode(self, boxes: Tensor, gt: Tensor) -> Tensor:
        boxes = ops.regularize_rboxes(boxes, angle_mode=self.angle_mode)
        gt = ops.regularize_rboxes(gt, angle_mode=self.angle_mode)
        px, py, pw, ph, pa = boxes.unbind(dim=-1)
        gx, gy, gw, gh, ga = gt.unbind(dim=-1)
        if self.proj_xy:
            dx = (torch.cos(pa) * (gx - px) + torch.sin(pa) * (gy - py)) / pw
            dy = (-torch.sin(pa) * (gx - px) + torch.cos(pa) * (gy - py)) / ph
        else:
            dx = (gx - px) / pw
            dy = (gy - py) / ph
        delta_angle = ops.regularize_rboxes(torch.stack((gx, gy, gw, gh, ga - pa), dim=-1), angle_mode=self.angle_mode)[:, 4]
        if self.edge_swap:
            alt = ops.regularize_rboxes(torch.stack((gx, gy, gh, gw, delta_angle + math.pi / 2), dim=-1), angle_mode=self.angle_mode)
            keep_alt = alt[:, 4].abs() < delta_angle.abs()
            gw = torch.where(keep_alt, alt[:, 2], gw)
            gh = torch.where(keep_alt, alt[:, 3], gh)
            delta_angle = torch.where(keep_alt, alt[:, 4], delta_angle)
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        da = delta_angle / (self.norm_factor * math.pi) if self.norm_factor else delta_angle
        deltas = torch.stack((dx, dy, dw, dh, da), dim=-1)
        return deltas / deltas.new_tensor(self.stds)

    def decode(self, boxes: Tensor, deltas: Tensor) -> Tensor:
        boxes = ops.regularize_rboxes(boxes, angle_mode=self.angle_mode)
        d = deltas * deltas.new_tensor(self.stds)
        px, py, pw, ph, pa = boxes.unbind(dim=-1)
        dx, dy, dw, dh, da = d.unbind(dim=-1)
        da = da * self.norm_factor * math.pi if self.norm_factor else da
        gw = pw * torch.exp(dw.clamp(min=-math.log(1000 / 16), max=math.log(1000 / 16)))
        gh = ph * torch.exp(dh.clamp(min=-math.log(1000 / 16), max=math.log(1000 / 16)))
        if self.proj_xy:
            gx = px + dx * pw * torch.cos(pa) - dy * ph * torch.sin(pa)
            gy = py + dx * pw * torch.sin(pa) + dy * ph * torch.cos(pa)
        else:
            gx = px + dx * pw
            gy = py + dy * ph
        rboxes = torch.stack((gx, gy, gw, gh, pa + da), dim=-1)
        return ops.regularize_rboxes(rboxes, angle_mode=self.angle_mode)


class MidpointOffsetCoder:
    """Oriented RPN coder used by Oriented R-CNN."""

    def __init__(self, stds=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), angle_mode="le90"):
        self.stds = tuple(stds)
        self.angle_mode = angle_mode

    def encode(self, boxes: Tensor, gt: Tensor) -> Tensor:
        boxes = boxes.float()
        gt = ops.regularize_rboxes(gt.float(), angle_mode=self.angle_mode)
        px = (boxes[:, 0] + boxes[:, 2]) * 0.5
        py = (boxes[:, 1] + boxes[:, 3]) * 0.5
        pw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        ph = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        poly = ops.xywhr2xyxyxyxy(gt).reshape(-1, 8)
        hbb = _rboxes_to_xyxy(gt)
        gx = (hbb[:, 0] + hbb[:, 2]) * 0.5
        gy = (hbb[:, 1] + hbb[:, 3]) * 0.5
        gw = (hbb[:, 2] - hbb[:, 0]).clamp(min=1e-6)
        gh = (hbb[:, 3] - hbb[:, 1]).clamp(min=1e-6)

        x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]
        y_min = y_coor.min(dim=1, keepdim=True).values
        x_max = x_coor.max(dim=1, keepdim=True).values

        masked_x = x_coor.clone()
        masked_x[(y_coor - y_min).abs() > 0.1] = -1000.0
        ga = masked_x.max(dim=1).values

        masked_y = y_coor.clone()
        masked_y[(x_coor - x_max).abs() > 0.1] = -1000.0
        gb = masked_y.max(dim=1).values

        deltas = torch.stack(
            ((gx - px) / pw, (gy - py) / ph, torch.log(gw / pw), torch.log(gh / ph), (ga - gx) / gw, (gb - gy) / gh),
            dim=-1,
        )
        return deltas / deltas.new_tensor(self.stds)

    def decode(self, boxes: Tensor, deltas: Tensor) -> Tensor:
        d = deltas * deltas.new_tensor(self.stds)
        dx, dy, dw, dh, da, db = d.unbind(dim=-1)
        px = (boxes[:, 0] + boxes[:, 2]) * 0.5
        py = (boxes[:, 1] + boxes[:, 3]) * 0.5
        pw = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
        ph = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
        gw = pw * torch.exp(dw.clamp(min=-math.log(1000 / 16), max=math.log(1000 / 16)))
        gh = ph * torch.exp(dh.clamp(min=-math.log(1000 / 16), max=math.log(1000 / 16)))
        gx = px + pw * dx
        gy = py + ph * dy
        ga = gx + da.clamp(-0.5, 0.5) * gw
        _ga = gx - da.clamp(-0.5, 0.5) * gw
        gb = gy + db.clamp(-0.5, 0.5) * gh
        _gb = gy - db.clamp(-0.5, 0.5) * gh
        polys = torch.stack((ga, gy - 0.5 * gh, gx + 0.5 * gw, gb, _ga, gy + 0.5 * gh, gx - 0.5 * gw, _gb), dim=-1)
        rboxes = ops.xyxyxyxy2xywhr(polys.view(-1, 4, 2), angle_mode=self.angle_mode)
        return ops.regularize_rboxes(rboxes, angle_mode=self.angle_mode)


class AnchorGenerator(nn.Module):
    def __init__(self, strides=(4, 8, 16, 32, 64), scales=(1, 2, 4), ratios=(0.5, 1.0, 2.0)):
        super().__init__()
        self.strides = tuple(strides)
        self.scales = tuple(scales)
        self.ratios = tuple(ratios)
        self.num_anchors = len(self.scales) * len(self.ratios)

    def grid_anchors(self, feat_shapes: list[tuple[int, int]], device: torch.device) -> list[Tensor]:
        anchors = []
        for (h, w), stride in zip(feat_shapes, self.strides):
            shift_x = torch.arange(w, device=device, dtype=torch.float32) * stride + 0.5 * stride
            shift_y = torch.arange(h, device=device, dtype=torch.float32) * stride + 0.5 * stride
            yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
            centers = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
            base = []
            for scale in self.scales:
                size = stride * scale
                for ratio in self.ratios:
                    area = size * size
                    aw = math.sqrt(area / ratio)
                    ah = aw * ratio
                    base.append(torch.tensor([-0.5 * aw, -0.5 * ah, 0.5 * aw, 0.5 * ah], device=device))
            base = torch.stack(base, dim=0)
            centers_xyxy = torch.cat((centers, centers), dim=-1)[:, None, :]
            anchors.append((centers_xyxy + base[None]).reshape(-1, 4))
        return anchors


class _RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, reg_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.obj = nn.Conv2d(in_channels, num_anchors, 1)
        self.reg = nn.Conv2d(in_channels, num_anchors * reg_dim, 1)
        for layer in (self.conv, self.obj, self.reg):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, feats: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        scores, deltas = [], []
        for feat in feats:
            x = F.relu(self.conv(feat), inplace=True)
            scores.append(self.obj(x))
            deltas.append(self.reg(x))
        return scores, deltas


class _TwoFCHead(nn.Module):
    def __init__(self, in_channels: int, pool_size: int = 7, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(in_channels * pool_size * pool_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return x


class _BBoxHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, reg_dim: int, pool_size: int = 7, hidden_dim: int = 1024):
        super().__init__()
        self.tower = _TwoFCHead(in_channels, pool_size=pool_size, hidden_dim=hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(hidden_dim, reg_dim)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.tower(x)
        return self.cls_score(feat), self.bbox_pred(feat)


class _MaskHead(nn.Module):
    def __init__(self, in_channels: int, dim: int = 256, num_convs: int = 4):
        super().__init__()
        blocks = []
        for i in range(num_convs):
            blocks.append(nn.Conv2d(in_channels if i == 0 else dim, dim, 3, padding=1))
            blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)
        self.up = nn.ConvTranspose2d(dim, dim, 2, stride=2)
        self.out = nn.Conv2d(dim, 1, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = F.relu(self.up(x), inplace=True)
        return self.out(x)


def _roi_align_multilevel(feats: list[Tensor], rois: Tensor, output_size: int, sampling_ratio: int, featmap_strides=(4, 8, 16, 32)) -> Tensor:
    if rois.numel() == 0:
        return feats[0].new_zeros((0, feats[0].shape[1], output_size, output_size))
    levels = _assign_levels_from_hboxes(rois[:, 1:5], max_level=min(5, len(featmap_strides) + 1))
    pooled = feats[0].new_zeros((rois.shape[0], feats[0].shape[1], output_size, output_size))
    for level, stride in enumerate(featmap_strides, start=2):
        idx = torch.where(levels == level)[0]
        if idx.numel() == 0:
            continue
        pooled[idx] = roi_align(
            feats[level - 2],
            rois[idx],
            output_size=output_size,
            spatial_scale=1.0 / float(stride),
            sampling_ratio=sampling_ratio,
            aligned=True,
        )
    return pooled


def _rotated_roi_align_single(feat: Tensor, rois: Tensor, output_size: int) -> Tensor:
    if rois.numel() == 0:
        return feat.new_zeros((0, feat.shape[1], output_size, output_size))
    _, _, h, w = feat.shape
    batch_idx = rois[:, 0].long()
    boxes = rois[:, 1:].to(dtype=feat.dtype)
    cx, cy, bw, bh, angle = boxes.unbind(dim=-1)
    xs = (torch.arange(output_size, device=feat.device, dtype=feat.dtype) + 0.5) / output_size - 0.5
    ys = (torch.arange(output_size, device=feat.device, dtype=feat.dtype) + 0.5) / output_size - 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx[None] * bw[:, None, None]
    yy = yy[None] * bh[:, None, None]
    cos_a = torch.cos(angle)[:, None, None]
    sin_a = torch.sin(angle)[:, None, None]
    gx = cx[:, None, None] + xx * cos_a - yy * sin_a
    gy = cy[:, None, None] + xx * sin_a + yy * cos_a
    grid = torch.stack((2 * gx / max(w - 1, 1) - 1, 2 * gy / max(h - 1, 1) - 1), dim=-1)
    pooled = feat.new_zeros((rois.shape[0], feat.shape[1], output_size, output_size))
    for bi in batch_idx.unique(sorted=True):
        idx = torch.where(batch_idx == bi)[0]
        pooled[idx] = F.grid_sample(
            feat[bi : bi + 1].expand(idx.numel(), -1, -1, -1),
            grid[idx],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    return pooled


def _rotated_roi_align_multilevel(feats: list[Tensor], rois: Tensor, output_size: int, featmap_strides=(4, 8, 16, 32)) -> Tensor:
    if rois.numel() == 0:
        return feats[0].new_zeros((0, feats[0].shape[1], output_size, output_size))
    levels = _assign_levels_from_rboxes(rois[:, 1:], max_level=min(5, len(featmap_strides) + 1))
    pooled = feats[0].new_zeros((rois.shape[0], feats[0].shape[1], output_size, output_size))
    for level, stride in enumerate(featmap_strides, start=2):
        idx = torch.where(levels == level)[0]
        if idx.numel() == 0:
            continue
        rois_scaled = rois[idx].clone()
        rois_scaled[:, 1:5] /= float(stride)
        pooled[idx] = _rotated_roi_align_single(feats[level - 2], rois_scaled, output_size)
    return pooled


def _sample_matches(pos_mask: Tensor, neg_mask: Tensor, num_samples: int, pos_fraction: float) -> Tensor:
    pos_inds = torch.where(pos_mask)[0]
    neg_inds = torch.where(neg_mask)[0]
    num_pos = min(int(num_samples * pos_fraction), pos_inds.numel())
    num_neg = min(num_samples - num_pos, neg_inds.numel())
    if pos_inds.numel() > num_pos:
        pos_inds = pos_inds[torch.randperm(pos_inds.numel(), device=pos_inds.device)[:num_pos]]
    if neg_inds.numel() > num_neg:
        neg_inds = neg_inds[torch.randperm(neg_inds.numel(), device=neg_inds.device)[:num_neg]]
    return torch.cat((pos_inds, neg_inds), dim=0)


def _match_hboxes(boxes: Tensor, gt_boxes: Tensor, gt_labels: Tensor, pos_iou: float, neg_iou: float, num_samples: int, pos_fraction: float):
    device = boxes.device
    if gt_boxes.numel() == 0:
        labels = torch.zeros((boxes.shape[0],), device=device, dtype=torch.long)
        sampled = _sample_matches(labels > 0, labels == 0, num_samples, pos_fraction)
        return sampled, labels[sampled], boxes[sampled], gt_boxes.new_zeros((sampled.numel(), 4)), gt_boxes.new_full((sampled.numel(),), -1, dtype=torch.long)
    iou = box_iou(boxes, gt_boxes)
    max_iou, matched_gt = iou.max(dim=1)
    labels = gt_labels[matched_gt] + 1
    labels[max_iou < neg_iou] = 0
    pos_mask = max_iou >= pos_iou
    labels[~pos_mask & (max_iou >= neg_iou)] = -1
    sampled = _sample_matches(labels > 0, labels == 0, num_samples, pos_fraction)
    return sampled, labels[sampled], boxes[sampled], gt_boxes[matched_gt[sampled]], matched_gt[sampled]


def _match_rboxes(boxes: Tensor, gt_boxes: Tensor, gt_labels: Tensor, pos_iou: float, neg_iou: float, num_samples: int, pos_fraction: float):
    device = boxes.device
    if gt_boxes.numel() == 0:
        labels = torch.zeros((boxes.shape[0],), device=device, dtype=torch.long)
        sampled = _sample_matches(labels > 0, labels == 0, num_samples, pos_fraction)
        return sampled, labels[sampled], boxes[sampled], gt_boxes.new_zeros((sampled.numel(), 5)), gt_boxes.new_full((sampled.numel(),), -1, dtype=torch.long)
    iou = batch_probiou(boxes, gt_boxes)
    max_iou, matched_gt = iou.max(dim=1)
    labels = gt_labels[matched_gt] + 1
    labels[max_iou < neg_iou] = 0
    pos_mask = max_iou >= pos_iou
    labels[~pos_mask & (max_iou >= neg_iou)] = -1
    sampled = _sample_matches(labels > 0, labels == 0, num_samples, pos_fraction)
    return sampled, labels[sampled], boxes[sampled], gt_boxes[matched_gt[sampled]], matched_gt[sampled]


def _crop_mask_targets(gt_masks: Tensor | None, proposals: Tensor, matched_gt_inds: Tensor, mask_size: int) -> Tensor:
    if gt_masks is None or proposals.numel() == 0 or matched_gt_inds.numel() == 0:
        return proposals.new_zeros((0, mask_size, mask_size))
    targets = []
    for prop, gt_idx in zip(proposals, matched_gt_inds):
        mask = gt_masks[gt_idx].float()
        x1, y1, x2, y2 = prop.round().long().tolist()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(mask.shape[1], x2)
        y2 = min(mask.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            crop = mask.new_zeros((mask_size, mask_size))
        else:
            crop = F.interpolate(mask[y1:y2, x1:x2][None, None], size=(mask_size, mask_size), mode="bilinear", align_corners=False)[0, 0]
        targets.append(crop)
    return torch.stack(targets, dim=0) if targets else proposals.new_zeros((0, mask_size, mask_size))


def _paste_masks(mask_logits: Tensor, boxes: Tensor, image_shape: tuple[int, int]) -> Tensor:
    if mask_logits.numel() == 0:
        return mask_logits.new_zeros((0, image_shape[0], image_shape[1]))
    masks = []
    probs = mask_logits.sigmoid()
    for mask, box in zip(probs, boxes):
        x1, y1, x2, y2 = box.round().long().tolist()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_shape[1], x2)
        y2 = min(image_shape[0], y2)
        canvas = mask.new_zeros(image_shape)
        if x2 > x1 and y2 > y1:
            resized = F.interpolate(mask[None, None], size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)[0, 0]
            canvas[y1:y2, x1:x2] = resized
        masks.append(canvas)
    return torch.stack(masks, dim=0)


class _AxisRCNNBase(nn.Module):
    default_cfg = {
        "rpn": {
            "anchor_scales": [1, 2, 4],
            "anchor_ratios": [0.5, 1.0, 2.0],
            "strides": [4, 8, 16, 32, 64],
            "pre_nms_topk_train": 2000,
            "post_nms_topk_train": 2000,
            "pre_nms_topk_test": 2000,
            "post_nms_topk_test": 1000,
            "nms_thresh": 0.7,
            "min_box_size": 0.0,
            "pos_iou": 0.7,
            "neg_iou": 0.3,
            "samples_per_img": 256,
            "pos_fraction": 0.5,
            "beta": 1.0 / 9.0,
        },
        "roi": {"pool_size": 7, "mask_pool_size": 14, "sampling_ratio": 2, "featmap_strides": [4, 8, 16, 32]},
        "train": {"pos_iou": 0.5, "neg_iou": 0.5, "samples_per_img": 512, "pos_fraction": 0.25},
        "test": {"score_thresh": 0.05, "nms_iou": 0.5, "max_dets": 300},
        "bbox_head": {"hidden_dim": 1024, "loss": "smooth_l1", "beta": 1.0},
        "mask_head": {"dim": 256, "num_convs": 4, "resolution": 28},
        "cascade": {"iou_thresholds": [0.5, 0.6, 0.7], "bbox_stds": [[0.1, 0.1, 0.2, 0.2], [0.05, 0.05, 0.1, 0.1], [0.033, 0.033, 0.067, 0.067]]},
    }

    def __init__(self, in_channels: list[int], nc: int, cfg: dict | None = None, with_mask: bool = False, cascade: bool = False):
        super().__init__()
        self.nc = nc
        self.with_mask = with_mask
        self.cascade = cascade
        self.cfg = _merge_dict(self.default_cfg, cfg)
        self.loss_names = ["rpn_cls", "rpn_box"]
        self.anchor_generator = AnchorGenerator(
            strides=self.cfg["rpn"]["strides"],
            scales=self.cfg["rpn"]["anchor_scales"],
            ratios=self.cfg["rpn"]["anchor_ratios"],
        )
        self.rpn_head = _RPNHead(in_channels[0], self.anchor_generator.num_anchors, reg_dim=4)
        stage_stds = self.cfg["cascade"]["bbox_stds"] if cascade else [self.cfg["cascade"]["bbox_stds"][0]]
        self.stage_coders = [HorizontalBoxCoder(stds=s) for s in stage_stds]
        self.bbox_heads = nn.ModuleList(
            [_BBoxHead(in_channels[0], num_classes=nc, reg_dim=4, pool_size=self.cfg["roi"]["pool_size"], hidden_dim=self.cfg["bbox_head"]["hidden_dim"]) for _ in stage_stds]
        )
        if cascade:
            for stage in range(1, 4):
                self.loss_names.extend((f"s{stage}_cls", f"s{stage}_box"))
        else:
            self.loss_names.extend(("rcnn_cls", "rcnn_box"))
        if with_mask:
            self.mask_head = _MaskHead(in_channels[0], dim=self.cfg["mask_head"]["dim"], num_convs=self.cfg["mask_head"]["num_convs"])
            self.loss_names.append("mask")

    def _image_shape_from_feats(self, feats: list[Tensor]) -> tuple[int, int]:
        stride = int(self.cfg["roi"]["featmap_strides"][0])
        return feats[0].shape[-2] * stride, feats[0].shape[-1] * stride

    def _rpn_forward(self, feats: list[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        scores, deltas = self.rpn_head(feats)
        shapes = [(x.shape[2], x.shape[3]) for x in feats]
        anchors = self.anchor_generator.grid_anchors(shapes, feats[0].device)
        return scores, deltas, anchors

    def _rpn_loss_and_proposals(self, feats: list[Tensor], gt_boxes: list[Tensor], image_shape: tuple[int, int], train: bool):
        obj_logits, bbox_deltas, anchors = self._rpn_forward(feats)
        total_obj, total_box, proposals = 0.0, 0.0, []
        batch_size = obj_logits[0].shape[0]
        coder = HorizontalBoxCoder()
        pre_nms = self.cfg["rpn"]["pre_nms_topk_train" if train else "pre_nms_topk_test"]
        post_nms = self.cfg["rpn"]["post_nms_topk_train" if train else "post_nms_topk_test"]
        for bi in range(batch_size):
            per_img_scores, per_img_boxes, level_ids = [], [], []
            if train:
                img_cls_loss = feats[0].new_tensor(0.0)
                img_box_loss = feats[0].new_tensor(0.0)
            for level, (lvl_logits, lvl_deltas, lvl_anchors) in enumerate(zip(obj_logits, bbox_deltas, anchors)):
                scores = lvl_logits[bi].permute(1, 2, 0).reshape(-1)
                deltas = lvl_deltas[bi].permute(1, 2, 0).reshape(-1, 4)
                if train:
                    labels = lvl_anchors.new_full((lvl_anchors.shape[0],), -1, dtype=torch.long)
                    box_targets = lvl_anchors.new_zeros((lvl_anchors.shape[0], 4))
                    if gt_boxes[bi].numel():
                        iou = box_iou(lvl_anchors, gt_boxes[bi])
                        max_iou, matched = iou.max(dim=1)
                        labels[max_iou < self.cfg["rpn"]["neg_iou"]] = 0
                        labels[max_iou >= self.cfg["rpn"]["pos_iou"]] = 1
                        labels[iou.argmax(dim=0)] = 1
                        pos = labels == 1
                        if pos.any():
                            box_targets[pos] = coder.encode(lvl_anchors[pos], gt_boxes[bi][matched[pos]])
                    sampled = _sample_matches(labels == 1, labels == 0, self.cfg["rpn"]["samples_per_img"] // len(anchors), self.cfg["rpn"]["pos_fraction"])
                    if sampled.numel():
                        cls_t = (labels[sampled] == 1).float()
                        img_cls_loss = img_cls_loss + F.binary_cross_entropy_with_logits(scores[sampled], cls_t, reduction="mean")
                        pos = labels[sampled] == 1
                        if pos.any():
                            img_box_loss = img_box_loss + F.smooth_l1_loss(deltas[sampled][pos], box_targets[sampled][pos], beta=self.cfg["rpn"]["beta"], reduction="mean")
                probs = scores.sigmoid()
                topk = min(pre_nms, probs.numel())
                idx = probs.topk(topk).indices
                boxes = _clip_boxes(coder.decode(lvl_anchors[idx], deltas[idx]), image_shape)
                keep = _remove_small_boxes(boxes, self.cfg["rpn"]["min_box_size"])
                per_img_scores.append(probs[idx][keep])
                per_img_boxes.append(boxes[keep])
                level_ids.append(boxes.new_full((keep.numel(),), level, dtype=torch.long))
            if train:
                total_obj = total_obj + img_cls_loss / max(len(anchors), 1)
                total_box = total_box + img_box_loss / max(len(anchors), 1)
            boxes = torch.cat(per_img_boxes, dim=0) if per_img_boxes else feats[0].new_zeros((0, 4))
            scores = torch.cat(per_img_scores, dim=0) if per_img_scores else feats[0].new_zeros((0,))
            lvl_ids = torch.cat(level_ids, dim=0) if level_ids else feats[0].new_zeros((0,), dtype=torch.long)
            keep = TorchNMS.batched_nms(boxes, scores, lvl_ids, self.cfg["rpn"]["nms_thresh"])[:post_nms]
            proposals.append(boxes[keep])
        return total_obj / batch_size, total_box / batch_size, proposals

    def _bbox_reg_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.numel() == 0:
            return pred.sum() * 0.0
        if self.cfg["bbox_head"]["loss"] == "l1":
            return F.l1_loss(pred, target, reduction="mean")
        return F.smooth_l1_loss(pred, target, beta=self.cfg["bbox_head"]["beta"], reduction="mean")

    def _single_stage_loss(self, feats: list[Tensor], proposals: list[Tensor], gt_boxes: list[Tensor], gt_labels: list[Tensor]):
        sampled_rois, labels_all, matched_boxes, matched_gt_inds = [], [], [], []
        for bi, props in enumerate(proposals):
            props = torch.cat((props, gt_boxes[bi]), dim=0) if gt_boxes[bi].numel() else props
            sampled, labels, boxes, targets, gt_inds = _match_hboxes(
                props, gt_boxes[bi], gt_labels[bi], self.cfg["train"]["pos_iou"], self.cfg["train"]["neg_iou"], self.cfg["train"]["samples_per_img"], self.cfg["train"]["pos_fraction"]
            )
            rois = torch.cat((boxes.new_full((boxes.shape[0], 1), bi), boxes), dim=1)
            sampled_rois.append(rois)
            labels_all.append(labels)
            matched_boxes.append(targets)
            matched_gt_inds.append(gt_inds)
        rois = torch.cat(sampled_rois, dim=0) if sampled_rois else feats[0].new_zeros((0, 5))
        labels = torch.cat(labels_all, dim=0) if labels_all else feats[0].new_zeros((0,), dtype=torch.long)
        matched = torch.cat(matched_boxes, dim=0) if matched_boxes else feats[0].new_zeros((0, 4))
        gt_inds = torch.cat(matched_gt_inds, dim=0) if matched_gt_inds else feats[0].new_zeros((0,), dtype=torch.long)
        pooled = _roi_align_multilevel(feats[:4], rois, self.cfg["roi"]["pool_size"], self.cfg["roi"]["sampling_ratio"], self.cfg["roi"]["featmap_strides"])
        if pooled.shape[0] == 0:
            zero = feats[0].sum() * 0.0
            return zero, zero, rois.new_zeros((0, 5)), gt_inds.new_zeros((0,), dtype=torch.long)
        cls_logits, box_deltas = self.bbox_heads[0](pooled)
        cls_loss = F.cross_entropy(cls_logits, labels)
        pos = labels > 0
        box_loss = self._bbox_reg_loss(box_deltas[pos], self.stage_coders[0].encode(rois[:, 1:5][pos], matched[pos])) if pos.any() else cls_loss * 0.0
        return cls_loss, box_loss, rois[pos], gt_inds[pos]

    def _cascade_stage_loss(self, feats: list[Tensor], proposals: list[Tensor], gt_boxes: list[Tensor], gt_labels: list[Tensor], image_shape: tuple[int, int]):
        stage_losses, stage_pos_rois, stage_gt_inds = [], None, None
        current_props = [torch.cat((props, gt_boxes[i]), dim=0) if gt_boxes[i].numel() else props for i, props in enumerate(proposals)]
        for stage_idx, (head, coder, iou_thr) in enumerate(zip(self.bbox_heads, self.stage_coders, self.cfg["cascade"]["iou_thresholds"])):
            all_rois = []
            counts = []
            for bi, props in enumerate(current_props):
                all_rois.append(torch.cat((props.new_full((props.shape[0], 1), bi), props), dim=1))
                counts.append(props.shape[0])
            rois_all = torch.cat(all_rois, dim=0) if all_rois else feats[0].new_zeros((0, 5))
            pooled = _roi_align_multilevel(feats[:4], rois_all, self.cfg["roi"]["pool_size"], self.cfg["roi"]["sampling_ratio"], self.cfg["roi"]["featmap_strides"])
            cls_logits_all, box_deltas_all = head(pooled)

            offset = 0
            sampled_global, labels_all, reg_targets, pos_rois, pos_gt_inds = [], [], [], [], []
            refined_props = []
            for bi, props in enumerate(current_props):
                num = counts[bi]
                logits = cls_logits_all[offset : offset + num]
                deltas = box_deltas_all[offset : offset + num]
                refined = _clip_boxes(coder.decode(props, deltas), image_shape)
                refined_props.append(refined.detach())
                sampled, labels, boxes, targets, gt_inds = _match_hboxes(props, gt_boxes[bi], gt_labels[bi], iou_thr, iou_thr, self.cfg["train"]["samples_per_img"], self.cfg["train"]["pos_fraction"])
                sampled_global.append(sampled + offset)
                labels_all.append(labels)
                reg_targets.append(coder.encode(boxes[labels > 0], targets[labels > 0]) if (labels > 0).any() else boxes.new_zeros((0, 4)))
                if (labels > 0).any():
                    pos_idx = sampled[labels > 0]
                    pos_rois.append(torch.cat((props.new_full((pos_idx.numel(), 1), bi), boxes[labels > 0]), dim=1))
                    pos_gt_inds.append(gt_inds[labels > 0])
                offset += num
            current_props = refined_props
            sample_idx = torch.cat(sampled_global, dim=0) if sampled_global else feats[0].new_zeros((0,), dtype=torch.long)
            labels = torch.cat(labels_all, dim=0) if labels_all else feats[0].new_zeros((0,), dtype=torch.long)
            cls_loss = F.cross_entropy(cls_logits_all[sample_idx], labels) if sample_idx.numel() else cls_logits_all.sum() * 0.0
            box_idx = labels > 0
            if box_idx.any():
                reg_pred = box_deltas_all[sample_idx][box_idx]
                reg_target = torch.cat(reg_targets, dim=0)
                box_loss = self._bbox_reg_loss(reg_pred, reg_target)
            else:
                box_loss = cls_loss * 0.0
            stage_losses.extend((cls_loss, box_loss))
            if pos_rois:
                stage_pos_rois = torch.cat(pos_rois, dim=0)
                stage_gt_inds = torch.cat(pos_gt_inds, dim=0)
        return stage_losses, current_props, stage_pos_rois, stage_gt_inds

    def _mask_loss(self, feats: list[Tensor], pos_rois: Tensor, gt_inds: Tensor, gt_masks: list[Tensor | None]) -> Tensor:
        if pos_rois is None or pos_rois.numel() == 0:
            return feats[0].sum() * 0.0
        pooled = _roi_align_multilevel(feats[:4], pos_rois, self.cfg["roi"]["mask_pool_size"], self.cfg["roi"]["sampling_ratio"], self.cfg["roi"]["featmap_strides"])
        logits = self.mask_head(pooled).squeeze(1)
        targets = []
        for bi_tensor in pos_rois[:, 0].long().unique(sorted=True):
            bi = int(bi_tensor.item())
            idx = pos_rois[:, 0].long() == bi
            targets.append(_crop_mask_targets(gt_masks[bi], pos_rois[idx, 1:5], gt_inds[idx], self.cfg["mask_head"]["resolution"]))
        target = torch.cat(targets, dim=0) if targets else logits.new_zeros((0, self.cfg["mask_head"]["resolution"], self.cfg["mask_head"]["resolution"]))
        return F.binary_cross_entropy_with_logits(logits, target)

    def loss(self, feats: list[Tensor], batch: dict) -> tuple[Tensor, Tensor]:
        gt_boxes, gt_labels, gt_masks = _split_targets(batch, "segment")
        image_shape = tuple(batch["img"].shape[2:])
        rpn_cls, rpn_box, proposals = self._rpn_loss_and_proposals(feats, gt_boxes, image_shape, train=True)
        if self.cascade:
            stage_losses, final_props, pos_rois, gt_inds = self._cascade_stage_loss(feats, proposals, gt_boxes, gt_labels, image_shape)
            losses = [rpn_cls, rpn_box, *stage_losses]
            if self.with_mask:
                losses.append(self._mask_loss(feats, pos_rois, gt_inds, gt_masks))
        else:
            cls_loss, box_loss, pos_rois, gt_inds = self._single_stage_loss(feats, proposals, gt_boxes, gt_labels)
            losses = [rpn_cls, rpn_box, cls_loss, box_loss]
            if self.with_mask:
                losses.append(self._mask_loss(feats, pos_rois, gt_inds, gt_masks))
        loss_items = torch.stack([x if isinstance(x, Tensor) else feats[0].new_tensor(float(x)) for x in losses])
        return loss_items.sum(), loss_items.detach()

    @torch.no_grad()
    def forward(self, feats: list[Tensor]) -> list[dict[str, Tensor]]:
        image_shape = self._image_shape_from_feats(feats)
        _, _, proposals = self._rpn_loss_and_proposals(feats, [feats[0].new_zeros((0, 4)) for _ in range(feats[0].shape[0])], image_shape, train=False)
        if self.cascade:
            current_props = proposals
            cls_logits_all = None
            offset_boxes = None
            for head, coder in zip(self.bbox_heads, self.stage_coders):
                rois = torch.cat([torch.cat((p.new_full((p.shape[0], 1), i), p), dim=1) for i, p in enumerate(current_props)], dim=0)
                pooled = _roi_align_multilevel(feats[:4], rois, self.cfg["roi"]["pool_size"], self.cfg["roi"]["sampling_ratio"], self.cfg["roi"]["featmap_strides"])
                cls_logits_all, bbox_deltas_all = head(pooled)
                new_props, offset = [], 0
                for props in current_props:
                    num = props.shape[0]
                    pred_boxes = _clip_boxes(coder.decode(props, bbox_deltas_all[offset : offset + num]), image_shape)
                    new_props.append(pred_boxes)
                    offset += num
                current_props = new_props
            per_image_props = current_props
            per_image_logits = []
            offset = 0
            for props in per_image_props:
                per_image_logits.append(cls_logits_all[offset : offset + props.shape[0]])
                offset += props.shape[0]
        else:
            per_image_props, per_image_logits = [], []
            for i, props in enumerate(proposals):
                if props.numel() == 0:
                    per_image_props.append(props.new_zeros((0, 4)))
                    per_image_logits.append(props.new_zeros((0, self.nc + 1)))
                    continue
                rois = torch.cat((props.new_full((props.shape[0], 1), i), props), dim=1)
                pooled = _roi_align_multilevel(feats[:4], rois, self.cfg["roi"]["pool_size"], self.cfg["roi"]["sampling_ratio"], self.cfg["roi"]["featmap_strides"])
                logits, deltas = self.bbox_heads[0](pooled)
                per_image_props.append(_clip_boxes(self.stage_coders[0].decode(props, deltas), image_shape))
                per_image_logits.append(logits)

        outputs = []
        for bi, (boxes, logits) in enumerate(zip(per_image_props, per_image_logits)):
            scores = logits.softmax(dim=-1)[:, 1:]
            num_classes = scores.shape[1]
            all_boxes = boxes[:, None, :].expand(boxes.shape[0], num_classes, 4).reshape(-1, 4)
            all_scores = scores.reshape(-1)
            all_labels = torch.arange(num_classes, device=boxes.device).repeat(boxes.shape[0])
            keep = all_scores > self.cfg["test"]["score_thresh"]
            all_boxes, all_scores, all_labels = all_boxes[keep], all_scores[keep], all_labels[keep]
            keep_nms = TorchNMS.batched_nms(all_boxes, all_scores, all_labels, self.cfg["test"]["nms_iou"])[: self.cfg["test"]["max_dets"]]
            pred_boxes = all_boxes[keep_nms]
            pred_scores = all_scores[keep_nms]
            pred_labels = all_labels[keep_nms]
            pred_masks = None
            if self.with_mask and pred_boxes.numel():
                rois = torch.cat((pred_boxes.new_full((pred_boxes.shape[0], 1), bi), pred_boxes), dim=1)
                pooled = _roi_align_multilevel(feats[:4], rois, self.cfg["roi"]["mask_pool_size"], self.cfg["roi"]["sampling_ratio"], self.cfg["roi"]["featmap_strides"])
                pred_masks = _paste_masks(self.mask_head(pooled).squeeze(1), pred_boxes, image_shape)
            outputs.append({"bboxes": pred_boxes, "conf": pred_scores, "cls": pred_labels.float(), "masks": pred_masks})
        return outputs


class _RotatedRCNNBase(nn.Module):
    default_cfg = {
        "angle_mode": "le90",
        "rpn": {
            "anchor_scales": [2, 4, 8],
            "anchor_ratios": [0.5, 1.0, 2.0],
            "strides": [4, 8, 16, 32, 64],
            "pre_nms_topk_train": 2000,
            "post_nms_topk_train": 2000,
            "pre_nms_topk_test": 2000,
            "post_nms_topk_test": 2000,
            "nms_thresh": 0.8,
            "min_box_size": 0.0,
            "pos_iou": 0.7,
            "neg_iou": 0.3,
            "samples_per_img": 256,
            "pos_fraction": 0.5,
            "beta": 1.0 / 9.0,
        },
        "roi": {"pool_size": 7, "sampling_ratio": 2, "featmap_strides": [4, 8, 16, 32]},
        "train": {"pos_iou": 0.5, "neg_iou": 0.5, "samples_per_img": 512, "pos_fraction": 0.25},
        "test": {"score_thresh": 0.05, "nms_iou": 0.1, "max_dets": 2000},
        "bbox_head": {"hidden_dim": 1024, "beta": 1.0},
    }

    def __init__(self, in_channels: list[int], nc: int, cfg: dict | None = None, oriented_proposals: bool = False):
        super().__init__()
        self.nc = nc
        self.cfg = _merge_dict(self.default_cfg, cfg)
        self.angle_mode = self.cfg["angle_mode"]
        self.oriented_proposals = oriented_proposals
        self.loss_names = ["rpn_cls", "rpn_box", "rcnn_cls", "rcnn_box"]
        self.anchor_generator = AnchorGenerator(
            strides=self.cfg["rpn"]["strides"],
            scales=self.cfg["rpn"]["anchor_scales"],
            ratios=self.cfg["rpn"]["anchor_ratios"],
        )
        reg_dim = 6 if oriented_proposals else 4
        self.rpn_head = _RPNHead(in_channels[0], self.anchor_generator.num_anchors, reg_dim=reg_dim)
        self.rpn_coder = MidpointOffsetCoder(angle_mode=self.angle_mode) if oriented_proposals else HorizontalBoxCoder()
        self.roi_coder = DeltaXYWHAOBBoxCoder(stds=(0.1, 0.1, 0.2, 0.2, 0.1), angle_mode=self.angle_mode, edge_swap=True, proj_xy=True) if oriented_proposals else DeltaXYWHAHBBoxCoder(stds=(0.1, 0.1, 0.2, 0.2, 0.1), angle_mode=self.angle_mode, norm_factor=2, edge_swap=True)
        self.bbox_head = _BBoxHead(in_channels[0], num_classes=nc, reg_dim=5, pool_size=self.cfg["roi"]["pool_size"], hidden_dim=self.cfg["bbox_head"]["hidden_dim"])

    def _image_shape_from_feats(self, feats: list[Tensor]) -> tuple[int, int]:
        stride = int(self.cfg["roi"]["featmap_strides"][0])
        return feats[0].shape[-2] * stride, feats[0].shape[-1] * stride

    def _rpn_loss_and_proposals(self, feats: list[Tensor], gt_boxes: list[Tensor], image_shape: tuple[int, int], train: bool):
        obj_logits, bbox_deltas = self.rpn_head(feats)
        anchors = self.anchor_generator.grid_anchors([(x.shape[2], x.shape[3]) for x in feats], feats[0].device)
        total_obj, total_box, proposals = 0.0, 0.0, []
        batch_size = obj_logits[0].shape[0]
        gt_hboxes = [_rboxes_to_xyxy(x) if x.numel() else x.new_zeros((0, 4)) for x in gt_boxes]
        pre_nms = self.cfg["rpn"]["pre_nms_topk_train" if train else "pre_nms_topk_test"]
        post_nms = self.cfg["rpn"]["post_nms_topk_train" if train else "post_nms_topk_test"]
        for bi in range(batch_size):
            img_cls_loss = feats[0].new_tensor(0.0)
            img_box_loss = feats[0].new_tensor(0.0)
            per_scores, per_boxes = [], []
            for lvl_logits, lvl_deltas, lvl_anchors in zip(obj_logits, bbox_deltas, anchors):
                scores = lvl_logits[bi].permute(1, 2, 0).reshape(-1)
                deltas = lvl_deltas[bi].permute(1, 2, 0).reshape(-1, 6 if self.oriented_proposals else 4)
                if train:
                    labels = lvl_anchors.new_full((lvl_anchors.shape[0],), -1, dtype=torch.long)
                    # Keep regression targets in anchor precision so AMP half deltas do not
                    # downcast the encoded targets before assignment.
                    box_targets = lvl_anchors.new_zeros((lvl_anchors.shape[0], deltas.shape[1]))
                    if gt_hboxes[bi].numel():
                        iou = box_iou(lvl_anchors, gt_hboxes[bi])
                        max_iou, matched = iou.max(dim=1)
                        labels[max_iou < self.cfg["rpn"]["neg_iou"]] = 0
                        labels[max_iou >= self.cfg["rpn"]["pos_iou"]] = 1
                        labels[iou.argmax(dim=0)] = 1
                        pos = labels == 1
                        if pos.any():
                            targets = gt_boxes[bi][matched[pos]] if self.oriented_proposals else gt_hboxes[bi][matched[pos]]
                            box_targets[pos] = self.rpn_coder.encode(lvl_anchors[pos], targets)
                    sampled = _sample_matches(labels == 1, labels == 0, self.cfg["rpn"]["samples_per_img"] // len(anchors), self.cfg["rpn"]["pos_fraction"])
                    if sampled.numel():
                        img_cls_loss = img_cls_loss + F.binary_cross_entropy_with_logits(scores[sampled], (labels[sampled] == 1).float(), reduction="mean")
                        pos = labels[sampled] == 1
                        if pos.any():
                            img_box_loss = img_box_loss + F.smooth_l1_loss(deltas[sampled][pos], box_targets[sampled][pos], beta=self.cfg["rpn"]["beta"], reduction="mean")
                probs = scores.sigmoid()
                topk = min(pre_nms, probs.numel())
                idx = probs.topk(topk).indices
                decoded = self.rpn_coder.decode(lvl_anchors[idx], deltas[idx].detach())
                decoded = _clip_rboxes(decoded, image_shape, self.angle_mode) if self.oriented_proposals else _clip_boxes(decoded, image_shape)
                keep = _remove_small_boxes(decoded, self.cfg["rpn"]["min_box_size"], rotated=self.oriented_proposals)
                per_scores.append(probs[idx][keep])
                per_boxes.append(decoded[keep])
            total_obj = total_obj + img_cls_loss / max(len(anchors), 1)
            total_box = total_box + img_box_loss / max(len(anchors), 1)
            boxes = torch.cat(per_boxes, dim=0) if per_boxes else feats[0].new_zeros((0, 5 if self.oriented_proposals else 4))
            scores = torch.cat(per_scores, dim=0) if per_scores else feats[0].new_zeros((0,))
            if self.oriented_proposals:
                keep = TorchNMS.fast_nms(boxes, scores, self.cfg["rpn"]["nms_thresh"], iou_func=batch_probiou)[:post_nms]
            else:
                keep = TorchNMS.nms(boxes, scores, self.cfg["rpn"]["nms_thresh"])[:post_nms]
            proposals.append(boxes[keep])
        return total_obj / batch_size, total_box / batch_size, proposals

    def _roi_pool(self, feats: list[Tensor], rois: Tensor) -> Tensor:
        if self.oriented_proposals:
            return _rotated_roi_align_multilevel(feats[:4], rois, self.cfg["roi"]["pool_size"], self.cfg["roi"]["featmap_strides"])
        return _roi_align_multilevel(feats[:4], rois, self.cfg["roi"]["pool_size"], self.cfg["roi"]["sampling_ratio"], self.cfg["roi"]["featmap_strides"])

    def loss(self, feats: list[Tensor], batch: dict) -> tuple[Tensor, Tensor]:
        gt_boxes, gt_labels, _ = _split_targets(batch, "obb")
        image_shape = tuple(batch["img"].shape[2:])
        rpn_cls, rpn_box, proposals = self._rpn_loss_and_proposals(feats, gt_boxes, image_shape, train=True)
        rois_all, labels_all, reg_targets_all = [], [], []
        for bi, props in enumerate(proposals):
            if self.oriented_proposals:
                props = torch.cat((props, gt_boxes[bi]), dim=0) if gt_boxes[bi].numel() else props
                sampled, labels, boxes, targets, _ = _match_rboxes(props, gt_boxes[bi], gt_labels[bi], self.cfg["train"]["pos_iou"], self.cfg["train"]["neg_iou"], self.cfg["train"]["samples_per_img"], self.cfg["train"]["pos_fraction"])
                rois = torch.cat((boxes.new_full((boxes.shape[0], 1), bi), boxes), dim=1)
                reg_targets = self.roi_coder.encode(boxes[labels > 0], targets[labels > 0]) if (labels > 0).any() else boxes.new_zeros((0, 5))
            else:
                gt_hboxes = _rboxes_to_xyxy(gt_boxes[bi]) if gt_boxes[bi].numel() else gt_boxes[bi].new_zeros((0, 4))
                props = torch.cat((props, gt_hboxes), dim=0) if gt_hboxes.numel() else props
                hprops = props
                sampled, labels, boxes, targets, gt_inds = _match_hboxes(_rboxes_to_xyxy(hprops) if hprops.shape[-1] == 5 else hprops, gt_hboxes, gt_labels[bi], self.cfg["train"]["pos_iou"], self.cfg["train"]["neg_iou"], self.cfg["train"]["samples_per_img"], self.cfg["train"]["pos_fraction"])
                rois = torch.cat((boxes.new_full((boxes.shape[0], 1), bi), boxes), dim=1)
                reg_targets = self.roi_coder.encode(boxes[labels > 0], gt_boxes[bi][gt_inds[labels > 0]]) if (labels > 0).any() else boxes.new_zeros((0, 5))
            rois_all.append(rois)
            labels_all.append(labels)
            reg_targets_all.append(reg_targets)
        rois = torch.cat(rois_all, dim=0) if rois_all else feats[0].new_zeros((0, 6 if self.oriented_proposals else 5))
        labels = torch.cat(labels_all, dim=0) if labels_all else feats[0].new_zeros((0,), dtype=torch.long)
        pooled = self._roi_pool(feats, rois)
        if pooled.shape[0] == 0:
            zero = feats[0].sum() * 0.0
            loss_items = torch.stack((rpn_cls, rpn_box, zero, zero))
            return loss_items.sum(), loss_items.detach()
        cls_logits, box_deltas = self.bbox_head(pooled)
        cls_loss = F.cross_entropy(cls_logits, labels)
        pos = labels > 0
        if pos.any():
            reg_targets = torch.cat(reg_targets_all, dim=0)
            box_loss = F.smooth_l1_loss(box_deltas[pos], reg_targets, beta=self.cfg["bbox_head"]["beta"], reduction="mean")
        else:
            box_loss = cls_loss * 0.0
        loss_items = torch.stack((rpn_cls, rpn_box, cls_loss, box_loss))
        return loss_items.sum(), loss_items.detach()

    @torch.no_grad()
    def forward(self, feats: list[Tensor]) -> list[dict[str, Tensor]]:
        image_shape = self._image_shape_from_feats(feats)
        _, _, proposals = self._rpn_loss_and_proposals(feats, [feats[0].new_zeros((0, 5)) for _ in range(feats[0].shape[0])], image_shape, train=False)
        outputs = []
        for bi, props in enumerate(proposals):
            if props.numel() == 0:
                outputs.append({"bboxes": props.new_zeros((0, 5)), "conf": props.new_zeros((0,)), "cls": props.new_zeros((0,))})
                continue
            rois = torch.cat((props.new_full((props.shape[0], 1), bi), props), dim=1)
            pooled = self._roi_pool(feats, rois)
            cls_logits, box_deltas = self.bbox_head(pooled)
            scores = cls_logits.softmax(dim=-1)[:, 1:]
            boxes = self.roi_coder.decode(props, box_deltas)
            all_boxes = boxes[:, None, :].expand(boxes.shape[0], scores.shape[1], 5).reshape(-1, 5)
            all_scores = scores.reshape(-1)
            all_labels = torch.arange(scores.shape[1], device=boxes.device).repeat(boxes.shape[0])
            keep = all_scores > self.cfg["test"]["score_thresh"]
            all_boxes, all_scores, all_labels = all_boxes[keep], all_scores[keep], all_labels[keep]
            keep_nms = TorchNMS.fast_nms(all_boxes, all_scores, self.cfg["test"]["nms_iou"], iou_func=batch_probiou)[: self.cfg["test"]["max_dets"]]
            outputs.append({"bboxes": all_boxes[keep_nms], "conf": all_scores[keep_nms], "cls": all_labels[keep_nms].float()})
        return outputs


class MaskRCNNHead(_AxisRCNNBase):
    def __init__(self, in_channels: list[int], nc: int, cfg: dict | None = None):
        super().__init__(in_channels, nc, cfg=cfg, with_mask=True, cascade=False)


class CascadeMaskRCNNHead(_AxisRCNNBase):
    def __init__(self, in_channels: list[int], nc: int, cfg: dict | None = None):
        super().__init__(in_channels, nc, cfg=cfg, with_mask=True, cascade=True)


class RotatedFasterRCNNHead(_RotatedRCNNBase):
    def __init__(self, in_channels: list[int], nc: int, cfg: dict | None = None):
        super().__init__(in_channels, nc, cfg=cfg, oriented_proposals=False)


class OrientedRCNNHead(_RotatedRCNNBase):
    def __init__(self, in_channels: list[int], nc: int, cfg: dict | None = None):
        super().__init__(in_channels, nc, cfg=cfg, oriented_proposals=True)
