# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


@dataclass
class RPNConfig:
    """Configuration for Region Proposal Network."""
    anchor_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512)
    ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
    pre_nms_topk: int = 1000
    post_nms_topk: int = 1000
    nms_thresh: float = 0.7
    min_box_size: float = 4.0
    fg_iou: float = 0.7
    bg_iou: float = 0.3
    samples_per_img: int = 256
    fg_fraction: float = 0.5


class AnchorGenerator(nn.Module):
    """Generate anchors per FPN level."""

    def __init__(self, sizes: Tuple[int, ...], ratios: Tuple[float, ...], strides: List[int]):
        """
        Initialize anchor generator.
        
        Args:
            sizes: Anchor sizes (one per FPN level)
            ratios: Aspect ratios for anchors
            strides: Feature map strides [8, 16, 32, 64, 128]
        """
        super().__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.strides = strides

    @torch.no_grad()
    def grid_anchors(self, feat: torch.Tensor, size: int, stride: int, device) -> torch.Tensor:
        """Generate anchors for one feature level."""
        h, w = feat.shape[-2:]
        shifts_x = (torch.arange(w, device=device) + 0.5) * stride
        shifts_y = (torch.arange(h, device=device) + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        
        # Base anchors for all ratios
        base = []
        for r in self.ratios:
            ar = math.sqrt(r)
            ws = size * ar
            hs = size / ar
            base.append(torch.tensor([[-ws / 2, -hs / 2, ws / 2, hs / 2]], device=device))
        base = torch.cat(base, dim=0)  # [R, 4]
        
        # Tile to grid
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1)  # [H, W, 4]
        anchors = base[None, None, :, :] + shifts[:, :, None, :]  # [H, W, R, 4]
        return anchors.reshape(-1, 4)

    @torch.no_grad()
    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Generate anchors for all FPN levels."""
        anchors = []
        device = feats[0].device
        for lvl, feat in enumerate(feats):
            size = self.sizes[min(lvl, len(self.sizes) - 1)]
            stride = self.strides[min(lvl, len(self.strides) - 1)]
            a = self.grid_anchors(feat, size=size, stride=stride, device=device)
            anchors.append(a)  # [Hi*Wi*R, 4] absolute xyxy
        return anchors


class RPNHead(nn.Module):
    """RPN head with 3x3 conv producing objectness and bbox deltas."""

    def __init__(self, in_channels: int, num_anchors: int):
        """
        Initialize RPN head.
        
        Args:
            in_channels: Input feature channels
            num_anchors: Number of anchors per location
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.obj = nn.Conv2d(in_channels, num_anchors, 1)
        self.reg = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        # Initialize weights
        for m in [self.conv, self.obj, self.reg]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass returning objectness logits and bbox deltas."""
        logits, bbox_deltas = [], []
        for x in feats:
            t = F.relu(self.conv(x))
            logits.append(self.obj(t))
            bbox_deltas.append(self.reg(t))
        return logits, bbox_deltas


def _apply_deltas_to_anchors(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Decode bbox deltas relative to anchors.
    
    Args:
        deltas: [N, 4] (tx, ty, tw, th)
        anchors: [N, 4] xyxy format
    
    Returns:
        Decoded boxes [N, 4] in xyxy format
    """
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha

    dx, dy, dw, dh = deltas.unbind(dim=1)
    # Prevent blow-ups
    dw = torch.clamp(dw, max=4.135)  # exp(4.135) ≈ 62
    dh = torch.clamp(dh, max=4.135)
    
    x = dx * wa + xa
    y = dy * ha + ya
    w = wa * torch.exp(dw)
    h = ha * torch.exp(dh)
    
    return torch.stack((x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h), dim=1)


@torch.no_grad()
def rpn_inference_single_image(
    logits_per_level: List[torch.Tensor],
    deltas_per_level: List[torch.Tensor],
    anchors_per_level: List[torch.Tensor],
    image_size: Tuple[int, int],
    cfg: RPNConfig,
) -> torch.Tensor:
    """
    Generate proposals for one image.
    
    Args:
        logits_per_level: List of objectness logits per level
        deltas_per_level: List of bbox deltas per level
        anchors_per_level: List of anchors per level
        image_size: (H, W) of image
        cfg: RPN configuration
    
    Returns:
        Proposals [N, 5] with [x1, y1, x2, y2, score]
    """
    device = logits_per_level[0].device
    H, W = image_size
    props = []
    
    for cls, reg, anchors in zip(logits_per_level, deltas_per_level, anchors_per_level):
        scores = cls.sigmoid().flatten()
        deltas = reg.permute(1, 2, 0).reshape(-1, 4)

        # Top-k before NMS
        num_pre = min(cfg.pre_nms_topk, scores.numel())
        topk = scores.topk(num_pre).indices
        scores = scores[topk]
        anchors = anchors[topk]
        deltas = deltas[topk]

        # Decode boxes
        boxes = _apply_deltas_to_anchors(deltas, anchors)
        
        # Clip to image
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=W - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=H - 1)
        
        # Filter by size
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        keep = (ws >= cfg.min_box_size) & (hs >= cfg.min_box_size)
        boxes, scores = boxes[keep], scores[keep]
        
        # Per-level NMS
        keep_idx = nms(boxes, scores, cfg.nms_thresh)
        keep_idx = keep_idx[: cfg.post_nms_topk]
        props.append(torch.cat([boxes[keep_idx], scores[keep_idx, None]], dim=1))
    
    if len(props) == 0:
        return torch.zeros((0, 5), device=device)
    
    props = torch.cat(props, dim=0)
    
    # Cross-level NMS
    keep = nms(props[:, :4], props[:, 4], cfg.nms_thresh)
    return props[keep]