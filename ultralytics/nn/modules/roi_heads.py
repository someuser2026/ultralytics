# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def encode_boxes(
    proposals: torch.Tensor, gt: torch.Tensor, std: Tuple[float, float, float, float]
) -> torch.Tensor:
    """
    Encode GT boxes relative to proposals.
    
    Args:
        proposals: [N, 4] in xyxy format
        gt: [N, 4] ground truth boxes in xyxy format
        std: Standard deviations for normalization
    
    Returns:
        Encoded deltas [N, 4] as (tx, ty, tw, th)
    """
    wp = proposals[:, 2] - proposals[:, 0]
    hp = proposals[:, 3] - proposals[:, 1]
    xp = proposals[:, 0] + 0.5 * wp
    yp = proposals[:, 1] + 0.5 * hp
    
    wg = gt[:, 2] - gt[:, 0]
    hg = gt[:, 3] - gt[:, 1]
    xg = gt[:, 0] + 0.5 * wg
    yg = gt[:, 1] + 0.5 * hg
    
    dx = (xg - xp) / (wp + 1e-8)
    dy = (yg - yp) / (hp + 1e-8)
    dw = torch.log((wg + 1e-8) / (wp + 1e-8))
    dh = torch.log((hg + 1e-8) / (hp + 1e-8))
    
    dx, dy, dw, dh = [dx / std[0], dy / std[1], dw / std[2], dh / std[3]]
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(
    proposals: torch.Tensor, deltas: torch.Tensor, std: Tuple[float, float, float, float]
) -> torch.Tensor:
    """
    Decode bbox deltas relative to proposals.
    
    Args:
        proposals: [N, 4] in xyxy format
        deltas: [N, 4] as (tx, ty, tw, th)
        std: Standard deviations used in encoding
    
    Returns:
        Decoded boxes [N, 4] in xyxy format
    """
    dx, dy, dw, dh = deltas.unbind(1)
    dx *= std[0]
    dy *= std[1]
    dw *= std[2]
    dh *= std[3]
    
    wp = proposals[:, 2] - proposals[:, 0]
    hp = proposals[:, 3] - proposals[:, 1]
    xp = proposals[:, 0] + 0.5 * wp
    yp = proposals[:, 1] + 0.5 * hp
    
    x = dx * wp + xp
    y = dy * hp + yp
    w = wp * torch.exp(dw.clamp(max=4.135))
    h = hp * torch.exp(dh.clamp(max=4.135))
    
    return torch.stack((x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h), dim=1)


class TwoFCBBoxHead(nn.Module):
    """Standard 2-FC bbox head producing class logits and bbox deltas."""

    def __init__(
        self, in_channels: int, pooler_resolution: int, nc: int, hidden: int = 1024, class_agnostic: bool = False
    ):
        """
        Initialize bbox head.
        
        Args:
            in_channels: Input feature channels
            pooler_resolution: ROI pooling resolution (e.g., 7)
            nc: Number of classes
            hidden: Hidden dimension
            class_agnostic: If True, predict class-agnostic deltas
        """
        super().__init__()
        self.class_agnostic = class_agnostic
        self.fc1 = nn.Linear(in_channels * pooler_resolution * pooler_resolution, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        
        out_reg = 4 if class_agnostic else 4 * nc
        self.cls = nn.Linear(hidden, nc)
        self.reg = nn.Linear(hidden, out_reg)
        
        # Initialize weights
        for m in [self.fc1, self.fc2, self.cls, self.reg]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning cls logits and bbox deltas."""
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return {"cls_logits": self.cls(x), "bbox_deltas": self.reg(x)}


class MaskHead(nn.Module):
    """Mask head producing class-specific mask logits per ROI."""

    def __init__(self, in_channels: int, nc: int, mask_size: int = 28):
        """
        Initialize mask head.
        
        Args:
            in_channels: Input feature channels
            nc: Number of classes
            mask_size: Output mask resolution
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.pred = nn.Conv2d(256, nc, 1)
        self.mask_size = mask_size
        
        # Initialize weights
        for m in [self.conv1, self.conv2, self.deconv, self.pred]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing mask logits.
        
        Args:
            x: ROI features [N, C, res, res]
        
        Returns:
            Mask logits [N, nc, 2*res, 2*res]
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        return self.pred(x)


def roi_align_pyramid(
    feats: List[torch.Tensor],
    boxes: List[torch.Tensor],
    levels: List[int],
    output_size: int,
    sampling_ratio: int,
) -> torch.Tensor:
    """
    Multi-level ROI align across FPN pyramid.
    
    Args:
        feats: List of feature maps per FPN level
        boxes: List of boxes per image [Ni, 4] in xyxy format
        levels: FPN level assignment for each box
        output_size: Pooled feature size
        sampling_ratio: Sampling ratio for ROI align
    
    Returns:
        Pooled features [sum(Ni), C, output_size, output_size]
    """
    assert len(feats) >= 1
    if len(boxes) == 0 or sum([len(b) for b in boxes]) == 0:
        return torch.zeros((0, feats[0].shape[1], output_size, output_size), device=feats[0].device)
    
    device = feats[0].device
    out = []
    lvl_unique = sorted(set(levels))
    
    # Build ROI tensor with batch indices
    rois = []
    for b_ix, b in enumerate(boxes):
        if b.numel():
            rois.append(torch.cat([torch.full((b.shape[0], 1), float(b_ix), device=device), b], dim=1))
    rois = torch.cat(rois, dim=0) if rois else torch.zeros((0, 5), device=device)

    # Process by pyramid level
    levels = torch.as_tensor(levels, device=device, dtype=torch.long)
    for l in lvl_unique:
        idx = torch.nonzero(levels == l, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        rois_l = rois.index_select(0, idx)
        feat_l = feats[l]
        out_l = roi_align(feat_l, rois_l, output_size=output_size, sampling_ratio=sampling_ratio, aligned=True)
        out.append((idx, out_l))
    
    # Reconstruct in original order
    total = rois.shape[0]
    C = feats[0].shape[1]
    pooled = torch.zeros((total, C, output_size, output_size), device=device)
    for idx, part in out:
        pooled[idx] = part
    
    return pooled