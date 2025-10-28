"""
Region Proposal Network for Cascade R-CNN
Path: ultralytics/nn/modules/rpn.py
"""

from __future__ import annotations
from typing import List, Tuple
import torch
from torch import Tensor, nn
from torchvision.ops import nms


class AnchorGenerator(nn.Module):
    """Generate anchors on multiple FPN levels."""
    
    def __init__(
        self,
        strides: Tuple[int, ...] = (4, 8, 16, 32),
        scales: Tuple[int, ...] = (32, 64, 128, 256),
        ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
    ):
        super().__init__()
        self.strides = strides
        self.scales = scales
        self.ratios = ratios
        self.num_anchors = len(ratios)
        
        # Cache for anchors
        self._anchor_cache = {}

    def _generate_base_anchors(self, scale: float, ratios: List[float], device) -> Tensor:
        """Generate base anchors for a single scale and multiple ratios."""
        ratios = torch.tensor(ratios, device=device)
        
        # Compute anchor dimensions
        ws = scale * torch.sqrt(1.0 / ratios)
        hs = scale * torch.sqrt(ratios)
        
        # Create anchors centered at (0, 0)
        anchors = torch.stack([
            -0.5 * ws,
            -0.5 * hs,
            0.5 * ws,
            0.5 * hs
        ], dim=1)
        
        return anchors

    @torch.no_grad()
    def grid_anchors(
        self,
        feat_shapes: List[Tuple[int, int]],
        device: torch.device
    ) -> List[Tensor]:
        """
        Generate anchors on grid for all FPN levels.
        
        Args:
            feat_shapes: List of (H, W) for each feature level
            device: device to create anchors on
        
        Returns:
            anchors_per_level: List of anchor tensors, each (num_anchors, 4)
        """
        anchors_per_level = []
        
        for (H, W), stride, scale in zip(feat_shapes, self.strides, self.scales):
            # Use cache if available
            cache_key = (H, W, stride, scale, device)
            if cache_key in self._anchor_cache:
                anchors_per_level.append(self._anchor_cache[cache_key])
                continue
            
            # Generate base anchors
            base_anchors = self._generate_base_anchors(scale, self.ratios, device)
            num_base = base_anchors.shape[0]
            
            # Create grid
            shifts_x = torch.arange(0, W, device=device, dtype=torch.float32) * stride
            shifts_y = torch.arange(0, H, device=device, dtype=torch.float32) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            
            shifts = torch.stack([
                shift_x.reshape(-1),
                shift_y.reshape(-1),
                shift_x.reshape(-1),
                shift_y.reshape(-1)
            ], dim=1)
            
            # Broadcast anchors to all grid locations
            anchors = base_anchors.view(1, num_base, 4) + shifts.view(-1, 1, 4)
            anchors = anchors.reshape(-1, 4)
            
            # Cache and append
            self._anchor_cache[cache_key] = anchors
            anchors_per_level.append(anchors)
        
        return anchors_per_level


class RPNHead(nn.Module):
    """RPN head with objectness and bbox regression."""
    
    def __init__(self, in_channels: int, num_anchors: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.obj_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_deltas = nn.Conv2d(in_channels, num_anchors * 4, 1)
        
        # Initialize weights
        for layer in [self.conv, self.obj_logits, self.bbox_deltas]:
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, feats: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Forward pass through RPN head.
        
        Args:
            feats: List of feature maps [P2, P3, P4, P5]
        
        Returns:
            obj_logits: List of objectness logits per level
            bbox_deltas: List of bbox regression deltas per level
        """
        obj_logits_list = []
        bbox_deltas_list = []
        
        for feat in feats:
            t = torch.relu(self.conv(feat))
            obj_logits_list.append(self.obj_logits(t))
            bbox_deltas_list.append(self.bbox_deltas(t))
        
        return obj_logits_list, bbox_deltas_list


@torch.no_grad()
def generate_proposals(
    anchors_per_level: List[Tensor],
    obj_logits_per_level: List[Tensor],
    bbox_deltas_per_level: List[Tensor],
    image_shape: Tuple[int, int],
    pre_nms_topk: int = 2000,
    post_nms_topk: int = 1000,
    nms_iou: float = 0.7,
    min_box_size: float = 0.0
) -> List[Tensor]:
    """
    Generate proposals from RPN outputs.
    
    Args:
        anchors_per_level: List of anchor tensors per FPN level
        obj_logits_per_level: List of objectness logits per level
        bbox_deltas_per_level: List of bbox deltas per level
        image_shape: (H, W) of input image
        pre_nms_topk: number of top proposals before NMS
        post_nms_topk: number of top proposals after NMS
        nms_iou: IoU threshold for NMS
        min_box_size: minimum box size to keep
    
    Returns:
        proposals_per_image: List of proposal tensors, one per image in batch
    """
    batch_size = obj_logits_per_level[0].shape[0]
    H_img, W_img = image_shape
    device = obj_logits_per_level[0].device
    
    proposals_per_image = []
    
    for batch_idx in range(batch_size):
        all_boxes = []
        all_scores = []
        
        for anchors, obj_logits, bbox_deltas in zip(
            anchors_per_level,
            obj_logits_per_level,
            bbox_deltas_per_level
        ):
            # Get batch element
            obj = obj_logits[batch_idx]  # (A, H, W)
            deltas = bbox_deltas[batch_idx]  # (4*A, H, W)
            
            A = obj.shape[0]
            H, W = obj.shape[1], obj.shape[2]
            
            # Reshape to (H*W*A,)
            obj_flat = obj.permute(1, 2, 0).reshape(-1).sigmoid()
            
            # Reshape to (H*W*A, 4)
            deltas_flat = deltas.permute(1, 2, 0).reshape(-1, 4)
            
            # Select top-k before NMS
            num_topk = min(pre_nms_topk, obj_flat.numel())
            topk_idx = torch.topk(obj_flat, k=num_topk).indices
            
            scores = obj_flat[topk_idx]
            deltas_topk = deltas_flat[topk_idx]
            anchors_topk = anchors[topk_idx]
            
            # Decode boxes
            boxes = decode_boxes(anchors_topk, deltas_topk)
            
            # Clip to image
            boxes[:, 0::2].clamp_(0, W_img)
            boxes[:, 1::2].clamp_(0, H_img)
            
            # Filter small boxes
            if min_box_size > 0:
                ws = boxes[:, 2] - boxes[:, 0]
                hs = boxes[:, 3] - boxes[:, 1]
                keep = (ws >= min_box_size) & (hs >= min_box_size)
                boxes = boxes[keep]
                scores = scores[keep]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
        
        if not all_boxes:
            proposals_per_image.append(torch.zeros((0, 4), device=device))
            continue
        
        # Concatenate all levels
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        if all_boxes.numel() == 0:
            proposals_per_image.append(torch.zeros((0, 4), device=device))
            continue
        
        # Apply NMS
        keep = nms(all_boxes, all_scores, nms_iou)
        keep = keep[:post_nms_topk]
        
        proposals_per_image.append(all_boxes[keep])
    
    return proposals_per_image


def decode_boxes(anchors: Tensor, deltas: Tensor) -> Tensor:
    """
    Decode bbox deltas to boxes.
    
    Args:
        anchors: (N, 4) anchor boxes in xyxy format
        deltas: (N, 4) regression deltas
    
    Returns:
        boxes: (N, 4) predicted boxes in xyxy format
    """
    wa = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-6)
    ha = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-6)
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha
    
    # Clamp deltas for stability
    dx = deltas[:, 0].clamp(min=-1000, max=1000)
    dy = deltas[:, 1].clamp(min=-1000, max=1000)
    dw = deltas[:, 2].clamp(min=-1000, max=1000)
    dh = deltas[:, 3].clamp(min=-1000, max=1000)
    
    x = dx * wa + xa
    y = dy * ha + ya
    w = wa * torch.exp(dw)
    h = ha * torch.exp(dh)
    
    x1 = x - 0.5 * w
    y1 = y - 0.5 * h
    x2 = x + 0.5 * w
    y2 = y + 0.5 * h
    
    return torch.stack([x1, y1, x2, y2], dim=1)