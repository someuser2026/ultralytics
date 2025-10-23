# Ultralytics 🚀 AGPL-3.0
from __future__ import annotations
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.transformer import (
    MSDeformAttn, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, MLP, LayerNorm2d
)

__all__ = ["Mask2FormerHead"]

def _conv_bn_act(c1, c2, k=1, s=1, p=None, g=1, bias=False):
    if p is None:
        p = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, p, groups=g, bias=bias),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True)
    )

class SimplePixelDecoder(nn.Module):
    """
    Lightweight pixel decoder producing a stride-`mask_stride` mask feature map.
    It projects each input level to `mask_dim`, upsamples to the finest level, and fuses by sum+conv.
    """
    def __init__(self, in_channels: List[int], mask_dim: int = 256, mask_stride: int = 4):
        super().__init__()
        self.mask_dim = mask_dim
        self.mask_stride = mask_stride
        self.proj = nn.ModuleList([_conv_bn_act(c, mask_dim, k=1) for c in in_channels])
        self.fuse = _conv_bn_act(mask_dim, mask_dim, k=3)
        self.out_norm = LayerNorm2d(mask_dim)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # feats expected from low->high stride order OR arbitrary; we align by spatial size
        # We upsample all to the smallest stride (largest HxW)
        sizes = [f.shape[-2:] for f in feats]
        target_size = max(sizes, key=lambda x: x[0]*x[1])  # largest spatial resolution
        proj = [self.proj[i](f) for i, f in enumerate(feats)]
        up  = [F.interpolate(p, size=target_size, mode="bilinear", align_corners=False) for p in proj]
        x = sum(up)
        x = self.fuse(x)
        x = self.out_norm(x)
        return x, proj  # mask_features (stride ~ of finest input), per-level projected features
        

class Mask2FormerHead(nn.Module):
    """
    Full Mask2Former head using Ultralytics' MSDeformAttn decoder.
    Training:
      returns dict with 'pred_logits', 'pred_masks', optionally 'aux'
    Inference:
      returns [raw_det_tensor, proto_like] to stay compatible with SegmentationValidator
    """
    export = False  # keep parity with other heads
    legacy = False

    def __init__(self,
                 nc: int,
                 num_queries: int = 100,
                 hidden_dim: int = 256,
                 nheads: int = 8,
                 nlevels: int = 3,
                 mask_dim: int = 256,
                 aux_loss: bool = True,
                 mask_stride: int = 4,
                 in_channels: List[int] | None = None  # appended by parse_model()
                 ):
        super().__init__()
        assert in_channels is not None and len(in_channels) == nlevels, "in_channels must be provided by parse_model()"
        self.nc = int(nc)
        self.num_queries = int(num_queries)
        self.hidden_dim = int(hidden_dim)
        self.nheads = int(nheads)
        self.nlevels = int(nlevels)
        self.mask_dim = int(mask_dim)
        self.aux_loss = bool(aux_loss)
        self.mask_stride = int(mask_stride)

        # Pixel decoder: produce per-pixel embeddings for mask projection
        self.pixel_decoder = SimplePixelDecoder(in_channels, mask_dim=self.mask_dim, mask_stride=self.mask_stride)

        # Input projections to hidden_dim for transformer memory per level
        self.input_proj = nn.ModuleList([_conv_bn_act(c, hidden_dim, k=1) for c in in_channels])

        # Learnable queries and decoder
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, n_levels=nlevels)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_layers=6, return_intermediate=self.aux_loss)

        # Per-query heads
        self.class_head = nn.Linear(hidden_dim, self.nc)
        self.mask_embed = MLP(hidden_dim, hidden_dim, self.mask_dim, 3)

        # Infer-time buffer
        self.stride = torch.tensor([mask_stride], dtype=torch.int)

    # ----- helper utils -----
    @staticmethod
    def _get_spatial_shapes(x_list: List[torch.Tensor]) -> torch.Tensor:
        # shapes: (n_levels, 2) as (H, W)
        return torch.as_tensor([x.shape[-2:] for x in x_list], dtype=torch.long, device=x_list[0].device)

    @staticmethod
    def _flatten_ms_feats(x_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # flatten per level to (B, H*W, C) and concatenate along HW
        bs = x_list[0].shape[0]
        outs, lvl_pos = [], []
        for x in x_list:
            B, C, H, W = x.shape
            outs.append(x.flatten(2).transpose(1, 2))  # (B, HW, C)
        memory = torch.cat(outs, dim=1)  # (B, sum(HW), C)
        return memory

    # ----- forward -----
    def forward(self, x: List[torch.Tensor]) -> Any:
        """
        x: list of multi-scale features from neck, typical P3..Pk, shapes [B, C_i, H_i, W_i]
        """
        # (1) Pixel decoder → mask features
        mask_features, per_level_proj = self.pixel_decoder(x)  # mask_features: [B, mask_dim, Hm, Wm]

        # (2) Prepare multi-scale memory for deformable decoder
        mem_levels = [self.input_proj[i](f) for i, f in enumerate(x)]          # -> [B, hidden_dim, Hi, Wi]
        spatial_shapes = self._get_spatial_shapes(mem_levels)                   # (L, 2)
        # flatten and concat along HW
        memory = torch.cat([m.flatten(2).transpose(1, 2) for m in mem_levels], dim=1)  # (B, sum(HW), hidden_dim)
        level_start_index = torch.as_tensor(
            [0] + [int(spatial_shapes[:i, 0].mul(spatial_shapes[:i, 1]).sum()) for i in range(1, len(mem_levels) + 1)],
            device=memory.device, dtype=torch.long
        )

        # (3) Queries
        bs = x[0].shape[0]
        tgt = torch.zeros(self.num_queries, bs, self.hidden_dim, device=memory.device)  # (Q, B, C)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)               # (Q, B, C)

        # (4) Decode
        hs, _ = self.decoder(tgt, memory, spatial_shapes, level_start_index, query_pos=query_pos)  # hs: (L, B, Q, C) or (B,Q,C)
        if isinstance(hs, list) or isinstance(hs, tuple):
            dec_out = hs  # list of (B, Q, C) per layer
        else:
            dec_out = [hs]  # final only

        # (5) Heads per layer (aux) or final
        def layer_heads(h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # h: (B, Q, C)
            class_logits = self.class_head(h)                     # (B, Q, nc)
            mask_embed   = self.mask_embed(h)                     # (B, Q, mask_dim)
            # mask logits via per-query projection onto pixel features
            # mask_features: (B, mask_dim, Hm, Wm) → logits: (B, Q, Hm, Wm)
            mask_logits  = torch.einsum("bqc, bchw -> bqhw", mask_embed, mask_features)
            return class_logits, mask_logits

        if self.training or self.aux_loss:
            outs = [layer_heads(h) for h in dec_out]              # list of (cls, mask)
            pred_logits, pred_masks = outs[-1]
            out: Dict[str, Any] = {"pred_logits": pred_logits, "pred_masks": pred_masks}
            if self.aux_loss and len(outs) > 1:
                out["aux"] = [{"pred_logits": c, "pred_masks": m} for (c, m) in outs[:-1]]
            return out

        # (6) Inference (validator compatibility):
        # Build YOLO-like det tensor (B, Q, 4 + 1 + nc + Ncoeff) and a proto-like tensor per image
        # a) get probabilities
        pred_logits, pred_masks = layer_heads(dec_out[-1])
        prob = pred_logits.sigmoid()                               # (B, Q, nc)
        conf, cls = prob.max(dim=-1)                               # (B, Q), (B, Q)

        # b) derive boxes from masks (tight xyxy on mask grid)
        B, Q, Hm, Wm = pred_masks.shape
        masks_bin = (pred_masks.sigmoid() > 0.5).float()
        # Compute boxes per query per image
        # (simple but robust; you may vectorize further if needed)
        boxes = pred_masks.new_zeros((B, Q, 4))
        for b in range(B):
            for q in range(Q):
                m = masks_bin[b, q]
                if m.any():
                    ys, xs = torch.where(m > 0)
                    y1, y2 = ys.min().float(), ys.max().float()
                    x1, x2 = xs.min().float(), xs.max().float()
                    boxes[b, q] = torch.tensor([x1, y1, x2, y2], device=m.device)
                else:
                    boxes[b, q] = 0

        # c) “proto-like” trick: stack instance mask logits as proto channels
        #    and emit one-hot coefficients (Q-dimensional) as 'extra' columns.
        #    The segmentation validator will reconstruct the same masks via process_mask[_native].
        dets = []
        protos = []
        for b in range(B):
            # proto: (C=Ninst, Hm, Wm) — keep all queries; NMS will filter later
            proto_b = pred_masks[b]                                # (Q, Hm, Wm)
            protos.append(proto_b)
            # coefficients: identity one-hot (Q, Q)
            coeff = torch.eye(Q, device=proto_b.device, dtype=proto_b.dtype)
            # YOLO format rows: [x1 y1 x2 y2 obj cls_logit...] before NMS; here obj=1 and we pass class probs
            obj = torch.ones((Q, 1), device=proto_b.device, dtype=proto_b.dtype)
            # Assemble per-query per-class logits -> NMS expects (B, Q, 4+1+nc+extra)
            det_b = torch.cat([boxes[b], obj, prob[b], coeff], dim=-1)  # (Q, 4+1+nc+Q)
            dets.append(det_b.unsqueeze(0))

        raw_det = torch.cat(dets, dim=0)    # (B, Q, 4+1+nc+Q)
        # The segmentation validator expects preds to be [raw_det, proto]
        # and will run NMS on raw_det then call process_mask(proto[i], coeff, bboxes, ...).
        return [raw_det, torch.stack(protos, dim=0)]
