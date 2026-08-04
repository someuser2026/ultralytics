# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RHINO-only transformer, rotated attention, and denoising components."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .head import RTDETRDecoder
from .transformer import DeformableTransformerDecoder, DeformableTransformerDecoderLayer, MLP, MSDeformAttn
from .utils import bias_init_with_prob, inverse_sigmoid, multi_scale_deformable_attn_pytorch


def _coordinate_to_encoding(
    coordinates: torch.Tensor,
    num_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
) -> torch.Tensor:
    """Encode normalized ``cxcywh`` coordinates as the DINO v2 sine query position."""
    if coordinates.shape[-1] != 4:
        raise ValueError(f"RHINO v2 query coordinates must be cxcywh, got shape {tuple(coordinates.shape)}.")
    dim_t = torch.arange(num_feats, dtype=coordinates.dtype, device=coordinates.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_feats)
    encoded = []
    # Reference DINO/RHINO ordering is y, x, w, h.
    for index in (1, 0, 2, 3):
        value = coordinates[..., index, None] * scale / dim_t
        value = torch.stack((value[..., 0::2].sin(), value[..., 1::2].cos()), dim=-1).flatten(-2)
        encoded.append(value)
    return torch.cat(encoded, dim=-1)


class _RhinoChannelMapper(nn.Module):
    """Reference RHINO ChannelMapper, private to ``RHINOOBBDecoder``."""

    def __init__(self, in_channels: tuple[int, ...], out_channels: int = 256):
        super().__init__()
        if len(in_channels) not in {3, 4}:
            raise ValueError(f"RHINO expects three or four input features, got {len(in_channels)}.")
        if out_channels % 32:
            raise ValueError(f"RHINO ChannelMapper output channels must be divisible by 32, got {out_channels}.")
        self.in_channels = tuple(int(channel) for channel in in_channels)
        self.out_channels = int(out_channels)
        self.input_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channel, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, out_channels),
            )
            for channel in self.in_channels
        )
        self.extra_conv = (
            nn.Sequential(
                nn.Conv2d(self.in_channels[-1], out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
            )
            if len(self.in_channels) == 3
            else None
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Map three/four raw backbone features to exactly four 256-channel levels."""
        if len(features) != len(self.input_convs):
            raise ValueError(f"RHINO received {len(features)} features but was built for {len(self.input_convs)}.")
        mapped = [projection(feature) for projection, feature in zip(self.input_convs, features)]
        if self.extra_conv is not None:
            mapped.append(self.extra_conv(features[-1]))
        if len(mapped) != 4:
            raise RuntimeError(f"RHINO ChannelMapper must emit four levels, emitted {len(mapped)}.")
        return mapped


class _RhinoSinePositionEncoding(nn.Module):
    """Reference normalized sine position encoding (temperature 20, offset 0)."""

    def __init__(
        self,
        num_feats: int = 128,
        temperature: int = 20,
        normalize: bool = True,
        offset: float = 0.0,
        scale: float = 2 * math.pi,
    ):
        super().__init__()
        self.num_feats = int(num_feats)
        self.temperature = int(temperature)
        self.normalize = bool(normalize)
        self.offset = float(offset)
        self.scale = float(scale)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Return ``[B, 2*num_feats, H, W]`` positional features for a padding mask."""
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.offset + eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.offset + eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        return torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)


class _RhinoEncoderLayer(nn.Module):
    """One RHINO deformable encoder layer."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(hidden_dim, num_levels, num_heads, num_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        source: torch.Tensor,
        position: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: list[list[int]],
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode flattened multiscale source features."""
        attended = self.self_attn(source + position, reference_points, source, spatial_shapes, padding_mask)
        source = self.norm1(source + self.dropout1(attended))
        ffn = self.linear2(self.dropout2(self.activation(self.linear1(source))))
        return self.norm2(source + self.dropout3(ffn))


class _RhinoDeformableEncoder(nn.Module):
    """Six-layer, four-scale RHINO deformable encoder."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_levels != 4:
            raise ValueError(f"RHINO encoder requires four feature levels, got {num_levels}.")
        self.num_layers = int(num_layers)
        self.num_levels = int(num_levels)
        self.layers = nn.ModuleList(
            [
                _RhinoEncoderLayer(hidden_dim, num_heads, num_levels, num_points, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.level_embeddings = nn.Parameter(torch.empty(num_levels, hidden_dim))
        nn.init.normal_(self.level_embeddings)
        self.spatial_shapes: torch.Tensor | None = None
        self.level_start_index: torch.Tensor | None = None
        self.valid_ratios: torch.Tensor | None = None

    @staticmethod
    def _valid_ratio(mask: torch.Tensor) -> torch.Tensor:
        """Compute valid width/height ratios for a feature mask."""
        height, width = mask.shape[-2:]
        valid_height = (~mask[:, :, 0]).sum(1)
        valid_width = (~mask[:, 0, :]).sum(1)
        return torch.stack((valid_width.float() / width, valid_height.float() / height), dim=-1)

    @staticmethod
    def _reference_points(
        spatial_shapes: list[list[int]], valid_ratios: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build Deformable-DETR encoder reference points."""
        references = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            references.append(torch.stack((ref_x, ref_y), dim=-1))
        reference_points = torch.cat(references, dim=1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        sources: list[torch.Tensor],
        masks: list[torch.Tensor],
        positions: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[list[int]], torch.Tensor, torch.Tensor]:
        """Flatten and encode four mapped feature levels."""
        source_flatten, mask_flatten, position_flatten, spatial_shapes = [], [], [], []
        for level, (source, mask, position) in enumerate(zip(sources, masks, positions)):
            batch_size, _, height, width = source.shape
            spatial_shapes.append([height, width])
            source_flatten.append(source.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            position = position.to(dtype=source.dtype).flatten(2).transpose(1, 2)
            position_flatten.append(position + self.level_embeddings[level].view(1, 1, -1))

        source = torch.cat(source_flatten, dim=1)
        padding_mask = torch.cat(mask_flatten, dim=1)
        position = torch.cat(position_flatten, dim=1)
        valid_ratios = torch.stack([self._valid_ratio(mask) for mask in masks], dim=1).to(source.dtype)
        reference_points = self._reference_points(spatial_shapes, valid_ratios, source.device, source.dtype)

        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source.device)
        level_start_index = torch.cat(
            (spatial_shapes_tensor.new_zeros(1), spatial_shapes_tensor.prod(1).cumsum(0)[:-1])
        )
        self.spatial_shapes = spatial_shapes_tensor
        self.level_start_index = level_start_index
        self.valid_ratios = valid_ratios

        for layer in self.layers:
            source = layer(source, position, reference_points, spatial_shapes, padding_mask)
        return source, spatial_shapes, padding_mask, valid_ratios


class RhinoMSDeformAttn(MSDeformAttn):
    """RHINO rotated deformable attention using ``angle/pi`` references."""

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply rotated multiscale sampling, converting the RHINO angle to radians."""
        batch_size, query_count = query.shape[:2]
        value_count = value.shape[1]
        if sum(shape[0] * shape[1] for shape in value_shapes) != value_count:
            raise ValueError("RHINO value shapes do not match flattened encoder memory.")

        if refer_bbox.ndim == 3:
            refer_bbox = refer_bbox.unsqueeze(2)
        if refer_bbox.shape[2] == 1:
            refer_bbox = refer_bbox.expand(-1, -1, self.n_levels, -1)
        elif refer_bbox.shape[2] != self.n_levels:
            raise ValueError(
                f"Expected reference boxes to have 1 or {self.n_levels} levels, got {refer_bbox.shape[2]}."
            )

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], 0.0)
        value = value.view(batch_size, value_count, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, query_count, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, query_count, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            batch_size, query_count, self.n_heads, self.n_levels, self.n_points
        )

        reference_dim = refer_bbox.shape[-1]
        if reference_dim == 2:
            normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            sampling_locations = (
                refer_bbox[:, :, None, :, None]
                + sampling_offsets / normalizer[None, None, None, :, None]
            )
        elif reference_dim == 4:
            sampling_locations = (
                refer_bbox[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            )
        elif reference_dim == 5:
            theta = refer_bbox[..., 4:] * math.pi
            cosine, sine = theta.cos(), theta.sin()
            rotation = torch.cat((cosine, -sine, sine, cosine), dim=-1).view(
                batch_size, query_count, self.n_levels, 2, 2
            )
            half_wh = refer_bbox[..., 2:4] * 0.5
            rotated_points = torch.einsum("bqlij,bqlj->bqli", rotation, half_wh)
            sampling_locations = (
                refer_bbox[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * rotated_points[:, :, None, :, None]
            )
        else:
            raise ValueError(f"RHINO reference boxes must have 2, 4, or 5 values, got {reference_dim}.")

        # Kept detached for focused RHINO parity diagnostics without retaining the training graph.
        self.last_sampling_locations = sampling_locations.detach()
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class RhinoTransformerDecoderLayer(DeformableTransformerDecoderLayer):
    """RHINO v2 decoder layer with rotated cross-attention."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 2048,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points, use_obb=False)
        self.cross_attn = RhinoMSDeformAttn(d_model, n_levels, n_heads, n_points)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode one layer using already-expanded per-level RHINO references."""
        query = key = self.with_pos_embed(embed, query_pos)
        attended = self.self_attn(
            query.transpose(0, 1),
            key.transpose(0, 1),
            embed.transpose(0, 1),
            attn_mask=attn_mask,
        )[0].transpose(0, 1)
        embed = self.norm1(embed + self.dropout1(attended))

        attended = self.cross_attn(
            self.with_pos_embed(embed, query_pos),
            refer_bbox,
            feats,
            shapes,
            padding_mask,
        )
        embed = self.norm2(embed + self.dropout2(attended))
        return self.forward_ffn(embed)


class RhinoTransformerDecoder(DeformableTransformerDecoder):
    """RHINO v2 decoder with per-layer normalization and Look Forward Twice."""

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        super().__init__(hidden_dim, decoder_layer, num_layers, eval_idx)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_references: list[torch.Tensor] = []
        self.attention_reference_inputs: list[torch.Tensor] = []

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        valid_ratios: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode with detached attention references and differentiable regression references."""
        output = embed
        decoder_boxes, decoder_scores = [], []
        attention_reference = refer_bbox.sigmoid().detach()
        regression_reference = attention_reference
        self.attention_references = []
        self.attention_reference_inputs = []
        if valid_ratios is None:
            valid_ratios = attention_reference.new_ones(
                (attention_reference.shape[0], len(shapes), 2)
            )
        if valid_ratios.shape != (attention_reference.shape[0], len(shapes), 2):
            raise ValueError(
                "RHINO valid_ratios must have shape "
                f"{(attention_reference.shape[0], len(shapes), 2)}, got {tuple(valid_ratios.shape)}."
            )
        reference_scale = torch.cat(
            (valid_ratios, valid_ratios, torch.ones_like(valid_ratios[..., :1])),
            dim=-1,
        )

        for index, layer in enumerate(self.layers):
            self.attention_references.append(attention_reference)
            attention_reference_input = attention_reference[:, :, None] * reference_scale[:, None]
            self.attention_reference_inputs.append(attention_reference_input.detach())
            query_sine = _coordinate_to_encoding(
                attention_reference_input[:, :, 0, :4], num_feats=self.hidden_dim // 2
            )
            output = layer(
                output,
                attention_reference_input,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(query_sine),
            )
            refinement_delta = bbox_head[index](output)
            next_reference = torch.sigmoid(
                refinement_delta + inverse_sigmoid(attention_reference)
            )

            normalized_output = self.norm(output)
            prediction_delta = bbox_head[index](normalized_output)
            reported_box = torch.sigmoid(
                prediction_delta + inverse_sigmoid(regression_reference)
            )
            if self.training:
                decoder_scores.append(score_head[index](normalized_output))
                decoder_boxes.append(reported_box)
            elif index == self.eval_idx:
                decoder_scores.append(score_head[index](normalized_output))
                decoder_boxes.append(reported_box)
                break

            # RHINO Look Forward Twice: raw-query refinement is detached only for subsequent attention.
            regression_reference = next_reference
            attention_reference = next_reference.detach()

        return torch.stack(decoder_boxes), torch.stack(decoder_scores)


class RotatedCdnQueryGenerator:
    """RHINO CDN generator operating on normalized ``cxcywh, angle/pi`` targets."""

    def __init__(
        self,
        num_classes: int,
        embed_dims: int,
        num_matching_queries: int,
        label_embedding: nn.Embedding,
        num_denoising_queries: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        group_mode: str = "dynamic",
        num_groups: int | None = None,
        max_num_groups: int | None = None,
    ):
        self.num_classes = int(num_classes)
        self.embed_dims = int(embed_dims)
        self.num_matching_queries = int(num_matching_queries)
        self.label_embedding = label_embedding
        self.num_denoising_queries = int(num_denoising_queries)
        self.label_noise_ratio = float(label_noise_ratio)
        self.box_noise_scale = float(box_noise_scale)
        self.group_mode = str(group_mode)
        self.num_groups = None if num_groups is None else int(num_groups)
        self.max_num_groups = None if max_num_groups is None else int(max_num_groups)

    def __call__(
        self, batch: dict[str, Any] | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]] | tuple[None, None, None, None]:
        """Generate CDN queries and a valid empty structure for all-empty batches."""
        if batch is None or self.num_denoising_queries <= 0:
            return None, None, None, None

        gt_groups = list(batch.get("gt_groups", []))
        batch_size = len(gt_groups)
        device = self.label_embedding.weight.device
        if not gt_groups or max(gt_groups, default=0) <= 0:
            label_query = self.label_embedding.weight.new_zeros((batch_size, 0, self.embed_dims))
            bbox_query = self.label_embedding.weight.new_zeros((batch_size, 0, 5))
            attention_mask = torch.zeros(
                (self.num_matching_queries, self.num_matching_queries), dtype=torch.bool, device=device
            )
            return label_query, bbox_query, attention_mask, {
                "num_denoising_queries": 0,
                "num_denoising_groups": 0,
                "dn_num_split": [0, self.num_matching_queries],
            }

        gt_labels = batch["cls"].to(device=device).view(-1).long()
        gt_bboxes = batch["bboxes"].to(device=device)
        batch_idx = batch["batch_idx"].to(device=device).view(-1).long()
        max_num_target = int(max(gt_groups))
        num_groups = self.get_num_groups(max_num_target)

        label_query = self.generate_dn_label_query(gt_labels, num_groups)
        bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)
        label_query, bbox_query = self.collate_dn_queries(
            label_query, bbox_query, batch_idx, batch_size, gt_groups, num_groups
        )
        attention_mask = self.generate_dn_mask(max_num_target, num_groups, label_query.device)
        num_dn = int(max_num_target * 2 * num_groups)
        return label_query, bbox_query, attention_mask, {
            "num_denoising_queries": num_dn,
            "num_denoising_groups": num_groups,
            "dn_num_split": [num_dn, self.num_matching_queries],
        }

    def get_num_groups(self, max_num_target: int) -> int:
        """Return static/dynamic group count with the RHINO cap."""
        if self.group_mode == "static":
            num_groups = self.num_groups if self.num_groups is not None else 1
        else:
            num_groups = self.num_denoising_queries // max(max_num_target, 1)
        num_groups = max(int(num_groups), 1)
        return min(num_groups, self.max_num_groups) if self.max_num_groups is not None else num_groups

    def generate_dn_label_query(self, gt_labels: torch.Tensor, num_groups: int) -> torch.Tensor:
        """Repeat positive/negative labels and apply reference label noise."""
        labels = gt_labels.repeat(2 * num_groups)
        if self.label_noise_ratio > 0:
            noise_mask = torch.rand(labels.shape, device=labels.device) < self.label_noise_ratio * 0.5
            if noise_mask.any():
                labels[noise_mask] = torch.randint(
                    0, self.num_classes, (int(noise_mask.sum()),), dtype=labels.dtype, device=labels.device
                )
        return self.label_embedding(labels)

    def generate_dn_bbox_query(self, gt_bboxes: torch.Tensor, num_groups: int) -> torch.Tensor:
        """Noise only normalized ``cxcywh`` while retaining the target angle."""
        expanded = gt_bboxes.repeat(2 * num_groups, 1).clone()
        num_targets = len(gt_bboxes)
        positive = torch.arange(num_targets, dtype=torch.long, device=gt_bboxes.device)
        positive = positive[None].repeat(num_groups, 1)
        positive += 2 * num_targets * torch.arange(num_groups, device=gt_bboxes.device)[:, None]
        positive = positive.flatten()
        negative = positive + num_targets

        random_sign = torch.randint(0, 2, expanded[:, :4].shape, device=expanded.device).to(expanded.dtype) * 2 - 1
        random_part = torch.rand_like(expanded[:, :4])
        random_part[negative] += 1.0
        random_part *= random_sign
        expanded[:, :4] += (
            random_part * expanded[:, 2:4].repeat(1, 2) * self.box_noise_scale / 2
        )
        expanded[:, :2].clamp_(0.0, 1.0)
        expanded[:, 2:4].clamp_(1e-6, 1.0)
        expanded[:, 4].clamp_(1e-6, 1 - 1e-6)
        return torch.logit(expanded, eps=1e-6)

    def collate_dn_queries(
        self,
        input_label_query: torch.Tensor,
        input_bbox_query: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
        gt_groups: list[int],
        num_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place repeated queries in per-image positive/negative group blocks."""
        device = input_label_query.device
        max_num_target = int(max(gt_groups))
        num_dn = int(max_num_target * 2 * num_groups)
        query_index = torch.cat([torch.arange(count, device=device) for count in gt_groups])
        query_index = torch.cat([query_index + max_num_target * index for index in range(2 * num_groups)]).long()
        batch_index = batch_idx.repeat(2 * num_groups)

        labels = input_label_query.new_zeros((batch_size, num_dn, self.embed_dims))
        boxes = input_bbox_query.new_zeros((batch_size, num_dn, 5))
        labels[batch_index, query_index] = input_label_query
        boxes[batch_index, query_index] = input_bbox_query
        return labels, boxes

    def generate_dn_mask(self, max_num_target: int, num_groups: int, device: torch.device) -> torch.Tensor:
        """Isolate matching queries and every CDN group."""
        num_dn = int(max_num_target * 2 * num_groups)
        total_queries = num_dn + self.num_matching_queries
        attention_mask = torch.zeros((total_queries, total_queries), dtype=torch.bool, device=device)
        attention_mask[num_dn:, :num_dn] = True
        for index in range(num_groups):
            start = max_num_target * 2 * index
            end = max_num_target * 2 * (index + 1)
            attention_mask[start:end, :start] = index > 0
            attention_mask[start:end, end:num_dn] = index < num_groups - 1
        return attention_mask


class RHINOOBBDecoder(RTDETRDecoder):
    """Reference RHINO v2 OBB decoder, isolated from the generic RT-DETR decoder."""

    export = False
    _ANGLE_LOGIT_BIAS = 0.0

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 900,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 2048,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = True,
    ):
        # Deliberately do not construct RTDETRDecoder's Conv+BN projections or decoder.
        nn.Module.__init__(self)
        self.hidden_dim = int(hd)
        self.nhead = int(nh)
        self.nl = 4
        self.num_feature_levels = 4
        self.nc = int(nc)
        self.num_queries = int(nq)
        self.num_decoder_layers = int(ndl)
        self.num_encoder_layers = 6
        self.num_denoising = int(nd)
        self.label_noise_ratio = float(label_noise_ratio)
        self.box_noise_scale = float(box_noise_scale)
        self.learnt_init_query = True

        self.channel_mapper = _RhinoChannelMapper(tuple(ch), hd)
        self.position_encoding = _RhinoSinePositionEncoding(num_feats=hd // 2)
        self.encoder = _RhinoDeformableEncoder(hd, 6, nh, 4, ndp, d_ffn, dropout)

        decoder_layer = RhinoTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, 4, ndp)
        self.decoder = RhinoTransformerDecoder(hd, decoder_layer, ndl, eval_idx)
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.query_embedding = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(2 * hd, hd, hd, num_layers=2)

        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 5, num_layers=3)
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 5, num_layers=3) for _ in range(ndl)])

        self.version = "v2"
        self.dn_group_mode = "dynamic"
        self.max_num_groups: int | None = None
        self.rhino_cfg: dict[str, Any] = {}
        self.dn_query_generator: RotatedCdnQueryGenerator | None = None
        self._reset_rhino_parameters()
        self.configure_rhino({})

    def configure_rhino(self, rhino_cfg: dict[str, Any] | None = None) -> None:
        """Apply RHINO-only runtime/loss configuration and reject unsupported decoder versions."""
        cfg = {
            "version": "v2",
            "num_queries": self.num_queries,
            "num_denoising_queries": self.num_denoising,
            "dn_group_mode": "dynamic",
            "max_num_groups": 30,
            "max_candidates": 500,
            "matcher_costs": None,
            "dn_matcher_costs": None,
            "loss_weights": None,
            "loss_types": None,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
        }
        if rhino_cfg:
            cfg.update(deepcopy(rhino_cfg))
        version = str(cfg["version"]).lower()
        if version != "v2":
            raise ValueError(f"Unsupported RHINO decoder version {cfg['version']!r}; only 'v2' is implemented.")

        requested_queries = int(cfg["num_queries"])
        if requested_queries != self.num_queries:
            self.num_queries = requested_queries
            self.query_embedding = nn.Embedding(self.num_queries, self.hidden_dim).to(self.enc_score_head.weight.device)
            xavier_uniform_(self.query_embedding.weight)

        self.version = version
        self.num_denoising = int(cfg["num_denoising_queries"])
        self.dn_group_mode = str(cfg["dn_group_mode"])
        self.max_num_groups = None if cfg["max_num_groups"] is None else int(cfg["max_num_groups"])
        self.max_candidates = int(cfg["max_candidates"])
        self.rhino_cfg = cfg
        self.dn_query_generator = RotatedCdnQueryGenerator(
            num_classes=self.nc,
            embed_dims=self.hidden_dim,
            num_matching_queries=self.num_queries,
            label_embedding=self.denoising_class_embed,
            num_denoising_queries=self.num_denoising,
            label_noise_ratio=self.label_noise_ratio,
            box_noise_scale=self.box_noise_scale,
            group_mode=self.dn_group_mode,
            max_num_groups=self.max_num_groups,
        )

    def _reset_rhino_parameters(self) -> None:
        """Initialize RHINO-private prediction and projection modules."""
        class_bias = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, class_bias)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for classifier, regressor in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(classifier.bias, class_bias)
            constant_(regressor.layers[-1].weight, 0.0)
            constant_(regressor.layers[-1].bias, 0.0)
        for module in (self.enc_output[0], self.query_embedding, self.denoising_class_embed):
            xavier_uniform_(module.weight)
        for projection in self.channel_mapper.input_convs:
            xavier_uniform_(projection[0].weight)
        if self.channel_mapper.extra_conv is not None:
            xavier_uniform_(self.channel_mapper.extra_conv[0].weight)

    @staticmethod
    def external_to_internal_angle(angle: torch.Tensor) -> torch.Tensor:
        """Convert radians to the RHINO ``angle/pi`` contract."""
        return torch.remainder(angle, math.pi) / math.pi

    @staticmethod
    def internal_to_external_angle(angle: torch.Tensor) -> torch.Tensor:
        """Convert the RHINO normalized angle to radians."""
        return angle * math.pi

    def _get_encoder_input(
        self,
        features: list[torch.Tensor],
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[list[int]], torch.Tensor, torch.Tensor]:
        """Map inputs to four levels and execute the private RHINO encoder."""
        mapped = self.channel_mapper(features)
        if padding_mask is None:
            masks = [
                torch.zeros((feature.shape[0], *feature.shape[-2:]), dtype=torch.bool, device=feature.device)
                for feature in mapped
            ]
        else:
            if padding_mask.ndim != 3 or padding_mask.shape[0] != mapped[0].shape[0]:
                raise ValueError(
                    "RHINO padding_mask must have shape [batch, height, width], "
                    f"got {tuple(padding_mask.shape)}."
                )
            padding_mask = padding_mask.to(device=mapped[0].device, dtype=torch.bool)
            masks = [
                F.interpolate(
                    padding_mask[:, None].float(),
                    size=feature.shape[-2:],
                    mode="nearest",
                )[:, 0].bool()
                for feature in mapped
            ]
        positions = [self.position_encoding(mask) for mask in masks]
        return self.encoder(mapped, masks, positions)

    def _generate_encoder_output_proposals(
        self,
        memory: torch.Tensor,
        padding_mask: torch.Tensor,
        shapes: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate mask-aware five-dimensional RHINO encoder proposals."""
        batch_size, value_count, _ = memory.shape
        if padding_mask.shape != (batch_size, value_count):
            raise ValueError(
                f"RHINO flattened padding mask must have shape {(batch_size, value_count)}, "
                f"got {tuple(padding_mask.shape)}."
            )

        proposals = []
        start = 0
        for level, (height, width) in enumerate(shapes):
            level_count = int(height * width)
            level_mask = padding_mask[:, start : start + level_count].view(batch_size, height, width, 1)
            valid_height = (~level_mask[:, :, 0, 0]).sum(1, keepdim=True)
            valid_width = (~level_mask[:, 0, :, 0]).sum(1, keepdim=True)

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, dtype=memory.dtype, device=memory.device),
                torch.arange(width, dtype=memory.dtype, device=memory.device),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), dim=-1)
            scale = torch.cat((valid_width, valid_height), dim=1).to(memory.dtype).view(batch_size, 1, 1, 2)
            grid = (grid[None] + 0.5) / scale
            wh = torch.ones_like(grid) * (0.05 * (2.0**level))
            angle = torch.full_like(grid[..., :1], 0.5)
            proposals.append(torch.cat((grid, wh, angle), dim=-1).view(batch_size, -1, 5))
            start += level_count

        proposals = torch.cat(proposals, dim=1)
        proposal_is_valid = ((proposals > 0.01) & (proposals < 0.99)).all(-1, keepdim=True)
        proposal_logits = torch.log(proposals / (1.0 - proposals))
        proposal_logits = proposal_logits.masked_fill(padding_mask[..., None], float("inf"))
        proposal_logits = proposal_logits.masked_fill(~proposal_is_valid, float("inf"))

        output_memory = memory.masked_fill(padding_mask[..., None], 0.0)
        output_memory = output_memory.masked_fill(~proposal_is_valid, 0.0)
        return self.enc_output(output_memory), proposal_logits

    def _get_decoder_input_rhino(
        self,
        memory: torch.Tensor,
        shapes: list[list[int]],
        padding_mask: torch.Tensor,
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k two-stage proposals and prepend CDN queries."""
        batch_size = memory.shape[0]
        transformed_memory, proposal_logits_all = self._generate_encoder_output_proposals(
            memory, padding_mask, shapes
        )
        encoder_scores_all = self.enc_score_head(transformed_memory)
        topk_indices = torch.topk(encoder_scores_all.max(-1).values, self.num_queries, dim=1).indices
        batch_indices = torch.arange(batch_size, device=memory.device)[:, None].expand_as(topk_indices)

        topk_features = transformed_memory[batch_indices, topk_indices]
        topk_proposals = proposal_logits_all[batch_indices, topk_indices]
        topk_logits = self.enc_bbox_head(topk_features) + topk_proposals
        encoder_boxes = topk_logits.sigmoid()
        encoder_scores = encoder_scores_all[batch_indices, topk_indices]

        query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        references = topk_logits.detach()
        if dn_embed is not None and dn_bbox is not None:
            query = torch.cat((dn_embed, query), dim=1)
            references = torch.cat((dn_bbox, references), dim=1)
        return query, references, encoder_boxes, encoder_scores

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the RHINO encoder, v2 decoder, and rotated denoising path."""
        image_padding_mask = batch.get("padding_mask") if batch is not None else None
        memory, shapes, padding_mask, valid_ratios = self._get_encoder_input(x, image_padding_mask)
        dn_embed = dn_bbox = attention_mask = dn_meta = None
        if self.training and self.dn_query_generator is not None:
            dn_embed, dn_bbox, attention_mask, dn_meta = self.dn_query_generator(batch)

        query, references, encoder_boxes, encoder_scores = self._get_decoder_input_rhino(
            memory, shapes, padding_mask, dn_embed, dn_bbox
        )
        decoder_boxes, decoder_scores = self.decoder(
            query,
            references,
            memory,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attention_mask,
            padding_mask=padding_mask,
            valid_ratios=valid_ratios,
        )

        if dn_meta is not None and dn_meta["dn_num_split"][0] == 0:
            # Preserve a zero-valued DDP dependency for all-empty batches.
            decoder_scores = decoder_scores + self.denoising_class_embed.weight[0, 0] * 0.0

        outputs = decoder_boxes, decoder_scores, encoder_boxes, encoder_scores, dn_meta
        if self.training:
            return outputs
        inference = torch.cat((decoder_boxes.squeeze(0), decoder_scores.squeeze(0).sigmoid()), dim=-1)
        return inference if self.export else (inference, outputs)
