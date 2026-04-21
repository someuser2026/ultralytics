# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
RHINO-oriented transformer decoder modules.

This file keeps RHINO-derived logic isolated from the default RT-DETR path.
"""

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
from .utils import bias_init_with_prob, inverse_sigmoid, linear_init


class RhinoMSDeformAttn(MSDeformAttn):
    """RHINO rotated deformable attention using 5D references."""

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        if refer_bbox.ndim == 3:
            refer_bbox = refer_bbox.unsqueeze(2)
        if refer_bbox.shape[2] == 1:
            refer_bbox = refer_bbox.expand(-1, -1, self.n_levels, -1)
        elif refer_bbox.shape[2] != self.n_levels:
            raise ValueError(
                f"Expected reference boxes to have 1 or {self.n_levels} levels, but got {refer_bbox.shape[2]}."
            )

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)

        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            sampling_locations = refer_bbox[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[
                None, None, None, :, None, :
            ]
        elif num_points == 4:
            sampling_locations = (
                refer_bbox[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            )
        elif num_points == 5:
            cosa = torch.cos(refer_bbox[..., 4:])
            sina = torch.sin(refer_bbox[..., 4:])
            rot = torch.cat([cosa, -sina, sina, cosa], dim=-1).view(bs, len_q, self.n_levels, 2, 2)
            wh = refer_bbox[..., 2:4] * 0.5
            rotated_points = torch.einsum("bqlij,bqlj->bqli", rot, wh)
            sampling_locations = (
                refer_bbox[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * rotated_points[:, :, None, :, None, :]
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2, 4 or 5, but got {num_points}.")

        output = self.output_proj(
            self._deformable_attn(value, value_shapes, sampling_locations, attention_weights)
        )
        return output

    @staticmethod
    def _deformable_attn(
        value: torch.Tensor,
        value_shapes: list,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        from .utils import multi_scale_deformable_attn_pytorch

        return multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)


class RhinoTransformerDecoderLayer(DeformableTransformerDecoderLayer):
    """Decoder layer using RHINO rotated cross-attention."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super().__init__(d_model, n_heads, d_ffn, dropout, act, n_levels, n_points, use_obb=False)
        self.cross_attn = RhinoMSDeformAttn(d_model, n_levels, n_heads, n_points)


class RhinoTransformerDecoder(DeformableTransformerDecoder):
    """RHINO decoder preserving 5D references across refinement."""

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = embed
        dec_bboxes = []
        dec_cls = []
        reference_points = refer_bbox.sigmoid()

        for i, layer in enumerate(self.layers):
            output = layer(
                output,
                reference_points,
                feats,
                shapes,
                padding_mask,
                attn_mask,
                pos_mlp(reference_points[..., :4]),
            )

            logits = bbox_head[i](output)
            refined_bbox = torch.sigmoid(logits + inverse_sigmoid(reference_points))

            if self.training:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            reference_points = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class RotatedCdnQueryGenerator:
    """RHINO rotated CDN query generator for normalized 5D boxes."""

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

    def __call__(self, batch: dict[str, Any] | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]] | tuple[None, None, None, None]:
        if batch is None or self.num_denoising_queries <= 0:
            return None, None, None, None

        gt_groups = batch.get("gt_groups", [])
        if not gt_groups or max(gt_groups) <= 0:
            return None, None, None, None

        gt_labels = batch["cls"].view(-1).long()
        gt_bboxes = batch["bboxes"]
        batch_idx = batch["batch_idx"].view(-1).long()
        batch_size = len(gt_groups)
        max_num_target = int(max(gt_groups))
        num_groups = self.get_num_groups(max_num_target)

        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, batch_idx, batch_size, gt_groups, num_groups
        )
        attn_mask = self.generate_dn_mask(max_num_target, num_groups, dn_label_query.device)
        num_dn = int(max_num_target * 2 * num_groups)
        dn_meta = {
            "num_denoising_queries": num_dn,
            "num_denoising_groups": num_groups,
            "dn_num_split": [num_dn, self.num_matching_queries],
        }
        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def get_num_groups(self, max_num_target: int) -> int:
        if self.group_mode == "static":
            num_groups = self.num_groups if self.num_groups is not None else 1
        else:
            num_groups = self.num_denoising_queries // max(max_num_target, 1)
        num_groups = max(int(num_groups), 1)
        if self.max_num_groups is not None:
            num_groups = min(num_groups, int(self.max_num_groups))
        return num_groups

    def generate_dn_label_query(self, gt_labels: torch.Tensor, num_groups: int) -> torch.Tensor:
        labels = gt_labels.repeat(2 * num_groups)
        if self.label_noise_ratio > 0:
            mask = torch.rand(labels.shape, device=labels.device) < (self.label_noise_ratio * 0.5)
            if mask.any():
                labels[mask] = torch.randint(0, self.num_classes, (int(mask.sum()),), device=labels.device, dtype=labels.dtype)
        return self.label_embedding(labels)

    def generate_dn_bbox_query(self, gt_bboxes: torch.Tensor, num_groups: int) -> torch.Tensor:
        if gt_bboxes.numel() == 0:
            return gt_bboxes.new_zeros((0, 5))

        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)
        gt_bboxes_expand = gt_bboxes_expand.clone()
        gt_bboxes_expand[:, 4] = RHINOOBBDecoder.external_to_internal_angle(gt_bboxes_expand[:, 4])

        positive_idx = torch.arange(len(gt_bboxes), dtype=torch.long, device=gt_bboxes.device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)
        positive_idx += 2 * len(gt_bboxes) * torch.arange(num_groups, dtype=torch.long, device=gt_bboxes.device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        rand_sign = (torch.randint_like(gt_bboxes_expand, low=0, high=2, dtype=torch.float32) * 2.0) - 1.0
        rand_part = torch.rand_like(gt_bboxes_expand)
        rand_part[negative_idx] += 1.0
        rand_part *= rand_sign

        noise_part = gt_bboxes_expand.new_zeros(gt_bboxes_expand.shape)
        noise_part[:, :4] = rand_part[:, :4] * gt_bboxes_expand[:, 2:4].repeat(1, 2) * self.box_noise_scale / 2
        noisy_bboxes = gt_bboxes_expand + noise_part
        noisy_bboxes[:, :2].clamp_(min=0.0, max=1.0)
        noisy_bboxes[:, 2:4].clamp_(min=1e-6, max=1.0)
        noisy_bboxes[:, 4:5].clamp_(min=1e-6, max=1 - 1e-6)
        return torch.logit(noisy_bboxes, eps=1e-6)

    def collate_dn_queries(
        self,
        input_label_query: torch.Tensor,
        input_bbox_query: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
        gt_groups: list[int],
        num_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = input_label_query.device
        max_num_target = int(max(gt_groups))
        num_dn = int(max_num_target * 2 * num_groups)

        map_query_index = torch.cat([torch.arange(num_target, device=device) for num_target in gt_groups], dim=0)
        map_query_index = torch.cat([map_query_index + max_num_target * i for i in range(2 * num_groups)]).long()
        batch_idx_expand = batch_idx.repeat(2 * num_groups, 1).view(-1)
        mapper = (batch_idx_expand, map_query_index)

        batched_label_query = torch.zeros(batch_size, num_dn, self.embed_dims, device=device)
        batched_bbox_query = torch.zeros(batch_size, num_dn, 5, device=device)
        batched_label_query[mapper] = input_label_query
        batched_bbox_query[mapper] = input_bbox_query
        return batched_label_query, batched_bbox_query

    def generate_dn_mask(self, max_num_target: int, num_groups: int, device: torch.device) -> torch.Tensor:
        num_dn = int(max_num_target * 2 * num_groups)
        tgt_size = num_dn + self.num_matching_queries
        attn_mask = torch.zeros((tgt_size, tgt_size), dtype=torch.bool, device=device)
        attn_mask[num_dn:, :num_dn] = True

        for i in range(num_groups):
            start = max_num_target * 2 * i
            end = max_num_target * 2 * (i + 1)
            if i > 0:
                attn_mask[start:end, :start] = True
            if i < num_groups - 1:
                attn_mask[start:end, end:num_dn] = True
        return attn_mask


class RHINOOBBDecoder(RTDETRDecoder):
    """RHINO OBB decoder with 5D references and rotated CDN."""

    export = False
    _ANGLE_LOGIT_BIAS = math.log(0.25 / 0.75)

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        super().__init__(
            nc=nc,
            ch=ch,
            hd=hd,
            nq=nq,
            ndp=ndp,
            nh=nh,
            ndl=ndl,
            d_ffn=d_ffn,
            dropout=dropout,
            act=act,
            eval_idx=eval_idx,
            nd=nd,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
        )
        self.num_feature_levels = len(ch)
        self.query_embedding = nn.Embedding(self.num_queries, hd)
        self.enc_bbox_head = MLP(hd, hd, 5, num_layers=3)
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 5, num_layers=3) for _ in range(ndl)])
        decoder_layer = RhinoTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = RhinoTransformerDecoder(hd, decoder_layer, ndl, eval_idx)
        self.version = "v2"
        self.dn_group_mode = "dynamic"
        self.max_num_groups = None
        self.rhino_cfg: dict[str, Any] = {}
        self.dn_query_generator: RotatedCdnQueryGenerator | None = None
        self._reset_rhino_parameters()
        self.configure_rhino({})

    def configure_rhino(self, rhino_cfg: dict[str, Any] | None = None) -> None:
        cfg = {
            "version": "v2",
            "num_queries": self.num_queries,
            "num_denoising_queries": self.num_denoising,
            "dn_group_mode": "dynamic",
            "max_num_groups": None,
            "matcher_costs": None,
            "dn_matcher_costs": None,
            "loss_weights": None,
            "loss_types": None,
        }
        if rhino_cfg:
            cfg.update(deepcopy(rhino_cfg))

        if int(cfg["num_queries"]) != self.num_queries:
            self.num_queries = int(cfg["num_queries"])
            self.query_embedding = nn.Embedding(self.num_queries, self.hidden_dim).to(self.query_embedding.weight.device)
            xavier_uniform_(self.query_embedding.weight)

        self.version = str(cfg["version"])
        self.num_denoising = int(cfg["num_denoising_queries"])
        self.dn_group_mode = str(cfg["dn_group_mode"])
        self.max_num_groups = cfg["max_num_groups"]
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
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        xavier_uniform_(self.query_embedding.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

    @staticmethod
    def external_to_internal_angle(angle: torch.Tensor) -> torch.Tensor:
        return (angle / math.pi) + 0.25

    @staticmethod
    def internal_to_external_angle(angle: torch.Tensor) -> torch.Tensor:
        return (angle - 0.25) * math.pi

    def _get_decoder_input_rhino(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = feats.shape[0]
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)
        enc_outputs_scores = self.enc_score_head(features)

        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype, device=feats.device).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)
        top_k_angle = torch.full_like(top_k_anchors[:, :, :1], self._ANGLE_LOGIT_BIAS)
        output_proposals = torch.cat([top_k_anchors, top_k_angle], dim=-1)

        topk_coords_unact = self.enc_bbox_head(top_k_features) + output_proposals
        topk_coords = topk_coords_unact.sigmoid()
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        query = self.query_embedding.weight.unsqueeze(0).expand(bs, -1, -1)
        reference_points = topk_coords_unact.detach() if self.training else topk_coords_unact

        if dn_embed is not None and dn_bbox is not None:
            query = torch.cat([dn_embed, query], dim=1)
            reference_points = torch.cat([dn_bbox, reference_points], dim=1)

        return query, reference_points, topk_coords, enc_scores

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        feats, shapes = self._get_encoder_input(x)
        dn_embed = dn_bbox = attn_mask = dn_meta = None
        if self.training and self.dn_query_generator is not None:
            dn_embed, dn_bbox, attn_mask, dn_meta = self.dn_query_generator(batch)

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input_rhino(feats, shapes, dn_embed, dn_bbox)
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        enc_bboxes = torch.cat([enc_bboxes[..., :4], self.internal_to_external_angle(enc_bboxes[..., 4:5])], dim=-1)
        dec_bboxes = torch.cat([dec_bboxes[..., :4], self.internal_to_external_angle(dec_bboxes[..., 4:5])], dim=-1)

        outputs = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return outputs
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), dim=-1)
        return y if self.export else (y, outputs)
