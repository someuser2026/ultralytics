# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Cascade R-CNN inspired heads that reuse YOLO detection modules."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import torch.nn as nn

from .head import Detect, Segment


class CascadeRCNNHead(nn.Module):
    """Cascade R-CNN detection head constructed from stacked YOLO Detect heads."""

    def __init__(
        self,
        nc: int = 80,
        num_stages: int = 3,
        ch: Sequence[int] = (),
        stage_weights: Iterable[float] | None = None,
        share_stem: bool = False,
    ) -> None:
        super().__init__()
        if not ch:
            raise ValueError("CascadeRCNNHead requires feature map channels to be supplied via the YAML config.")
        if num_stages < 1:
            raise ValueError("CascadeRCNNHead expects at least one cascade stage.")

        self.nc = nc
        self.num_stages = int(num_stages)
        self.share_stem = bool(share_stem)
        self.stage_heads = nn.ModuleList()

        first_head = self._build_stage_head(nc, ch)
        self.stage_heads.append(first_head)
        for _ in range(1, self.num_stages):
            self.stage_heads.append(first_head if self.share_stem else self._build_stage_head(nc, ch))

        weights = list(stage_weights) if stage_weights is not None else [1.0] * self.num_stages
        if len(weights) != self.num_stages:
            raise ValueError("stage_weights must match num_stages when provided.")
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if torch.any(weight_tensor < 0) or weight_tensor.sum() <= 0:
            raise ValueError("stage_weights must be non-negative and sum to a positive value.")
        self.stage_weights = (weight_tensor / weight_tensor.sum()).tolist()

        self._last_stage_outputs: List[Sequence] | None = None
        self._sync_attributes()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_tensor_list(self, data) -> List[torch.Tensor] | None:
        if isinstance(data, torch.Tensor):
            return None
        if isinstance(data, (list, tuple)):
            if data and all(torch.is_tensor(item) for item in data):
                return list(data)
            for item in data:
                result = self._find_tensor_list(item)
                if result is not None:
                    return result
        return None

    def _find_reference_tensor(self, data) -> torch.Tensor | None:
        if torch.is_tensor(data):
            return data
        if isinstance(data, (list, tuple)):
            for item in data:
                result = self._find_reference_tensor(item)
                if result is not None:
                    return result
        return None

    def _build_stage_head(self, nc: int, ch: Sequence[int]) -> nn.Module:
        return Detect(nc=nc, ch=tuple(ch))

    def _clone_inputs(self, x: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [xi for xi in x]

    def _sync_attributes(self) -> None:
        ref = self.stage_heads[-1]
        for attr in ("nc", "nl", "reg_max", "no", "stride", "anchors", "strides", "export", "format", "max_det", "inplace"):
            if hasattr(ref, attr):
                setattr(self, attr, getattr(ref, attr))

    # ---------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: List[torch.Tensor]):
        stage_results: List[Sequence] = []
        for head in self.stage_heads:
            stage_results.append(head(self._clone_inputs(x)))

        self._last_stage_outputs = stage_results
        final_output = stage_results[-1]
        return final_output

    # ---------------------------------------------------------------------
    # Initialisation utilities
    # ------------------------------------------------------------------
    def bias_init(self) -> None:
        stride = self.stride
        if isinstance(stride, torch.Tensor):
            stride_tensor = stride.flatten().detach().clone()
        else:
            stride_tensor = torch.tensor(stride, dtype=torch.float32)

        features, ref_tensor = None, None
        if self._last_stage_outputs:
            features = self._find_tensor_list(self._last_stage_outputs[-1])
            ref_tensor = self._find_reference_tensor(self._last_stage_outputs[-1])

        for head in self.stage_heads:
            if hasattr(head, "stride"):
                head_stride = stride_tensor
                if head_stride.numel() != getattr(head, "nl", head_stride.numel()):
                    if features is not None and ref_tensor is not None:
                        base_stride = stride_tensor.flatten()[0].item() if stride_tensor.numel() else 1.0
                        base_size = base_stride * ref_tensor.shape[-2]
                        computed = [base_size / feat.shape[-2] for feat in features[: getattr(head, "nl", len(features))]]
                        head_stride = torch.tensor(computed, dtype=torch.float32, device=ref_tensor.device)
                    else:
                        head_stride = head_stride[-getattr(head, "nl", head_stride.numel()):]
                head.stride = head_stride.clone()
            if hasattr(head, "bias_init"):
                head.bias_init()
        if hasattr(self.stage_heads[-1], "stride"):
            final_stride = self.stage_heads[-1].stride
            if isinstance(final_stride, torch.Tensor):
                self.stride = final_stride.clone()
            else:
                self.stride = torch.tensor(final_stride, dtype=torch.float32)
        self._sync_attributes()


class CascadeMaskRCNNHead(CascadeRCNNHead):
    """Cascade head for instance segmentation using YOLO Segment modules."""

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        num_stages: int = 3,
        ch: Sequence[int] = (),
        stage_weights: Iterable[float] | None = None,
        share_stem: bool = False,
    ) -> None:
        self.nm = nm
        self.npr = npr
        super().__init__(nc=nc, num_stages=num_stages, ch=ch, stage_weights=stage_weights, share_stem=share_stem)

    def _build_stage_head(self, nc: int, ch: Sequence[int]) -> nn.Module:
        return Segment(nc=nc, nm=self.nm, npr=self.npr, ch=tuple(ch))

    def _sync_attributes(self) -> None:
        ref = self.stage_heads[-1]
        for attr in ("nc", "nl", "reg_max", "no", "stride", "anchors", "strides", "export", "format", "max_det", "inplace"):
            if hasattr(ref, attr):
                setattr(self, attr, getattr(ref, attr))
        for attr in ("nm", "npr"):
            if hasattr(ref, attr):
                setattr(self, attr, getattr(ref, attr))

    def bias_init(self) -> None:
        super().bias_init()
