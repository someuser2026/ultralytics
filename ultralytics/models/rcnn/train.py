# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import RCNNOBBModel, RCNNSegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .val import RCNNOBBValidator, RCNNSegmentationValidator


class RCNNSegmentationTrainer(yolo.segment.SegmentationTrainer):
    """Mask R-CNN / Cascade Mask R-CNN trainer."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        super().__init__(cfg, {**(overrides or {}), "task": "segment"}, _callbacks)

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        model = RCNNSegmentationModel(
            cfg, nc=self.data["nc"], ch=self.data.get("input_channels", self.data["channels"]), verbose=verbose and RANK == -1
        )
        return self._finalize_model_build(model, weights)

    def get_validator(self):
        model = unwrap_model(self.model)
        head = getattr(model, "model", [None])[-1]
        self.loss_names = tuple(getattr(head, "loss_names", ("loss",)))
        return RCNNSegmentationValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


class RCNNOBBTrainer(yolo.obb.OBBTrainer):
    """Rotated Faster R-CNN / Oriented R-CNN trainer."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: list[Any] | None = None):
        super().__init__(cfg, {**(overrides or {}), "task": "obb"}, _callbacks)

    def get_model(self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True):
        model = RCNNOBBModel(
            cfg, nc=self.data["nc"], ch=self.data.get("input_channels", self.data["channels"]), verbose=verbose and RANK == -1
        )
        self.args.angle_mode = model.yaml.get("angle_mode", getattr(self.args, "angle_mode", "le90"))
        return self._finalize_model_build(model, weights)

    def get_validator(self):
        head = getattr(getattr(self, "model", None), "model", [None])[-1]
        self.loss_names = tuple(getattr(head, "loss_names", ("loss",)))
        return RCNNOBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
