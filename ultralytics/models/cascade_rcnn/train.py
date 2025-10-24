# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Training hooks for Cascade R-CNN models."""

from __future__ import annotations

from pathlib import Path

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import CascadeMaskRCNNModel, CascadeRCNNDetectionModel
from ultralytics.utils import RANK


class CascadeRCNNTrainer(DetectionTrainer):
    """Detection trainer for Cascade R-CNN."""

    def get_model(self, cfg: dict | str | None = None, weights: str | None = None, verbose: bool = True):
        model = CascadeRCNNDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model


class CascadeMaskRCNNTrainer(SegmentationTrainer):
    """Segmentation trainer for Cascade Mask R-CNN."""

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        model = CascadeMaskRCNNModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
