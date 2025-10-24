# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Cascade R-CNN model interface built on Ultralytics Model API."""

from __future__ import annotations

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import CascadeMaskRCNNModel, CascadeRCNNDetectionModel

from .predict import CascadeMaskRCNNPredictor, CascadeRCNNPredictor
from .train import CascadeMaskRCNNTrainer, CascadeRCNNTrainer
from .val import CascadeMaskRCNNValidator, CascadeRCNNValidator


class CascadeRCNN(Model):
    """Entry point for Cascade R-CNN style detection and segmentation models."""

    def __init__(self, model: str = "cascade-rcnn.yaml", task: str | None = None, verbose: bool = False) -> None:
        super().__init__(model=model, task=task or "detect", verbose=verbose)

    @property
    def task_map(self) -> dict:
        return {
            "detect": {
                "model": CascadeRCNNDetectionModel,
                "trainer": CascadeRCNNTrainer,
                "validator": CascadeRCNNValidator,
                "predictor": CascadeRCNNPredictor,
            },
            "segment": {
                "model": CascadeMaskRCNNModel,
                "trainer": CascadeMaskRCNNTrainer,
                "validator": CascadeMaskRCNNValidator,
                "predictor": CascadeMaskRCNNPredictor,
            },
        }
