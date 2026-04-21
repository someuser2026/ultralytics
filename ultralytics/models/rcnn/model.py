# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RCNNOBBModel, RCNNSegmentationModel

from .predict import RCNNOBBPredictor, RCNNSegmentationPredictor
from .train import RCNNOBBTrainer, RCNNSegmentationTrainer
from .val import RCNNOBBValidator, RCNNSegmentationValidator


class RCNN(Model):
    """Interface for native RCNN-style segmentation and OBB models."""

    def __init__(self, model: str = "mask-rcnn.yaml", task: str | None = None) -> None:
        super().__init__(model=model, task=task)
        if self.task == "segment":
            self.overrides["overlap_mask"] = False
            if hasattr(self.model, "args") and isinstance(self.model.args, dict):
                self.model.args["overlap_mask"] = False

    @property
    def task_map(self) -> dict:
        return {
            "segment": {
                "predictor": RCNNSegmentationPredictor,
                "validator": RCNNSegmentationValidator,
                "trainer": RCNNSegmentationTrainer,
                "model": RCNNSegmentationModel,
            },
            "obb": {
                "predictor": RCNNOBBPredictor,
                "validator": RCNNOBBValidator,
                "trainer": RCNNOBBTrainer,
                "model": RCNNOBBModel,
            },
        }
