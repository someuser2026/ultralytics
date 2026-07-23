"""Public model wrapper for native Mask2Former instance segmentation."""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import SegmentationModel

from .predict import Mask2FormerPredictor
from .train import Mask2FormerTrainer
from .val import Mask2FormerValidator


class Mask2Former(Model):
    """Mask2Former family interface with reference instance inference."""

    def __init__(self, model: str = "mask2former-yolo12-seg.yaml", task: str | None = None) -> None:
        super().__init__(model=model, task=task)

    @property
    def task_map(self) -> dict:
        return {
            "segment": {
                "predictor": Mask2FormerPredictor,
                "validator": Mask2FormerValidator,
                "trainer": Mask2FormerTrainer,
                "model": SegmentationModel,
            }
        }

