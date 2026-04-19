# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RHINOOBBModel
from ultralytics.utils.torch_utils import TORCH_1_11

from .predict import RHINOOBBPredictor
from .train import RHINOOBBTrainer
from .val import RHINOOBBValidator


class RHINO(Model):
    """Interface for RHINO OBB models."""

    def __init__(self, model: str = "rhino-r50-obb.yaml", task: str | None = None) -> None:
        assert TORCH_1_11, "RHINO requires torch>=1.11"
        super().__init__(model=model, task=task)

    @property
    def task_map(self) -> dict:
        return {
            "obb": {
                "predictor": RHINOOBBPredictor,
                "validator": RHINOOBBValidator,
                "trainer": RHINOOBBTrainer,
                "model": RHINOOBBModel,
            }
        }
