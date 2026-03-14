# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Interface for Baidu's RT-DETR, a Vision Transformer-based real-time object detector.

RT-DETR offers real-time performance and high accuracy, excelling in accelerated backends like CUDA with TensorRT.
It features an efficient hybrid encoder and IoU-aware query selection for enhanced detection accuracy.

References:
    https://arxiv.org/pdf/2304.08069.pdf
"""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel, RTDETRSegmentModel, RTDETROBBModel
from ultralytics.utils.torch_utils import TORCH_1_11

from .predict import RTDETRPredictor, RTDETRSegmentPredictor, RTDETROBBPredictor
from .train import RTDETRTrainer, RTDETRSegmentTrainer, RTDETROBBTrainer
from .val import RTDETRValidator, RTDETRSegmentValidator, RTDETROBBValidator


class RTDETR(Model):
    """
    Interface for Baidu's RT-DETR model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware
    query selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained model.

    Methods:
        task_map: Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

    Examples:
        Initialize RT-DETR with a pre-trained model
        >>> from ultralytics import RTDETR
        >>> model = RTDETR("rtdetr-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "rtdetr-l.pt", task: str | None = None) -> None:
        """
        Initialize the RT-DETR model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained model. Supports .pt, .yaml, and .yml formats.
        """
        assert TORCH_1_11, "RTDETR requires torch>=1.11"
        super().__init__(model=model, task=task)

    @property
    def task_map(self) -> dict:
        """
        Return a task map for RT-DETR, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the RT-DETR model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": RTDETRTrainer,
                "model": RTDETRDetectionModel,
            },
            "segment": {
                "predictor": RTDETRSegmentPredictor,
                "validator": RTDETRSegmentValidator,
                "trainer": RTDETRSegmentTrainer,
                "model": RTDETRSegmentModel,
            },
            "obb": {
                "predictor": RTDETROBBPredictor,
                "validator": RTDETROBBValidator,
                "trainer": RTDETROBBTrainer,
                "model": RTDETROBBModel,
            },
        }
