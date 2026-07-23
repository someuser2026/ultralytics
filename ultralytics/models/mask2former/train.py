"""Training adapter for Mask2Former segmentation models."""

from copy import copy

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.torch_utils import unwrap_model

from .val import Mask2FormerValidator


class Mask2FormerTrainer(SegmentationTrainer):
    """Reuse generic segmentation training while routing validation through native inference."""

    def get_validator(self):
        self.loss_names = ("cls_loss", "mask_loss", "dice_loss")
        model = unwrap_model(self.model)
        head = model.model[-1]
        if getattr(head, "point_rend_enabled", False) and "point_loss" not in self.loss_names:
            self.loss_names = (*self.loss_names, "point_loss")
        return Mask2FormerValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

