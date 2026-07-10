# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model


def on_train_epoch_start(trainer) -> None:
    """Linearly ramp the shoreline auxiliary loss weight during early training."""
    target_weight = float(getattr(trainer.args, "shoreline_aux_weight", 0.20))
    warmup_epochs = max(int(getattr(trainer.args, "shoreline_aux_warmup_epochs", 10)), 0)
    active_weight = target_weight if warmup_epochs == 0 else target_weight * min(max(trainer.epoch, 0) / warmup_epochs, 1.0)
    trainer.args.active_shoreline_aux_weight = active_weight
    if getattr(trainer, "model", None) is not None:
        unwrap_model(trainer.model).args.active_shoreline_aux_weight = active_weight


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    This trainer specializes in handling segmentation tasks, extending the detection trainer with segmentation-specific
    functionality including model initialization, validation, and visualization.

    Attributes:
        loss_names (tuple[str]): Names of the loss components used during training.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationTrainer
        >>> args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml", epochs=3)
        >>> trainer = SegmentationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks=None):
        """
        Initialize a SegmentationTrainer object.

        Args:
            cfg (dict): Configuration dictionary with default training settings.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)
        self.args.active_shoreline_aux_weight = 0.0
        self.add_callback("on_train_epoch_start", on_train_epoch_start)

    def get_model(self, cfg: dict | str | None = None, weights: str | Path | None = None, verbose: bool = True):
        """
        Initialize and return a SegmentationModel with specified configuration and weights.

        Args:
            cfg (dict | str, optional): Model configuration. Can be a dictionary, a path to a YAML file, or None.
            weights (str | Path, optional): Path to pretrained weights file.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (SegmentationModel): Initialized segmentation model with loaded weights if specified.

        Examples:
            >>> trainer = SegmentationTrainer()
            >>> model = trainer.get_model(cfg="yolo11n-seg.yaml")
            >>> model = trainer.get_model(weights="yolo11n-seg.pt", verbose=False)
        """
        model = SegmentationModel(
            cfg, nc=self.data["nc"], ch=self.data.get("input_channels", self.data["channels"]), verbose=verbose and RANK == -1
        )
        return self._finalize_model_build(model, weights)

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        if self._uses_mask2former_head():
            self.loss_names = "cls_loss", "mask_loss", "dice_loss"
        else:
            self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "shoreline_prior_loss", "land_water_prior_loss", "shore_aux_loss"
        from ultralytics.nn.modules.pointrend import has_pointrend

        if has_pointrend(unwrap_model(self.model)):
            self.loss_names = (*self.loss_names, "point_loss")
        return yolo.segment.SegmentationValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _uses_mask2former_head(self) -> bool:
        """Return True when the current model ends with the native Mask2Former head."""
        model = getattr(self, "model", None)
        if model is None:
            return False
        model = unwrap_model(model)
        return bool(getattr(model, "model", None)) and model.model[-1].__class__.__name__ == "Mask2FormerHead"
