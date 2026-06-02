# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel
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


class OBBTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Oriented Bounding Box (OBB) model.

    This trainer specializes in training YOLO models that detect oriented bounding boxes, which are useful for
    detecting objects at arbitrary angles rather than just axis-aligned rectangles.

    Attributes:
        loss_names (tuple): Names of the loss components used during training including box_loss, cls_loss,
            and dfl_loss.

    Methods:
        get_model: Return OBBModel initialized with specified config and weights.
        get_validator: Return an instance of OBBValidator for validation of YOLO model.

    Examples:
        >>> from ultralytics.models.yolo.obb import OBBTrainer
        >>> args = dict(model="yolo11n-obb.pt", data="dota8.yaml", epochs=3)
        >>> trainer = OBBTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: list[Any] | None = None):
        """
        Initialize an OBBTrainer object for training Oriented Bounding Box (OBB) models.

        Args:
            cfg (dict, optional): Configuration dictionary for the trainer. Contains training parameters and
                model configuration.
            overrides (dict, optional): Dictionary of parameter overrides for the configuration. Any values here
                will take precedence over those in cfg.
            _callbacks (list[Any], optional): List of callback functions to be invoked during training.
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)
        self.args.active_shoreline_aux_weight = 0.0
        self.add_callback("on_train_epoch_start", on_train_epoch_start)

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> OBBModel:
        """
        Return OBBModel initialized with specified config and weights.

        Args:
            cfg (str | dict, optional): Model configuration. Can be a path to a YAML config file, a dictionary
                containing configuration parameters, or None to use default configuration.
            weights (str | Path, optional): Path to pretrained weights file. If None, random initialization is used.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (OBBModel): Initialized OBBModel with the specified configuration and weights.

        Examples:
            >>> trainer = OBBTrainer()
            >>> model = trainer.get_model(cfg="yolo11n-obb.yaml", weights="yolo11n-obb.pt")
        """
        model = OBBModel(
            cfg, nc=self.data["nc"], ch=self.data.get("input_channels", self.data["channels"]), verbose=verbose and RANK == -1
        )
        self.args.angle_mode = model.yaml.get("angle_mode", getattr(self.args, "angle_mode", "oc"))
        return self._finalize_model_build(model, weights)

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        head = getattr(getattr(self, "model", None), "model", [None])[-1]
        self.loss_names = (
            ("box_loss", "cls_loss", "ctr_loss", "shoreline_prior_loss", "land_water_prior_loss", "shore_aux_loss")
            if head.__class__.__name__ == "RotatedFCOS"
            else ("box_loss", "cls_loss", "dfl_loss", "shoreline_prior_loss", "land_water_prior_loss", "shore_aux_loss")
        )
        return yolo.obb.OBBValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
