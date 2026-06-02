# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models.rtdetr.train import RTDETROBBTrainer
from ultralytics.utils import RANK


class RHINOOBBTrainer(RTDETROBBTrainer):
    """Trainer for RHINO OBB models."""

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        from ultralytics.nn.tasks import RHINOOBBModel

        model = RHINOOBBModel(
            cfg, nc=self.data["nc"], ch=self.data.get("input_channels", self.data["channels"]), verbose=verbose and RANK == -1
        )
        return self._finalize_model_build(model, weights)

    def get_validator(self):
        from .val import RHINOOBBValidator

        self.loss_names = (
            "giou_loss",
            "cls_loss",
            "l1_loss",
            "shoreline_prior_loss",
            "land_water_prior_loss",
        )
        return RHINOOBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
