# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models.rtdetr.train import RTDETROBBTrainer
from ultralytics.utils import RANK, colorstr

from .dataset import RHINODataset


class RHINOOBBTrainer(RTDETROBBTrainer):
    """Trainer for RHINO OBB models."""

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build the RHINO-only dataset without changing generic RT-DETR datasets."""
        return RHINODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            task=self.args.task,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        from ultralytics.nn.tasks import RHINOOBBModel

        model = RHINOOBBModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("input_channels", self.data["channels"]),
            verbose=verbose and RANK == -1,
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
