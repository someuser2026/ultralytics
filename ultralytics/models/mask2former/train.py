"""Training adapter for Mask2Former segmentation models."""

from copy import copy

from torch import nn, optim

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .val import Mask2FormerValidator


class Mask2FormerTrainer(SegmentationTrainer):
    """Reuse generic segmentation training while routing validation through native inference."""

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build an optimizer with an optional lower LR for the Mask2Former backbone."""
        backbone_lr_multiplier = float(getattr(self.args, "backbone_lr_multiplier", 1.0))
        if backbone_lr_multiplier == 1.0:
            return super().build_optimizer(model, name, lr, momentum, decay, iterations)
        if backbone_lr_multiplier <= 0:
            raise ValueError("backbone_lr_multiplier must be greater than 0.")
        if name.lower() != "adamw":
            raise ValueError("Mask2Former backbone_lr_multiplier currently requires optimizer=AdamW.")

        model = unwrap_model(model)
        if not getattr(model, "model", None):
            raise ValueError("Mask2Former backbone LR grouping requires a parsed model with model[0] as its backbone.")

        backbone_param_ids = {id(param) for param in model.model[0].parameters()}
        norm_types = tuple(module_type for type_name, module_type in nn.__dict__.items() if "Norm" in type_name)
        grouped = {
            "head_bias": [],
            "head_decay": [],
            "head_no_decay": [],
            "backbone_bias": [],
            "backbone_decay": [],
            "backbone_no_decay": [],
        }

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                scope = "backbone" if id(param) in backbone_param_ids else "head"
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    kind = "bias"
                elif (
                    isinstance(module, (*norm_types, nn.Embedding))
                    or "logit_scale" in fullname
                    or "absolute_pos_embed" in fullname
                    or param_name.startswith("lora_")
                ):
                    kind = "no_decay"
                else:
                    kind = "decay"
                grouped[f"{scope}_{kind}"].append(param)

        param_groups = []
        group_summary = []
        for scope in ("head", "backbone"):
            group_lr = lr if scope == "head" else lr * backbone_lr_multiplier
            for kind in ("bias", "decay", "no_decay"):
                params = grouped[f"{scope}_{kind}"]
                if not params:
                    continue
                group_decay = decay if kind == "decay" else 0.0
                param_groups.append(
                    {
                        "params": params,
                        "lr": group_lr,
                        "weight_decay": group_decay,
                        "is_bias": kind == "bias",
                    }
                )
                group_summary.append(f"{len(params)} {scope} {kind}(lr={group_lr:g}, decay={group_decay:g})")

        if not param_groups:
            raise ValueError("No trainable Mask2Former parameters remain after freeze/LoRA configuration.")

        optimizer = optim.AdamW(param_groups, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        LOGGER.info(
            f"{colorstr('optimizer:')} AdamW(lr={lr:g}, backbone_lr={lr * backbone_lr_multiplier:g}, "
            f"momentum={momentum:g}) with parameter groups " + ", ".join(group_summary)
        )
        return optimizer

    def get_validator(self):
        self.loss_names = ("cls_loss", "mask_loss", "dice_loss")
        model = unwrap_model(self.model)
        head = model.model[-1]
        if getattr(head, "point_rend_enabled", False) and "point_loss" not in self.loss_names:
            self.loss_names = (*self.loss_names, "point_loss")
        return Mask2FormerValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
