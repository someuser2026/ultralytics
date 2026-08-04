# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.models.rtdetr.val import RTDETROBBValidator
from ultralytics.utils import colorstr

from .dataset import RHINODataset
from .postprocess import rhino_postprocess


class RHINOOBBValidator(RTDETROBBValidator):
    """RHINO validator with an isolated dataset and reference top-k postprocessing."""

    _VALIDATION_CONTEXT_KEY = "_rhino_inference_context"

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build the RHINO-only validation dataset."""
        return RHINODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task=self.args.task,
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics and record whether the backend can receive RHINO mask metadata."""
        super().init_metrics(model)
        is_backend = hasattr(model, "pt") or hasattr(model, "nn_module")
        self._rhino_mask_transport_supported = not is_backend or bool(
            getattr(model, "pt", False) or getattr(model, "nn_module", False)
        )

    def preprocess(self, batch: dict) -> dict:
        """Move tensors to the validation device and attach the RHINO-only inference context."""
        batch = super().preprocess(batch)
        image_shapes = batch.get("img_shapes")
        if image_shapes is None:
            height, width = batch["img"].shape[-2:]
            image_shapes = torch.tensor(
                [[height, width]] * batch["img"].shape[0],
                dtype=torch.long,
                device=batch["img"].device,
            )
            batch["img_shapes"] = image_shapes
        self._rhino_image_shapes = [
            (int(shape[0]), int(shape[1]))
            for shape in torch.as_tensor(image_shapes).detach().cpu()
        ]

        padding_mask = batch.get("padding_mask")
        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch["img"].shape[0], *batch["img"].shape[-2:]),
                dtype=torch.bool,
                device=batch["img"].device,
            )
            batch["padding_mask"] = padding_mask
        if padding_mask.any() and not getattr(self, "_rhino_mask_transport_supported", True):
            raise RuntimeError(
                "RHINO padded validation batches require a PyTorch backend because exported graphs "
                "do not expose the RHINO padding-mask input."
            )

        batch["metadata_vec"] = {
            self._VALIDATION_CONTEXT_KEY: True,
            "metadata_vec": batch.get("metadata_vec"),
            "padding_mask": padding_mask,
            "img_shapes": image_shapes,
        }
        return batch

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """Apply NMS-free flattened multiclass RHINO top-k selection."""
        predictions = preds[0] if isinstance(preds, (list, tuple)) else preds
        image_shapes = getattr(self, "_rhino_image_shapes", None)
        if image_shapes is None:
            imgsz = self.args.imgsz
            image_shape = tuple(imgsz) if isinstance(imgsz, (tuple, list)) else (imgsz, imgsz)
            image_shapes = [image_shape] * predictions.shape[0]
        max_candidates = int(getattr(getattr(self, "rtdetr_head", None), "max_candidates", 500))
        return rhino_postprocess(
            predictions,
            image_shapes,
            conf=self.args.conf,
            classes=self.args.classes,
            max_candidates=max_candidates,
        )

    def _prepare_batch(self, si: int, batch: dict) -> dict:
        """Prepare RHINO targets using the valid pre-padding image shape."""
        selected = batch["batch_idx"] == si
        classes = batch["cls"][selected].squeeze(-1)
        boxes = batch["bboxes"][selected].clone()
        valid_shape_tensor = torch.as_tensor(batch["img_shapes"][si], device=self.device)
        valid_shape = (int(valid_shape_tensor[0]), int(valid_shape_tensor[1]))
        if classes.shape[0]:
            boxes[..., :4] *= valid_shape_tensor[[1, 0, 1, 0]]
            areas = boxes[..., 2] * boxes[..., 3]
        else:
            areas = torch.zeros(0, device=self.device)
        return {
            "cls": classes,
            "bboxes": boxes,
            "areas": areas,
            "ori_shape": batch["ori_shape"][si],
            "imgsz": valid_shape,
            "ratio_pad": batch["ratio_pad"][si],
            "im_file": batch["im_file"][si],
        }
