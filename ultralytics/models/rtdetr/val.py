# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, PrepareAuxiliaryMaskInputs, SelectInputBands, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.metrics import SegmentMetrics, OBBMetrics
from ultralytics.utils.nms import TorchNMS
import numpy as np

__all__ = ("RTDETRValidator", "RTDETRSegmentValidator", "RTDETROBBValidator")  # tuple or list


def _rtdetr_head(model):
    """Return the RT-DETR head module through optional backend/model wrappers."""
    module = model
    for _ in range(2):
        candidate = getattr(module, "model", None)
        if candidate is None:
            break
        module = candidate
    return module[-1] if hasattr(module, "__getitem__") else None


class RTDETRDataset(YOLODataset):
    """
    Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.

    Attributes:
        augment (bool): Whether to apply data augmentation.
        rect (bool): Whether to use rectangular training.
        use_segments (bool): Whether to use segmentation masks.
        use_keypoints (bool): Whether to use keypoint annotations.
        imgsz (int): Target image size for training.

    Methods:
        load_image: Load one image from dataset index.
        build_transforms: Build transformation pipeline for the dataset.

    Examples:
        Initialize an RT-DETR dataset
        >>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
        >>> image, hw = dataset.load_image(0)
    """

    def __init__(self, *args, data=None, **kwargs):
        """
        Initialize the RTDETRDataset class by inheriting from the YOLODataset class.

        This constructor sets up a dataset specifically optimized for the RT-DETR (Real-Time DEtection and TRacking)
        model, building upon the base YOLODataset functionality.

        Args:
            *args (Any): Variable length argument list passed to the parent YOLODataset class.
            data (dict | None): Dictionary containing dataset information. If None, default values will be used.
            **kwargs (Any): Additional keyword arguments passed to the parent YOLODataset class.
        """
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        """
        Load one image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool, optional): Whether to use rectangular mode for batch inference.

        Returns:
            im (torch.Tensor): The loaded image.
            resized_hw (tuple): Height and width of the resized image with shape (2,).

        Examples:
            Load an image from the dataset
            >>> dataset = RTDETRDataset(img_path="path/to/images")
            >>> image, hw = dataset.load_image(0)
        """
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """
        Build transformation pipeline for the dataset.

        Args:
            hyp (dict, optional): Hyperparameters for transformations.

        Returns:
            (Compose): Composition of transformation functions.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scale_fill=True)])
            transforms = Compose([])
        transforms.append(
            PrepareAuxiliaryMaskInputs(
                bands=self.data.get("bands", {}),
                band_scale_factors=self.data.get("band_scale_factors", {}),
                use_shoreline_prior_loss=bool(getattr(hyp, "use_shoreline_prior_loss", False)),
                use_land_water_prior_loss=bool(getattr(hyp, "use_land_water_prior_loss", False)),
                shoreline_prior_max_dist=int(getattr(hyp, "shoreline_prior_max_dist", 128)),
            )
        )
        transforms.append(SelectInputBands(self.data.get("input_bands")))
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                channel_scale_factors=self.data.get(
                    "input_band_scale_factors", self.data.get("band_scale_factors", {})
                ),
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                return_obb=self.use_obb,
                bgr=1.0 if self.data.get("input_bands_explicit", False) else 0.0,
            )
        )
        return transforms


class RTDETRValidator(DetectionValidator):
    """
    RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Attributes:
        args (Namespace): Configuration arguments for validation.
        data (dict): Dataset configuration dictionary.

    Methods:
        build_dataset: Build an RTDETR Dataset for validation.
        postprocess: Apply Non-maximum suppression to prediction outputs.

    Examples:
        Initialize and run RT-DETR validation
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    Notes:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str, optional): `train` mode or `val` mode, users are able to customize different augmentations for
                each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (RTDETRDataset): Dataset configured for RT-DETR validation.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
            task=self.args.task,
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize validator metrics and cache the RT-DETR head for task-specific postprocessing."""
        super().init_metrics(model)
        self.rtdetr_head = _rtdetr_head(model)

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor | list | tuple): Raw predictions from the model. If tensor, should have shape
                (batch_size, num_predictions, num_classes + 4) where last dimension contains bbox coords and class scores.

        Returns:
            (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing:
                - 'bboxes': Tensor of shape (N, 4) with bounding box coordinates
                - 'conf': Tensor of shape (N,) with confidence scores
                - 'cls': Tensor of shape (N,) with class indices
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            keep = score > self.args.conf
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)[keep]
            outputs[i] = pred[pred[:, 4].argsort(descending=True)]

        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys
                with bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        box = ops.xyxy2xywh(box)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )


class RTDETRSegmentValidator(SegmentationValidator, RTDETRValidator):
    """
    RTDETRSegmentValidator extends RTDETRValidator to provide validation capabilities for RT-DETR segmentation models.

    The class handles both bounding box and mask predictions, processing segmentation detections and updating
    evaluation metrics accordingly.

    Attributes:
        args (Namespace): Configuration arguments for validation.
        data (dict): Dataset configuration dictionary.
        metrics (SegmentMetrics): Metrics calculator for segmentation tasks.

    Methods:
        postprocess: Apply Non-maximum suppression and process segmentation predictions.
        preprocess: Preprocess batch to ensure masks are float type.
        init_metrics: Initialize SegmentMetrics for validation.

    Examples:
        Initialize and run RT-DETR segmentation validation
        >>> from ultralytics.models.rtdetr import RTDETRSegmentValidator
        >>> args = dict(model="rtdetr-l-seg.pt", data="coco8-seg.yaml")
        >>> validator = RTDETRSegmentValidator(args=args)
        >>> validator()
    """

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression and process segmentation predictions.

        Args:
            preds (torch.Tensor | list | tuple): Raw predictions from the model. For segmentation, expects
                (predictions, extra) where extra contains protos.

        Returns:
            (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing:
                - 'bboxes': Tensor of shape (N, 4) with bounding box coordinates
                - 'conf': Tensor of shape (N,) with confidence scores
                - 'cls': Tensor of shape (N,) with class indices
                - 'masks': Tensor of shape (N, H, W) with segmentation masks
        """
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        # Extract protos if available
        if isinstance(preds[1], (list, tuple)) and len(preds[1]) > 6:
            protos = preds[1][6]
        elif preds[1] is not None:
            protos = preds[1] if isinstance(preds[1], torch.Tensor) else None
        else:
            protos = None

        bs, _, nd = preds[0].shape
        nm = 32
        head = getattr(self, "rtdetr_head", None)
        if head is not None and hasattr(head, "nm"):
            nm = head.nm
        nc = len(self.names) if self.names is not None else nd - 4 - nm
        imgsz = self._last_imgsz
        scale = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], device=preds[0].device, dtype=preds[0].dtype)

        bboxes = preds[0][..., :4] * scale
        scores = preds[0][..., 4 : 4 + nc]
        mask_coeffs = preds[0][..., 4 + nc :]
        results = []
        for i in range(bs):
            bbox = ops.xywh2xyxy(bboxes[i])
            score, cls = scores[i].max(-1)
            pred = torch.cat([bbox, score[..., None], cls[..., None], mask_coeffs[i]], dim=-1)
            pred = pred[score > self.args.conf]
            pred = pred[pred[:, 4].argsort(descending=True)]
            proto_i = None if protos is None else (protos if protos.ndim == 3 else protos[i])
            masks = (
                self.process(proto_i, pred[:, 6:], pred[:, :4], shape=imgsz)
                if protos is not None and pred.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native or proto_i is None else proto_i.shape[1:])),
                    dtype=torch.uint8,
                    device=pred.device,
                )
            )
            results.append({"bboxes": pred[:, :4], "conf": pred[:, 4], "cls": pred[:, 5], "masks": masks})

        return results


class RTDETROBBValidator(OBBValidator, RTDETRValidator):
    """
    RTDETROBBValidator extends RTDETRValidator to provide validation capabilities for RT-DETR OBB models.

    The class handles rotated bounding box predictions, processing OBB detections and updating evaluation
    metrics accordingly.

    Attributes:
        args (Namespace): Configuration arguments for validation.
        data (dict): Dataset configuration dictionary.
        metrics (OBBMetrics): Metrics calculator for OBB tasks.
        is_dota (bool): Flag indicating whether the validation dataset is in DOTA format.

    Methods:
        postprocess: Apply Non-maximum suppression and process OBB predictions.
        _process_batch: Process batch with OBB-specific IoU calculation.
        init_metrics: Initialize OBBMetrics for validation.

    Examples:
        Initialize and run RT-DETR OBB validation
        >>> from ultralytics.models.rtdetr import RTDETROBBValidator
        >>> args = dict(model="rtdetr-l-obb.pt", data="dota8.yaml")
        >>> validator = RTDETROBBValidator(args=args)
        >>> validator()
    """

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression to OBB prediction outputs.

        Args:
            preds (torch.Tensor | list | tuple): Raw predictions from the model. Should have shape
                (batch_size, num_predictions, num_classes + 5) where last dimension contains rbox coords (5D) and class scores.

        Returns:
            (list[dict[str, torch.Tensor]]): List of dictionaries for each image, each containing:
                - 'bboxes': Tensor of shape (N, 5) with rotated bounding box coordinates (x, y, w, h, angle)
                - 'conf': Tensor of shape (N,) with confidence scores
                - 'cls': Tensor of shape (N,) with class indices
        """
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        imgsz = self.args.imgsz if isinstance(self.args.imgsz, (tuple, list)) else (self.args.imgsz, self.args.imgsz)
        scale = torch.tensor([imgsz[1], imgsz[0], imgsz[1], imgsz[0]], device=preds[0].device, dtype=preds[0].dtype)
        rboxes = preds[0][..., :5].clone()
        rboxes[..., :4] *= scale
        scores = preds[0][..., 5:]
        outputs = [torch.zeros((0, 7), device=rboxes.device)] * bs
        for i in range(bs):
            score, cls = scores[i].max(-1)
            pred_rboxes = ops.regularize_rboxes(rboxes[i])
            pred = torch.cat([pred_rboxes, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score > self.args.conf]
            outputs[i] = pred[pred[:, 5].argsort(descending=True)]

        return [{"bboxes": x[:, :5], "conf": x[:, 5], "cls": x[:, 6]} for x in outputs]
