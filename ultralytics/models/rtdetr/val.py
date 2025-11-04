# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.metrics import SegmentMetrics, OBBMetrics
from ultralytics.utils.nms import TorchNMS
import numpy as np

__all__ = ("RTDETRValidator", "RTDETRSegmentValidator", "RTDETROBBValidator")  # tuple or list


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
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                return_obb=self.use_obb,
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
        )

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
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)  # filter
            # Sort by confidence to correctly get internal metrics
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred[score > self.args.conf]

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


class RTDETRSegmentValidator(RTDETRValidator):
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

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize RTDETRSegmentValidator and set task to 'segment', metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (namespace, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "segment"
        self.metrics = SegmentMetrics(fitness_weights=getattr(self.args, "fitness_weights", None))

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess batch of images for RT-DETR segmentation validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        batch = super().preprocess(batch)
        if "masks" in batch:
            batch["masks"] = batch["masks"].float()
        return batch

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
        # Extract dimensions
        nm = 32  # default number of masks
        if hasattr(self.model, "model") and hasattr(self.model.model[-1], "nm"):
            nm = self.model.model[-1].nm
        nc = len(self.model.names) if hasattr(self.model, "names") else nd - 4 - nm

        bboxes = preds[0][..., :4] * self.args.imgsz
        scores = preds[0][..., 4 : 4 + nc]
        mask_coeffs = preds[0][..., 4 + nc :]

        outputs = [torch.zeros((0, 7 + nm), device=bboxes.device)] * bs
        for i in range(bs):
            bbox = ops.xywh2xyxy(bboxes[i])
            score, cls = scores[i].max(-1)
            pred = torch.cat([bbox, score[..., None], cls[..., None], mask_coeffs[i]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred[score > self.args.conf]

        # Process masks with protos if available
        results = []
        for i, pred in enumerate(outputs):
            result = {"bboxes": pred[:, :4], "conf": pred[:, 4], "cls": pred[:, 5]}
            if protos is not None and pred.shape[0] > 0:
                # Process masks similar to SegmentationValidator
                masks = ops.process_mask(
                    protos[i : i + 1], pred[:, 6:], pred[:, :4], (self.args.imgsz, self.args.imgsz), upsample=True
                )
                result["masks"] = masks
            else:
                result["masks"] = None
            results.append(result)

        return results


class RTDETROBBValidator(RTDETRValidator):
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

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize RTDETROBBValidator and set task to 'obb', metrics to OBBMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (namespace, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics(fitness_weights=getattr(self.args, "fitness_weights", None))

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for RT-DETR OBB validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # check if dataset is DOTA format
        if hasattr(self, "confusion_matrix"):
            self.confusion_matrix.task = "obb"  # set confusion matrix task to 'obb'

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """
        Compute the correct prediction matrix for a batch of OBB detections and ground truth boxes.

        Args:
            preds (dict[str, torch.Tensor]): Prediction dictionary containing 'cls' and 'bboxes' keys with detected
                class labels and rotated bounding boxes (5D).
            batch (dict[str, torch.Tensor]): Batch dictionary containing 'cls' and 'bboxes' keys with ground truth
                class labels and rotated bounding boxes (5D).

        Returns:
            (dict[str, np.ndarray]): Dictionary containing 'tp' key with the correct prediction matrix.
        """
        from ultralytics.utils.metrics import batch_probiou

        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {
                "tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool),
                "matched_gt_idx": np.full(preds["cls"].shape[0], -1, dtype=np.int32),
            }
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        tp, matched_gt_idx = self.match_predictions(
            preds["cls"], batch["cls"], iou, return_matched_indices=True
        )

        return {
            "tp": tp.cpu().numpy(),
            "matched_gt_idx": matched_gt_idx,
        }

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
        rboxes = preds[0][..., :5] * self.args.imgsz
        scores = preds[0][..., 5:]
        outputs = [torch.zeros((0, 7), device=rboxes.device)] * bs
        for i in range(bs):
            score, cls = scores[i].max(-1)
            pred = torch.cat([rboxes[i], score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred[score > self.args.conf]

        return [{"bboxes": x[:, :5], "conf": x[:, 5], "cls": x[:, 6]} for x in outputs]
