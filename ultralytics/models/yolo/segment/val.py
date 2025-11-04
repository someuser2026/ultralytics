# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, mask_iou


class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    This validator handles the evaluation of segmentation models, processing both bounding box and mask predictions
    to compute metrics such as mAP for both detection and segmentation tasks.

    Attributes:
        plot_masks (list): List to store masks for plotting.
        process (callable): Function to process masks based on save_json and save_txt flags.
        args (namespace): Arguments for the validator.
        metrics (SegmentMetrics): Metrics calculator for segmentation tasks.
        stats (dict): Dictionary to store statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.segment import SegmentationValidator
        >>> args = dict(model="yolo11n-seg.pt", data="coco8-seg.yaml")
        >>> validator = SegmentationValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (namespace, optional): Arguments for the validator.
            _callbacks (list, optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.process = None
        self.args.task = "segment"
        # print("-"*50)
        # print("Inside segment/val.py 54")
        # print(self.args.fitness_weights)
        # print("-"*50)
        self.metrics = SegmentMetrics(fitness_weights = self.args.fitness_weights)
    
    def __call__(self, trainer=None, model=None):
        """Run validation."""
        # Reset aggregates at START of validation
        if hasattr(self.metrics, 'reset_mask_aggregates'):
            self.metrics.reset_mask_aggregates()
        return super().__call__(trainer, model)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Preprocess batch of images for YOLO segmentation validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].float()
        self._last_imgsz = tuple(batch["img"].shape[2:])
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize metrics and select mask processing function based on save_json flag.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        if self.args.save_json:
            check_requirements("faster-coco-eval>=1.6.7")
        # More accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask

    def get_desc(self) -> str:
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """
        Post-process YOLO predictions and return output detections with proto.

        Args:
            preds (list[torch.Tensor]): Raw predictions from the model.

        Returns:
            list[dict[str, torch.Tensor]]: Processed detection predictions with masks.
        """
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        preds = super().postprocess(preds[0])
        # imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto
        imgsz = self._last_imgsz
        for i, pred in enumerate(preds):
            # print("-"*50)
            # print("Inside postprocess function in segment/val.py 126")
            # print("imgsz", imgsz)
            # print("pred:", pred)
            # print("-"*50)
            coefficient = pred.pop("extra")
            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare a batch for training or inference by processing images and targets.

        Args:
            si (int): Batch index.
            batch (Dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (Dict[str, Any]): Prepared batch with processed annotations.
        """
        prepared_batch = super()._prepare_batch(si, batch)
        midx = [si] if self.args.overlap_mask else batch["batch_idx"] == si
        prepared_batch["masks"] = batch["masks"][midx]
        return prepared_batch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """
        Compute correct prediction matrix for a batch based on bounding boxes and optional masks.
        Also updates per-class aggregates for Dice, mIoU, and boundary-F1 from matched mask pairs.

        Notes:
            - Uses greedy one-to-one matching among (gt, pred) pairs with IoU >= first threshold.
            - Matching is deterministic and GPU-friendly (no Python sets in tight loops).
        """
        tp = super()._process_batch(preds, batch)
        gt_cls, gt_masks = batch["cls"], batch["masks"]

        # Early return if no ground truth or predictions
        if len(gt_cls) == 0 or len(preds["cls"]) == 0:
            tp_m = np.zeros((len(preds["cls"]), self.niou), dtype=bool)
            tp.update({"tp_m": tp_m})
            return tp

        pred_masks = preds["masks"]
        device = gt_masks.device
        
        # Handle overlap_mask case by expanding instance channels from an overlapped label mask
        if getattr(self.args, "overlap_mask", False):
            nl = len(gt_cls)
            index = torch.arange(nl, device=device).view(nl, 1, 1) + 1
            gt_masks = gt_masks.repeat(nl, 1, 1)  # (1,H,W) -> (nl,H,W)
            gt_masks = torch.where(gt_masks == index, 1.0, 0.0)

        # Shape alignment: resize GT to pred size if needed
        if gt_masks.shape[1:] != pred_masks.shape[1:]:
            gt_masks = F.interpolate(
                gt_masks[None].float(), 
                pred_masks.shape[1:], 
                mode="nearest", 
                align_corners=False
            )[0]
            gt_masks = gt_masks.gt_(0.5)
        
        # print("-"*50)
        # print("Inside _process_batch function in segment/val.py 196")
        # print("gt_masks.shape", gt_masks.shape)
        # print("pred_masks.shape", pred_masks.shape)
        # print("-"*50)

        # Compute mask IoU on flattened masks
        iou = mask_iou(
            gt_masks.reshape(gt_masks.shape[0], -1).float(),
            pred_masks.reshape(pred_masks.shape[0], -1).float()
        )

        # Regular mask mAP TPs (per thresholds)
        tp_m, _ = self.match_predictions(preds["cls"], gt_cls, iou)
        tp_m = tp_m.cpu().numpy()

        # -------- Aggregates: Dice / mIoU / Boundary-F1 (one-to-one greedy on first IoU threshold) --------
        with torch.no_grad():
            # Zero IoU for mismatched classes
            class_ok = (gt_cls[:, None] == preds["cls"][None])
            iou_c = iou * class_ok.float()  # Ensure float for proper masking
            
            # Get threshold (default to 0.5 if not available)
            thr = float(self.iouv[0].item()) if hasattr(self, "iouv") and self.iouv is not None else 0.5

            # Find all potential matches above threshold
            matches = torch.nonzero(iou_c >= thr, as_tuple=False)  # [K, 2] (g, p)
            
            if matches.numel() > 0:
                # Sort by IoU in descending order for greedy matching
                ious_flat = iou_c[matches[:, 0], matches[:, 1]]
                order = torch.argsort(ious_flat, descending=True)
                matches = matches[order]

                # Greedy unique assignment using boolean masks (no Python loops in critical path)
                taken_g = torch.zeros(gt_cls.shape[0], dtype=torch.bool, device=device)
                taken_p = torch.zeros(preds["cls"].shape[0], dtype=torch.bool, device=device)
                
                # Vectorized filtering of matches
                keep_mask = torch.zeros(matches.shape[0], dtype=torch.bool, device=device)
                for k in range(matches.shape[0]):
                    g, p = matches[k, 0].item(), matches[k, 1].item()
                    if not taken_g[g] and not taken_p[p]:
                        keep_mask[k] = True
                        taken_g[g] = True
                        taken_p[p] = True

                if keep_mask.any():
                    # Get matched indices
                    matched = matches[keep_mask]
                    g_idx = matched[:, 0]
                    p_idx = matched[:, 1]
                    cls_indices = gt_cls[g_idx].to(torch.long)

                    # Get matched masks (already aligned, convert to boolean)
                    gsel = gt_masks[g_idx].bool()
                    psel = pred_masks[p_idx].bool()

                    # Validate mask shapes before updating
                    if gsel.shape == psel.shape and gsel.numel() > 0:
                        # Update per-class aggregates
                        boundary_tolerance = getattr(self.args, "boundary_tolerance", 1)
                        self.metrics.update_mask_aggregates(
                            cls_indices, 
                            gsel, 
                            psel,
                            boundary_tolerance=boundary_tolerance
                        )

        tp.update({"tp_m": tp_m})
        return tp

    def plot_predictions(self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int) -> None:
        """
        Plot batch predictions with masks and bounding boxes.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
        """
        for p in preds:
            masks = p["masks"]
            if masks.shape[0] > self.args.max_det:
                LOGGER.warning(f"Limiting validation plots to 'max_det={self.args.max_det}' items.")
            p["masks"] = torch.as_tensor(masks[: self.args.max_det], dtype=torch.uint8).cpu()
        super().plot_predictions(batch, preds, ni, max_det=self.args.max_det)  # plot bboxes

    def save_one_txt(self, predn: torch.Tensor, save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            masks=torch.as_tensor(predn["masks"], dtype=torch.uint8),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """
        Save one JSON result for COCO evaluation.

        Args:
            predn (dict[str, torch.Tensor]): Predictions containing bboxes, masks, confidence scores, and classes.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """
        from faster_coco_eval.core.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        pred_masks = np.transpose(predn["masks"], (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pred_masks)
        super().pred_to_json(predn, pbatch)
        for i, r in enumerate(rles):
            self.jdict[-len(rles) + i]["segmentation"] = r  # segmentation

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **super().scale_preds(predn, pbatch),
            "masks": ops.scale_image(
                torch.as_tensor(predn["masks"], dtype=torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Return COCO-style instance segmentation evaluation metrics."""
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return super().coco_evaluate(stats, pred_json, anno_json, ["bbox", "segm"], suffix=["Box", "Mask"])
