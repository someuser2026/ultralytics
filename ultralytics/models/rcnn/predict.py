# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.obb.predict import OBBPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, ops

_RCNN_MASK_THRESHOLD = 0.5


class RCNNSegmentationPredictor(SegmentationPredictor):
    """Prediction adapter for Mask R-CNN and Cascade Mask R-CNN."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        if isinstance(preds, dict):
            preds = [preds]
        return [self.construct_result(pred, img, orig_img, img_path) for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])]

    def construct_result(self, pred, img, orig_img, img_path):
        boxes = pred["bboxes"].clone()
        if boxes.numel():
            boxes = ops.scale_boxes(img.shape[2:], boxes, orig_img.shape)
            det = torch.cat((boxes, pred["conf"].unsqueeze(-1), pred["cls"].unsqueeze(-1)), dim=-1)
        else:
            det = boxes.new_zeros((0, 6))

        masks = pred.get("masks")
        if masks is not None and masks.numel():
            masks = ops.scale_image(
                masks.permute(1, 2, 0).contiguous().float().cpu().numpy(),
                orig_img.shape,
            )
            masks = torch.as_tensor(np.transpose(masks, (2, 0, 1))) > _RCNN_MASK_THRESHOLD
            keep = masks.flatten(1).any(dim=1)
            det = det[keep.to(det.device)]
            masks = masks[keep]
        else:
            det = det[:0]
            masks = det.new_zeros((0, orig_img.shape[0], orig_img.shape[1]), dtype=torch.bool)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=det, masks=masks)


class RCNNOBBPredictor(OBBPredictor):
    """Prediction adapter for rotated Faster R-CNN and Oriented R-CNN."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"

    def postprocess(self, preds, img, orig_imgs):
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            if pred["bboxes"].numel():
                tensor = torch.cat(
                    (
                        pred["bboxes"][:, :4],
                        pred["conf"].unsqueeze(-1),
                        pred["cls"].unsqueeze(-1),
                        pred["bboxes"][:, 4:5],
                    ),
                    dim=-1,
                )
            else:
                tensor = pred["bboxes"].new_zeros((0, 7))
            results.append(self.construct_result(tensor, img, orig_img, img_path))
        return results
