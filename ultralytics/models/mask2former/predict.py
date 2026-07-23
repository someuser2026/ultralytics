"""Ultralytics Results adapter for native Mask2Former inference."""

from __future__ import annotations

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor, _segment_head
from ultralytics.utils import DEFAULT_CFG, ops

from .postprocess import (
    finalize_mask2former_instances,
    refine_selected_masks,
    select_mask2former_instances,
    unpack_mask2former_outputs,
)


class Mask2FormerPredictor(SegmentationPredictor):
    """Run reference Mask2Former instance inference and return standard Ultralytics Results."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        preds = unpack_mask2former_outputs(preds)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        head = _segment_head(self.model)
        head_limit = int(getattr(head, "max_per_image", 100))
        max_per_image = min(head_limit, int(self.args.max_det))
        selections = select_mask2former_instances(
            preds["pred_logits"], preds["pred_masks"], len(self.model.names), max_per_image
        )
        point_features = preds.get("pointrend_features")
        results = []
        for i, (selection, orig_img, img_path) in enumerate(zip(selections, orig_imgs, self.batch[0])):
            mask_logits = refine_selected_masks(head, selection, point_features, i, tuple(img.shape[2:]))
            mask_logits = ops.scale_masks(mask_logits[:, None], orig_img.shape[:2], padding=True)[:, 0]
            pred = finalize_mask2former_instances(
                selection, mask_logits, float(getattr(head, "mask_threshold", 0.5))
            )
            boxes = torch.cat((pred["bboxes"], pred["conf"][:, None], pred["cls"][:, None]), dim=1)
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=boxes, masks=pred["masks"])
            )
        return results
