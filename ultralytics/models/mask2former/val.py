"""Reference-inference validator for Mask2Former instance segmentation."""

from __future__ import annotations

from ultralytics.models.yolo.segment.predict import _segment_head
from ultralytics.models.yolo.segment.val import SegmentationValidator

from .postprocess import (
    finalize_mask2former_instances,
    refine_selected_masks,
    select_mask2former_instances,
    unpack_mask2former_outputs,
)


class Mask2FormerValidator(SegmentationValidator):
    """Feed reference Mask2Former instances into the standard segmentation metrics pipeline."""

    def postprocess(self, preds):
        preds = unpack_mask2former_outputs(preds)
        head = getattr(self, "segment_head", None)
        if head is None and getattr(self, "model", None) is not None:
            head = _segment_head(self.model)
        head_limit = int(getattr(head, "max_per_image", 100))
        max_per_image = min(head_limit, int(self.args.max_det))
        selections = select_mask2former_instances(
            preds["pred_logits"], preds["pred_masks"], self.nc, max_per_image
        )
        point_features = preds.get("pointrend_features")
        outputs = []
        for i, selection in enumerate(selections):
            mask_logits = refine_selected_masks(
                head, selection, point_features, i, tuple(self._last_imgsz)
            )
            outputs.append(
                finalize_mask2former_instances(
                    selection, mask_logits, float(getattr(head, "mask_threshold", 0.5))
                )
            )
        return outputs
