# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models import yolo


class RCNNSegmentationValidator(yolo.segment.SegmentationValidator):
    """Validation adapter for native RCNN segmentation models."""

    def postprocess(self, preds):
        return preds if isinstance(preds, list) else [preds]


class RCNNOBBValidator(yolo.obb.OBBValidator):
    """Validation adapter for native RCNN OBB models."""

    def postprocess(self, preds):
        return preds if isinstance(preds, list) else [preds]
