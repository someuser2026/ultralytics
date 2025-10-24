# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validation hooks for Cascade R-CNN models."""

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator


class CascadeRCNNValidator(DetectionValidator):
    """Detection validator for Cascade R-CNN."""


class CascadeMaskRCNNValidator(SegmentationValidator):
    """Segmentation validator for Cascade Mask R-CNN."""
