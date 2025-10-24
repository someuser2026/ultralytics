# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prediction interfaces for Cascade R-CNN models."""

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor


class CascadeRCNNPredictor(DetectionPredictor):
    """Detection predictor using the Cascade R-CNN head."""


class CascadeMaskRCNNPredictor(SegmentationPredictor):
    """Segmentation predictor using the Cascade Mask R-CNN head."""
