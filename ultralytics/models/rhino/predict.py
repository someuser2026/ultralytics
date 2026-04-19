# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.rtdetr.predict import RTDETROBBPredictor


class RHINOOBBPredictor(RTDETROBBPredictor):
    """RHINO OBB predictor reusing the RT-DETR OBB postprocess path."""

