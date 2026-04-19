# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.rtdetr.val import RTDETROBBValidator


class RHINOOBBValidator(RTDETROBBValidator):
    """RHINO validator reusing the RT-DETR OBB dataset and metrics path."""

