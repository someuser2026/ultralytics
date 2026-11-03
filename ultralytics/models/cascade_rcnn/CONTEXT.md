# Cascade helper routing

Current Cascade Mask R-CNN models use [../rcnn/CONTEXT.md](../rcnn/CONTEXT.md). The local `model.py`, `train.py`, `predict.py`, and `val.py` are empty placeholders, not active adapters.

`targets.py` contains RPN/stage/mask target and ROI-sampling helpers. Inspect it only when a caller or the requested comparison points here; its presence does not establish that the active cascade head uses it. Follow the RCNN guide for current implementation, configurations, and tests.
