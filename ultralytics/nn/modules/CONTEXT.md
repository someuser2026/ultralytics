# Layer and head routing

Select a file from the configured class/head; inspect the required class rather than reading every module.

| Component | Files |
| --- | --- |
| General convolution, blocks, activations, attention | `conv.py`, `block.py`, `activation.py`, `transformer.py` |
| Feature pyramids and metadata-conditioned necks | `neck.py`; RCNN backbone adapters in `rcnn_backbone.py` |
| YOLO task heads, rotated FCOS, RT-DETR decoders | `head.py` |
| Native Mask/Cascade Mask/rotated/oriented R-CNN heads and losses | `rcnn.py` |
| RHINO decoder | `rhino.py` |
| Active Mask2Former head | `mask2former.py`; `head.py` rebinds its same-named head to this implementation |
| LEGNet blocks | `legnet.py` |
| Mamba blocks, selective scan, HRNet fusion | `mamba_yolo.py`, `common_utils_mbyolo.py`, `hrvmamba.py` |
| PointRend refinement | `pointrend.py`, `pointrend_rcnn.py` |
| LoRA adapters | `lora.py` |

Follow active imports before using older `head_cascade.py`, `head_cascade_mask.py`, `roi.py`, or `rpn.py` helpers. Export/parser wiring belongs to [../CONTEXT.md](../CONTEXT.md); family target/output contracts belong to [../../models/CONTEXT.md](../../models/CONTEXT.md).

Check feature order, channels, stride, and tensor layout at each caller boundary. Mamba CPU fallbacks used for construction probes do not establish support for normal CPU training/inference. Extension issues route to [../../../selective_scan/](../../../selective_scan/) or [../../../dcnv4/](../../../dcnv4/) after identifying the failing import.

Module and family regression tests are indexed in [../../../tests/CONTEXT.md](../../../tests/CONTEXT.md).
