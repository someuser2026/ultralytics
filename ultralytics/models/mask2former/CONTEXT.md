# Mask2Former instance segmentation

The `Mask2Former` wrapper uses `SegmentationModel` with family-specific training, prediction, and validation adapters.

| Concern | Entry point |
| --- | --- |
| Public wrapper/task map | `model.py` |
| Optimizer groups, backbone LR multiplier, validator/loss labels | `train.py` |
| Native output unpacking, query/class selection, mask rescoring/boxes, refinement | `postprocess.py` |
| Prediction and validation integration | `predict.py`, `val.py` |

Native fields are `pred_logits` and `pred_masks`; the unpacker also accepts their two-tensor exported form. Query/class selection happens before mask resizing, thresholding, and mask-quality rescoring. Preserve this family pipeline when changing inference; generic YOLO NMS is not its selection step.

The active head is [../../nn/modules/mask2former.py](../../nn/modules/mask2former.py). `SegmentationModel.init_criterion` selects `Mask2FormerInstanceLoss` in [../../utils/loss.py](../../utils/loss.py); several other Mask2Former-named classes coexist there. For a non-default `backbone_lr_multiplier`, the trainer requires AdamW and groups backbone parameters from `model[0]`.

Configurations: [../../cfg/models/transformer/](../../cfg/models/transformer/). Checks: [Mask2Former](../../../tests/test_mask2former.py), [PointRend](../../../tests/test_pointrend.py), [PointRend YAML wiring](../../../tests/test_pointrend_yaml.py).
