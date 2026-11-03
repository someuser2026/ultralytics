# Native RCNN adapters

This is the active adapter for Mask R-CNN, Cascade Mask R-CNN, rotated Faster R-CNN, and Oriented R-CNN. Supported tasks are `segment` and `obb`.

| Concern | Entry point |
| --- | --- |
| Public wrapper and task map | `model.py`: `RCNN` |
| Model construction and task-specific loss labels | `train.py` |
| Prediction dictionaries to user-facing results | `predict.py` |
| Validation adaptation | `val.py` |

The tensor models in [../../nn/tasks.py](../../nn/tasks.py) require a **batch dictionary containing `img`**; direct tensor calls are unsupported. `_RCNNModel` dispatches loss/prediction, and the selected head computes training losses internally. Segmentation forces `overlap_mask=False` so targets remain per-instance.

Active heads, proposal assignment/sampling, ROI processing, cascade stages, and losses live in [../../nn/modules/rcnn.py](../../nn/modules/rcnn.py). PointRend-specific RCNN behavior lives in [../../nn/modules/pointrend_rcnn.py](../../nn/modules/pointrend_rcnn.py). Preserve per-image prediction dictionaries and trace mask/box scaling through the adapter before converting outputs.

Configurations: [../../cfg/models/rcnn/](../../cfg/models/rcnn/). Checks: [RCNN variants](../../../tests/test_rcnn_variants.py), [RCNN PointRend](../../../tests/test_pointrend_rcnn.py), [dictionary outputs](../../../tests/test_dict_outputs.py). Current cascade models are covered here; the sibling `cascade_rcnn/` adapter files are placeholders.
