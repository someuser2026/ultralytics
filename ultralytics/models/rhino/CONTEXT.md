# RHINO OBB

RHINO has its own OBB wrapper and dataset boundary while extending RT-DETR adapters. Keep RHINO-specific changes scoped here or in its dedicated decoder/loss implementations.

| Concern | Entry point |
| --- | --- |
| Wrapper/task map | `model.py`: `RHINO` |
| Model, dataset, validator construction | `train.py` |
| Angle formatting and padded-batch construction | `dataset.py`: `RHINODataset` |
| Native prediction selection | `postprocess.py`: `rhino_postprocess` |
| Result conversion and metric integration | `predict.py`, `val.py` |

`RHINODataset` sets the final `Format` transform to `le90`. It pads spatial tensors at the bottom/right and supplies `padding_mask` (`True` for padding) and original `img_shapes`. Preserve these fields and the angle convention across model, loss, and postprocessing; do not change generic RT-DETR formatting to implement RHINO behavior.

Tensor-model assembly: `RHINOOBBModel` in [../../nn/tasks.py](../../nn/tasks.py). Decoder: [../../nn/modules/rhino.py](../../nn/modules/rhino.py). Matching, geometry costs, denoising assignment, and `RHINOOBBLoss`: [../../utils/rhino.py](../../utils/rhino.py). Follow [../rtdetr/CONTEXT.md](../rtdetr/CONTEXT.md) only for inherited behavior.

Configurations: [../../cfg/models/rhino/](../../cfg/models/rhino/). Checks: [variants](../../../tests/test_rhino_variants.py), [family isolation](../../../tests/test_rhino_isolation.py), [reference parity](../../../tests/test_rhino_reference_parity.py).
