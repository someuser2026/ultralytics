# YOLO task adapters

[model.py](model.py) owns the YOLO wrapper and `task_map`. Follow the resolved family from the parent guide if construction switches wrappers.

| Task / concern | Route |
| --- | --- |
| OBB training, validation, prediction | `obb/train.py`, `obb/val.py`, `obb/predict.py` |
| Instance segmentation | `segment/train.py`, `segment/val.py`, `segment/predict.py` |
| Detection and shared adapter bases | `detect/` |
| Classification, pose, text/prompt variants | `classify/`, `pose/`, `world/`, `yoloe/` only as requested |

OBB and segmentation adapters extend detection behavior. Read the relevant override before following its superclass. OBB training propagates the model YAML's `angle_mode` and distinguishes rotated FCOS loss labels from YOLO OBB labels. For segmentation, follow the selected head's mask/refinement behavior rather than assuming all segmentation outputs share one layout.

Custom backbones such as LEGNet and Mamba can use these adapters; their layers are routed through [../../nn/modules/CONTEXT.md](../../nn/modules/CONTEXT.md). Tensor-model and criterion selection are in [../../nn/CONTEXT.md](../../nn/CONTEXT.md); common batch preparation is in [../../data/CONTEXT.md](../../data/CONTEXT.md).

Checks by affected feature: [rotated FCOS](../../../tests/test_rotated_fcos.py), [LEGNet](../../../tests/test_legnet.py), [YOLO26](../../../tests/test_yolo26.py), [OBB auxiliary masks](../../../tests/test_obb_aux_masks.py), [PointRend](../../../tests/test_pointrend.py).
