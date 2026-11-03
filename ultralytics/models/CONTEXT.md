# Model-family routing

Family adapters connect the public model interface to task-specific trainers, validators, and predictors. Shared lifecycle code belongs to [../engine/CONTEXT.md](../engine/CONTEXT.md); tensor models and layers belong to [../nn/CONTEXT.md](../nn/CONTEXT.md).

| Family / task | Local guide |
| --- | --- |
| YOLO, including OBB/segment adapters used by custom backbones and FCOS | [yolo/CONTEXT.md](yolo/CONTEXT.md) |
| Mask/Cascade Mask R-CNN and rotated/oriented R-CNN | [rcnn/CONTEXT.md](rcnn/CONTEXT.md) |
| RHINO OBB | [rhino/CONTEXT.md](rhino/CONTEXT.md) |
| RT-DETR detection, segmentation, OBB | [rtdetr/CONTEXT.md](rtdetr/CONTEXT.md) |
| Mask2Former instance segmentation | [mask2former/CONTEXT.md](mask2former/CONTEXT.md) |
| Older cascade helper files | [cascade_rcnn/CONTEXT.md](cascade_rcnn/CONTEXT.md); current cascade models use `rcnn/` |

`yolo/model.py` can switch a constructed model to another family based on its head. Identify the resolved wrapper and its `task_map` before following generic YOLO adapters.

`sam/`, `fastsam/`, and `nas/` are separate family implementations. `utils/` contains shared model helpers, including DETR utilities; follow actual imports because similarly named loss implementations also exist under `../utils/`. Architecture locations are indexed separately in [../cfg/models/CONTEXT.md](../cfg/models/CONTEXT.md).
