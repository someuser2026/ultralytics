# Shared utilities and losses

For loss questions, first identify the selected model's criterion or head loss through [../nn/CONTEXT.md](../nn/CONTEXT.md). Similar class names do not guarantee the same active implementation.

| Need | File / folder |
| --- | --- |
| YOLO task losses, rotated FCOS, RT-DETR, Mask2Former criteria, spatial priors | `loss.py`; search the selected criterion class |
| RHINO matching, geometry costs, loss | `rhino.py` |
| Task-aligned assignment | `tal.py` |
| Metrics, IoU/probabilistic IoU | `metrics.py` |
| Box/mask conversion, scaling, angle conventions | `ops.py`; suppression in `nms.py` |
| Instance/segment container transformations | `instance.py` |
| W&B and other training integrations | `callbacks/`; W&B implementation in `callbacks/wb.py` |
| Plotting, device/distributed helpers, settings, export helpers | `plotting.py`, `torch_utils.py`, `dist.py`, `__init__.py`, `export/` as relevant |

Native RCNN losses live in their head implementation, not automatically in `loss_cascade.py`. DETR loss classes also exist in `../models/utils/loss.py`; follow the consumer's import. The active Mask2Former criterion is selected by `SegmentationModel.init_criterion`, not by choosing the first Mask2Former-named class in this file.

When changing a loss, trace assignment, coordinate/angle units, normalization, empty targets, and auxiliary branches together. Logging loss items may be only a subset of the optimized total. Relevant feature/family checks are indexed in [../../tests/CONTEXT.md](../../tests/CONTEXT.md).
