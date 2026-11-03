# Architecture YAML routing

Choose the requested family, backbone, task, and variant before opening YAMLs. This is the architecture catalogue; adapter behavior belongs to [../../models/CONTEXT.md](../../models/CONTEXT.md).

| Architecture/configuration | Folder |
| --- | --- |
| Versioned YOLO models | `11/`, `12/`, `26/`, `v3/`, `v5/`, `v6/`, `v8/`, `v9/`, `v10/` |
| Additional YOLO task variants | `yolo/obb/`, `yolo/segment_no_p2/` |
| ConvNeXt/ConvNeXtV2 backbones | `convnext/`, `convnextv2/`, then `tiny/`, `base/`, or `large/` |
| timm backbone variants | `timm/obb/`, `timm/segment/` |
| LEGNet and rotated FCOS | `legnet/`, `fcos/` |
| Mamba-YOLO and Mamba/HRNet variants | `mamba-yolo/` |
| Mask/Cascade Mask R-CNN, rotated/oriented R-CNN | `rcnn/` |
| RHINO | `rhino/` |
| RT-DETR | `rt-detr/`; also inspect a specifically requested config in `transformer/` |
| Mask2Former and transformer segmentation variants | `transformer/` |

Suffixes such as `-obb`, `-seg`, `pointrend`, `shoreaux`, and `smallobj` narrow the search; confirm the head and arguments inside the YAML. Names alone do not establish equivalent behavior. Inspect `nc`, scaling, feature sources, and task-specific options before changing a variant.

For parser/registration changes, follow [../../nn/CONTEXT.md](../../nn/CONTEXT.md). For verification, use [../../../tests/test_benchmark_model_configs.py](../../../tests/test_benchmark_model_configs.py) and the selected family's tests from [../../../tests/CONTEXT.md](../../../tests/CONTEXT.md). The existing `README.md` is general usage guidance, not the custom-fork routing map.
