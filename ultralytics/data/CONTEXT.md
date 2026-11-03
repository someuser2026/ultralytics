# Data loading and transformation

This folder owns runtime image/label preparation, not the external dataset-generation pipeline.

| Need | Entry point |
| --- | --- |
| Dataset/dataloader construction and worker seeding | `build.py`: `build_yolo_dataset`, `build_dataloader`, `seed_worker` |
| Image loading/caching and dataset iteration | `base.py` |
| Label cache, metadata sidecars, transform assembly, batch collation | `dataset.py`: `YOLODataset` |
| Augmentation, auxiliary fields, input-band selection, tensor formatting | `augment.py`: `v8_transforms`, `PrepareAuxiliaryMaskInputs`, `SelectInputBands`, `Format` |
| Dataset validation, band/channel semantics, spatial transforms, metadata encoding | `utils.py` |
| Prediction input sources | `loaders.py`, `build.py:load_inference_source` |
| Conversion, annotation, splitting, download utilities | `converter.py`, `annotator.py`, `split.py`, `split_dota.py`, `scripts/` |

`channels` describes loaded imagery; explicit `input_bands` uses unique 1-based indices and determines `input_channels`. Auxiliary fields are prepared before band selection. Keep their spatial transforms aligned with imagery and labels. Metadata-enabled data requires the configured JSON sidecars and produces `metadata_vec`.

`Format` produces CHW images and task-dependent targets; collation stacks images and offsets `batch_idx`. OBB formatting normalizes center/size while retaining the angle convention. Mask layout depends on `overlap_mask`. Do not assume every family uses the generic transform chain: RT-DETR and RHINO have local dataset behavior in [../models/CONTEXT.md](../models/CONTEXT.md).

Checks: [dataloader seeds](../../tests/test_dataloader_seed.py), [padding](../../tests/test_augmentation_padding.py), [OBB auxiliary masks](../../tests/test_obb_aux_masks.py), [metadata](../../tests/test_metadata_modulation.py).
