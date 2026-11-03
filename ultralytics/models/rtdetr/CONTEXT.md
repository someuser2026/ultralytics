# RT-DETR adapters

[model.py](model.py) maps `detect`, `segment`, and `obb` to separate tensor models and adapters. Start with the class for the requested task.

| Concern | Entry point |
| --- | --- |
| Trainer, dataset/model construction, loss labels | `train.py`: `RTDETRTrainer`, `RTDETRSegmentTrainer`, `RTDETROBBTrainer` |
| Dataset transforms and validation | `val.py`: `RTDETRDataset` plus task validators |
| Task-specific decoding and results | `predict.py`: detection, segment, and OBB predictors |

The dataset class lives in `val.py` but is also used for training. It assembles its own transform chain; generic YOLO preprocessing is not a complete account of RT-DETR inputs.

The `RTDETR*Model` classes in [../../nn/tasks.py](../../nn/tasks.py) build grouped targets, split denoising queries, and connect encoder/decoder predictions to criteria. The detection loss optimizes more terms than its three displayed loss items. Decoders live in [../../nn/modules/head.py](../../nn/modules/head.py); follow the concrete model's criterion import into [../../utils/CONTEXT.md](../../utils/CONTEXT.md).

RHINO-specific formatting and postprocessing belong to [../rhino/CONTEXT.md](../rhino/CONTEXT.md). Configurations are primarily in [../../cfg/models/rt-detr/](../../cfg/models/rt-detr/); the architecture guide also routes transformer variants. Checks: [RT-DETR variants](../../../tests/test_rtdetr_variants.py), [PointRend](../../../tests/test_pointrend.py), [dictionary outputs](../../../tests/test_dict_outputs.py).
