# Shared execution lifecycle

Use this folder for behavior shared across model families. Check the selected [family adapter](../models/CONTEXT.md) first when only one family/task is affected.

| Behavior | File / entry point |
| --- | --- |
| Public load/train/val/predict/export dispatch | [model.py](model.py): `Model` |
| Training setup/loop, optimization, checkpoint saving, resume | [trainer.py](trainer.py): `BaseTrainer`, `_setup_train`, `_do_train`, `check_resume`, `resume_training` |
| Validation lifecycle and metric aggregation | [validator.py](validator.py): `BaseValidator` |
| Prediction source setup, preprocessing, streaming, callbacks | [predictor.py](predictor.py): `BasePredictor` |
| User-facing boxes, masks, OBBs, serialization | [results.py](results.py): `Results` and its tensor wrappers |
| Format-specific model export | [exporter.py](exporter.py) |
| Hyperparameter tuning | [tuner.py](tuner.py) |

Family subclasses supply task-specific preprocessing, datasets, models, postprocessing, and validation behavior. Keep those responsibilities distinct from the shared loop. Resume restores training state; loading checkpoint weights for a new run is a different path.

Trace input/backend details through [../data/CONTEXT.md](../data/CONTEXT.md) and [../nn/CONTEXT.md](../nn/CONTEXT.md) only when involved. Logging integrations route through [../utils/CONTEXT.md](../utils/CONTEXT.md). Relevant checks: [../../tests/test_engine.py](../../tests/test_engine.py), [../../tests/test_dict_outputs.py](../../tests/test_dict_outputs.py), [../../tests/test_exports.py](../../tests/test_exports.py).
