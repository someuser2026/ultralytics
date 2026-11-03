# Runtime routing

The public API selects a model family; that family connects the shared engine to model construction, data preparation, and task-specific outputs. Use the selected route below before opening implementation files.

| Question | Route |
| --- | --- |
| Architecture/dataset YAML, experiment settings, CLI/defaults | [cfg/CONTEXT.md](cfg/CONTEXT.md) |
| Which model, trainer, validator, or predictor handles this family? | [models/CONTEXT.md](models/CONTEXT.md) |
| YAML parsing, model construction, backend, layer implementation | [nn/CONTEXT.md](nn/CONTEXT.md) |
| Shared training/resume, validation/prediction lifecycle, export, results | [engine/CONTEXT.md](engine/CONTEXT.md) |
| Image/label loading, transforms, auxiliary inputs, batching | [data/CONTEXT.md](data/CONTEXT.md) |
| Losses, assignment, geometry, metrics, callbacks/logging | [utils/CONTEXT.md](utils/CONTEXT.md) |

`cfg/models/` contains architecture declarations; `models/` contains family/task adapters; `nn/` constructs and executes networks. A family-specific error should start with its adapter, then follow the relevant shared implementation. A different backbone does not necessarily imply a different adapter family: check the selected head and task map.

Public exports live in [__init__.py](__init__.py). `hub/`, `solutions/`, and `trackers/` handle their named features; `assets/` contains sample media. Enter these only for the requested feature or a traced dependency. Test selection lives in [../tests/CONTEXT.md](../tests/CONTEXT.md).
