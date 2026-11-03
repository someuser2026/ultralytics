# Configuration routing

Configuration declarations and CLI argument handling live here. Runtime interpretation belongs to the consuming model, data loader, or engine component.

| Need | Start here |
| --- | --- |
| Select or change an architecture/variant | [models/CONTEXT.md](models/CONTEXT.md) |
| Dataset paths, classes, bands, metadata declarations | `datasets/`; choose the YAML actually passed as `data` |
| Experiment-specific configuration | `experiments/`; inspect the requested experiment, not all model YAMLs |
| Default runtime arguments | [default.yaml](default.yaml) |
| CLI parsing, override merging, argument validation, save-directory selection | [__init__.py](__init__.py): `entrypoint`, `get_cfg`, `check_cfg`, `get_save_dir` |
| Tracker parameters | `trackers/` |

The `yolo` command is registered in [../../pyproject.toml](../../pyproject.toml). For a new runtime option, check its default, validation, and consumer together. Dataset YAMLs may be external to this repository; the supplied `data` path is the starting point, not a similarly named bundled example.

Dataset band/metadata validation and derived input channels are owned by [../data/CONTEXT.md](../data/CONTEXT.md). Model YAML parsing is owned by [../nn/CONTEXT.md](../nn/CONTEXT.md). Relevant CLI and architecture checks are indexed in [../../tests/CONTEXT.md](../../tests/CONTEXT.md).
