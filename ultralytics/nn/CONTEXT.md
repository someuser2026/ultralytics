# Network construction and execution

This folder turns configuration into executable tensor models. Public family adapters live under [../models/CONTEXT.md](../models/CONTEXT.md).

| Question | Entry point |
| --- | --- |
| YAML resolution, parsing, channel/repeat handling, task inference | [tasks.py](tasks.py): `yaml_model_load`, `parse_model`, `guess_model_task` |
| Forward/loss dispatch and criterion selection | `tasks.py`: `BaseModel` and the selected task/family model class |
| Building blocks and head implementations | [modules/CONTEXT.md](modules/CONTEXT.md) |
| Native/exported inference backends, input/output adaptation | [autobackend.py](autobackend.py) |
| Text encoders for text-conditioned models | [text_model.py](text_model.py) |

For a YAML-visible module, trace its export in `modules/__init__.py`, import into `tasks.py`, and matching `parse_model` branch. A class definition alone does not establish that its YAML arguments or feature channels are wired correctly.

Choose the concrete model class before inspecting loss code: task `init_criterion` methods select criteria, while RCNN models delegate losses to their heads. Family-specific batch/output contracts belong to their family contexts. Head behavior can differ between construction probes, training, evaluation, and export; verify the relevant mode.

Related checks: [../../tests/test_dict_outputs.py](../../tests/test_dict_outputs.py), [../../tests/test_benchmark_model_configs.py](../../tests/test_benchmark_model_configs.py), and the selected family tests.
