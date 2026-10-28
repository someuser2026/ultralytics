# Repo Context

Custom `ultralytics` fork for rip current detection on satellite imagery. Primary tasks: `obb` and `segment`. The active model runtime is under `ultralytics/`. Paths below are relative to the repository root unless absolute.

## Route Before Reading

1. Identify the requested task, model family, and any named file or error. Choose the narrowest relevant route below before reading implementation files.
2. Along that route, read applicable `AGENTS.md` guidance and existing `CONTEXT.md` files from parent to child, including intermediate folders even when starting from a named file. Read `CONTEXT.md` explicitly; do not assume it was automatically loaded.
3. Parent contexts route to children; child contexts supply local entry points, behavior, configs, tests, and data/output locations. Read only the selected branch, not sibling contexts or every file in a folder. Reuse context already read unless it changes.
4. If a selected folder lacks `CONTEXT.md`, report the missing path. Use the nearest parent guidance and a scoped filename/symbol search to proceed; do not create context files unless requested. Ask if the missing information leaves the task ambiguous.
5. Start with `rg --files <selected-folder>` or `rg -n '<symbol>' <selected-folder>`, narrowed by filename/glob where possible. Read the matching sections and relevant nearby tests/configs. Avoid repository-wide content searches and recursive file dumps as a first step.
6. Expand only when a relevant import, caller, config, test, or unsuccessful scoped search points elsewhere. Follow that dependency's context chain before reading its implementation. Stop expanding once the task and affected dependencies are understood.

## Task Routing

| Task / question | Start here |
| --- | --- |
| Model YAMLs, datasets, experiment settings, CLI defaults | `ultralytics/cfg/`: `models/` for architectures, `datasets/` for data, `experiments/` for experiments, `default.yaml` and `__init__.py` for defaults/CLI |
| Family/task-specific training, validation, prediction | `ultralytics/models/`; YOLO OBB and segmentation use `yolo/obb/` and `yolo/segment/`. For another family, choose its own subfolder. |
| Shared training loop, checkpoint/resume, prediction/validation lifecycle, export, result objects | `ultralytics/engine/` |
| Model construction, YAML parsing, task dispatch | `ultralytics/nn/tasks.py`; follow the selected family adapter in `ultralytics/models/` as needed |
| Backbones, necks, heads, attention, custom layers | `ultralytics/nn/modules/`; select by the model YAML or imported class |
| Losses, assignment, metrics, box/mask geometry, NMS | `ultralytics/utils/`; trace the selected model's criterion first, since some families keep helpers in `ultralytics/models/` |
| Dataset loading, labels, augmentation, preprocessing, batches | `ultralytics/data/`; dataset configuration lives in `ultralytics/cfg/datasets/` |
| Logging, W&B, integration callbacks | `ultralytics/utils/callbacks/`; job-specific prediction logging is in `jobs/infer/` |
| HPC training submission, PBS resources, batch/resume scripts | `jobs/train/hpc/` |
| Site inference, prediction JSONs, batch combination, inference PBS jobs | `jobs/infer/`; scheduler scripts in `jobs/infer/hpc/` |
| Regression tests and test fixtures | `tests/`; choose the feature/family test, then its fixtures and `tests/configs/` as needed |
| CUDA extension builds or extension import/runtime errors | `selective_scan/`, `dcnv4/`; first identify which extension the active module requires |
| Installation, dependencies, containers, usage examples, documentation | `pyproject.toml`, `docker/`, `examples/`, or `docs/`, according to the request |

`ultralytics/hub/`, `ultralytics/solutions/`, and `ultralytics/trackers/` are specialized areas; enter them only for those features or a traced dependency. Inspect `runs/`, `wandb/`, and `datasets/` only when the task needs particular outputs, logs, or data; keep them out of general source searches.

## Reference Sources

`mmdetection/`, `LEGNet/`, `RHINO/`, `Mamba-YOLO/`, and `detectron2/` are reference-only checkouts for comparison or porting into `ultralytics/`. They are not the normal model construction, training, or testing path unless explicitly stated. Start with the active implementation even when a model shares a reference repository's name. Other upstream trees such as `HRVMamba/` and `Mask2Former/` should likewise only be explored for a relevant comparison, port, or traced dependency.

## Environment and Checks

Use `ultralytics_contrib` unless told otherwise. Local interpreter: `/Users/manishagupta/miniforge3/envs/ultralytics_contrib/bin/python`.

```bash
conda activate ultralytics_contrib
```

Run the checks relevant to the selected route; these are examples, not a mandatory full suite:

```bash
pytest tests/test_hpc_submission.py
pytest tests/test_legnet.py
pytest tests/test_rhino_variants.py
pytest tests/test_rotated_fcos.py
```

## Smoke-Test Dataset Paths

- OBB: `/Users/manishagupta/Desktop/PhD/rip-detection/data_processed/Global/Annotated/variants/obb/planet_full_c448_ov35_kf20_10075-single_sh-lw-d-prx-cl-hz-sdw_smoke10_seed0/data.yaml`
- Segment: `/Users/manishagupta/Desktop/PhD/rip-detection/data_processed/Global/Annotated/variants/segment/planet_full_c448_ov35_kf20_10075-single_sh-lw-d-prx-cl-hz-sdw_smoke10_seed0/data.yaml`

## Smoke Test Priorities

When running a smoke test, run **2 full epochs**, not just a few steps. Check in this order:

1. Logic is correct: finite losses and gradients alone do not establish correctness. Ask for clarification if the intended behavior is unclear; do not assume it.
2. Gradients are finite and nonzero in the layers where they should be.
3. Losses are finite (no NaNs) and not stuck at one value.
4. Data loads correctly.

Focus on data loading, loss logic, and architecture/module logic. Do not edit code during a smoke test unless specifically asked. Report what worked, what failed, and anything that could not be verified.

## Agent Rules

- Only edit `ultralytics/` rahter than the reference repos.
- Reuse existing components before adding new abstractions.
- Check nearby tests/configs before making assumptions.
- Be careful with absolute dataset paths, scratch paths, and HPC env variables.
- In HPC submit scripts, keep `plots=False` to prevent large image uploads to `wandb` when `wandb=True`.
- Ask before deleting non-temporary files.
- When asked to `check`/`find`/`investigate`/`brainstorm` do not edit any code. Only read the required files.
- Always ask for clarification if there is ambiguity in an instruction or information provided. Do not assume anything.
