# Repo Context

Custom `ultralytics` fork for rip current detection on satellite imagery. Primary tasks: `obb` and `segment`. The active model runtime is under `ultralytics/`. Paths below are relative to the repository root unless absolute.

## Route Before Reading

1. Choose the narrowest route for the requested task, model family, file, or error before reading implementation files.
2. Read applicable `AGENTS.md` guidance and existing `CONTEXT.md` files along that route, parent to child, including intermediate folders. Read contexts explicitly; reuse unchanged guidance already read. Skip unrelated branches.
3. Contexts mark routing/workflow boundaries, not every folder. Report a missing linked context; otherwise inherit the nearest parent without a missing-file warning. Do not create contexts unless requested. Ask if missing information leaves the task ambiguous.
4. Use scoped `rg --files <folder>` or `rg -n '<symbol>' <folder>` searches, narrowed by filename/glob. Read matching sections and relevant nearby tests/configs, not entire folders. Avoid repository-wide content searches as a first step.
5. Expand only when an import, caller, config, test, or unsuccessful scoped search points elsewhere; read that dependency's context chain first. Stop when the task and affected dependencies are understood.

Context paths are relative to their folder; commands run from the repository root unless stated otherwise. Follow cross-folder links only when needed. Shared rules belong here, child routing in parent contexts, and local contracts in the owning context. Update affected routes when paths or responsibilities change; avoid copying parent guidance.

## Task Routing

| Task / question | Start here |
| --- | --- |
| Model configuration, construction, training, validation, inference, data, losses | [ultralytics/CONTEXT.md](ultralytics/CONTEXT.md) |
| HPC training submission, PBS resources, batch/resume scripts | [jobs/train/hpc/CONTEXT.md](jobs/train/hpc/CONTEXT.md) |
| Site inference, prediction JSONs, batch combination, inference PBS jobs | [jobs/infer/CONTEXT.md](jobs/infer/CONTEXT.md) |
| Regression tests, fixtures, targeted checks | [tests/CONTEXT.md](tests/CONTEXT.md) |
| CUDA extension builds or extension import/runtime errors | `selective_scan/`, `dcnv4/`; first identify which extension the active module requires |
| Installation, dependencies, containers, usage examples, documentation | `pyproject.toml`, `docker/`, `examples/`, or `docs/`, according to the request |

Inspect `runs/`, `wandb/`, and `datasets/` only when the task needs particular outputs, logs, or data; keep them out of general source searches.

## Reference Sources

`mmdetection/`, `LEGNet/`, `RHINO/`, `Mamba-YOLO/`, `detectron2/`, `HRVMamba/`, and `Mask2Former/` are reference checkouts. Consult them for a relevant comparison, port, or traced dependency. Start with the active implementation in `ultralytics/`, even when a model shares a reference repository's name.

## Environment and Checks

Use `ultralytics_contrib` unless told otherwise. Local interpreter: `/Users/manishagupta/miniforge3/envs/ultralytics_contrib/bin/python`.

```bash
conda activate ultralytics_contrib
```

Select relevant checks through [tests/CONTEXT.md](tests/CONTEXT.md); do not run the full suite by default.

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

- Edit the active implementation and supporting project files, including `ultralytics/`, `jobs/`, `tests/`, and routing documentation. Do not edit the reference repositories unless explicitly requested.
- Reuse existing components before adding new abstractions.
- Be careful with absolute dataset paths, scratch paths, and HPC env variables.
- In HPC submit scripts, keep `plots=False` to prevent large image uploads to `wandb` when `wandb=True`.
- Ask before deleting non-temporary files.
- When asked to `check`/`find`/`investigate`/`brainstorm` do not edit any code. Only read the required files.
- Always ask for clarification if there is ambiguity in an instruction or information provided. Do not assume anything.
