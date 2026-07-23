# Repo Context

This repo is a custom `ultralytics` fork for training and testing rip current detection models on satellite imagery.

Primary tasks:
- `obb`
- `segment`

## What Matters
- `ultralytics/`: main codebase; use this for model building, training, validation, and inference
- `ultralytics/cfg/models/`: custom model YAMLs
- `ultralytics/models/` and `ultralytics/nn/`: model/task implementations
- `ultralytics/cfg/datasets/`: dataset YAMLs
- `tests/`: local tests and smoke coverage
- `jobs/train/hpc/`: PBS/HPC submission scripts

## Reference-Only Folders
- `mmdetection/`
- `LEGNet/`
- `RHINO/`
- `Mamba-YOlO/`
- `detectron2/`

These are only for reference while porting ideas into `ultralytics/`. They are not the normal runtime path for training, testing, or model construction unless explicitly stated.

## Environment
- Use the `ultralytics_contrib` environment unless told otherwise.
- Typical setup:

```bash
conda activate ultralytics_contrib
```

## Useful Checks
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

Run the smoke test for 2 full epochs and not just steps. During the smoke test focus on these things first:

1. the logic is correct [incorrect logic can still have finite loss and gradients]. If you do not have enough clarification for the checking the correctness of the logic, simply ask for clarification. do not assume anything.
2. the graidients are finite and non zero for the layers they should be
3. the losses are finite [not nan] or stuck at one value
4. Data is loading correctly

Key areas for problems include:

1. Data loading process
2. logical errors in the loss functions
3. logical errors in model architecture [example modules etc]

Do not edit the code during the smoke test unless specifcally asked to. Simply report the results from the smoke tests, including what worked and what did not.

## Agent Rules
- Prefer editing `ultralytics/` over the reference repos.
- Reuse existing components before adding new abstractions.
- Check nearby tests/configs before making assumptions.
- Be careful with absolute dataset paths, scratch paths, and HPC env variables.
- Ask before deleting non-temporary files.
- When asked to `check`/`find`/`investigate`/`brainstorm` do not edit any code. Only read the required files.
- Always ask for clarification if there is ambiguity in a instruction or information provided. Do not assume anything.
