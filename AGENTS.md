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
- OBB: `/Users/manishagupta/Desktop/PhD/rip-detection/data_processed/Global/Annotated/variants/obb/planet_full_c448_ov35_kf20_10075-single-shoreline4_sample10_seed0`
- Segment: `/Users/manishagupta/Desktop/PhD/rip-detection/data_processed/Global/Annotated/variants/segment/planet_full_c448_ov35_kf20_10075-single-shoreline4_sample10_seed0`

## Agent Rules
- Prefer editing `ultralytics/` over the reference repos.
- Reuse existing components before adding new abstractions.
- Check nearby tests/configs before making assumptions.
- Be careful with absolute dataset paths, scratch paths, and HPC env variables.
- Ask before deleting non-temporary files.
