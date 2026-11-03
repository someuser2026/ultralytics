# HPC training jobs

Submission wrappers choose runs and pass environment variables to PBS scripts, which build the `yolo` command. Submit from the repository root; PBS execution uses `PBS_O_WORKDIR` when set.

| Need | Route |
| --- | --- |
| Full/short training execution and scheduler resources | `planet_full.pbs`, `planet_full_2hr_walltime.pbs`; inspect the selected profile's header and argument assembly |
| Benchmark/model-comparison/architecture-ablation submissions | `bash_scripts_joint/`; select the named `submit_*.sh` |
| Resume unfinished experiment groups | `bash_scripts_joint/resume_*.sh`; shared selection logic in its `*_lib.sh` files |
| Resume explicit checkpoints | `submit_resume_training.sh`: full/2hr profile plus `last.pt` paths |
| Segmentation input-band experiments | `bash_scripts_seg/submit_input_band_segment_models.sh` |

`bash_scripts_obb/` is currently empty; OBB jobs also come from joint launchers.

- PBS scripts bootstrap micromamba using `MAMBA_EXE` and `MAMBA_ROOT_PREFIX`; their defaults point to `/srv/scratch/z5428587/mamba`. Inspect the selected script before changing environment setup.
- Dataset variants resolve beneath `$SCRATCH/data_processed/Global/Annotated/variants/`. `planet_full.pbs` also supports `DATA_YAML`; do not assume the short profile has identical overrides.
- Fresh/fine-tuned outputs use `$SCRATCH/runs/cuda/<task>/imgsz_<imgsz>/yolo/`, optional `PROJECT`, then the run name. True resume follows the checkpoint's saved training state.
- Input bands belong in the dataset YAML; removed `USE_SHORELINE_INPUT`/`USE_LAND_WATER_INPUT` switches are rejected. Plot/upload policy is in root `AGENTS.md`.

Check wrapper argument positions before using their dry-run mode. [../../../tests/test_hpc_submission.py](../../../tests/test_hpc_submission.py) exercises submission behavior with stubs. Runtime changes route to [../../../ultralytics/CONTEXT.md](../../../ultralytics/CONTEXT.md).
