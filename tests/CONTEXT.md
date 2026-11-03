# Test routing

Choose checks for the affected feature. Commands run from the repository root in the environment specified by `AGENTS.md`; its smoke-test policy and dataset paths also apply.

| Feature | Test files in this folder |
| --- | --- |
| Model YAML catalogue and CLI | `test_benchmark_model_configs.py`, `test_cli.py` |
| LEGNet / rotated FCOS | `test_legnet.py`, `test_rotated_fcos.py` |
| Mamba / HRVMamba | `test_mamba.py`, `test_hrvmamba_dvss.py` |
| RCNN and cascade variants | `test_rcnn_variants.py`, `test_pointrend_rcnn.py` |
| RHINO | `test_rhino_variants.py`, `test_rhino_isolation.py`, `test_rhino_reference_parity.py` |
| RT-DETR / Mask2Former / YOLO26 | `test_rtdetr_variants.py`, `test_mask2former.py`, `test_yolo26.py` |
| PointRend and YAML wiring | `test_pointrend.py`, `test_pointrend_yaml.py` |
| Data, auxiliary fields, metadata | `test_dataloader_seed.py`, `test_augmentation_padding.py`, `test_obb_aux_masks.py`, `test_metadata_modulation.py` |
| Neck/pooling behavior | `test_fpn_reference.py`, `test_sppf.py` |
| Engine, output contracts, fitness, exports | `test_engine.py`, `test_dict_outputs.py`, `test_task_fitness_weights.py`, `test_exports.py` |
| PBS submission/resume wrappers | `test_hpc_submission.py` |
| Site inference, combination, control | `test_site_predict_json_script.py`, `test_combine_site_predict_json_batches.py`, `test_site_prediction_pipeline.py` |
| W&B callbacks and inference logging | `test_wb_callback_json.py`, `test_inference_wandb_script.py` |

Example: `python -m pytest tests/test_rcnn_variants.py -k cascade`. Use `python -m pytest tests/test_hpc_submission.py` for submission changes. A unit-test function named "smoke" does not replace the required two-epoch training smoke test.

`configs/` holds LEGNet smoke YAMLs. `conftest.py` initializes seeds, recreates `tests/tmp`, excludes `slow` tests unless `--slow` is passed, and performs export-artifact cleanup at session end; inspect it before using existing outputs in a test session. Check selected tests for downloads, data, device, and extension requirements; report skips separately from passes.

`test_cuda.py`, `test_integrations.py`, `test_python.py`, and `test_solutions.py` provide additional coverage when those surfaces are involved. Root `pytest.ini` supplies the current pytest configuration.
