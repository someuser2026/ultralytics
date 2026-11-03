# Site inference jobs

These scripts run site predictions and manage their artifacts. Model-level prediction behavior belongs to [../../ultralytics/models/CONTEXT.md](../../ultralytics/models/CONTEXT.md).

| Need | Entry point |
| --- | --- |
| Image selection, prediction mode, JSON writing, output paths | `yolo_site_predict_json.py` |
| Merge split-batch JSONs | `combine_yolo_site_predict_json_batches.py` |
| Optional durable PBS control and recovery | `site_predict_job_control.py`; enabled through `SITE_PREDICT_CONTROL` |
| Prepared image inventories, record validation, batch-manifest contracts | `prepared_prediction_contract.py` |
| Load the configured external segmentation policy | `segmentation_filter_bridge.py` |
| Upload existing artifacts | `log_predictions_to_wandb.py`; matching submit/PBS files under `hpc/` |
| Submit GPU prediction batches | `hpc/submit_yolo_site_predict_json.sh`, `hpc/yolo_site_predict_json.pbs` |
| Launch batches plus dependent CPU combination | `hpc/yolo_site_predict_json_pipeline.pbs`, `hpc/combine_yolo_site_predict_json_batches.pbs` |
| Multi-site wrapper | `hpc/submit_all_sites_yolo_predict_json_pngs_c448_ov35_kf20.sh` |

Inputs resolve under `$SCRATCH/data_processed/<site>/PSScene/<img-dir>`; `img-dir` must be relative and cannot traverse upward. The predictor expects `.../<run>/weights/best.pt` and infers `obb`/`segment` from checkpoint path components.

Outputs are under `$SCRATCH/data_processed/<site>/predictions/<task>/<run>__<img-dir-slug>/`; split jobs write `batch_<index>/predictions.json`, and combination writes the parent `predictions.json`. The configured segmentation-policy file is an external dependency validated by the bridge; do not silently substitute another implementation.

Check script help for differing Python/wrapper defaults, batching, W&B, and dry-run arguments. Tests are grouped under site inference in [../../tests/CONTEXT.md](../../tests/CONTEXT.md).
