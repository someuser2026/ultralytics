#!/usr/bin/env python3
"""Export train/val/test predictions for a trained run and upload them to Weights & Biases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ultralytics import YOLO
from ultralytics.utils import LOGGER

try:
    import wandb as wb

    assert hasattr(wb, "__version__")
except (ImportError, AssertionError):
    wb = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for standalone prediction export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Path to the trained best.pt checkpoint.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the dataset YAML file.")
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Optional original W&B run ID to resume instead of creating a sibling inference run.",
    )
    parser.add_argument("--device", type=str, default=0, help="Optional device override, e.g. 0 or cpu.")
    parser.add_argument("--batch", type=int, default=None, help="Optional batch-size override.")
    parser.add_argument("--imgsz", type=int, default=448, help="Optional image-size override.")
    parser.add_argument("--conf", type=float, default=0.01, help="Confidence threshold for prediction export.")
    return parser.parse_args(argv)


def infer_run_name(weights: str | Path) -> str:
    """Infer the run name from a <run_name>/weights/best.pt style checkpoint path."""
    return Path(weights).expanduser().resolve().parent.parent.name


def load_saved_args(run_dir: str | Path) -> dict[str, Any]:
    """Load args.yaml from a run directory when it exists."""
    args_path = Path(run_dir) / "args.yaml"
    if not args_path.is_file():
        return {}

    data = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {args_path}, but found {type(data).__name__}.")
    return data


def clean_wandb_project_name(project_path: str | Path | None, task: str | None) -> str:
    """Match the training-side W&B project cleaning logic from the existing callback."""
    if not project_path:
        return "Ultralytics"

    proj_name_parts = str(project_path).split("/")
    required_parts = [part for part in proj_name_parts if "imgsz" in part]
    if task:
        required_parts.append(str(task))
    if proj_name_parts and proj_name_parts[-1]:
        required_parts.append(proj_name_parts[-1])

    project_cleaned = "-".join(required_parts) if required_parts else "Ultralytics"
    return project_cleaned.replace("yolo", "")


def build_run_context(
    weights: str | Path,
    data: str | Path,
    *,
    device: str | None = None,
    batch: int | None = None,
    imgsz: int | None = None,
    conf: float = 0.01,
    saved_args: dict[str, Any] | None = None,
    model_task: str | None = None,
) -> dict[str, Any]:
    """Resolve run metadata from the checkpoint path, args.yaml, and CLI overrides."""
    weights_path = Path(weights).expanduser().resolve()
    data_path = Path(data).expanduser().resolve()
    run_dir = weights_path.parent.parent
    run_name = run_dir.name

    saved_args = saved_args or load_saved_args(run_dir)
    project_value = saved_args.get("project")
    project_path = Path(project_value) if project_value not in (None, "") else run_dir.parent
    task = saved_args.get("task") or model_task
    if task in (None, ""):
        raise ValueError(
            f"Could not infer task for run '{run_name}'. Expected 'task' in {run_dir / 'args.yaml'} or model.task."
        )

    context = {
        "weights_path": weights_path,
        "data_path": data_path,
        "run_dir": run_dir,
        "run_name": run_name,
        "saved_args": saved_args,
        "project_path": project_path,
        "task": task,
        "device": device if device is not None else saved_args.get("device"),
        "batch": batch if batch is not None else saved_args.get("batch"),
        "imgsz": imgsz if imgsz is not None else saved_args.get("imgsz"),
        "conf": conf,
    }
    context["wandb_project"] = clean_wandb_project_name(project_path, task)
    return context


def format_context_value(value: Any) -> str:
    """Format resolved context values for readable console output."""
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def has_split_source(split_source: Any) -> bool:
    """Return whether a dataset split source is configured."""
    return split_source not in (None, [])


def log_run_context(context: dict[str, Any], active_run_name: str, *, resumed_original_run: bool) -> None:
    """Print the resolved model and inference context before prediction export starts."""
    LOGGER.info("=========================================")
    LOGGER.info("Standalone W&B Inference Export")
    LOGGER.info("=========================================")
    LOGGER.info(f"Model weights: {context['weights_path']}")
    LOGGER.info(f"Run directory: {context['run_dir']}")
    LOGGER.info(f"Source run name: {context['run_name']}")
    LOGGER.info(f"Task: {context['task']}")
    LOGGER.info(f"Data YAML: {context['data_path']}")
    LOGGER.info(f"Project path: {context['project_path']}")
    LOGGER.info(f"W&B project: {context['wandb_project']}")
    LOGGER.info(f"W&B run name: {active_run_name}")
    LOGGER.info(f"Resume original run: {resumed_original_run}")
    LOGGER.info(f"Device: {format_context_value(context.get('device'))}")
    LOGGER.info(f"Batch: {format_context_value(context.get('batch'))}")
    LOGGER.info(f"Image size: {format_context_value(context.get('imgsz'))}")
    LOGGER.info(f"Confidence: {format_context_value(context.get('conf'))}")
    LOGGER.info("Local prediction output directories:")
    LOGGER.info(f"  train: {context['run_dir'] / 'predictions' / 'train'}")
    LOGGER.info(f"  val:  {context['run_dir'] / 'predictions' / 'val'}")
    LOGGER.info(f"  test: {context['run_dir'] / 'predictions' / 'test'}")
    LOGGER.info("=========================================")


def resolve_split_source(dataset_root, split_spec):
    """Resolve dataset split specs that may be relative paths, absolute paths, or lists of either."""
    if split_spec in (None, ""):
        return None
    if isinstance(split_spec, (list, tuple)):
        resolved = [resolve_split_source(dataset_root, item) for item in split_spec]
        return [item for item in resolved if item is not None]

    split_path = Path(split_spec)
    if split_path.is_absolute() or dataset_root in (None, ""):
        return split_path
    return Path(dataset_root) / split_path


def load_data_split_sources(
    data_yaml: str | Path,
) -> tuple[dict[str, Any], Path | list[Path] | None, Path | list[Path] | None, Path | list[Path] | None]:
    """Load dataset YAML and resolve train/val/test sources."""
    data_path = Path(data_yaml).expanduser().resolve()
    data_dict = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data_dict, dict):
        raise ValueError(f"Expected a mapping in {data_path}, but found {type(data_dict).__name__}.")

    train_source = resolve_split_source(data_dict.get("path"), data_dict.get("train"))
    val_source = resolve_split_source(data_dict.get("path"), data_dict.get("val"))
    test_source = resolve_split_source(data_dict.get("path"), data_dict.get("test"))
    return data_dict, train_source, val_source, test_source


def prediction_task_name(result) -> str:
    """Infer the prediction task name from a Results object."""
    if result.obb is not None:
        return "obb"
    if result.probs is not None:
        return "classify"
    if result.keypoints is not None:
        return "pose"
    if result.masks is not None:
        return "segment"
    return "detect"


def safe_result_summary(result) -> list[dict[str, Any]]:
    """Return a serializable prediction summary even if Results.summary() fails for a custom result type."""
    try:
        summary = result.summary(normalize=True)
        if isinstance(summary, list):
            return summary
    except Exception as exc:
        LOGGER.warning(f"Falling back to manual prediction JSON export for '{result.path}': {exc}")

    h, w = result.orig_shape
    scale_x = float(w) if w else 1.0
    scale_y = float(h) if h else 1.0
    predictions = []
    data = result.obb if result.obb is not None else result.boxes
    if data is None:
        return predictions

    is_obb = result.obb is not None
    for row in data:
        try:
            class_id = int(row.cls)
            conf = round(float(row.conf), 5)
            coords = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2).tolist()
            box = {}
            for i, (x, y) in enumerate(coords, start=1):
                box[f"x{i}"] = round(float(x) / scale_x, 5)
                box[f"y{i}"] = round(float(y) / scale_y, 5)
            predictions.append(
                {
                    "name": result.names[class_id],
                    "class": class_id,
                    "confidence": conf,
                    "box": box,
                }
            )
        except Exception as exc:
            LOGGER.warning(f"Skipping malformed prediction row for '{result.path}': {exc}")
    return predictions


def prediction_json_payload(result) -> dict[str, Any]:
    """Build a serializable per-image prediction payload."""
    speed = getattr(result, "speed", {}) or {}
    if not isinstance(speed, dict):
        speed = {}
    return {
        "image_path": str(result.path),
        "image_name": Path(result.path).name,
        "image_stem": Path(result.path).stem,
        "task": prediction_task_name(result),
        "orig_shape": {"height": int(result.orig_shape[0]), "width": int(result.orig_shape[1])},
        "speed_ms": {k: None if v is None else float(v) for k, v in speed.items()},
        "predictions": safe_result_summary(result),
    }


def prediction_result_key(result, source_root: Path | None, index: int, used_keys: set[str]) -> str:
    """Create a stable key for one prediction result inside the aggregate JSON payload."""
    source_path = Path(result.path)
    if source_root is not None:
        try:
            relative_path = source_path.resolve().relative_to(source_root.resolve())
            key = relative_path.with_suffix("").as_posix()
        except Exception:
            key = source_path.stem
    else:
        key = source_path.stem

    if key in used_keys:
        key = f"{key}_{index:06d}"

    used_keys.add(key)
    return key


def save_predictions_json(results, output_dir: str | Path, source_root=None) -> Path:
    """Save prediction summaries as one aggregate JSON file for the entire split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(source_root, (list, tuple)):
        source_root = next((Path(item) for item in source_root if item is not None), None)
    else:
        source_root = Path(source_root) if source_root is not None else None

    used_keys: set[str] = set()
    payload = {"count": 0, "predictions": {}}

    for index, result in enumerate(results):
        key = prediction_result_key(result, source_root, index, used_keys)
        payload["predictions"][key] = prediction_json_payload(result)

    payload["count"] = len(payload["predictions"])
    with open(output_dir / "predictions.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_dir


def log_predictions(pred_dir: str | Path, run_name: str, subset: str, wandb_module=None) -> bool:
    """Upload one split's predictions directory to W&B while keeping the local export on disk."""
    wandb_module = wandb_module or wb
    if wandb_module is None:
        raise RuntimeError("wandb is not installed or unavailable in this environment.")

    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        LOGGER.warning(f"Skipping W&B upload; predictions directory does not exist: {pred_dir}")
        return False

    try:
        LOGGER.info(f"Logging {subset} predictions from {pred_dir} to wandb")
        artifact = wandb_module.Artifact(f"{run_name}_predictions_{subset}", type=f"predictions_{subset}")
        artifact.add_dir(str(pred_dir))
        logged_artifact = wandb_module.log_artifact(artifact)
        if hasattr(logged_artifact, "wait"):
            logged_artifact.wait()

        LOGGER.info(f"Uploaded {subset} predictions to wandb and kept local directory: {pred_dir}")
        return True
    except Exception as exc:
        LOGGER.warning(f"Failed to upload {subset} predictions to wandb. Local directory retained: {pred_dir}. Error: {exc}")
        return False


def export_split_predictions(
    model,
    split_source,
    output_dir: str | Path,
    run_name: str,
    subset: str,
    *,
    wandb_module=None,
    **predict_kwargs,
) -> bool:
    """Export one split's predictions to JSON and upload them to W&B."""
    if not has_split_source(split_source):
        LOGGER.info(f"No '{subset}' path found in data.yaml; skipping {subset} prediction artifact export.")
        return False

    try:
        LOGGER.info(f"Exporting {subset} predictions from source: {format_context_value(split_source)}")
        prediction_results = list(model.predict(split_source, stream=True, **predict_kwargs))
        save_predictions_json(prediction_results, output_dir, source_root=split_source)
        return log_predictions(output_dir, run_name, subset, wandb_module=wandb_module)
    except Exception as exc:
        LOGGER.warning(f"Failed to export {subset} predictions to wandb: {exc}")
        return False


def initialize_wandb_run(context: dict[str, Any], *, wandb_run_id: str | None = None, wandb_module=None):
    """Create a sibling inference run or resume the original run when an explicit run ID is provided."""
    wandb_module = wandb_module or wb
    if wandb_module is None:
        raise RuntimeError("wandb is not installed or unavailable in this environment.")

    requested_name = context["run_name"] if wandb_run_id else f"{context['run_name']}_inference"
    config = {
        "weights": str(context["weights_path"]),
        "data": str(context["data_path"]),
        "source_run_name": context["run_name"],
        "source_run_dir": str(context["run_dir"]),
        "source_project_path": str(context["project_path"]),
        "inference_export": True,
        "resumed_original_run": bool(wandb_run_id),
    }
    init_kwargs = {
        "project": context["wandb_project"],
        "name": requested_name,
        "config": config,
    }
    if wandb_run_id:
        init_kwargs["id"] = wandb_run_id
        init_kwargs["resume"] = "must"

    try:
        run = wandb_module.init(**init_kwargs)
    except Exception as exc:
        if wandb_run_id:
            raise RuntimeError(
                f"Failed to resume W&B run '{wandb_run_id}' for '{context['run_name']}'."
            ) from exc
        raise RuntimeError(f"Failed to initialize W&B run '{requested_name}'.") from exc

    active_run_name = getattr(run, "name", None) or requested_name
    return run, active_run_name


def build_predict_kwargs(context: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for YOLO.predict from resolved context."""
    predict_kwargs = {
        "conf": context["conf"],
        "save": False,
        "verbose": False,
    }
    for key in ("device", "batch", "imgsz"):
        value = context.get(key)
        if value is not None:
            predict_kwargs[key] = value
    return predict_kwargs


def run_inference_exports(args: argparse.Namespace, *, wandb_module=None, model_cls=YOLO) -> dict[str, Any]:
    """Run standalone train/val/test prediction export and upload results to W&B."""
    weights_path = args.weights.expanduser().resolve()
    data_path = args.data.expanduser().resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Data YAML file not found: {data_path}")

    run_dir = weights_path.parent.parent
    saved_args = load_saved_args(run_dir)
    model = model_cls(str(weights_path))
    context = build_run_context(
        weights_path,
        data_path,
        device=args.device,
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        saved_args=saved_args,
        model_task=getattr(model, "task", None),
    )

    run = None
    active_run_name = None
    try:
        run, active_run_name = initialize_wandb_run(
            context,
            wandb_run_id=args.wandb_run_id,
            wandb_module=wandb_module,
        )
        log_run_context(context, active_run_name, resumed_original_run=bool(args.wandb_run_id))
        _, train_source, val_source, test_source = load_data_split_sources(context["data_path"])
        output_root = context["run_dir"] / "predictions"
        predict_kwargs = build_predict_kwargs(context)

        split_sources = (("train", train_source), ("val", val_source), ("test", test_source))
        LOGGER.info("Resolved dataset split sources:")
        for subset, split_source in split_sources:
            LOGGER.info(f"  {subset}: {format_context_value(split_source)}")

        exported_subsets = []
        failed_subsets = []
        skipped_subsets = []
        for subset, split_source in split_sources:
            output_dir = output_root / subset
            if export_split_predictions(
                model,
                split_source,
                output_dir,
                active_run_name,
                subset,
                wandb_module=wandb_module,
                **predict_kwargs,
            ):
                exported_subsets.append(subset)
            elif has_split_source(split_source):
                failed_subsets.append(subset)
            else:
                skipped_subsets.append(subset)

        LOGGER.info(f"Exported prediction artifacts: {format_context_value(exported_subsets)}")
        if skipped_subsets:
            LOGGER.info(f"Skipped prediction artifacts: {format_context_value(skipped_subsets)}")
        if failed_subsets:
            raise RuntimeError(f"Prediction export failed for configured split(s): {', '.join(failed_subsets)}")

        if not exported_subsets:
            LOGGER.warning("No train/val/test prediction artifacts were exported.")

        return {
            "active_run_name": active_run_name,
            "context": context,
            "exported_subsets": exported_subsets,
            "skipped_subsets": skipped_subsets,
            "failed_subsets": failed_subsets,
        }
    finally:
        if run is not None and hasattr(run, "finish"):
            run.finish()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for standalone train/val/test prediction artifact export."""
    args = parse_args(argv)
    try:
        run_inference_exports(args)
    except Exception as exc:
        LOGGER.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
