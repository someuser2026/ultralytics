#!/usr/bin/env python3
"""Run YOLO prediction for one site/image folder and save one aggregate predictions.json."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from ultralytics import YOLO
from ultralytics.utils.callbacks.wb import _prediction_json_payload, _prediction_result_key

try:
    import wandb as wb

    assert hasattr(wb, "__version__")
except (ImportError, AssertionError):
    wb = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--site-name", required=True, help="Site name under $SCRATCH/data_processed/<site_name>")
    parser.add_argument("--img-dir", required=True, help="Relative directory under $SCRATCH/data_processed/<site>/PSScene")
    parser.add_argument("--imgsz", type=int, default=640, help="Starting inference image size")
    parser.add_argument("--conf", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=None, help="Optional cap on detections per image")
    parser.add_argument("--batch", type=int, default=None, help="Optional batch-size override for directory mode.")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0, 0,1, or cpu")
    parser.add_argument(
        "--predict-mode",
        choices=("per-image", "directory"),
        default="per-image",
        help="Use one predict call per image or one predict call on the whole image directory.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to upload the prediction directory to a sibling W&B inference run.",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Optional W&B run ID to resume instead of creating a sibling inference run.",
    )
    parser.add_argument("--job-batch-index", type=int, default=None, help="Optional 1-based batch index for job splitting.")
    parser.add_argument(
        "--job-batch-start",
        type=int,
        default=None,
        help="Optional inclusive start index into the sorted PNG list for job splitting.",
    )
    parser.add_argument(
        "--job-batch-end",
        type=int,
        default=None,
        help="Optional exclusive end index into the sorted PNG list for job splitting.",
    )
    return parser.parse_args(argv)


def is_oom_error(exc: Exception) -> bool:
    """Return True when an exception indicates CUDA OOM."""
    message = str(exc).lower()
    return "out of memory" in message or "cuda out of memory" in message


def derive_run_name(ckpt: Path) -> str:
    """Infer run name from a .../<run_name>/weights/best.pt checkpoint layout."""
    if ckpt.name != "best.pt" or ckpt.parent.name != "weights":
        raise ValueError(f"Checkpoint must be .../<run_name>/weights/best.pt, got: {ckpt}")
    return ckpt.parent.parent.name


def derive_task(ckpt: Path) -> str:
    """Infer task from checkpoint path parts."""
    parts = {part.lower() for part in ckpt.parts}
    if "obb" in parts:
        return "obb"
    if "segment" in parts:
        return "segment"
    raise ValueError(f"Could not infer task (obb/segment) from checkpoint path: {ckpt}")


def validate_img_dir(img_dir: str) -> Path:
    """Validate that img_dir is a relative path under PSScene."""
    img_dir_path = Path(img_dir)
    if img_dir_path.is_absolute():
        raise ValueError(f"img_dir must be relative to PSScene, got absolute path: {img_dir}")
    if any(part == ".." for part in img_dir_path.parts):
        raise ValueError(f"img_dir must not traverse outside PSScene: {img_dir}")
    normalized_parts = [part for part in img_dir_path.parts if part not in ("", ".")]
    if not normalized_parts:
        raise ValueError("img_dir must not be empty")
    return Path(*normalized_parts)


def slugify_img_dir(img_dir: str) -> str:
    """Create a deterministic filesystem-safe slug for img_dir."""
    return "__".join(validate_img_dir(img_dir).parts)


def resolve_image_dir(scratch: str | Path, site_name: str, img_dir: str) -> Path:
    """Resolve the input image directory under $SCRATCH/data_processed/<site>/PSScene."""
    return Path(scratch) / "data_processed" / site_name / "PSScene" / validate_img_dir(img_dir)


def derive_output_dir(
    scratch: str | Path,
    site_name: str,
    task: str,
    run_name: str,
    img_dir: str,
    *,
    batch_index: int | None = None,
) -> Path:
    """Derive the output directory that will contain predictions.json."""
    folder_name = f"{run_name}__{slugify_img_dir(img_dir)}"
    output_dir = Path(scratch) / "data_processed" / site_name / "predictions" / task / folder_name
    if batch_index is not None:
        if batch_index < 1:
            raise ValueError(f"batch_index must be >= 1, got: {batch_index}")
        output_dir = output_dir / f"batch_{batch_index}"
    return output_dir


def list_input_images(img_dir: Path) -> list[Path]:
    """Return sorted top-level PNG files in the image directory."""
    return sorted(path for path in img_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")


def resolve_job_batch(
    args: argparse.Namespace,
    images: list[Path],
    *,
    scratch: str | Path,
    site_name: str,
    task: str,
    run_name: str,
    img_dir: str,
) -> tuple[list[Path], Path]:
    """Validate optional job-batch arguments and return the selected images/output dir."""
    batch_fields = (args.job_batch_index, args.job_batch_start, args.job_batch_end)
    if all(value is None for value in batch_fields):
        return images, derive_output_dir(scratch, site_name, task, run_name, img_dir)
    if any(value is None for value in batch_fields):
        raise ValueError("job batch arguments must be provided together: --job-batch-index/start/end")

    batch_index = args.job_batch_index
    batch_start = args.job_batch_start
    batch_end = args.job_batch_end
    if batch_index is None or batch_start is None or batch_end is None:
        raise ValueError("job batch arguments must not be None once provided")
    if batch_index < 1:
        raise ValueError(f"--job-batch-index must be >= 1, got: {batch_index}")
    if batch_start < 0:
        raise ValueError(f"--job-batch-start must be >= 0, got: {batch_start}")
    if batch_end <= batch_start:
        raise ValueError(
            f"--job-batch-end must be greater than --job-batch-start, got start={batch_start}, end={batch_end}"
        )
    if batch_end > len(images):
        raise ValueError(f"--job-batch-end={batch_end} exceeds image count {len(images)}")

    selected_images = images[batch_start:batch_end]
    if not selected_images:
        raise ValueError(f"Batch slice [{batch_start}:{batch_end}] did not select any images")

    return selected_images, derive_output_dir(scratch, site_name, task, run_name, img_dir, batch_index=batch_index)


def resolve_source_root(source_root: str | Path | list[Path] | tuple[Path, ...] | None) -> Path | None:
    """Normalize source_root to the same shape used by the aggregate JSON helpers."""
    if isinstance(source_root, (list, tuple)):
        return next((Path(item) for item in source_root if item is not None), None)
    return Path(source_root) if source_root is not None else None


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


def build_wandb_context(ckpt: Path, task: str) -> dict[str, Any]:
    """Resolve W&B run context from the checkpoint layout and saved args."""
    run_dir = ckpt.parent.parent
    saved_args = load_saved_args(run_dir)
    project_value = saved_args.get("project")
    project_path = Path(project_value) if project_value not in (None, "") else run_dir.parent
    return {
        "weights_path": ckpt,
        "run_dir": run_dir,
        "run_name": run_dir.name,
        "saved_args": saved_args,
        "project_path": project_path,
        "task": saved_args.get("task") or task,
        "wandb_project": clean_wandb_project_name(project_path, saved_args.get("task") or task),
    }


def initialize_wandb_run(context: dict[str, Any], *, wandb_run_id: str | None = None, wandb_module=None):
    """Create a sibling inference run or resume the original run when an explicit run ID is provided."""
    wandb_module = wandb_module or wb
    if wandb_module is None:
        raise RuntimeError("wandb is not installed or unavailable in this environment.")

    requested_name = context["run_name"] if wandb_run_id else f"{context['run_name']}_inference"
    config = {
        "weights": str(context["weights_path"]),
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

    run = wandb_module.init(**init_kwargs)
    active_run_name = getattr(run, "name", None) or requested_name
    return run, active_run_name


def log_predictions(pred_dir: str | Path, run_name: str, site_name: str, img_dir_slug: str, wandb_module=None) -> bool:
    """Upload one site prediction directory to W&B while keeping the local export on disk."""
    wandb_module = wandb_module or wb
    if wandb_module is None:
        raise RuntimeError("wandb is not installed or unavailable in this environment.")

    pred_dir = Path(pred_dir)
    if not pred_dir.exists():
        raise RuntimeError(f"Prediction directory does not exist: {pred_dir}")

    artifact_name = f"{run_name}_predictions_site_{site_name}_{img_dir_slug}"
    artifact_type = "predictions_site"
    artifact = wandb_module.Artifact(artifact_name, type=artifact_type)
    artifact.add_dir(str(pred_dir))
    logged_artifact = wandb_module.log_artifact(artifact)
    if hasattr(logged_artifact, "wait"):
        logged_artifact.wait()
    return True


class PredictionJsonWriter:
    """Stream prediction records to disk so large folder runs do not accumulate Results in RAM."""

    def __init__(self, output_dir: str | Path, source_root=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_root = resolve_source_root(source_root)
        self.entries_tmp_path = self.output_dir / ".predictions_entries.tmp"
        self.used_keys: set[str] = set()
        self.count = 0
        self.first = True
        self._closed = False
        self._entries_handle = self.entries_tmp_path.open("w", encoding="utf-8")

    def add_result(self, result) -> None:
        """Serialize one prediction result directly to the temp entries file."""
        key = _prediction_result_key(result, self.source_root, self.count, self.used_keys)
        payload = _prediction_json_payload(result)
        if not self.first:
            self._entries_handle.write(",\n")
        self._entries_handle.write(json.dumps(key))
        self._entries_handle.write(": ")
        json.dump(payload, self._entries_handle, indent=2)
        self.first = False
        self.count += 1

    def close(self) -> Path:
        """Build the final predictions.json and remove temporary files."""
        if self._closed:
            return self.output_dir / "predictions.json"

        self._entries_handle.close()
        final_path = self.output_dir / "predictions.json"
        with final_path.open("w", encoding="utf-8") as handle:
            handle.write("{\n")
            handle.write(f'  "count": {self.count},\n')
            handle.write('  "predictions": {\n')
            with self.entries_tmp_path.open("r", encoding="utf-8") as entries:
                shutil.copyfileobj(entries, handle)
            if not self.first:
                handle.write("\n")
            handle.write("  }\n")
            handle.write("}\n")

        self.entries_tmp_path.unlink(missing_ok=True)
        self._closed = True
        return final_path

    def abort(self) -> None:
        """Remove partial temp output after a fatal error."""
        if not self._entries_handle.closed:
            self._entries_handle.close()
        self.entries_tmp_path.unlink(missing_ok=True)
        self._closed = True


def build_predict_kwargs(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    run_name: str,
    use_half: bool,
    max_det: int,
    batch: int | None,
) -> dict[str, Any]:
    """Build kwargs for YOLO.predict."""
    predict_kwargs = {
        "project": str(output_dir),
        "name": run_name,
        "exist_ok": True,
        "save": False,
        "stream": True,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": max_det,
        "half": use_half,
        "device": args.device,
        "verbose": False,
    }
    if batch is not None:
        predict_kwargs["batch"] = batch
    return predict_kwargs


def run_per_image_predictions(
    model,
    images: list[Path],
    writer: PredictionJsonWriter,
    predict_kwargs: dict[str, Any],
) -> list[str]:
    """Run one predict call per image and stream results into the aggregate JSON writer."""
    failed = []
    for index, image_path in enumerate(images, start=1):
        try:
            results = list(model.predict(source=str(image_path), **predict_kwargs))
            for result in results:
                writer.add_result(result.cpu())
            del results
            if getattr(model, "predictor", None) is not None:
                model.predictor.results = None
                model.predictor.batch = None
        except RuntimeError as exc:
            if is_oom_error(exc):
                print(f"[OOM] {image_path.name}: {exc}")
                writer.abort()
                raise
            print(f"[FAIL] {image_path.name}: {exc}")
            failed.append(image_path.name)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {image_path.name}: {exc}")
            failed.append(image_path.name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if index % 20 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)}")

    return failed


def run_directory_predictions(
    model,
    source: Path | list[Path],
    writer: PredictionJsonWriter,
    predict_kwargs: dict[str, Any],
) -> None:
    """Run one predict call on the entire image directory and stream results into the aggregate JSON writer."""
    processed = 0
    try:
        if isinstance(source, Path):
            predict_source: str | list[str] = str(source)
        else:
            predict_source = [str(path) for path in source]
        for result in model.predict(source=predict_source, **predict_kwargs):
            writer.add_result(result.cpu())
            processed += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if processed % 20 == 0:
                print(f"Processed {processed}")
    except RuntimeError as exc:
        if is_oom_error(exc):
            print(f"[OOM] directory mode: {exc}")
            writer.abort()
            raise
        raise


def main(argv: list[str] | None = None) -> Path:
    """Run prediction and write one aggregate JSON output directory."""
    args = parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH is required")

    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    run_name = derive_run_name(ckpt)
    task = derive_task(ckpt)
    max_det = args.max_det if args.max_det is not None else (100 if task == "segment" else 300)
    image_dir = resolve_image_dir(scratch, args.site_name, args.img_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = list_input_images(image_dir)
    if not images:
        raise RuntimeError(f"No PNG images found in {image_dir}")

    selected_images, output_dir = resolve_job_batch(
        args,
        images,
        scratch=scratch,
        site_name=args.site_name,
        task=task,
        run_name=run_name,
        img_dir=args.img_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ckpt))
    use_half = args.device != "cpu" and torch.cuda.is_available()
    use_half = False
    batch = args.batch if args.predict_mode == "directory" else 1
    predict_kwargs = build_predict_kwargs(
        args,
        output_dir=output_dir,
        run_name=run_name,
        use_half=use_half,
        max_det=max_det,
        batch=batch,
    )
    writer = PredictionJsonWriter(output_dir, source_root=image_dir)

    print(f"Images: {len(images)}")
    print(f"Selected images: {len(selected_images)}")
    print(f"Site: {args.site_name}")
    print(f"Task: {task}")
    print(f"Run: {run_name}")
    print(f"Input: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Predict mode: {args.predict_mode}")
    print(f"Device: {args.device}, half={use_half}")
    print(f"NMS: conf={args.conf}, iou={args.iou}, max_det={max_det}")
    print(f"Batch: {batch}")
    print(f"W&B upload: {args.wandb}")
    if args.job_batch_index is not None:
        print(
            "Job batch: "
            f"index={args.job_batch_index}, start={args.job_batch_start}, end={args.job_batch_end}"
        )

    failed = []
    wandb_run = None
    active_run_name = run_name

    if args.wandb:
        context = build_wandb_context(ckpt, task)
        wandb_run, active_run_name = initialize_wandb_run(context, wandb_run_id=args.wandb_run_id)

    try:
        if args.predict_mode == "directory":
            directory_source = selected_images if args.job_batch_index is not None else image_dir
            run_directory_predictions(model, directory_source, writer, predict_kwargs)
        else:
            failed = run_per_image_predictions(model, selected_images, writer, predict_kwargs)
    except Exception:
        writer.abort()
        raise
    finally:
        if getattr(model, "predictor", None) is not None:
            model.predictor.results = None
            model.predictor.batch = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    predictions_json = writer.close()

    print("Done")
    print(f"Predictions json: {predictions_json}")
    print(f"Failed images: {len(failed)}")
    if failed:
        failure_log = output_dir / "failed_images.txt"
        failure_log.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"Failure log: {failure_log}")

    try:
        if args.wandb:
            log_predictions(output_dir, active_run_name, args.site_name, slugify_img_dir(args.img_dir))
            print(f"W&B artifact logged for: {output_dir}")
    finally:
        if wandb_run is not None and hasattr(wandb_run, "finish"):
            wandb_run.finish()

    return output_dir


if __name__ == "__main__":
    main()
