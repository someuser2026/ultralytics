#!/usr/bin/env python3
"""Run YOLO prediction for one site/image folder and save one aggregate predictions.json."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils.callbacks.wb import _save_predictions_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--site-name", required=True, help="Site name under $SCRATCH/data_processed/<site_name>")
    parser.add_argument("--img-dir", required=True, help="Relative directory under $SCRATCH/data_processed/<site>/PSScene")
    parser.add_argument("--imgsz", type=int, default=640, help="Starting inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="0", help="Inference device, e.g. 0, 0,1, or cpu")
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


def derive_output_dir(scratch: str | Path, site_name: str, task: str, run_name: str, img_dir: str) -> Path:
    """Derive the output directory that will contain predictions.json."""
    folder_name = f"{run_name}__{slugify_img_dir(img_dir)}"
    return Path(scratch) / "data_processed" / site_name / "predictions" / task / folder_name


def list_input_images(img_dir: Path) -> list[Path]:
    """Return sorted top-level PNG files in the image directory."""
    return sorted(path for path in img_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")


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
    image_dir = resolve_image_dir(scratch, args.site_name, args.img_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = list_input_images(image_dir)
    if not images:
        raise RuntimeError(f"No PNG images found in {image_dir}")

    output_dir = derive_output_dir(scratch, args.site_name, task, run_name, args.img_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ckpt))
    use_half = args.device != "cpu" and torch.cuda.is_available()

    print(f"Images: {len(images)}")
    print(f"Site: {args.site_name}")
    print(f"Task: {task}")
    print(f"Run: {run_name}")
    print(f"Input: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}, half={use_half}")

    all_results = []
    failed = []

    for index, image_path in enumerate(images, start=1):
        try:
            with torch.inference_mode():
                results = list(
                    model.predict(
                        source=str(image_path),
                        project=str(output_dir),
                        name=run_name,
                        exist_ok=True,
                        save=False,
                        stream=True,
                        batch=1,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        half=use_half,
                        device=args.device,
                        verbose=False,
                    )
                )
            all_results.extend(results)
        except RuntimeError as exc:
            if is_oom_error(exc):
                print(f"[OOM] {image_path.name}: {exc}")
                raise

            print(f"[FAIL] {image_path.name}: {exc}")
            failed.append(image_path.name)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {image_path.name}: {exc}")
            failed.append(image_path.name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if index % 20 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)}")

    _save_predictions_json(all_results, output_dir, source_root=image_dir)
    predictions_json = output_dir / "predictions.json"

    print("Done")
    print(f"Predictions json: {predictions_json}")
    print(f"Failed images: {len(failed)}")
    if failed:
        failure_log = output_dir / "failed_images.txt"
        failure_log.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print(f"Failure log: {failure_log}")

    return output_dir


if __name__ == "__main__":
    main()
