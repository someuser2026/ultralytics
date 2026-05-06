#!/usr/bin/env python3
"""Merge batched site-level YOLO predictions into one aggregate predictions.json."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--site-name", required=True, help="Site name under $SCRATCH/data_processed/<site_name>")
    parser.add_argument("--img-dir", required=True, help="Relative directory under $SCRATCH/data_processed/<site>/PSScene")
    return parser.parse_args(argv)


def load_inference_module():
    """Load the site prediction script so output-path derivation stays shared."""
    script_path = Path(__file__).with_name("yolo_site_predict_json.py")
    spec = importlib.util.spec_from_file_location("yolo_site_predict_json", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    spec.loader.exec_module(module)
    return module


def list_batch_prediction_jsons(output_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted batch prediction files under the parent output directory."""
    batch_files: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("batch_"):
            continue
        suffix = child.name.removeprefix("batch_")
        if not suffix.isdigit():
            continue
        predictions_json = child / "predictions.json"
        if predictions_json.is_file():
            batch_files.append((int(suffix), predictions_json))
    return sorted(batch_files, key=lambda item: item[0])


def load_predictions(path: Path) -> dict[str, Any]:
    """Load one batch predictions.json payload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, found {type(payload).__name__}")
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError(f"Expected 'predictions' mapping in {path}")
    return predictions


def combine_predictions(output_dir: Path) -> Path:
    """Merge all batch predictions.json files into the parent output directory."""
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    batch_files = list_batch_prediction_jsons(output_dir)
    if not batch_files:
        raise FileNotFoundError(f"No batch predictions.json files found under {output_dir}")

    merged_predictions: dict[str, Any] = {}
    for batch_index, predictions_json in batch_files:
        predictions = load_predictions(predictions_json)
        duplicate_keys = set(merged_predictions).intersection(predictions)
        if duplicate_keys:
            preview = ", ".join(sorted(duplicate_keys)[:5])
            raise ValueError(f"Duplicate prediction keys found while merging batch_{batch_index}: {preview}")
        merged_predictions.update(predictions)

    merged_payload = {"count": len(merged_predictions), "predictions": merged_predictions}
    output_path = output_dir / "predictions.json"
    output_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> Path:
    """Resolve the normal output directory and merge all batch prediction payloads."""
    args = parse_args(argv)
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise EnvironmentError("SCRATCH is required")
    inference_module = load_inference_module()

    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    run_name = inference_module.derive_run_name(ckpt)
    task = inference_module.derive_task(ckpt)
    output_dir = inference_module.derive_output_dir(scratch, args.site_name, task, run_name, args.img_dir)
    merged_path = combine_predictions(output_dir)
    print(f"Merged predictions json: {merged_path}")
    return merged_path


if __name__ == "__main__":
    main()
