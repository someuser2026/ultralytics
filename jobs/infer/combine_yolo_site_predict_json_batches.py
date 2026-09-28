#!/usr/bin/env python3
"""Merge batched site-level YOLO predictions into one aggregate predictions.json."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Run directory containing batch_*/predictions.json")
    parser.add_argument("--data-processed-root", type=Path, help="Defaults to $SCRATCH/data_processed")
    parser.add_argument("--site-name", "--site", help="Site name under the processed data root")
    parser.add_argument("--task", choices=("obb", "segment"))
    parser.add_argument("--run", help="Full prediction folder name, including the image-directory suffix")
    parser.add_argument("--ckpt", help="Path to YOLO weights (.pt), for the existing checkpoint-based interface")
    parser.add_argument("--img-dir", help="Relative directory under $SCRATCH/data_processed/<site>/PSScene")
    parser.add_argument("--expected-batches", type=int, help="Require exactly batch_1 through batch_N")
    parser.add_argument("--expected-count", type=int, help="Require one prediction record per submitted image")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing")
    args = parser.parse_args(argv)
    if args.expected_batches is not None and args.expected_batches < 1:
        parser.error("--expected-batches must be >= 1")
    if args.expected_count is not None and args.expected_count < 0:
        parser.error("--expected-count must be >= 0")
    if args.output_dir:
        if any((args.data_processed_root, args.site_name, args.task, args.run, args.ckpt, args.img_dir)):
            parser.error("--output-dir cannot be combined with other path-selection arguments")
    elif args.ckpt:
        if not args.site_name or not args.img_dir:
            parser.error("--ckpt requires --site-name and --img-dir")
        if any((args.data_processed_root, args.task, args.run)):
            parser.error("--ckpt cannot be combined with --data-processed-root, --task or --run")
    elif not (args.site_name and args.task and args.run) or args.img_dir:
        parser.error("Provide --output-dir, --site-name/--task/--run, or --ckpt/--site-name/--img-dir")
    return args


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
    records: dict[str, Any] = {}
    for raw_key, record in predictions.items():
        if not isinstance(record, dict):
            raise ValueError(f"{path} prediction[{raw_key!r}] must be a JSON object")
        key = raw_key
        for field in ("image_stem", "image_name"):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                key = value
                break
        stem = Path(key).stem
        if stem in records:
            raise ValueError(f"Duplicate prediction keys after stem normalization: {stem!r} in {path}")
        records[stem] = record
    return records


def combine_predictions(
    output_dir: Path, *, expected_batches: int | None = None, expected_count: int | None = None, dry_run: bool = False
) -> Path:
    """Merge all batch predictions.json files into the parent output directory."""
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    batch_files = list_batch_prediction_jsons(output_dir)
    if not batch_files:
        raise FileNotFoundError(f"No batch predictions.json files found under {output_dir}")
    if expected_batches is not None:
        if expected_batches < 1:
            raise ValueError("expected_batches must be >= 1")
        indices = [index for index, _ in batch_files]
        if indices != list(range(1, expected_batches + 1)):
            raise ValueError(f"Expected exactly batch_1 through batch_{expected_batches}, found batch indices {indices}")

    merged_predictions: dict[str, Any] = {}
    for batch_index, predictions_json in batch_files:
        predictions = load_predictions(predictions_json)
        duplicate_keys = set(merged_predictions).intersection(predictions)
        if duplicate_keys:
            preview = ", ".join(sorted(duplicate_keys)[:5])
            raise ValueError(f"Duplicate prediction keys found while merging batch_{batch_index}: {preview}")
        merged_predictions.update(predictions)

    if expected_count is not None and len(merged_predictions) != expected_count:
        raise ValueError(f"Expected {expected_count} prediction records, found {len(merged_predictions)}")
    merged_payload = {"count": len(merged_predictions), "predictions": merged_predictions}
    output_path = output_dir / "predictions.json"
    print(f"{output_dir}: {len(batch_files)} batches, {len(merged_predictions)} records -> {output_path}")
    if dry_run:
        return output_path
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output_dir, prefix=".predictions.json.", suffix=".tmp", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(merged_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        tmp_path.replace(output_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return output_path


def main(argv: list[str] | None = None) -> Path:
    """Resolve the normal output directory and merge all batch prediction payloads."""
    args = parse_args(argv)
    if args.output_dir:
        output_dir = args.output_dir.expanduser()
    elif args.ckpt:
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
    else:
        root = args.data_processed_root
        if root is None:
            scratch = os.environ.get("SCRATCH")
            if not scratch:
                raise EnvironmentError("Set SCRATCH or provide --data-processed-root")
            root = Path(scratch) / "data_processed"
        for name in (args.site_name, args.run):
            if Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Site and run must be directory names, got {name!r}")
        output_dir = root.expanduser() / args.site_name / "predictions" / args.task / args.run
    merged_path = combine_predictions(
        output_dir, expected_batches=args.expected_batches, expected_count=args.expected_count, dry_run=args.dry_run
    )
    print(f"{'Validated' if args.dry_run else 'Merged'} predictions json: {merged_path}")
    return merged_path


if __name__ == "__main__":
    main()
