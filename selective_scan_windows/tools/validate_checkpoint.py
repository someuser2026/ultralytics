"""Validate a local research checkpoint; never download inputs or upload predictions."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_baseline(baseline_path, identity, arrays):
    """Require identical inputs/settings and conservatively compare ordered outputs."""
    import numpy as np

    baseline_path = Path(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["identity"] != identity:
        raise ValueError("Baseline checkpoint/image hashes or prediction settings differ.")
    with np.load(baseline_path.with_name("predictions.npz"), allow_pickle=False) as prior:
        if set(prior.files) != set(arrays):
            raise AssertionError("Baseline prediction fields differ.")
        for key, value in arrays.items():
            if value.shape != prior[key].shape:
                raise AssertionError(f"Baseline prediction shape differs for {key}: {value.shape} vs {prior[key].shape}")
            if key.endswith("_masks"):
                np.testing.assert_array_equal(value, prior[key], err_msg=key)
            else:
                np.testing.assert_allclose(value, prior[key], rtol=6e-4, atol=2e-3, err_msg=key)


def run(args, report):
    import numpy as np
    import torch
    from selective_scan_windows.diagnostics import require_backend

    report["backend"] = require_backend()
    checkpoint = args.checkpoint.resolve(strict=True)
    images = [p.resolve(strict=True) for p in args.images]
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".pt":
        raise ValueError("--checkpoint must be an existing local .pt file.")
    from ultralytics import YOLO
    from ultralytics.nn.modules import common_utils_mbyolo
    from selective_scan_windows import fwd, bwd

    if (common_utils_mbyolo.selective_scan_cuda_core.fwd is not fwd or
            common_utils_mbyolo.selective_scan_cuda_core.bwd is not bwd):
        raise RuntimeError("Ultralytics is not using the new Windows backend.")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    model = YOLO(str(checkpoint))
    mamba_modules = [name for name, module in model.model.named_modules()
                     if module.__class__.__module__ == "ultralytics.nn.modules.mamba_yolo"]
    if not mamba_modules:
        raise ValueError("Checkpoint contains no active Mamba modules; it cannot qualify selective scan.")
    # Let prediction letterbox source images of any size to the fixed model input.
    settings = dict(imgsz=448, batch=1, half=False, conf=args.conf, iou=args.iou,
                    max_det=args.max_det, retina_masks=args.retina_masks, rect=False,
                    augment=False, agnostic_nms=False)
    report["identity"] = {
        "checkpoint_sha256": sha256(checkpoint),
        "image_sha256": [sha256(path) for path in images], "settings": settings,
    }
    report["checkpoint"] = str(checkpoint)
    report["images"] = [str(path) for path in images]
    report["mamba_module_count"] = len(mamba_modules)
    arrays = {}
    timings = []
    memory = []

    def predict(path):
        return model.predict(source=str(path), device=args.device, **settings, verbose=False,
                             save=False, save_txt=False, save_crop=False, save_conf=False,
                             show=False, plots=False, project=str(args.output), name="inference", exist_ok=True)

    with torch.inference_mode():
        for _ in range(2):
            for path in images:
                results = predict(path)
                del results
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        for repeat in range(args.repeats):
            for index, path in enumerate(images):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                results = predict(path)
                torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - start)
                if len(results) != 1:
                    raise AssertionError("Expected exactly one result per input tile.")
                for field in ("boxes", "masks", "obb", "probs", "keypoints"):
                    item = getattr(results[0], field, None)
                    if item is not None:
                        tensor = item.data
                        if not torch.isfinite(tensor).all().item():
                            raise AssertionError(f"Non-finite {field} on {path}")
                        if repeat == 0:
                            arrays[f"image_{index}_{field}"] = tensor.cpu().numpy().copy()
                del results
            torch.cuda.synchronize(device)
            memory.append({"allocated": torch.cuda.memory_allocated(device),
                           "reserved": torch.cuda.memory_reserved(device)})

    report["memory_bytes"] = {
        "per_repeat": memory, "peak_allocated": torch.cuda.max_memory_allocated(device),
        "peak_reserved": torch.cuda.max_memory_reserved(device),
        "device_total": torch.cuda.get_device_properties(device).total_memory,
    }
    report["timing_seconds_per_tile"] = {"median": statistics.median(timings),
                                         "minimum": min(timings), "maximum": max(timings)}
    report["prediction_shapes"] = {key: list(value.shape) for key, value in arrays.items()}
    # Compare the same end-of-pass point after warming up all input shapes.
    # Reserved-memory growth alone can be allocator caching, so record it separately.
    growth_limit = max(16 * 1024**2, int(memory[0]["allocated"] * 0.05))
    growth = max(m["allocated"] for m in memory[1:]) - memory[0]["allocated"]
    report["allocated_memory_growth"] = {"bytes": growth, "limit_bytes": growth_limit}
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    if growth > growth_limit:
        raise AssertionError("Repeated inference increased live GPU allocation beyond the documented threshold.")
    report["baseline_status"] = "not_provided"
    if args.baseline:
        compare_baseline(args.baseline, report["identity"], arrays)
        report["baseline_status"] = "passed"
    report["inference_status"] = "passed"
    report["note"] = "Inference verification only; run numerical tests separately. Full training is not qualified."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--images", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory for report.json and predictions.npz")
    parser.add_argument("--baseline", type=Path, help="Prior report.json with neighboring predictions.npz")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--retina-masks", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.device < 0 or args.max_det < 1:
        parser.error("Require repeats >= 3, device >= 0, and max-det >= 1.")
    if not (0 <= args.conf <= 1 and 0 <= args.iou <= 1):
        parser.error("conf and iou must be in [0, 1].")
    if args.output.exists():
        parser.error("--output must be a new directory to preserve earlier reports.")
    args.output.mkdir(parents=True)
    report = {"schema_version": 1, "inference_status": "failed"}
    try:
        run(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["inference_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
