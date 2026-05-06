from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "jobs" / "infer" / "combine_yolo_site_predict_json_batches.py"


def load_script_module():
    """Load the standalone batch combine script as a Python module."""
    spec = importlib.util.spec_from_file_location("combine_yolo_site_predict_json_batches", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def combine_module():
    """Provide a fresh copy of the combine script module for each test."""
    return load_script_module()


def write_run_layout(tmp_path: Path, *, task: str = "obb", run_name: str = "demo_run") -> Path:
    """Create a checkpoint layout matching <run_name>/weights/best.pt."""
    run_dir = tmp_path / "scratch" / "runs" / "cuda" / task / "imgsz_448" / "yolo" / "project_x" / run_name
    weights_path = run_dir / "weights" / "best.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"weights")
    return weights_path


def write_batch_payload(output_dir: Path, batch_index: int, predictions: dict[str, object]) -> None:
    """Write one batch predictions.json payload."""
    batch_dir = output_dir / f"batch_{batch_index}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    payload = {"count": len(predictions), "predictions": predictions}
    (batch_dir / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_main_merges_batch_prediction_jsons(combine_module, tmp_path: Path, monkeypatch):
    scratch = tmp_path / "scratch"
    weights_path = write_run_layout(tmp_path, task="obb", run_name="demo_run")
    inference_module = combine_module.load_inference_module()
    output_dir = inference_module.derive_output_dir(scratch, "Treachery", "obb", "demo_run", "tiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    write_batch_payload(output_dir, 2, {"b": {"score": 0.8}})
    write_batch_payload(output_dir, 1, {"a": {"score": 0.9}})
    (output_dir / "predictions.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("SCRATCH", str(scratch))
    merged_path = combine_module.main(
        [
            "--ckpt",
            str(weights_path),
            "--site-name",
            "Treachery",
            "--img-dir",
            "tiles",
        ]
    )

    payload = json.loads(merged_path.read_text(encoding="utf-8"))
    assert merged_path == output_dir / "predictions.json"
    assert payload["count"] == 2
    assert list(payload["predictions"]) == ["a", "b"]


def test_combine_predictions_rejects_duplicate_keys(combine_module, tmp_path: Path):
    output_dir = tmp_path / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_batch_payload(output_dir, 1, {"dup": {"score": 0.9}})
    write_batch_payload(output_dir, 2, {"dup": {"score": 0.8}})

    with pytest.raises(ValueError, match="Duplicate prediction keys"):
        combine_module.combine_predictions(output_dir)


def test_combine_predictions_requires_batch_jsons(combine_module, tmp_path: Path):
    output_dir = tmp_path / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="No batch predictions.json files found"):
        combine_module.combine_predictions(output_dir)
