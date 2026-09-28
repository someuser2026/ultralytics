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


def test_normalizes_keys_using_reference_precedence(combine_module, tmp_path: Path):
    write_batch_payload(tmp_path, 10, {"elsewhere/c.png": {"score": 0.5}})
    write_batch_payload(tmp_path, 2, {"ignored": {"image_name": "folder/b.png"}})
    write_batch_payload(tmp_path, 1, {"ignored": {"image_stem": "a", "image_name": "wrong.png"}})
    result = combine_module.combine_predictions(tmp_path)
    payload = json.loads(result.read_text())
    assert list(payload["predictions"]) == ["a", "b", "c"]
    assert payload["count"] == 3
    assert payload["predictions"]["a"]["image_name"] == "wrong.png"


@pytest.mark.parametrize("same_batch", [True, False])
def test_rejects_normalized_duplicates(combine_module, tmp_path: Path, same_batch: bool):
    records = {"a.png": {}}
    if same_batch:
        records["folder/a.jpg"] = {}
    else:
        write_batch_payload(tmp_path, 2, {"another": {"image_name": "a.png"}})
    write_batch_payload(tmp_path, 1, records)
    with pytest.raises(ValueError, match="Duplicate prediction keys"):
        combine_module.combine_predictions(tmp_path)


@pytest.mark.parametrize("indices", [(1, 3), (1, 2, 3), (1, 1)])
def test_expected_batches_rejects_missing_or_stale_batches(combine_module, tmp_path: Path, indices):
    for index in indices:
        write_batch_payload(tmp_path, index, {str(index): {}})
    with pytest.raises(ValueError, match="Expected exactly"):
        combine_module.combine_predictions(tmp_path, expected_batches=2)
    assert not (tmp_path / "predictions.json").exists()


def test_hpc_directory_selection_and_dry_run(combine_module, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    run_dir = tmp_path / "data_processed" / "Treachery" / "predictions" / "segment" / "model__visual__pngs_c448"
    write_batch_payload(run_dir, 1, {"a.png": {}})
    args = ["--site", "Treachery", "--task", "segment", "--run", run_dir.name, "--expected-batches", "1"]
    output = combine_module.main([*args, "--dry-run"])
    assert output == run_dir / "predictions.json"
    assert not output.exists()
    monkeypatch.delenv("SCRATCH")
    combine_module.main([*args, "--data-processed-root", str(tmp_path / "data_processed")])
    assert json.loads(output.read_text())["count"] == 1
    combine_module.main(["--output-dir", str(run_dir)])


def test_failed_write_preserves_previous_output(combine_module, tmp_path: Path, monkeypatch):
    write_batch_payload(tmp_path, 1, {"a": {}})
    output = tmp_path / "predictions.json"
    output.write_text("previous result")

    def fail_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(combine_module.json, "dump", fail_dump)
    with pytest.raises(OSError, match="disk full"):
        combine_module.combine_predictions(tmp_path)
    assert output.read_text() == "previous result"
    assert not list(tmp_path.glob(".predictions.json.*.tmp"))


def test_rejects_non_object_record(combine_module, tmp_path: Path):
    write_batch_payload(tmp_path, 1, {"a": []})
    with pytest.raises(ValueError, match="must be a JSON object"):
        combine_module.combine_predictions(tmp_path)


def test_rejects_skipped_images_without_overwriting(combine_module, tmp_path: Path):
    write_batch_payload(tmp_path, 1, {"a": {}})
    output = tmp_path / "predictions.json"
    output.write_text("previous result")
    with pytest.raises(ValueError, match="Expected 2 prediction records, found 1"):
        combine_module.combine_predictions(tmp_path, expected_batches=1, expected_count=2)
    assert output.read_text() == "previous result"
