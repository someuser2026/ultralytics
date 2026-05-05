from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics.engine.results import Results


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "jobs" / "infer" / "yolo_site_predict_json.py"


def load_script_module():
    """Load the standalone site prediction script as a Python module."""
    spec = importlib.util.spec_from_file_location("yolo_site_predict_json", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inference_module():
    """Provide a fresh copy of the standalone script module for each test."""
    return load_script_module()


def make_obb_result(image_path: Path) -> Results:
    """Create one small OBB result payload for JSON export tests."""
    return Results(
        orig_img=np.zeros((32, 48, 3), dtype=np.uint8),
        path=str(image_path),
        names={0: "background", 1: "rip"},
        obb=torch.tensor([[24.0, 16.0, 10.0, 6.0, 0.0, 0.91, 1.0]], dtype=torch.float32),
    )


def write_run_layout(tmp_path: Path, *, task: str = "obb", run_name: str = "demo_run") -> Path:
    """Create a checkpoint layout matching <run_name>/weights/best.pt."""
    run_dir = tmp_path / "scratch" / "runs" / "cuda" / task / "imgsz_448" / "yolo" / "project_x" / run_name
    weights_path = run_dir / "weights" / "best.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"weights")
    return weights_path


class FakeYOLO:
    def __init__(self, weights: str, *, failures: dict[str, Exception] | None = None):
        self.weights = weights
        self.failures = failures or {}
        self.predict_calls = []

    def predict(self, source, stream: bool = True, **kwargs):
        image_path = Path(source)
        self.predict_calls.append({"source": source, "stream": stream, "kwargs": kwargs})
        if image_path.name in self.failures:
            raise self.failures[image_path.name]
        return [make_obb_result(image_path)]


def test_derive_run_name_and_task_from_checkpoint(inference_module, tmp_path: Path):
    weights_path = write_run_layout(tmp_path, task="segment", run_name="seed42_run")

    assert inference_module.derive_run_name(weights_path) == "seed42_run"
    assert inference_module.derive_task(weights_path) == "segment"


def test_resolve_image_and_output_dir(inference_module, tmp_path: Path):
    scratch = tmp_path / "scratch"
    image_dir = inference_module.resolve_image_dir(scratch, "Treachery", "visual/pngs/images_c448_ov35_kf20")
    output_dir = inference_module.derive_output_dir(
        scratch,
        "Treachery",
        "obb",
        "demo_run",
        "visual/pngs/images_c448_ov35_kf20",
    )

    assert image_dir == scratch / "data_processed" / "Treachery" / "PSScene" / "visual" / "pngs" / "images_c448_ov35_kf20"
    assert output_dir == (
        scratch
        / "data_processed"
        / "Treachery"
        / "predictions"
        / "obb"
        / "demo_run__visual__pngs__images_c448_ov35_kf20"
    )


def test_main_writes_aggregate_predictions_json_without_txt_outputs(inference_module, tmp_path: Path, monkeypatch):
    scratch = tmp_path / "scratch"
    image_dir = scratch / "data_processed" / "Treachery" / "PSScene" / "visual" / "pngs" / "images_c448_ov35_kf20"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "a.png").write_bytes(b"")
    (image_dir / "b.png").write_bytes(b"")
    (image_dir / "ignore.jpg").write_bytes(b"")

    weights_path = write_run_layout(tmp_path, task="obb", run_name="demo_run")
    fake_model = FakeYOLO(str(weights_path))

    monkeypatch.setenv("SCRATCH", str(scratch))
    monkeypatch.setattr(inference_module, "YOLO", lambda weights: fake_model)
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference_module.torch.cuda, "empty_cache", lambda: None)

    output_dir = inference_module.main(
        [
            "--ckpt",
            str(weights_path),
            "--site-name",
            "Treachery",
            "--img-dir",
            "visual/pngs/images_c448_ov35_kf20",
        ]
    )

    payload = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))

    assert output_dir.name == "demo_run__visual__pngs__images_c448_ov35_kf20"
    assert payload["count"] == 2
    assert set(payload["predictions"]) == {"a", "b"}
    assert not (output_dir / "labels").exists()
    assert list(output_dir.rglob("*.txt")) == []
    assert not (output_dir / "failed_images.txt").exists()
    assert len(fake_model.predict_calls) == 2


def test_main_logs_failed_images_and_keeps_successful_json(inference_module, tmp_path: Path, monkeypatch):
    scratch = tmp_path / "scratch"
    image_dir = scratch / "data_processed" / "Shipstern" / "PSScene" / "tiles"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "ok.png").write_bytes(b"")
    (image_dir / "fail.png").write_bytes(b"")

    weights_path = write_run_layout(tmp_path, task="obb", run_name="demo_run")
    fake_model = FakeYOLO(str(weights_path), failures={"fail.png": RuntimeError("bad image")})

    monkeypatch.setenv("SCRATCH", str(scratch))
    monkeypatch.setattr(inference_module, "YOLO", lambda weights: fake_model)
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference_module.torch.cuda, "empty_cache", lambda: None)

    output_dir = inference_module.main(
        [
            "--ckpt",
            str(weights_path),
            "--site-name",
            "Shipstern",
            "--img-dir",
            "tiles",
        ]
    )

    payload = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
    failed_images = (output_dir / "failed_images.txt").read_text(encoding="utf-8").splitlines()

    assert payload["count"] == 1
    assert set(payload["predictions"]) == {"ok"}
    assert failed_images == ["fail.png"]
    assert list(output_dir.rglob("*.txt")) == [output_dir / "failed_images.txt"]


def test_main_raises_immediately_on_oom(inference_module, tmp_path: Path, monkeypatch):
    scratch = tmp_path / "scratch"
    image_dir = scratch / "data_processed" / "Shipstern" / "PSScene" / "tiles"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "oom.png").write_bytes(b"")

    weights_path = write_run_layout(tmp_path, task="obb", run_name="demo_run")
    fake_model = FakeYOLO(str(weights_path), failures={"oom.png": RuntimeError("CUDA out of memory")})

    monkeypatch.setenv("SCRATCH", str(scratch))
    monkeypatch.setattr(inference_module, "YOLO", lambda weights: fake_model)
    monkeypatch.setattr(inference_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference_module.torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        inference_module.main(
            [
                "--ckpt",
                str(weights_path),
                "--site-name",
                "Shipstern",
                "--img-dir",
                "tiles",
            ]
        )
