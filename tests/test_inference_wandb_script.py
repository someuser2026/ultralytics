from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ultralytics.engine.results import Results


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "jobs" / "infer" / "log_predictions_to_wandb.py"


def load_script_module():
    """Load the standalone inference export script as a Python module."""
    spec = importlib.util.spec_from_file_location("log_predictions_to_wandb", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inference_module():
    """Provide a fresh copy of the standalone script module for each test."""
    return load_script_module()


class FakeLoggedArtifact:
    def wait(self):
        """Mirror the W&B artifact wait API."""


class FakeArtifact:
    def __init__(self, name: str, artifact_type: str):
        self.name = name
        self.type = artifact_type
        self.added_dirs = []

    def add_dir(self, path: str):
        self.added_dirs.append(Path(path))


class FakeRun:
    def __init__(self, name: str):
        self.name = name
        self.finished = False

    def finish(self):
        self.finished = True


class FakeWandb:
    __version__ = "0.test"

    def __init__(self, *, fail_resume: bool = False):
        self.fail_resume = fail_resume
        self.init_calls = []
        self.artifacts = []
        self.logged_artifacts = []
        self.run = None

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail_resume and kwargs.get("id"):
            raise RuntimeError("resume failed")
        self.run = FakeRun(kwargs.get("name", "unnamed"))
        return self.run

    def Artifact(self, name: str, type: str):
        artifact = FakeArtifact(name, type)
        self.artifacts.append(artifact)
        return artifact

    def log_artifact(self, artifact):
        self.logged_artifacts.append(artifact)
        return FakeLoggedArtifact()


class FakeYOLO:
    def __init__(self, weights: str):
        self.weights = weights
        self.task = "obb"
        self.predict_calls = []

    def predict(self, source, stream: bool = True, **kwargs):
        self.predict_calls.append({"source": source, "stream": stream, "kwargs": kwargs})
        if isinstance(source, (list, tuple)):
            source = source[0]
        source_path = Path(source)
        image_path = next(source_path.rglob("*.png")) if source_path.is_dir() else source_path
        return [make_obb_result(image_path)]


def make_obb_result(image_path: Path) -> Results:
    """Create one small OBB result payload for JSON export tests."""
    return Results(
        orig_img=np.zeros((32, 48, 3), dtype=np.uint8),
        path=str(image_path),
        names={0: "background", 1: "rip"},
        obb=torch.tensor([[24.0, 16.0, 10.0, 6.0, 0.0, 0.91, 1.0]], dtype=torch.float32),
    )


def write_dataset_yaml(
    root: Path,
    *,
    include_val: bool = True,
    include_test: bool = True,
    test_as_list: bool = False,
) -> tuple[Path, Path, Path]:
    """Create a small dataset layout and corresponding YAML file."""
    val_dir = root / "images" / "val"
    test_dir = root / "images" / "test"
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "sample.png").write_bytes(b"")
    (test_dir / "sample.png").write_bytes(b"")

    payload = {"path": str(root)}
    if include_val:
        payload["val"] = "images/val"
    if include_test:
        payload["test"] = ["images/test"] if test_as_list else "images/test"

    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return data_yaml, val_dir, test_dir


def write_run_layout(tmp_path: Path, run_name: str = "demo_run") -> tuple[Path, Path]:
    """Create a checkpoint layout matching <run_name>/weights/best.pt."""
    run_dir = tmp_path / "scratch" / "runs" / "cuda" / "obb" / "imgsz_448" / "yolo" / "project_x" / run_name
    weights_path = run_dir / "weights" / "best.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"weights")
    return run_dir, weights_path


def test_infer_run_name_from_weights_path(inference_module, tmp_path: Path):
    run_dir, weights_path = write_run_layout(tmp_path, run_name="seed42_run")

    assert inference_module.infer_run_name(weights_path) == "seed42_run"
    assert run_dir.name == "seed42_run"


def test_build_run_context_recovers_saved_args(inference_module, tmp_path: Path):
    run_dir, weights_path = write_run_layout(tmp_path)
    data_yaml, _, _ = write_dataset_yaml(tmp_path / "dataset")
    args_yaml = {
        "project": "/srv/scratch/z5428587/runs/cuda/obb/imgsz_448/yolo/project_x",
        "task": "obb",
        "batch": 8,
        "imgsz": 448,
        "device": "0",
    }
    (run_dir / "args.yaml").write_text(yaml.safe_dump(args_yaml), encoding="utf-8")

    context = inference_module.build_run_context(weights_path, data_yaml, saved_args=args_yaml, model_task="segment")

    assert context["run_name"] == "demo_run"
    assert context["project_path"] == Path(args_yaml["project"])
    assert context["task"] == "obb"
    assert context["batch"] == 8
    assert context["imgsz"] == 448
    assert context["device"] == "0"
    assert context["wandb_project"] == "imgsz_448-obb-project_x"


def test_build_run_context_falls_back_without_args_yaml(inference_module, tmp_path: Path):
    run_dir, weights_path = write_run_layout(tmp_path)
    data_yaml, _, _ = write_dataset_yaml(tmp_path / "dataset")

    context = inference_module.build_run_context(
        weights_path,
        data_yaml,
        device="cpu",
        batch=4,
        imgsz=640,
        model_task="segment",
    )

    assert context["run_name"] == "demo_run"
    assert context["project_path"] == run_dir.parent
    assert context["task"] == "segment"
    assert context["device"] == "cpu"
    assert context["batch"] == 4
    assert context["imgsz"] == 640


def test_build_run_context_raises_when_task_cannot_be_inferred(inference_module, tmp_path: Path):
    _, weights_path = write_run_layout(tmp_path)
    data_yaml, _, _ = write_dataset_yaml(tmp_path / "dataset")

    with pytest.raises(ValueError, match="Could not infer task"):
        inference_module.build_run_context(weights_path, data_yaml, model_task=None)


def test_load_data_split_sources_supports_relative_absolute_and_lists(inference_module, tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    absolute_test = tmp_path / "absolute" / "test"
    absolute_test.mkdir(parents=True)
    data_yaml = dataset_root / "data.yaml"
    dataset_root.mkdir(parents=True)
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "val": "images/val",
                "test": ["images/test", str(absolute_test)],
            }
        ),
        encoding="utf-8",
    )

    _, val_source, test_source = inference_module.load_data_split_sources(data_yaml)

    assert val_source == dataset_root / "images" / "val"
    assert test_source == [dataset_root / "images" / "test", absolute_test]


def test_save_predictions_json_matches_callback_shape(inference_module, tmp_path: Path):
    split_root = tmp_path / "images" / "val"
    image_path = split_root / "nested" / "example.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"")

    output_dir = tmp_path / "predictions" / "val"
    inference_module.save_predictions_json([make_obb_result(image_path)], output_dir, source_root=split_root)

    payload = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
    entry = payload["predictions"]["nested/example"]

    assert payload["count"] == 1
    assert entry["task"] == "obb"
    assert entry["image_name"] == "example.png"
    assert entry["image_stem"] == "example"
    assert entry["orig_shape"] == {"height": 32, "width": 48}
    assert len(entry["predictions"]) == 1


def test_initialize_wandb_run_uses_default_inference_name(inference_module, tmp_path: Path):
    run_dir, weights_path = write_run_layout(tmp_path)
    data_yaml, _, _ = write_dataset_yaml(tmp_path / "dataset")
    fake_wandb = FakeWandb()

    context = inference_module.build_run_context(weights_path, data_yaml, model_task="obb")
    run, active_name = inference_module.initialize_wandb_run(context, wandb_module=fake_wandb)

    assert fake_wandb.init_calls[0]["name"] == "demo_run_inference"
    assert active_name == "demo_run_inference"
    assert run.name == "demo_run_inference"
    assert context["project_path"] == run_dir.parent


def test_initialize_wandb_run_raises_on_failed_resume(inference_module, tmp_path: Path):
    _, weights_path = write_run_layout(tmp_path)
    data_yaml, _, _ = write_dataset_yaml(tmp_path / "dataset")
    fake_wandb = FakeWandb(fail_resume=True)
    context = inference_module.build_run_context(weights_path, data_yaml, model_task="obb")

    with pytest.raises(RuntimeError, match="Failed to resume W&B run 'abc123'"):
        inference_module.initialize_wandb_run(context, wandb_run_id="abc123", wandb_module=fake_wandb)


def test_log_run_context_prints_resolved_details(inference_module, tmp_path: Path, monkeypatch):
    _, weights_path = write_run_layout(tmp_path)
    data_yaml, _, _ = write_dataset_yaml(tmp_path / "dataset")
    context = inference_module.build_run_context(weights_path, data_yaml, model_task="segment")
    messages = []
    monkeypatch.setattr(inference_module.LOGGER, "info", lambda message: messages.append(str(message)))

    inference_module.log_run_context(context, "demo_run_inference", resumed_original_run=False)

    output = "\n".join(messages)
    assert "Standalone W&B Inference Export" in output
    assert "Model weights:" in output
    assert "Task: segment" in output
    assert "W&B run name: demo_run_inference" in output
    assert str(context["run_dir"] / "predictions" / "val") in output


def test_run_inference_exports_logs_active_run_artifacts_and_preserves_local_exports(inference_module, tmp_path: Path):
    run_dir, weights_path = write_run_layout(tmp_path)
    data_yaml, _, _ = write_dataset_yaml(tmp_path / "dataset")
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump({"project": str(run_dir.parent), "task": "obb", "batch": 2, "imgsz": 448, "device": "0"}),
        encoding="utf-8",
    )
    fake_wandb = FakeWandb()
    args = inference_module.parse_args(["--weights", str(weights_path), "--data", str(data_yaml)])

    result = inference_module.run_inference_exports(args, wandb_module=fake_wandb, model_cls=FakeYOLO)

    assert result["active_run_name"] == "demo_run_inference"
    assert result["exported_subsets"] == ["val", "test"]
    assert [artifact.name for artifact in fake_wandb.artifacts] == [
        "demo_run_inference_predictions_val",
        "demo_run_inference_predictions_test",
    ]
    assert (run_dir / "predictions" / "val" / "predictions.json").is_file()
    assert (run_dir / "predictions" / "test" / "predictions.json").is_file()
    assert fake_wandb.run is not None and fake_wandb.run.finished is True


def test_run_inference_exports_skips_missing_test_cleanly(inference_module, tmp_path: Path):
    _, weights_path = write_run_layout(tmp_path)
    data_yaml, _, _ = write_dataset_yaml(tmp_path / "dataset", include_test=False)
    fake_wandb = FakeWandb()
    args = inference_module.parse_args(["--weights", str(weights_path), "--data", str(data_yaml)])

    result = inference_module.run_inference_exports(args, wandb_module=fake_wandb, model_cls=FakeYOLO)

    assert result["exported_subsets"] == ["val"]
    assert [artifact.name for artifact in fake_wandb.artifacts] == ["demo_run_inference_predictions_val"]
