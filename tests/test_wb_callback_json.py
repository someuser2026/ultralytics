import json
from pathlib import Path

import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.utils.callbacks.wb import _resolve_split_source, _save_predictions_json


def test_save_predictions_json_writes_one_aggregate_obb_json(tmp_path):
    split_root = tmp_path / "images" / "val"
    image_path = split_root / "nested" / "example.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"")

    result = Results(
        orig_img=np.zeros((32, 48, 3), dtype=np.uint8),
        path=str(image_path),
        names={0: "background", 1: "rip"},
        obb=torch.tensor([[24.0, 16.0, 10.0, 6.0, 0.0, 0.91, 1.0]], dtype=torch.float32),
    )

    output_dir = tmp_path / "predictions" / "val"
    _save_predictions_json([result], output_dir, source_root=split_root)

    payload = json.loads((output_dir / "predictions.json").read_text())
    entry = payload["predictions"]["nested/example"]

    assert payload["count"] == 1
    assert set(payload["predictions"]) == {"nested/example"}
    assert entry["task"] == "obb"
    assert entry["image_name"] == "example.png"
    assert entry["image_stem"] == "example"
    assert entry["orig_shape"] == {"height": 32, "width": 48}
    assert len(entry["predictions"]) == 1
    assert set(entry["predictions"][0]["box"]) == {f"{axis}{i}" for axis in ("x", "y") for i in range(1, 5)}
    assert all(0.0 <= value <= 1.0 for value in entry["predictions"][0]["box"].values())


def test_save_predictions_json_falls_back_when_summary_raises(tmp_path):
    split_root = tmp_path / "images" / "test"
    image_path = split_root / "example.png"
    split_root.mkdir(parents=True)
    image_path.write_bytes(b"")

    result = Results(
        orig_img=np.zeros((32, 48, 3), dtype=np.uint8),
        path=str(image_path),
        names={0: "background", 1: "rip"},
        obb=torch.tensor([[24.0, 16.0, 10.0, 6.0, 0.0, 0.91, 1.0]], dtype=torch.float32),
    )
    result.summary = lambda normalize=True: (_ for _ in ()).throw(TypeError("string indices must be integers, not 'str'"))

    output_dir = tmp_path / "predictions" / "test"
    _save_predictions_json([result], output_dir, source_root=split_root)

    payload = json.loads((output_dir / "predictions.json").read_text())
    entry = payload["predictions"]["example"]

    assert payload["count"] == 1
    assert entry["task"] == "obb"
    assert len(entry["predictions"]) == 1
    assert entry["predictions"][0]["name"] == "rip"


def test_resolve_split_source_supports_lists_and_absolute_paths(tmp_path):
    root = tmp_path / "dataset"
    absolute = tmp_path / "abs" / "test.txt"
    resolved = _resolve_split_source(root, ["images/val", absolute])

    assert resolved == [root / "images/val", absolute]
