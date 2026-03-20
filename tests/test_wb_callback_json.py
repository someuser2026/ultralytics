import json
from pathlib import Path

import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.utils.callbacks.wb import _save_predictions_json


def test_save_predictions_json_writes_obb_predictions_and_manifest(tmp_path):
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

    prediction_path = output_dir / "nested" / "example.json"
    payload = json.loads(prediction_path.read_text())
    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert payload["task"] == "obb"
    assert payload["image_name"] == "example.png"
    assert payload["orig_shape"] == {"height": 32, "width": 48}
    assert payload["json_file"] == str(Path("nested") / "example.json")
    assert len(payload["predictions"]) == 1
    assert set(payload["predictions"][0]["box"]) == {f"{axis}{i}" for axis in ("x", "y") for i in range(1, 5)}
    assert all(0.0 <= value <= 1.0 for value in payload["predictions"][0]["box"].values())
    assert manifest["count"] == 1
    assert manifest["predictions"][0]["json_file"] == str(Path("nested") / "example.json")
