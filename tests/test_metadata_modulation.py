from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tests import TMP
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import (
    DEFAULT_METADATA_FIELDS,
    build_metadata_root_mappings,
    check_det_dataset,
    encode_metadata_properties,
    resolve_metadata_path,
)
from ultralytics.nn.modules import FPN
from ultralytics.nn.tasks import OBBModel, SegmentationModel

ULTRA_READY = find_spec("cv2") is not None and find_spec("torch") is not None
METADATA_DIM = len(DEFAULT_METADATA_FIELDS) + 2  # two azimuth fields expand to sin/cos pairs


def _write_image(path: Path, shape: tuple[int, int, int] = (32, 32, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full(shape, 127, dtype=np.uint8))


def _write_metadata_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "properties": {
                    "anomalous_pixels": 9,
                    "clear_confidence_percent": 80,
                    "clear_percent": 75.0,
                    "cloud_cover": 0.125,
                    "cloud_percent": 12.5,
                    "ground_control": True,
                    "gsd": 3.5,
                    "heavy_haze_percent": 0.0,
                    "light_haze_percent": 5.0,
                    "pixel_resolution": 3,
                    "satellite_azimuth": 90.0,
                    "shadow_percent": 1.0,
                    "snow_ice_percent": 0.0,
                    "sun_azimuth": 180.0,
                    "sun_elevation": 45.0,
                    "view_angle": 30.0,
                    "visible_confidence_percent": 70.0,
                    "udm2_confidence_mean": 60.0,
                    "unusable_pixels_percent": 15.0,
                }
            }
        ),
        encoding="utf-8",
    )


def _make_detection_dataset_root(name: str) -> Path:
    root = TMP / name
    for split in ("train", "val"):
        image_path = root / "images" / split / "nested" / "sample.jpg"
        label_path = root / "labels" / split / "nested" / "sample.txt"
        metadata_path = root / "metadata" / split / "nested" / "sample.json"
        _write_image(image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        _write_metadata_json(metadata_path)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "metadata: metadata",
                "names:",
                "  0: foreground",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def _metadata_enabled_obb_cfg(mode: str) -> dict:
    return {
        "nc": 1,
        "stride": 64,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
        ],
        "head": [
            [[1, 2, 3], 1, "FPN", [32, {"metadata_cfg": {"enabled": True, "mode": mode, "hidden_dim": 16}}]],
            [4, 1, "Index", [0]],
            [4, 1, "Index", [1]],
            [4, 1, "Index", [2]],
            [[5, 6, 7], 1, "OBB", [1, 1]],
        ],
    }


def _metadata_enabled_segment_cfg(mode: str) -> dict:
    return {
        "nc": 1,
        "stride": 64,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
        ],
        "head": [
            [[1, 2, 3], 1, "FPN", [32, {"metadata_cfg": {"enabled": True, "mode": mode, "hidden_dim": 16}}]],
            [4, 1, "Index", [0]],
            [4, 1, "Index", [1]],
            [4, 1, "Index", [2]],
            [[5, 6, 7], 1, "Segment", [1, 8, 32]],
        ],
    }


def _build_obb_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "img": torch.randn(batch_size, 3, 64, 64),
        "batch_idx": torch.zeros((1, 1), dtype=torch.float32),
        "cls": torch.zeros((1, 1), dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.2, 0.0]], dtype=torch.float32),
        "cls_probs": torch.ones((1, 1), dtype=torch.float32),
        "metadata_vec": torch.randn(batch_size, METADATA_DIM),
    }


def _build_segment_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    masks = torch.zeros((1, 64, 64), dtype=torch.float32)
    masks[0, 20:44, 18:42] = 1.0
    return {
        "img": torch.randn(batch_size, 3, 64, 64),
        "batch_idx": torch.zeros((1, 1), dtype=torch.float32),
        "cls": torch.zeros((1, 1), dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.375, 0.375]], dtype=torch.float32),
        "cls_probs": torch.ones((1, 1), dtype=torch.float32),
        "masks": masks,
        "metadata_vec": torch.randn(batch_size, METADATA_DIM),
    }


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_metadata_path_resolution() -> None:
    """Resolve JSON metadata sidecars by mirrored relative path and stem."""
    root = TMP / "metadata_paths"
    image_root = root / "images" / "val"
    metadata_root = root / "metadata"
    image_path = image_root / "nested" / "sample.jpg"
    _write_image(image_path)

    data = {"val": str(image_root), "metadata": str(metadata_root)}
    resolved = resolve_metadata_path(image_path, build_metadata_root_mappings(data), required=True)

    assert resolved["split"] == "val"
    assert Path(resolved["metadata_file"]) == metadata_root / "val" / "nested" / "sample.json"


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_check_det_dataset_accepts_metadata_root_folders() -> None:
    """Dataset parsing should expand metadata root folders to split-specific paths."""
    data_yaml = _make_detection_dataset_root("metadata_yaml")
    data = check_det_dataset(str(data_yaml), autodownload=False)

    assert Path(data["metadata"]["train"]).name == "train"
    assert Path(data["metadata"]["val"]).name == "val"
    assert data["metadata_fields"] == list(DEFAULT_METADATA_FIELDS)


def test_encode_metadata_properties_matches_expected_vector() -> None:
    """Metadata encoding should apply the documented per-field transforms."""
    properties = {
        "anomalous_pixels": 9,
        "clear_confidence_percent": 80,
        "clear_percent": 75.0,
        "cloud_cover": 0.125,
        "cloud_percent": 12.5,
        "ground_control": True,
        "gsd": 3.5,
        "heavy_haze_percent": 0.0,
        "light_haze_percent": 5.0,
        "pixel_resolution": 3,
        "satellite_azimuth": 90.0,
        "shadow_percent": 1.0,
        "snow_ice_percent": 0.0,
        "sun_azimuth": 180.0,
        "sun_elevation": 45.0,
        "view_angle": 30.0,
        "visible_confidence_percent": 70.0,
        "udm2_confidence_mean": 60.0,
        "unusable_pixels_percent": 15.0,
    }

    encoded = encode_metadata_properties(properties)

    assert encoded.shape == (METADATA_DIM,)
    assert encoded[0] == pytest.approx(np.log1p(9.0))
    assert encoded[1] == pytest.approx(0.80)
    assert encoded[3] == pytest.approx(0.125)
    assert encoded[5] == pytest.approx(1.0)
    assert encoded[10] == pytest.approx(1.0, abs=1e-6)  # sin(90°)
    assert encoded[11] == pytest.approx(0.0, abs=1e-6)  # cos(90°)
    assert encoded[14] == pytest.approx(0.0, abs=1e-6)  # sin(180°)
    assert encoded[15] == pytest.approx(-1.0, abs=1e-6)  # cos(180°)
    assert encoded[16] == pytest.approx(0.5)
    assert encoded[17] == pytest.approx(30.0 / 90.0)


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_dataset_collates_metadata_vec() -> None:
    """Dataset samples should expose metadata_vec and stack it in collate_fn."""
    data_yaml = _make_detection_dataset_root("metadata_dataset")
    data = check_det_dataset(str(data_yaml), autodownload=False)
    hyp = SimpleNamespace(mask_ratio=4, overlap_mask=False, bgr=0.0)
    dataset = YOLODataset(
        img_path=data["train"],
        imgsz=32,
        batch_size=1,
        augment=False,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="test: ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
    )

    sample = dataset[0]
    batch = dataset.collate_fn([sample, sample])

    assert sample["metadata_vec"].shape == (METADATA_DIM,)
    assert batch["metadata_vec"].shape == (2, METADATA_DIM)


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
@pytest.mark.parametrize("mode", ["film_affine", "film_gate"])
def test_fpn_metadata_modulation_changes_outputs(mode: str) -> None:
    """Metadata-conditioned FPN outputs should change when metadata changes."""
    torch.manual_seed(0)
    inputs = [
        torch.randn(1, 128, 32, 32),
        torch.randn(1, 256, 16, 16),
        torch.randn(1, 512, 8, 8),
    ]
    neck = FPN([128, 256, 512], 64, {"metadata_cfg": {"enabled": True, "mode": mode, "hidden_dim": 16}})
    zero_meta = torch.zeros(1, METADATA_DIM)
    one_meta = torch.ones(1, METADATA_DIM)

    outs_zero = neck(inputs, metadata_vec=zero_meta)
    outs_one = neck(inputs, metadata_vec=one_meta)

    assert all(torch.isfinite(out).all() for out in outs_zero)
    assert outs_zero[0].shape == (1, 64, 32, 32)
    assert any(not torch.allclose(a, b) for a, b in zip(outs_zero, outs_one))


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_metadata_enabled_obb_model_runs_loss() -> None:
    """A metadata-conditioned OBB model should run forward/loss on a synthetic batch."""
    model = OBBModel(_metadata_enabled_obb_cfg("film_affine"), ch=3, nc=1, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, angle_mode="oc")

    loss, loss_items = model(_build_obb_batch())

    assert torch.isfinite(loss).all()
    assert torch.isfinite(loss_items).all()


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_metadata_enabled_segment_model_runs_loss() -> None:
    """A metadata-conditioned segment model should run forward/loss on a synthetic batch."""
    model = SegmentationModel(_metadata_enabled_segment_cfg("film_gate"), ch=3, nc=1, verbose=False)
    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
        overlap_mask=False,
        mask_ratio=4,
        mask_weight=1.0,
        bgr=0.0,
        seg_use_mixed_loss=False,
        use_soft_ignore_band=False,
        imgsz=64,
        use_shoreline_prior_loss=False,
        use_land_water_prior_loss=False,
    )

    loss, loss_items = model(_build_segment_batch())

    assert torch.isfinite(loss).all()
    assert torch.isfinite(loss_items).all()


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_metadata_enabled_model_raises_when_metadata_missing() -> None:
    """A metadata-conditioned YOLO model should reject train-time batches without metadata_vec."""
    model = OBBModel(_metadata_enabled_obb_cfg("film_affine"), ch=3, nc=1, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, angle_mode="oc")
    batch = _build_obb_batch()
    batch.pop("metadata_vec")

    with pytest.raises(ValueError, match="metadata-conditioned neck"):
        model(batch)
