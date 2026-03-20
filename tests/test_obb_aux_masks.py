from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from tests import TMP
from ultralytics.data.augment import PrepareAuxiliaryMaskInputs
from ultralytics.data.utils import build_auxiliary_root_mappings, resolve_auxiliary_mask_paths
from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils.loss import _compute_obb_spatial_prior_losses


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


def test_auxiliary_mask_path_resolution() -> None:
    """Resolve shoreline and land/water sidecars by mirrored relative path and stem."""
    root = TMP / "obb_aux_paths"
    image_root = root / "images" / "val"
    shore_root = root / "masks" / "shoreline" / "val"
    land_root = root / "masks" / "land_water" / "val"
    image_path = image_root / "nested" / "sample.jpg"
    _write_png(image_path, np.zeros((4, 4, 3), dtype=np.uint8))

    data = {
        "val": str(image_root),
        "shoreline_masks": {"val": str(shore_root)},
        "land_water_masks": {"val": str(land_root)},
    }
    resolved = resolve_auxiliary_mask_paths(image_path, build_auxiliary_root_mappings(data), True, True)

    assert resolved["split"] == "val"
    assert Path(resolved["shoreline_mask_file"]) == shore_root / "nested" / "sample.png"
    assert Path(resolved["land_water_mask_file"]) == land_root / "nested" / "sample.png"


def test_prepare_auxiliary_mask_inputs_builds_channels_and_prior_maps() -> None:
    """Append shoreline and one-hot land/water channels and emit prior-loss tensors."""
    transform = PrepareAuxiliaryMaskInputs(
        use_shoreline_input=True,
        use_land_water_input=True,
        use_shoreline_prior_loss=True,
        use_land_water_prior_loss=True,
        shoreline_prior_max_dist=4,
    )
    shoreline_mask = np.zeros((5, 5), dtype=np.uint8)
    shoreline_mask[:, 2] = 1
    land_water_mask = np.full((5, 5), 2, dtype=np.uint8)
    land_water_mask[:, 0] = 1
    land_water_mask[0, 0] = 0

    labels = transform(
        {
            "img": np.zeros((5, 5, 3), dtype=np.uint8),
            "shoreline_mask": shoreline_mask,
            "land_water_mask": land_water_mask,
        }
    )

    assert labels["img"].shape == (5, 5, 6)
    assert torch.equal(labels["land_water_mask"], torch.from_numpy(land_water_mask[None].astype(np.int64)))
    assert labels["shoreline_distance_map"].shape == (1, 5, 5)
    assert labels["shoreline_distance_map"][0, 0, 0].item() == 0.0  # no-data ignored
    assert labels["shoreline_distance_map"][0, 2, 0].item() == 0.0  # land ignored
    assert labels["shoreline_distance_map"][0, 2, 4].item() > 0.0  # water away from shoreline is penalized


def test_spatial_prior_losses_support_land_and_closest_corner_modes() -> None:
    """Land priors penalize land-centered negatives and closest-corner mode reduces shoreline penalty."""
    land_water_mask = torch.zeros((1, 1, 8, 8), dtype=torch.long)
    land_water_mask[:, :, :, 1:8] = 2
    land_water_mask[:, :, :, 0] = 1

    shoreline_distance = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    shoreline_distance[:, :, :, :] = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)

    pred_rboxes = torch.tensor(
        [
            [
                [0.5, 3.5, 1.0, 1.0, 0.0],  # center on land
                [6.0, 3.5, 6.0, 2.0, 0.0],  # center far from shoreline, left corner near shoreline
            ]
        ],
        dtype=torch.float32,
    )
    conf_scores = torch.tensor([[0.8, 0.6]], dtype=torch.float32)
    negative_mask = torch.tensor([[True, True]])

    shoreline_center, land_penalty = _compute_obb_spatial_prior_losses(
        pred_rboxes,
        conf_scores,
        negative_mask,
        land_water_mask,
        shoreline_distance,
        point_mode="center",
        shoreline_prior_max_dist=8.0,
    )
    shoreline_corner, _ = _compute_obb_spatial_prior_losses(
        pred_rboxes,
        conf_scores,
        negative_mask,
        land_water_mask,
        shoreline_distance,
        point_mode="closest_corner",
        shoreline_prior_max_dist=8.0,
    )

    assert land_penalty.item() > 0.0
    assert shoreline_center.item() > shoreline_corner.item()


def test_predict_auxiliary_images_append_mask_channels() -> None:
    """Predict helper appends shoreline and land/water channels for file-based OBB sources."""
    root = TMP / "obb_aux_predict"
    image_root = root / "images" / "val"
    shore_root = root / "masks" / "shoreline" / "val"
    land_root = root / "masks" / "land_water" / "val"
    image_path = image_root / "sample.jpg"
    shoreline_path = shore_root / "sample.png"
    land_water_path = land_root / "sample.png"

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    shoreline = np.zeros((8, 8), dtype=np.uint8)
    shoreline[:, 3] = 255
    land_water = np.full((8, 8), 255, dtype=np.uint8)
    land_water[:, :2] = 128

    _write_png(image_path, image)
    _write_png(shoreline_path, shoreline)
    _write_png(land_water_path, land_water)

    predictor = BasePredictor(overrides={"task": "obb", "imgsz": 8, "batch": 1, "rect": False})
    predictor.model = SimpleNamespace(task="obb", pt=True, dynamic=False, imx=False, stride=32)
    predictor.imgsz = (8, 8)
    predictor.data = {
        "val": str(image_root),
        "shoreline_masks": {"val": str(shore_root)},
        "land_water_masks": {"val": str(land_root)},
    }
    predictor.args.use_shoreline_input = True
    predictor.args.use_land_water_input = True
    predictor.source_type = SimpleNamespace(stream=False, screenshot=False, from_img=False, tensor=False)
    predictor.dataset = SimpleNamespace(video_flag=[False])

    predictor._setup_auxiliary_predict_context()
    prepared = predictor._prepare_auxiliary_predict_images([str(image_path)], [image])[0]

    assert prepared.shape == (8, 8, 6)
    assert prepared[..., 3].max() == 255  # shoreline
    assert prepared[..., 4].max() == 255  # land
    assert prepared[..., 5].max() == 255  # water
