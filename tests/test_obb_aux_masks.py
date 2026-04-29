from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tests import TMP
from ultralytics.data.augment import PrepareAuxiliaryMaskInputs, RandomFlip
from ultralytics.data.utils import build_auxiliary_root_mappings, check_det_dataset, resolve_auxiliary_mask_paths
from ultralytics.engine.predictor import BasePredictor
from ultralytics.nn.tasks import OBBModel, SegmentationModel
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import _compute_obb_spatial_prior_losses, _compute_segmentation_spatial_prior_losses


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


def test_auxiliary_mask_path_resolution() -> None:
    """Resolve shoreline and land/water sidecars by mirrored relative path and stem."""
    root = TMP / "obb_aux_paths"
    image_root = root / "images" / "val"
    shore_root = root / "masks" / "shoreline"
    land_root = root / "masks" / "land_water"
    image_path = image_root / "nested" / "sample.jpg"
    _write_png(image_path, np.zeros((4, 4, 3), dtype=np.uint8))

    data = {
        "val": str(image_root),
        "shoreline_masks": str(shore_root),
        "land_water_masks": str(land_root),
    }
    resolved = resolve_auxiliary_mask_paths(image_path, build_auxiliary_root_mappings(data), True, True)

    assert resolved["split"] == "val"
    assert Path(resolved["shoreline_mask_file"]) == shore_root / "val" / "nested" / "sample.png"
    assert Path(resolved["land_water_mask_file"]) == land_root / "val" / "nested" / "sample.png"


def test_check_det_dataset_accepts_auxiliary_root_folders() -> None:
    """Dataset parsing should expand auxiliary root folders to split-specific paths."""
    root = TMP / "obb_aux_yaml"
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "masks" / "shoreline" / "train").mkdir(parents=True, exist_ok=True)
    (root / "masks" / "shoreline" / "val").mkdir(parents=True, exist_ok=True)
    (root / "masks" / "land_water" / "train").mkdir(parents=True, exist_ok=True)
    (root / "masks" / "land_water" / "val").mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "shoreline_masks: masks/shoreline",
                "land_water_masks: masks/land_water",
                "names:",
                "  0: foreground",
                "",
            ]
        ),
        encoding="utf-8",
    )

    data = check_det_dataset(str(data_yaml), autodownload=False)

    assert Path(data["shoreline_masks"]["train"]) == root / "masks" / "shoreline" / "train"
    assert Path(data["shoreline_masks"]["val"]) == root / "masks" / "shoreline" / "val"
    assert Path(data["land_water_masks"]["train"]) == root / "masks" / "land_water" / "train"
    assert Path(data["land_water_masks"]["val"]) == root / "masks" / "land_water" / "val"


def test_prepare_auxiliary_mask_inputs_builds_channels_and_prior_maps() -> None:
    """Append shoreline plus a single semantic land/water channel and emit prior-loss tensors."""
    transform = PrepareAuxiliaryMaskInputs(
        use_shoreline_input=True,
        use_land_water_input=True,
        use_shoreline_prior_loss=True,
        use_land_water_prior_loss=True,
        shoreline_prior_max_dist=4,
    )
    shoreline_mask = np.zeros((5, 5), dtype=np.uint8)
    shoreline_mask[:, 2] = 1
    land_water_mask = np.full((5, 5), 255, dtype=np.uint8)
    land_water_mask[:, 0] = 64
    land_water_mask[:, 1] = 128
    land_water_mask[0, 0] = 0

    labels = transform(
        {
            "img": np.zeros((5, 5, 3), dtype=np.uint8),
            "shoreline_mask": shoreline_mask,
            "land_water_mask": land_water_mask,
        }
    )

    assert labels["img"].shape == (5, 5, 5)
    assert set(np.unique(labels["img"][..., 4]).tolist()) == {0, 64, 128, 255}
    assert torch.equal(labels["land_water_mask"], torch.from_numpy(land_water_mask[None].astype(np.int64)))
    assert labels["shoreline_distance_map"].shape == (1, 5, 5)
    assert labels["shoreline_distance_map"][0, 0, 0].item() > 0.0
    assert labels["shoreline_distance_map"][0, 2, 0].item() > 0.0
    assert labels["shoreline_distance_map"][0, 2, 1].item() > 0.0
    assert labels["shoreline_distance_map"][0, 2, 4].item() > 0.0  # water away from shoreline is penalized


def test_auxiliary_mask_channels_follow_geometric_augmentation() -> None:
    """Shoreline and land/water inputs must match the exact geometric transforms applied to the image."""
    shoreline_mask = np.zeros((4, 6), dtype=np.uint8)
    shoreline_mask[:, 1] = 1
    land_water_mask = np.full((4, 6), 255, dtype=np.uint8)
    land_water_mask[:, :1] = 64
    land_water_mask[:, 1:2] = 128
    land_water_mask[0, 0] = 0
    encoded_land_water = land_water_mask.copy()

    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[..., 0] = shoreline_mask * 255
    img[..., 1] = encoded_land_water
    labels = {
        "img": img,
        "instances": Instances(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xywh",
            normalized=False,
        ),
        "shoreline_mask": shoreline_mask,
        "land_water_mask": land_water_mask,
    }

    labels = RandomFlip(p=1.0, direction="horizontal")(labels)
    expected_shoreline = labels["shoreline_mask"].copy()
    expected_land_water = labels["land_water_mask"].copy()
    transformed_img = labels["img"].copy()

    prepared = PrepareAuxiliaryMaskInputs(use_shoreline_input=True, use_land_water_input=True)(labels)

    assert prepared["img"].shape == (4, 6, 5)
    assert np.array_equal(prepared["img"][..., 3], expected_shoreline * 255)
    assert np.array_equal(prepared["img"][..., 4], expected_land_water.astype(np.uint8))
    assert np.array_equal(prepared["img"][..., 0], expected_shoreline * 255)
    assert np.array_equal(prepared["img"][..., 1], expected_land_water.astype(np.uint8))
    assert np.array_equal(transformed_img[..., 0], prepared["img"][..., 0])
    assert np.array_equal(transformed_img[..., 1], prepared["img"][..., 1])


def test_spatial_prior_losses_support_threshold_area_and_closest_corner_modes() -> None:
    """Land priors use thresholded area overlap and closest-corner mode reduces shoreline penalty."""
    land_water_mask = torch.zeros((1, 1, 8, 8), dtype=torch.long)
    land_water_mask[:, :, :, 1:8] = 192
    land_water_mask[:, :, :, 0] = 64

    shoreline_distance = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    shoreline_distance[:, :, :, :] = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)

    pred_rboxes = torch.tensor(
        [
            [
                [0.5, 3.5, 1.0, 1.0, 0.0],  # 100% land support
                [2.0, 3.5, 1.0, 1.0, 0.0],  # 0% land support
                [6.0, 3.5, 6.0, 2.0, 0.0],  # center far from shoreline, left corner near shoreline
            ]
        ],
        dtype=torch.float32,
    )
    conf_scores = torch.tensor([[0.8, 0.7, 0.6]], dtype=torch.float32)

    shoreline_center, land_penalty = _compute_obb_spatial_prior_losses(
        pred_rboxes,
        conf_scores,
        land_water_mask,
        shoreline_distance,
        point_mode="center",
        shoreline_prior_max_dist=8.0,
        land_threshold=0.05,
        land_beta=4.0,
    )
    shoreline_corner, _ = _compute_obb_spatial_prior_losses(
        pred_rboxes,
        conf_scores,
        land_water_mask,
        shoreline_distance,
        point_mode="closest_corner",
        shoreline_prior_max_dist=8.0,
        land_threshold=0.05,
        land_beta=4.0,
    )

    assert land_penalty.item() > 0.0
    assert shoreline_center.item() > shoreline_corner.item()

    land_low = _compute_obb_spatial_prior_losses(
        pred_rboxes[:, 1:2],
        conf_scores[:, 1:2],
        land_water_mask,
        shoreline_distance,
        shoreline_prior_max_dist=8.0,
        land_threshold=0.05,
        land_beta=4.0,
    )[1]
    assert land_low.item() == pytest.approx(0.0, abs=1e-6)


def test_segment_spatial_prior_losses_cover_all_predictions() -> None:
    """Segment priors use all predictions with thresholded land overlap and shoreline distance aggregation."""
    proto = torch.tensor(
        [
            [
                [
                        [-6.0, -6.0, 6.0, 6.0],
                        [-12.0, -12.0, 12.0, 12.0],
                        [-12.0, -12.0, 12.0, 12.0],
                        [-12.0, -12.0, 12.0, 12.0],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
    pred_masks = torch.tensor([[[1.0], [-1.0]]], dtype=torch.float32)
    pred_scores = torch.tensor([[0.9, 0.9]], dtype=torch.float32)
    land_water_mask = torch.tensor(
        [[[[64, 64, 192, 255], [64, 128, 192, 255], [64, 128, 192, 255], [64, 64, 192, 255]]]],
        dtype=torch.long,
    )
    shoreline_distance = torch.tensor(
        [[[[0.0, 0.0, 0.2, 0.8], [0.0, 0.0, 0.2, 0.8], [0.0, 0.0, 0.2, 0.8], [0.0, 0.0, 0.2, 0.8]]]],
        dtype=torch.float32,
    )

    shoreline_loss, land_loss = _compute_segmentation_spatial_prior_losses(
        pred_scores[:, :1],
        pred_masks[:, :1],
        proto,
        land_water_mask,
        shoreline_distance,
        shoreline_prior_max_dist=1.0,
        land_threshold=0.05,
        land_beta=4.0,
        chunk_size=1,
    )
    shoreline_land, land_high = _compute_segmentation_spatial_prior_losses(
        pred_scores[:, 1:],
        pred_masks[:, 1:],
        proto,
        land_water_mask,
        shoreline_distance,
        shoreline_prior_max_dist=1.0,
        land_threshold=0.05,
        land_beta=4.0,
        chunk_size=1,
    )

    assert shoreline_loss.item() > 0.0
    assert land_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert shoreline_land.item() < shoreline_loss.item() * 0.05
    assert land_high.item() > 0.0


def test_predict_auxiliary_images_append_mask_channels() -> None:
    """Predict helper appends shoreline and land/water channels for file-based OBB sources."""
    root = TMP / "obb_aux_predict"
    image_root = root / "images" / "val"
    shore_root = root / "masks" / "shoreline"
    land_root = root / "masks" / "land_water"
    image_path = image_root / "sample.jpg"
    shoreline_path = shore_root / "val" / "sample.png"
    land_water_path = land_root / "val" / "sample.png"

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    shoreline = np.zeros((8, 8), dtype=np.uint8)
    shoreline[:, 3] = 255
    land_water = np.full((8, 8), 255, dtype=np.uint8)
    land_water[:, :1] = 64
    land_water[:, 1:2] = 128
    land_water[:, 2:3] = 192

    _write_png(image_path, image)
    _write_png(shoreline_path, shoreline)
    _write_png(land_water_path, land_water)

    predictor = BasePredictor(overrides={"task": "obb", "imgsz": 8, "batch": 1, "rect": False})
    predictor.model = SimpleNamespace(task="obb", pt=True, dynamic=False, imx=False, stride=32)
    predictor.imgsz = (8, 8)
    predictor.data = {
        "val": str(image_root),
        "shoreline_masks": str(shore_root),
        "land_water_masks": str(land_root),
    }
    predictor.args.use_shoreline_input = True
    predictor.args.use_land_water_input = True
    predictor.source_type = SimpleNamespace(stream=False, screenshot=False, from_img=False, tensor=False)
    predictor.dataset = SimpleNamespace(video_flag=[False])

    predictor._setup_auxiliary_predict_context()
    prepared = predictor._prepare_auxiliary_predict_images([str(image_path)], [image])[0]

    assert prepared.shape == (8, 8, 5)
    assert prepared[..., 3].max() == 255  # shoreline
    assert 64 in np.unique(prepared[..., 4])  # land
    assert 128 in np.unique(prepared[..., 4])  # land
    assert 192 in np.unique(prepared[..., 4])  # water
    assert 255 in np.unique(prepared[..., 4])  # whitewater


def test_dual_branch_yolo12_models_build_with_aux_channels() -> None:
    """Dual-branch YOLO12 OBB and segment models should build and run with auxiliary channels."""
    obb_cfg = "ultralytics/cfg/models/yolo/obb/yolo12-pafpn-dual-obb.yaml"
    seg_cfg = "ultralytics/cfg/models/yolo/segment_no_p2/yolo12-pafpn-dual-segment.yaml"

    obb_model = OBBModel(obb_cfg, ch=5, nc=1, verbose=False)
    seg_model = SegmentationModel(seg_cfg, ch=5, nc=1, verbose=False)

    obb_out = obb_model.predict(torch.randn(1, 5, 64, 64))
    seg_out = seg_model.predict(torch.randn(1, 5, 64, 64))

    assert obb_out is not None
    assert seg_out is not None


def test_dual_branch_models_require_auxiliary_channels() -> None:
    """Dual-branch models should fail fast when there are no auxiliary channels to route."""
    obb_cfg = "ultralytics/cfg/models/yolo/obb/yolo12-pafpn-dual-obb.yaml"

    with pytest.raises(ValueError, match="ChannelSplit requires input channels > rgb_channels"):
        OBBModel(obb_cfg, ch=3, nc=1, verbose=False)
