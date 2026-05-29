from __future__ import annotations

import math
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import tifffile

from tests import TMP
from ultralytics.cfg import check_cfg
from ultralytics.data.augment import (
    Albumentations,
    LetterBox,
    Mosaic,
    PrepareAuxiliaryMaskInputs,
    RandomFlip,
    RandomPerspective,
    RandomUnsharpMask,
)
from ultralytics.data.build import load_inference_source
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import (
    check_det_dataset,
    get_auxiliary_mask_flags,
    verify_image,
    verify_image_label,
)
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.obb.train import on_train_epoch_start as obb_on_train_epoch_start
from ultralytics.models.yolo.segment.train import on_train_epoch_start as seg_on_train_epoch_start
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import OBBModel, SegmentationModel
from ultralytics.utils.instance import Instances
from ultralytics.utils.patches import imread
from ultralytics.utils.loss import (
    _compute_obb_spatial_prior_losses,
    _compute_segmentation_spatial_prior_losses,
    _compute_shoreline_aux_loss,
)

def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, array, metadata={"axes": "YX" if array.ndim == 2 else "YXS"})


def _band_image(
    shoreline: np.ndarray | None = None,
    land_water: np.ndarray | None = None,
    shoreline_distance: np.ndarray | None = None,
    shoreline_proximity: np.ndarray | None = None,
    dtype: np.dtype = np.uint8,
) -> np.ndarray:
    shape = next(x.shape for x in (shoreline, land_water, shoreline_distance, shoreline_proximity) if x is not None)
    img = np.zeros((*shape, 3), dtype=dtype)
    channels = [img[..., 0], img[..., 1], img[..., 2]]
    for band in (shoreline, land_water, shoreline_distance, shoreline_proximity):
        if band is not None:
            channels.append(band.astype(dtype, copy=False))
    return np.stack(channels, axis=2)


AUX_BANDS = {4: "shoreline", 5: "land_water", 6: "shoreline_distance", 7: "shoreline_proximity"}


def _empty_instances() -> Instances:
    return Instances(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0, 1000, 2), dtype=np.float32),
        bbox_format="xywh",
        normalized=False,
    )


def _empty_labels(img: np.ndarray) -> dict:
    return {
        "img": img,
        "cls": np.zeros((0, 1), dtype=np.float32),
        "instances": _empty_instances(),
        "im_file": "sample.tif",
        "ori_shape": img.shape[:2],
        "resized_shape": img.shape[:2],
    }


def _shoreaux_args(**overrides) -> SimpleNamespace:
    base = {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "angle_mode": "oc",
        "overlap_mask": False,
        "mask_ratio": 4,
        "mask_weight": 1.0,
        "bgr": 0.0,
        "seg_use_mixed_loss": False,
        "use_soft_ignore_band": False,
        "imgsz": 64,
        "use_shoreline_prior_loss": False,
        "use_land_water_prior_loss": False,
        "shoreline_aux_weight": 0.2,
        "active_shoreline_aux_weight": 0.2,
        "shoreline_aux_bce_weight": 1.0,
        "shoreline_aux_dice_weight": 1.0,
        "shoreline_aux_warmup_epochs": 10,
        "shoreline_prior_gt_margin": 0.05,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _build_shoreaux_obb_batch(empty: bool = False) -> dict[str, torch.Tensor]:
    field = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    if not empty:
        field[:, :, :, 31:33] = 1.0
    return {
        "img": torch.randn(1, 3, 64, 64),
        "batch_idx": torch.zeros((1, 1), dtype=torch.float32),
        "cls": torch.zeros((1, 1), dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.2, 0.0]], dtype=torch.float32),
        "cls_probs": torch.ones((1, 1), dtype=torch.float32),
        "shoreline_proximity_field": field,
    }


def _build_shoreaux_segment_batch(empty: bool = False) -> dict[str, torch.Tensor]:
    field = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    if not empty:
        field[:, :, :, 31:33] = 1.0
    masks = torch.zeros((1, 64, 64), dtype=torch.float32)
    masks[0, 20:44, 18:42] = 1.0
    return {
        "img": torch.randn(1, 3, 64, 64),
        "batch_idx": torch.zeros((1, 1), dtype=torch.float32),
        "cls": torch.zeros((1, 1), dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.375, 0.375]], dtype=torch.float32),
        "cls_probs": torch.ones((1, 1), dtype=torch.float32),
        "masks": masks,
        "shoreline_proximity_field": field,
    }


def test_check_det_dataset_normalizes_bands_and_infers_channels() -> None:
    """Dataset parsing should preserve bands metadata and infer raw channel count from it."""
    root = TMP / "obb_aux_yaml"
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "bands:",
                "  4: shoreline",
                "  5: land_water",
                "  6: shoreline_distance",
                "  7: shoreline_proximity",
                "names:",
                "  0: foreground",
                "",
            ]
        ),
        encoding="utf-8",
    )

    data = check_det_dataset(str(data_yaml), autodownload=False)

    assert data["bands"] == {4: "shoreline", 5: "land_water", 6: "shoreline_distance", 7: "shoreline_proximity"}
    assert data["channels"] == 7


def test_check_det_dataset_normalizes_band_scale_factors() -> None:
    """Dataset parsing should normalize per-band input scale divisors from YAML."""
    root = TMP / "obb_aux_yaml_scale_factors"
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "channels: 7",
                "bands:",
                "  4: shoreline",
                "  5: land_water",
                "  6: shoreline_distance",
                "  7: shoreline_proximity",
                "band_scale_factors:",
                "  '6': 103",
                "  '7': 65535",
                "names:",
                "  0: foreground",
                "",
            ]
        ),
        encoding="utf-8",
    )

    data = check_det_dataset(str(data_yaml), autodownload=False)

    assert data["band_scale_factors"] == {6: 103.0, 7: 65535.0}


def test_check_det_dataset_rejects_reserved_rgb_band_indices() -> None:
    """Semantic TIFF bands must not overwrite the first three RGB channels."""
    root = TMP / "obb_aux_yaml_reserved"
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "bands:",
                "  3: shoreline",
                "names:",
                "  0: foreground",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SyntaxError, match="reserved for RGB"):
        check_det_dataset(str(data_yaml), autodownload=False)


def test_check_det_dataset_rejects_channels_smaller_than_bands() -> None:
    """Raw channel count must cover the highest configured TIFF band."""
    root = TMP / "obb_aux_yaml_channels"
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "channels: 5",
                "bands:",
                "  6: shoreline_distance",
                "names:",
                "  0: foreground",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SyntaxError, match="at least 6 channels"):
        check_det_dataset(str(data_yaml), autodownload=False)


def test_prepare_auxiliary_mask_inputs_emit_prior_tensors_from_embedded_bands() -> None:
    """Embedded TIFF bands should drive prior-loss tensors without appending new image channels."""
    transform = PrepareAuxiliaryMaskInputs(
        bands={4: "shoreline", 5: "land_water", 6: "shoreline_distance", 7: "shoreline_proximity"},
        use_shoreline_input=True,
        use_land_water_input=True,
        use_shoreline_prior_loss=True,
        use_land_water_prior_loss=True,
        use_shoreline_aux_loss=True,
        shoreline_prior_max_dist=4,
    )
    shoreline_mask = np.zeros((5, 5), dtype=np.uint8)
    shoreline_mask[:, 2] = 255
    land_water_mask = np.full((5, 5), 255, dtype=np.uint8)
    land_water_mask[:, 0] = 64
    land_water_mask[:, 1] = 128
    land_water_mask[0, 0] = 0
    shoreline_distance = np.tile(np.arange(5, dtype=np.uint8), (5, 1))
    shoreline_proximity = np.zeros((5, 5), dtype=np.uint8)
    shoreline_proximity[:, 2] = 255

    img = _band_image(shoreline_mask, land_water_mask, shoreline_distance, shoreline_proximity)
    labels = transform({"img": img})

    assert labels["img"].shape == (5, 5, 7)
    assert torch.equal(labels["land_water_mask"], torch.from_numpy(land_water_mask[None].astype(np.int64)))
    assert labels["shoreline_distance_map"].shape == (1, 5, 5)
    assert labels["shoreline_distance_map"].dtype == torch.float32
    assert torch.equal(labels["shoreline_distance_map"][0], torch.from_numpy(shoreline_distance.astype(np.float32)))
    assert labels["shoreline_proximity_field"].shape == (1, 5, 5)
    assert labels["shoreline_proximity_field"].dtype == torch.float32
    assert labels["shoreline_proximity_field"][0, 0, 2].item() == pytest.approx(1.0, abs=1e-6)
    assert labels["shoreline_proximity_field"].max().item() == pytest.approx(1.0, abs=1e-6)


def test_prepare_auxiliary_mask_inputs_use_configured_proximity_scale_factor() -> None:
    """Configured band scale factors should normalize shoreline proximity fields from uint16 TIFF bands."""
    transform = PrepareAuxiliaryMaskInputs(
        bands={4: "shoreline_proximity"},
        band_scale_factors={4: 65535.0},
        use_shoreline_aux_loss=True,
    )
    shoreline_proximity = np.zeros((4, 4), dtype=np.uint16)
    shoreline_proximity[:, 1] = 65535
    shoreline_proximity[:, 2] = 32768

    labels = transform({"img": _band_image(shoreline_proximity=shoreline_proximity, dtype=np.uint16)})
    field = labels["shoreline_proximity_field"][0]

    assert field[:, 1].min().item() == pytest.approx(1.0, abs=1e-6)
    assert field[0, 2].item() == pytest.approx(32768.0 / 65535.0, abs=1e-6)


def test_prepare_auxiliary_mask_inputs_use_configured_distance_scale_factor() -> None:
    """Configured band scale factors should normalize shoreline distance maps used by prior losses."""
    transform = PrepareAuxiliaryMaskInputs(
        bands={4: "shoreline", 5: "land_water", 6: "shoreline_distance"},
        band_scale_factors={6: 8.0},
        use_shoreline_prior_loss=True,
    )
    shoreline_mask = np.zeros((4, 4), dtype=np.uint16)
    land_water_mask = np.full((4, 4), 255, dtype=np.uint16)
    shoreline_distance = np.full((4, 4), 24, dtype=np.uint16)

    labels = transform({"img": _band_image(shoreline_mask, land_water_mask, shoreline_distance, dtype=np.uint16)})
    distance = labels["shoreline_distance_map"][0]

    assert distance.dtype == torch.float32
    assert distance[0, 0].item() == pytest.approx(3.0, abs=1e-6)


def test_blank_shoreline_mask_produces_zero_distance_map() -> None:
    """Blank shoreline masks should suppress shoreline prior rather than max it out."""
    transform = PrepareAuxiliaryMaskInputs(
        bands={4: "shoreline", 5: "land_water"},
        use_shoreline_prior_loss=True,
        use_land_water_prior_loss=True,
        shoreline_prior_max_dist=8,
    )
    img = _band_image(np.zeros((4, 4), dtype=np.uint8), np.full((4, 4), 192, dtype=np.uint8))
    labels = transform({"img": img})

    assert torch.count_nonzero(labels["shoreline_distance_map"]) == 0


def test_shoreline_gaussian_field_straight_line_is_peak_on_shore_and_truncated() -> None:
    """Gaussian shoreline targets should peak on-shore, decay smoothly, and zero beyond the truncation radius."""
    transform = PrepareAuxiliaryMaskInputs(
        bands={4: "shoreline"},
        use_shoreline_aux_loss=True,
        shoreline_aux_gaussian_sigma_ratio=0.20,
        shoreline_aux_gaussian_truncate_sigmas=2.0,
    )
    shoreline_mask = np.zeros((9, 9), dtype=np.uint8)
    shoreline_mask[:, 4] = 255
    labels = transform({"img": _band_image(shoreline_mask)})
    field = labels["shoreline_proximity_field"][0]

    assert field[:, 4].min().item() == pytest.approx(1.0, abs=1e-6)
    assert field[4, 4].item() > field[4, 5].item() > field[4, 6].item()
    assert field[4, 8].item() == pytest.approx(0.0, abs=1e-6)


def test_shoreline_gaussian_field_handles_curved_masks() -> None:
    """Gaussian shoreline targets should preserve curved shoreline geometry."""
    transform = PrepareAuxiliaryMaskInputs(bands={4: "shoreline"}, use_shoreline_aux_loss=True)
    shoreline_mask = np.zeros((11, 11), dtype=np.uint8)
    shoreline_mask[2:9, 5] = 255
    shoreline_mask[8, 5:9] = 255
    labels = transform({"img": _band_image(shoreline_mask)})
    field = labels["shoreline_proximity_field"][0]

    assert field[2, 5].item() == pytest.approx(1.0, abs=1e-6)
    assert field[8, 8].item() == pytest.approx(1.0, abs=1e-6)
    assert field[7, 7].item() > field[5, 0].item()


def test_blank_shoreline_mask_produces_zero_proximity_field() -> None:
    """Empty shoreline masks should emit an all-zero Gaussian proximity field."""
    transform = PrepareAuxiliaryMaskInputs(bands={4: "shoreline"}, use_shoreline_aux_loss=True)
    labels = transform({"img": _band_image(np.zeros((6, 6), dtype=np.uint8))})

    assert torch.count_nonzero(labels["shoreline_proximity_field"]) == 0


def test_auxiliary_bands_follow_geometric_augmentation() -> None:
    """Embedded shoreline and land/water bands must follow the exact geometric transforms applied to the image."""
    shoreline_mask = np.zeros((4, 6), dtype=np.uint8)
    shoreline_mask[:, 1] = 255
    land_water_mask = np.full((4, 6), 255, dtype=np.uint8)
    land_water_mask[:, :1] = 64
    land_water_mask[:, 1:2] = 128
    land_water_mask[0, 0] = 0
    shoreline_distance = np.tile(np.arange(6, dtype=np.uint8), (4, 1))
    shoreline_proximity = shoreline_mask.copy()
    img = _band_image(shoreline_mask, land_water_mask, shoreline_distance, shoreline_proximity)
    labels = {
        "img": img,
        "instances": Instances(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xywh",
            normalized=False,
        ),
    }

    labels = RandomFlip(p=1.0, direction="horizontal")(labels)
    prepared = PrepareAuxiliaryMaskInputs(
        bands={4: "shoreline", 5: "land_water", 6: "shoreline_distance", 7: "shoreline_proximity"},
        use_shoreline_prior_loss=True,
        use_land_water_prior_loss=True,
        use_shoreline_aux_loss=True,
    )(labels)

    assert prepared["img"].shape == (4, 6, 7)
    assert torch.equal(
        prepared["land_water_mask"][0],
        torch.from_numpy(np.fliplr(land_water_mask).astype(np.int64)),
    )
    assert torch.equal(
        prepared["shoreline_distance_map"][0],
        torch.from_numpy(np.fliplr(shoreline_distance).astype(np.float32)),
    )
    assert prepared["shoreline_proximity_field"][0, :, 4].min().item() == pytest.approx(1.0, abs=1e-6)


def test_letterbox_uses_nearest_geometry_for_categorical_embedded_bands() -> None:
    """Letterbox resizing must not invent intermediate land/water classes."""
    shoreline_mask = np.zeros((8, 8), dtype=np.uint8)
    shoreline_mask[:, 3:5] = 255
    land_water_mask = np.full((8, 8), 255, dtype=np.uint8)
    land_water_mask[:, :2] = 64
    land_water_mask[:, 2:4] = 128
    land_water_mask[:, 4:6] = 192
    shoreline_distance = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
    shoreline_proximity = shoreline_mask.copy()
    img = _band_image(shoreline_mask, land_water_mask, shoreline_distance, shoreline_proximity)

    out = LetterBox(new_shape=(13, 13), scale_fill=True, bands=AUX_BANDS)(image=img)

    assert out.shape == (13, 13, 7)
    assert set(np.unique(out[..., 4]).tolist()) <= {0, 64, 128, 192, 255}
    assert set(np.unique(out[..., 3]).tolist()) <= {0, 255}


def test_random_perspective_uses_nearest_geometry_for_categorical_embedded_bands() -> None:
    """Affine/perspective transforms must keep categorical TIFF bands valid."""
    np.random.seed(0)
    random.seed(0)
    shoreline_mask = np.zeros((18, 18), dtype=np.uint8)
    shoreline_mask[4:14, 8:10] = 255
    land_water_mask = np.full((18, 18), 255, dtype=np.uint8)
    land_water_mask[:, :5] = 64
    land_water_mask[:, 5:9] = 128
    land_water_mask[:, 9:13] = 192
    shoreline_distance = np.tile(np.arange(18, dtype=np.uint8), (18, 1))
    shoreline_proximity = shoreline_mask.copy()
    img = _band_image(shoreline_mask, land_water_mask, shoreline_distance, shoreline_proximity)

    labels = RandomPerspective(
        degrees=12,
        translate=0.15,
        scale=0.2,
        shear=5,
        perspective=0.0,
        bands=AUX_BANDS,
    )(_empty_labels(img))

    assert labels["img"].shape == (18, 18, 7)
    assert set(np.unique(labels["img"][..., 4]).tolist()) <= {0, 64, 128, 192, 255}
    assert set(np.unique(labels["img"][..., 3]).tolist()) <= {0, 255}


def test_mosaic_preserves_multiband_dtype_and_valid_categorical_values() -> None:
    """Mosaic canvases should preserve TIFF dtype/channel count and zero-fill non-RGB bands."""
    shoreline_mask = np.zeros((6, 6), dtype=np.uint16)
    shoreline_mask[:, 2:4] = 255
    land_water_mask = np.full((6, 6), 255, dtype=np.uint16)
    land_water_mask[:, :2] = 64
    land_water_mask[:, 2:4] = 128
    land_water_mask[:, 4:] = 192
    shoreline_distance = np.full((6, 6), 103, dtype=np.uint16)
    shoreline_proximity = np.full((6, 6), 65535, dtype=np.uint16)
    img = _band_image(shoreline_mask, land_water_mask, shoreline_distance, shoreline_proximity, dtype=np.uint16)
    labels = _empty_labels(img.copy())
    labels["mix_labels"] = [_empty_labels(img.copy()) for _ in range(3)]

    mosaic = Mosaic(SimpleNamespace(cache=None, bands=AUX_BANDS), imgsz=8, p=1.0)
    out = mosaic._mosaic4(labels)["img"]

    assert out.dtype == np.uint16
    assert out.shape[-1] == 7
    assert set(np.unique(out[..., 4]).tolist()) <= {0, 64, 128, 192, 255}
    assert set(np.unique(out[..., 3]).tolist()) <= {0, 255}


def test_rgb_photometric_transforms_leave_embedded_bands_unchanged() -> None:
    """Sharpening and Albumentations photometric transforms must only modify RGB channels."""
    img = np.zeros((16, 16, 7), dtype=np.uint8)
    img[4:12, 4:12, :3] = 96
    img[6:10, 6:10, :3] = 180
    img[:, 7, 3] = 255
    img[:, :4, 4] = 64
    img[:, 4:8, 4] = 128
    img[:, 8:12, 4] = 192
    img[:, 12:, 4] = 255
    img[..., 5] = 103
    img[..., 6] = 255

    sharpened = RandomUnsharpMask(
        kernel_size_range=(3, 3),
        sigma_limit=1.0,
        amount_range=(1.0, 1.0),
        threshold=0,
        p=1.0,
    )({"img": img.copy()})["img"]

    assert not np.array_equal(sharpened[..., :3], img[..., :3])
    assert np.array_equal(sharpened[..., 3:], img[..., 3:])

    cfg = SimpleNamespace(
        multi_ch_albu=True,
        gaussian_blur_p=1.0,
        motion_blur_p=0.0,
        additive_noise_p=0.0,
        multi_spec_noise_p=0.0,
    )
    albu = Albumentations(cfg=cfg, p=1.0)
    if albu.transform is None:
        pytest.skip("Albumentations transform is unavailable in this environment.")
    blurred = albu({"img": img.copy(), "cls": np.zeros((0, 1), dtype=np.float32), "instances": _empty_instances()})[
        "img"
    ]

    assert not np.array_equal(blurred[..., :3], img[..., :3])
    assert np.array_equal(blurred[..., 3:], img[..., 3:])


def test_multiband_tiff_verify_and_load_round_trip() -> None:
    """7-band TIFF fixtures should verify cleanly and load with preserved shape and dtype."""
    root = TMP / "multiband_tiff_verify"
    image_path = root / "images" / "train" / "sample.tif"
    label_path = root / "labels" / "train" / "sample.txt"
    image = np.zeros((16, 16, 7), dtype=np.uint16)
    image[..., 0] = 255
    image[..., 5] = 103
    image[..., 6] = 65535
    _write_tiff(image_path, image)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("", encoding="utf-8")

    (_, _), nf, nc, msg = verify_image(((str(image_path), 0), "verify: "))
    result = verify_image_label((str(image_path), str(label_path), "verify: ", False, 1, 0, 0, False))
    loaded = imread(str(image_path))
    loader = load_inference_source(str(image_path), channels=7)
    _, images, _ = next(iter(loader))

    assert nf == 1
    assert nc == 0
    assert msg == ""
    assert result[7] == 1
    assert result[9] == 0
    assert loaded.shape == (16, 16, 7)
    assert loaded.dtype == np.uint16
    assert images[0].shape == (16, 16, 7)
    assert images[0].dtype == np.uint16

    chw_path = root / "images" / "train" / "sample_chw.tif"
    chw_image = np.moveaxis(image, -1, 0)
    tifffile.imwrite(chw_path, chw_image, metadata={"axes": "SYX"})
    chw_loaded = imread(str(chw_path), expected_channels=7)

    assert chw_loaded.shape == (16, 16, 7)
    assert chw_loaded.dtype == np.uint16
    assert np.array_equal(chw_loaded, image)


def test_multiband_segment_dataset_sample_scales_prior_targets_and_model_inputs() -> None:
    """A segment dataset sample should scale configured prior targets and model input channels."""
    root = TMP / "multiband_segment_dataset"
    image_path = root / "images" / "train" / "sample.tif"
    label_path = root / "labels" / "train" / "sample.txt"
    val_image_path = root / "images" / "val" / "sample.tif"
    val_label_path = root / "labels" / "val" / "sample.txt"
    for path in (image_path, val_image_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in (label_path, val_label_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    image = np.zeros((16, 16, 7), dtype=np.uint16)
    image[..., 0] = 255
    image[..., 1] = 128
    image[..., 2] = 64
    image[:, 7, 3] = 255
    image[:, :4, 4] = 64
    image[:, 4:8, 4] = 128
    image[:, 8:12, 4] = 192
    image[:, 12:, 4] = 255
    image[..., 5] = 103
    image[..., 6] = 65535
    _write_tiff(image_path, image)
    _write_tiff(val_image_path, image)
    segment_row = "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
    label_path.write_text(segment_row, encoding="utf-8")
    val_label_path.write_text(segment_row, encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "channels: 7",
                "bands:",
                "  4: shoreline",
                "  5: land_water",
                "  6: shoreline_distance",
                "  7: shoreline_proximity",
                "band_scale_factors:",
                "  6: 103",
                "  7: 65535",
                "names:",
                "  0: foreground",
                "",
            ]
        ),
        encoding="utf-8",
    )
    data = check_det_dataset(str(data_yaml), autodownload=False, hyp=_shoreaux_args(use_shoreline_prior_loss=True))
    dataset = YOLODataset(
        img_path=data["train"],
        imgsz=16,
        batch_size=1,
        augment=False,
        rect=False,
        hyp=_shoreaux_args(use_shoreline_prior_loss=True, use_land_water_prior_loss=True, use_shoreline_aux_loss=True),
        prefix="test: ",
        data=data,
        task="segment",
    )

    sample = dataset[0]

    assert sample["img"].shape == (7, 16, 16)
    assert sample["img"].dtype == torch.float32
    assert sample["img"][0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert sample["img"][5, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert sample["img"][6, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert sample["land_water_mask"].dtype == torch.int64
    assert sample["land_water_mask"].shape == (1, 16, 16)
    assert sample["shoreline_distance_map"][0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert sample["shoreline_proximity_field"][0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_shoreaux_model_yaml_auto_requires_only_shoreline_bands() -> None:
    """Shoreline auxiliary heads should auto-enable shoreline targets without requiring land/water bands."""
    flags = get_auxiliary_mask_flags(
        SimpleNamespace(model="ultralytics/cfg/models/12/yolo12-obb-shoreaux.yaml", use_shoreline_aux_loss=False)
    )

    assert flags["use_shoreline_aux_loss"] is True
    assert flags["require_shoreline"] is True
    assert flags["require_land_water"] is False


def test_check_det_dataset_accepts_shoreaux_models_without_land_water_bands() -> None:
    """Shoreline auxiliary heads should validate datasets that provide only shoreline bands."""
    root = TMP / "shoreaux_yaml"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "bands:",
                "  4: shoreline",
                "names:",
                "  0: foreground",
                "",
            ]
        ),
        encoding="utf-8",
    )

    data = check_det_dataset(
        str(data_yaml),
        autodownload=False,
        hyp=SimpleNamespace(model="ultralytics/cfg/models/12/yolo12-obb-shoreaux.yaml"),
    )

    assert data["bands"] == {4: "shoreline"}
    assert data["channels"] == 4


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
    """Segment priors use all predictions with thresholded land overlap and Gaussian shoreline penalties."""
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
    assert shoreline_land.item() > 0.0
    assert land_high.item() > 0.0


def _segment_prior_topk_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    proto = torch.tensor([[[[8.0, -8.0], [8.0, -8.0]]]], dtype=torch.float32)
    pred_masks = torch.tensor([[[1.0], [-1.0], [1.0], [-1.0]]], dtype=torch.float32)
    pred_scores = torch.tensor([[0.1, 0.9, 0.8, 0.7]], dtype=torch.float32)
    land_water_mask = torch.tensor([[[[64, 255], [64, 255]]]], dtype=torch.long)
    return pred_scores, pred_masks, proto, land_water_mask


def test_segment_spatial_prior_topk_disabled_matches_all_anchor_behavior() -> None:
    """segment_prior_topk=-1 or inf should preserve all-anchor segment prior behavior."""
    pred_scores, pred_masks, proto, land_water_mask = _segment_prior_topk_fixture()

    baseline = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        None,
        segment_prior_topk=pred_scores.shape[1],
    )
    disabled = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        None,
        segment_prior_topk=-1,
    )
    inf_disabled = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        None,
        segment_prior_topk=math.inf,
    )

    torch.testing.assert_close(disabled[0], baseline[0])
    torch.testing.assert_close(disabled[1], baseline[1])
    torch.testing.assert_close(inf_disabled[0], baseline[0])
    torch.testing.assert_close(inf_disabled[1], baseline[1])


def test_segment_spatial_prior_topk_selects_highest_confidence_unassigned_anchor() -> None:
    """Finite segment_prior_topk should rank unassigned anchors by detached confidence."""
    pred_scores, pred_masks, proto, land_water_mask = _segment_prior_topk_fixture()

    actual = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        None,
        segment_prior_topk=1,
    )
    expected = _compute_segmentation_spatial_prior_losses(
        pred_scores[:, 1:2],
        pred_masks[:, 1:2],
        proto,
        land_water_mask,
        None,
        segment_prior_topk=-1,
    )

    torch.testing.assert_close(actual[1], expected[1])


def test_segment_spatial_prior_topk_keeps_low_confidence_foreground_anchor() -> None:
    """Foreground anchors should be evaluated even when top-k unassigned count is zero."""
    pred_scores, pred_masks, proto, land_water_mask = _segment_prior_topk_fixture()
    fg_mask = torch.tensor([[True, False, False, False]], dtype=torch.bool)

    actual = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        None,
        fg_mask=fg_mask,
        segment_prior_topk=0,
    )
    expected = _compute_segmentation_spatial_prior_losses(
        pred_scores[:, :1],
        pred_masks[:, :1],
        proto,
        land_water_mask,
        None,
        segment_prior_topk=-1,
    )

    torch.testing.assert_close(actual[1], expected[1])


def test_segment_spatial_prior_topk_rejects_invalid_values() -> None:
    """Only non-negative integers, -1, and inf are valid segment_prior_topk values."""
    pred_scores, pred_masks, proto, land_water_mask = _segment_prior_topk_fixture()

    for bad_value in (1.5, -2, math.nan):
        with pytest.raises(ValueError):
            _compute_segmentation_spatial_prior_losses(
                pred_scores,
                pred_masks,
                proto,
                land_water_mask,
                None,
                segment_prior_topk=bad_value,
            )


def test_segment_prior_topk_cfg_accepts_numeric_and_inf_strings() -> None:
    """The CLI/config checker should coerce segment_prior_topk as a float key."""
    finite_cfg = {"segment_prior_topk": "512"}
    check_cfg(finite_cfg, hard=False)
    assert finite_cfg["segment_prior_topk"] == pytest.approx(512.0)

    inf_cfg = {"segment_prior_topk": "inf"}
    check_cfg(inf_cfg, hard=False)
    assert math.isinf(inf_cfg["segment_prior_topk"])


def test_segment_shoreline_prior_respects_assigned_offshore_gt_mask() -> None:
    """Assigned masks that match offshore GT should not be penalized just for matching that GT distance."""
    proto = torch.tensor([[[[-12.0, -12.0, 12.0, 12.0]] * 4]], dtype=torch.float32)
    pred_masks = torch.tensor([[[1.0]]], dtype=torch.float32)
    pred_scores = torch.tensor([[0.9]], dtype=torch.float32)
    land_water_mask = torch.full((1, 1, 4, 4), 255, dtype=torch.long)
    shoreline_distance = torch.tensor([[[[0.0, 0.0, 0.8, 0.8]] * 4]], dtype=torch.float32)
    gt_masks = torch.zeros((1, 4, 4), dtype=torch.long)
    gt_masks[:, :, 2:] = 1

    fallback_loss, fallback_land = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        shoreline_distance,
        shoreline_prior_max_dist=1.0,
    )
    gt_relative_loss, gt_relative_land = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        shoreline_distance,
        shoreline_prior_max_dist=1.0,
        masks=gt_masks,
        target_gt_idx=torch.zeros((1, 1), dtype=torch.long),
        fg_mask=torch.ones((1, 1), dtype=torch.bool),
        batch_idx=torch.tensor([[0]]),
        overlap=True,
    )

    assert fallback_loss.item() > 0.0
    assert gt_relative_loss.item() < fallback_loss.item() * 0.05
    assert gt_relative_land.item() == pytest.approx(fallback_land.item(), abs=1e-6)


def test_segment_shoreline_prior_gates_low_iou_assigned_masks() -> None:
    """Low-IoU assigned masks should not receive a strong shoreline-prior signal."""
    proto = torch.tensor([[[[-12.0, -12.0, 12.0, 12.0]] * 4]], dtype=torch.float32)
    pred_masks = torch.tensor([[[1.0]]], dtype=torch.float32)
    pred_scores = torch.tensor([[0.9]], dtype=torch.float32)
    land_water_mask = torch.full((1, 1, 4, 4), 255, dtype=torch.long)
    shoreline_distance = torch.tensor([[[[0.0, 0.0, 1.0, 1.0]] * 4]], dtype=torch.float32)
    gt_masks = torch.zeros((1, 4, 4), dtype=torch.long)
    gt_masks[:, :, :2] = 1

    assigned_loss, _ = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        shoreline_distance,
        shoreline_prior_max_dist=1.0,
        masks=gt_masks,
        target_gt_idx=torch.zeros((1, 1), dtype=torch.long),
        fg_mask=torch.ones((1, 1), dtype=torch.bool),
        batch_idx=torch.tensor([[0]]),
        overlap=True,
    )
    unassigned_loss, _ = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        shoreline_distance,
        shoreline_prior_max_dist=1.0,
        masks=gt_masks,
        target_gt_idx=torch.zeros((1, 1), dtype=torch.long),
        fg_mask=torch.zeros((1, 1), dtype=torch.bool),
        batch_idx=torch.tensor([[0]]),
        overlap=True,
    )

    assert assigned_loss.item() < unassigned_loss.item() * 0.05
    assert unassigned_loss.item() > 0.9


def test_segment_shoreline_prior_keeps_unassigned_offshore_penalty_with_gt_context() -> None:
    """Unassigned confident offshore masks should still receive the Gaussian false-positive prior."""
    proto = torch.tensor([[[[-12.0, -12.0, 12.0, 12.0]] * 4]], dtype=torch.float32)
    pred_masks = torch.tensor([[[1.0]]], dtype=torch.float32)
    pred_scores = torch.tensor([[0.9]], dtype=torch.float32)
    land_water_mask = torch.full((1, 1, 4, 4), 255, dtype=torch.long)
    shoreline_distance = torch.tensor([[[[0.0, 0.0, 1.0, 1.0]] * 4]], dtype=torch.float32)
    gt_masks = torch.zeros((1, 4, 4), dtype=torch.long)
    gt_masks[:, :, :2] = 1

    shoreline_loss, land_loss = _compute_segmentation_spatial_prior_losses(
        pred_scores,
        pred_masks,
        proto,
        land_water_mask,
        shoreline_distance,
        shoreline_prior_max_dist=1.0,
        masks=gt_masks,
        target_gt_idx=torch.zeros((1, 1), dtype=torch.long),
        fg_mask=torch.zeros((1, 1), dtype=torch.bool),
        batch_idx=torch.tensor([[0]]),
        overlap=True,
    )

    assert shoreline_loss.item() > 0.9
    assert land_loss.item() == pytest.approx(0.0, abs=1e-6)


def test_predict_multichannel_tiff_validates_channel_count() -> None:
    """Predict helpers should accept file-based multichannel TIFFs and reject channel mismatches."""
    root = TMP / "obb_aux_predict"
    image_path = root / "images" / "val" / "sample.tif"

    shoreline = np.zeros((8, 8), dtype=np.uint8)
    shoreline[:, 3] = 255
    land_water = np.full((8, 8), 255, dtype=np.uint8)
    land_water[:, :1] = 64
    land_water[:, 1:2] = 128
    land_water[:, 2:3] = 192
    shoreline_distance = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
    shoreline_proximity = shoreline.copy()
    image = _band_image(shoreline, land_water, shoreline_distance, shoreline_proximity)
    _write_tiff(image_path, image)

    predictor = BasePredictor(overrides={"task": "obb", "imgsz": 8, "batch": 1, "rect": False})
    predictor.model = SimpleNamespace(task="obb", pt=True, dynamic=False, imx=False, stride=32, ch=7)
    predictor.imgsz = (8, 8)
    predictor.dataset = load_inference_source(str(image_path), batch=1, channels=7)
    predictor.source_type = predictor.dataset.source_type
    _, images, _ = next(iter(predictor.dataset))

    prepared = predictor.pre_transform(images)[0]
    predictor._validate_predict_channels([prepared])

    assert prepared.shape == (8, 8, 7)
    with pytest.raises(ValueError, match="model expects 7"):
        predictor._validate_predict_channels([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_shoreaux_configs_build_while_standard_configs_remain_unchanged() -> None:
    """Canonical shoreaux configs should build while standard YOLO12 configs keep their original head classes."""
    obb_model = OBBModel("ultralytics/cfg/models/12/yolo12-obb.yaml", ch=3, nc=1, verbose=False)
    seg_model = SegmentationModel("ultralytics/cfg/models/12/yolo12-seg.yaml", ch=3, nc=1, verbose=False)
    shore_obb_model = OBBModel("ultralytics/cfg/models/12/yolo12-obb-shoreaux.yaml", ch=3, nc=1, verbose=False)
    shore_seg_model = SegmentationModel("ultralytics/cfg/models/12/yolo12-seg-shoreaux.yaml", ch=3, nc=1, verbose=False)

    assert shore_obb_model.model[-1].__class__.__name__ == "OBBShoreAux"
    assert shore_seg_model.model[-1].__class__.__name__ == "SegmentShoreAux"
    assert obb_model.model[-1].__class__.__name__ == "OBB"
    assert seg_model.model[-1].__class__.__name__ == "Segment"


def test_shoreaux_heads_return_train_time_aux_logits_and_eval_compatibility() -> None:
    """Shoreaux heads should emit train-time auxiliary maps without changing eval output contracts."""
    img = torch.randn(1, 3, 64, 64)
    obb_model = OBBModel("ultralytics/cfg/models/12/yolo12-obb-shoreaux.yaml", ch=3, nc=1, verbose=False)
    seg_model = SegmentationModel("ultralytics/cfg/models/12/yolo12-seg-shoreaux.yaml", ch=3, nc=1, verbose=False)

    obb_model.train()
    seg_model.train()
    obb_train = obb_model.predict(img)
    seg_train = seg_model.predict(img)

    assert set(obb_train.keys()) == {"main", "shore_aux_logits"}
    assert set(seg_train.keys()) == {"main", "shore_aux_logits"}
    assert obb_train["shore_aux_logits"].shape == (1, 1, 16, 16)
    assert seg_train["shore_aux_logits"].shape == (1, 1, 16, 16)

    obb_model.eval()
    seg_model.eval()
    obb_eval = obb_model.predict(img)
    seg_eval = seg_model.predict(img)

    assert isinstance(obb_eval, tuple) and len(obb_eval) == 2
    assert isinstance(seg_eval, tuple) and len(seg_eval) == 2


@pytest.mark.parametrize("empty", [False, True])
def test_shoreaux_obb_loss_is_finite_for_positive_and_empty_targets(empty: bool) -> None:
    """OBB shoreline auxiliary loss should remain finite for both positive and empty shoreline targets."""
    model = OBBModel("ultralytics/cfg/models/12/yolo12-obb-shoreaux.yaml", ch=3, nc=1, verbose=False)
    model.args = _shoreaux_args()

    loss, loss_items = model(_build_shoreaux_obb_batch(empty=empty))

    assert torch.isfinite(loss).all()
    assert torch.isfinite(loss_items).all()
    assert loss_items[-1].item() >= 0.0


@pytest.mark.parametrize("empty", [False, True])
def test_shoreaux_segment_loss_is_finite_for_positive_and_empty_targets(empty: bool) -> None:
    """Segment shoreline auxiliary loss should remain finite for both positive and empty shoreline targets."""
    model = SegmentationModel("ultralytics/cfg/models/12/yolo12-seg-shoreaux.yaml", ch=3, nc=1, verbose=False)
    model.args = _shoreaux_args()

    loss, loss_items = model(_build_shoreaux_segment_batch(empty=empty))

    assert torch.isfinite(loss).all()
    assert torch.isfinite(loss_items).all()
    assert loss_items[-1].item() >= 0.0


@pytest.mark.parametrize(
    ("cfg", "model_cls"),
    [
        ("ultralytics/cfg/models/12/yolo12-obb-shoreaux.yaml", OBBModel),
        ("ultralytics/cfg/models/12/yolo12-seg-shoreaux.yaml", SegmentationModel),
    ],
)
def test_shoreaux_loss_backpropagates_into_shared_features(cfg: str, model_cls) -> None:
    """The shoreline auxiliary loss should reach shared backbone/neck parameters, not only aux-branch weights."""
    model = model_cls(cfg, ch=3, nc=1, verbose=False)
    model.train()
    preds = model.predict(torch.randn(1, 3, 64, 64))
    field = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    field[:, :, :, 31:33] = 1.0

    loss = _compute_shoreline_aux_loss(preds["shore_aux_logits"], field)
    loss.backward()

    shared_grad = next(model.model[0].parameters()).grad
    aux_grad = next(model.model[-1].shore_aux_decoder.parameters()).grad
    assert shared_grad is not None
    assert aux_grad is not None
    assert shared_grad.abs().sum() > 0
    assert aux_grad.abs().sum() > 0


@pytest.mark.parametrize("callback", [obb_on_train_epoch_start, seg_on_train_epoch_start])
def test_shoreaux_weight_schedule_ramps_linearly(callback) -> None:
    """The shoreline auxiliary loss weight should ramp from zero to the configured target."""
    model = torch.nn.Linear(1, 1)
    model.args = SimpleNamespace()
    trainer = SimpleNamespace(
        args=SimpleNamespace(shoreline_aux_weight=0.2, shoreline_aux_warmup_epochs=10),
        model=model,
        epoch=0,
    )

    callback(trainer)
    assert trainer.args.active_shoreline_aux_weight == pytest.approx(0.0)
    assert model.args.active_shoreline_aux_weight == pytest.approx(0.0)

    trainer.epoch = 5
    callback(trainer)
    assert trainer.args.active_shoreline_aux_weight == pytest.approx(0.1)

    trainer.epoch = 10
    callback(trainer)
    assert trainer.args.active_shoreline_aux_weight == pytest.approx(0.2)


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


def test_dual_branch_yolo_constructor_can_infer_channels_from_data_yaml() -> None:
    """Generic YOLO YAML construction should use supplied local data channels when available."""
    seg_cfg = "ultralytics/cfg/models/yolo/segment_no_p2/yolo12-pafpn-dual-segment.yaml"
    model = YOLO(seg_cfg, task="segment", data={"channels": 5}, verbose=False)

    assert model.model.yaml["channels"] == 5


def test_dual_branch_models_require_auxiliary_channels() -> None:
    """Dual-branch models should fail fast when there are no auxiliary channels to route."""
    obb_cfg = "ultralytics/cfg/models/yolo/obb/yolo12-pafpn-dual-obb.yaml"

    with pytest.raises(ValueError, match="ChannelSplit requires input channels > rgb_channels"):
        OBBModel(obb_cfg, ch=3, nc=1, verbose=False)
