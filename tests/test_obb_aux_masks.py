from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tests import TMP
from ultralytics.data.augment import PrepareAuxiliaryMaskInputs, RandomFlip
from ultralytics.data.build import load_inference_source
from ultralytics.data.utils import (
    check_det_dataset,
    get_auxiliary_mask_flags,
)
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.obb.train import on_train_epoch_start as obb_on_train_epoch_start
from ultralytics.models.yolo.segment.train import on_train_epoch_start as seg_on_train_epoch_start
from ultralytics.nn.tasks import OBBModel, SegmentationModel
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import (
    _compute_obb_spatial_prior_losses,
    _compute_segmentation_spatial_prior_losses,
    _compute_shoreline_aux_loss,
)

def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        stack = array[None]
    else:
        stack = array.transpose(2, 0, 1)
    assert cv2.imwritemulti(str(path), stack)


def _band_image(
    shoreline: np.ndarray | None = None,
    land_water: np.ndarray | None = None,
    shoreline_distance: np.ndarray | None = None,
    shoreline_proximity: np.ndarray | None = None,
) -> np.ndarray:
    shape = next(x.shape for x in (shoreline, land_water, shoreline_distance, shoreline_proximity) if x is not None)
    img = np.zeros((*shape, 3), dtype=np.uint8)
    channels = [img[..., 0], img[..., 1], img[..., 2]]
    for band in (shoreline, land_water, shoreline_distance, shoreline_proximity):
        if band is not None:
            channels.append(band.astype(np.uint8, copy=False))
    return np.stack(channels, axis=2)


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


def test_dual_branch_models_require_auxiliary_channels() -> None:
    """Dual-branch models should fail fast when there are no auxiliary channels to route."""
    obb_cfg = "ultralytics/cfg/models/yolo/obb/yolo12-pafpn-dual-obb.yaml"

    with pytest.raises(ValueError, match="ChannelSplit requires input channels > rgb_channels"):
        OBBModel(obb_cfg, ch=3, nc=1, verbose=False)
