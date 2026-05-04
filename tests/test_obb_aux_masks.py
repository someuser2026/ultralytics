from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from tests import TMP
from ultralytics.data.augment import PrepareAuxiliaryMaskInputs, RandomFlip
from ultralytics.data.utils import (
    build_auxiliary_root_mappings,
    check_det_dataset,
    get_auxiliary_mask_flags,
    resolve_auxiliary_mask_paths,
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


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), array)


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


def test_blank_shoreline_mask_produces_zero_distance_map() -> None:
    """Blank shoreline masks should suppress shoreline prior rather than max it out."""
    transform = PrepareAuxiliaryMaskInputs(
        use_shoreline_prior_loss=True,
        use_land_water_prior_loss=True,
        shoreline_prior_max_dist=8,
    )
    labels = transform(
        {
            "img": np.zeros((4, 4, 3), dtype=np.uint8),
            "shoreline_mask": np.zeros((4, 4), dtype=np.uint8),
            "land_water_mask": np.full((4, 4), 192, dtype=np.uint8),
        }
    )

    assert torch.count_nonzero(labels["shoreline_distance_map"]) == 0


def test_shoreline_gaussian_field_straight_line_is_peak_on_shore_and_truncated() -> None:
    """Gaussian shoreline targets should peak on-shore, decay smoothly, and zero beyond the truncation radius."""
    transform = PrepareAuxiliaryMaskInputs(
        use_shoreline_aux_loss=True,
        shoreline_aux_gaussian_sigma_ratio=0.20,
        shoreline_aux_gaussian_truncate_sigmas=2.0,
    )
    shoreline_mask = np.zeros((9, 9), dtype=np.uint8)
    shoreline_mask[:, 4] = 1

    labels = transform({"img": np.zeros((9, 9, 3), dtype=np.uint8), "shoreline_mask": shoreline_mask})
    field = labels["shoreline_proximity_field"][0]

    assert field[:, 4].min().item() == pytest.approx(1.0, abs=1e-6)
    assert field[4, 4].item() > field[4, 5].item() > field[4, 6].item()
    assert field[4, 8].item() == pytest.approx(0.0, abs=1e-6)


def test_shoreline_gaussian_field_handles_curved_masks() -> None:
    """Gaussian shoreline targets should preserve curved shoreline geometry."""
    transform = PrepareAuxiliaryMaskInputs(use_shoreline_aux_loss=True)
    shoreline_mask = np.zeros((11, 11), dtype=np.uint8)
    shoreline_mask[2:9, 5] = 1
    shoreline_mask[8, 5:9] = 1

    labels = transform({"img": np.zeros((11, 11, 3), dtype=np.uint8), "shoreline_mask": shoreline_mask})
    field = labels["shoreline_proximity_field"][0]

    assert field[2, 5].item() == pytest.approx(1.0, abs=1e-6)
    assert field[8, 8].item() == pytest.approx(1.0, abs=1e-6)
    assert field[7, 7].item() > field[5, 0].item()


def test_blank_shoreline_mask_produces_zero_proximity_field() -> None:
    """Empty shoreline masks should emit an all-zero Gaussian proximity field."""
    transform = PrepareAuxiliaryMaskInputs(use_shoreline_aux_loss=True)
    labels = transform({"img": np.zeros((6, 6, 3), dtype=np.uint8), "shoreline_mask": np.zeros((6, 6), dtype=np.uint8)})

    assert torch.count_nonzero(labels["shoreline_proximity_field"]) == 0


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


def test_shoreaux_model_yaml_auto_requires_only_shoreline_masks() -> None:
    """Shoreline auxiliary heads should auto-enable shoreline targets without requiring land/water masks."""
    flags = get_auxiliary_mask_flags(
        SimpleNamespace(model="ultralytics/cfg/models/12/yolo12-obb-shoreaux.yaml", use_shoreline_aux_loss=False)
    )

    assert flags["use_shoreline_aux_loss"] is True
    assert flags["require_shoreline"] is True
    assert flags["require_land_water"] is False


def test_check_det_dataset_accepts_shoreaux_models_without_land_water_masks() -> None:
    """Shoreline auxiliary heads should validate datasets that provide only shoreline masks."""
    root = TMP / "shoreaux_yaml"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / "shoreline" / split).mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "shoreline_masks: masks/shoreline",
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

    assert Path(data["shoreline_masks"]["train"]).name == "train"
    assert "land_water_masks" not in data


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
