from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _block(**overrides):
    values = {
        "enabled": True,
        "feature_levels": [0],
        "project_channels": 8,
        "hidden_channels": 12,
        "num_fcs": 2,
        "coarse_resolution": 8,
        "subdivision_steps": 2,
        "subdivision_num_points": 16,
        "scale_factor": 2,
    }
    values.update(overrides)
    return values


def _tiny_cfg(pointrend=...):
    cfg = {
        "nc": 1,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
        ],
        "head": [[[0, 1, 2], 1, "Segment", [1, 8, 16]]],
    }
    if pointrend is not ...:
        cfg["pointrend"] = pointrend
    return cfg


def _model(cfg):
    from ultralytics.nn.tasks import SegmentationModel

    return SegmentationModel(deepcopy(cfg), ch=3, nc=1, verbose=False)


@pytest.mark.parametrize("value", [..., {"enabled": False}])
def test_missing_or_disabled_yaml_does_not_attach_pointrend(value):
    from ultralytics.nn.modules.pointrend import has_pointrend

    model = _model(_tiny_cfg(value))

    assert not has_pointrend(model)
    assert not any(".point_rend." in name for name, _ in model.named_parameters())


def test_yaml_dimensions_and_normalized_defaults_are_model_owned():
    model = _model(
        _tiny_cfg(
            {
                "enabled": True,
                "feature_levels": [0, 1],
                "project_channels": 7,
                "hidden_channels": 11,
                "num_fcs": 2,
                "coarse_resolution": 10,
            }
        )
    )
    refiner = model.model[-1].point_rend

    assert [tuple(layer.weight.shape) for layer in refiner.projections] == [(7, 16, 1, 1), (7, 32, 1, 1)]
    assert tuple(refiner.point_head.fcs[0].weight.shape) == (11, 15, 1)
    assert tuple(refiner.point_head.fcs[1].weight.shape) == (11, 12, 1)
    assert model.yaml["pointrend"] == {
        "enabled": True,
        "feature_levels": [0, 1],
        "project_channels": 7,
        "hidden_channels": 11,
        "num_fcs": 2,
        "coarse_resolution": 10,
        "subdivision_steps": 3,
        "subdivision_num_points": 784,
        "scale_factor": 2,
    }


@pytest.mark.parametrize(
    ("block", "match"),
    [
        (_block(feature_levels=[3]), "feature_levels"),
        (_block(project_channels=0), "project_channels"),
        (_block(hidden_channels=-1), "hidden_channels"),
        (_block(num_fcs=0), "num_fcs"),
        (_block(coarse_resolution=0), "coarse_resolution"),
        (_block(subdivision_steps=-1), "subdivision_steps"),
        (_block(scale_factor=0), "scale_factor"),
    ],
)
def test_invalid_yaml_fails_during_model_construction(block, match):
    with pytest.raises(ValueError, match=match):
        _model(_tiny_cfg(block))


def test_enabled_pointrend_rejects_unsupported_head():
    from ultralytics.nn.tasks import DetectionModel

    cfg = _tiny_cfg(_block())
    cfg["head"] = [[[0, 1, 2], 1, "Detect", [1]]]
    with pytest.raises(TypeError, match="unsupported"):
        DetectionModel(cfg, ch=3, nc=1, verbose=False)


@pytest.mark.parametrize(
    ("relative_path", "model_class", "head_name"),
    [
        ("ultralytics/cfg/models/11/yolo11-seg-pointrend.yaml", "SegmentationModel", "Segment"),
        ("ultralytics/cfg/models/rt-detr/rtdetr-l-seg-pointrend.yaml", "RTDETRSegmentModel", "RTDETRSegmentDecoder"),
        (
            "ultralytics/cfg/models/transformer/mask2former-yolo12-seg-pointrend.yaml",
            "SegmentationModel",
            "Mask2FormerHead",
        ),
        (
            "ultralytics/cfg/models/rcnn/mask_rcnn_r50_fpn_pointrend.yaml",
            "RCNNSegmentationModel",
            "MaskRCNNHead",
        ),
        (
            "ultralytics/cfg/models/rcnn/cascade_mask_rcnn_r50_fpn_pointrend.yaml",
            "RCNNSegmentationModel",
            "CascadeMaskRCNNHead",
        ),
    ],
)
def test_canonical_yaml_constructs_native_adapter_family(relative_path, model_class, head_name):
    from ultralytics.nn import tasks
    from ultralytics.nn.modules.pointrend import get_pointrend_adapter, has_pointrend

    model = getattr(tasks, model_class)(ROOT / relative_path, ch=3, nc=1, verbose=False)
    head = model.model[-1]

    assert type(head).__name__ == head_name
    assert has_pointrend(model)
    assert get_pointrend_adapter(head).refiner is head.point_rend
    assert model.yaml["pointrend"]["enabled"] is True


@pytest.mark.parametrize(
    "key",
    [
        "pointrend",
        "pointrend_feature_levels",
        "pointrend_project_channels",
        "pointrend_hidden_channels",
        "pointrend_num_fcs",
        "pointrend_coarse_resolution",
        "pointrend_subdivision_steps",
        "pointrend_subdivision_num_points",
        "pointrend_scale_factor",
    ],
)
def test_old_architecture_cli_arguments_are_rejected(key):
    from ultralytics.cfg import get_cfg

    with pytest.raises(SyntaxError, match=key):
        get_cfg(overrides={key: 1})


def test_training_policy_cli_arguments_remain_valid():
    from ultralytics.cfg import get_cfg

    cfg = get_cfg(
        overrides={
            "pointrend_mode": "frozen",
            "pointrend_train_num_points": 32,
            "pointrend_oversample_ratio": 2.0,
            "pointrend_importance_sample_ratio": 0.5,
            "pointrend_train_max_instances": 7,
            "pointrend_loss_weight": 0.25,
        }
    )
    assert cfg.pointrend_mode == "frozen"
    assert cfg.pointrend_train_num_points == 32
    assert cfg.pointrend_loss_weight == 0.25


def test_base_checkpoint_into_pointrend_yaml_preserves_new_refiner():
    torch.manual_seed(1)
    incoming = _model(_tiny_cfg())
    torch.manual_seed(2)
    target = _model(_tiny_cfg(_block()))
    point_before = {
        name: tensor.clone() for name, tensor in target.state_dict().items() if ".point_rend." in name
    }

    target.load(incoming, verbose=False)
    incoming_state = incoming.state_dict()
    for name, tensor in target.state_dict().items():
        if ".point_rend." in name:
            assert torch.equal(tensor, point_before[name])
        elif name in incoming_state and tensor.shape == incoming_state[name].shape:
            assert torch.equal(tensor, incoming_state[name])


def test_matching_pointrend_checkpoint_restores_refiner_tensors():
    torch.manual_seed(3)
    incoming = _model(_tiny_cfg(_block()))
    torch.manual_seed(4)
    target = _model(_tiny_cfg(_block()))

    target.load(incoming, verbose=False)

    for name, tensor in target.state_dict().items():
        if ".point_rend." in name:
            assert torch.equal(tensor, incoming.state_dict()[name])


def test_pointrend_architecture_mismatch_raises_before_loading():
    incoming = _model(_tiny_cfg(_block(project_channels=9)))
    target = _model(_tiny_cfg(_block(project_channels=8)))
    base_before = target.model[0].conv.weight.detach().clone()

    with pytest.raises(ValueError, match="incompatible"):
        target.load(incoming, verbose=False)

    assert torch.equal(target.model[0].conv.weight, base_before)


def test_legacy_runtime_config_checkpoint_is_migrated_with_weights():
    incoming = _model(_tiny_cfg(_block()))
    incoming_refiner = incoming.model[-1].point_rend
    incoming_refiner.config = incoming_refiner.model_config
    del incoming_refiner.model_config
    del incoming_refiner.train_config
    incoming.yaml.pop("pointrend")
    incoming_state = {
        name: tensor.clone() for name, tensor in incoming.state_dict().items() if ".point_rend." in name
    }
    target = _model(_tiny_cfg())

    target.load(incoming, verbose=False)

    assert target.yaml["pointrend"]["enabled"] is True
    assert hasattr(target.model[-1].point_rend, "model_config")
    for name, tensor in target.state_dict().items():
        if ".point_rend." in name:
            assert torch.equal(tensor, incoming_state[name])


def test_checkpoint_reload_preserves_normalized_yaml(tmp_path):
    from ultralytics.nn.tasks import load_checkpoint

    model = _model(_tiny_cfg({"enabled": True, "project_channels": 8}))
    checkpoint = tmp_path / "pointrend.pt"
    torch.save({"model": model, "train_args": {}}, checkpoint)

    loaded, _ = load_checkpoint(checkpoint)

    assert loaded.yaml["pointrend"] == model.yaml["pointrend"]
    assert loaded.model[-1].point_rend.model_config.to_dict() == model.yaml["pointrend"]
