from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "ultralytics/cfg/models/rcnn/pointrend_rcnn_r50_fpn_smallobj.yaml"


def _head_config(**point_overrides):
    point = {
        "coarse_pool_resolution": 14,
        "coarse_conv_dim": 8,
        "coarse_fc_dim": 16,
        "coarse_num_fcs": 2,
        "coarse_output_resolution": 7,
        "point_hidden_dim": 8,
        "point_num_fcs": 3,
        "coarse_pred_each_layer": True,
        "train_num_points": 16,
        "oversample_ratio": 3.0,
        "importance_sample_ratio": 0.75,
        "subdivision_steps": 5,
        "subdivision_num_points": 784,
        "scale_factor": 2,
    }
    point.update(point_overrides)
    return {
        "rpn": {
            "anchor_scales": [1, 2, 4],
            "anchor_ratios": [0.5, 1.0, 2.0],
            "strides": [4, 8, 16, 32, 64],
            "pre_nms_topk_train": 32,
            "post_nms_topk_train": 16,
            "pre_nms_topk_test": 16,
            "post_nms_topk_test": 8,
            "samples_per_img": 32,
            "pos_fraction": 0.5,
            "beta": 0.0,
        },
        "roi": {"pool_size": 7, "sampling_ratio": 0, "featmap_strides": [4, 8, 16, 32]},
        "train": {"pos_iou": 0.5, "neg_iou": 0.5, "samples_per_img": 16, "pos_fraction": 0.25},
        "test": {"score_thresh": 0.05, "nms_iou": 0.5, "max_dets": 8, "mask_threshold": 0.5},
        "bbox_head": {
            "hidden_dim": 16,
            "loss": "smooth_l1",
            "beta": 0.0,
            "train_on_pred_boxes": True,
            "class_agnostic": False,
        },
        "pointrend": point,
    }


def _tiny_head(nc=2, **point_overrides):
    from ultralytics.nn.modules.pointrend_rcnn import PointRendRCNNHead

    return PointRendRCNNHead([8, 8, 8, 8, 8], nc, _head_config(**point_overrides))


def _features(requires_grad=False):
    return [
        torch.randn(1, 8, 16, 16, requires_grad=requires_grad),
        torch.randn(1, 8, 8, 8, requires_grad=requires_grad),
        torch.randn(1, 8, 4, 4, requires_grad=requires_grad),
        torch.randn(1, 8, 2, 2, requires_grad=requires_grad),
        torch.randn(1, 8, 1, 1, requires_grad=requires_grad),
    ]


def _batch(class_index=1):
    mask = torch.zeros((1, 64, 64), dtype=torch.float32)
    mask[:, 14:51, 17:47] = 1.0
    return {
        "img": torch.randn(1, 3, 64, 64),
        "batch_idx": torch.zeros((1, 1), dtype=torch.float32),
        "cls": torch.tensor([[class_index]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5078125, 0.46875, 0.578125]], dtype=torch.float32),
        "masks": mask,
    }


def test_yaml_matches_detectron2_style_architecture_and_small_anchors():
    from ultralytics.nn.tasks import guess_model_task, yaml_model_load

    cfg = yaml_model_load(MODEL_YAML)
    backbone_args = cfg["backbone"][0][3]
    fpn_cfg = cfg["head"][0][3][1]
    head_cfg = cfg["head"][-1][3][1]
    point_cfg = head_cfg["pointrend"]

    assert guess_model_task(cfg) == "segment"
    assert cfg["head"][-1][-2] == "PointRendRCNNHead"
    assert backbone_args == ["resnet50", "DEFAULT", 1, True, True]
    assert fpn_cfg["implementation"] == "reference"
    assert fpn_cfg["num_outs"] == 5
    assert fpn_cfg["weight_init"] == "detectron2"
    assert head_cfg["rpn"]["anchor_scales"] == [1, 2, 4]
    assert head_cfg["roi"] == {"pool_size": 7, "sampling_ratio": 0, "featmap_strides": [4, 8, 16, 32]}
    assert point_cfg["coarse_pool_resolution"] == 14
    assert point_cfg["coarse_output_resolution"] == 7
    assert point_cfg["coarse_fc_dim"] == 1024
    assert point_cfg["coarse_num_fcs"] == 2
    assert point_cfg["point_hidden_dim"] == 256
    assert point_cfg["point_num_fcs"] == 3
    assert point_cfg["train_num_points"] == 196
    assert point_cfg["subdivision_steps"] == 5
    assert point_cfg["subdivision_num_points"] == 784


def test_dedicated_head_topology_initialization_and_class_specific_outputs():
    torch.manual_seed(0)
    head = _tiny_head(nc=3)
    branch = head.point_rend
    coarse = branch.coarse_head
    point = branch.point_head

    assert isinstance(coarse.reduce_channel_dim, nn.Identity)
    assert coarse.reduce_spatial_dim.kernel_size == (2, 2)
    assert coarse.reduce_spatial_dim.stride == (2, 2)
    assert len(coarse.fcs) == 2
    assert coarse.predictor.out_features == 3 * 7 * 7
    assert len(point.fcs) == 3
    assert point.fcs[0].in_channels == 8 + 3
    assert point.fcs[1].in_channels == 8 + 3
    assert point.predictor.out_channels == 3
    assert branch.source_channels == (8,)
    assert not any("project" in name for name, _ in branch.named_modules())

    roi_features = torch.randn(2, 8, 14, 14)
    coarse_logits = coarse(roi_features)
    class_logits, box_deltas = head.bbox_head(torch.randn(2, 8, 7, 7))
    assert coarse_logits.shape == (2, 3, 7, 7)
    assert class_logits.shape == (2, 4)
    assert box_deltas.shape == (2, 3, 4)

    conv_expected = (2.0 / (8 * 2 * 2)) ** 0.5
    fc_expected = (1.0 / coarse.fcs[0].in_features) ** 0.5
    assert coarse.reduce_spatial_dim.weight.std().item() == pytest.approx(conv_expected, rel=0.2)
    assert coarse.fcs[0].weight.std().item() == pytest.approx(fc_expected, rel=0.2)
    assert coarse.predictor.weight.std().item() == pytest.approx(0.001, rel=0.15)
    assert point.predictor.weight.std().item() == pytest.approx(0.001, rel=0.2)
    assert head.bbox_head.bbox_pred.weight.std().item() == pytest.approx(0.001, rel=0.15)


def test_coarse_grid_uses_direct_p2_and_reference_regular_coordinates(monkeypatch):
    head = _tiny_head(nc=2)
    branch = head.point_rend
    recorded = {}

    def fake_sample(p2, boxes, batch_indices, point_coords, image_shape):
        recorded["coords"] = point_coords.detach().clone()
        return p2.new_zeros((boxes.shape[0], p2.shape[1], point_coords.shape[1]))

    monkeypatch.setattr(branch, "_sample_p2", fake_sample)
    logits = branch.coarse_logits(
        torch.randn(1, 8, 16, 16),
        torch.tensor([[3.25, 4.5, 43.75, 52.25]]),
        torch.zeros(1, dtype=torch.long),
        (64, 64),
    )
    coords = recorded["coords"].reshape(1, 14, 14, 2)

    assert logits.shape == (1, 2, 7, 7)
    assert torch.allclose(coords[0, 0, 0], torch.tensor([0.5 / 14, 0.5 / 14]))
    assert torch.allclose(coords[0, -1, -1], torch.tensor([13.5 / 14, 13.5 / 14]))


def test_effective_schedule_dense_initialization_and_final_shape(monkeypatch):
    head = _tiny_head(nc=2)
    branch = head.point_rend
    calls = []
    original = branch._point_logits

    def record(*args, **kwargs):
        point_coords = args[3]
        calls.append(point_coords.shape[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(branch, "_point_logits", record)
    assert branch.effective_subdivision_schedule() == (28, 3)
    assert branch.final_resolution() == 224

    refined = branch.refine(
        torch.randn(1, 8, 16, 16),
        torch.tensor([[2.5, 3.25, 55.75, 58.5]]),
        torch.zeros(1, dtype=torch.long),
        torch.tensor([1]),
        (64, 64),
    )
    assert calls == [28 * 28, 784, 784, 784]
    assert refined.shape == (1, 2, 224, 224)
    assert torch.isfinite(refined).all()


def test_fractional_aligned_coarse_targets_are_binary_and_handle_degenerate_boxes():
    from ultralytics.nn.modules.pointrend_rcnn import _PointRendMaskBranch

    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(40), indexing="ij")
    mask = ((xx + 2 * yy) > 36).float()[None, None]
    fractional = torch.tensor([[3.2, 4.4, 28.7, 25.6]])
    shifted = fractional + torch.tensor([[0.75, 0.0, 0.75, 0.0]])
    rounded = fractional.round()
    target = _PointRendMaskBranch.aligned_coarse_targets(mask, fractional, 7)
    shifted_target = _PointRendMaskBranch.aligned_coarse_targets(mask, shifted, 7)
    rounded_target = _PointRendMaskBranch.aligned_coarse_targets(mask, rounded, 7)
    degenerate = _PointRendMaskBranch.aligned_coarse_targets(mask, torch.tensor([[4.0, 5.0, 4.0, 8.0]]), 7)

    assert target.dtype == torch.bool
    assert target.shape == shifted_target.shape == rounded_target.shape == degenerate.shape == (1, 7, 7)
    assert torch.logical_xor(target, shifted_target).any()
    assert torch.isfinite(degenerate.float()).all()


def test_aligned_coarse_targets_pass_fractional_boxes_without_rounding(monkeypatch):
    import ultralytics.nn.modules.pointrend_rcnn as pointrend_rcnn

    recorded = {}

    def fake_roi_align(input, rois, output_size, spatial_scale, sampling_ratio, aligned):
        recorded["rois"] = rois.clone()
        recorded["settings"] = (output_size, spatial_scale, sampling_ratio, aligned)
        return input.new_full((rois.shape[0], 1, output_size, output_size), 0.6)

    monkeypatch.setattr(pointrend_rcnn, "torchvision_native_roi_align", fake_roi_align)
    boxes = torch.tensor([[3.2, 4.4, 28.7, 25.6]])
    targets = pointrend_rcnn._PointRendMaskBranch.aligned_coarse_targets(
        torch.zeros((1, 1, 32, 40)),
        boxes,
        7,
    )

    assert torch.equal(recorded["rois"][:, 1:], boxes)
    assert recorded["settings"] == (7, 1.0, 0, True)
    assert targets.dtype == torch.bool and targets.all()


def test_mask_training_uses_gt_class_predicted_boxes():
    head = _tiny_head(nc=2)
    features = _features()
    proposal = torch.tensor([[17.0, 14.0, 47.0, 51.0]])
    gt_boxes = [proposal.clone()]
    gt_labels = [torch.tensor([1])]
    with torch.no_grad():
        for parameter in head.bbox_head.parameters():
            parameter.zero_()
        head.bbox_head.bbox_pred.bias[4] = 0.25

    _, _, mask_rois, gt_indices, classes = head._box_losses_and_mask_rois(
        features,
        [proposal],
        gt_boxes,
        gt_labels,
        (64, 64),
    )

    assert mask_rois.shape == (2, 5)
    assert torch.all(classes == 1)
    assert torch.all(gt_indices == 0)
    assert torch.all(mask_rois[:, 1] > proposal[0, 0])
    assert not torch.equal(mask_rois[:, 1:5], proposal.expand_as(mask_rois[:, 1:5]))


def test_joint_loss_has_finite_nonzero_gradients_for_every_expected_group():
    torch.manual_seed(4)
    head = _tiny_head(nc=2)
    features = _features(requires_grad=True)
    total, items = head.loss(features, _batch(class_index=1))
    total.backward()

    groups = {
        "rpn": head.rpn_head.parameters(),
        "box": head.bbox_head.parameters(),
        "coarse": head.point_rend.coarse_head.parameters(),
        "point": head.point_rend.point_head.parameters(),
    }
    assert items.shape == (6,)
    assert torch.isfinite(total) and torch.isfinite(items).all()
    for parameters in groups.values():
        grads = [parameter.grad for parameter in parameters]
        assert grads and all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert all(grad.abs().sum() > 0 for grad in grads)
    assert features[0].grad is not None
    assert torch.isfinite(features[0].grad).all() and features[0].grad.abs().sum() > 0


def test_frozen_loss_trains_only_complete_coarse_and_point_branches():
    from ultralytics.nn.modules.pointrend import PointRendTrainConfig

    torch.manual_seed(5)
    head = _tiny_head(nc=2)
    head.point_rend.train_config = PointRendTrainConfig(
        mode="frozen",
        train_num_points=16,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )
    for name, parameter in head.named_parameters():
        parameter.requires_grad = name.startswith("point_rend.")

    total, items = head.loss(_features(), _batch(class_index=1))
    total.backward()

    assert torch.isfinite(total) and torch.isfinite(items).all()
    for name, parameter in head.named_parameters():
        if name.startswith("point_rend."):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert parameter.grad is None, name


def test_dedicated_checkpoint_compatibility_rejects_generic_adapter():
    from ultralytics.nn.modules.pointrend import (
        PointRendConfig,
        PointRendRefiner,
        prepare_pointrend_weight_transfer,
    )

    target = SimpleNamespace(model=[_tiny_head(nc=2)], yaml={})
    same = SimpleNamespace(model=[_tiny_head(nc=2)], yaml={})
    different = SimpleNamespace(model=[_tiny_head(nc=1)], yaml={})
    generic_cfg = PointRendConfig.from_yaml(
        {
            "pointrend": {
                "enabled": True,
                "feature_levels": [0],
                "project_channels": 8,
                "hidden_channels": 8,
                "num_fcs": 2,
                "coarse_resolution": 8,
            }
        }
    )
    generic_head = SimpleNamespace(point_rend=PointRendRefiner([8], generic_cfg))
    generic = SimpleNamespace(model=[generic_head], yaml={})

    prepare_pointrend_weight_transfer(target, same)
    with pytest.raises(ValueError, match="architecture is incompatible"):
        prepare_pointrend_weight_transfer(target, different)
    with pytest.raises(ValueError, match="generic adapter-based"):
        prepare_pointrend_weight_transfer(target, generic)


def test_dedicated_predictor_pastes_at_original_resolution_and_keeps_empty_masks():
    from ultralytics.models.rcnn.predict import RCNNSegmentationPredictor

    predictor = object.__new__(RCNNSegmentationPredictor)
    predictor.model = SimpleNamespace(names={0: "rip"})
    original = np.zeros((37, 53, 3), dtype=np.uint8)
    network_input = torch.zeros((1, 3, 64, 64))
    prediction = {
        "bboxes": torch.tensor([[8.25, 10.5, 39.75, 49.25]]),
        "conf": torch.tensor([0.8]),
        "cls": torch.tensor([0.0]),
        "masks": torch.ones((1, 64, 64), dtype=torch.bool),
        "mask_roi_logits": torch.full((1, 1, 224, 224), -100.0),
    }

    result = predictor.construct_result(prediction, network_input, original, "image.png")

    assert result.boxes.data.shape == (1, 6)
    assert result.masks.data.shape == (1, 37, 53)
    assert not result.masks.data.any()
    assert result.cpu().boxes.data.shape == (1, 6)
    assert result.numpy().masks.data.shape == (1, 37, 53)


def test_empty_dedicated_prediction_preserves_standard_shapes():
    from ultralytics.models.rcnn.predict import RCNNSegmentationPredictor

    predictor = object.__new__(RCNNSegmentationPredictor)
    predictor.model = SimpleNamespace(names={0: "rip"})
    original = np.zeros((31, 47, 3), dtype=np.uint8)
    prediction = {
        "bboxes": torch.zeros((0, 4)),
        "conf": torch.zeros((0,)),
        "cls": torch.zeros((0,)),
        "masks": torch.zeros((0, 64, 64), dtype=torch.bool),
        "mask_roi_logits": torch.zeros((0, 1, 224, 224)),
    }
    result = predictor.construct_result(prediction, torch.zeros((1, 3, 64, 64)), original, "empty.png")

    assert result.boxes.data.shape == (0, 6)
    assert result.masks.data.shape == (0, 31, 47)


def test_memory_bounded_binary_paste_matches_probability_reference():
    from ultralytics.nn.modules.pointrend import paste_roi_probabilities
    from ultralytics.nn.modules.pointrend_rcnn import _paste_binary_masks

    torch.manual_seed(9)
    logits = torch.randn(7, 1, 13, 17)
    boxes = torch.tensor(
        [
            [-2.5, 1.25, 18.5, 21.75],
            [2.2, -3.0, 20.7, 18.1],
            [1.0, 2.0, 8.0, 11.0],
            [7.5, 5.25, 25.5, 24.75],
            [4.0, 4.0, 4.0, 10.0],
            [0.0, 0.0, 27.0, 23.0],
            [10.2, 3.7, 16.8, 15.3],
        ]
    )
    reference = paste_roi_probabilities(logits, boxes, (23, 27), max_chunk_size=2) >= 0.5
    actual = _paste_binary_masks(logits, boxes, (23, 27), 0.5, chunk_size=3)

    assert torch.equal(actual, reference)


def test_model_build_overrides_dataset_classes_without_generic_adapter(monkeypatch):
    from ultralytics.nn.modules.pointrend import get_pointrend_adapter, has_pointrend
    from ultralytics.nn.tasks import RCNNSegmentationModel, yaml_model_load

    cfg = deepcopy(yaml_model_load(MODEL_YAML))
    cfg["backbone"][0][3][1] = None
    model = RCNNSegmentationModel(cfg, nc=1, verbose=False)
    head = model.model[-1]

    assert head.nc == 1
    assert head.point_rend.num_classes == 1
    assert has_pointrend(model)
    with pytest.raises(TypeError, match="No PointRend adapter"):
        get_pointrend_adapter(head)
