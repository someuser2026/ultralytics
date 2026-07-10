from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _point_args(**overrides):
    values = {
        "pointrend": True,
        "pointrend_mode": "joint",
        "pointrend_feature_levels": [0],
        "pointrend_project_channels": 8,
        "pointrend_hidden_channels": 8,
        "pointrend_num_fcs": 2,
        "pointrend_coarse_resolution": 8,
        "pointrend_train_num_points": 16,
        "pointrend_oversample_ratio": 3.0,
        "pointrend_importance_sample_ratio": 0.75,
        "pointrend_train_max_instances": 8,
        "pointrend_subdivision_steps": 2,
        "pointrend_subdivision_num_points": 16,
        "pointrend_scale_factor": 2,
        "pointrend_loss_weight": 1.0,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "overlap_mask": False,
        "mask_weight": 1.0,
        "seg_use_mixed_loss": False,
        "use_soft_ignore_band": False,
        "imgsz": 64,
        "use_shoreline_prior_loss": False,
        "use_land_water_prior_loss": False,
        "shoreline_aux_weight": 0.0,
        "active_shoreline_aux_weight": 0.0,
        "shoreline_aux_bce_weight": 1.0,
        "shoreline_aux_dice_weight": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _tiny_segment_cfg():
    return {
        "nc": 1,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
        ],
        "head": [[[0, 1, 2], 1, "Segment", [1, 8, 16]]],
    }


def _segment_batch(imgsz=64):
    masks = torch.zeros((1, imgsz, imgsz), dtype=torch.float32)
    masks[0, 18:46, 20:44] = 1.0
    return {
        "img": torch.randn(1, 3, imgsz, imgsz),
        "batch_idx": torch.zeros((1, 1), dtype=torch.float32),
        "cls": torch.zeros((1, 1), dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.375, 0.4375]], dtype=torch.float32),
        "cls_probs": torch.ones((1, 1), dtype=torch.float32),
        "masks": masks,
    }


def test_uncertain_point_selection_prefers_logits_near_zero():
    from ultralytics.nn.modules.pointrend import select_uncertain_points_test

    logits = torch.tensor([[[[-5.0, -1.0], [0.01, 3.0]]]])
    indices, coords = select_uncertain_points_test(logits, 1)

    assert indices.item() == 2
    assert torch.allclose(coords[0, 0], torch.tensor([0.25, 0.75]))


def test_pointrend_refinement_shapes_and_gradients():
    from ultralytics.nn.modules.pointrend import PointRendConfig, PointRendInstances, PointRendRefiner

    cfg = PointRendConfig.from_args(_point_args())
    refiner = PointRendRefiner([4], cfg)
    source = torch.randn(2, 4, 16, 16, requires_grad=True)
    fine = refiner.project_features([source])
    coarse = torch.randn(2, 1, 8, 8, requires_grad=True)
    gt = torch.zeros(2, 1, 32, 32)
    gt[:, :, 8:24, 8:24] = 1
    instances = PointRendInstances(
        coarse_logits=coarse,
        boxes=torch.tensor([[4.0, 4.0, 28.0, 28.0], [2.0, 3.0, 26.0, 30.0]]),
        batch_indices=torch.tensor([0, 1]),
        fine_features=fine,
        image_shape=(32, 32),
        gt_masks=gt,
    )

    loss = refiner.point_loss(instances)
    refined = refiner.refine(instances)
    loss.backward()

    assert refined.shape == (2, 1, 32, 32)
    assert torch.isfinite(loss)
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert coarse.grad is not None and coarse.grad.abs().sum() > 0
    assert refiner.point_head.fc_logits.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["joint", "frozen"])
def test_yolo_segment_pointrend_loss_and_trainability(mode):
    from ultralytics.nn.modules.pointrend import configure_pointrend
    from ultralytics.nn.tasks import SegmentationModel

    torch.manual_seed(0)
    model = SegmentationModel(_tiny_segment_cfg(), ch=3, nc=1, verbose=False)
    args = _point_args(pointrend_mode=mode)
    model.args = args
    configure_pointrend(model, args)
    if mode == "frozen":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = ".point_rend." in name

    loss, items = model.loss(_segment_batch())
    loss.sum().backward()
    point_grads = [p.grad for n, p in model.named_parameters() if ".point_rend." in n]
    base_grads = [p.grad for n, p in model.named_parameters() if ".point_rend." not in n]

    assert items.shape == (8,)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    assert any(grad is not None and grad.abs().sum() > 0 for grad in point_grads)
    if mode == "joint":
        assert any(grad is not None and grad.abs().sum() > 0 for grad in base_grads)
    else:
        assert all(grad is None for grad in base_grads)


def test_pointrend_adapter_empty_instances():
    from ultralytics.nn.modules.pointrend import (
        PointRendConfig,
        PointRendRefiner,
        YOLOPrototypePointRendAdapter,
    )

    refiner = PointRendRefiner([4], PointRendConfig.from_args(_point_args()))
    fine = refiner.project_features([torch.randn(1, 4, 8, 8)])
    adapter = YOLOPrototypePointRendAdapter(refiner)
    instances = adapter.from_coefficients(
        coefficients=torch.zeros(0, 8),
        prototypes=torch.randn(1, 8, 8, 8),
        boxes=torch.zeros(0, 4),
        batch_indices=torch.zeros(0, dtype=torch.long),
        fine_features=fine,
        image_shape=(32, 32),
    )

    assert refiner.refine(instances).shape == (0, 1, 32, 32)


def test_mask2former_pointrend_joint_loss():
    from ultralytics.nn.modules import Mask2FormerHead
    from ultralytics.nn.modules.pointrend import configure_pointrend
    from ultralytics.utils.loss import Mask2FormerInstanceLoss

    cfg = {
        "feature_strides": [2, 4, 8, 16],
        "transformer_in_features": [1, 2, 3],
        "common_stride": 2,
        "conv_dim": 32,
        "mask_dim": 32,
        "hidden_dim": 32,
        "num_queries": 4,
        "nheads": 4,
        "dec_layers": 2,
        "enc_layers": 1,
        "dim_feedforward": 32,
        "encoder_dim_feedforward": 32,
        "train_num_points": 16,
    }
    head = Mask2FormerHead(1, cfg, ch=[8, 16, 32, 64])

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.model = torch.nn.ModuleList([module])
            self.yaml = {"nc": 1}
            self.args = _point_args(overlap_mask=False)

    model = Wrapper(head)
    configure_pointrend(model, model.args)
    criterion = Mask2FormerInstanceLoss(model)
    features = [
        torch.randn(1, 8, 32, 32),
        torch.randn(1, 16, 16, 16),
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 64, 4, 4),
    ]
    batch = _segment_batch()

    loss, items = criterion(head.train()(features), batch)
    loss.backward()

    assert items.shape == (4,)
    assert torch.isfinite(loss) and torch.isfinite(items).all()
    assert head.point_rend.point_head.fc_logits.weight.grad.abs().sum() > 0


def test_mask_rcnn_pointrend_joint_loss():
    from ultralytics.nn.modules import MaskRCNNHead
    from ultralytics.nn.modules.pointrend import configure_pointrend

    head = MaskRCNNHead(
        [8],
        1,
        cfg={
            "rpn": {
                "strides": [4],
                "anchor_scales": [1],
                "anchor_ratios": [1.0],
                "pre_nms_topk_train": 32,
                "post_nms_topk_train": 16,
                "pre_nms_topk_test": 16,
                "post_nms_topk_test": 8,
                "samples_per_img": 16,
            },
            "roi": {"featmap_strides": [4], "pool_size": 4, "mask_pool_size": 8},
            "train": {"samples_per_img": 16, "pos_fraction": 0.5},
            "bbox_head": {"hidden_dim": 32},
            "mask_head": {"dim": 8, "num_convs": 1, "resolution": 16},
        },
    )

    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.model = torch.nn.ModuleList([module])

    wrapper = Wrapper(head)
    configure_pointrend(wrapper, _point_args())
    features = [torch.randn(1, 8, 16, 16, requires_grad=True)]
    batch = _segment_batch()

    loss, items = head.loss(features, batch)
    loss.backward()

    assert items.shape == (6,)
    assert torch.isfinite(loss) and torch.isfinite(items).all()
    assert head.point_rend.point_head.fc_logits.weight.grad.abs().sum() > 0


def test_rtdetr_pointrend_final_match_loss():
    from ultralytics.nn.modules.pointrend import PointRendConfig, PointRendRefiner
    from ultralytics.utils.loss import RTDETRSegmentLoss

    torch.manual_seed(0)
    refiner = PointRendRefiner([8], PointRendConfig.from_args(_point_args()))
    criterion = RTDETRSegmentLoss(
        nc=1,
        aux_loss=False,
        overlap_mask=False,
        point_rend=refiner,
    )
    pred_bboxes = torch.rand(2, 1, 4, 4, requires_grad=True)
    pred_bboxes.data[..., 2:] = pred_bboxes.data[..., 2:] * 0.4 + 0.2
    pred_scores = torch.randn(2, 1, 4, 1, requires_grad=True)
    dec_coefficients = torch.randn(1, 1, 4, 8, requires_grad=True)
    enc_coefficients = torch.randn(1, 4, 8, requires_grad=True)
    prototypes = torch.randn(1, 8, 16, 16, requires_grad=True)
    source = torch.randn(1, 8, 16, 16, requires_grad=True)
    fine = refiner.project_features([source])
    masks = torch.zeros(1, 64, 64)
    masks[0, 18:46, 20:44] = 1
    batch = {
        "cls": torch.zeros(1, dtype=torch.long),
        "bboxes": torch.tensor([[0.5, 0.5, 0.375, 0.4375]]),
        "batch_idx": torch.zeros(1, dtype=torch.long),
        "gt_groups": [1],
        "masks": masks,
        "imgsz": torch.tensor([64, 64]),
    }

    losses = criterion(
        (pred_bboxes, pred_scores),
        batch,
        masks=(dec_coefficients, enc_coefficients, prototypes),
        pointrend_features=fine,
    )
    total = sum(losses.values())
    total.backward()

    assert "loss_point" in losses
    assert torch.isfinite(total)
    assert refiner.point_head.fc_logits.weight.grad.abs().sum() > 0
