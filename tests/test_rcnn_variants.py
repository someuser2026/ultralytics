from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

RCNN_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "rcnn"
RCNN_VARIANTS = {
    "oriented_rcnn_legnet_small_fpn_le90_smallobj.yaml": ("OrientedRCNNHead", "obb"),
    "oriented_rcnn_r50_fpn_le90.yaml": ("OrientedRCNNHead", "obb"),
    "oriented_rcnn_r50_fpn_le90_smallobj.yaml": ("OrientedRCNNHead", "obb"),
    "rotated_faster_rcnn_r50_fpn_le90_smallobj.yaml": ("RotatedFasterRCNNHead", "obb"),
    "rotated_faster_rcnn_unravelnet_fpn_le90.yaml": ("RotatedFasterRCNNHead", "obb"),
    "rotated_faster_rcnn_unravelnet_fpn_le90_smallobj.yaml": ("RotatedFasterRCNNHead", "obb"),
    "mask_rcnn_r50_fpn.yaml": ("MaskRCNNHead", "segment"),
    "mask_rcnn_r50_fpn_smallobj.yaml": ("MaskRCNNHead", "segment"),
    "cascade_mask_rcnn_r50_fpn.yaml": ("CascadeMaskRCNNHead", "segment"),
    "cascade_mask_rcnn_r50_fpn_smallobj.yaml": ("CascadeMaskRCNNHead", "segment"),
    "pointrend_rcnn_r50_fpn_smallobj.yaml": ("PointRendRCNNHead", "segment"),
}


def _disable_test_only_pretrained_download(model_cfg):
    """Keep architecture tests independent of local pretrained-weight caches or network access."""
    model_cfg = deepcopy(model_cfg)
    for layer in model_cfg.get("backbone", []):
        if layer[-2] == "ResNetBackbone" and len(layer[-1]) > 1 and layer[-1][1] == "DEFAULT":
            layer[-1][1] = None
        elif layer[-2] == "Timm" and len(layer[-1]) > 1 and layer[-1][1] is True:
            layer[-1][1] = False
    return model_cfg


@pytest.mark.parametrize(("model_name", "expected"), RCNN_VARIANTS.items())
def test_rcnn_variant_yaml_parses(model_name, expected):
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    head_name, _ = expected
    model_cfg = _disable_test_only_pretrained_download(yaml_model_load(RCNN_ROOT / model_name))
    model, save, backbone_layers, head_layers = parse_model(deepcopy(model_cfg), ch=3, verbose=False)

    assert len(model) > 0
    assert isinstance(save, list)
    assert backbone_layers
    assert head_layers
    assert model[-1].__class__.__name__ == head_name


@pytest.mark.parametrize(("model_name", "expected"), RCNN_VARIANTS.items())
def test_rcnn_variant_task_inference(model_name, expected, tmp_path):
    from ultralytics import RCNN, YOLO
    from ultralytics.nn.tasks import guess_model_task, yaml_model_load
    from ultralytics.utils import YAML

    _, task = expected
    model_path = RCNN_ROOT / model_name
    cfg = yaml_model_load(model_path)

    assert guess_model_task(cfg) == task
    if any(
        len(layer[-1]) > 1
        and (
            (layer[-2] == "ResNetBackbone" and layer[-1][1] == "DEFAULT")
            or (layer[-2] == "Timm" and layer[-1][1] is True)
        )
        for layer in cfg.get("backbone", [])
    ):
        model_path = tmp_path / model_name
        YAML.save(model_path, _disable_test_only_pretrained_download(cfg))

    model = RCNN(str(model_path))
    assert model.task == task
    assert model.model.task == task

    generic_model = YOLO(str(model_path))
    assert generic_model.__class__.__name__ == "RCNN"
    assert generic_model.task == task


def test_smallobj_rcnn_variants_use_reduced_anchor_scales():
    """Every explicit small-object RCNN config should use the requested [1, 2, 4] RPN scales."""
    from ultralytics.nn.tasks import yaml_model_load

    smallobj_configs = sorted(RCNN_ROOT.glob("*smallobj.yaml"))
    assert smallobj_configs
    for model_path in smallobj_configs:
        cfg = yaml_model_load(model_path)
        head_cfg = cfg["head"][-1][3][1]
        assert head_cfg["rpn"]["anchor_scales"] == [1, 2, 4], model_path


def test_anchor_generator_supports_detectron2_and_native_offsets():
    from ultralytics.nn.modules.rcnn import AnchorGenerator

    detectron2 = AnchorGenerator(strides=[4], scales=[1], ratios=[1.0], offset=0.0)
    native = AnchorGenerator(strides=[4], scales=[1], ratios=[1.0])
    detectron2_anchors = detectron2.grid_anchors([(1, 3)], torch.device("cpu"))[0]
    native_anchors = native.grid_anchors([(1, 3)], torch.device("cpu"))[0]
    detectron2_centers = (detectron2_anchors[:, :2] + detectron2_anchors[:, 2:]) * 0.5
    native_centers = (native_anchors[:, :2] + native_anchors[:, 2:]) * 0.5

    assert torch.equal(detectron2_centers, torch.tensor([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]))
    assert torch.equal(native_centers, torch.tensor([[2.0, 2.0], [6.0, 2.0], [10.0, 2.0]]))


@pytest.mark.parametrize("offset", [-0.01, 1.0])
def test_anchor_generator_rejects_invalid_offsets(offset):
    from ultralytics.nn.modules.rcnn import AnchorGenerator

    with pytest.raises(ValueError, match=r"Anchor offset must be in \[0, 1\)"):
        AnchorGenerator(offset=offset)


def test_rcnn_bbox_coders_roundtrip():
    from ultralytics.nn.modules.rcnn import DeltaXYWHAHBBoxCoder, DeltaXYWHAOBBoxCoder, HorizontalBoxCoder

    hanchors = torch.tensor([[10.0, 12.0, 42.0, 60.0], [30.0, 18.0, 70.0, 58.0]])
    htargets = torch.tensor([[12.0, 15.0, 40.0, 54.0], [32.0, 20.0, 66.0, 56.0]])
    hcoder = HorizontalBoxCoder(stds=(1.0, 1.0, 1.0, 1.0))
    assert torch.allclose(hcoder.decode(hanchors, hcoder.encode(hanchors, htargets)), htargets, atol=1e-4)

    rtargets = torch.tensor([[26.0, 34.0, 28.0, 18.0, 0.20], [48.0, 39.0, 20.0, 14.0, -0.35]])
    h2r = DeltaXYWHAHBBoxCoder(stds=(1.0, 1.0, 1.0, 1.0, 1.0), angle_mode="le90", norm_factor=2, edge_swap=True)
    decoded = h2r.decode(hanchors, h2r.encode(hanchors, rtargets))
    assert torch.allclose(decoded[:, :4], rtargets[:, :4], atol=1e-4)

    ranchors = torch.tensor([[26.0, 34.0, 28.0, 18.0, 0.20], [48.0, 39.0, 20.0, 14.0, -0.35]])
    rtargets2 = torch.tensor([[28.0, 33.0, 26.0, 20.0, 0.15], [46.0, 40.0, 18.0, 16.0, -0.20]])
    rcoder = DeltaXYWHAOBBoxCoder(stds=(1.0, 1.0, 1.0, 1.0, 1.0), angle_mode="le90", edge_swap=True, proj_xy=True)
    decoded = rcoder.decode(ranchors, rcoder.encode(ranchors, rtargets2))
    assert torch.allclose(decoded, rtargets2, atol=1e-4)


def test_rotated_roi_align_shape_sanity():
    from ultralytics.nn.modules.rcnn import _rotated_roi_align_multilevel

    feats = [
        torch.randn(2, 16, 32, 32),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 16, 4, 4),
    ]
    rois = torch.tensor(
        [
            [0.0, 48.0, 52.0, 24.0, 16.0, 0.2],
            [1.0, 64.0, 40.0, 36.0, 20.0, -0.3],
        ],
        dtype=torch.float32,
    )
    pooled = _rotated_roi_align_multilevel(feats, rois, output_size=7, sampling_ratio=2, featmap_strides=(4, 8, 16, 32))
    assert pooled.shape == (2, 16, 7, 7)
    assert torch.isfinite(pooled).all()


def test_rotated_roi_align_accepts_half_features():
    from ultralytics.nn.modules.rcnn import _rotated_roi_align_multilevel

    feats = [
        torch.randn(2, 16, 32, 32, dtype=torch.float16),
        torch.randn(2, 16, 16, 16, dtype=torch.float16),
        torch.randn(2, 16, 8, 8, dtype=torch.float16),
        torch.randn(2, 16, 4, 4, dtype=torch.float16),
    ]
    rois = torch.tensor(
        [
            [0.0, 48.0, 52.0, 24.0, 16.0, 0.2],
            [0.0, 32.0, 28.0, 18.0, 14.0, -0.1],
            [1.0, 64.0, 40.0, 36.0, 20.0, -0.3],
        ],
        dtype=torch.float32,
    )

    pooled = _rotated_roi_align_multilevel(feats, rois, output_size=7, sampling_ratio=2, featmap_strides=(4, 8, 16, 32))

    assert pooled.dtype == torch.float16
    assert pooled.shape == (3, 16, 7, 7)
    assert torch.isfinite(pooled.float()).all()


def test_rotated_roi_align_adaptive_sampling_is_finite():
    from ultralytics.nn.modules.rcnn import _rotated_roi_align_multilevel

    feats = [
        torch.randn(2, 16, 32, 32),
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 16, 4, 4),
    ]
    rois = torch.tensor(
        [
            [0.0, 24.0, 18.0, 9.0, 11.0, 0.2],
            [1.0, 64.0, 40.0, 36.0, 20.0, -0.3],
        ],
        dtype=torch.float32,
    )

    pooled = _rotated_roi_align_multilevel(feats, rois, output_size=7, sampling_ratio=0, featmap_strides=(4, 8, 16, 32))

    assert pooled.shape == (2, 16, 7, 7)
    assert torch.isfinite(pooled).all()


def test_rotated_roi_align_offset_cache_reuses_same_key():
    import ultralytics.nn.modules.rcnn as rcnn_module

    rcnn_module._ROTATED_ROI_ALIGN_OFFSET_CACHE.clear()

    first = rcnn_module._get_rotated_roi_align_offset_templates(torch.device("cpu"), 7, 2, 2)
    second = rcnn_module._get_rotated_roi_align_offset_templates(torch.device("cpu"), 7, 2, 2)
    third = rcnn_module._get_rotated_roi_align_offset_templates(torch.device("cpu"), 7, 3, 2)

    assert len(rcnn_module._ROTATED_ROI_ALIGN_OFFSET_CACHE) == 2
    assert first[0] is second[0]
    assert first[1] is second[1]
    assert third[0] is not first[0]
    assert third[1] is not first[1]


def test_rotated_roi_align_adaptive_sampling_reuses_cached_templates():
    import ultralytics.nn.modules.rcnn as rcnn_module

    rcnn_module._ROTATED_ROI_ALIGN_OFFSET_CACHE.clear()
    feat = torch.randn(1, 16, 32, 32)
    rois = torch.tensor([[24.0, 18.0, 9.0, 11.0, 0.2], [24.0, 18.0, 9.0, 11.0, 0.2]], dtype=torch.float32)

    pooled = rcnn_module._rotated_roi_align_single(feat, rois, output_size=7, sampling_ratio=0)

    assert pooled.shape == (2, 16, 7, 7)
    assert len(rcnn_module._ROTATED_ROI_ALIGN_OFFSET_CACHE) == 1


def test_rotated_roi_align_offset_cache_is_bounded(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    rcnn_module._ROTATED_ROI_ALIGN_OFFSET_CACHE.clear()
    monkeypatch.setattr(rcnn_module, "_ROTATED_ROI_ALIGN_OFFSET_CACHE_MAXSIZE", 3)

    for sampling_ratio in range(1, 6):
        rcnn_module._get_rotated_roi_align_offset_templates(torch.device("cpu"), 7, sampling_ratio, sampling_ratio)

    assert len(rcnn_module._ROTATED_ROI_ALIGN_OFFSET_CACHE) == 3
    assert ("cpu", None, 7, 1, 1) not in rcnn_module._ROTATED_ROI_ALIGN_OFFSET_CACHE
    assert ("cpu", None, 7, 2, 2) not in rcnn_module._ROTATED_ROI_ALIGN_OFFSET_CACHE


def test_torchvision_native_roi_align_prefers_registered_op(monkeypatch):
    import ultralytics.nn.modules.roi as roi_module

    calls = {}

    def fake_native(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
        calls["args"] = {
            "input_shape": tuple(input.shape),
            "rois_shape": tuple(rois.shape),
            "spatial_scale": spatial_scale,
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "sampling_ratio": sampling_ratio,
            "aligned": aligned,
        }
        return input.new_zeros((rois.shape[0], input.shape[1], pooled_height, pooled_width))

    def fail_fallback(*args, **kwargs):
        raise AssertionError("native torchvision ROIAlign op should be preferred when it is registered")

    monkeypatch.setattr(roi_module.torch.ops.torchvision, "roi_align", fake_native)
    monkeypatch.setattr(roi_module, "_roi_align_fallback", fail_fallback)

    feat = torch.randn(2, 16, 8, 8)
    rois = torch.tensor([[0.0, 4.0, 4.0, 20.0, 20.0], [1.0, 8.0, 6.0, 24.0, 26.0]], dtype=torch.float32)
    out = roi_module.torchvision_native_roi_align(feat, rois, output_size=7, spatial_scale=0.25, sampling_ratio=0, aligned=True)

    assert out.shape == (2, 16, 7, 7)
    assert calls["args"] == {
        "input_shape": (2, 16, 8, 8),
        "rois_shape": (2, 5),
        "spatial_scale": 0.25,
        "pooled_height": 7,
        "pooled_width": 7,
        "sampling_ratio": 0,
        "aligned": True,
    }


def test_axis_roi_align_multilevel_uses_native_helper(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    calls = []

    def fake_native(feat, rois, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
        calls.append((tuple(feat.shape), tuple(rois.shape), output_size, spatial_scale, sampling_ratio, aligned))
        return feat.new_zeros((rois.shape[0], feat.shape[1], output_size, output_size))

    monkeypatch.setattr(rcnn_module, "torchvision_native_roi_align", fake_native)

    feats = _rcnn_feats()
    rois = torch.tensor(
        [
            [0.0, 8.0, 8.0, 24.0, 24.0],
            [0.0, 40.0, 40.0, 96.0, 96.0],
            [1.0, 72.0, 72.0, 120.0, 120.0],
        ],
        dtype=torch.float32,
    )

    pooled = rcnn_module._roi_align_multilevel(feats[:4], rois, output_size=7, sampling_ratio=0, featmap_strides=(4, 8, 16, 32))

    assert pooled.shape == (3, 16, 7, 7)
    assert calls
    assert all(call[2] == 7 for call in calls)
    assert all(call[4] == 0 for call in calls)
    assert all(call[5] is True for call in calls)


def test_rcnn_segment_model_defaults_overlap_mask_false():
    from ultralytics import YOLO

    model = YOLO(str(RCNN_ROOT / "mask_rcnn_r50_fpn.yaml"))

    assert model.overrides["overlap_mask"] is False
    assert model.model.args["overlap_mask"] is False


def test_split_targets_rejects_lossy_overlap_masks():
    from ultralytics.nn.modules.rcnn import _split_targets

    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0, 0], dtype=torch.long),
        "cls": torch.tensor([[0], [0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]], dtype=torch.float32),
        "masks": torch.ones(1, 8, 8, dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="overlap_mask=False"):
        _split_targets(batch, "segment")


def _rcnn_feats(dtype=torch.float32):
    return [
        torch.randn(2, 16, 32, 32, dtype=dtype),
        torch.randn(2, 16, 16, 16, dtype=dtype),
        torch.randn(2, 16, 8, 8, dtype=dtype),
        torch.randn(2, 16, 4, 4, dtype=dtype),
        torch.randn(2, 16, 2, 2, dtype=dtype),
    ]


class _FixedCascadeBBoxHead(torch.nn.Module):
    """Deterministic class-agnostic bbox head for proposal-flow tests."""

    def __init__(self, dx=0.0):
        super().__init__()
        self.dx = float(dx)

    def forward(self, pooled):
        logits = pooled.new_zeros((pooled.shape[0], 2))
        deltas = pooled.new_zeros((pooled.shape[0], 4))
        deltas[:, 0] = self.dx
        return logits, deltas


def _cascade_flow_test_head(samples_per_img=8, dx=0.0):
    from ultralytics.nn.modules.rcnn import CascadeMaskRCNNHead

    head = CascadeMaskRCNNHead(
        [1],
        1,
        cfg={
            "rpn": {"strides": [4], "anchor_scales": [1], "anchor_ratios": [1.0]},
            "roi": {"pool_size": 1, "mask_pool_size": 1, "sampling_ratio": 0, "featmap_strides": [4]},
            "train": {"samples_per_img": samples_per_img, "pos_fraction": 0.25},
            "bbox_head": {"hidden_dim": 4},
            "mask_head": {"dim": 2, "num_convs": 1, "resolution": 2},
        },
    )
    head.bbox_heads = torch.nn.ModuleList([_FixedCascadeBBoxHead(dx) for _ in range(head.num_stages)])
    return head


def test_rpn_negative_only_image_has_objectness_loss_and_gradient():
    from ultralytics.nn.modules.rcnn import MaskRCNNHead

    head = MaskRCNNHead(
        [8, 8],
        1,
        cfg={
            "rpn": {
                "strides": [4, 8],
                "anchor_scales": [1],
                "anchor_ratios": [1.0],
                "pre_nms_topk_train": 20,
                "post_nms_topk_train": 10,
                "samples_per_img": 8,
            },
            "roi": {"featmap_strides": [4, 8]},
        },
    )
    feats = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 2, 2)]
    cls_loss, box_loss, _ = head._rpn_loss_and_proposals(
        feats, [torch.empty((0, 4))], image_shape=(16, 16), train=True
    )
    (cls_loss + box_loss).backward()

    assert torch.isfinite(cls_loss) and cls_loss > 0
    assert box_loss == 0
    assert head.rpn_head.obj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(("reassign_gt", "expected_target"), [(True, 10.0), (False, 20.0)])
def test_rpn_low_quality_match_regression_gt_semantics(reassign_gt, expected_target):
    from ultralytics.nn.modules.rcnn import _assign_and_sample_rpn

    anchors = torch.zeros(4, 4)
    match_gt = torch.zeros(2, 4)
    regression_gt = torch.tensor([[10.0, 0, 0, 0], [20.0, 0, 0, 0]])
    overlaps = torch.tensor([[0.4, 0.5], [0.1, 0.8], [0.0, 0.0], [0.0, 0.0]])
    cfg = {
        "pos_iou": 0.7,
        "neg_iou": 0.3,
        "min_pos_iou": 0.3,
        "match_low_quality": True,
        "gt_max_assign_all": True,
        "low_quality_reassign_gt": reassign_gt,
        "samples_per_img": 4,
        "pos_fraction": 1.0,
    }

    sampled, labels, targets = _assign_and_sample_rpn(
        anchors,
        match_gt,
        regression_gt,
        cfg,
        lambda _anchors, _gt: overlaps,
        lambda _anchors, target: target,
        4,
    )

    assert sampled.numel() <= cfg["samples_per_img"]
    assert labels[:2].tolist() == [1, 1]
    assert targets[0, 0] == expected_target
    assert targets[1, 0] == 20


def test_rpn_low_quality_matching_respects_minimum_iou():
    from ultralytics.nn.modules.rcnn import _assign_and_sample_rpn

    anchors = torch.zeros(3, 4)
    gt = torch.zeros(1, 4)
    cfg = {
        "pos_iou": 0.7,
        "neg_iou": 0.3,
        "min_pos_iou": 0.3,
        "match_low_quality": True,
        "gt_max_assign_all": True,
        "samples_per_img": 3,
        "pos_fraction": 1.0,
    }
    _, labels, _ = _assign_and_sample_rpn(
        anchors,
        gt,
        gt,
        cfg,
        lambda _anchors, _gt: torch.zeros(3, 1),
        lambda source, _target: source,
        4,
    )

    assert not (labels == 1).any()


def test_rpn_detectron2_low_quality_matching_promotes_best_anchor_below_threshold():
    from ultralytics.nn.modules.rcnn import _assign_and_sample_rpn

    anchors = torch.zeros(3, 4)
    gt = torch.zeros(1, 4)
    cfg = {
        "pos_iou": 0.7,
        "neg_iou": 0.3,
        "min_pos_iou": 0.0,
        "match_low_quality": True,
        "gt_max_assign_all": True,
        "low_quality_reassign_gt": False,
        "samples_per_img": 3,
        "pos_fraction": 1.0,
    }
    _, labels, _ = _assign_and_sample_rpn(
        anchors,
        gt,
        gt,
        cfg,
        lambda _anchors, _gt: torch.tensor([[0.1], [0.2], [0.05]]),
        lambda source, _target: source,
        4,
    )

    assert labels.tolist() == [0, 1, 0]


@pytest.mark.parametrize(
    ("normalizer", "expected"),
    [("sampled_anchors", torch.log(torch.tensor(2.0))), ("fixed_batch_size", torch.log(torch.tensor(2.0)) / 8)],
)
def test_rpn_loss_normalization_modes_on_restricted_anchor_set(normalizer, expected):
    from ultralytics.nn.modules.rcnn import MaskRCNNHead

    head = MaskRCNNHead(
        [1],
        1,
        cfg={
            "rpn": {
                "strides": [4],
                "anchor_scales": [1],
                "anchor_ratios": [1.0],
                "pre_nms_topk_train": 1,
                "post_nms_topk_train": 1,
                "samples_per_img": 8,
                "loss_normalizer": normalizer,
            },
            "roi": {"featmap_strides": [4]},
        },
    )
    with torch.no_grad():
        for parameter in head.rpn_head.parameters():
            parameter.zero_()

    cls_loss, box_loss, _ = head._rpn_loss_and_proposals(
        [torch.zeros(1, 1, 1, 1)],
        [torch.empty((0, 4))],
        image_shape=(4, 4),
        train=True,
    )

    assert torch.allclose(cls_loss, expected)
    assert box_loss == 0


def test_rpn_loss_normalizer_rejects_unknown_mode():
    from ultralytics.nn.modules.rcnn import _rpn_loss_normalizer

    with pytest.raises(ValueError, match="sampled_anchors.*fixed_batch_size"):
        _rpn_loss_normalizer({"loss_normalizer": "unknown", "samples_per_img": 256}, 1, 1)


def test_match_hboxes_preserves_assignment_and_ignored_interval():
    from ultralytics.nn.modules.rcnn import _assign_hboxes, _match_hboxes

    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_labels = torch.tensor([0])
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 8.0, 8.0],
            [20.0, 20.0, 30.0, 30.0],
        ]
    )

    labels, targets, gt_inds = _assign_hboxes(boxes, gt_boxes, gt_labels, pos_iou=0.7, neg_iou=0.5)
    sampled, sampled_labels, sampled_boxes, sampled_targets, sampled_gt_inds = _match_hboxes(
        boxes, gt_boxes, gt_labels, pos_iou=0.7, neg_iou=0.5, num_samples=3, pos_fraction=0.5
    )

    assert labels.tolist() == [1, -1, 0]
    assert torch.equal(targets, gt_boxes.expand_as(boxes))
    assert gt_inds.tolist() == [0, 0, 0]
    assert sampled.tolist() == [0, 2]
    assert sampled_labels.tolist() == [1, 0]
    assert torch.equal(sampled_boxes, boxes[sampled])
    assert torch.equal(sampled_targets, targets[sampled])
    assert torch.equal(sampled_gt_inds, gt_inds[sampled])


def test_gt_aware_hbox_sampling_handles_empty_inputs_and_positive_quota():
    from ultralytics.nn.modules.rcnn import _assign_and_sample_hboxes_with_gt

    gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt_labels = torch.tensor([0])
    proposals = torch.tensor([[1.0, 0.0, 11.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    boxes, labels, targets, gt_inds, is_gt = _assign_and_sample_hboxes_with_gt(
        proposals, gt_boxes, gt_labels, 0.5, 0.5, num_samples=8, pos_fraction=0.5
    )
    assert boxes.shape[0] == 3
    assert labels.tolist() == [1, 1, 0]
    assert is_gt.tolist() == [True, False, False]
    assert torch.equal(boxes[0], gt_boxes[0])
    assert torch.equal(targets[labels > 0], gt_boxes.expand(2, -1))
    assert gt_inds.tolist() == [0, 0, 0]

    no_gt = gt_boxes[:0]
    no_labels = gt_labels[:0]
    boxes, labels, targets, gt_inds, is_gt = _assign_and_sample_hboxes_with_gt(
        proposals, no_gt, no_labels, 0.5, 0.5, num_samples=8, pos_fraction=0.5
    )
    assert torch.equal(boxes, proposals)
    assert labels.tolist() == [0, 0]
    assert targets.shape == (2, 4)
    assert gt_inds.tolist() == [-1, -1]
    assert not is_gt.any()

    boxes, labels, targets, gt_inds, is_gt = _assign_and_sample_hboxes_with_gt(
        proposals[:0], gt_boxes, gt_labels, 0.5, 0.5, num_samples=8, pos_fraction=0.5
    )
    assert torch.equal(boxes, gt_boxes)
    assert labels.tolist() == [1]
    assert torch.equal(targets, gt_boxes)
    assert gt_inds.tolist() == [0]
    assert is_gt.tolist() == [True]

    empty = _assign_and_sample_hboxes_with_gt(
        proposals[:0], no_gt, no_labels, 0.5, 0.5, num_samples=8, pos_fraction=0.5
    )
    assert all(value.shape[0] == 0 for value in empty)

    positive_proposals = gt_boxes.expand(4, -1).clone()
    negative_proposals = torch.tensor(
        [
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
            [60.0, 60.0, 70.0, 70.0],
            [80.0, 80.0, 90.0, 90.0],
        ]
    )
    boxes, labels, _, _, _ = _assign_and_sample_hboxes_with_gt(
        torch.cat((positive_proposals, negative_proposals)),
        gt_boxes,
        gt_labels,
        0.5,
        0.5,
        num_samples=4,
        pos_fraction=0.25,
    )
    assert boxes.shape[0] == 4
    assert (labels > 0).sum() == 1
    assert (labels == 0).sum() == 3


def test_cascade_samples_before_bbox_forward_and_readds_exact_gt(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    head = _cascade_flow_test_head(samples_per_img=8, dx=1.0)
    bbox_rois, mask_calls = [], []

    def fake_roi_align(feats, rois, output_size, sampling_ratio, featmap_strides):
        bbox_rois.append(rois.detach().clone())
        return feats[0].new_zeros((rois.shape[0], feats[0].shape[1], output_size, output_size))

    def fake_mask_loss(feats, pos_rois, gt_inds, class_indices, gt_masks, *, stage_idx, include_point):
        mask_calls.append((stage_idx, pos_rois.detach().clone(), gt_inds.clone(), class_indices.clone(), include_point))
        return feats[0].sum() * 0.0

    monkeypatch.setattr(rcnn_module, "_roi_align_multilevel", fake_roi_align)
    monkeypatch.setattr(head, "_mask_loss", fake_mask_loss)

    gt = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    proposals = [torch.tensor([[8.0, 10.0, 18.0, 20.0], [40.0, 40.0, 50.0, 50.0]])]
    feats = [torch.zeros(1, 1, 16, 16)]
    losses, final_carrier, point_loss = head._cascade_stage_loss(
        feats,
        proposals,
        [gt],
        [torch.tensor([0])],
        [torch.ones(1, 64, 64)],
        image_shape=(64, 64),
    )

    stage1_refined_gt = head.stage_coders[0].decode(gt, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    stage2_boxes = bbox_rois[1][:, 1:5]
    assert len(bbox_rois) == head.num_stages
    assert all(rois.shape[0] <= head.cfg["train"]["samples_per_img"] for rois in bbox_rois)
    assert ((stage2_boxes - gt).abs() < 1e-6).all(dim=1).any()
    assert not ((stage2_boxes - stage1_refined_gt).abs() < 1e-6).all(dim=1).any()
    assert final_carrier[0].shape[0] == proposals[0].shape[0]
    assert not ((final_carrier[0] - gt).abs() < 1e-6).all(dim=1).any()
    assert [call[4] for call in mask_calls] == [False, False, True]
    assert ((mask_calls[-1][1][:, 1:5] - gt).abs() < 1e-6).all(dim=1).any()
    assert all(torch.isfinite(loss) for loss in losses)
    assert point_loss is None


def test_cascade_bbox_forward_is_bounded_by_stage_sampler(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    samples_per_img = 512
    head = _cascade_flow_test_head(samples_per_img=samples_per_img)
    bbox_counts = []

    def fake_roi_align(feats, rois, output_size, sampling_ratio, featmap_strides):
        bbox_counts.append(rois.shape[0])
        return feats[0].new_zeros((rois.shape[0], feats[0].shape[1], output_size, output_size))

    monkeypatch.setattr(rcnn_module, "_roi_align_multilevel", fake_roi_align)
    monkeypatch.setattr(
        head,
        "_mask_loss",
        lambda feats, *args, **kwargs: feats[0].sum() * 0.0,
    )

    proposals = [torch.tensor([[20.0, 20.0, 30.0, 30.0]]).expand(2000, -1).clone()]
    head._cascade_stage_loss(
        [torch.zeros(1, 1, 16, 16)],
        proposals,
        [torch.tensor([[0.0, 0.0, 10.0, 10.0]])],
        [torch.tensor([0])],
        [torch.ones(1, 64, 64)],
        image_shape=(64, 64),
    )

    assert bbox_counts == [samples_per_img] * head.num_stages


def test_cascade_sampling_preserves_multi_image_alignment_with_empty_inputs(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    head = _cascade_flow_test_head(samples_per_img=8)
    bbox_rois = []

    def fake_roi_align(feats, rois, output_size, sampling_ratio, featmap_strides):
        bbox_rois.append(rois.detach().clone())
        return feats[0].new_zeros((rois.shape[0], feats[0].shape[1], output_size, output_size))

    monkeypatch.setattr(rcnn_module, "_roi_align_multilevel", fake_roi_align)
    monkeypatch.setattr(
        head,
        "_mask_loss",
        lambda feats, *args, **kwargs: feats[0].sum() * 0.0,
    )

    losses, final_carrier, _ = head._cascade_stage_loss(
        [torch.zeros(2, 1, 16, 16)],
        [torch.empty((0, 4)), torch.tensor([[20.0, 20.0, 30.0, 30.0]])],
        [torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.empty((0, 4))],
        [torch.tensor([0]), torch.empty((0,), dtype=torch.long)],
        [torch.ones(1, 64, 64), None],
        image_shape=(64, 64),
    )

    assert len(bbox_rois) == head.num_stages
    assert all(rois[:, 0].tolist() == [0.0, 1.0] for rois in bbox_rois)
    assert final_carrier[0].shape == (0, 4)
    assert final_carrier[1].shape == (1, 4)
    assert all(torch.isfinite(loss) for loss in losses)


def test_cascade_builds_per_stage_class_specific_masks_and_all_stages_receive_gradients():
    from ultralytics.nn.modules.rcnn import CascadeMaskRCNNHead

    cfg = {
        "rpn": {
            "strides": [4, 8, 16, 32, 64],
            "anchor_scales": [1],
            "anchor_ratios": [1.0],
            "pre_nms_topk_train": 30,
            "post_nms_topk_train": 20,
            "samples_per_img": 16,
        },
        "train": {"samples_per_img": 16},
        "bbox_head": {"hidden_dim": 32},
        "mask_head": {"dim": 8, "num_convs": 1, "class_agnostic": False},
    }
    head = CascadeMaskRCNNHead([16] * 5, 2, cfg)
    batch = _segment_batch()
    batch["cls"] = torch.tensor([[0], [1]], dtype=torch.float32)
    loss, items = head.loss(_rcnn_feats(), batch)
    loss.backward()

    assert head.num_stages == 3
    assert head.stage_loss_weights == (1.0, 0.5, 0.25)
    assert len(head.mask_heads) == 3
    assert all(mask_head.out.out_channels == 2 for mask_head in head.mask_heads)
    assert items.numel() == len(head.loss_names)
    for stage_head in [*head.bbox_heads, *head.mask_heads]:
        gradients = [parameter.grad for parameter in stage_head.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_cascade_reference_flow_preserves_state_dict_schema():
    from ultralytics.nn.modules.rcnn import CascadeMaskRCNNHead

    cfg = {
        "rpn": {"strides": [4], "anchor_scales": [1], "anchor_ratios": [1.0]},
        "roi": {"featmap_strides": [4]},
        "bbox_head": {"hidden_dim": 8},
        "mask_head": {"dim": 4, "num_convs": 1},
    }
    source = CascadeMaskRCNNHead([4], 2, cfg)
    target = CascadeMaskRCNNHead([4], 2, cfg)

    result = target.load_state_dict(source.state_dict(), strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert source.state_dict().keys() == target.state_dict().keys()


def test_cascade_rejects_inconsistent_stage_configuration():
    from ultralytics.nn.modules.rcnn import CascadeMaskRCNNHead

    with pytest.raises(ValueError, match="lengths"):
        CascadeMaskRCNNHead(
            [8],
            1,
            {"cascade": {"iou_thresholds": [0.5, 0.6], "stage_loss_weights": [1.0]}},
        )


def test_cascade_loads_legacy_shared_mask_head_into_every_stage(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module
    from ultralytics.nn.modules.rcnn import CascadeMaskRCNNHead

    cfg = {
        "rpn": {"strides": [4], "anchor_scales": [1], "anchor_ratios": [1.0]},
        "roi": {"featmap_strides": [4]},
        "bbox_head": {"hidden_dim": 8},
        "mask_head": {"dim": 4, "num_convs": 1, "class_agnostic": False},
    }
    source = CascadeMaskRCNNHead([4], 2, cfg)
    target = CascadeMaskRCNNHead([4], 2, cfg)
    legacy = {f"mask_head.{key}": value.clone() for key, value in source.mask_heads[0].state_dict().items()}
    warnings = []
    monkeypatch.setattr(rcnn_module.LOGGER, "warning", warnings.append)

    target.load_state_dict(legacy, strict=False)

    assert any("legacy shared RCNN mask-head weights" in warning for warning in warnings)
    for stage_head in target.mask_heads:
        for key, value in stage_head.state_dict().items():
            assert torch.equal(value, source.mask_heads[0].state_dict()[key])


def test_cascade_inference_averages_stage_logits_and_mask_probabilities(monkeypatch):
    from ultralytics.nn.modules.rcnn import CascadeMaskRCNNHead

    class FixedBBoxHead(torch.nn.Module):
        def __init__(self, class_logit):
            super().__init__()
            self.class_logit = class_logit

        def forward(self, pooled):
            logits = pooled.new_tensor([0.0, self.class_logit]).repeat(pooled.shape[0], 1)
            return logits, pooled.new_zeros((pooled.shape[0], 4))

    class FixedMaskHead(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, pooled):
            return pooled.new_full((pooled.shape[0], 1, 28, 28), self.value)

    head = CascadeMaskRCNNHead(
        [4],
        1,
        {
            "rpn": {"strides": [4], "anchor_scales": [1], "anchor_ratios": [1.0]},
            "roi": {"featmap_strides": [4]},
            "test": {"score_thresh": 0.87, "max_dets": 10},
            "bbox_head": {"hidden_dim": 8},
            "mask_head": {"dim": 4, "num_convs": 1},
        },
    )
    head.bbox_heads = torch.nn.ModuleList([FixedBBoxHead(1.0), FixedBBoxHead(2.0), FixedBBoxHead(3.0)])
    head.mask_heads = torch.nn.ModuleList([FixedMaskHead(10.0), FixedMaskHead(10.0), FixedMaskHead(-10.0)])
    proposal = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    monkeypatch.setattr(head, "_rpn_loss_and_proposals", lambda *args, **kwargs: (torch.tensor(0.0), torch.tensor(0.0), [proposal]))

    pred = head.eval()([torch.zeros(1, 4, 1, 1)])[0]
    expected_score = torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[1]
    legacy_score = torch.stack([torch.softmax(torch.tensor([0.0, value]), 0)[1] for value in (1.0, 2.0, 3.0)]).mean()

    assert pred["conf"].shape == (1,)
    assert torch.allclose(pred["conf"][0], expected_score)
    assert not torch.allclose(pred["conf"][0], legacy_score)
    assert pred["masks"].shape == (1, 4, 4)
    assert pred["masks"].all()  # The final-stage mask alone is negative; probability merging keeps it.


def test_oriented_rcnn_routes_through_rotated_roi_align(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    calls = {"axis": 0, "rotated": 0}

    def fake_axis(*args, **kwargs):
        calls["axis"] += 1
        raise AssertionError("OrientedRCNNHead should not use axis-aligned ROI pooling")

    def fake_rotated(feats, rois, output_size, sampling_ratio=0, featmap_strides=(4, 8, 16, 32)):
        calls["rotated"] += 1
        assert sampling_ratio == 3
        return feats[0].new_zeros((rois.shape[0], feats[0].shape[1], output_size, output_size))

    monkeypatch.setattr(rcnn_module, "_roi_align_multilevel", fake_axis)
    monkeypatch.setattr(rcnn_module, "_rotated_roi_align_multilevel", fake_rotated)

    head = rcnn_module.OrientedRCNNHead([16], 1, cfg={"roi": {"sampling_ratio": 3, "featmap_strides": [4, 8, 16, 32]}})
    rois = torch.tensor([[0.0, 48.0, 52.0, 24.0, 16.0, 0.2], [1.0, 64.0, 40.0, 36.0, 20.0, -0.3]], dtype=torch.float32)
    pooled = head._roi_pool(_rcnn_feats(), rois)

    assert pooled.shape == (2, 16, 7, 7)
    assert calls == {"axis": 0, "rotated": 1}


def test_rotated_faster_rcnn_routes_through_axis_roi_align(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    calls = {"axis": 0, "rotated": 0}

    def fake_axis(feats, rois, output_size, sampling_ratio, featmap_strides=(4, 8, 16, 32)):
        calls["axis"] += 1
        assert sampling_ratio == 3
        return feats[0].new_zeros((rois.shape[0], feats[0].shape[1], output_size, output_size))

    def fake_rotated(*args, **kwargs):
        calls["rotated"] += 1
        raise AssertionError("RotatedFasterRCNNHead should not use rotated ROI pooling")

    monkeypatch.setattr(rcnn_module, "_roi_align_multilevel", fake_axis)
    monkeypatch.setattr(rcnn_module, "_rotated_roi_align_multilevel", fake_rotated)

    head = rcnn_module.RotatedFasterRCNNHead([16], 1, cfg={"roi": {"sampling_ratio": 3, "featmap_strides": [4, 8, 16, 32]}})
    rois = torch.tensor([[0.0, 10.0, 12.0, 42.0, 60.0], [1.0, 30.0, 18.0, 70.0, 58.0]], dtype=torch.float32)
    pooled = head._roi_pool(_rcnn_feats(), rois)

    assert pooled.shape == (2, 16, 7, 7)
    assert calls == {"axis": 1, "rotated": 0}


def test_oriented_rcnn_rpn_targets_accept_amp_deltas():
    from ultralytics.nn.modules.rcnn import OrientedRCNNHead

    class DummyRPN(torch.nn.Module):
        def forward(self, feats):
            return [torch.zeros(1, 1, 1, 1)], [torch.zeros(1, 6, 1, 1, dtype=torch.float16)]

    head = OrientedRCNNHead(
        in_channels=[8],
        nc=1,
        cfg={
            "rpn": {
                "strides": [4],
                "anchor_scales": [1],
                "anchor_ratios": [1.0],
                "pre_nms_topk_train": 1,
                "post_nms_topk_train": 1,
                "pre_nms_topk_test": 1,
                "post_nms_topk_test": 1,
                "samples_per_img": 1,
            },
            "roi": {"featmap_strides": [4]},
        },
    )
    head.rpn_head = DummyRPN()
    feats = [torch.zeros(1, 8, 1, 1)]
    gt_boxes = [torch.tensor([[2.0, 2.0, 4.0, 4.0, 0.0]], dtype=torch.float32)]

    cls_loss, box_loss, proposals = head._rpn_loss_and_proposals(feats, gt_boxes, image_shape=(4, 4), train=True)

    assert torch.isfinite(cls_loss)
    assert torch.isfinite(box_loss)
    assert len(proposals) == 1
    assert proposals[0].shape[-1] == 5


def test_rotated_faster_rcnn_rpn_uses_level_aware_batched_nms(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    class DummyRPN(torch.nn.Module):
        def forward(self, feats):
            return [
                torch.tensor([[[[2.0]], [[1.0]]]], dtype=torch.float32),
                torch.tensor([[[[0.5]]]], dtype=torch.float32),
            ], [
                torch.zeros(1, 8, 1, 1, dtype=torch.float32),
                torch.zeros(1, 4, 1, 1, dtype=torch.float32),
            ]

    calls = {}

    def fake_batched_nms(boxes, scores, idxs, iou_threshold, use_fast_nms=False):
        calls["boxes"] = boxes.clone()
        calls["scores"] = scores.clone()
        calls["idxs"] = idxs.clone()
        calls["iou_threshold"] = iou_threshold
        calls["use_fast_nms"] = use_fast_nms
        return torch.tensor([1, 0], device=boxes.device)

    monkeypatch.setattr(rcnn_module.TorchNMS, "batched_nms", staticmethod(fake_batched_nms))

    head = rcnn_module.RotatedFasterRCNNHead(
        in_channels=[8],
        nc=1,
        cfg={
            "rpn": {
                "strides": [4, 8],
                "anchor_scales": [1],
                "anchor_ratios": [1.0, 2.0],
                "pre_nms_topk_train": 1,
                "post_nms_topk_train": 2,
                "pre_nms_topk_test": 1,
                "post_nms_topk_test": 2,
                "samples_per_img": 1,
            },
            "roi": {"featmap_strides": [4]},
        },
    )
    head.rpn_head = DummyRPN()

    feats = [torch.zeros(1, 8, 1, 1), torch.zeros(1, 8, 1, 1)]
    _, _, proposals = head._rpn_loss_and_proposals(feats, [torch.zeros((0, 5), dtype=torch.float32)], image_shape=(8, 8), train=False)

    assert calls["boxes"].shape == (2, 4)
    assert calls["scores"].shape == (2,)
    assert torch.equal(calls["idxs"], torch.tensor([0, 1]))
    assert calls["iou_threshold"] == pytest.approx(head.cfg["rpn"]["nms_thresh"])
    assert calls["use_fast_nms"] is False
    assert proposals[0].shape == (2, 4)
    assert torch.allclose(proposals[0], calls["boxes"][torch.tensor([1, 0])])


def test_oriented_rcnn_rpn_uses_hbox_level_aware_batched_nms(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    class DummyRPN(torch.nn.Module):
        def forward(self, feats):
            return [
                torch.tensor([[[[2.0]]]], dtype=torch.float32),
                torch.tensor([[[[1.0]]]], dtype=torch.float32),
            ], [
                torch.zeros(1, 6, 1, 1, dtype=torch.float32),
                torch.zeros(1, 6, 1, 1, dtype=torch.float32),
            ]

    calls = {}

    def fake_batched_nms(boxes, scores, idxs, iou_threshold, use_fast_nms=False):
        calls["boxes"] = boxes.clone()
        calls["scores"] = scores.clone()
        calls["idxs"] = idxs.clone()
        calls["iou_threshold"] = iou_threshold
        calls["use_fast_nms"] = use_fast_nms
        return torch.tensor([1, 0], device=boxes.device)

    monkeypatch.setattr(rcnn_module.TorchNMS, "batched_nms", staticmethod(fake_batched_nms))

    head = rcnn_module.OrientedRCNNHead(
        in_channels=[8],
        nc=1,
        cfg={
            "rpn": {
                "strides": [4, 8],
                "anchor_scales": [1],
                "anchor_ratios": [1.0],
                "pre_nms_topk_train": 1,
                "post_nms_topk_train": 2,
                "pre_nms_topk_test": 1,
                "post_nms_topk_test": 2,
                "samples_per_img": 1,
            },
            "roi": {"featmap_strides": [4]},
        },
    )
    head.rpn_head = DummyRPN()

    feats = [torch.zeros(1, 8, 1, 1), torch.zeros(1, 8, 1, 1)]
    _, _, proposals = head._rpn_loss_and_proposals(feats, [torch.zeros((0, 5), dtype=torch.float32)], image_shape=(8, 8), train=False)

    assert calls["boxes"].shape == (2, 4)
    assert calls["scores"].shape == (2,)
    assert torch.equal(calls["idxs"], torch.tensor([0, 1]))
    assert calls["iou_threshold"] == pytest.approx(head.cfg["rpn"]["nms_thresh"])
    assert calls["use_fast_nms"] is False
    assert proposals[0].shape == (2, 5)
    assert torch.allclose(calls["boxes"][torch.tensor([1, 0])], rcnn_module._rboxes_to_xyxy(proposals[0]))


@pytest.mark.parametrize("head_name", ["MaskRCNNHead", "CascadeMaskRCNNHead"])
def test_segment_rcnn_heads_do_not_use_rotated_roi_align(monkeypatch, head_name):
    import ultralytics.nn.modules.rcnn as rcnn_module

    calls = {"axis": 0, "rotated": 0}

    def fake_axis(feats, rois, output_size, sampling_ratio, featmap_strides=(4, 8, 16, 32)):
        calls["axis"] += 1
        return feats[0].new_zeros((rois.shape[0], feats[0].shape[1], output_size, output_size))

    def fake_rotated(*args, **kwargs):
        calls["rotated"] += 1
        raise AssertionError("Segment RCNN heads should not use rotated ROI pooling")

    monkeypatch.setattr(rcnn_module, "_roi_align_multilevel", fake_axis)
    monkeypatch.setattr(rcnn_module, "_rotated_roi_align_multilevel", fake_rotated)

    head_cls = getattr(rcnn_module, head_name)
    head = head_cls([16], 1)
    loss, loss_items = head.loss(_rcnn_feats(), _segment_batch())

    assert torch.isfinite(loss)
    assert torch.isfinite(loss_items).all()
    assert calls["rotated"] == 0
    assert calls["axis"] >= 2


def test_autobackend_preserves_single_rcnn_prediction_dict():
    from ultralytics.nn.autobackend import AutoBackend

    class FakeRCNNModel(torch.nn.Module):
        uses_batch_dict = True

        def forward(self, batch, mode="auto", **kwargs):
            assert mode == "predict"
            assert set(batch) == {"img"}
            assert batch["img"].shape[0] == 1
            device = batch["img"].device
            return [
                {
                    "bboxes": torch.tensor([[1.0, 2.0, 10.0, 12.0]], device=device),
                    "conf": torch.tensor([0.9], device=device),
                    "cls": torch.tensor([0.0], device=device),
                    "masks": None,
                }
            ]

    backend = AutoBackend.__new__(AutoBackend)
    torch.nn.Module.__init__(backend)
    backend.fp16 = False
    backend.nhwc = False
    backend.pt = True
    backend.nn_module = False
    backend.model = FakeRCNNModel()
    backend.names = {i: f"class{i}" for i in range(999)}
    backend.task = "segment"
    backend.device = torch.device("cpu")

    out = backend.forward(torch.zeros(1, 3, 32, 32))

    assert isinstance(out, list)
    assert len(out) == 1
    assert isinstance(out[0], dict)
    assert {"bboxes", "conf", "cls", "masks"} <= set(out[0])


def test_mask_targets_use_fractional_aligned_roi_align_and_binary_threshold(monkeypatch):
    import ultralytics.nn.modules.rcnn as rcnn_module

    calls = {}

    def fake_roi_align(input, rois, output_size, spatial_scale, sampling_ratio, aligned):
        calls.update(
            input=input.clone(),
            rois=rois.clone(),
            output_size=output_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )
        return input.new_tensor([[[[0.49, 0.50], [0.51, 1.00]]]])

    monkeypatch.setattr(rcnn_module, "torchvision_native_roi_align", fake_roi_align)
    gt_masks = torch.stack((torch.zeros(8, 10), torch.ones(8, 10)))
    proposals = torch.tensor([[-0.25, 1.25, 9.60, 8.75]])
    targets = rcnn_module._crop_mask_targets(gt_masks, proposals, torch.tensor([1]), mask_size=2)

    assert torch.equal(calls["input"], gt_masks[1:2, None])
    assert torch.allclose(calls["rois"], torch.tensor([[0.0, 0.0, 1.25, 9.60, 8.0]]))
    assert calls["output_size"] == 2
    assert calls["spatial_scale"] == 1.0
    assert calls["sampling_ratio"] == 0
    assert calls["aligned"] is True
    assert torch.equal(targets, torch.tensor([[[0.0, 1.0], [1.0, 1.0]]]))


def test_paste_masks_matches_continuous_probability_reference():
    import ultralytics.nn.modules.rcnn as rcnn_module
    from ultralytics.nn.modules.pointrend import paste_roi_probabilities

    logits = torch.linspace(-4.0, 4.0, 16).reshape(1, 4, 4)
    boxes = torch.tensor([[1.25, 0.75, 7.60, 6.40]])

    masks = rcnn_module._paste_masks(logits, boxes, image_shape=(7, 9), mask_threshold=0.5)
    expected = paste_roi_probabilities(logits[:, None], boxes, (7, 9)) >= 0.5

    assert torch.equal(masks, expected)


def test_paste_masks_thresholds_to_binary_masks():
    import ultralytics.nn.modules.rcnn as rcnn_module

    logits = torch.logit(torch.tensor([0.49, 0.50, 0.51], dtype=torch.float32)).view(3, 1, 1).expand(3, 2, 2).clone()
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]], dtype=torch.float32)

    masks = rcnn_module._paste_masks(logits, boxes, image_shape=(2, 2), mask_threshold=0.5)

    assert masks.dtype == torch.bool
    assert masks.flatten(1).sum(1).tolist() == [0, 4, 4]


def test_mask_threshold_falls_back_for_old_checkpoint_cfg():
    import ultralytics.nn.modules.rcnn as rcnn_module

    assert rcnn_module._mask_threshold_from_cfg({"test": {"score_thresh": 0.05}}) == pytest.approx(0.5)
    assert rcnn_module._mask_threshold_from_cfg({"test": {"mask_threshold": 0.25}}) == pytest.approx(0.25)


@pytest.mark.parametrize(("head_name", "num_stages"), [("MaskRCNNHead", 1), ("CascadeMaskRCNNHead", 3)])
def test_empty_mask_predictions_retain_detections(monkeypatch, head_name, num_stages):
    import ultralytics.nn.modules.rcnn as rcnn_module

    class FixedBBoxHead(torch.nn.Module):
        def forward(self, pooled):
            return pooled.new_tensor([[0.0, 10.0]]).expand(pooled.shape[0], -1), pooled.new_zeros(
                (pooled.shape[0], 4)
            )

    class EmptyMaskHead(torch.nn.Module):
        def forward(self, pooled):
            return pooled.new_full((pooled.shape[0], 1, 28, 28), -100.0)

    head = getattr(rcnn_module, head_name)(
        [4],
        1,
        {
            "rpn": {"strides": [4], "anchor_scales": [1], "anchor_ratios": [1.0]},
            "roi": {"featmap_strides": [4]},
            "test": {"score_thresh": 0.0, "max_dets": 10},
            "bbox_head": {"hidden_dim": 8},
            "mask_head": {"dim": 4, "num_convs": 1},
        },
    )
    head.bbox_heads = torch.nn.ModuleList([FixedBBoxHead() for _ in range(num_stages)])
    head.mask_heads = torch.nn.ModuleList([EmptyMaskHead() for _ in range(num_stages)])
    proposal = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    monkeypatch.setattr(
        head,
        "_rpn_loss_and_proposals",
        lambda *args, **kwargs: (torch.tensor(0.0), torch.tensor(0.0), [proposal]),
    )

    pred = head.eval()([torch.zeros(1, 4, 1, 1)])[0]

    assert pred["bboxes"].shape == (1, 4)
    assert pred["conf"].shape == (1,)
    assert pred["cls"].shape == (1,)
    assert pred["masks"].shape == (1, 4, 4)
    assert not pred["masks"].any()
    assert pred["mask_roi_logits"].shape == (1, 1, 28, 28)


def test_rcnn_segmentation_predictor_keeps_fractional_non_empty_masks():
    from ultralytics.models.rcnn.predict import RCNNSegmentationPredictor

    predictor = RCNNSegmentationPredictor.__new__(RCNNSegmentationPredictor)
    predictor.batch = (["example.png"],)
    predictor.model = SimpleNamespace(names={0: "rip"})

    masks = torch.zeros(1, 32, 32, dtype=torch.float32)
    masks[0, 5:20, 4:18] = 0.9
    pred = {
        "bboxes": torch.tensor([[4.0, 5.0, 18.0, 20.0]], dtype=torch.float32),
        "conf": torch.tensor([0.95], dtype=torch.float32),
        "cls": torch.tensor([0.0], dtype=torch.float32),
        "masks": masks,
    }
    img = torch.zeros(1, 3, 32, 32)
    orig_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]

    results = predictor.postprocess(pred, img, orig_imgs)

    assert len(results) == 1
    assert results[0].path == "example.png"
    assert results[0].boxes is not None
    assert results[0].boxes.shape[0] == 1
    assert results[0].masks is not None
    assert results[0].masks.data.dtype == torch.bool
    assert int(results[0].masks.data.sum()) > 0


@pytest.mark.parametrize("masks", [None, torch.zeros(1, 32, 32, dtype=torch.float32)])
def test_rcnn_segmentation_predictor_retains_predictions_with_empty_masks(masks):
    from ultralytics.models.rcnn.predict import RCNNSegmentationPredictor

    predictor = RCNNSegmentationPredictor.__new__(RCNNSegmentationPredictor)
    predictor.batch = (["example.png"],)
    predictor.model = SimpleNamespace(names={0: "rip"})

    pred = {
        "bboxes": torch.tensor([[4.0, 5.0, 18.0, 20.0]], dtype=torch.float32),
        "conf": torch.tensor([0.95], dtype=torch.float32),
        "cls": torch.tensor([0.0], dtype=torch.float32),
        "masks": masks,
    }
    img = torch.zeros(1, 3, 32, 32)
    orig_imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]

    results = predictor.postprocess(pred, img, orig_imgs)

    assert len(results) == 1
    assert results[0].boxes is not None
    assert results[0].boxes.shape[0] == 1
    assert results[0].masks is not None
    assert results[0].masks.data.shape == (1, 32, 32)
    assert not results[0].masks.data.any()


def _segment_batch():
    img = torch.rand(2, 3, 128, 128)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.tensor([[0], [0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.35, 0.25], [0.45, 0.55, 0.30, 0.28]], dtype=torch.float32),
        "masks": torch.zeros(2, 128, 128, dtype=torch.float32),
        "metadata_vec": torch.randn(2, 4),
    }
    batch["masks"][0, 40:88, 36:92] = 1
    batch["masks"][1, 44:92, 28:86] = 1
    return batch


def _obb_batch():
    return {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.tensor([[0], [0]], dtype=torch.float32),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.28, 0.18, 0.10], [0.42, 0.58, 0.24, 0.20, -0.25]],
            dtype=torch.float32,
        ),
        "metadata_vec": torch.randn(2, 4),
    }


def _metadata_enabled_rcnn_cfg():
    return {
        "nc": 1,
        "stride": 16,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
        ],
        "head": [
            [[0, 1, 2, 3], 1, "FPN", [16, {"num_outs": 4, "metadata_cfg": {"enabled": True, "mode": "film_gate", "hidden_dim": 8}}]],
            [[4], 1, "MaskRCNNHead", [1, {
                "rpn": {
                    "anchor_scales": [1],
                    "anchor_ratios": [1.0],
                    "strides": [2, 4, 8, 16],
                    "pre_nms_topk_train": 50,
                    "post_nms_topk_train": 50,
                    "pre_nms_topk_test": 20,
                    "post_nms_topk_test": 20,
                    "nms_thresh": 0.7,
                    "min_box_size": 0.0,
                    "pos_iou": 0.7,
                    "neg_iou": 0.3,
                    "samples_per_img": 16,
                    "pos_fraction": 0.5,
                    "beta": 0.1111111111111111,
                },
                "roi": {"pool_size": 7, "mask_pool_size": 14, "sampling_ratio": 0, "featmap_strides": [2, 4, 8, 16]},
                "train": {"pos_iou": 0.5, "neg_iou": 0.5, "samples_per_img": 16, "pos_fraction": 0.25},
                "test": {"score_thresh": 0.05, "nms_iou": 0.5, "max_dets": 20},
                "bbox_head": {"hidden_dim": 64, "loss": "smooth_l1", "beta": 1.0},
                "mask_head": {"dim": 16, "num_convs": 2, "resolution": 28},
            }]],
        ],
    }


@pytest.mark.parametrize(
    ("model_name", "task"),
    [
        ("mask_rcnn_r50_fpn.yaml", "segment"),
        ("cascade_mask_rcnn_r50_fpn.yaml", "segment"),
        ("oriented_rcnn_legnet_small_fpn_le90_smallobj.yaml", "obb"),
        ("oriented_rcnn_r50_fpn_le90.yaml", "obb"),
        ("rotated_faster_rcnn_r50_fpn_le90_smallobj.yaml", "obb"),
        ("rotated_faster_rcnn_unravelnet_fpn_le90.yaml", "obb"),
    ],
)
def test_rcnn_variant_forward_and_loss_smoke(model_name, task):
    from ultralytics.nn.tasks import RCNNOBBModel, RCNNSegmentationModel

    model_cls = RCNNSegmentationModel if task == "segment" else RCNNOBBModel
    batch = _segment_batch() if task == "segment" else _obb_batch()
    model = model_cls(str(RCNN_ROOT / model_name), nc=1, ch=3, verbose=False)

    model.train()
    loss, loss_items = model(batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_items).all()
    loss.backward()

    if task == "segment":
        model.model[-1].cfg["test"].pop("mask_threshold", None)
    model.eval()
    with torch.no_grad():
        with pytest.raises(TypeError, match="batch dict"):
            model(batch["img"])
        preds = model({"img": batch["img"], "metadata_vec": batch["metadata_vec"]}, mode="predict")
    assert len(preds) == batch["img"].shape[0]
    for pred in preds:
        assert {"bboxes", "conf", "cls"} <= set(pred)
        if task == "segment":
            assert "masks" in pred
            assert pred["masks"] is not None
            assert pred["masks"].shape[0] == pred["bboxes"].shape[0]
            assert pred["masks"].dtype == torch.bool
            assert pred["mask_roi_logits"] is not None
            assert pred["mask_roi_logits"].shape[0] == pred["bboxes"].shape[0]
        else:
            if pred["bboxes"].numel():
                assert pred["bboxes"].shape[1] == 5


def test_metadata_enabled_rcnn_requires_and_uses_batch_metadata():
    from ultralytics.nn.tasks import RCNNSegmentationModel

    model = RCNNSegmentationModel(_metadata_enabled_rcnn_cfg(), nc=1, ch=3, verbose=False)
    batch = _segment_batch()

    missing_metadata = dict(batch)
    missing_metadata.pop("metadata_vec")
    model.train()
    with pytest.raises(ValueError, match="metadata-conditioned neck"):
        model(missing_metadata, mode="loss")

    loss, loss_items = model(batch, mode="loss")
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_items).all()

    model.eval()
    with torch.no_grad():
        preds = model({"img": batch["img"], "metadata_vec": batch["metadata_vec"]}, mode="predict")
    assert len(preds) == batch["img"].shape[0]
