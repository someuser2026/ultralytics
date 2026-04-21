from copy import deepcopy
from pathlib import Path

import pytest
import torch

RCNN_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "rcnn"
RCNN_VARIANTS = {
    "oriented_rcnn_r50_fpn_le90.yaml": ("OrientedRCNNHead", "obb"),
    "oriented_rcnn_r50_fpn_le90_smallobj.yaml": ("OrientedRCNNHead", "obb"),
    "rotated_faster_rcnn_unravelnet_fpn_le90.yaml": ("RotatedFasterRCNNHead", "obb"),
    "rotated_faster_rcnn_unravelnet_fpn_le90_smallobj.yaml": ("RotatedFasterRCNNHead", "obb"),
    "mask_rcnn_r50_fpn.yaml": ("MaskRCNNHead", "segment"),
    "mask_rcnn_r50_fpn_smallobj.yaml": ("MaskRCNNHead", "segment"),
    "cascade_mask_rcnn_r50_fpn.yaml": ("CascadeMaskRCNNHead", "segment"),
    "cascade_mask_rcnn_r50_fpn_smallobj.yaml": ("CascadeMaskRCNNHead", "segment"),
}


@pytest.mark.parametrize(("model_name", "expected"), RCNN_VARIANTS.items())
def test_rcnn_variant_yaml_parses(model_name, expected):
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    head_name, _ = expected
    model_cfg = yaml_model_load(RCNN_ROOT / model_name)
    model, save, backbone_layers, head_layers = parse_model(deepcopy(model_cfg), ch=3, verbose=False)

    assert len(model) > 0
    assert isinstance(save, list)
    assert backbone_layers
    assert head_layers
    assert model[-1].__class__.__name__ == head_name


@pytest.mark.parametrize(("model_name", "expected"), RCNN_VARIANTS.items())
def test_rcnn_variant_task_inference(model_name, expected):
    from ultralytics import RCNN, YOLO
    from ultralytics.nn.tasks import guess_model_task, yaml_model_load

    _, task = expected
    model_path = RCNN_ROOT / model_name
    cfg = yaml_model_load(model_path)

    assert guess_model_task(cfg) == task

    model = RCNN(str(model_path))
    assert model.task == task
    assert model.model.task == task

    generic_model = YOLO(str(model_path))
    assert generic_model.__class__.__name__ == "RCNN"
    assert generic_model.task == task


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


def _rcnn_feats(dtype=torch.float32):
    return [
        torch.randn(2, 16, 32, 32, dtype=dtype),
        torch.randn(2, 16, 16, 16, dtype=dtype),
        torch.randn(2, 16, 8, 8, dtype=dtype),
        torch.randn(2, 16, 4, 4, dtype=dtype),
        torch.randn(2, 16, 2, 2, dtype=dtype),
    ]


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


def _segment_batch():
    img = torch.rand(2, 3, 128, 128)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0, 1], dtype=torch.long),
        "cls": torch.tensor([[0], [0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.35, 0.25], [0.45, 0.55, 0.30, 0.28]], dtype=torch.float32),
        "masks": torch.zeros(2, 128, 128, dtype=torch.float32),
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
    }


@pytest.mark.parametrize(
    ("model_name", "task"),
    [
        ("mask_rcnn_r50_fpn.yaml", "segment"),
        ("cascade_mask_rcnn_r50_fpn.yaml", "segment"),
        ("oriented_rcnn_r50_fpn_le90.yaml", "obb"),
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

    model.eval()
    with torch.no_grad():
        preds = model(batch["img"])
    assert len(preds) == batch["img"].shape[0]
    for pred in preds:
        assert {"bboxes", "conf", "cls"} <= set(pred)
        if task == "segment":
            assert "masks" in pred
        else:
            if pred["bboxes"].numel():
                assert pred["bboxes"].shape[1] == 5
