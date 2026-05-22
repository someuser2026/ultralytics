from importlib.util import find_spec

import pytest

TORCH_READY = find_spec("torch") is not None


if TORCH_READY:
    import torch


def _features(ch=(16, 32, 64), sizes=(16, 8, 4)):
    return [torch.randn(2, c, s, s) for c, s in zip(ch, sizes)]


def _set_stride(head, values=(8.0, 16.0, 32.0)):
    head.stride = torch.tensor(values)
    return head


def _assert_base_raw(raw, nl: int, batch: int = 2):
    assert {"boxes", "scores", "feats"} <= set(raw)
    assert len(raw["feats"]) == nl
    assert raw["boxes"].shape[0] == batch
    assert raw["scores"].shape[0] == batch


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_detect_segment_obb_pose_training_outputs_are_dicts():
    from ultralytics.nn.modules.head import Detect, OBB, Pose, Segment

    detect = _set_stride(Detect(nc=3, ch=(16, 32, 64))).train()
    raw = detect(_features())
    _assert_base_raw(raw, nl=3)

    segment = _set_stride(Segment(nc=3, nm=8, npr=16, ch=(16, 32, 64))).train()
    raw = segment(_features())
    _assert_base_raw(raw, nl=3)
    assert {"mask_coefficient", "proto"} <= set(raw)

    obb = _set_stride(OBB(nc=3, ch=(16, 32, 64))).train()
    raw = obb(_features())
    _assert_base_raw(raw, nl=3)
    assert "angle" in raw

    pose = _set_stride(Pose(nc=3, kpt_shape=(5, 3), ch=(16, 32, 64))).train()
    raw = pose(_features())
    _assert_base_raw(raw, nl=3)
    assert "kpts" in raw


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_detect_family_eval_outputs_keep_raw_dicts():
    from ultralytics.nn.modules.head import Detect, OBB, Pose, Segment

    detect = _set_stride(Detect(nc=3, ch=(16, 32, 64))).eval()
    pred, raw = detect(_features())
    _assert_base_raw(raw, nl=3)
    assert pred.shape[1] == 4 + detect.nc

    segment = _set_stride(Segment(nc=3, nm=8, npr=16, ch=(16, 32, 64))).eval()
    (pred, proto), raw = segment(_features())
    _assert_base_raw(raw, nl=3)
    assert pred.shape[1] == 4 + segment.nc + segment.nm
    assert proto.shape[1] == segment.nm

    obb = _set_stride(OBB(nc=3, ch=(16, 32, 64))).eval()
    pred, raw = obb(_features())
    _assert_base_raw(raw, nl=3)
    assert pred.shape[1] == 4 + obb.nc + obb.ne

    pose = _set_stride(Pose(nc=3, kpt_shape=(5, 3), ch=(16, 32, 64))).eval()
    pred, raw = pose(_features())
    _assert_base_raw(raw, nl=3)
    assert pred.shape[1] == 4 + pose.nc + pose.nk


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_shoreaux_heads_wrap_main_dict_during_training_only():
    from ultralytics.nn.modules.head import OBBShoreAux, SegmentShoreAux

    obb = _set_stride(OBBShoreAux(nc=2, ch=(16, 32, 64))).train()
    raw = obb(_features())
    assert set(raw) == {"main", "shore_aux_logits"}
    _assert_base_raw(raw["main"], nl=3)
    assert "angle" in raw["main"]

    segment = _set_stride(SegmentShoreAux(nc=2, nm=8, npr=16, ch=(16, 32, 64))).train()
    raw = segment(_features())
    assert set(raw) == {"main", "shore_aux_logits"}
    _assert_base_raw(raw["main"], nl=3)
    assert {"mask_coefficient", "proto"} <= set(raw["main"])

    obb.eval()
    _, eval_raw = obb(_features())
    _assert_base_raw(eval_raw, nl=3)
    segment.eval()
    (_, _), eval_raw = segment(_features())
    _assert_base_raw(eval_raw, nl=3)


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_rotated_fcos_outputs_are_dicts():
    from ultralytics.nn.modules.head import RotatedFCOS

    head = RotatedFCOS(nc=2, ch=(16, 16, 16, 16, 16))
    feats = _features(ch=(16, 16, 16, 16, 16), sizes=(16, 8, 4, 2, 1))
    raw = head.train()(feats)
    assert {"cls_scores", "bbox_preds", "angle_preds", "centernesses"} <= set(raw)

    pred, raw = head.eval()(_features(ch=(16, 16, 16, 16, 16), sizes=(16, 8, 4, 2, 1)))
    assert {"cls_scores", "bbox_preds", "angle_preds", "centernesses"} <= set(raw)
    assert pred.shape[1] == 4 + head.nc + 1
