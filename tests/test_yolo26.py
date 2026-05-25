from __future__ import annotations

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch

YOLO26_READY = find_spec("cv2") is not None and find_spec("torch") is not None


def _args(**overrides):
    values = {
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
        "angle_mode": "oc",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _segment_batch(imgsz: int = 64):
    masks = torch.zeros((1, imgsz, imgsz), dtype=torch.float32)
    masks[0, 20:44, 18:42] = 1.0
    return {
        "img": torch.randn(1, 3, imgsz, imgsz),
        "batch_idx": torch.zeros((1, 1), dtype=torch.float32),
        "cls": torch.zeros((1, 1), dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.375, 0.375]], dtype=torch.float32),
        "cls_probs": torch.ones((1, 1), dtype=torch.float32),
        "masks": masks,
    }


def _obb_batch(imgsz: int = 64):
    return {
        "img": torch.randn(1, 3, imgsz, imgsz),
        "batch_idx": torch.zeros((1, 1), dtype=torch.float32),
        "cls": torch.zeros((1, 1), dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.30, 0.20, 0.0]], dtype=torch.float32),
        "cls_probs": torch.ones((1, 1), dtype=torch.float32),
    }


@pytest.mark.skipif(not YOLO26_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_yolo26_segment_model_builds_and_loss_is_finite():
    from ultralytics.nn.tasks import SegmentationModel, guess_model_task

    model = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=1, verbose=False)
    model.args = _args()

    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    assert model.end2end is True
    assert model.model[-1].reg_max == 1
    assert model.model[-1].__class__.__name__ == "Segment26"
    assert guess_model_task(model) == "segment"

    loss, items = model.loss(_segment_batch())
    assert torch.isfinite(loss).all()
    assert torch.isfinite(items).all()
    loss.sum().backward()


@pytest.mark.skipif(not YOLO26_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_yolo26_obb_model_builds_and_loss_is_finite():
    from ultralytics.nn.tasks import OBBModel, guess_model_task

    model = OBBModel("yolo26n-obb.yaml", ch=3, nc=1, verbose=False)
    model.args = _args()

    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    assert model.end2end is True
    assert model.model[-1].reg_max == 1
    assert model.model[-1].__class__.__name__ == "OBB26"
    assert guess_model_task(model) == "obb"

    loss, items = model.loss(_obb_batch())
    assert torch.isfinite(loss).all()
    assert torch.isfinite(items).all()
    loss.sum().backward()
