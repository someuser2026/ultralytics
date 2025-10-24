# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Smoke tests for Cascade R-CNN detection and segmentation models."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cv2", reason="OpenCV is required for Cascade model smoke tests")

from ultralytics.nn.tasks import CascadeMaskRCNNModel, CascadeRCNNDetectionModel


def _make_detection_batch(img_size: int = 256, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Generate a minimal detection batch compatible with v8 losses."""

    imgs = torch.rand(2, 3, img_size, img_size, device=device)
    cls = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    bboxes = torch.tensor(
        [
            [0.5, 0.5, 0.4, 0.4],
            [0.3, 0.7, 0.2, 0.2],
        ],
        dtype=torch.float32,
        device=device,
    )
    batch_idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    return {"img": imgs, "cls": cls, "bboxes": bboxes, "batch_idx": batch_idx}


def _make_segmentation_batch(
    img_size: int = 256, mask_size: int = 64, device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """Generate a minimal segmentation batch with square instance masks."""

    batch = _make_detection_batch(img_size=img_size, device=device)
    masks = torch.zeros(2, mask_size, mask_size, device=device)
    masks[0, 16:48, 16:48] = 1.0
    masks[1, 8:40, 24:56] = 1.0
    batch["masks"] = masks
    return batch


def test_cascade_rcnn_detection_loss_smoke():
    """Ensure the Cascade R-CNN detection model produces finite training losses."""

    model = CascadeRCNNDetectionModel(
        cfg="ultralytics/cfg/models/cascade/cascade-rcnn.yaml", nc=2, ch=3, verbose=False
    )
    model.train()
    batch = _make_detection_batch()

    loss_vec, loss_items = model.loss(batch)
    assert loss_vec.shape == loss_items.shape == (3,)
    assert torch.isfinite(loss_vec).all()
    assert torch.isfinite(loss_items).all()

    model.zero_grad(set_to_none=True)
    loss_vec.sum().backward()


def test_cascade_mask_rcnn_segmentation_loss_smoke():
    """Ensure the Cascade Mask R-CNN segmentation model produces finite training losses."""

    model = CascadeMaskRCNNModel(
        cfg="ultralytics/cfg/models/cascade/cascade-mask-rcnn.yaml", nc=2, ch=3, verbose=False
    )
    model.train()
    batch = _make_segmentation_batch()

    loss_vec, loss_items = model.loss(batch)
    assert loss_vec.shape == loss_items.shape == (4,)
    assert torch.isfinite(loss_vec).all()
    assert torch.isfinite(loss_items).all()

    model.zero_grad(set_to_none=True)
    loss_vec.sum().backward()
