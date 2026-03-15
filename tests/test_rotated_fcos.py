from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest

ULTRA_READY = find_spec("cv2") is not None and find_spec("torch") is not None
SHAPELY_READY = find_spec("shapely") is not None
LEGNET_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "legnet"


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_fpn_supports_extra_outputs():
    """FPN should preserve the default output count and optionally emit extra levels."""
    import torch

    from ultralytics.nn.modules import FPN

    inputs = [
        torch.randn(1, 128, 80, 80),
        torch.randn(1, 256, 40, 40),
        torch.randn(1, 512, 20, 20),
    ]
    fpn_default = FPN([128, 256, 512], 256, {})
    outs_default = fpn_default(inputs)
    assert len(outs_default) == 3

    fpn_extra = FPN(
        [128, 256, 512],
        256,
        {"num_outs": 5, "add_extra_convs": "on_output", "relu_before_extra_convs": True},
    )
    outs_extra = fpn_extra(inputs)
    assert len(outs_extra) == 5
    assert outs_extra[0].shape == (1, 256, 80, 80)
    assert outs_extra[1].shape == (1, 256, 40, 40)
    assert outs_extra[2].shape == (1, 256, 20, 20)
    assert outs_extra[3].shape == (1, 256, 10, 10)
    assert outs_extra[4].shape == (1, 256, 5, 5)


@pytest.mark.skipif(not SHAPELY_READY or not ULTRA_READY, reason="torch, cv2, and shapely are required")
def test_rotated_box_iou_matches_shapely():
    """Pure-Torch rotated IoU should track a Shapely polygon IoU oracle."""
    import torch
    from shapely.geometry import Polygon

    from ultralytics.utils.loss import rotated_box_iou
    from ultralytics.utils.ops import regularize_rboxes, xywhr2xyxyxyxy

    torch.manual_seed(0)
    boxes1 = torch.rand(64, 5)
    boxes2 = torch.rand(64, 5)
    boxes1[:, :2] *= 256
    boxes2[:, :2] *= 256
    boxes1[:, 2:4] = boxes1[:, 2:4] * 64 + 8
    boxes2[:, 2:4] = boxes2[:, 2:4] * 64 + 8
    boxes1[:, 4] = (torch.rand(64) - 0.5) * torch.pi
    boxes2[:, 4] = (torch.rand(64) - 0.5) * torch.pi
    boxes1 = regularize_rboxes(boxes1, angle_mode="le90")
    boxes2 = regularize_rboxes(boxes2, angle_mode="le90")

    torch_iou = rotated_box_iou(boxes1, boxes2)
    polys1 = xywhr2xyxyxyxy(boxes1).cpu().numpy()
    polys2 = xywhr2xyxyxyxy(boxes2).cpu().numpy()
    oracle = []
    for poly1, poly2 in zip(polys1, polys2):
        p1 = Polygon(poly1)
        p2 = Polygon(poly2)
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        oracle.append(0.0 if union <= 0 else inter / union)
    oracle = torch.tensor(oracle, dtype=torch.float32)

    assert torch.allclose(torch_iou.cpu(), oracle, atol=2e-2, rtol=5e-2)


def _build_synthetic_batch(batch_size: int, boxes_per_image: list[int], imgsz: int = 256):
    """Create a synthetic OBB batch for FCOS smoke tests."""
    import torch

    img = torch.randn(batch_size, 3, imgsz, imgsz)
    batch_idx, cls, bboxes, cls_probs = [], [], [], []
    for image_idx, count in enumerate(boxes_per_image):
        for box_idx in range(count):
            batch_idx.append([image_idx])
            cls.append([float(box_idx % 2)])
            x = 0.2 + 0.15 * (box_idx + 1)
            y = 0.2 + 0.1 * (box_idx + 1)
            w = 0.15 + 0.02 * box_idx
            h = 0.10 + 0.01 * box_idx
            angle = -0.6 + 0.2 * box_idx
            bboxes.append([x, y, w, h, angle])
            cls_probs.append([1.0])
    return {
        "img": img,
        "batch_idx": torch.tensor(batch_idx, dtype=torch.float32) if batch_idx else torch.zeros((0, 1)),
        "cls": torch.tensor(cls, dtype=torch.float32) if cls else torch.zeros((0, 1)),
        "bboxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 5)),
        "cls_probs": torch.tensor(cls_probs, dtype=torch.float32) if cls_probs else torch.zeros((0, 1)),
    }


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
@pytest.mark.parametrize("bbox_loss_type", ["probiou", "rotated_iou"])
@pytest.mark.parametrize("boxes_per_image", [[0, 0], [1, 0], [2, 1]])
def test_rotated_fcos_forward_and_backward(boxes_per_image, bbox_loss_type):
    """Rotated FCOS should build, run, and backpropagate on synthetic OBB batches."""
    import torch

    from ultralytics.nn.tasks import OBBModel, yaml_model_load

    cfg = yaml_model_load(LEGNET_ROOT / "legnet-small-fcos.yaml")
    cfg["bbox_loss_type"] = bbox_loss_type
    head_cfg = cfg["head"][-1][3][1]
    head_cfg["bbox_loss_type"] = bbox_loss_type

    model = OBBModel(cfg, ch=3, nc=2, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, angle_mode=cfg["angle_mode"])
    model.train()
    batch = _build_synthetic_batch(batch_size=2, boxes_per_image=boxes_per_image)
    total_loss, loss_items = model.loss(batch)
    assert total_loss.isfinite()
    assert torch.isfinite(loss_items).all()
    total_loss.backward()

    head = model.model[-1]
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters() if p.requires_grad)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model[4].parameters() if p.requires_grad)

    model.eval()
    preds = model(batch["img"])
    infer = preds[0] if isinstance(preds, tuple) else preds
    assert infer.shape[1] == 4 + model.model[-1].nc + 1
