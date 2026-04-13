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


def _build_single_box_batch(imgsz: int, cx: float, cy: float, w: float, h: float, angle: float = 0.0):
    """Create a single-box normalized OBB batch."""
    import torch

    return {
        "img": torch.zeros(1, 3, imgsz, imgsz),
        "batch_idx": torch.tensor([[0.0]], dtype=torch.float32),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[cx / imgsz, cy / imgsz, w / imgsz, h / imgsz, angle]], dtype=torch.float32),
        "cls_probs": torch.tensor([[1.0]], dtype=torch.float32),
    }


def _build_rotated_fcos_criterion(model_name: str = "legnet-small-fcos-smallobj.yaml", nc: int = 1, box: float = 7.5, cls: float = 0.5):
    """Instantiate a RotatedFCOS model and criterion for unit tests."""
    from ultralytics.nn.tasks import OBBModel, yaml_model_load
    from ultralytics.utils.loss import RotatedFCOSLoss

    cfg = yaml_model_load(LEGNET_ROOT / model_name)
    model = OBBModel(cfg, ch=3, nc=nc, verbose=False)
    model.args = SimpleNamespace(box=box, cls=cls, angle_mode=cfg["angle_mode"])
    return model, RotatedFCOSLoss(model)


def _build_manual_fcos_preds(head, batch_size: int = 1, imgsz: int = 64, bbox_value: float = 1.0):
    """Create deterministic multi-level RotatedFCOS predictions with valid shapes."""
    import torch

    cls_scores, bbox_preds, angle_preds, centernesses = [], [], [], []
    for stride in head.stride.tolist():
        size = max(int(imgsz // stride), 1)
        cls_scores.append(torch.zeros(batch_size, head.nc, size, size))
        bbox_preds.append(torch.full((batch_size, 4, size, size), bbox_value))
        angle_preds.append(torch.zeros(batch_size, 1, size, size))
        centernesses.append(torch.zeros(batch_size, 1, size, size))
    return cls_scores, bbox_preds, angle_preds, centernesses


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_rotated_fcos_default_bbox_loss_type_is_rotated_iou():
    """RotatedFCOS should default to the MMRotate-style rotated IoU loss path."""
    from ultralytics.nn.modules.head import RotatedFCOS

    head = RotatedFCOS(nc=1, ch=(256, 256, 256, 256, 256))
    assert head.bbox_loss_type == "rotated_iou"


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_rotated_fcos_loss_keeps_small_targets(monkeypatch):
    """Targets below 2 pixels should still reach RotatedFCOS assignment."""
    _, criterion = _build_rotated_fcos_criterion()
    preds = _build_manual_fcos_preds(criterion, imgsz=64)
    batch = _build_single_box_batch(imgsz=64, cx=6.0, cy=6.0, w=1.5, h=1.5)

    captured = {}
    original_preprocess = criterion.preprocess

    def wrapped_preprocess(targets, batch_size):
        captured["num_targets"] = int(targets.shape[0])
        return original_preprocess(targets, batch_size)

    monkeypatch.setattr(criterion, "preprocess", wrapped_preprocess)
    criterion(preds, batch)

    assert captured["num_targets"] == 1


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_rotated_fcos_bbox_loss_matches_weighted_log_iou():
    """The rotated_iou path should be a centerness-weighted -log(IoU) reduction."""
    import torch

    from ultralytics.utils.loss import rotated_box_iou

    _, criterion = _build_rotated_fcos_criterion(box=123.0, cls=0.01)
    pred_boxes = torch.tensor([[32.0, 32.0, 10.0, 6.0, 0.20], [48.0, 20.0, 12.0, 8.0, -0.35]], dtype=torch.float32)
    target_boxes = torch.tensor([[31.0, 33.0, 9.5, 6.5, 0.15], [49.0, 18.0, 11.5, 7.5, -0.30]], dtype=torch.float32)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float32)

    expected = (-torch.log(rotated_box_iou(pred_boxes, target_boxes).clamp_min(1e-6)) * weights).sum()
    expected = expected / weights.sum().clamp_min(1e-6)

    assert torch.allclose(criterion._bbox_loss(pred_boxes, target_boxes, weights), expected)


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_rotated_fcos_loss_ignores_global_box_and_cls_gains():
    """RotatedFCOSLoss should not use Ultralytics global box/cls gains."""
    criterion_model_a, criterion_a = _build_rotated_fcos_criterion(box=1.0, cls=1.0)
    criterion_model_b, criterion_b = _build_rotated_fcos_criterion(box=99.0, cls=0.01)

    preds_a = _build_manual_fcos_preds(criterion_model_a.model[-1], imgsz=64)
    preds_b = _build_manual_fcos_preds(criterion_model_b.model[-1], imgsz=64)
    batch = _build_single_box_batch(imgsz=64, cx=6.0, cy=6.0, w=8.0, h=8.0)

    total_a, items_a = criterion_a(preds_a, batch)
    total_b, items_b = criterion_b(preds_b, batch)

    assert total_a.item() == pytest.approx(total_b.item(), rel=1e-6, abs=1e-6)
    assert items_a[:3].tolist() == pytest.approx(items_b[:3].tolist(), rel=1e-6, abs=1e-6)


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_rotated_fcos_cls_and_centerness_use_positive_count(monkeypatch):
    """Classification and centerness losses should divide by the number of positives."""
    import torch

    _, criterion = _build_rotated_fcos_criterion()
    preds = (
        [torch.zeros(1, criterion.nc, 1, 1) for _ in criterion.stride],
        [torch.zeros(1, 4, 1, 1) for _ in criterion.stride],
        [torch.zeros(1, 1, 1, 1) for _ in criterion.stride],
        [torch.zeros(1, 1, 1, 1) for _ in criterion.stride],
    )
    batch = {
        "img": torch.zeros(1, 3, 4, 4),
        "batch_idx": torch.zeros((0, 1)),
        "cls": torch.zeros((0, 1)),
        "bboxes": torch.zeros((0, 5)),
        "cls_probs": torch.zeros((0, 1)),
    }

    monkeypatch.setattr(
        criterion,
        "preprocess",
        lambda targets, batch_size: (
            torch.zeros((1, 0, 1), dtype=torch.long),
            torch.zeros((1, 0, 5)),
            torch.zeros((1, 0, 1)),
            torch.zeros((1, 0, 1), dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        criterion,
        "get_targets",
        lambda *args, **kwargs: (
            torch.tensor([[0, 0, criterion.nc, criterion.nc, criterion.nc]], dtype=torch.long),
            torch.ones((1, 5, 4), dtype=torch.float32),
            torch.zeros((1, 5, 1), dtype=torch.float32),
            torch.ones((1, 5, 1), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(criterion, "_bbox_loss", lambda pred, target, weight: pred.new_tensor(0.0))
    monkeypatch.setattr(criterion, "loss_cls", lambda pred, target: pred.new_tensor(6.0))
    monkeypatch.setattr(criterion, "loss_centerness", lambda pred, target: pred.new_tensor(8.0))

    _, items = criterion(preds, batch)

    assert items[1].item() == pytest.approx(3.0)
    assert items[2].item() == pytest.approx(4.0)


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
