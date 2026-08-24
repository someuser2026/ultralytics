from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest

ULTRA_READY = find_spec("cv2") is not None and find_spec("torch") is not None
SHAPELY_READY = find_spec("shapely") is not None
LEGNET_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "legnet"
FCOS_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "fcos"


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_rotated_fcos_r50_yaml_parses_with_reference_fpn():
    """The benchmark timm ResNet-50 RFCOS config should expose canonical P3-P7 features."""
    from ultralytics.nn.modules import FPN, Timm
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    cfg = yaml_model_load(FCOS_ROOT / "rotated_fcos_r50_fpn_le90.yaml")
    backbone_args = cfg["backbone"][0][3]
    assert cfg["backbone"][0][-2] == "Timm"
    assert backbone_args == [
        "resnet50.tv2_in1k",
        True,
        3,
        True,
        [1, 2, 3, 4],
        32,
        None,
        "auto",
        False,
        False,
        False,
        False,
        0.0,
        0.0,
    ]

    # Keep architecture tests independent of network access and local pretrained-weight caches.
    cfg["backbone"][0][3][1] = False
    model, save, backbone_layers, head_layers = parse_model(deepcopy(cfg), ch=3, verbose=False)

    assert save and backbone_layers and head_layers
    assert isinstance(model[0], Timm)
    assert model[0].model_name == "resnet50.tv2_in1k"
    assert model[0].channels == [256, 512, 1024, 2048]
    assert model[-1].__class__.__name__ == "RotatedFCOS"
    assert model[-1].stride.tolist() == [8.0, 16.0, 32.0, 64.0, 128.0]
    assert next(module for module in model if isinstance(module, FPN)).implementation == "reference"


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


def _build_rotated_fcos_criterion(
    model_name: str = "legnet-small-fcos-smallobj.yaml",
    nc: int = 1,
    box: float = 7.5,
    cls: float = 0.5,
    bbox_loss_type: str | None = None,
):
    """Instantiate a RotatedFCOS model and criterion for unit tests."""
    from ultralytics.nn.tasks import OBBModel, yaml_model_load
    from ultralytics.utils.loss import RotatedFCOSLoss

    cfg = yaml_model_load(LEGNET_ROOT / model_name)
    if bbox_loss_type is not None:
        cfg["head"][-1][3][1]["bbox_loss_type"] = bbox_loss_type
    model = OBBModel(cfg, ch=3, nc=nc, verbose=False)
    model.args = SimpleNamespace(box=box, cls=cls, angle_mode=cfg["angle_mode"])
    return model, RotatedFCOSLoss(model)


def _build_obb_model(nc: int = 1, box: float = 7.5, cls: float = 0.5, dfl: float = 1.5, bbox_loss_type: str = "probiou"):
    """Instantiate an OBB model configured with the selected bbox loss type."""
    from ultralytics.nn.tasks import OBBModel, yaml_model_load

    cfg = yaml_model_load(LEGNET_ROOT / "legnet-small-obb.yaml")
    cfg["head"][-1][3][1] = {"ne": 1, "bbox_loss_type": bbox_loss_type}
    model = OBBModel(cfg, ch=3, nc=nc, verbose=False)
    model.args = SimpleNamespace(box=box, cls=cls, dfl=dfl, angle_mode=cfg.get("angle_mode", "oc"))
    return model


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
    return {
        "cls_scores": cls_scores,
        "bbox_preds": bbox_preds,
        "angle_preds": angle_preds,
        "centernesses": centernesses,
    }


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_rotated_fcos_default_bbox_loss_type_is_rotated_iou():
    """RotatedFCOS should default to the MMRotate-style rotated IoU loss path."""
    from ultralytics.nn.modules.head import RotatedFCOS

    head = RotatedFCOS(nc=1, ch=(256, 256, 256, 256, 256))
    assert head.bbox_loss_type == "rotated_iou"


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_kfiou_similarity_matches_theoretical_2d_upper_bound_for_identical_boxes():
    """Raw KFIoU should hit the known 2D upper bound for identical boxes."""
    import torch

    from ultralytics.utils.loss import kfiou_similarity

    boxes = torch.tensor([[32.0, 24.0, 10.0, 6.0, 0.25], [12.0, 40.0, 5.0, 14.0, -0.4]], dtype=torch.float32)
    similarity = kfiou_similarity(boxes, boxes)

    assert torch.allclose(similarity, torch.full_like(similarity, 1.0 / 3.0), atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_kfiou_loss_increases_with_center_offset():
    """Raw KFIoU stays center-invariant while the paper-style loss penalizes larger offsets."""
    import torch

    from ultralytics.utils.loss import kfiou_loss, kfiou_similarity

    base = torch.tensor([[32.0, 32.0, 16.0, 8.0, 0.2]], dtype=torch.float32)
    near = base.clone()
    far = base.clone()
    near[:, 0] += 2.0
    far[:, 0] += 8.0

    assert torch.allclose(kfiou_similarity(base, near), kfiou_similarity(base, far), atol=1e-6, rtol=1e-5)
    assert kfiou_loss(base, far).item() > kfiou_loss(base, near).item()


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
def test_rotated_fcos_bbox_loss_matches_weighted_kfiou():
    """The kfiou path should match the shared paper-style KFIoU helper."""
    import torch

    from ultralytics.utils.loss import kfiou_loss

    _, criterion = _build_rotated_fcos_criterion(box=123.0, cls=0.01, bbox_loss_type="kfiou")
    pred_boxes = torch.tensor([[32.0, 32.0, 10.0, 6.0, 0.20], [48.0, 20.0, 12.0, 8.0, -0.35]], dtype=torch.float32)
    target_boxes = torch.tensor([[31.0, 33.0, 9.5, 6.5, 0.15], [49.0, 18.0, 11.5, 7.5, -0.30]], dtype=torch.float32)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float32)
    center_strides = torch.tensor([[8.0], [16.0]], dtype=torch.float32)

    expected = kfiou_loss(
        pred_boxes,
        target_boxes,
        pred_centers=pred_boxes[:, :2] / center_strides,
        target_centers=target_boxes[:, :2] / center_strides,
    )
    expected = (expected * weights).sum() / weights.sum().clamp_min(1e-6)

    assert torch.allclose(criterion._bbox_loss(pred_boxes, target_boxes, weights, center_strides=center_strides), expected)


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_rotated_fcos_loss_ignores_global_box_and_cls_gains():
    """RotatedFCOSLoss should not use Ultralytics global box/cls gains."""
    import torch

    criterion_model_a, criterion_a = _build_rotated_fcos_criterion(box=1.0, cls=1.0, bbox_loss_type="kfiou")
    criterion_model_b, criterion_b = _build_rotated_fcos_criterion(box=99.0, cls=0.01, bbox_loss_type="kfiou")

    preds_a = _build_manual_fcos_preds(criterion_model_a.model[-1], imgsz=64)
    preds_b = _build_manual_fcos_preds(criterion_model_b.model[-1], imgsz=64)
    batch = _build_single_box_batch(imgsz=64, cx=6.0, cy=6.0, w=8.0, h=8.0)

    total_a, items_a = criterion_a(preds_a, batch)
    total_b, items_b = criterion_b(preds_b, batch)

    assert torch.allclose(total_a, total_b, rtol=1e-6, atol=1e-6)
    assert items_a[:3].tolist() == pytest.approx(items_b[:3].tolist(), rel=1e-6, abs=1e-6)


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_rotated_fcos_cls_and_centerness_use_positive_count(monkeypatch):
    """Classification and centerness losses should divide by the number of positives."""
    import torch

    _, criterion = _build_rotated_fcos_criterion()
    preds = {
        "cls_scores": [torch.zeros(1, criterion.nc, 1, 1) for _ in criterion.stride],
        "bbox_preds": [torch.zeros(1, 4, 1, 1) for _ in criterion.stride],
        "angle_preds": [torch.zeros(1, 1, 1, 1) for _ in criterion.stride],
        "centernesses": [torch.zeros(1, 1, 1, 1) for _ in criterion.stride],
    }
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
@pytest.mark.parametrize("bbox_loss_type", ["probiou", "rotated_iou", "kfiou"])
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
    assert torch.isfinite(total_loss).all()
    assert torch.isfinite(loss_items).all()
    total_loss.sum().backward()

    head = model.model[-1]
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters() if p.requires_grad)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model[4].parameters() if p.requires_grad)

    model.eval()
    preds = model(batch["img"])
    infer = preds[0] if isinstance(preds, tuple) else preds
    assert isinstance(preds[1], dict)
    assert {"cls_scores", "bbox_preds", "angle_preds", "centernesses"} <= set(preds[1])
    assert infer.shape[1] == 4 + model.model[-1].nc + 1


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_obb_head_legacy_loss_type_defaults_to_probiou():
    """Legacy OBB head args should still default to probiou."""
    from ultralytics.nn.tasks import OBBModel, yaml_model_load

    cfg = yaml_model_load(LEGNET_ROOT / "legnet-small-obb.yaml")
    model = OBBModel(cfg, ch=3, nc=1, verbose=False)

    assert model.model[-1].bbox_loss_type == "probiou"
    assert model.model[-1].ne == 1


@pytest.mark.skipif(not ULTRA_READY, reason="cv2 and torch are required")
def test_obb_kfiou_forward_and_backward():
    """OBB should build, run, and backpropagate with KFIoU selected on the head."""
    import torch

    model = _build_obb_model(nc=2, bbox_loss_type="kfiou")
    model.train()
    batch = _build_synthetic_batch(batch_size=2, boxes_per_image=[2, 1])

    total_loss, loss_items = model.loss(batch)

    assert model.model[-1].bbox_loss_type == "kfiou"
    assert torch.isfinite(total_loss).all()
    assert torch.isfinite(loss_items).all()
    total_loss.sum().backward()

    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model[-1].parameters() if p.requires_grad)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.model[0].parameters() if p.requires_grad)
