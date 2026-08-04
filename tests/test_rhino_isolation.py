import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "ultralytics/cfg/models"


def test_only_rhino_yamls_use_rhino_decoder():
    users = []
    for path in MODEL_ROOT.rglob("*.yaml"):
        if "RHINOOBBDecoder" in path.read_text():
            users.append(path)
    assert users
    assert all(path.parent.name == "rhino" for path in users)


def test_dataset_and_trainer_changes_are_rhino_only():
    from ultralytics.data.augment import Format
    from ultralytics.models.rhino.dataset import RHINODataset
    from ultralytics.models.rhino.train import RHINOOBBTrainer
    from ultralytics.models.rhino.val import RHINOOBBValidator
    from ultralytics.models.rtdetr.train import RTDETROBBTrainer
    from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETROBBValidator

    assert "RHINODataset" in inspect.getsource(RHINOOBBTrainer.build_dataset)
    assert "RHINODataset" in inspect.getsource(RHINOOBBValidator.build_dataset)
    assert "RTDETRDataset" in inspect.getsource(RTDETROBBTrainer.build_dataset)
    assert "RTDETRDataset" in inspect.getsource(RTDETROBBValidator.build_dataset)

    hyp = SimpleNamespace(
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        mask_ratio=4,
        overlap_mask=True,
        use_shoreline_prior_loss=False,
        use_land_water_prior_loss=False,
        shoreline_prior_max_dist=128,
    )
    attributes = {
        "augment": False,
        "rect": False,
        "imgsz": 64,
        "data": {},
        "use_segments": False,
        "use_keypoints": False,
        "use_obb": True,
    }
    generic = object.__new__(RTDETRDataset)
    rhino = object.__new__(RHINODataset)
    for dataset in (generic, rhino):
        dataset.__dict__.update(attributes)
    generic_format = next(
        transform for transform in reversed(generic.build_transforms(hyp).transforms) if isinstance(transform, Format)
    )
    rhino_format = next(
        transform for transform in reversed(rhino.build_transforms(hyp).transforms) if isinstance(transform, Format)
    )
    assert generic_format.angle_mode == "oc"
    assert rhino_format.angle_mode == "le90"
    assert Format().angle_mode == "oc"


def test_protected_shared_modules_have_no_rhino_coupling():
    from ultralytics.models.rtdetr.val import RTDETRDataset
    from ultralytics.nn.modules.block import Timm
    from ultralytics.nn.modules.head import RTDETROBBDecoder
    from ultralytics.utils.loss import RTDETROBBLoss

    for protected in (Timm, RTDETRDataset, RTDETROBBDecoder, RTDETROBBLoss):
        assert "RHINO" not in inspect.getsource(protected)


def test_generic_rtdetr_collation_does_not_add_rhino_padding_metadata():
    from ultralytics.models.rtdetr.val import RTDETRDataset

    samples = [
        {
            "img": torch.zeros(3, 8, 8),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]]),
            "cls": torch.tensor([[0.0]]),
            "batch_idx": torch.zeros(1),
        }
        for _ in range(2)
    ]
    batch = RTDETRDataset.collate_fn(samples)
    assert "padding_mask" not in batch
    assert "img_shapes" not in batch


@pytest.mark.parametrize(
    ("config_path", "head_name"),
    [
        ("11/yolo11-obb.yaml", "OBB"),
        ("rt-detr/rtdetr-l-obb.yaml", "RTDETROBBDecoder"),
        ("fcos/rotated_fcos_r50_fpn_le90.yaml", "RotatedFCOS"),
        ("rcnn/oriented_rcnn_r50_fpn_le90.yaml", "OrientedRCNNHead"),
        (
            "timm/obb/final/augfpn/resnet/resnet50/1cls/resnet50-augfpn_512c-obb.yaml",
            "OBB",
        ),
    ],
)
def test_non_rhino_model_head_construction_is_unchanged(config_path, head_name):
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    config = yaml_model_load(MODEL_ROOT / config_path)
    model, *_ = parse_model(deepcopy(config), ch=3, verbose=False)
    assert model[-1].__class__.__name__ == head_name
    assert all(module.__class__.__name__ != "RHINOOBBDecoder" for module in model.modules())


def test_generic_rtdetr_obb_loss_remains_finite():
    from ultralytics.utils.loss import RTDETROBBLoss

    criterion = RTDETROBBLoss(nc=1)
    predicted_boxes = torch.tensor([[[[0.5, 0.5, 0.2, 0.1, 0.1]]]])
    predicted_scores = torch.zeros(1, 1, 1, 1)
    batch = {
        "cls": torch.tensor([0]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.1]]),
        "gt_groups": [1],
    }
    losses = criterion((predicted_boxes, predicted_scores), batch)
    assert all(torch.isfinite(loss) for loss in losses.values())
