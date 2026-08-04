from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

RHINO_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "rhino"
RHINO_VARIANTS = {
    "rhino-r50-obb.yaml": ("RHINOOBBDecoder", "obb"),
    "rhino-dinov3-augfpn-obb.yaml": ("RHINOOBBDecoder", "obb"),
}
RHINO_TEST_READY = find_spec("cv2") is not None and find_spec("torch") is not None
TIMM_READY = find_spec("timm") is not None


def _skip_if_variant_dependency_missing(model_name: str):
    if "dinov3" in model_name and not TIMM_READY:
        pytest.skip("timm is required to parse the timm RHINO compatibility YAML")


@pytest.mark.skipif(not RHINO_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
@pytest.mark.parametrize(("model_name", "expected"), RHINO_VARIANTS.items())
def test_rhino_variant_yaml_parses(model_name, expected):
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    _skip_if_variant_dependency_missing(model_name)
    head_name, _ = expected
    model_cfg = yaml_model_load(RHINO_ROOT / model_name)
    model, save, backbone_layers, head_layers = parse_model(deepcopy(model_cfg), ch=3, verbose=False)

    assert len(model) > 0
    assert isinstance(save, list)
    assert backbone_layers
    assert head_layers
    assert model[-1].__class__.__name__ == head_name


@pytest.mark.skipif(not RHINO_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_rhino_variant_task_inference():
    from ultralytics import RHINO, YOLO
    from ultralytics.nn.tasks import guess_model_task, yaml_model_load

    model_path = RHINO_ROOT / "rhino-r50-obb.yaml"
    cfg = yaml_model_load(model_path)

    assert guess_model_task(cfg) == "obb"

    model = RHINO(str(model_path))
    assert model.task == "obb"
    assert model.model.task == "obb"

    generic_model = YOLO(str(model_path))
    assert generic_model.__class__.__name__ == "RHINO"
    assert generic_model.task == "obb"


@pytest.mark.skipif(not RHINO_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
@pytest.mark.parametrize("num_levels", [3, 4])
def test_rhino_decoder_forward_and_loss_smoke(num_levels):
    from ultralytics.nn.modules import RHINOOBBDecoder
    from ultralytics.utils.rhino import RHINOOBBLoss

    torch.manual_seed(0)
    decoder = RHINOOBBDecoder(
        nc=2,
        ch=tuple([32] * num_levels),
        hd=32,
        nq=12,
        ndp=4,
        nh=4,
        ndl=2,
        d_ffn=64,
        nd=16,
    )
    decoder.configure_rhino(
        {
            "version": "v2",
            "num_queries": 12,
            "num_denoising_queries": 16,
            "dn_group_mode": "dynamic",
            "max_num_groups": 4,
        }
    )
    decoder.train()

    spatial_sizes = [(16, 16), (8, 8), (4, 4), (2, 2)][:num_levels]
    feats = [torch.randn(2, 32, h, w) for h, w in spatial_sizes]
    batch = {
        "cls": torch.tensor([0, 1, 1], dtype=torch.long),
        "bboxes": torch.tensor(
            [
                [0.25, 0.25, 0.20, 0.10, 0.00],
                [0.70, 0.60, 0.15, 0.12, 0.20],
                [0.52, 0.40, 0.10, 0.08, 0.90],
            ],
            dtype=torch.float32,
        ),
        "batch_idx": torch.tensor([0, 0, 1], dtype=torch.long),
        "gt_groups": [2, 1],
        "img_shapes": [(128, 128), (128, 128)],
    }

    dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = decoder(feats, batch=batch)
    assert dec_bboxes.shape[-1] == 5
    assert enc_bboxes.shape[-1] == 5
    assert dn_meta is not None
    assert dn_meta["dn_num_split"][0] == dn_meta["num_denoising_queries"]

    dn_bboxes, matching_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
    dn_scores, matching_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

    criterion = RHINOOBBLoss(nc=2)
    losses = criterion(
        (torch.cat([enc_bboxes.unsqueeze(0), matching_bboxes], dim=0), torch.cat([enc_scores.unsqueeze(0), matching_scores], dim=0)),
        batch,
        dn_bboxes=dn_bboxes,
        dn_scores=dn_scores,
        dn_meta=dn_meta,
    )
    total_loss = sum(losses.values())
    assert torch.isfinite(total_loss)
    total_loss.backward()


@pytest.mark.skipif(not RHINO_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_rhino_hausdorff_cost_stable():
    from ultralytics.utils.rhino import hausdorff_pairwise_cost

    boxes1 = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.10, 0.00],
            [0.30, 0.30, 0.15, 0.12, 0.10],
        ],
        dtype=torch.float32,
    )
    boxes2 = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.10, 0.00],
            [0.75, 0.70, 0.10, 0.08, -0.15],
        ],
        dtype=torch.float32,
    )

    cost = hausdorff_pairwise_cost(boxes1, boxes2, num_points=8)
    assert cost.shape == (2, 2)
    assert torch.isfinite(cost).all()
    assert cost[0, 0] <= cost[0, 1]


@pytest.mark.skipif(not RHINO_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_rhino_probiou_loss_stable_for_small_boxes():
    from ultralytics.utils.rhino import RHINOOBBLoss

    criterion = RHINOOBBLoss(nc=1, loss_types={"bbox": "l1", "giou": "probiou"})
    criterion.device = torch.device("cpu")

    pred_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.00, 0.00, 0.00],
            [0.50, 0.50, 1e-9, 1e-9, 0.00],
            [0.50, 0.50, 0.20, 0.30, 0.10],
        ],
        dtype=torch.float32,
    )
    gt_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.20, 0.00],
            [0.50, 0.50, 0.20, 0.20, 0.00],
            [0.50, 0.50, 0.20, 0.30, 0.10],
        ],
        dtype=torch.float32,
    )

    losses = criterion._get_loss_bbox(pred_boxes, gt_boxes)
    assert torch.isfinite(losses["loss_bbox"])
    assert torch.isfinite(losses["loss_giou"])


@pytest.mark.skipif(not RHINO_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_rhino_dn_group_assigner_expected_matches():
    from ultralytics.utils.rhino import DNGroupHungarianAssigner

    assigner = DNGroupHungarianAssigner()
    gt_boxes = torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.10, 0.00],
            [0.75, 0.75, 0.15, 0.12, 0.20],
        ],
        dtype=torch.float32,
    )
    gt_cls = torch.tensor([0, 1], dtype=torch.long)

    pred_boxes = gt_boxes + torch.tensor([[0.08, 0.00, 0.00, 0.00, 0.00], [-0.08, 0.00, 0.00, 0.00, 0.00]])
    pred_scores = torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32)
    dn_boxes = gt_boxes.repeat(2, 1)
    dn_scores = torch.tensor([[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0], [-8.0, 8.0]], dtype=torch.float32)

    assigned = assigner.assign(pred_boxes, pred_scores, dn_boxes, dn_scores, gt_boxes, gt_cls, num_groups=2)
    assert assigned.tolist() == [0, 1, 0, 1]
