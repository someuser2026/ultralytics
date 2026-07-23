from importlib.util import find_spec
import types

import pytest

TORCH_READY = find_spec("torch") is not None


if TORCH_READY:
    import torch


def _mask2former_cfg(**overrides):
    cfg = {
        "feature_strides": [4, 8, 16, 32],
        "transformer_in_features": [1, 2, 3],
        "common_stride": 4,
        "conv_dim": 32,
        "mask_dim": 32,
        "hidden_dim": 32,
        "num_queries": 6,
        "nheads": 4,
        "dec_layers": 3,
        "enc_layers": 1,
        "dim_feedforward": 64,
        "encoder_dim_feedforward": 64,
        "train_num_points": 32,
    }
    cfg.update(overrides)
    return cfg


def _features(batch=2):
    return [
        torch.randn(batch, 8, 16, 16),
        torch.randn(batch, 16, 8, 8),
        torch.randn(batch, 32, 4, 4),
        torch.randn(batch, 64, 2, 2),
    ]


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_mask2former_head_train_and_eval_contracts():
    from ultralytics.nn.modules import Mask2FormerHead

    head = Mask2FormerHead(2, _mask2former_cfg(), ch=[8, 16, 32, 64])

    raw = head.train()(_features())
    assert {"pred_logits", "pred_masks", "aux_outputs", "feats"} <= set(raw)
    assert raw["pred_logits"].shape == (2, 6, 3)
    assert raw["pred_masks"].shape == (2, 6, 16, 16)
    assert len(raw["aux_outputs"]) == 2

    eval_raw = head.eval()(_features())
    assert {"pred_logits", "pred_masks", "feats"} <= set(eval_raw)
    assert eval_raw["pred_logits"].shape == (2, 6, 3)

    head.export = True
    logits, masks = head(_features())
    assert logits.shape == (2, 6, 3)
    assert masks.shape == (2, 6, 16, 16)


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_mask2former_loss_is_finite_and_backpropagates():
    from ultralytics.nn.modules import Mask2FormerHead
    from ultralytics.utils.loss import Mask2FormerInstanceLoss

    torch.manual_seed(0)
    head = Mask2FormerHead(1, _mask2former_cfg(), ch=[8, 16, 32, 64])
    model = types.SimpleNamespace(
        model=[head],
        yaml={"nc": 1},
        args=types.SimpleNamespace(overlap_mask=False),
    )
    loss_fn = Mask2FormerInstanceLoss(model)

    masks = torch.zeros(3, 16, 16)
    masks[0, 2:7, 2:7] = 1
    masks[1, 8:13, 9:14] = 1
    masks[2, 4:10, 10:15] = 1
    batch = {
        "img": torch.randn(2, 3, 64, 64),
        "cls": torch.zeros(3, 1),
        "batch_idx": torch.tensor([0, 0, 1.0]),
        "masks": masks,
    }

    loss, items = loss_fn(head.train()(_features()), batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(items).all()

    loss.backward()
    assert head.predictor.class_embed.weight.grad.abs().sum() > 0
    assert head.pixel_decoder.mask_features.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_mask2former_label_loss_moves_empty_weight_to_logits_device(monkeypatch):
    from ultralytics.utils.loss import Mask2FormerHungarianMatcher, Mask2FormerSetCriterion
    import ultralytics.utils.loss as loss_module

    criterion = Mask2FormerSetCriterion(
        1,
        matcher=Mask2FormerHungarianMatcher(cost_class=1.0, cost_mask=1.0, cost_dice=1.0, num_points=8),
        eos_coef=0.1,
        num_points=8,
        oversample_ratio=2.0,
        importance_sample_ratio=0.5,
    )
    outputs = {"pred_logits": torch.randn(1, 3, 2)}
    targets = [{"labels": torch.zeros(1, dtype=torch.long)}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    def cross_entropy(input, target, weight):
        assert weight.device == input.device
        return input.sum() * 0.0

    monkeypatch.setattr(loss_module.F, "cross_entropy", cross_entropy)
    loss = criterion.loss_labels(outputs, targets, indices)["loss_ce"]
    assert torch.isfinite(loss)


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_mask2former_builds_from_segmentation_yaml_style_config():
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.loss import Mask2FormerInstanceLoss

    cfg = {
        "nc": 1,
        "backbone": [
            [-1, 1, "Conv", [8, 3, 2]],
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
        ],
        "head": [
            [
                [0, 1, 2, 3],
                1,
                "Mask2FormerHead",
                [
                    1,
                    _mask2former_cfg(
                        feature_strides=[2, 4, 8, 16],
                        common_stride=2,
                    ),
                ],
            ]
        ],
    }

    model = SegmentationModel(cfg, ch=3, nc=1, verbose=False)
    head = model.model[-1]
    assert type(head).__name__ == "Mask2FormerHead"
    assert head.stride.tolist() == [2.0, 4.0, 8.0, 16.0]
    assert isinstance(model.init_criterion(), Mask2FormerInstanceLoss)


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_mask2former_reference_instance_postprocess_matches_expected_scores_and_boxes():
    from ultralytics.models.mask2former.postprocess import (
        finalize_mask2former_instances,
        select_mask2former_instances,
    )

    logits = torch.tensor([[[4.0, 0.0, -3.0], [3.0, 2.0, -2.0]]])
    masks = torch.full((1, 2, 4, 4), -2.0)
    masks[0, 0, 1:3, 1:3] = 2.0
    selection = select_mask2former_instances(logits, masks, num_classes=2, max_per_image=2)[0]
    result = finalize_mask2former_instances(selection, selection["mask_logits"])

    assert result["bboxes"].tolist() == [[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0]]
    assert result["masks"].shape == (2, 4, 4)
    assert result["conf"][0] > 0
    assert result["conf"][1] == 0


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_mask2former_reference_selection_keeps_overlapping_instances_without_nms():
    from ultralytics.models.mask2former.postprocess import select_mask2former_instances

    logits = torch.tensor([[[6.0, -3.0], [5.0, -3.0]]])
    masks = torch.ones(1, 2, 4, 4)
    selection = select_mask2former_instances(logits, masks, num_classes=1, max_per_image=2)[0]

    assert selection["query_indices"].numel() == 2
    assert set(selection["query_indices"].tolist()) == {0, 1}


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_mask2former_predictor_returns_standard_results_without_generic_nms():
    import numpy as np
    from types import SimpleNamespace

    from ultralytics.models.mask2former.predict import Mask2FormerPredictor

    predictor = Mask2FormerPredictor.__new__(Mask2FormerPredictor)
    predictor.args = SimpleNamespace(max_det=2)
    predictor.model = SimpleNamespace(names={0: "foreground"})
    predictor.batch = (["image.png"],)
    raw = {
        "pred_logits": torch.tensor([[[6.0, -3.0], [5.0, -3.0]]]),
        "pred_masks": torch.ones(1, 2, 4, 4),
    }
    results = predictor.postprocess(raw, torch.zeros(1, 3, 8, 8), [np.zeros((8, 8, 3), dtype=np.uint8)])

    assert len(results) == 1
    assert results[0].boxes.data.shape == (2, 6)
    assert results[0].masks.data.shape == (2, 8, 8)


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_mask2former_validator_returns_standard_segmentation_dictionary():
    from ultralytics.models.mask2former.val import Mask2FormerValidator

    validator = Mask2FormerValidator.__new__(Mask2FormerValidator)
    validator.args = types.SimpleNamespace(max_det=2)
    validator.nc = 1
    validator._last_imgsz = (8, 8)
    validator.segment_head = types.SimpleNamespace(
        max_per_image=2,
        mask_threshold=0.5,
        point_rend_enabled=False,
    )
    raw = {
        "pred_logits": torch.tensor([[[6.0, -3.0], [5.0, -3.0]]]),
        "pred_masks": torch.ones(1, 2, 4, 4),
    }

    predictions = validator.postprocess(raw)

    assert len(predictions) == 1
    assert set(predictions[0]) == {"bboxes", "conf", "cls", "masks"}
    assert predictions[0]["bboxes"].shape == (2, 4)
    assert predictions[0]["masks"].shape == (2, 8, 8)


@pytest.mark.skipif(not TORCH_READY, reason="torch is required")
def test_yolo_routes_mask2former_head_to_dedicated_family(tmp_path):
    import numpy as np

    from ultralytics import YOLO
    from ultralytics.models.mask2former import Mask2Former
    from ultralytics.utils import YAML

    cfg = {
        "nc": 1,
        "backbone": [
            [-1, 1, "Conv", [8, 3, 2]],
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
        ],
        "head": [[[0, 1, 2, 3], 1, "Mask2FormerHead", [1, _mask2former_cfg(feature_strides=[2, 4, 8, 16], common_stride=2)]]],
    }
    model_yaml = tmp_path / "tiny-mask2former.yaml"
    YAML.save(model_yaml, cfg)
    model = YOLO(model_yaml, task="segment", verbose=False)

    assert isinstance(model, Mask2Former)
    results = model.predict(np.zeros((32, 32, 3), dtype=np.uint8), imgsz=32, max_det=2, verbose=False)
    assert len(results) == 1
    assert results[0].boxes.data.shape == (2, 6)
    assert results[0].masks.data.shape == (2, 32, 32)
