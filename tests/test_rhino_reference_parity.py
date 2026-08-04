from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
R50_CONFIG = ROOT / "ultralytics/cfg/models/rhino/rhino-r50-obb.yaml"


def _small_decoder(**kwargs):
    from ultralytics.nn.modules.rhino import RHINOOBBDecoder

    settings = dict(nc=2, ch=(32, 32, 32), hd=32, nq=12, nh=4, ndl=2, d_ffn=64, nd=16)
    settings.update(kwargs)
    return RHINOOBBDecoder(**settings)


def test_channel_mapper_and_reference_layer_counts():
    decoder = _small_decoder()
    features = [
        torch.randn(1, 32, 16, 16),
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 32, 4, 4),
    ]
    mapped = decoder.channel_mapper(features)

    assert [tuple(feature.shape[1:]) for feature in mapped] == [
        (32, 16, 16),
        (32, 8, 8),
        (32, 4, 4),
        (32, 2, 2),
    ]
    assert len(decoder.encoder.layers) == 6
    assert len(decoder.decoder.layers) == 2
    for projection in [*decoder.channel_mapper.input_convs, decoder.channel_mapper.extra_conv]:
        assert isinstance(projection[1], nn.GroupNorm)
        assert projection[1].num_groups == 32
        assert len(projection) == 2


def test_r50_uses_reference_query_and_denoising_counts():
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    config = yaml_model_load(R50_CONFIG)
    model, *_ = parse_model(config, ch=3, verbose=False)
    head = model[-1]
    assert head.num_queries == 900
    assert head.num_denoising == 100
    assert head.max_num_groups == 30
    assert head.max_candidates == 500
    assert len(head.encoder.layers) == 6
    assert len(head.decoder.layers) == 6


def test_rotated_sampling_converts_normalized_angle_to_pi():
    from ultralytics.nn.modules.rhino import RhinoMSDeformAttn

    attention = RhinoMSDeformAttn(d_model=8, n_levels=1, n_heads=1, n_points=1)
    with torch.no_grad():
        attention.sampling_offsets.weight.zero_()
        attention.sampling_offsets.bias.copy_(torch.tensor([1.0, 0.0]))
    query = torch.zeros(1, 1, 8)
    value = torch.zeros(1, 4, 8)
    reference = torch.tensor([[[[0.5, 0.5, 0.4, 0.2, 0.5]]]])
    attention(query, reference, value, [[2, 2]])
    assert torch.allclose(attention.last_sampling_locations[0, 0, 0, 0, 0], torch.tensor([0.4, 0.5]), atol=1e-5)


def test_look_forward_twice_uses_raw_refinement_and_normalized_predictions():
    from ultralytics.nn.modules.rhino import RhinoTransformerDecoder
    from ultralytics.nn.modules.utils import inverse_sigmoid

    class PassThroughLayer(nn.Module):
        def forward(
            self,
            embed,
            refer_bbox,
            feats,
            shapes,
            padding_mask=None,
            attn_mask=None,
            query_pos=None,
        ):
            return embed

    class ZeroPositionHead(nn.Module):
        def forward(self, value):
            return value.new_zeros((*value.shape[:-1], 4))

    class RecordingBBoxHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 5, bias=False)
            nn.init.zeros_(self.projection.weight)
            self.projection.weight.data[0, 0] = 1.0
            self.inputs = []
            self.outputs = []

        def forward(self, value):
            self.inputs.append(value)
            output = self.projection(value)
            output.retain_grad()
            self.outputs.append(output)
            return output

    decoder = RhinoTransformerDecoder(4, PassThroughLayer(), 2)
    bbox_heads = nn.ModuleList([RecordingBBoxHead(), RecordingBBoxHead()])
    score_heads = nn.ModuleList([nn.Linear(4, 2), nn.Linear(4, 2)])
    query = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
    reference_logits = torch.zeros(1, 1, 5)
    boxes, _ = decoder(
        query,
        reference_logits,
        torch.zeros(1, 1, 4),
        [[1, 1]],
        bbox_heads,
        score_heads,
        ZeroPositionHead(),
    )

    normalized_query = decoder.norm(query)
    initial_reference = reference_logits.sigmoid()
    raw_next_reference = torch.sigmoid(
        bbox_heads[0].outputs[0] + inverse_sigmoid(initial_reference)
    )
    normalized_next_reference = torch.sigmoid(
        bbox_heads[0].outputs[1] + inverse_sigmoid(initial_reference)
    )
    expected_layer_1 = normalized_next_reference
    expected_layer_2 = torch.sigmoid(
        bbox_heads[1].outputs[1] + inverse_sigmoid(raw_next_reference)
    )

    assert all(len(head.inputs) == 2 for head in bbox_heads)
    for head in bbox_heads:
        assert torch.equal(head.inputs[0], query)
        assert torch.equal(head.inputs[1], normalized_query)
    assert torch.allclose(decoder.attention_references[0], initial_reference)
    assert torch.allclose(decoder.attention_references[1], raw_next_reference.detach())
    assert not torch.allclose(decoder.attention_references[1], normalized_next_reference.detach())
    assert torch.allclose(boxes[0], expected_layer_1)
    assert torch.allclose(boxes[1], expected_layer_2)
    assert all(not reference.requires_grad for reference in decoder.attention_references)

    boxes[1].sum().backward()
    raw_refinement_grad = bbox_heads[0].outputs[0].grad
    normalized_prediction_grad = bbox_heads[0].outputs[1].grad
    assert raw_refinement_grad is not None and raw_refinement_grad.abs().sum() > 0
    assert normalized_prediction_grad is None or normalized_prediction_grad.count_nonzero() == 0
    assert bbox_heads[0].projection.weight.grad is not None
    assert bbox_heads[0].projection.weight.grad.abs().sum() > 0


def test_empty_gt_backward_keeps_dn_embedding_in_graph():
    decoder = _small_decoder().train()
    features = [
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 32, 2, 2),
    ]
    batch = {
        "cls": torch.empty(0, dtype=torch.long),
        "bboxes": torch.empty(0, 5),
        "batch_idx": torch.empty(0, dtype=torch.long),
        "gt_groups": [0, 0],
    }
    _, scores, _, _, metadata = decoder(features, batch)
    assert metadata["dn_num_split"] == [0, 12]
    scores.sum().backward()
    assert decoder.denoising_class_embed.weight.grad is not None


def test_phc_rejected_positive_becomes_classification_negative_but_retains_regression():
    from ultralytics.utils.rhino import RHINOOBBLoss

    criterion = RHINOOBBLoss(nc=2)

    class ControlledAssigner:
        @staticmethod
        def assign(**_):
            return torch.tensor([0, -1])

    criterion.dn_assigner = ControlledAssigner()
    dn_boxes = torch.tensor(
        [[[0.2, 0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.1, 0.2], [0.1] * 5, [0.9] * 5]],
        requires_grad=True,
    )
    dn_scores = torch.zeros(1, 4, 2, requires_grad=True)
    matching_boxes = torch.rand(1, 3, 5)
    matching_scores = torch.zeros(1, 3, 2)
    batch = {
        "cls": torch.tensor([0, 1]),
        "bboxes": torch.tensor([[0.2, 0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.2, 0.1, 0.2]]),
        "gt_groups": [2],
        "img_shapes": [(100, 200)],
    }
    losses = criterion._dn_single(
        dn_boxes,
        dn_scores,
        matching_boxes,
        matching_scores,
        batch,
        {"num_denoising_groups": 1, "num_denoising_queries": 4},
    )
    assert criterion.last_dn_targets["labels"].tolist() == [[0, 2, 2, 2]]
    assert criterion.last_dn_targets["original_positive_count"] == 2
    assert criterion.last_dn_targets["new_positive_count"] == 1
    sum(losses).backward()
    assert dn_boxes.grad[0, :2].abs().sum() > 0


def test_l1_is_normalized_while_geometry_uses_pixel_radian_boxes():
    from ultralytics.utils.rhino import RHINOOBBLoss, rhino_boxes_to_physical

    box = torch.tensor([[0.25, 0.5, 0.2, 0.1, 0.5]])
    physical = rhino_boxes_to_physical(box, (100, 400))
    assert torch.allclose(physical, torch.tensor([[100.0, 50.0, 80.0, 10.0, torch.pi / 2]]))

    criterion = RHINOOBBLoss(nc=1)
    criterion.device = torch.device("cpu")
    prediction = box + torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.05]])
    losses = criterion._get_loss_bbox(
        prediction, box, image_shapes=[(100, 400)], image_indices=torch.tensor([0]), normalizer=1
    )
    assert torch.allclose(losses["loss_bbox"], torch.tensor(5.0 * 0.15), atol=1e-6)
    assert torch.isfinite(losses["loss_giou"])


def test_rhino_model_regularizes_targets_in_pixel_space_before_normalizing():
    from ultralytics.nn.tasks import RHINOOBBModel

    captured = {}

    class CaptureCriterion:
        def __call__(self, predictions, batch, **_):
            captured.update(batch)
            return {"loss_bbox": predictions[0].sum() * 0.0}

    model = object.__new__(RHINOOBBModel)
    nn.Module.__init__(model)
    model.criterion = CaptureCriterion()
    model.train()
    predictions = (
        torch.zeros(1, 1, 1, 5),
        torch.zeros(1, 1, 1, 1),
        torch.zeros(1, 1, 5),
        torch.zeros(1, 1, 1),
        None,
    )
    batch = {
        "img": torch.zeros(1, 3, 100, 400),
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor(
            [
                [0.5, 0.5, 0.25, 0.60, 0.0],
                [0.5, 0.5, 0.15, 1.00, -torch.pi / 2],
            ]
        ),
        "img_shapes": torch.tensor([[100, 400]]),
    }

    model.loss(batch, predictions)

    expected = torch.tensor([[0.5, 0.5, 0.25, 0.60, 0.0]]).repeat(2, 1)
    assert torch.allclose(captured["bboxes"], expected, atol=1e-6)
    physical = captured["bboxes"].clone()
    physical[:, [0, 2]] *= 400
    physical[:, [1, 3]] *= 100
    assert torch.allclose(physical[:, 2:4], torch.tensor([[100.0, 60.0], [100.0, 60.0]]), atol=1e-5)


def test_hausdorff_matching_is_normalized_and_resolution_invariant():
    from ultralytics.utils.rhino import DNGroupHungarianAssigner, RHINOHungarianMatcher

    costs = [{"type": "hausdorff", "weight": 1.0, "num_points": 4}]
    matcher = RHINOHungarianMatcher(costs=costs)
    predictions = torch.tensor(
        [[0.4, 0.5, 0.2, 0.1, 0.25], [0.8, 0.5, 0.2, 0.1, 0.25]]
    )
    targets = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.25]])
    scores = torch.zeros(2, 1)
    classes = torch.zeros(1, dtype=torch.long)

    cost_448 = matcher.cost_matrix(predictions, scores, targets, classes, image_shape=(448, 448))
    cost_896 = matcher.cost_matrix(predictions, scores, targets, classes, image_shape=(896, 896))

    assert torch.allclose(cost_448, cost_896, atol=1e-6)
    assert cost_448[0, 0] == pytest.approx(0.1, abs=1e-6)
    assert cost_448[0, 0] != pytest.approx(44.8, abs=1e-3)

    dn_assigner = DNGroupHungarianAssigner(costs=costs)
    dn_targets = torch.tensor(
        [[0.5, 0.5, 0.2, 0.1, 0.25], [0.8, 0.5, 0.2, 0.1, 0.25]]
    )
    dn_scores = torch.zeros(2, 1)
    gt_classes = torch.zeros(2, dtype=torch.long)
    assignment_448 = dn_assigner.assign(
        dn_targets,
        dn_scores,
        dn_targets,
        dn_scores,
        dn_targets,
        gt_classes,
        num_groups=1,
        image_shape=(448, 448),
    )
    assignment_896 = dn_assigner.assign(
        dn_targets,
        dn_scores,
        dn_targets,
        dn_scores,
        dn_targets,
        gt_classes,
        num_groups=1,
        image_shape=(896, 896),
    )
    assert assignment_448.tolist() == assignment_896.tolist() == [0, 1]


def test_optional_rotated_iou_matching_uses_pixel_radian_boxes(monkeypatch):
    import ultralytics.utils.rhino as rhino_utils

    calls = []

    def capture_probiou(gt_bboxes, pred_bboxes):
        calls.append((gt_bboxes.clone(), pred_bboxes.clone()))
        return gt_bboxes.new_ones((len(gt_bboxes), len(pred_bboxes)))

    monkeypatch.setattr(rhino_utils, "batch_probiou", capture_probiou)
    costs = [{"type": "rotated_iou", "weight": 1.0}]
    matcher = rhino_utils.RHINOHungarianMatcher(costs=costs)
    assigner = rhino_utils.DNGroupHungarianAssigner(costs=costs)
    predictions = torch.tensor([[0.4, 0.5, 0.2, 0.1, 0.25]])
    targets = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.25]])
    scores = torch.zeros(1, 1)
    classes = torch.zeros(1, dtype=torch.long)

    matcher.cost_matrix(predictions, scores, targets, classes, image_shape=(100, 400))
    assigner.assign(
        predictions,
        scores,
        predictions,
        scores,
        targets,
        classes,
        num_groups=1,
        image_shape=(100, 400),
    )

    expected_gt = torch.tensor([[200.0, 50.0, 80.0, 10.0, torch.pi / 4]])
    expected_pred = torch.tensor([[160.0, 50.0, 80.0, 10.0, torch.pi / 4]])
    assert len(calls) == 3
    for physical_gt, physical_pred in calls:
        assert torch.allclose(physical_gt, expected_gt, atol=1e-6)
        assert torch.allclose(physical_pred, expected_pred, atol=1e-6)


def test_rhino_collate_pads_spatial_inputs_and_records_valid_shapes():
    from ultralytics.models.rhino.dataset import RHINODataset

    samples = []
    for image_index, (height, width) in enumerate(((8, 12), (4, 6))):
        samples.append(
            {
                "img": torch.full((3, height, width), float(image_index + 1)),
                "land_water_mask": torch.full((1, height, width), 192, dtype=torch.long),
                "shoreline_distance_map": torch.ones(1, height, width),
                "shoreline_proximity_field": torch.ones(1, height, width),
                "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.25]]),
                "cls": torch.tensor([[0.0]]),
                "batch_idx": torch.zeros(1),
            }
        )

    batch = RHINODataset.collate_fn(samples)

    assert batch["img"].shape == (2, 3, 8, 12)
    assert batch["img_shapes"].tolist() == [[8, 12], [4, 6]]
    assert not batch["padding_mask"][0].any()
    assert not batch["padding_mask"][1, :4, :6].any()
    assert batch["padding_mask"][1, 4:, :].all()
    assert batch["padding_mask"][1, :, 6:].all()
    for key in ("img", "land_water_mask", "shoreline_distance_map", "shoreline_proximity_field"):
        assert not batch[key][1, ..., 4:, :].any()
        assert not batch[key][1, ..., :, 6:].any()
    assert torch.allclose(batch["bboxes"], torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.25]]).repeat(2, 1))


def test_padding_masks_drive_feature_valid_ratios_and_proposals():
    decoder = _small_decoder(nd=0).eval()
    features = [
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 32, 2, 2),
    ]
    image_mask = torch.zeros(2, 32, 32, dtype=torch.bool)
    image_mask[1, 16:, :] = True
    image_mask[1, :, 24:] = True

    with torch.no_grad():
        memory, shapes, flattened_mask, valid_ratios = decoder._get_encoder_input(features, image_mask)

    assert shapes == [[8, 8], [4, 4], [2, 2], [1, 1]]
    assert not flattened_mask[0].any()
    assert flattened_mask[1].any()
    expected = torch.tensor(
        [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[0.75, 0.50], [0.75, 0.50], [1.0, 0.50], [1.0, 1.0]],
        ]
    )
    assert torch.allclose(valid_ratios, expected)

    decoder.enc_output = nn.Identity()
    proposal_shapes = [[2, 4], [1, 2], [1, 1], [1, 1]]
    proposal_mask = torch.tensor(
        [[False, False, False, True, True, True, True, True, False, False, False, False]]
    )
    proposal_memory = torch.ones(1, 12, 32)
    transformed, proposal_logits = decoder._generate_encoder_output_proposals(
        proposal_memory, proposal_mask, proposal_shapes
    )
    proposals = proposal_logits.sigmoid()

    assert torch.allclose(proposals[0, 0, :2], torch.tensor([1 / 6, 0.5]), atol=1e-6)
    assert (transformed[0, proposal_mask[0]] == 0).all()
    assert torch.isinf(proposal_logits[0, proposal_mask[0]]).all()


def test_decoder_scales_only_attention_references_by_valid_ratios():
    from ultralytics.nn.modules.rhino import RhinoTransformerDecoder, RhinoTransformerDecoderLayer
    from ultralytics.nn.modules.transformer import MLP

    layer = RhinoTransformerDecoderLayer(8, 1, 16, 0.0, nn.ReLU(), 2, 1)
    decoder = RhinoTransformerDecoder(8, layer, 1)
    bbox_heads = nn.ModuleList([MLP(8, 8, 5, 3)])
    score_heads = nn.ModuleList([nn.Linear(8, 2)])
    position = MLP(16, 8, 8, 2)
    reference_logits = torch.logit(torch.tensor([[[0.4, 0.6, 0.2, 0.3, 0.25]]]))
    valid_ratios = torch.tensor([[[1.0, 1.0], [0.5, 0.75]]])

    boxes, _ = decoder(
        torch.randn(1, 1, 8),
        reference_logits,
        torch.randn(1, 5, 8),
        [[2, 2], [1, 1]],
        bbox_heads,
        score_heads,
        position,
        valid_ratios=valid_ratios,
    )

    attention_input = decoder.attention_reference_inputs[0]
    assert torch.allclose(attention_input[0, 0, 0], torch.tensor([0.4, 0.6, 0.2, 0.3, 0.25]))
    assert torch.allclose(
        attention_input[0, 0, 1],
        torch.tensor([0.2, 0.45, 0.1, 0.225, 0.25]),
    )
    assert torch.allclose(decoder.attention_references[0][0, 0], torch.tensor([0.4, 0.6, 0.2, 0.3, 0.25]))
    assert torch.isfinite(boxes).all()


def test_omitted_mask_matches_explicit_all_false_mask():
    torch.manual_seed(7)
    decoder = _small_decoder(nd=0).eval()
    features = [
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 32, 2, 2),
    ]
    with torch.no_grad():
        implicit = decoder(features)[0]
        explicit = decoder(
            features,
            {"padding_mask": torch.zeros(2, 32, 32, dtype=torch.bool)},
        )[0]
    assert torch.allclose(implicit, explicit, atol=1e-6, rtol=1e-5)


def test_padded_rhino_backward_is_finite_and_reaches_features():
    decoder = _small_decoder(nd=0).train()
    features = [
        torch.randn(2, 32, 8, 8, requires_grad=True),
        torch.randn(2, 32, 4, 4, requires_grad=True),
        torch.randn(2, 32, 2, 2, requires_grad=True),
    ]
    image_mask = torch.zeros(2, 32, 32, dtype=torch.bool)
    image_mask[1, 16:, :] = True
    image_mask[1, :, 24:] = True
    outputs = decoder(features, {"padding_mask": image_mask})
    loss = sum(tensor.float().sum() for tensor in outputs[:4])
    loss.backward()

    assert torch.isfinite(loss)
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features)
    assert sum(feature.grad.abs().sum() for feature in features) > 0


def test_validator_uses_valid_shapes_for_predictions_and_targets():
    from types import SimpleNamespace

    from ultralytics.models.rhino.val import RHINOOBBValidator

    validator = object.__new__(RHINOOBBValidator)
    validator.args = SimpleNamespace(imgsz=448, conf=0.0, classes=None)
    validator.rtdetr_head = SimpleNamespace(max_candidates=1)
    validator._rhino_image_shapes = [(100, 200), (50, 80)]
    predictions = torch.tensor(
        [
            [[0.5, 0.5, 0.2, 0.1, 0.25, 0.9]],
            [[0.5, 0.5, 0.2, 0.1, 0.25, 0.9]],
        ]
    )
    processed = validator.postprocess(predictions)
    assert torch.allclose(processed[0]["bboxes"][0, :4], torch.tensor([100.0, 50.0, 40.0, 10.0]))
    assert torch.allclose(processed[1]["bboxes"][0, :4], torch.tensor([40.0, 25.0, 16.0, 5.0]))

    validator.device = torch.device("cpu")
    batch = {
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.1, 0.25], [0.5, 0.5, 0.2, 0.1, 0.25]]
        ),
        "img_shapes": torch.tensor([[100, 200], [50, 80]]),
        "ori_shape": ((100, 200), (50, 80)),
        "ratio_pad": (((1.0, 1.0), (0.0, 0.0)), ((1.0, 1.0), (0.0, 0.0))),
        "im_file": ("a.jpg", "b.jpg"),
    }
    prepared = validator._prepare_batch(1, batch)
    assert prepared["imgsz"] == (50, 80)
    assert torch.allclose(prepared["bboxes"][0, :4], torch.tensor([40.0, 25.0, 16.0, 5.0]))


def test_rhino_model_unwraps_validation_mask_context(monkeypatch):
    from ultralytics.nn.tasks import RHINOOBBModel, RTDETROBBModel

    captured = {}

    def fake_predict(
        self,
        x,
        profile=False,
        visualize=False,
        batch=None,
        augment=False,
        embed=None,
        metadata_vec=None,
    ):
        captured.update(batch=batch, metadata_vec=metadata_vec)
        return x

    monkeypatch.setattr(RTDETROBBModel, "predict", fake_predict)
    model = object.__new__(RHINOOBBModel)
    image = torch.zeros(2, 3, 8, 8)
    padding_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    image_shapes = torch.tensor([[8, 8], [6, 7]])
    metadata = torch.randn(2, 3)
    context = {
        RHINOOBBModel._VALIDATION_CONTEXT_KEY: True,
        "metadata_vec": metadata,
        "padding_mask": padding_mask,
        "img_shapes": image_shapes,
    }

    assert model.predict(image, metadata_vec=context) is image
    assert captured["metadata_vec"] is metadata
    assert captured["batch"]["padding_mask"] is padding_mask
    assert captured["batch"]["img_shapes"] is image_shapes


def test_multiclass_topk_can_return_multiple_labels_for_one_query():
    from ultralytics.models.rhino.postprocess import rhino_postprocess

    predictions = torch.tensor([[[0.5, 0.5, 0.2, 0.1, 0.25, 0.95, 0.90], [0.2, 0.2, 0.1, 0.1, 0.0, 0.1, 0.2]]])
    result = rhino_postprocess(predictions, [(100, 200)], conf=0.5, max_candidates=4)[0]
    assert result["cls"].tolist() == [0, 1]
    assert torch.allclose(result["bboxes"][0], result["bboxes"][1])


def test_unsupported_rhino_version_fails_explicitly():
    decoder = _small_decoder()
    with pytest.raises(ValueError, match="only 'v2'"):
        decoder.configure_rhino({"version": "v4"})
