from pathlib import Path

import pytest
import torch
from torch import nn

from ultralytics.nn.modules.neck import FPN
from ultralytics.utils import YAML


def test_reference_fpn_matches_manual_unsmoothed_top_down_computation():
    torch.manual_seed(3)
    neck = FPN([2, 3, 4], 2, {"implementation": "reference", "num_outs": 3})
    inputs = [torch.randn(1, 2, 8, 8), torch.randn(1, 3, 4, 4), torch.randn(1, 4, 2, 2)]

    actual = neck(inputs)
    laterals = [conv(x) for conv, x in zip(neck.laterals, inputs)]
    for i in range(len(laterals) - 1, 0, -1):
        laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(
            laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
        )
    expected = [conv(x) for conv, x in zip(neck.smooth, laterals)]

    assert all(torch.allclose(a, e) for a, e in zip(actual, expected))


def test_reference_fpn_uses_learned_bare_convs_for_every_level():
    neck = FPN([8, 16, 32], 8, {"implementation": "reference"})

    assert all(isinstance(module, nn.Conv2d) for module in neck.laterals)
    assert all(isinstance(module, nn.Conv2d) for module in neck.smooth)
    assert neck.laterals[0].weight.shape == (8, 8, 1, 1)
    assert not any(isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.SiLU)) for module in neck.modules())


@pytest.mark.parametrize("source", ["on_input", "on_lateral", "on_output"])
def test_reference_fpn_supports_all_extra_convolution_sources(source):
    neck = FPN(
        [4, 8],
        4,
        {"implementation": "reference", "num_outs": 4, "add_extra_convs": source},
    )
    outputs = neck([torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4)])

    assert [x.shape[-2:] for x in outputs] == [(8, 8), (4, 4), (2, 2), (1, 1)]


def test_reference_fpn_uses_max_pool_for_extra_levels_without_extra_convs():
    neck = FPN([4, 8], 4, {"implementation": "reference", "num_outs": 4})
    outputs = neck([torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4)])

    assert torch.equal(outputs[2], torch.nn.functional.max_pool2d(outputs[1], 1, 2))
    assert torch.equal(outputs[3], torch.nn.functional.max_pool2d(outputs[2], 1, 2))


def test_enhanced_fpn_remains_the_default():
    neck = FPN([4, 8], 4, {"num_outs": 3})
    outputs = neck([torch.randn(1, 4, 8, 8), torch.randn(1, 8, 4, 4)])

    assert neck.implementation == "enhanced"
    assert len(outputs) == 3


def test_all_rcnn_family_yamls_select_reference_fpn():
    root = Path(__file__).parents[1] / "ultralytics" / "cfg" / "models"
    head_names = {"MaskRCNNHead", "CascadeMaskRCNNHead", "RotatedFasterRCNNHead", "OrientedRCNNHead"}
    checked = 0
    for path in root.rglob("*.yaml"):
        text = path.read_text()
        if not any(name in text for name in head_names):
            continue
        cfg = YAML.load(path)
        fpn_layers = [layer for layer in cfg.get("head", []) if layer[2] == "FPN"]
        assert fpn_layers, f"{path} has an RCNN head but no FPN layer"
        assert all(layer[3][1].get("implementation") == "reference" for layer in fpn_layers), path
        checked += 1
    assert checked == 12

