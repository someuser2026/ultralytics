from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import pytest

LEGNET_MODELS = {
    "legnet-small.yaml": ("Detect", 3),
    "legnet-small-seg.yaml": ("Segment", 3),
    "legnet-small-obb.yaml": ("OBB", 3),
    "legnet-small-fcos.yaml": ("RotatedFCOS", 5),
    "legnet-small-fcos-smallobj.yaml": ("RotatedFCOS", 5),
}
LEGNET_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "legnet"
LEGNET_TEST_READY = find_spec("cv2") is not None and find_spec("torch") is not None


@pytest.mark.skipif(not LEGNET_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
@pytest.mark.parametrize(("model_name", "expected"), LEGNET_MODELS.items())
def test_legnet_model_yaml_parses(model_name, expected):
    """Ensure LEGNet model definitions register cleanly and expose three task features."""
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    head_name, num_levels = expected
    model_cfg = yaml_model_load(LEGNET_ROOT / model_name)
    model, save, backbone_layers, head_layers = parse_model(deepcopy(model_cfg), ch=3, verbose=False)

    assert len(model) > 0
    assert isinstance(save, list)
    assert backbone_layers
    assert head_layers
    assert model[-1].__class__.__name__ == head_name
    assert model[-1].nl == num_levels


@pytest.mark.skipif(not LEGNET_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
def test_legnet_backbone_output_shapes():
    """Validate the LEGNet-small backbone returns the expected P2-P5 pyramid."""
    import torch

    from ultralytics.nn.modules import LWEGNet

    backbone = LWEGNet()
    outputs = backbone(torch.randn(1, 3, 640, 640))

    assert backbone.channels == [64, 128, 256, 512]
    assert len(outputs) == 4
    assert outputs[0].shape == (1, 64, 160, 160)
    assert outputs[1].shape == (1, 128, 80, 80)
    assert outputs[2].shape == (1, 256, 40, 40)
    assert outputs[3].shape == (1, 512, 20, 20)
