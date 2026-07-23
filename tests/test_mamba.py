from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

MAMBA_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "mamba-yolo"
MAMBA_MODELS = tuple(path.name for path in sorted(MAMBA_ROOT.glob("*.yaml")))
MAMBA_TEST_READY = find_spec("cv2") is not None and find_spec("einops") is not None
MAMBA_BUILD_CASES = (
    ("Mamba-YOLO-L-obb-demo.yaml", "obb"),
    ("Mamba-YOLO-L-obb-demo-edgevss.yaml", "obb"),
    ("Mamba-YOLO-T.yaml", "detect"),
    ("legnet-mamba-seg-edgevss-all-shoreaux.yaml", "segment"),
    ("mamba-hrnet-obb.yaml", "obb"),
    ("mamba-hrnet-obb-edgevss.yaml", "obb"),
    ("mamba-hrnet-obb-shoreaux.yaml", "obb"),
    ("mamba-hrnet-seg.yaml", "segment"),
    ("mamba-hrnet-seg-dvss.yaml", "segment"),
    ("mamba-yolo-B-hrnet-seg.yaml", "segment"),
    ("mamba-yolo-B-hrnet-seg-pointrend.yaml", "segment"),
    ("yolo-mamba-seg-edgevss-backbone.yaml", "segment"),
    ("yolo-mamba-seg-edgevss-all.yaml", "segment"),
    ("mamba-hrnet-seg-edgevss.yaml", "segment"),
)


@pytest.mark.skipif(not MAMBA_TEST_READY, reason="cv2 and einops are required to import Mamba-YOLO blocks")
@pytest.mark.parametrize("model_name", MAMBA_MODELS)
def test_mamba_model_yaml_parses(model_name):
    """Ensure Mamba-YOLO model definitions register cleanly with the research-branch parser."""
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    model_cfg = yaml_model_load(MAMBA_ROOT / model_name)
    input_channels = 4 if any(layer[2] == "ChannelSplit" for layer in model_cfg["backbone"]) else 3
    model, save, backbone_layers, head_layers = parse_model(deepcopy(model_cfg), ch=input_channels, verbose=False)

    assert len(model) > 0
    assert isinstance(save, list)
    assert backbone_layers
    assert head_layers


def test_mamba_configs_are_single_class():
    """All Mamba-YOLO research configs in this fork should now be single-class."""
    from ultralytics.nn.tasks import yaml_model_load

    for model_name in MAMBA_MODELS:
        model_cfg = yaml_model_load(MAMBA_ROOT / model_name)
        assert model_cfg["nc"] == 1


@pytest.mark.parametrize(
    "model_name",
    ("mamba-yolo-B-hrnet-seg.yaml", "mamba-yolo-B-hrnet-seg-pointrend.yaml"),
)
def test_mamba_hrnet_b_configs_select_b_scale(model_name):
    """B-specific Mamba-HRNet filenames should select the B compound scale."""
    from ultralytics.nn.tasks import yaml_model_load

    model_cfg = yaml_model_load(MAMBA_ROOT / model_name)

    assert model_cfg["scale"] == "B"
    assert model_cfg["scales"]["B"] == [0.33, 0.50, 1024]


@pytest.mark.skipif(not MAMBA_TEST_READY, reason="cv2 and einops are required to import Mamba-YOLO blocks")
@pytest.mark.parametrize(("model_name", "task"), MAMBA_BUILD_CASES)
def test_mamba_model_construction_uses_build_only_cpu_fallback(model_name, task):
    """Mamba-YOLO models should finish CPU construction and resolve their head strides."""
    from ultralytics import YOLO

    model = YOLO(MAMBA_ROOT / model_name, task=task)

    expected_stride = (
        torch.tensor([4.0, 8.0, 16.0, 32.0])
        if model_name
        in {
            "mamba-hrnet-seg.yaml",
            "mamba-hrnet-seg-dvss.yaml",
            "mamba-hrnet-seg-edgevss.yaml",
            "mamba-yolo-B-hrnet-seg.yaml",
            "mamba-yolo-B-hrnet-seg-pointrend.yaml",
        }
        else torch.tensor([8.0, 16.0, 32.0])
    )
    assert torch.equal(model.model.stride.cpu(), expected_stride)


@pytest.mark.skipif(not MAMBA_TEST_READY, reason="cv2 and einops are required to import Mamba-YOLO blocks")
def test_edge_stem_output_shape():
    """EdgeStem should match the Mamba stem output contract."""
    from ultralytics.nn.modules import EdgeStem

    stem = EdgeStem(3, 128, 3)
    output = stem(torch.randn(1, 3, 640, 640))

    assert output.shape == (1, 128, 160, 160)


@pytest.mark.skipif(not MAMBA_TEST_READY, reason="cv2 and einops are required to import Mamba-YOLO blocks")
def test_mamba_cpu_forward_still_requires_build_context():
    """The CPU fallback should remain limited to stride probing during model construction."""
    from ultralytics import YOLO

    model = YOLO(MAMBA_ROOT / "Mamba-YOLO-T.yaml", task="detect")

    with pytest.raises(RuntimeError, match="CPU fallback is available only during model construction stride probing"):
        model.model(torch.zeros(1, 3, 64, 64))
