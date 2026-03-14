from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import pytest

RTDETR_VARIANTS = {
    "rtdetr-l-seg.yaml": ("RTDETRSegmentDecoder", "segment"),
    "rtdetr-l-obb.yaml": ("RTDETROBBDecoder", "obb"),
}
RTDETR_ROOT = Path(__file__).resolve().parents[1] / "ultralytics" / "cfg" / "models" / "rt-detr"
RTDETR_TEST_READY = find_spec("cv2") is not None and find_spec("torch") is not None


@pytest.mark.skipif(not RTDETR_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
@pytest.mark.parametrize(("model_name", "expected"), RTDETR_VARIANTS.items())
def test_rtdetr_variant_yaml_parses(model_name, expected):
    """Ensure RT-DETR task variants parse into the expected decoder head."""
    from ultralytics.nn.tasks import parse_model, yaml_model_load

    head_name, _ = expected
    model_cfg = yaml_model_load(RTDETR_ROOT / model_name)
    model, save, backbone_layers, head_layers = parse_model(deepcopy(model_cfg), ch=3, verbose=False)

    assert len(model) > 0
    assert isinstance(save, list)
    assert backbone_layers
    assert head_layers
    assert model[-1].__class__.__name__ == head_name


@pytest.mark.skipif(not RTDETR_TEST_READY, reason="cv2 and torch are required to import Ultralytics models")
@pytest.mark.parametrize(("model_name", "expected"), RTDETR_VARIANTS.items())
def test_rtdetr_variant_task_inference(model_name, expected):
    """Ensure RT-DETR variant YAMLs resolve to the correct task without explicit overrides."""
    from ultralytics import RTDETR, YOLO
    from ultralytics.nn.tasks import guess_model_task, yaml_model_load

    _, task = expected
    model_path = RTDETR_ROOT / model_name
    cfg = yaml_model_load(model_path)

    assert guess_model_task(cfg) == task

    model = RTDETR(str(model_path))
    assert model.task == task
    assert model.model.task == task

    generic_model = YOLO(str(model_path))
    assert generic_model.__class__.__name__ == "RTDETR"
    assert generic_model.task == task
