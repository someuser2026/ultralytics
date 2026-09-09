from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "ultralytics" / "cfg" / "models"
BENCHMARK_SUBMITTER = ROOT / "jobs" / "train" / "hpc" / "bash_scripts_joint" / "submit_benchmark_models.sh"


def _load(path: Path) -> dict:
    """Load a model YAML for architecture-level comparisons."""
    return yaml.safe_load(path.read_text())


def test_benchmark_yolox_configs_keep_original_architectures() -> None:
    """The one-class x configs must differ from stock YOLO only in configuration metadata."""
    cases = (
        (MODEL_ROOT / "11/yolo11-obb.yaml", MODEL_ROOT / "11/yolo11x-obb-1cls.yaml"),
        (
            MODEL_ROOT / "11/yolo11-seg.yaml",
            MODEL_ROOT / "timm/segment/final/yolo_neck/yolo/yolo11x/flat_no_p2/1cls/yolo11x-yolo11x-segment.yaml",
        ),
        (MODEL_ROOT / "12/yolo12-obb.yaml", MODEL_ROOT / "12/yolo12x-obb-1cls.yaml"),
        (
            MODEL_ROOT / "12/yolo12-seg.yaml",
            MODEL_ROOT / "timm/segment/final/yolo_neck/yolo/yolo12x/1cls/yolo12x-yolo12x-segment.yaml",
        ),
    )

    for stock_path, benchmark_path in cases:
        stock = _load(stock_path)
        benchmark = _load(benchmark_path)
        assert benchmark["nc"] == 1
        assert benchmark["scales"] == {"x": stock["scales"]["x"]}
        assert benchmark["backbone"] == stock["backbone"]
        assert benchmark["head"] == stock["head"]


def test_benchmark_mamba_yolo_models_share_official_l_architecture() -> None:
    """Mamba-YOLO benchmark variants must share the L backbone and neck, excluding the terminal task head."""
    reference = _load(MODEL_ROOT / "mamba-yolo/Mamba-YOLO-L.yaml")
    variants = {
        "OBB": _load(MODEL_ROOT / "mamba-yolo/Mamba-YOLO-L-obb-demo.yaml"),
        "Segment": _load(MODEL_ROOT / "mamba-yolo/Mamba-YOLO-L-seg.yaml"),
    }

    for terminal, variant in variants.items():
        assert variant["nc"] == 1
        assert variant["scales"] == {"L": reference["scales"]["L"]}
        assert variant["backbone"] == reference["backbone"]
        assert variant["head"][:-1] == reference["head"][:-1]
        assert variant["head"][-1][2] == terminal


def test_benchmark_resnet_rcnn_models_use_trainable_pretrained_timm_backbones() -> None:
    """Benchmark ResNet RCNN variants should use pretrained timm backbones without configuration-level freezing."""
    configs = (
        MODEL_ROOT / "rcnn/rotated_faster_rcnn_r50_fpn_le90_smallobj.yaml",
        MODEL_ROOT / "rcnn/cascade_mask_rcnn_r50_fpn_smallobj.yaml",
        MODEL_ROOT / "rcnn/mask_rcnn_r50_fpn_smallobj.yaml",
    )

    for config_path in configs:
        backbone_layer = _load(config_path)["backbone"][0]
        assert backbone_layer[:3] == [-1, 1, "Timm"]
        args = backbone_layer[3]
        assert args[0] == "resnet50.tv2_in1k"
        assert args[1] is True  # pretrained
        assert args[8] is False  # freeze_stem
        assert args[9] is False  # freeze


def test_benchmark_submitter_uses_faithful_yolo_and_mamba_configs() -> None:
    """Both dataset launch paths share this submitter, so keep its canonical config references explicit."""
    submitter = BENCHMARK_SUBMITTER.read_text()
    expected = (
        "ultralytics/cfg/models/11/yolo11x-obb-1cls.yaml",
        "ultralytics/cfg/models/12/yolo12x-obb-1cls.yaml",
        "ultralytics/cfg/models/26/yolo26-obb.yaml",
        "ultralytics/cfg/models/26/yolo26-seg.yaml",
        "ultralytics/cfg/models/timm/segment/final/yolo_neck/yolo/yolo11x/flat_no_p2/1cls/"
        "yolo11x-yolo11x-segment.yaml",
        "ultralytics/cfg/models/timm/segment/final/yolo_neck/yolo/yolo12x/1cls/yolo12x-yolo12x-segment.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-seg.yaml",
    )
    replaced = (
        "yolo11x-augfpn_512c-obb.yaml",
        "yolo12x-augfpn_512c-obb.yaml",
        "ultralytics/cfg/models/26/yolo26-seg-pointrend.yaml",
        "ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml",
    )

    assert all(path in submitter for path in expected)
    assert all(path not in submitter for path in replaced)


def test_benchmark_dino_and_mamba_hr_task_pairs_share_necks() -> None:
    """Each OBB/segment pair must have the same feature-building neck before its task-specific head."""
    cases = (
        (
            MODEL_ROOT
            / "timm/obb/final/yolo_neck/transformer/dinov3_7_12_17_22/1cls/"
            "dinov3_7_12_17_22-yolo11x-obb.yaml",
            MODEL_ROOT
            / "timm/segment/final/yolo_neck/transformer/dinov3_7_12_17_22/1cls/"
            "dinov3_7_12_17_22-yolo11x-segment.yaml",
            [25, 28, 31],
        ),
        (
            MODEL_ROOT / "mamba-yolo/mamba-hrnet-obb.yaml",
            MODEL_ROOT / "mamba-yolo/mamba-hrnet-seg.yaml",
            [70, 73, 76, 79],
        ),
    )

    for obb_path, segment_path, expected_levels in cases:
        obb = _load(obb_path)
        segment = _load(segment_path)
        assert obb["backbone"] == segment["backbone"]
        assert obb["head"][:-1] == segment["head"][:-1]
        assert obb["head"][-1][0] == segment["head"][-1][0] == expected_levels
        assert obb["head"][-1][2] == "OBB"
        assert segment["head"][-1][2] == "Segment"
