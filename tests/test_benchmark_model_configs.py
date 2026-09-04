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


def test_benchmark_submitter_uses_faithful_yolo_and_mamba_configs() -> None:
    """Both dataset launch paths share this submitter, so keep its canonical config references explicit."""
    submitter = BENCHMARK_SUBMITTER.read_text()
    expected = (
        "ultralytics/cfg/models/11/yolo11x-obb-1cls.yaml",
        "ultralytics/cfg/models/12/yolo12x-obb-1cls.yaml",
        "ultralytics/cfg/models/timm/segment/final/yolo_neck/yolo/yolo11x/flat_no_p2/1cls/"
        "yolo11x-yolo11x-segment.yaml",
        "ultralytics/cfg/models/timm/segment/final/yolo_neck/yolo/yolo12x/1cls/yolo12x-yolo12x-segment.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-seg.yaml",
    )
    replaced = (
        "yolo11x-augfpn_512c-obb.yaml",
        "yolo12x-augfpn_512c-obb.yaml",
        "ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml",
    )

    assert all(path in submitter for path in expected)
    assert all(path not in submitter for path in replaced)
