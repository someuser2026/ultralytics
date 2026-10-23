"""Load the configured rip-detection policy without importing its parent package."""
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(maxsize=8)
def load_segmentation_filter(path):
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"Configured segmentation filter is missing: {source}")
    spec = importlib.util.spec_from_file_location("site_shared_segmentation_filter", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load segmentation filter: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (not callable(getattr(module, "filter_segmentation_record", None))
            or not callable(getattr(module, "segmentation_polygon_px", None))
            or getattr(module, "MIN_SEGMENT_AREA_PX2", None) != 1.0
            or getattr(module, "SEGMENT_FILTER_VERSION", None) != "segment-pixel-area-v1"):
        raise ValueError(f"Incompatible segmentation filter: {source}")
    return module
