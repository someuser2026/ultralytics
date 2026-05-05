# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    ROOT,
    SETTINGS_FILE,
    TQDM,
    YAML,
    clean_url,
    colorstr,
    emojis,
    is_dir_writeable,
)
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
from ultralytics.utils.ops import segments2boxes
from ultralytics.utils.patches import read_tiff

HELP_URL = "See https://docs.ultralytics.com/datasets for dataset formatting guidance."
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}  # image suffixes
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
AUX_MASK_SPLITS = ("train", "val", "test", "minival")
BAND_KEY = "bands"
BAND_SCALE_FACTORS_KEY = "band_scale_factors"
METADATA_KEY = "metadata"
RGB_BAND_COUNT = 3
DEFAULT_BAND_SCALE_FACTOR = 255.0
CORE_AUXILIARY_BANDS = frozenset({"shoreline", "land_water", "shoreline_distance", "shoreline_proximity"})
CATEGORICAL_AUXILIARY_BANDS = frozenset({"shoreline", "land_water"})
DEFAULT_METADATA_FIELDS = (
    "anomalous_pixels",
    "clear_confidence_percent",
    "clear_percent",
    "cloud_cover",
    "cloud_percent",
    "ground_control",
    "gsd",
    "heavy_haze_percent",
    "light_haze_percent",
    "pixel_resolution",
    "satellite_azimuth",
    "shadow_percent",
    "snow_ice_percent",
    "sun_azimuth",
    "sun_elevation",
    "view_angle",
    "visible_confidence_percent",
    "udm2_confidence_mean",
    "unusable_pixels_percent",
)
_METADATA_PERCENT_FIELDS = {
    "clear_confidence_percent",
    "clear_percent",
    "cloud_percent",
    "heavy_haze_percent",
    "light_haze_percent",
    "shadow_percent",
    "snow_ice_percent",
    "visible_confidence_percent",
    "udm2_confidence_mean",
    "unusable_pixels_percent",
}
_METADATA_AZIMUTH_FIELDS = {"satellite_azimuth", "sun_azimuth"}
_METADATA_DIV90_FIELDS = {"sun_elevation", "view_angle"}
_METADATA_RAW_FIELDS = {"cloud_cover", "gsd", "pixel_resolution"}


def img2label_paths(img_paths: list[str]) -> list[str]:
    """Convert image paths to label paths by replacing 'images' with 'labels' and extension with '.txt'."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]


def _model_uses_shoreline_aux_loss(model_spec: Any) -> bool:
    """Return True when the selected model YAML uses a shoreline auxiliary YOLO head."""
    if isinstance(model_spec, dict):
        cfg = model_spec
    elif isinstance(model_spec, (str, Path)):
        spec = str(model_spec)
        if not spec.endswith((".yaml", ".yml")):
            return False
        resolved = check_file(spec, suffix=(".yaml", ".yml"), download=False, hard=False)
        yaml_path = Path(resolved or spec)
        if not yaml_path.is_file():
            return False
        cfg = YAML.load(yaml_path)
    else:
        return False

    if bool(cfg.get("use_shoreline_aux_loss", False)):
        return True

    head = cfg.get("head") or []
    if not head:
        return False
    last = head[-1]
    return isinstance(last, (list, tuple)) and len(last) >= 3 and last[2] in {"OBBShoreAux", "SegmentShoreAux"}


def get_auxiliary_mask_flags(hyp: Any = None) -> dict[str, bool]:
    """Return mask-input and prior-loss requirements derived from the current args/config."""
    use_shoreline_input = bool(getattr(hyp, "use_shoreline_input", False))
    use_land_water_input = bool(getattr(hyp, "use_land_water_input", False))
    use_shoreline_prior_loss = bool(getattr(hyp, "use_shoreline_prior_loss", False))
    use_land_water_prior_loss = bool(getattr(hyp, "use_land_water_prior_loss", False))
    model_spec = hyp.get("model") if isinstance(hyp, dict) else getattr(hyp, "model", None)
    use_shoreline_aux_loss = bool(getattr(hyp, "use_shoreline_aux_loss", False)) or _model_uses_shoreline_aux_loss(
        model_spec
    )
    return {
        "use_shoreline_input": use_shoreline_input,
        "use_land_water_input": use_land_water_input,
        "use_shoreline_prior_loss": use_shoreline_prior_loss,
        "use_land_water_prior_loss": use_land_water_prior_loss,
        "use_shoreline_aux_loss": use_shoreline_aux_loss,
        "require_shoreline": use_shoreline_input or use_shoreline_prior_loss or use_shoreline_aux_loss,
        "require_land_water": use_land_water_input or use_land_water_prior_loss or use_shoreline_prior_loss,
        "enabled": any(
            (use_shoreline_input, use_land_water_input, use_shoreline_prior_loss, use_land_water_prior_loss, use_shoreline_aux_loss)
        ),
    }


def normalize_bands_config(data: dict[str, Any]) -> None:
    """Normalize and validate optional multichannel TIFF band metadata from data.yaml."""
    raw_bands = data.get(BAND_KEY)
    if raw_bands is None:
        data[BAND_KEY] = {}
        return
    if not isinstance(raw_bands, dict) or not raw_bands:
        raise SyntaxError("bands must be a non-empty dict when present in data.yaml.")

    normalized = {}
    seen_names = set()
    for raw_idx, raw_name in raw_bands.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise SyntaxError(f"Band key '{raw_idx}' is invalid. bands keys must be integers >= 4.") from exc
        if idx <= RGB_BAND_COUNT:
            raise SyntaxError("bands keys must be >= 4 because bands 1, 2, and 3 are reserved for RGB.")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise SyntaxError(f"Band {idx} must map to a non-empty string name.")
        name = raw_name.strip()
        if name in seen_names:
            raise SyntaxError(f"Duplicate band name '{name}' in bands config.")
        seen_names.add(name)
        normalized[idx] = name

    data[BAND_KEY] = dict(sorted(normalized.items()))


def get_band_name_to_index(bands: dict[int, str] | None) -> dict[str, int]:
    """Return a zero-based image-channel lookup from a normalized 1-based bands config."""
    if not bands:
        return {}
    return {name: idx - 1 for idx, name in bands.items()}


def normalize_band_scale_factors_config(data: dict[str, Any]) -> None:
    """Normalize and validate optional per-band input scale divisors from data.yaml."""
    raw_scale_factors = data.get(BAND_SCALE_FACTORS_KEY)
    if not raw_scale_factors:
        data[BAND_SCALE_FACTORS_KEY] = {}
        return
    if not isinstance(raw_scale_factors, dict):
        raise SyntaxError("band_scale_factors must be a dict when present in data.yaml.")

    channels = int(data.get("channels", RGB_BAND_COUNT))
    normalized = {}
    for raw_idx, raw_scale in raw_scale_factors.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise SyntaxError(
                f"Band scale factor key '{raw_idx}' is invalid. band_scale_factors keys must be integers >= 1."
            ) from exc
        if idx < 1 or idx > channels:
            raise SyntaxError(
                f"band_scale_factors key {idx} is invalid because channels={channels}. "
                "Scale-factor keys must be within [1, channels]."
            )
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError) as exc:
            raise SyntaxError(f"band_scale_factors[{idx}] must be a positive numeric value.") from exc
        if not math.isfinite(scale) or scale <= 0:
            raise SyntaxError(f"band_scale_factors[{idx}] must be a positive finite numeric value.")
        normalized[idx] = scale

    data[BAND_SCALE_FACTORS_KEY] = dict(sorted(normalized.items()))


def get_channel_scale_factors(channels: int, band_scale_factors: dict[int, float] | None = None) -> np.ndarray:
    """Return per-channel input divisors for tensor normalization."""
    scales = np.full(int(channels), DEFAULT_BAND_SCALE_FACTOR, dtype=np.float32)
    for idx, scale in (band_scale_factors or {}).items():
        channel_idx = int(idx) - 1
        if 0 <= channel_idx < len(scales):
            scales[channel_idx] = float(scale)
    return scales


def get_categorical_band_indices(bands: dict[int, str] | None, channels: int | None = None) -> set[int]:
    """Return zero-based indices for embedded TIFF bands that must use nearest-neighbor geometry."""
    max_channels = None if channels is None else int(channels)
    indices = set()
    for idx, name in (bands or {}).items():
        channel_idx = int(idx) - 1
        if name in CATEGORICAL_AUXILIARY_BANDS and (max_channels is None or channel_idx < max_channels):
            indices.add(channel_idx)
    return indices


def _channel_padding_values(channels: int, dtype: np.dtype, padding_value: int | float = 0) -> np.ndarray:
    """Return per-channel padding values, preserving RGB fill and zero-filling non-RGB bands."""
    values = np.zeros(int(channels), dtype=dtype)
    values[: min(3, int(channels))] = padding_value
    return values


def resize_image_with_band_roles(
    img: np.ndarray,
    size: tuple[int, int],
    bands: dict[int, str] | None = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize an image while using nearest-neighbor interpolation for categorical embedded bands."""
    if img.ndim == 2:
        return cv2.resize(img, size, interpolation=interpolation)

    channels = int(img.shape[2])
    nearest = get_categorical_band_indices(bands, channels)
    if channels <= 4 and not nearest:
        resized = cv2.resize(img, size, interpolation=interpolation)
        return resized[..., None] if resized.ndim == 2 else resized

    resized_channels = []
    for channel_idx in range(channels):
        channel_interp = cv2.INTER_NEAREST if channel_idx in nearest else interpolation
        resized_channels.append(cv2.resize(img[..., channel_idx], size, interpolation=channel_interp))
    return np.stack(resized_channels, axis=-1).astype(img.dtype, copy=False)


def warp_image_with_band_roles(
    img: np.ndarray,
    matrix: np.ndarray,
    dsize: tuple[int, int],
    bands: dict[int, str] | None = None,
    perspective: bool = False,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int | float = 0,
) -> np.ndarray:
    """Apply affine/perspective geometry while preserving categorical embedded-band values."""
    if img.ndim == 2:
        warp = cv2.warpPerspective if perspective else cv2.warpAffine
        matrix_arg = matrix if perspective else matrix[:2]
        return warp(img, matrix_arg, dsize=dsize, flags=interpolation, borderValue=border_value)

    channels = int(img.shape[2])
    nearest = get_categorical_band_indices(bands, channels)
    if channels == 3 and not nearest:
        warp = cv2.warpPerspective if perspective else cv2.warpAffine
        matrix_arg = matrix if perspective else matrix[:2]
        warped = warp(
            img,
            matrix_arg,
            dsize=dsize,
            flags=interpolation,
            borderValue=(border_value,) * 3,
        )
        return warped[..., None] if warped.ndim == 2 else warped

    padding_values = _channel_padding_values(channels, img.dtype, padding_value=border_value)
    warped_channels = []
    for channel_idx in range(channels):
        channel_interp = cv2.INTER_NEAREST if channel_idx in nearest else interpolation
        channel_border = padding_values[channel_idx].item()
        if perspective:
            warped = cv2.warpPerspective(
                img[..., channel_idx],
                matrix,
                dsize=dsize,
                flags=channel_interp,
                borderValue=channel_border,
            )
        else:
            warped = cv2.warpAffine(
                img[..., channel_idx],
                matrix[:2],
                dsize=dsize,
                flags=channel_interp,
                borderValue=channel_border,
            )
        warped_channels.append(warped)
    return np.stack(warped_channels, axis=-1).astype(img.dtype, copy=False)


def pad_image_with_band_roles(
    img: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    padding_value: int | float = 0,
) -> np.ndarray:
    """Pad RGB channels with the requested image value and non-RGB channels with zero."""
    if img.ndim == 2:
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)

    h, w, channels = img.shape
    if channels == 3:
        return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(padding_value,) * 3,
        )

    pad_img = np.empty((h + top + bottom, w + left + right, channels), dtype=img.dtype)
    pad_img[...] = _channel_padding_values(channels, img.dtype, padding_value=padding_value).reshape(1, 1, channels)
    pad_img[top : top + h, left : left + w] = img
    return pad_img


def make_image_canvas_with_band_roles(
    shape: tuple[int, int, int],
    dtype: np.dtype,
    padding_value: int | float = 0,
) -> np.ndarray:
    """Allocate an image canvas with RGB fill and zero-filled non-RGB bands."""
    canvas = np.empty(shape, dtype=dtype)
    canvas[...] = _channel_padding_values(shape[2], dtype, padding_value=padding_value).reshape(1, 1, shape[2])
    return canvas


def validate_bands_config(data: dict[str, Any], hyp: Any = None) -> None:
    """Validate multichannel TIFF band metadata against the current training or predict flags."""
    for legacy_key in ("shoreline_masks", "land_water_masks"):
        if legacy_key in data:
            raise SyntaxError(f"'{legacy_key}' is no longer supported. Store auxiliary masks in TIFF bands instead.")
    normalize_bands_config(data)
    bands = data.get(BAND_KEY, {})
    band_names = set(bands.values())
    max_band_index = max(bands, default=RGB_BAND_COUNT)

    raw_channels = data.get("channels")
    if raw_channels is None:
        raw_channels = max(RGB_BAND_COUNT, max_band_index)
    try:
        raw_channels = int(raw_channels)
    except (TypeError, ValueError) as exc:
        raise SyntaxError("channels must be an integer when present in data.yaml.") from exc
    if raw_channels < max_band_index:
        raise SyntaxError(f"channels={raw_channels} is invalid because bands require at least {max_band_index} channels.")
    data["channels"] = raw_channels
    normalize_band_scale_factors_config(data)

    flags = get_auxiliary_mask_flags(hyp)
    if flags["use_shoreline_input"] and "shoreline" not in band_names:
        raise SyntaxError("use_shoreline_input requires bands to define 'shoreline'.")
    if flags["use_land_water_input"] and "land_water" not in band_names:
        raise SyntaxError("use_land_water_input requires bands to define 'land_water'.")
    if flags["use_land_water_prior_loss"] and "land_water" not in band_names:
        raise SyntaxError("use_land_water_prior_loss requires bands to define 'land_water'.")
    if flags["use_shoreline_prior_loss"]:
        if "land_water" not in band_names:
            raise SyntaxError("use_shoreline_prior_loss requires bands to define 'land_water'.")
        if "shoreline_distance" not in band_names and "shoreline" not in band_names:
            raise SyntaxError("use_shoreline_prior_loss requires bands to define 'shoreline_distance' or 'shoreline'.")
    if flags["use_shoreline_aux_loss"] and "shoreline_proximity" not in band_names and "shoreline" not in band_names:
        raise SyntaxError("shoreline auxiliary loss requires bands to define 'shoreline_proximity' or 'shoreline'.")


def _resolve_dataset_path(path: Path, spec: str) -> Path:
    """Resolve a dataset-relative path spec to an absolute path."""
    x = (path / spec).resolve()
    if not x.exists() and spec.startswith("../"):
        x = (path / spec[3:]).resolve()
    return x


def _resolve_split_entry(path: Path, spec: str | list[str]) -> str | list[str]:
    """Resolve a split or auxiliary-root spec to absolute path(s)."""
    if isinstance(spec, str):
        return str(_resolve_dataset_path(path, spec))
    return [str(_resolve_dataset_path(path, s)) for s in spec]


def _normalize_split_roots(spec: str | list[str] | None) -> list[Path]:
    """Return directory-style roots for split matching, tolerating txt/csv list inputs."""
    if spec is None:
        return []
    entries = spec if isinstance(spec, list) else [spec]
    roots = []
    for entry in entries:
        p = Path(entry)
        roots.append((p.parent if p.is_file() else p).resolve())
    return roots


def _append_split_to_aux_root(spec: str | list[str], split: str) -> str | list[str]:
    """Append a dataset split name to a base auxiliary-mask root or list of roots."""
    if isinstance(spec, str):
        return str(Path(spec) / split)
    return [str(Path(entry) / split) for entry in spec]


def _get_auxiliary_split_spec(spec: Any, split: str) -> str | list[str] | None:
    """Return a split-specific auxiliary-mask spec from either a root-folder or per-split config."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        return spec.get(split)
    return _append_split_to_aux_root(spec, split)


def build_metadata_root_mappings(data: dict[str, Any]) -> list[dict[str, Path | str]]:
    """Build longest-prefix image-root to metadata-root mappings from a resolved data dict."""
    metadata_cfg = data.get(METADATA_KEY)
    if not metadata_cfg:
        return []

    mappings = []
    for split in AUX_MASK_SPLITS:
        image_roots = _normalize_split_roots(data.get(split))
        metadata_roots = _normalize_split_roots(_get_auxiliary_split_spec(metadata_cfg, split))
        if metadata_roots and len(metadata_roots) not in {1, len(image_roots)}:
            raise ValueError(f"metadata.{split} must define either 1 root or {len(image_roots)} roots to match {split}.")
        if len(metadata_roots) == 1 and len(image_roots) > 1:
            metadata_roots *= len(image_roots)
        for i, image_root in enumerate(image_roots):
            metadata_root = metadata_roots[i] if i < len(metadata_roots) else None
            if metadata_root is not None:
                mappings.append({"split": split, "image_root": image_root, "metadata_root": metadata_root})
    return sorted(mappings, key=lambda x: len(str(x["image_root"])), reverse=True)


def resolve_metadata_path(
    image_file: str | Path, mappings: list[dict[str, Path | str]], required: bool = False
) -> dict[str, str | None]:
    """Resolve a mirrored JSON metadata sidecar path for an image using longest-prefix root matching."""
    image_path = Path(image_file).resolve()
    for mapping in mappings:
        image_root = Path(mapping["image_root"])
        try:
            rel = image_path.relative_to(image_root)
        except ValueError:
            symlink_candidate = image_root / image_path.name
            try:
                if not symlink_candidate.exists() or symlink_candidate.resolve() != image_path:
                    continue
            except OSError:
                continue
            rel = symlink_candidate.relative_to(image_root)

        metadata_file = str((Path(mapping["metadata_root"]) / rel).with_suffix(".json"))
        out = {"split": mapping["split"], "metadata_file": metadata_file}
        if required and not metadata_file:
            raise FileNotFoundError(f"No metadata root matched image '{image_path}'.")
        return out

    if required:
        raise FileNotFoundError(f"Could not match image '{image_path}' to any configured metadata split root.")
    return {"split": None, "metadata_file": None}


def validate_metadata_config(data: dict[str, Any]) -> None:
    """Validate optional metadata split roots and requested metadata fields."""
    metadata_cfg = data.get(METADATA_KEY)
    metadata_fields = data.get("metadata_fields")
    if metadata_fields and not metadata_cfg:
        raise SyntaxError("metadata_fields is present but 'metadata:' is missing from data.yaml.")
    if not metadata_cfg:
        return
    if metadata_fields is None:
        data["metadata_fields"] = list(DEFAULT_METADATA_FIELDS)
    elif not isinstance(metadata_fields, list) or not metadata_fields:
        raise SyntaxError("metadata_fields must be a non-empty list when metadata sidecars are configured.")
    for split in AUX_MASK_SPLITS:
        if data.get(split) and not metadata_cfg.get(split):
            raise SyntaxError(f"metadata.{split} is required when the dataset defines a '{split}' split.")
    build_metadata_root_mappings(data)


def encode_metadata_properties(properties: dict[str, Any], fields: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
    """Encode selected metadata properties into a fixed float vector for FiLM modulation."""
    fields = tuple(fields or DEFAULT_METADATA_FIELDS)
    encoded = []
    for field in fields:
        if field not in properties:
            raise KeyError(f"Metadata field '{field}' is missing from JSON properties.")
        value = properties[field]
        if value is None:
            raise ValueError(f"Metadata field '{field}' is null and can not be encoded.")
        if field == "ground_control":
            encoded.append(1.0 if bool(value) else 0.0)
        elif field == "anomalous_pixels":
            encoded.append(math.log1p(float(value)))
        elif field in _METADATA_PERCENT_FIELDS:
            encoded.append(float(value) / 100.0)
        elif field in _METADATA_AZIMUTH_FIELDS:
            radians = math.radians(float(value))
            encoded.extend((math.sin(radians), math.cos(radians)))
        elif field in _METADATA_DIV90_FIELDS:
            encoded.append(float(value) / 90.0)
        elif field in _METADATA_RAW_FIELDS:
            encoded.append(float(value))
        else:
            raise ValueError(f"Metadata field '{field}' is not supported by the v1 encoder.")
    return np.asarray(encoded, dtype=np.float32)


def check_file_speeds(
    files: list[str], threshold_ms: float = 10, threshold_mb: float = 50, max_files: int = 5, prefix: str = ""
):
    """
    Check dataset file access speed and provide performance feedback.

    This function tests the access speed of dataset files by measuring ping (stat call) time and read speed.
    It samples up to 5 files from the provided list and warns if access times exceed the threshold.

    Args:
        files (list[str]): List of file paths to check for access speed.
        threshold_ms (float, optional): Threshold in milliseconds for ping time warnings.
        threshold_mb (float, optional): Threshold in megabytes per second for read speed warnings.
        max_files (int, optional): The maximum number of files to check.
        prefix (str, optional): Prefix string to add to log messages.

    Examples:
        >>> from pathlib import Path
        >>> image_files = list(Path("dataset/images").glob("*.jpg"))
        >>> check_file_speeds(image_files, threshold_ms=15)
    """
    if not files:
        LOGGER.warning(f"{prefix}Image speed checks: No files to check")
        return

    # Sample files (max 5)
    files = random.sample(files, min(max_files, len(files)))

    # Test ping (stat time)
    ping_times = []
    file_sizes = []
    read_speeds = []

    for f in files:
        try:
            # Measure ping (stat call)
            start = time.perf_counter()
            file_size = os.stat(f).st_size
            ping_times.append((time.perf_counter() - start) * 1000)  # ms
            file_sizes.append(file_size)

            # Measure read speed
            start = time.perf_counter()
            with open(f, "rb") as file_obj:
                _ = file_obj.read()
            read_time = time.perf_counter() - start
            if read_time > 0:  # Avoid division by zero
                read_speeds.append(file_size / (1 << 20) / read_time)  # MB/s
        except Exception:
            pass

    if not ping_times:
        LOGGER.warning(f"{prefix}Image speed checks: failed to access files")
        return

    # Calculate stats with uncertainties
    avg_ping = np.mean(ping_times)
    std_ping = np.std(ping_times, ddof=1) if len(ping_times) > 1 else 0
    size_msg = f", size: {np.mean(file_sizes) / (1 << 10):.1f} KB"
    ping_msg = f"ping: {avg_ping:.1f}±{std_ping:.1f} ms"

    if read_speeds:
        avg_speed = np.mean(read_speeds)
        std_speed = np.std(read_speeds, ddof=1) if len(read_speeds) > 1 else 0
        speed_msg = f", read: {avg_speed:.1f}±{std_speed:.1f} MB/s"
    else:
        speed_msg = ""

    if avg_ping < threshold_ms or avg_speed < threshold_mb:
        LOGGER.info(f"{prefix}Fast image access ✅ ({ping_msg}{speed_msg}{size_msg})")
    else:
        LOGGER.warning(
            f"{prefix}Slow image access detected ({ping_msg}{speed_msg}{size_msg}). "
            f"Use local storage instead of remote/mounted storage for better performance. "
            f"See https://docs.ultralytics.com/guides/model-training-tips/"
        )


def get_hash(paths: list[str]) -> str:
    """Return a single hash value of a list of paths (files or dirs)."""
    size = 0
    for p in paths:
        try:
            size += os.stat(p).st_size
        except OSError:
            continue
    h = __import__("hashlib").sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img: Image.Image) -> tuple[int, int]:
    """Return exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in {6, 8}:  # rotation 270 or 90
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def _verify_tiff_image_file(im_file: str) -> tuple[tuple[int, int], str]:
    """Return TIFF image shape and format without using PIL verification."""
    im = read_tiff(im_file)
    shape = im.shape[:2]
    image_format = Path(im_file).suffix[1:].lower()
    return shape, image_format


def verify_image(args: tuple) -> tuple:
    """Verify one image."""
    (im_file, cls), prefix = args
    # Number (found, corrupt), message
    nf, nc, msg = 0, 0, ""
    try:
        if str(im_file).lower().endswith((".tif", ".tiff")):
            shape, image_format = _verify_tiff_image_file(im_file)
        else:
            with Image.open(im_file) as im:
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                shape = (shape[1], shape[0])  # hw
                image_format = im.format.lower()
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert image_format in IMG_FORMATS, f"Invalid image format {image_format}. {FORMATS_HELP_MSG}"
        if image_format in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"
        nf = 1
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
    return (im_file, cls), nf, nc, msg


def verify_image_label(args: tuple) -> list:
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    cls_probs = np.zeros((0, 1), dtype=np.float32)
    try:
        # Verify images
        if str(im_file).lower().endswith((".tif", ".tiff")):
            shape, image_format = _verify_tiff_image_file(im_file)
        else:
            with Image.open(im_file) as im:
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                shape = (shape[1], shape[0])  # hw
                image_format = im.format.lower()
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert image_format in IMG_FORMATS, f"invalid image format {image_format}. {FORMATS_HELP_MSG}"
        if image_format in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}{im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
                    classes, probs = [], []
                    for x in lb:
                        row = np.array(x, dtype=np.float32)
                        has_prob = len(x) >= 8 and (len(x) - 2) % 2 == 0  # cls + (xy)*n + prob
                        seg = row[1:-1] if has_prob else row[1:]
                        classes.append(row[0])
                        probs.append(row[-1] if has_prob else 1.0)
                        segments.append(seg.reshape(-1, 2))  # (xy1...)
                    classes = np.array(classes, dtype=np.float32)
                    cls_probs = np.array(probs, dtype=np.float32).reshape(-1, 1)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            if nl := len(lb):
                if keypoint:
                    assert lb.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    points = lb[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    if lb.shape[1] == 6:  # optional soft-label probability for detect labels: cls xywh prob
                        cls_probs = lb[:, 5:6]
                        lb = lb[:, :5]
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    if len(cls_probs) != nl:
                        cls_probs = np.ones((nl, 1), dtype=np.float32)
                    points = lb[:, 1:]
                # Coordinate points check with 1% tolerance
                assert points.max() <= 1.01, f"non-normalized or out of bounds coordinates {points[points > 1.01]}"
                assert lb.min() >= -0.01, f"negative class labels or coordinate {lb[lb < -0.01]}"
                if cls_probs.size:
                    assert cls_probs.min() >= -0.01, f"negative label probability {cls_probs[cls_probs < -0.01]}"
                    assert cls_probs.max() <= 1.01, (
                        f"label probability out of bounds {cls_probs[cls_probs > 1.01]}"
                    )

                # All labels
                max_cls = 0 if single_cls else lb[:, 0].max()  # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                dedupe = np.concatenate((lb, cls_probs), axis=1) if len(cls_probs) else lb
                _, i = np.unique(dedupe, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    cls_probs = cls_probs[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}{im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
                cls_probs = np.zeros((0, 1), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
            cls_probs = np.zeros((0, 1), dtype=np.float32)
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
            cls_probs = np.ones((lb.shape[0], 1), dtype=np.float32)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, cls_probs, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}{im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, None, nm, nf, ne, nc, msg]


def visualize_image_annotations(image_path: str, txt_path: str, label_map: dict[int, str]):
    """
    Visualize YOLO annotations (bounding boxes and class labels) on an image.

    This function reads an image and its corresponding annotation file in YOLO format, then
    draws bounding boxes around detected objects and labels them with their respective class names.
    The bounding box colors are assigned based on the class ID, and the text color is dynamically
    adjusted for readability, depending on the background color's luminance.

    Args:
        image_path (str): The path to the image file to annotate, and it can be in formats supported by PIL.
        txt_path (str): The path to the annotation file in YOLO format, that should contain one line per object.
        label_map (dict[int, str]): A dictionary that maps class IDs (integers) to class labels (strings).

    Examples:
        >>> label_map = {0: "cat", 1: "dog", 2: "bird"}  # It should include all annotated classes details
        >>> visualize_image_annotations("path/to/image.jpg", "path/to/annotations.txt", label_map)
    """
    import matplotlib.pyplot as plt

    from ultralytics.utils.plotting import colors

    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    _, ax = plt.subplots(1)  # Plot the image and annotations
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, True))  # Get and normalize the RGB color
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")  # Create a rectangle
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # Formula for luminance
        ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    plt.show()


def polygon2mask(
    imgsz: tuple[int, int], polygons: list[np.ndarray], color: int = 1, downsample_ratio: int = 1
) -> np.ndarray:
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple[int, int]): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape (N, M), where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask.
        downsample_ratio (int, optional): Factor by which to downsample the mask.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(
    imgsz: tuple[int, int], polygons: list[np.ndarray], color: int, downsample_ratio: int = 1
) -> np.ndarray:
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple[int, int]): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape (N, M), where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(
    imgsz: tuple[int, int], segments: list[np.ndarray], downsample_ratio: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for segment in segments:
        mask = polygon2mask(
            imgsz,
            [segment.reshape(-1)],
            downsample_ratio=downsample_ratio,
            color=1,
        )
        ms.append(mask.astype(masks.dtype))
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect, Segment or Pose dataset.

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it
    performs a recursive search. It prefers YAML files that have the same stem as the provided path.

    Args:
        path (Path): The directory path to search for the YAML file.

    Returns:
        (Path): The path of the found YAML file.
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # try root level first and then recursive
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # prefer *.yaml files that match
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]

def compute_channels(channels, hyp):
    return int(channels)


def check_det_dataset(dataset: str, autodownload: bool = True, hyp: dict = None) -> dict[str, Any]:
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found.

    Returns:
        (dict[str, Any]): Parsed dataset information and paths.
    """
    file = check_file(dataset)

    # Download (optional)
    extract_dir = ""
    if zipfile.is_zipfile(file) or is_tarfile(file):
        new_dir = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)
        file = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = file.parent, False

    # Read YAML
    data = YAML.load(file, append_filename=True)  # dictionary

    # Checks
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            LOGGER.warning("renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val"] = data.pop("validation")  # replace 'validation' key with 'val' key
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])
    validate_bands_config(data, hyp)
    data["channels"] = compute_channels(data.get("channels", RGB_BAND_COUNT), hyp)

    # Resolve paths
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    if not path.exists() and not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()  # path relative to DATASETS_DIR

    # Set paths
    data["path"] = path  # download scripts
    for k in "train", "val", "test", "minival":
        if data.get(k):  # prepend path
            data[k] = _resolve_split_entry(path, data[k])
    if data.get(METADATA_KEY):
        spec = data[METADATA_KEY]
        data[METADATA_KEY] = {
            split: _resolve_split_entry(path, split_spec)
            for split in AUX_MASK_SPLITS
            if data.get(split) and (split_spec := _get_auxiliary_split_spec(spec, split))
        }

    validate_metadata_config(data)

    # Parse YAML
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name with URL auth stripped
            LOGGER.info("")
            m = f"Dataset '{name}' images not found, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_FILE}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # success
            if s.startswith("http") and s.endswith(".zip"):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                subprocess.run(s.split(), check=True)
            else:  # python script
                exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}\n")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf")  # download fonts

    return data  # dictionary


def check_cls_dataset(dataset: str | Path, split: str = "", hyp: dict = None) -> dict[str, Any]:
    """
    Check a classification dataset such as Imagenet.

    This function accepts a `dataset` name and attempts to retrieve the corresponding dataset information.
    If the dataset is not found locally, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str | Path): The name of the dataset.
        split (str, optional): The split of the dataset. Either 'val', 'test', or ''.

    Returns:
        (dict[str, Any]): A dictionary containing the following keys:

            - 'train' (Path): The directory path containing the training set of the dataset.
            - 'val' (Path): The directory path containing the validation set of the dataset.
            - 'test' (Path): The directory path containing the test set of the dataset.
            - 'nc' (int): The number of classes in the dataset.
            - 'names' (dict[int, str]): A dictionary of class names in the dataset.
    """
    # Download (optional if dataset=https://file.zip is passed directly)
    if str(dataset).startswith(("http:/", "https:/")):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)
    elif str(dataset).endswith((".zip", ".tar", ".gz")):
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)

    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()
    if not data_dir.is_dir():
        if data_dir.suffix != "":
            raise ValueError(
                f'Classification datasets must be a directory (data="path/to/dir") not a file (data="{dataset}"), '
                "See https://docs.ultralytics.com/datasets/classify/"
            )
        LOGGER.info("")
        LOGGER.warning(f"Dataset not found, missing path {data_dir}, attempting download...")
        t = time.time()
        if str(dataset) == "imagenet":
            subprocess.run(["bash", str(ROOT / "data/scripts/get_imagenet.sh")], check=True)
        else:
            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{dataset}.zip"
            download(url, dir=data_dir.parent)
        LOGGER.info(f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n")
    train_set = data_dir / "train"
    if not train_set.is_dir():
        LOGGER.warning(f"Dataset 'split=train' not found at {train_set}")
        if image_files := list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png")):
            from ultralytics.data.split import split_classify_dataset

            LOGGER.info(f"Found {len(image_files)} images in subdirectories. Attempting to split...")
            data_dir = split_classify_dataset(data_dir, train_ratio=0.8)
            train_set = data_dir / "train"
        else:
            LOGGER.error(f"No images found in {data_dir} or its subdirectories.")
    val_set = (
        data_dir / "val"
        if (data_dir / "val").exists()
        else data_dir / "validation"
        if (data_dir / "validation").exists()
        else data_dir / "valid"
        if (data_dir / "valid").exists()
        else None
    )  # data/test or data/val
    test_set = data_dir / "test" if (data_dir / "test").exists() else None  # data/val or data/test
    if split == "val" and not val_set:
        LOGGER.warning("Dataset 'split=val' not found, using 'split=test' instead.")
        val_set = test_set
    elif split == "test" and not test_set:
        LOGGER.warning("Dataset 'split=test' not found, using 'split=val' instead.")
        test_set = val_set

    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # number of classes
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]  # class names list
    names = dict(enumerate(sorted(names)))

    # Print to console
    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():
        prefix = f"{colorstr(f'{k}:')} {v}..."
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            nf = len(files)  # number of files
            nd = len({file.parent for file in files})  # number of directories
            if nf == 0:
                if k == "train":
                    raise FileNotFoundError(f"{dataset} '{k}:' no training images found")
                else:
                    LOGGER.warning(f"{prefix} found {nf} images in {nd} classes (no images found)")
            elif nd != nc:
                LOGGER.error(f"{prefix} found {nf} images in {nd} classes (requires {nc} classes, not {nd})")
            else:
                LOGGER.info(f"{prefix} found {nf} images in {nd} classes ✅ ")

    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names, "channels": 3}


class HUBDatasetStats:
    """
    A class for generating HUB dataset JSON and `-hub` dataset directory.

    Args:
        path (str): Path to data.yaml or data.zip (with data.yaml inside data.zip).
        task (str): Dataset task. Options are 'detect', 'segment', 'pose', 'classify'.
        autodownload (bool): Attempt to download dataset if not found locally.

    Attributes:
        task (str): Dataset task type.
        hub_dir (Path): Directory path for HUB dataset files.
        im_dir (Path): Directory path for compressed images.
        stats (dict): Statistics dictionary containing dataset information.
        data (dict): Dataset configuration data.

    Methods:
        get_json: Return dataset JSON for Ultralytics HUB.
        process_images: Compress images for Ultralytics HUB.

    Note:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
        i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.

    Examples:
        >>> from ultralytics.data.utils import HUBDatasetStats
        >>> stats = HUBDatasetStats("path/to/coco8.zip", task="detect")  # detect dataset
        >>> stats = HUBDatasetStats("path/to/coco8-seg.zip", task="segment")  # segment dataset
        >>> stats = HUBDatasetStats("path/to/coco8-pose.zip", task="pose")  # pose dataset
        >>> stats = HUBDatasetStats("path/to/dota8.zip", task="obb")  # OBB dataset
        >>> stats = HUBDatasetStats("path/to/imagenet10.zip", task="classify")  # classification dataset
        >>> stats.get_json(save=True)
        >>> stats.process_images()
    """

    def __init__(self, path: str = "coco8.yaml", task: str = "detect", autodownload: bool = False):
        """Initialize class."""
        path = Path(path).resolve()
        LOGGER.info(f"Starting HUB dataset checks for {path}....")

        self.task = task  # detect, segment, pose, classify, obb
        if self.task == "classify":
            unzip_dir = unzip_file(path)
            data = check_cls_dataset(unzip_dir)
            data["path"] = unzip_dir
        else:  # detect, segment, pose, obb
            _, data_dir, yaml_path = self._unzip(Path(path))
            try:
                # Load YAML with checks
                data = YAML.load(yaml_path)
                data["path"] = ""  # strip path since YAML should be in dataset root for all HUB datasets
                YAML.save(yaml_path, data)
                data = check_det_dataset(yaml_path, autodownload)  # dict
                data["path"] = data_dir  # YAML path should be set to '' (relative) or parent (absolute)
            except Exception as e:
                raise Exception("error/HUB/dataset_stats/init") from e

        self.hub_dir = Path(f"{data['path']}-hub")
        self.im_dir = self.hub_dir / "images"
        self.stats = {"nc": len(data["names"]), "names": list(data["names"].values())}  # statistics dictionary
        self.data = data

    @staticmethod
    def _unzip(path: Path) -> tuple[bool, str, Path]:
        """Unzip data.zip."""
        if not str(path).endswith(".zip"):  # path is data.yaml
            return False, None, path
        unzip_dir = unzip_file(path, path=path.parent)
        assert unzip_dir.is_dir(), (
            f"Error unzipping {path}, {unzip_dir} not found. path/to/abc.zip MUST unzip to path/to/abc/"
        )
        return True, str(unzip_dir), find_dataset_yaml(unzip_dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f: str):
        """Save a compressed image for HUB previews."""
        compress_one_image(f, self.im_dir / Path(f).name)  # save to dataset-hub

    def get_json(self, save: bool = False, verbose: bool = False) -> dict:
        """Return dataset JSON for Ultralytics HUB."""

        def _round(labels):
            """Update labels to integer class and 4 decimal place floats."""
            if self.task == "detect":
                coordinates = labels["bboxes"]
            elif self.task in {"segment", "obb"}:  # Segment and OBB use segments. OBB segments are normalized xyxyxyxy
                coordinates = [x.flatten() for x in labels["segments"]]
            elif self.task == "pose":
                n, nk, nd = labels["keypoints"].shape
                coordinates = np.concatenate((labels["bboxes"], labels["keypoints"].reshape(n, nk * nd)), 1)
            else:
                raise ValueError(f"Undefined dataset task={self.task}.")
            zipped = zip(labels["cls"], coordinates)
            return [[int(c[0]), *(round(float(x), 4) for x in points)] for c, points in zipped]

        for split in "train", "val", "test":
            self.stats[split] = None  # predefine
            path = self.data.get(split)

            # Check split
            if path is None:  # no split
                continue
            files = [f for f in Path(path).rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]  # image files in split
            if not files:  # no images
                continue

            # Get dataset statistics
            if self.task == "classify":
                from torchvision.datasets import ImageFolder  # scope for faster 'import ultralytics'

                dataset = ImageFolder(self.data[split])

                x = np.zeros(len(dataset.classes)).astype(int)
                for im in dataset.imgs:
                    x[im[1]] += 1

                self.stats[split] = {
                    "instance_stats": {"total": len(dataset), "per_class": x.tolist()},
                    "image_stats": {"total": len(dataset), "unlabelled": 0, "per_class": x.tolist()},
                    "labels": [{Path(k).name: v} for k, v in dataset.imgs],
                }
            else:
                from ultralytics.data import YOLODataset

                dataset = YOLODataset(img_path=self.data[split], data=self.data, task=self.task)
                x = np.array(
                    [
                        np.bincount(label["cls"].astype(int).flatten(), minlength=self.data["nc"])
                        for label in TQDM(dataset.labels, total=len(dataset), desc="Statistics")
                    ]
                )  # shape(128x80)
                self.stats[split] = {
                    "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
                    "image_stats": {
                        "total": len(dataset),
                        "unlabelled": int(np.all(x == 0, 1).sum()),
                        "per_class": (x > 0).sum(0).tolist(),
                    },
                    "labels": [{Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)],
                }

        # Save, print and return
        if save:
            self.hub_dir.mkdir(parents=True, exist_ok=True)  # makes dataset-hub/
            stats_path = self.hub_dir / "stats.json"
            LOGGER.info(f"Saving {stats_path.resolve()}...")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self) -> Path:
        """Compress images for Ultralytics HUB."""
        from ultralytics.data import YOLODataset  # ClassificationDataset

        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes dataset-hub/images/
        for split in "train", "val", "test":
            if self.data.get(split) is None:
                continue
            dataset = YOLODataset(img_path=self.data[split], data=self.data)
            with ThreadPool(NUM_THREADS) as pool:
                for _ in TQDM(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f"{split} images"):
                    pass
        LOGGER.info(f"Done. All images saved to {self.im_dir}")
        return self.im_dir


def compress_one_image(f: str, f_new: str = None, max_dim: int = 1920, quality: int = 50):
    """
    Compress a single image file to reduced size while preserving its aspect ratio and quality using either the Python
    Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will not be
    resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image.
        quality (int, optional): The image compression quality as a percentage.

    Examples:
        >>> from pathlib import Path
        >>> from ultralytics.data.utils import compress_one_image
        >>> for f in Path("path/to/dataset").rglob("*.jpg"):
        >>>    compress_one_image(f)
    """
    try:  # use PIL
        Image.MAX_IMAGE_PIXELS = None  # Fix DecompressionBombError, allow optimization of image > ~178.9 million pixels
        im = Image.open(f)
        if im.mode in {"RGBA", "LA"}:  # Convert to RGB if needed (for JPEG)
            im = im.convert("RGB")
        r = max_dim / max(im.height, im.width)  # ratio
        if r < 1.0:  # image too large
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, "JPEG", quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.warning(f"HUB ops PIL failure {f}: {e}")
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1.0:  # image too large
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)


def load_dataset_cache_file(path: Path) -> dict:
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix: str, path: Path, x: dict, version: str):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = version  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        with open(str(path), "wb") as file:  # context manager here fixes windows async np.save bug
            np.save(file, x)
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}Cache directory {path.parent} is not writeable, cache not saved.")
