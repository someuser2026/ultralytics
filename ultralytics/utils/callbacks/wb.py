# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import yaml
from pathlib import Path
import os
from copy import deepcopy
import numpy as np
from PIL import Image
# from ultralytics.models.yolo.model import YOLO

from ultralytics.utils import SETTINGS, TESTS_RUNNING, LOGGER
from ultralytics.utils.torch_utils import model_info_for_loggers
from ultralytics.models.yolo.model import YOLO

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["wandb"] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, "__version__")  # verify package is not directory
    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


def _custom_table(x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """
    Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of the default wandb precision-recall
    curve while allowing for enhanced customization. The visual metric is useful for monitoring model performance across
    different classes.

    Args:
        x (list): Values for the x-axis; expected to have length N.
        y (list): Corresponding values for the y-axis; also expected to have length N.
        classes (list): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot.
        x_title (str, optional): Label for the x-axis.
        y_title (str, optional): Label for the y-axis.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    import polars as pl  # scope for faster 'import ultralytics'
    import polars.selectors as cs

    df = pl.DataFrame({"class": classes, "y": y, "x": x}).with_columns(cs.numeric().round(3))
    data = df.select(["class", "y", "x"]).rows()

    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wb.plot_table(
        "wandb/area-under-curve/v0",
        wb.Table(data=data, columns=["class", "y", "x"]),
        fields=fields,
        string_fields=string_fields,
    )


def _ux_make_pair_tile(pred_path: Path, label_path: Path, tile_h: int = 256) -> Image.Image:
    """
    Create a single horizontal tile showing prediction and ground-truth side-by-side.
    
    Combines two overlay images into a single tile with format: [pred | label]
    Both images are resized to the same height while preserving aspect ratio.
    
    Args:
        pred_path (Path): Path to prediction overlay image (*_pred.jpg)
        label_path (Path): Path to ground-truth overlay image (*_labels.jpg)
        tile_h (int): Target height in pixels for both images. Default: 256
    
    Returns:
        Image.Image: Combined PIL Image with predictions on left, labels on right
    
    Notes:
        - Missing/corrupt images are replaced with light gray placeholders
        - Images are resized to matching heights to create uniform tiles
        - Original aspect ratios are preserved during resizing
    """
    def _safe_open(p):
        """Safely open image or return placeholder if file is missing/corrupt."""
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            # Return light gray placeholder (245, 245, 245) if image fails to load
            return Image.new("RGB", (tile_h, tile_h), (245, 245, 245))

    # Load both prediction and label images (or placeholders)
    im_p = _safe_open(pred_path)
    im_l = _safe_open(label_path)

    def _resize_h(img):
        """Resize image to target height while preserving aspect ratio."""
        if img.height == 0:
            return img  # Avoid division by zero
        # Calculate new width to maintain aspect ratio at target height
        new_w = max(1, int(round(img.width * (tile_h / img.height))))
        return img.resize((new_w, tile_h))

    # Resize both images to same height
    im_p = _resize_h(im_p)
    im_l = _resize_h(im_l)

    # Create horizontal tile: [prediction | label]
    tile = Image.new("RGB", (im_p.width + im_l.width, tile_h), (255, 255, 255))
    tile.paste(im_p, (0, 0))           # Prediction on left
    tile.paste(im_l, (im_p.width, 0))  # Label on right
    return tile


def _ux_save_grid(pairs: list[tuple[Path, Path]], out_path: Path, rows: int = 4, cols: int = 4, tile_h: int = 256) -> Path:
    """
    Assemble a rows×cols grid of (pred|label) tiles and save to disk.
    
    Creates a uniform grid where each cell contains a side-by-side comparison
    of predictions vs ground-truth for a single validation image.
    
    Args:
        pairs (list[tuple[Path, Path]]): List of (pred_path, label_path) tuples
            Each tuple points to the prediction and label overlay images
        out_path (Path): Destination path for the assembled grid image
        rows (int): Number of grid rows. Default: 4
        cols (int): Number of grid columns. Default: 4
        tile_h (int): Height in pixels for each tile. Default: 256
    
    Returns:
        Path: The output path where the grid was saved (same as out_path)
    
    Notes:
        - Only the first (rows × cols) pairs are used; extras are ignored
        - Tiles are centered in their grid cells with 6px padding between cells
        - All tiles are resized to the width of the widest tile for uniformity
        - Output directory is created automatically if it doesn't exist
    """
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate tiles for the grid (up to rows×cols images)
    tiles = [_ux_make_pair_tile(a, b, tile_h) for a, b in pairs[: rows * cols]]

    # Calculate grid dimensions with uniform cell sizing
    max_w = max([t.width for t in tiles], default=2 * tile_h)  # Width of widest tile
    cell_w, cell_h, pad = max_w, tile_h, 6  # Cell dimensions and inter-cell padding
    grid_w = cols * cell_w + (cols - 1) * pad  # Total grid width
    grid_h = rows * cell_h + (rows - 1) * pad  # Total grid height
    
    # Create white canvas for the grid
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    # Place each tile in its grid position
    for i in range(rows * cols):
        r, c = divmod(i, cols)  # Calculate row and column indices
        x = c * (cell_w + pad)  # X position (left edge of cell)
        y = r * (cell_h + pad)  # Y position (top edge of cell)
        
        if i < len(tiles):
            t = tiles[i]
            # Center tile horizontally within its cell
            x_off = x + (cell_w - t.width) // 2
            canvas.paste(t, (x_off, y))
    
    # Save assembled grid to disk
    canvas.save(out_path)
    return out_path


def _ux_build_and_log_f2_grids(trainer, per_grid: int = 8):
    """
    Build and log 4 W&B grids showing best and worst F2 score examples.
    
    Creates visualization grids to help diagnose model performance:
      - Top-F2 grids (2):   Show 16 examples with highest F2 scores (best predictions)
      - Worst-F2 grids (2): Show 16 examples with lowest F2 scores (worst predictions)
    
    Each grid is 4×4 (8 examples per grid, 16 total per category).
    Each cell displays: [prediction overlay | ground-truth overlay]
    
    Args:
        trainer: Ultralytics trainer instance with validator and metrics
        per_grid (int): Number of examples per grid (default: 8 for 4×4 grids)
    
    Returns:
        None. Logs 4 grid images to W&B or returns silently if:
        - W&B is not installed
        - Validator or metrics are unavailable
        - Per-image F2 scores haven't been computed
    
    Notes:
        - Sources per-image overlays from <save_dir>/per_image/<stem>_{pred,labels}.jpg
        - F2 scores must be stored in validator.metrics.per_image["box"]["f2"]
        - Image IDs are retrieved from "image_id" or "img" keys in metrics
        - All grids are saved to <save_dir>/per_image/grids/ before W&B upload
    """
    # Check if W&B is available
    # try:
    #     import wandb as wb
    # except Exception:
    #     return  # W&B not installed, exit silently

    # Retrieve validator from trainer
    validator = getattr(trainer, "validator", None)
    if validator is None:
        return  # No validator available
    
    # Access per-image metrics
    m = getattr(validator.metrics, "per_image", None)
    if not m or "box" not in m:
        return  # Per-image metrics not available

    # Extract per-image F2 scores and image identifiers
    box = m["box"]
    img_ids = np.asarray(box.get("image_id") or box.get("img") or [])
    f2 = np.asarray(box.get("f2") or [])
    
    # Validate data availability
    if len(img_ids) == 0 or len(f2) == 0:
        return  # No F2 scores computed

    # Sort images by F2 score
    order_asc = np.argsort(f2)       # Ascending: worst -> best F2 scores
    order_desc = order_asc[::-1]     # Descending: best -> worst F2 scores

    # Locate per-image overlay directory
    per_img_dir = Path(getattr(validator.metrics, "save_dir", getattr(validator, "save_dir", Path(".")))) / "per_image"

    def _pairs(indices):
        """Build list of (pred_path, label_path) tuples for given image indices."""
        out = []
        for i in indices:
            stem = str(img_ids[i])  # Image filename stem
            out.append((
                per_img_dir / f"{stem}_pred.jpg",    # Prediction overlay
                per_img_dir / f"{stem}_labels.jpg"   # Ground-truth overlay
            ))
        return out

    # Select examples for grids
    top_pairs = _pairs(order_desc[: 2 * per_grid])  # Best 16 examples (highest F2)
    low_pairs = _pairs(order_asc[: 2 * per_grid])   # Worst 16 examples (lowest F2)

    # Assemble and save grids
    grids_dir = per_img_dir / "grids"
    
    # Top F2 grids (split into 2 grids of 8 examples each)
    top_g1 = _ux_save_grid(top_pairs[:per_grid], grids_dir / "topF2_grid_1.jpg")
    top_g2 = _ux_save_grid(top_pairs[per_grid: 2 * per_grid], grids_dir / "topF2_grid_2.jpg")
    
    # Worst F2 grids (split into 2 grids of 8 examples each)
    low_g1 = _ux_save_grid(low_pairs[:per_grid], grids_dir / "lowF2_grid_1.jpg")
    low_g2 = _ux_save_grid(low_pairs[per_grid: 2 * per_grid], grids_dir / "lowF2_grid_2.jpg")

    # Log all grids to W&B with descriptive keys
    wb.log({
        "examples/topF2_grid_1": wb.Image(str(top_g1)),  # Best 8 examples (grid 1)
        "examples/topF2_grid_2": wb.Image(str(top_g2)),  # Best 8 examples (grid 2)
        "examples/lowF2_grid_1": wb.Image(str(low_g1)),  # Worst 8 examples (grid 1)
        "examples/lowF2_grid_2": wb.Image(str(low_g2)),  # Worst 8 examples (grid 2)
    })


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """
    Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb.
    The curve can represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.

    Args:
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape (C, N), where C is the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C.
        id (str, optional): Unique identifier for the logged data in wandb.
        title (str, optional): Title for the visualization plot.
        x_title (str, optional): Label for the x-axis.
        y_title (str, optional): Label for the y-axis.
        num_x (int, optional): Number of interpolated data points for visualization.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted.

    Notes:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    import numpy as np

    # Create new x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # Create arrays for logging
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # add new x
            y_log.extend(np.interp(x_new, x, yi))  # interpolate y to new x
            classes.extend([names[i]] * len(x_new))  # add class names
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(plots, step):
    """
    Log plots to WandB at a specific step if they haven't been logged already.

    This function checks each plot in the input dictionary against previously processed plots and logs
    new or updated plots to WandB at the specified step.

    Args:
        plots (dict): Dictionary of plots to log, where keys are plot names and values are dictionaries
            containing plot metadata including timestamps.
        step (int): The step/epoch at which to log the plots in the WandB run.

    Notes:
        The function uses a shallow copy of the plots dictionary to prevent modification during iteration.
        Plots are identified by their stem name (filename without extension).
        Each plot is logged as a WandB Image object.
    """
    for name, params in plots.copy().items():  # shallow copy to prevent plots dict changing during iteration
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    """Initialize and start wandb project if module is present."""
    if not wb.run:
        # Create a copy of the config
        config = deepcopy(vars(trainer.args))
        
        # Shorten the 'data' path if it exists
        if 'data' in config and config['data']:
            config['data'] = str(Path(config['data']).parts[-2:]) if len(Path(config['data']).parts) >= 2 else os.path.basename(config['data'])
        
        # Shorten the 'model' path if it exists
        if 'model' in config and config['model']:
            config['model'] = os.path.basename(config['model'])
        
        project_cleaned = "-".join(str(trainer.args.project).split("/")[-3:]) if trainer.args.project else "Ultralytics"
        
        wb.init(
            project=project_cleaned,
            name=str(trainer.args.name).replace("/", "-"),
            config=config,
        )


def on_fit_epoch_end(trainer):
    """Log training metrics and model information at the end of an epoch."""
    wb.run.log(trainer.metrics, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)

def _get_val_test_dir(data_spec) -> tuple[bool, Path, Path]:
    """
    Return True iff 'test:' exists in the dataset YAML.
    Handles multiple data specification formats:
    - Path to YAML file
    - Dictionary with 'test' key
    - Inline string (returns True optimistically)
    """
    val_dir = None
    test_dir = None
    try:
        # Case 1: data_spec is a Path or string pointing to a YAML file
        if isinstance(data_spec, (str, Path)):
            p = Path(str(data_spec))
            if p.exists() and p.suffix in {".yaml", ".yml"}:
                content = p.read_text(encoding="utf-8")
                data_dict = yaml.safe_load(content)
                
                if isinstance(data_dict, dict):
                    # Check if 'test' key exists and is not None/empty
                    test_dir = data_dict.get("test")
                    val_dir = data_dict.get("val")
                    return test_dir is not None and test_dir != "", Path(data_dict.get("path")) / test_dir, Path(data_dict.get("path")) / val_dir
        
        # Case 2: data_spec is already a dictionary
        elif isinstance(data_spec, dict):
            test_dir = data_spec.get("test")
            val_dir = data_spec.get("val")
            return test_dir is not None and test_dir != "", Path(data_spec.get("path")) / test_dir, Path(data_spec.get("path")) / val_dir
        
    except Exception as e:
        # If parsing fails (e.g., custom YAML loader, encoding issues),
        # return True optimistically and let Ultralytics validator handle it
        import warnings
        warnings.warn(f"Could not parse data spec for test split detection: {e}. Assuming test split exists.")
    
    # Optimistic default: if we can't determine, assume test exists
    # This prevents skipping test eval when it might be available
    return False

def _log_predictions(pred_dir, run_name, subset):
    LOGGER.info(f"Logging test labels from {pred_dir} to wandb")
    artifact = wb.Artifact(run_name + "_predictions_" + subset, type = "predictions_" + subset)
    artifact.add_dir(str(pred_dir))
    wb.log_artifact(artifact)


def on_train_end(trainer):
    """
    End-of-training:
      1) Log final train/val plots (existing behavior).
      2) Evaluate the BEST checkpoint on the TEST split (if present), with plots disabled.
      3) Log test metrics into the same W&B run.
      4) Finish the W&B run.
    """
    step = trainer.epoch + 1

    # Existing behavior: log final train/val plots
    # _log_plots(trainer.validator.plots, step=step)
    # _log_plots(trainer.plots, step=step)

    # Existing: optional curves (val)
    if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
        for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(trainer.validator.metrics.names.values()),
                id=f"curves/{curve_name}",
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )

    # NEW: post-training evaluation on TEST split using BEST model; no plot saving/logging
    try:
        data = getattr(trainer.args, "data", None)
        if not data:
            LOGGER.info("No data config provided; skipping test evaluation.")
        else:
            has_test, test_dir, val_dir = _get_val_test_dir(data)
        
        # Determine which model to use for evaluation
        best_path = getattr(trainer, "best", None)
        best_exists = bool(best_path) and Path(best_path).exists()
        if best_exists:
            LOGGER.info(f"Evaluating best checkpoint on test split: {best_path}")
            best_model = YOLO(str(best_path))
        else:
            LOGGER.info("Best checkpoint not found; using current model for test evaluation.")
            best_model = trainer.model
        
        # prediction on val set
        list(best_model.predict(val_dir, True, save_conf = True,
            save_txt = True, conf = 0.01, project = trainer.args.project, name = os.path.join(trainer.args.name, "labels", "val")
        ))
        labels_dir = Path(trainer.args.project) / trainer.args.name / "labels"
        val_labels = labels_dir / "val"
        _log_predictions(val_labels, trainer.args.name, "val")

        if not has_test:
            LOGGER.info("No 'test' split in data.yaml; skipping test evaluation.")
        else:
            # Run test evaluation with plots disabled
            test_results = best_model.val(
                data=data,
                split="test",
                device=getattr(trainer.args, "device", None),
                batch=getattr(trainer.args, "batch", None),
                imgsz=getattr(trainer.args, "imgsz", None),
                conf=getattr(trainer.args, "conf", 0.001),
                iou=getattr(trainer.args, "iou", 0.7),
                fitness_weights = getattr(trainer.args, "fitness_weights", None),
                plots=False,  # Disable plot generation/saving
                save_json=False,  # Optional: disable JSON saving
                verbose=False,  # Optional: reduce console output
            )

            # Log test metrics to the SAME run (both series and summary)
            if hasattr(test_results, "results_dict") and isinstance(test_results.results_dict, dict):
                prefixed_metrics = {f"test/{k}": v for k, v in test_results.results_dict.items()}
                
                # Log as timestep entry
                wb.run.log(prefixed_metrics, step=step)
                
                # Also update summary for easy access
                wb.run.summary.update(prefixed_metrics)
                
                # Optional: log a summary message
                if "metrics/mAP50-95(B)" in test_results.results_dict:
                    map_value = test_results.results_dict["metrics/mAP50-95(B)"]
                    LOGGER.info(f"Test evaluation complete. mAP50-95: {map_value:.4f}")

                # store predictions on val and test set for downstream processing
                list(best_model.predict(test_dir, True, conf = 0.01, save_conf = True,
                    save_txt = True, batch = 2, project = trainer.args.project, name = os.path.join(trainer.args.name, "labels", "test")
                ))
                test_labels = labels_dir / "test"
                if test_labels.exists():
                    _log_predictions(test_labels, trainer.args.name, "test")
            else:
                LOGGER.info("Test results object missing 'results_dict' attribute.")

    except Exception as e:
        # Log detailed error information
        error_info = {
            "test_eval_error": str(e),
            "test_eval_error_type": type(e).__name__,
            "test_eval_traceback": e,
        }
        # LOGGER.info(error_info)
        LOGGER.error(e)
        wb.run.summary.update({"test_eval_failed": True})
    
    try:
        # Generate and log F2 visualization grids
        # Creates 4 grids: 2 showing best predictions, 2 showing worst predictions
        # Each grid contains 8 examples in 4×4 layout
        _ux_build_and_log_f2_grids(trainer, per_grid=8)
    except Exception as e:
        # Log failures gracefully without interrupting training cleanup
        # from ultralytics.utils import LOGGER
        LOGGER.warning(f"W&B F2 grids skipped: {e}")

    # Finish the run (existing behavior)
    wb.run.finish()



callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
