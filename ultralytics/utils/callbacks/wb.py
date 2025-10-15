# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import yaml
from pathlib import Path
import os
from copy import deepcopy
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
        
        wb.init(
            project=str(trainer.args.project).replace("/", "-") if trainer.args.project else "Ultralytics",
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

def _has_test_split(data_spec) -> bool:
    """
    Return True iff 'test:' exists in the dataset YAML.
    Handles multiple data specification formats:
    - Path to YAML file
    - Dictionary with 'test' key
    - Inline string (returns True optimistically)
    """
    try:
        # Case 1: data_spec is a Path or string pointing to a YAML file
        if isinstance(data_spec, (str, Path)):
            p = Path(str(data_spec))
            if p.exists() and p.suffix in {".yaml", ".yml"}:
                content = p.read_text(encoding="utf-8")
                data_dict = yaml.safe_load(content)
                
                if isinstance(data_dict, dict):
                    # Check if 'test' key exists and is not None/empty
                    test_value = data_dict.get("test")
                    return test_value is not None and test_value != ""
        
        # Case 2: data_spec is already a dictionary
        elif isinstance(data_spec, dict):
            test_value = data_spec.get("test")
            return test_value is not None and test_value != ""
        
    except Exception as e:
        # If parsing fails (e.g., custom YAML loader, encoding issues),
        # return True optimistically and let Ultralytics validator handle it
        import warnings
        warnings.warn(f"Could not parse data spec for test split detection: {e}. Assuming test split exists.")
    
    # Optimistic default: if we can't determine, assume test exists
    # This prevents skipping test eval when it might be available
    return False


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
        elif not _has_test_split(data):
            LOGGER.info("No 'test' split in data.yaml; skipping test evaluation.")
        else:
            # Determine which model to use for test evaluation
            best_path = getattr(trainer, "best", None)
            best_exists = bool(best_path) and Path(best_path).exists()

            if best_exists:
                LOGGER.info(f"Evaluating best checkpoint on test split: {best_path}")
                best_model = YOLO(str(best_path))
            else:
                LOGGER.info("Best checkpoint not found; using current model for test evaluation.")
                best_model = trainer.model

            # Run test evaluation with plots disabled
            test_results = best_model.val(
                data=data,
                split="test",
                device=getattr(trainer.args, "device", None),
                batch=getattr(trainer.args, "batch", None),
                imgsz=getattr(trainer.args, "imgsz", None),
                conf=getattr(trainer.args, "conf", 0.001),
                iou=getattr(trainer.args, "iou", 0.7),
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
            else:
                LOGGER.info("Test results object missing 'results_dict' attribute.")

    except Exception as e:
        # Log detailed error information
        error_info = {
            "test_eval_error": str(e),
            "test_eval_error_type": type(e).__name__,
            # "test_eval_traceback": traceback.format_exc(),
        }
        LOGGER.info(error_info)
        wb.run.summary.update({"test_eval_failed": True})

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
