# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo11n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import cfg2dict, get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import (
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.metrics import calculate_fitness
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)


class BaseTrainer:
    """
    A base class for creating trainers.

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.

    Methods:
        train: Execute the training process.
        validate: Run validation on the test set.
        save_model: Save model training checkpoints.
        get_dataset: Get train and validation datasets.
        setup_model: Load, create, or download model.
        build_optimizer: Construct an optimizer for the model.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        overrides = cfg2dict(overrides) if overrides else {}
        self._explicit_epoch_limit = "epochs" in self._get_explicit_arg_keys(cfg, overrides)
        self.hub_session = overrides.pop("session", None)  # HUB
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device)
        # Update "-1" devices so post-training val does not repeat search
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self. best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo11n -> yolo11n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            # print("-*"*50)
            # print("inside base/trainer.py 160")
            # print("args:", self.args)
            self.data = self.get_dataset()
            # print("data:", self.data)
            # print("-*"*50)

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        self.ddp = world_size > 1 and "LOCAL_RANK" not in os.environ
        self.world_size = world_size
        # Run subprocess if DDP training, else train normally
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            # Start console logging immediately at trainer initialization
            self.run_callbacks("on_pretrain_routine_start")

    @staticmethod
    def _get_explicit_arg_keys(cfg, overrides):
        """Return config keys that were explicitly supplied by the caller."""
        explicit_keys = set(overrides)
        if cfg is DEFAULT_CFG or cfg is DEFAULT_CFG_DICT:
            return explicit_keys
        if isinstance(cfg, dict):
            if cfg == DEFAULT_CFG_DICT:
                return explicit_keys
            explicit_keys.update(cfg)
        elif isinstance(cfg, (str, Path)):
            if Path(cfg).expanduser().resolve() == DEFAULT_CFG_PATH.resolve():
                return explicit_keys
            explicit_keys.update(cfg2dict(cfg))
        return explicit_keys

    def _should_adjust_epochs_for_time(self):
        """Return True when timed training should estimate epochs dynamically."""
        return bool(self.args.time) and not self._explicit_epoch_limit

    def _time_exceeded(self):
        """Return True when the configured training time budget has been exhausted."""
        return bool(self.args.time) and (time.time() - self.train_time_start) > (self.args.time * 3600)

    def _training_duration_description(self):
        """Describe the active stop criteria for training logs."""
        if self.args.time and self._explicit_epoch_limit:
            return f"up to {self.epochs} epochs or {self.args.time} hours (whichever comes first)..."
        return f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs..."

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        # Run subprocess if DDP training, else train normally
        if self.ddp:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                )

            # Command
            cmd, file = generate_ddp_command(self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train()

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self):
        """Initialize and set the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=self.world_size,
        )

    @staticmethod
    def _resolve_layer_indices(value):
        """Normalize freeze-style args into a concrete list of outer model layer indices."""
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f"Layer counts must be >= 0, but received {value}.")
            return list(range(value))
        if isinstance(value, range):
            return list(value)
        if isinstance(value, (list, tuple)):
            if not all(isinstance(x, int) for x in value):
                raise TypeError(f"Layer index lists must contain only ints, but received {value!r}.")
            return list(value)
        if value in {None, False}:
            return []
        raise TypeError(f"Unsupported layer index spec {value!r}. Use an int, list[int], or tuple[int, ...].")

    @staticmethod
    def _has_layer_spec(value):
        """Return True when a freeze-style arg contains an actionable value."""
        return not (
            value is None
            or value is False
            or value == []
            or value == ()
            or (isinstance(value, range) and len(value) == 0)
        )

    @staticmethod
    def _is_lora_parameter_name(name):
        """Return True for trainable LoRA adapter parameters."""
        return ".lora_A" in name or ".lora_B" in name

    @staticmethod
    def _is_lora_base_parameter_name(name):
        """Return True for frozen base weights wrapped by a LoRA module."""
        return ".base_layer.weight" in name or ".base_layer.bias" in name

    @staticmethod
    def _weights_have_lora(weights):
        """Return True when an incoming checkpoint/model already contains LoRA parameters."""
        return weights is not None and hasattr(weights, "state_dict") and any(
            ".lora_A" in name or ".lora_B" in name for name in weights.state_dict()
        )

    def _get_timm_trainable_non_lora_parameter_names(self, timm_layer):
        """Return trainable timm parameter names excluding LoRA adapter tensors."""
        return [
            name for name, param in timm_layer.m.named_parameters() if param.requires_grad and not self._is_lora_parameter_name(name)
        ]

    def _get_timm_backbone_layers(self, model=None):
        """Return timm backbone entries as (outer_idx, outer_prefix, param_prefix, layer)."""
        model = unwrap_model(model if model is not None else self.model)
        backbone_layers = getattr(model, "backbone_layers", [])
        timm_layers = []
        for idx, layer in enumerate(backbone_layers):
            if layer.__class__.__name__ == "Timm" and hasattr(layer, "m"):
                timm_layers.append((idx, f"model.{idx}.", f"model.{idx}.m.", layer))
        return timm_layers

    @staticmethod
    def _get_timm_unfreeze_context(timm_layer):
        """Return the inner timm module and container used for selective unfreezing/LoRA layer targeting."""
        candidates = [("", timm_layer.m)]
        for prefix in ("model", "body", "backbone", "module"):
            child = getattr(timm_layer.m, prefix, None)
            if isinstance(child, nn.Module):
                candidates.append((f"{prefix}.", child))

        for root_prefix, root_module in candidates:
            for attr in ("blocks", "stages", "layers"):
                units = getattr(root_module, attr, None)
                if isinstance(units, (nn.ModuleList, nn.Sequential, list, tuple)) and len(units):
                    return root_module, root_prefix, attr, list(units)
        raise ValueError(
            "Timm selective unfreezing requires the wrapped timm model to expose a non-empty "
            "'blocks', 'stages', or 'layers' container."
        )

    @staticmethod
    def _get_timm_unfreeze_units(timm_layer):
        """Return the timm backbone container name and units used for selective unfreezing."""
        _, _, container_name, units = BaseTrainer._get_timm_unfreeze_context(timm_layer)
        return container_name, units

    def _resolve_timm_unfreeze_indices(self, spec, num_units, outer_idx, container_name):
        """Resolve timm unfreeze config into specific inner block/stage indices."""
        if isinstance(spec, int):
            if spec < 0:
                raise ValueError(f"'unfreeze' must be >= 0, but received {spec}.")
            if spec == 0:
                LOGGER.warning(
                    f"'unfreeze=0' leaves timm backbone layer {outer_idx} fully frozen. No {container_name} will be unfrozen."
                )
                return []
            if spec > num_units:
                raise ValueError(
                    f"'unfreeze={spec}' exceeds the {num_units} available timm {container_name} in backbone layer {outer_idx}."
                )
            return list(range(num_units - spec, num_units))

        if isinstance(spec, range):
            spec = list(spec)
        if isinstance(spec, tuple):
            spec = list(spec)
        if isinstance(spec, list):
            if not spec:
                LOGGER.warning(
                    f"'unfreeze=[]' leaves timm backbone layer {outer_idx} fully frozen. No {container_name} will be unfrozen."
                )
                return []
            if not all(isinstance(i, int) for i in spec):
                raise TypeError(f"'unfreeze' lists must contain only ints, but received {spec!r}.")
            bad = sorted({i for i in spec if i < 0 or i >= num_units})
            if bad:
                raise ValueError(
                    f"'unfreeze={spec}' contains invalid timm {container_name} indices {bad}. "
                    f"Valid indices for backbone layer {outer_idx} are 0 to {num_units - 1}."
                )
            return sorted(set(spec))

        raise TypeError(
            f"'unfreeze' only supports int or list[int] for timm backbones, but received {spec!r}."
        )

    def _apply_timm_unfreeze(self, timm_layers):
        """Selectively unfreeze the last few inner timm backbone units after the backbone has been frozen."""
        unfreeze_spec = self.args.unfreeze
        if not self._has_layer_spec(unfreeze_spec):
            return []

        if not timm_layers:
            raise ValueError(
                "'unfreeze' is only supported for models with a timm backbone. "
                "Remove 'unfreeze' or switch to a timm-backed model."
            )
        if len(timm_layers) > 1:
            raise ValueError(
                f"'unfreeze' is ambiguous for models with multiple timm backbones ({len(timm_layers)} found). "
                "Use a single timm backbone when applying selective unfreezing."
            )

        outer_idx, outer_prefix, param_prefix, timm_layer = timm_layers[0]
        trainable_non_lora = self._get_timm_trainable_non_lora_parameter_names(timm_layer)
        if trainable_non_lora:
            sample = ", ".join(trainable_non_lora[:3])
            suffix = " ..." if len(trainable_non_lora) > 3 else ""
            raise ValueError(
                f"'unfreeze' requires timm backbone layer {outer_idx} to be fully frozen first. "
                "Freeze the full timm backbone through trainer 'freeze' before applying selective unfreeze. "
                f"Found trainable non-LoRA timm parameters: {sample}{suffix}"
            )

        root_module, root_prefix, container_name, units = self._get_timm_unfreeze_context(timm_layer)
        selected = self._resolve_timm_unfreeze_indices(unfreeze_spec, len(units), outer_idx, container_name)
        if not selected:
            return []

        for idx in selected:
            for param in units[idx].parameters():
                param.requires_grad = True

        self._register_timm_train_mode_override(
            outer_idx=outer_idx,
            timm_layer=timm_layer,
            root_module=root_module,
            container_name=container_name,
            selected=selected,
        )

        LOGGER.info(
            f"Unfreezing timm backbone layer {outer_idx} {container_name} indices {selected} "
            f"(outer prefix '{outer_prefix}')."
        )
        LOGGER.warning(
            f"Timm backbone layer {outer_idx} is partially unfrozen. Trainer will restore train mode for the "
            f"selected {container_name} each epoch and keep the still-frozen timm submodules in eval mode."
        )
        return [f"{param_prefix}{root_prefix}{container_name}.{idx}." for idx in selected]

    def _normalize_lora_targets(self):
        """Normalize LoRA target names into a non-empty list of strings."""
        targets = self.args.lora_targets
        if isinstance(targets, str):
            normalized = [x.strip() for x in targets.split(",") if x.strip()]
        elif isinstance(targets, (list, tuple, set)):
            normalized = [str(x).strip() for x in targets if str(x).strip()]
        else:
            raise TypeError(
                f"'lora_targets' must be a comma-separated string or list[str], but received {targets!r}."
            )
        if not normalized:
            raise ValueError("'lora_targets' must contain at least one target module name.")
        return normalized

    def _resolve_timm_lora_indices(self, spec, num_units, outer_idx, container_name):
        """Resolve timm LoRA unit selection into concrete inner indices."""
        if isinstance(spec, int):
            if spec < 0:
                raise ValueError(f"'lora_layers' must be >= 0, but received {spec}.")
            if spec == 0:
                raise ValueError(
                    f"'lora_layers=0' would disable LoRA injection for timm backbone layer {outer_idx}. "
                    "Set 'lora=False' instead."
                )
            if spec > num_units:
                raise ValueError(
                    f"'lora_layers={spec}' exceeds the {num_units} available timm {container_name} "
                    f"in backbone layer {outer_idx}."
                )
            return list(range(num_units - spec, num_units))

        if isinstance(spec, range):
            spec = list(spec)
        if isinstance(spec, tuple):
            spec = list(spec)
        if isinstance(spec, list):
            if not spec:
                raise ValueError("'lora_layers=[]' is empty. Set 'lora=False' or select at least one inner layer.")
            if not all(isinstance(i, int) for i in spec):
                raise TypeError(f"'lora_layers' lists must contain only ints, but received {spec!r}.")
            bad = sorted({i for i in spec if i < 0 or i >= num_units})
            if bad:
                raise ValueError(
                    f"'lora_layers={spec}' contains invalid timm {container_name} indices {bad}. "
                    f"Valid indices for backbone layer {outer_idx} are 0 to {num_units - 1}."
                )
            return sorted(set(spec))

        raise TypeError(
            f"'lora_layers' only supports int or list[int] for timm backbones, but received {spec!r}."
        )

    def _register_timm_train_mode_override(self, outer_idx, timm_layer, root_module, container_name=None, selected=None):
        """Merge train-mode overrides for timm backbones affected by selective unfreezing or LoRA."""
        selected = None if selected is None else set(selected)
        existing = self._timm_train_mode_overrides.get(outer_idx)
        if existing is None:
            self._timm_train_mode_overrides[outer_idx] = {
                "layer": timm_layer,
                "root_module": root_module,
                "container_name": container_name,
                "selected": selected,
                "all_units": selected is None,
            }
            return

        if container_name and existing["container_name"] and existing["container_name"] != container_name:
            raise ValueError(
                f"Conflicting timm train-mode overrides for backbone layer {outer_idx}: "
                f"'{existing['container_name']}' vs '{container_name}'."
            )

        if container_name and existing["container_name"] is None:
            existing["container_name"] = container_name
        if root_module is not None:
            existing["root_module"] = root_module
        if selected is None or existing["all_units"]:
            existing["all_units"] = True
            existing["selected"] = None
        else:
            existing["selected"] = (existing["selected"] or set()) | selected

    def _configure_timm_lora(self, model):
        """Inject LoRA adapters into a single timm backbone before checkpoint weights are loaded."""
        if not self.args.lora:
            if self._has_layer_spec(self.args.lora_layers):
                raise ValueError("'lora_layers' requires 'lora=True'.")
            return model

        timm_layers = self._get_timm_backbone_layers(model)
        if not timm_layers:
            raise ValueError(
                "'lora=True' is only supported for models with a timm backbone. "
                "Remove 'lora' or switch to a timm-backed model."
            )
        if len(timm_layers) > 1:
            raise ValueError(
                f"'lora=True' is ambiguous for models with multiple timm backbones ({len(timm_layers)} found). "
                "Use a single timm backbone when applying LoRA."
            )

        rank = self.args.lora_rank
        alpha = self.args.lora_alpha
        dropout = self.args.lora_dropout
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"'lora_rank' must be a positive int, but received {rank!r}.")
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ValueError(f"'lora_alpha' must be > 0, but received {alpha!r}.")
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"'lora_dropout' must satisfy 0 <= dropout < 1, but received {dropout!r}.")

        outer_idx, _, _, timm_layer = timm_layers[0]
        if getattr(timm_layer, "_ultralytics_lora", None):
            return model

        root_module, root_prefix, container_name_for_layers, units = None, "", None, None
        try:
            root_module, root_prefix, container_name_for_layers, units = self._get_timm_unfreeze_context(timm_layer)
        except ValueError:
            if self._has_layer_spec(self.args.lora_layers):
                raise

        container_name = None
        selected = None
        if self._has_layer_spec(self.args.lora_layers):
            container_name = container_name_for_layers
            selected = self._resolve_timm_lora_indices(self.args.lora_layers, len(units), outer_idx, container_name)

        from ultralytics.nn.modules.lora import inject_lora_into_timm

        targets = self._normalize_lora_targets()
        summary = inject_lora_into_timm(
            timm_layer,
            rank=rank,
            alpha=float(alpha),
            dropout=float(dropout),
            target_modules=targets,
            container_name=container_name,
            unit_indices=selected,
            target_root=root_module,
        )
        if summary["num_matched"] == 0:
            selection = (
                f"{container_name} indices {selected}" if selected is not None else "the full wrapped timm backbone"
            )
            raise ValueError(
                f"LoRA did not match any nn.Linear modules in timm backbone layer {outer_idx} for targets "
                f"{targets} within {selection}."
            )

        timm_layer._ultralytics_lora = {
            "outer_idx": outer_idx,
            "container_name": container_name,
            "selected": None if selected is None else set(selected),
            "targets": targets,
            "num_matched": summary["num_matched"],
            "trainable_params": summary["trainable_params"],
        }
        model._ultralytics_lora_configured = True

        selection = (
            f"{container_name} indices {selected}" if selected is not None else "all matching inner linear modules"
        )
        LOGGER.info(
            f"Injected LoRA into timm backbone layer {outer_idx}: matched {summary['num_matched']} linear modules "
            f"({summary['trainable_params']:,} trainable params) for targets {targets} across {selection}."
        )
        LOGGER.warning(
            f"Timm backbone layer {outer_idx} contains LoRA adapters. Trainer will restore timm train mode each epoch "
            "so the wrapped backbone does not stay stuck in eval due to frozen base parameters."
        )
        return model

    def _sync_timm_lora_train_mode_overrides(self, timm_layers):
        """Register train-mode overrides for timm backbones that contain LoRA adapters."""
        for outer_idx, _, _, timm_layer in timm_layers:
            meta = getattr(timm_layer, "_ultralytics_lora", None)
            if not meta:
                continue

            self._register_timm_train_mode_override(
                outer_idx=outer_idx,
                timm_layer=timm_layer,
                root_module=self._get_timm_unfreeze_context(timm_layer)[0] if meta.get("container_name") else timm_layer.m,
                container_name=meta.get("container_name"),
                selected=meta.get("selected"),
            )

            non_lora_trainable = any(
                param.requires_grad
                for name, param in timm_layer.m.named_parameters()
                if not self._is_lora_parameter_name(name) and not self._is_lora_base_parameter_name(name)
            )
            if non_lora_trainable and not self._has_layer_spec(self.args.unfreeze):
                LOGGER.warning(
                    f"LoRA adapters are active on timm backbone layer {outer_idx}, but non-LoRA timm parameters "
                    "remain trainable. Use trainer 'freeze' to freeze the full timm backbone for adapter-only "
                    "fine-tuning."
                )

    def _restore_timm_train_modes(self):
        """Override Timm.train() eval fallback for timm backbones with partial unfreeze and/or LoRA adapters."""
        for entry in getattr(self, "_timm_train_mode_overrides", {}).values():
            timm_layer = entry["layer"]
            root_module = entry["root_module"]
            container_name = entry["container_name"]

            # Re-enable train mode for the wrapped timm backbone, then selectively return frozen subtrees to eval.
            timm_layer.m.train()
            for name, child in root_module.named_children():
                if container_name and name == container_name:
                    units = list(getattr(root_module, container_name))
                    for unit in units:
                        if not any(param.requires_grad for param in unit.parameters()):
                            unit.eval()
                elif not any(param.requires_grad for param in child.parameters()):
                    child.eval()

    def _finalize_model_build(self, model, weights=None):
        """Apply model-level adapter configuration before loading checkpoint weights."""
        if not self.args.lora and self._has_layer_spec(self.args.lora_layers):
            raise ValueError("'lora_layers' requires 'lora=True'.")

        weights_have_lora = self._weights_have_lora(weights)
        if weights_have_lora and not self.args.lora:
            raise ValueError(
                "Checkpoint weights contain LoRA parameters, but 'lora=False'. "
                "Enable 'lora=True' to resume or fine-tune this checkpoint."
            )

        if self.args.lora and not getattr(model, "_ultralytics_lora_configured", False):
            self._configure_timm_lora(model)

        if weights is not None:
            model.load(weights)
        return model

    def _setup_train(self):
        """Build dataloaders and optimizer on correct rank process."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Compile model
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # Freeze layers
        freeze_list = self._resolve_layer_indices(self.args.freeze)
        timm_layers = self._get_timm_backbone_layers()
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        timm_param_prefixes = tuple(param_prefix for _, _, param_prefix, _ in timm_layers)
        self._timm_train_mode_overrides = {}
        if timm_layers and (self._has_layer_spec(self.args.freeze) or self._has_layer_spec(self.args.unfreeze)):
            LOGGER.warning(
                "Timm backbone detected. Trainer freeze/unfreeze args are authoritative for timm trainability. "
                "'freeze' still uses outer Ultralytics layer indices, while 'unfreeze' targets inner timm "
                "blocks/stages/layers."
            )

        self.freeze_layer_names = freeze_layer_names
        self.unfreeze_layer_names = []
        timm_reenabled = False
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                if self._is_lora_parameter_name(k):
                    v.requires_grad = True
                else:
                    LOGGER.info(f"Freezing layer '{k}'")
                    v.requires_grad = False
            elif timm_param_prefixes and k.startswith(timm_param_prefixes) and not v.requires_grad:
                if self._is_lora_base_parameter_name(k):
                    continue
                if not timm_reenabled:
                    LOGGER.info(
                        "Re-enabling pre-frozen timm backbone parameters so trainer freeze/unfreeze args remain "
                        "the single source of truth."
                    )
                    timm_reenabled = True
                v.requires_grad = True
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                if self._is_lora_base_parameter_name(k):
                    continue
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        self.unfreeze_layer_names = self._apply_timm_unfreeze(timm_layers)
        self._sync_timm_lora_train_mode_overrides(timm_layers)

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training
        # print("-"*50)
        # print("Inside trainer.py 307")
        # print("gs:", gs)
        # print("stride:", self.stride)
        # print("model.stride:", self.model.stride)
        # print("args.imgsz:", self.args.imgsz)
        # print("-"*50)

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self):
        """Train the model with the specified world size."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for {self._training_duration_description()}"
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            time_limit_reached = False
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    if self.args.compile:
                        # Decouple inference and loss calculations for improved compile performance
                        if getattr(unwrap_model(self.model), "uses_batch_dict", False):
                            loss, self.loss_items = self.model(batch, mode="loss")
                        else:
                            preds = self.model(batch["img"], metadata_vec=batch.get("metadata_vec"))
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                    else:
                        loss, self.loss_items = self.model(batch)
                    self.loss = loss.sum()
                    if RANK != -1:
                        self.loss *= self.world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        time_limit_reached |= self._time_exceeded()
                        if RANK != -1:  # if DDP training
                            broadcast_list = [time_limit_reached if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast timed stop state to all ranks
                            time_limit_reached = broadcast_list[0]

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                time_limit_reached |= self._time_exceeded()
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop or time_limit_reached:
                    self._clear_memory(threshold=0.5)  # prevent VRAM spike
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch or time_limit_reached

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self._should_adjust_epochs_for_time():
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory(0.5)  # clear if memory utilization > 50%

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using polars."""
        import polars as pl  # scope for faster 'import ultralytics'

        return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        self._restore_timm_train_modes()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if (
                any(f in n for f in self.freeze_layer_names)
                and not any(u in n for u in self.unfreeze_layer_names)
                and isinstance(m, nn.BatchNorm2d)
            ):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(unwrap_model(self.ema.ema)).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # save as dict
                "train_meta": {"explicit_epoch_limit": self._explicit_epoch_limit},
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'

    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] == "ndjson":
                # Convert NDJSON to YOLO format
                import asyncio

                from ultralytics.data.converter import convert_ndjson_to_yolo

                yaml_path = asyncio.run(convert_ndjson_to_yolo(self.args.data))
                self.args.data = str(yaml_path)
                data = check_det_dataset(self.args.data, True, self.args)
                # print("-"*50)
                # print("Inside base/trainer.py 651")
                # print(data)
                # print("-"*50)
            elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data, True, self.args)
                # print("-"*50)
                # print("Inside base/trainer.py 651")
                # print(data)
                # print("-"*50)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """
        Load, create, or download model for any task.

        Returns:
            (dict): Optional checkpoint to resume training from.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            self.model = self._finalize_model_build(self.model)
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allow custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Run validation on val set using self.validator.

        Returns:
            metrics (dict): Dictionary of validation metrics.
            fitness (float): Fitness score for the validation.
        """
        metrics = self.validator(self)
        
        # Calculate comparison metric based on configuration
        # if self.comparision_metric == "fitness":
        #     # Use the built-in fitness from metrics
        #     self.comp_metric = metrics.pop("fitness", -self.loss.detach().cpu().numpy())
        # elif self.comparision_metric == "f1":
        #     # Extract F1 from metrics (for segmentation tasks)
        #     self.comp_metric = metrics.get("metrics/f1(M)", -self.loss.detach().cpu().numpy())
        # elif self.comparision_metric == "f2":
        #     # Extract F2 from metrics (for segmentation tasks)
        #     self.comp_metric = metrics.get("metrics/f2(M)", -self.loss.detach().cpu().numpy())
        # elif self.comparision_metric == "custom":
            # Use custom fitness calculation with configured weights
        # fitness_weights = getattr(self.args, 'fitness_weights', None)
        # self.comp_metric = calculate_fitness(metrics, fitness_weights)
        # else:
        
        # Update best metric if current is better
        # if not self.best_comp_metric or self.best_comp_metric < self.comp_metric:
        #     self.best_comp_metric = copy(self.comp_metric)
        self.fitness = metrics.pop("fitness", None)
        if not self.best_fitness or self.best_fitness < self.fitness:
            self.best_fitness = copy(self.fitness)
            
        return metrics, copy(self.fitness)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Return a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Return dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plot training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time"] + keys)).rstrip(",") + "\n")  # header
        t = time.time() - self.train_time_start
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.validator.args.compile = False  # disable final val compile as too slow
                    self.metrics = self.validator(model=f)
                    # self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                _, ckpt = load_checkpoint(last)
                ckpt_args = {**DEFAULT_CFG_DICT, **ckpt.get("train_args", {})}
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self._explicit_epoch_limit = ckpt.get("train_meta", {}).get(
                    "explicit_epoch_limit",
                    self._explicit_epoch_limit or (bool(ckpt_args.get("time")) and ckpt_args.get("epochs") != DEFAULT_CFG.epochs),
                )
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            self.best_fitness = ckpt["best_fitness"]
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif (
                    isinstance(module, bn)
                    or "logit_scale" in fullname
                    or param_name.startswith("lora_")
                ):  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if not any(g):
            raise ValueError("No trainable parameters remain after freeze/LoRA configuration. Adjust your training args.")

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer
