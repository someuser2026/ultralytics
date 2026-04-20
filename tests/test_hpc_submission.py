from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_stub(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _parse_call_log(path: Path) -> List[List[str]]:
    calls: List[List[str]] = []
    current: Optional[List[str]] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "CALL":
            current = []
        elif line == "END":
            if current is not None:
                calls.append(current)
                current = None
        elif current is not None:
            current.append(line)
    return calls


def _parse_varlist(call: List[str]) -> Dict[str, str]:
    varlist = call[call.index("-v") + 1]
    values = {}
    for item in varlist.split(","):
        key, value = item.split("=", 1)
        values[key] = value
    return values


def test_hpc_wrappers_submit_local_configs(tmp_path: Path) -> None:
    """Smoke-test representative migrated wrappers with a stubbed qsub."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qsub_log = tmp_path / "qsub.log"

    _write_stub(
        bin_dir / "qsub",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{\n"
        "  echo CALL\n"
        "  for arg in \"$@\"; do\n"
        "    printf '%s\\n' \"$arg\"\n"
        "  done\n"
        "  echo END\n"
        "} >> \"$QSUB_LOG\"\n",
    )
    _write_stub(bin_dir / "sleep", "#!/usr/bin/env bash\nexit 0\n")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["QSUB_LOG"] = str(qsub_log)

    scripts = [
        ["bash", "jobs/train/hpc/bash_scripts_seg/final/final_backbone.sh", "224", "8", "0", "5"],
        ["bash", "jobs/train/hpc/bash_scripts_obb/final_backbone_neck.sh", "224", "16", "0"],
        ["bash", "jobs/train/hpc/bash_scripts_joint/model_sweep_fixed_data_tile.sh", "224", "0", "4", "4", "5", "0"],
        ["bash", "jobs/train/hpc/bash_scripts_joint/backbone_sweep.sh", "obb_base", "224", "16", "0"],
        ["bash", "jobs/train/hpc/bash_scripts_joint/neck_sweep.sh", "seg_non512", "224", "8"],
        ["bash", "jobs/train/hpc/bash_scripts_joint/lora_dinov3_1cls_pn10075s.sh", "224", "4", "4", "5", "0"],
        ["bash", "jobs/train/hpc/bash_scripts_joint/unfreeze_dinov3_1cls_pn10075s.sh", "224", "4", "4", "5", "0"],
    ]

    for cmd in scripts:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    calls = _parse_call_log(qsub_log)
    assert calls, "expected qsub calls from migrated wrappers"

    config_paths = []
    saw_lora = False
    saw_unfreeze = False
    for call in calls:
        assert call[-1] == "jobs/train/hpc/planet_full.pbs"
        vars_map = _parse_varlist(call)
        config_yaml = vars_map["CONFIG_YAML"]
        config_paths.append(config_yaml)
        assert "/Users/manishagupta/Desktop/PhD/Code" not in config_yaml
        assert not config_yaml.startswith("configs/ultralytics")
        assert Path(REPO_ROOT / config_yaml).is_file()
        if "LORA" in vars_map:
            saw_lora = True
        if "UNFREEZE" in vars_map:
            saw_unfreeze = True

    assert any(path.startswith("ultralytics/cfg/models/timm/") for path in config_paths)
    assert any(path.startswith("ultralytics/cfg/models/yolo/") for path in config_paths)
    assert saw_lora
    assert saw_unfreeze


def test_rotatedfcos_submitter_resolves_local_configs(tmp_path: Path) -> None:
    """Smoke-test the RotatedFCOS submitter with alias, path, and dry-run flows."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qsub_log = tmp_path / "qsub.log"

    _write_stub(
        bin_dir / "qsub",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{\n"
        "  echo CALL\n"
        "  for arg in \"$@\"; do\n"
        "    printf '%s\\n' \"$arg\"\n"
        "  done\n"
        "  echo END\n"
        "} >> \"$QSUB_LOG\"\n",
    )
    _write_stub(bin_dir / "sleep", "#!/usr/bin/env bash\nexit 0\n")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["QSUB_LOG"] = str(qsub_log)

    subprocess.run(
        ["bash", "jobs/train/hpc/bash_scripts_obb/submit_rotatedfcos.sh", "legnet-small-fcos", "224", "16", "pn10075s", "0"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "bash",
            "jobs/train/hpc/bash_scripts_obb/submit_rotatedfcos.sh",
            str(REPO_ROOT / "ultralytics/cfg/models/legnet/legnet-small-fcos-smallobj.yaml"),
            "256",
            "8",
            "0",
            "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    dry_run = subprocess.run(
        ["bash", "jobs/train/hpc/bash_scripts_obb/submit_rotatedfcos.sh", "legnet-small-fcos-smallobj", "224", "16", "21", "1"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = _parse_call_log(qsub_log)
    assert len(calls) == 2

    alias_vars = _parse_varlist(calls[0])
    path_vars = _parse_varlist(calls[1])

    for call, vars_map in zip(calls, [alias_vars, path_vars]):
        assert call[-1] == "jobs/train/hpc/planet_full.pbs"
        assert vars_map["TASK"] == "obb"
        assert vars_map["PROJECT"] == "rotatedfcos"
        assert "/Users/manishagupta/Desktop/PhD/Code" not in vars_map["CONFIG_YAML"]
        assert Path(REPO_ROOT / vars_map["CONFIG_YAML"]).is_file()

    assert alias_vars["CONFIG_YAML"] == "ultralytics/cfg/models/legnet/legnet-small-fcos.yaml"
    assert alias_vars["MULTISPECTRAL"] == "21"

    assert path_vars["CONFIG_YAML"] == "ultralytics/cfg/models/legnet/legnet-small-fcos-smallobj.yaml"
    assert path_vars["MULTISPECTRAL"] == "0"

    assert "[DRY RUN] qsub -V -v" in dry_run.stdout
    assert len(_parse_call_log(qsub_log)) == 2


def test_rcnn_submitters_resolve_smallobj_configs(tmp_path: Path) -> None:
    """Smoke-test the RCNN small-object submitters with aliases, paths, and dry-run flows."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qsub_log = tmp_path / "qsub.log"

    _write_stub(
        bin_dir / "qsub",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{\n"
        "  echo CALL\n"
        "  for arg in \"$@\"; do\n"
        "    printf '%s\\n' \"$arg\"\n"
        "  done\n"
        "  echo END\n"
        "} >> \"$QSUB_LOG\"\n",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["QSUB_LOG"] = str(qsub_log)

    cases = [
        (
            ["bash", "jobs/train/hpc/bash_scripts_obb/submit_oriented_rcnn.sh", "oriented-rcnn-smallobj", "224", "8", "pn10075s", "0"],
            "obb",
            "rcnn_obb",
            "ultralytics/cfg/models/rcnn/oriented_rcnn_r50_fpn_le90_smallobj.yaml",
            "21",
        ),
        (
            ["bash", "jobs/train/hpc/bash_scripts_obb/submit_rotated_faster_rcnn.sh", "rotated-faster-rcnn-smallobj", "224", "8", "0", "0"],
            "obb",
            "rcnn_obb",
            "ultralytics/cfg/models/rcnn/rotated_faster_rcnn_unravelnet_fpn_le90_smallobj.yaml",
            "0",
        ),
        (
            ["bash", "jobs/train/hpc/bash_scripts_seg/submit_mask_rcnn.sh", "mask-rcnn-smallobj", "224", "4", "pn10075s", "0"],
            "segment",
            "rcnn_segment",
            "ultralytics/cfg/models/rcnn/mask_rcnn_r50_fpn_smallobj.yaml",
            "21",
        ),
        (
            ["bash", "jobs/train/hpc/bash_scripts_seg/submit_cascade_mask_rcnn.sh", "cascade-mask-rcnn-smallobj", "224", "4", "21", "0"],
            "segment",
            "rcnn_segment",
            "ultralytics/cfg/models/rcnn/cascade_mask_rcnn_r50_fpn_smallobj.yaml",
            "21",
        ),
    ]

    for cmd, _, _, _, _ in cases:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    dry_run = subprocess.run(
        [
            "bash",
            "jobs/train/hpc/bash_scripts_seg/submit_mask_rcnn.sh",
            str(REPO_ROOT / "ultralytics/cfg/models/rcnn/mask_rcnn_r50_fpn_smallobj.yaml"),
            "256",
            "2",
            "0",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = _parse_call_log(qsub_log)
    assert len(calls) == len(cases)

    for call, (_, task, project, config_yaml, multispectral) in zip(calls, cases):
        assert call[-1] == "jobs/train/hpc/planet_full.pbs"
        vars_map = _parse_varlist(call)
        assert vars_map["TASK"] == task
        assert vars_map["PROJECT"] == project
        assert vars_map["CONFIG_YAML"] == config_yaml
        assert vars_map["MULTISPECTRAL"] == multispectral
        assert "/Users/manishagupta/Desktop/PhD/Code" not in vars_map["CONFIG_YAML"]
        assert Path(REPO_ROOT / vars_map["CONFIG_YAML"]).is_file()

    assert "[DRY RUN] qsub -V -v" in dry_run.stdout
    assert len(_parse_call_log(qsub_log)) == len(cases)


def test_planet_full_pbs_builds_native_yolo_command(tmp_path: Path) -> None:
    """Smoke-test the PBS wrapper with stubbed micromamba and yolo commands."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scratch = tmp_path / "scratch"
    dataset_dir = scratch / "data_processed/Global/Annotated/variants/segment/planet_full_c224_ov35_kf20_seed0"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "data.yaml").write_text("path: .\ntrain: train\nval: val\n", encoding="utf-8")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_text("stub", encoding="utf-8")
    yolo_log = tmp_path / "yolo.log"

    _write_stub(
        bin_dir / "micromamba",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"$1\" == \"shell\" && \"$2\" == \"hook\" ]]; then\n"
        "cat <<'EOF'\n"
        "micromamba() {\n"
        "  if [[ \"$1\" == \"activate\" ]]; then\n"
        "    export CONDA_DEFAULT_ENV=\"$2\"\n"
        "    return 0\n"
        "  fi\n"
        "  return 0\n"
        "}\n"
        "EOF\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
    )
    _write_stub(
        bin_dir / "yolo",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{\n"
        "  echo CALL\n"
        "  for arg in \"$@\"; do\n"
        "    printf '%s\\n' \"$arg\"\n"
        "  done\n"
        "  echo END\n"
        "} >> \"$YOLO_LOG\"\n",
    )
    _write_stub(bin_dir / "nvidia-smi", "#!/usr/bin/env bash\nexit 0\n")

    env_base = os.environ.copy()
    env_base["PATH"] = f"{bin_dir}:{env_base['PATH']}"
    env_base["SCRATCH"] = str(scratch)
    env_base["MAMBA_EXE"] = str(bin_dir / "micromamba")
    env_base["MAMBA_ROOT_PREFIX"] = str(tmp_path / "mamba-root")
    env_base["YOLO_LOG"] = str(yolo_log)
    env_base["PBS_O_WORKDIR"] = str(REPO_ROOT)
    env_base["PBS_JOBID"] = "123.server"
    env_base["PBS_JOBNAME"] = "test-hpc"
    env_base["TASK"] = "segment"
    env_base["IMGSZ"] = "224"
    env_base["CONFIG_YAML"] = "ultralytics/cfg/models/timm/segment/final/panet_adaptive/resnet/resnet50/no_p2/1cls/resnet50-panet_adaptive-segment.yaml"
    env_base["TIME_FLOAT"] = "null"
    env_base["EPOCHS"] = "5"
    env_base["DEVICE"] = "0"
    env_base["OVERLAP"] = "35"
    env_base["KEEP_FRAC"] = "20"
    env_base["BATCH"] = "4"
    env_base["CHECKPOINT"] = str(checkpoint)
    env_base["FREEZE"] = "1"
    env_base["LORA"] = "true"
    env_base["LORA_RANK"] = "8"
    env_base["LORA_ALPHA"] = "16"
    env_base["LORA_DROPOUT"] = "0.1"
    env_base["USE_SOFT_IGNORE"] = "true"
    env_base["SDICE"] = "1"
    env_base["GAUSSIAN_BLUR_P"] = "0.25"
    env_base["WANDB"] = "true"
    env_base["SEED"] = "0"
    env_base["MULTISPECTRAL"] = "0"
    env_base["PROJECT"] = "demo_project"

    cases = [
        ("auto", "demo_project", "resnet50-panet_adaptive-planet_full_c224_ov35_kf20_seed0"),
        ("literal_name", "demo_project", "literal_name"),
        ("null", "null", "yolo_segment_"),
    ]

    for experiment_mode, project, _ in cases:
        env = env_base.copy()
        env["EXPERIMENT_MODE"] = experiment_mode
        env["PROJECT"] = project
        subprocess.run(["bash", "jobs/train/hpc/planet_full.pbs"], cwd=REPO_ROOT, env=env, check=True)

    calls = _parse_call_log(yolo_log)
    assert len(calls) == 6

    for settings_call, train_call in zip(calls[0::2], calls[1::2]):
        assert settings_call == ["settings", "wandb=True"]
        train_map = {}
        for arg in train_call:
            if "=" in arg:
                key, value = arg.split("=", 1)
                train_map[key] = value
        assert train_map["task"] == "segment"
        assert train_map["mode"] == "train"
        assert train_map["model"] == str(checkpoint)
        assert train_map["data"] == str(dataset_dir / "data.yaml")
        assert train_map["freeze"] == "1"
        assert train_map["lora"] == "true"
        assert train_map["lora_rank"] == "8"
        assert train_map["lora_alpha"] == "16"
        assert train_map["lora_dropout"] == "0.1"
        assert train_map["use_soft_ignore_band"] == "true"
        assert train_map["seg_w_dice"] == "1"
        assert train_map["gaussian_blur_p"] == "0.25"

    auto_run = {k: v for k, v in (arg.split("=", 1) for arg in calls[1] if "=" in arg)}
    literal_run = {k: v for k, v in (arg.split("=", 1) for arg in calls[3] if "=" in arg)}
    null_run = {k: v for k, v in (arg.split("=", 1) for arg in calls[5] if "=" in arg)}

    assert auto_run["name"] == "resnet50-panet_adaptive-planet_full_c224_ov35_kf20_seed0"
    assert auto_run["project"] == str(scratch / "runs/cuda/segment/imgsz_224/yolo/demo_project")
    assert literal_run["name"] == "literal_name"
    assert literal_run["project"] == str(scratch / "runs/cuda/segment/imgsz_224/yolo/demo_project")
    assert null_run["name"].startswith("yolo_segment_")
    assert null_run["project"] == str(scratch / "runs/cuda/segment/imgsz_224/yolo")
