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
        ["bash", "jobs/train/hpc/bash_scripts_joint/mamba_models_448_pn10075s.sh", "4", "4", "5", "0", "0"],
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


def test_joint_mamba_launcher_includes_edgevss_variants(tmp_path: Path) -> None:
    """Smoke-test the joint Mamba launcher after adding EdgeVSS variants."""
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
        ["bash", "jobs/train/hpc/bash_scripts_joint/mamba_models_448_pn10075s.sh", "4", "4", "5", "0", "0"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    calls = _parse_call_log(qsub_log)
    assert len(calls) == 9

    config_paths = {_parse_varlist(call)["CONFIG_YAML"] for call in calls}
    expected = {
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml",
        "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-obb.yaml",
        "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo-edgevss.yaml",
        "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-obb-edgevss.yaml",
        "ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml",
        "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml",
        "ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg-edgevss-backbone.yaml",
        "ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg-edgevss-all.yaml",
        "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg-edgevss.yaml",
    }

    assert config_paths == expected
    for config_yaml in config_paths:
        assert "/Users/manishagupta/Desktop/PhD/Code" not in config_yaml
        assert Path(REPO_ROOT / config_yaml).is_file()


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


def test_mamba_yolo_submitters_resolve_local_configs(tmp_path: Path) -> None:
    """Smoke-test the Mamba-YOLO submitters with aliases, paths, and dry-run flows."""
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
            ["bash", "jobs/train/hpc/bash_scripts_seg/submit_yolo_mamba_seg.sh", "yolo-mamba-seg", "224", "8", "pn10075s", "0"],
            "segment",
            "mamba_yolo_segment",
            "ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml",
            "21",
        ),
        (
            ["bash", "jobs/train/hpc/bash_scripts_seg/submit_yolo_mamba_seg.sh", "mamba-hrnet-seg", "448", "8", "pn10075s", "0"],
            "segment",
            "mamba_yolo_segment",
            "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml",
            "21",
        ),
        (
            [
                "bash",
                "jobs/train/hpc/bash_scripts_obb/submit_mamba_yolo_obb.sh",
                str(REPO_ROOT / "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml"),
                "256",
                "8",
                "0",
                "0",
            ],
            "obb",
            "mamba_yolo_obb",
            "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L-obb-demo.yaml",
            "0",
        ),
        (
            ["bash", "jobs/train/hpc/bash_scripts_obb/submit_mamba_yolo_obb.sh", "mamba-hrnet-obb", "448", "8", "21", "0"],
            "obb",
            "mamba_yolo_obb",
            "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-obb.yaml",
            "21",
        ),
    ]

    for cmd, _, _, _, _ in cases:
        subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    dry_run = subprocess.run(
        ["bash", "jobs/train/hpc/bash_scripts_obb/submit_mamba_yolo_obb.sh", "mamba-yolo-l-obb", "224", "16", "21", "1"],
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


def test_shoreline_segment_submitter_passes_requested_variants(tmp_path: Path) -> None:
    """Smoke-test the fixed shoreline segment launcher with the requested knobs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qsub_log = tmp_path / "qsub.log"
    scratch = tmp_path / "scratch"

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
    env["SCRATCH"] = str(scratch)
    env["RUN_TAG"] = "testrun"

    subprocess.run(
        ["bash", "jobs/train/hpc/bash_scripts_seg/submit_shoreline_segment_models.sh", "8", "100", "0", "0"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    calls = _parse_call_log(qsub_log)
    assert len(calls) == 10

    expected_data = (
        scratch
        / "data_processed/Global/Annotated/variants/segment/"
        / "planet_full_c448_ov35_kf20_10075-single_sh-lw-d-prx-cl-hz-sdw_seed0/data.yaml"
    )
    by_name = {call[call.index("-N") + 1]: _parse_varlist(call) for call in calls}

    expected_configs = {
        "yolo12n_seg_shore_lw_loss": "ultralytics/cfg/models/12/yolo12-seg.yaml",
        "yolo12n_seg_shore_lw_input": "ultralytics/cfg/models/12/yolo12-seg.yaml",
        "yolo12n_seg_shore_aux_head": "ultralytics/cfg/models/12/yolo12-seg-shoreaux.yaml",
        "yolo26n_seg_normal": "ultralytics/cfg/models/26/yolo26-seg.yaml",
        "yolo26n_seg_shore_lw_input": "ultralytics/cfg/models/26/yolo26-seg.yaml",
        "mamba_hrnet_seg_shore_lw_loss": "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml",
        "mamba_hrnet_seg_shore_lw_input": "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-seg.yaml",
        "mamba_hrnet_yolo26_seg_normal": "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-yolo26-seg.yaml",
        "mamba_hrnet_yolo26_seg_shore_lw_input": "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-yolo26-seg.yaml",
        "mamba_hrnet_cascade_mask_rcnn_normal": "ultralytics/cfg/models/mamba-yolo/mamba-hrnet-cascade-mask-rcnn.yaml",
    }
    assert set(by_name) == set(expected_configs)

    for name, config_yaml in expected_configs.items():
        vars_map = by_name[name]
        assert vars_map["TASK"] == "segment"
        assert vars_map["IMGSZ"] == "448"
        assert vars_map["EPOCHS"] == "100"
        assert vars_map["BATCH"] == "8"
        assert vars_map["DEVICE"] == "0"
        assert vars_map["PROJECT"] == "shoreline_segment_models"
        assert vars_map["DATA_YAML"] == str(expected_data)
        assert vars_map["CONFIG_YAML"] == config_yaml
        assert vars_map["EXPERIMENT_MODE"] == f"testrun_{name}"
        assert vars_map["SDICE"] == "1"
        assert vars_map["CLAHE_P"] == "0.0"
        assert vars_map["UNSHARP_P"] == "0.0"
        assert vars_map["GAUSSIAN_BLUR_P"] == "0.0"
        assert vars_map["MOTION_BLUR_P"] == "0.0"
        assert vars_map["MULTI_SPEC_NOISE_P"] == "0.0"
        assert vars_map["MOSAIC"] == "0.0"
        assert vars_map["MIXUP"] == "0.0"
        assert vars_map["COPY_PASTE"] == "0.0"
        assert vars_map["CLOSE_MOSAIC"] == "0"
        assert Path(REPO_ROOT / config_yaml).is_file()

    for name in ("yolo12n_seg_shore_lw_loss", "mamba_hrnet_seg_shore_lw_loss"):
        assert by_name[name]["USE_SHORELINE_PRIOR_LOSS"] == "true"
        assert by_name[name]["USE_LAND_WATER_PRIOR_LOSS"] == "true"
        assert by_name[name]["USE_SHORELINE_INPUT"] == "false"
        assert by_name[name]["USE_LAND_WATER_INPUT"] == "false"

    for name in (
        "yolo12n_seg_shore_lw_input",
        "yolo26n_seg_shore_lw_input",
        "mamba_hrnet_seg_shore_lw_input",
        "mamba_hrnet_yolo26_seg_shore_lw_input",
    ):
        assert by_name[name]["USE_SHORELINE_INPUT"] == "true"
        assert by_name[name]["USE_LAND_WATER_INPUT"] == "true"
        assert by_name[name]["USE_SHORELINE_PRIOR_LOSS"] == "false"
        assert by_name[name]["USE_LAND_WATER_PRIOR_LOSS"] == "false"

    aux_vars = by_name["yolo12n_seg_shore_aux_head"]
    assert aux_vars["USE_SHORELINE_AUX_LOSS"] == "true"
    assert aux_vars["SHORELINE_AUX_WARMUP_EPOCHS"] == "2"


def test_rhino_submitter_resolves_local_configs(tmp_path: Path) -> None:
    """Smoke-test the RHINO submitter with aliases, paths, and dry-run flows."""
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

    subprocess.run(
        ["bash", "jobs/train/hpc/bash_scripts_obb/submit_rhino.sh", "rhino-r50-obb", "448", "4", "pn10075s", "0"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "bash",
            "jobs/train/hpc/bash_scripts_obb/submit_rhino.sh",
            str(REPO_ROOT / "ultralytics/cfg/models/rhino/rhino-dinov3-augfpn-obb.yaml"),
            "448",
            "4",
            "0",
            "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    dry_run = subprocess.run(
        ["bash", "jobs/train/hpc/bash_scripts_obb/submit_rhino.sh", "rhino-dinov3-augfpn-obb", "448", "4", "21", "1"],
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
        assert vars_map["PROJECT"] == "rhino_obb"
        assert "/Users/manishagupta/Desktop/PhD/Code" not in vars_map["CONFIG_YAML"]
        assert Path(REPO_ROOT / vars_map["CONFIG_YAML"]).is_file()

    assert alias_vars["CONFIG_YAML"] == "ultralytics/cfg/models/rhino/rhino-r50-obb.yaml"
    assert alias_vars["MULTISPECTRAL"] == "21"
    assert path_vars["CONFIG_YAML"] == "ultralytics/cfg/models/rhino/rhino-dinov3-augfpn-obb.yaml"
    assert path_vars["MULTISPECTRAL"] == "0"

    assert "[DRY RUN] qsub -V -v" in dry_run.stdout
    assert len(_parse_call_log(qsub_log)) == 2


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
    env_base["USE_SHORELINE_INPUT"] = "true"
    env_base["USE_LAND_WATER_INPUT"] = "true"
    env_base["USE_SHORELINE_PRIOR_LOSS"] = "true"
    env_base["USE_LAND_WATER_PRIOR_LOSS"] = "true"
    env_base["USE_SHORELINE_AUX_LOSS"] = "true"
    env_base["SHORELINE_AUX_WARMUP_EPOCHS"] = "2"
    env_base["SDICE"] = "1"
    env_base["GAUSSIAN_BLUR_P"] = "0.25"
    env_base["MOSAIC"] = "0.0"
    env_base["MIXUP"] = "0.0"
    env_base["COPY_PASTE"] = "0.0"
    env_base["CLOSE_MOSAIC"] = "0"
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
        assert train_map["time"] == "11"
        assert train_map["freeze"] == "1"
        assert train_map["lora"] == "true"
        assert train_map["lora_rank"] == "8"
        assert train_map["lora_alpha"] == "16"
        assert train_map["lora_dropout"] == "0.1"
        assert train_map["use_soft_ignore_band"] == "true"
        assert train_map["use_shoreline_input"] == "true"
        assert train_map["use_land_water_input"] == "true"
        assert train_map["use_shoreline_prior_loss"] == "true"
        assert train_map["use_land_water_prior_loss"] == "true"
        assert train_map["use_shoreline_aux_loss"] == "true"
        assert train_map["shoreline_aux_warmup_epochs"] == "2"
        assert train_map["seg_w_dice"] == "1"
        assert train_map["gaussian_blur_p"] == "0.25"
        assert train_map["mosaic"] == "0.0"
        assert train_map["mixup"] == "0.0"
        assert train_map["copy_paste"] == "0.0"
        assert train_map["close_mosaic"] == "0"

    auto_run = {k: v for k, v in (arg.split("=", 1) for arg in calls[1] if "=" in arg)}
    literal_run = {k: v for k, v in (arg.split("=", 1) for arg in calls[3] if "=" in arg)}
    null_run = {k: v for k, v in (arg.split("=", 1) for arg in calls[5] if "=" in arg)}

    assert auto_run["name"] == "resnet50-panet_adaptive-planet_full_c224_ov35_kf20_seed0"
    assert auto_run["project"] == str(scratch / "runs/cuda/segment/imgsz_224/yolo/demo_project")
    assert literal_run["name"] == "literal_name"
    assert literal_run["project"] == str(scratch / "runs/cuda/segment/imgsz_224/yolo/demo_project")
    assert null_run["name"].startswith("yolo_segment_")
    assert null_run["project"] == str(scratch / "runs/cuda/segment/imgsz_224/yolo")


def test_site_prediction_submitter_passes_site_and_img_dir(tmp_path: Path) -> None:
    """Smoke-test the site prediction submitter with live and dry-run flows."""
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

    checkpoint = tmp_path / "scratch" / "runs" / "cuda" / "obb" / "imgsz_448" / "yolo" / "project_x" / "demo_run" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["QSUB_LOG"] = str(qsub_log)
    env["PREDICT_MODE"] = "directory"
    env["WANDB"] = "false"
    env["BATCH"] = "2"
    env["IOU"] = "0.55"
    env["MAX_DET"] = "77"
    env["JOB_BATCH_SIZE"] = "0"

    subprocess.run(
        [
            "bash",
            "jobs/infer/hpc/submit_yolo_site_predict_json.sh",
            str(checkpoint),
            "Treachery",
            "visual/pngs/images_c448_ov35_kf20",
            "512",
            "0.33",
            "cpu",
            "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    dry_run = subprocess.run(
        [
            "bash",
            "jobs/infer/hpc/submit_yolo_site_predict_json.sh",
            str(checkpoint),
            "Treachery",
            "visual/pngs/images_c448_ov35_kf20",
            "640",
            "0.25",
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
    assert len(calls) == 1

    call = calls[0]
    vars_map = _parse_varlist(call)
    assert call[-1] == "jobs/infer/hpc/yolo_site_predict_json.pbs"
    assert vars_map["CHECKPOINT"] == str(checkpoint)
    assert vars_map["SITE_NAME"] == "Treachery"
    assert vars_map["IMG_DIR"] == "visual/pngs/images_c448_ov35_kf20"
    assert vars_map["IMGSZ"] == "512"
    assert vars_map["CONF"] == "0.33"
    assert vars_map["IOU"] == "0.55"
    assert vars_map["MAX_DET"] == "77"
    assert vars_map["BATCH"] == "2"
    assert vars_map["DEVICE"] == "cpu"
    assert vars_map["PREDICT_MODE"] == "directory"
    assert vars_map["WANDB"] == "false"

    assert "[DRY RUN] qsub -V -v" in dry_run.stdout
    assert len(_parse_call_log(qsub_log)) == 1


def test_site_prediction_submitter_splits_batches_into_multiple_jobs(tmp_path: Path) -> None:
    """Smoke-test batched PBS submission for site prediction export."""
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

    scratch = tmp_path / "scratch"
    checkpoint = scratch / "runs" / "cuda" / "obb" / "imgsz_448" / "yolo" / "project_x" / "demo_run" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")

    image_dir = scratch / "data_processed" / "Treachery" / "PSScene" / "visual" / "pngs" / "images_c448_ov35_kf20"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in ("a.png", "b.png", "c.png", "d.png", "e.png"):
        (image_dir / name).write_bytes(b"")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["QSUB_LOG"] = str(qsub_log)
    env["SCRATCH"] = str(scratch)
    env["PREDICT_MODE"] = "directory"
    env["WANDB"] = "true"
    env["BATCH"] = "4"
    env["JOB_BATCH_SIZE"] = "2"

    subprocess.run(
        [
            "bash",
            "jobs/infer/hpc/submit_yolo_site_predict_json.sh",
            str(checkpoint),
            "Treachery",
            "visual/pngs/images_c448_ov35_kf20",
            "512",
            "0.33",
            "cpu",
            "0",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    calls = _parse_call_log(qsub_log)
    assert len(calls) == 3

    expected_ranges = [("1", "0", "2"), ("2", "2", "4"), ("3", "4", "5")]
    for call, (batch_index, batch_start, batch_end) in zip(calls, expected_ranges):
        vars_map = _parse_varlist(call)
        assert call[-1] == "jobs/infer/hpc/yolo_site_predict_json.pbs"
        assert call[call.index("-N") + 1] == f"demo_run_Treachery_b{batch_index}"
        assert vars_map["IMG_DIR"] == "visual/pngs/images_c448_ov35_kf20"
        assert vars_map["BATCH"] == "4"
        assert vars_map["WANDB"] == "false"
        assert vars_map["JOB_BATCH_INDEX"] == batch_index
        assert vars_map["JOB_BATCH_START"] == batch_start
        assert vars_map["JOB_BATCH_END"] == batch_end


def test_site_prediction_submitter_can_disable_batching_from_command_line(tmp_path: Path) -> None:
    """Smoke-test that the positional batching flag can force a single qsub."""
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

    scratch = tmp_path / "scratch"
    checkpoint = scratch / "runs" / "cuda" / "obb" / "imgsz_448" / "yolo" / "project_x" / "demo_run" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")

    image_dir = scratch / "data_processed" / "Treachery" / "PSScene" / "visual" / "pngs" / "images_c448_ov35_kf20"
    image_dir.mkdir(parents=True, exist_ok=True)
    for name in ("a.png", "b.png", "c.png", "d.png", "e.png"):
        (image_dir / name).write_bytes(b"")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["QSUB_LOG"] = str(qsub_log)
    env["SCRATCH"] = str(scratch)
    env["PREDICT_MODE"] = "directory"
    env["WANDB"] = "true"
    env["BATCH"] = "4"
    env["JOB_BATCH_SIZE"] = "2"

    subprocess.run(
        [
            "bash",
            "jobs/infer/hpc/submit_yolo_site_predict_json.sh",
            str(checkpoint),
            "Treachery",
            "visual/pngs/images_c448_ov35_kf20",
            "512",
            "0.33",
            "cpu",
            "0",
            "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )

    calls = _parse_call_log(qsub_log)
    assert len(calls) == 1

    call = calls[0]
    vars_map = _parse_varlist(call)
    assert call[-1] == "jobs/infer/hpc/yolo_site_predict_json.pbs"
    assert call[call.index("-N") + 1] == "demo_run_Treachery"
    assert vars_map["WANDB"] == "true"
    assert "JOB_BATCH_INDEX" not in vars_map
    assert "JOB_BATCH_START" not in vars_map
    assert "JOB_BATCH_END" not in vars_map


def test_log_predictions_submitter_passes_expected_env_vars(tmp_path: Path) -> None:
    """Smoke-test the train/val/test prediction export submitter with live and dry-run flows."""
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

    checkpoint = tmp_path / "scratch" / "runs" / "cuda" / "segment" / "imgsz_448" / "yolo" / "project_x" / "demo_run" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"weights")

    data_yaml = tmp_path / "scratch" / "data_processed" / "Global" / "Annotated" / "variants" / "segment" / "planet_full_c448_ov35_kf20_seed0" / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("path: .\nval: val\ntest: test\n", encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["QSUB_LOG"] = str(qsub_log)
    env["WANDB_RUN_ID"] = "abc123"

    subprocess.run(
        [
            "bash",
            "jobs/infer/hpc/submit_log_predictions_to_wandb.sh",
            str(checkpoint),
            str(data_yaml),
            "cpu",
            "2",
            "512",
            "0.02",
            "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    dry_run = subprocess.run(
        [
            "bash",
            "jobs/infer/hpc/submit_log_predictions_to_wandb.sh",
            str(checkpoint),
            str(data_yaml),
            "0",
            "4",
            "448",
            "0.01",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    calls = _parse_call_log(qsub_log)
    assert len(calls) == 1

    call = calls[0]
    vars_map = _parse_varlist(call)
    assert call[-1] == "jobs/infer/hpc/log_predictions_to_wandb.pbs"
    assert vars_map["CHECKPOINT"] == str(checkpoint)
    assert vars_map["DATA"] == str(data_yaml)
    assert vars_map["DEVICE"] == "cpu"
    assert vars_map["BATCH"] == "2"
    assert vars_map["IMGSZ"] == "512"
    assert vars_map["CONF"] == "0.02"
    assert vars_map["WANDB_RUN_ID"] == "abc123"

    assert "[DRY RUN] qsub -V -v" in dry_run.stdout
    assert len(_parse_call_log(qsub_log)) == 1
