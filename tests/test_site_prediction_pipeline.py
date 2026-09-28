from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "jobs/infer/hpc/yolo_site_predict_json_pipeline.pbs"


@pytest.fixture
def pipeline_env(tmp_path):
    scratch = tmp_path / "scratch"
    checkpoint = scratch / "runs/segment/demo/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    images = scratch / "data_processed/Treachery/PSScene/visual/pngs_c448"
    images.mkdir(parents=True)
    for index in range(5):
        (images / f"{index}.png").touch()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qsub = bin_dir / "qsub"
    qsub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "log = pathlib.Path(os.environ['QSUB_LOG'])\n"
        "lines = log.read_text().splitlines() if log.exists() else []\n"
        "with log.open('a') as handle: handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "index = len(lines) + 1\n"
        "if str(index) == os.environ.get('FAIL_BATCH'): sys.exit(1)\n"
        "print('invalid job id' if os.environ.get('BAD_JOB_ID') else f'{index}.server')\n"
    )
    qsub.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CHECKPOINT": str(checkpoint),
        "SCRATCH": str(scratch),
        "SITE_NAME": "Treachery",
        "IMG_DIR": "visual/pngs_c448",
        "JOB_BATCH_SIZE": "2",
        "PBS_O_WORKDIR": str(REPO_ROOT),
        "QSUB_LOG": str(tmp_path / "qsub.jsonl"),
        "DRY_RUN": "0",
    }


def run_pipeline(env):
    return subprocess.run(["bash", str(PIPELINE)], cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def read_calls(env):
    return [json.loads(line) for line in Path(env["QSUB_LOG"]).read_text().splitlines()]


def test_pipeline_waits_for_all_successful_batches(pipeline_env):
    result = run_pipeline(pipeline_env)
    assert result.returncode == 0, result.stderr
    calls = read_calls(pipeline_env)
    assert len(calls) == 4
    assert all(call[-1].endswith("/yolo_site_predict_json.pbs") for call in calls[:3])
    combine = calls[-1]
    assert combine[-1].endswith("/combine_yolo_site_predict_json_batches.pbs")
    assert combine[combine.index("-W") + 1] == "depend=afterok:1.server:2.server:3.server"
    assert "EXPECTED_BATCHES=3" in combine[combine.index("-v") + 1]
    assert "EXPECTED_COUNT=5" in combine[combine.index("-v") + 1]


def test_pipeline_dry_run_does_not_submit(pipeline_env):
    result = run_pipeline({**pipeline_env, "DRY_RUN": "1"})
    assert result.returncode == 0, result.stderr
    assert not Path(pipeline_env["QSUB_LOG"]).exists()
    assert result.stdout.count("[DRY RUN] qsub") == 4
    assert "depend=afterok:1.dryrun:2.dryrun:3.dryrun" in result.stdout


@pytest.mark.parametrize("override,call_count", [({"FAIL_BATCH": "2"}, 2), ({"BAD_JOB_ID": "1"}, 1)])
def test_pipeline_stops_on_submission_failure(pipeline_env, override, call_count):
    result = run_pipeline({**pipeline_env, **override})
    assert result.returncode != 0
    calls = read_calls(pipeline_env)
    assert len(calls) == call_count
    assert all(call[-1].endswith("/yolo_site_predict_json.pbs") for call in calls)


def test_combine_pbs_executes_combiner(pipeline_env, tmp_path):
    run_dir = Path(pipeline_env["SCRATCH"]) / "data_processed/Treachery/predictions/segment/demo__visual__pngs_c448"
    for index in (1, 2):
        batch_dir = run_dir / f"batch_{index}"
        batch_dir.mkdir(parents=True)
        (batch_dir / "predictions.json").write_text(json.dumps({"predictions": {f"{index}.png": {}}}))
    mamba = tmp_path / "micromamba"
    mamba.write_text('#!/bin/bash\nset -euo pipefail\nshift 3\nexec "$@"\n')
    mamba.chmod(0o755)
    env = {
        **pipeline_env,
        "MAMBA_EXE": str(mamba),
        "EXPECTED_BATCHES": "2",
        "EXPECTED_COUNT": "2",
        "PATH": f"{Path(sys.executable).parent}:{pipeline_env['PATH']}",
    }
    result = subprocess.run(
        ["bash", "jobs/infer/hpc/combine_yolo_site_predict_json_batches.pbs"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((run_dir / "predictions.json").read_text())
    assert payload == {"count": 2, "predictions": {"1": {}, "2": {}}}
