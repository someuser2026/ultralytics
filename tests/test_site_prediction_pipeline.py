from __future__ import annotations

import json
import argparse
import importlib.util
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
        "if '-d' in sys.argv[1:]: sys.exit('qsub: invalid option -- d')\n"
        "assert pathlib.Path.cwd() == pathlib.Path(os.environ['EXPECTED_QSUB_CWD'])\n"
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
        "EXPECTED_QSUB_CWD": str(REPO_ROOT),
        "DRY_RUN": "0",
    }


def run_pipeline(env):
    return subprocess.run(["bash", str(PIPELINE)], cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def read_calls(env):
    return [json.loads(line) for line in Path(env["QSUB_LOG"]).read_text().splitlines()]


@pytest.mark.parametrize("batch_size,batch_count", [(2, 3), (5000, 1)])
def test_pipeline_waits_for_all_successful_batches(pipeline_env, batch_size, batch_count):
    result = run_pipeline({**pipeline_env, "JOB_BATCH_SIZE": str(batch_size)})
    assert result.returncode == 0, result.stderr
    calls = read_calls(pipeline_env)
    assert len(calls) == batch_count + 1
    assert all(call[-1].endswith("/yolo_site_predict_json.pbs") for call in calls[:-1])
    combine = calls[-1]
    assert combine[-1].endswith("/combine_yolo_site_predict_json_batches.pbs")
    dependency = ":".join(f"{index}.server" for index in range(1, batch_count + 1))
    assert combine[combine.index("-W") + 1] == f"depend=afterok:{dependency}"
    assert f"EXPECTED_BATCHES={batch_count}" in combine[combine.index("-v") + 1]
    assert "EXPECTED_COUNT=5" in combine[combine.index("-v") + 1]


def test_submitter_uses_repo_directory_when_called_elsewhere(pipeline_env, tmp_path):
    result = subprocess.run(
        [
            "bash", str(REPO_ROOT / "jobs/infer/hpc/submit_yolo_site_predict_json.sh"),
            pipeline_env["CHECKPOINT"], pipeline_env["SITE_NAME"], pipeline_env["IMG_DIR"],
        ],
        cwd=tmp_path,
        env={**pipeline_env, "COMBINE_BATCHES": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(read_calls(pipeline_env)) == 4


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


@pytest.mark.parametrize('broken_setup', [False, True])
def test_deferred_combine_pbs_receipt_logging_and_frozen_environment(pipeline_env, tmp_path, monkeypatch, broken_setup):
    from PIL import Image

    helper_path = REPO_ROOT / 'jobs/infer/site_predict_job_control.py'
    spec = importlib.util.spec_from_file_location('compute_combine_helper', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    logging = Path(os.environ.get('SITE_PIPELINE_REPO', REPO_ROOT.parent.parent / 'rip-detection')) / (
        'Code/time_series_analysis/scripts/analysis/site_pipeline_job_logging.sh')
    assert logging.is_file(), 'Set SITE_PIPELINE_REPO to the pipeline checkout'
    image_root = Path(pipeline_env['SCRATCH']) / 'data_processed/Treachery/PSScene/visual/pngs_c448'
    for path in image_root.glob('*.png'):
        Image.new('RGB', (8, 6)).save(path)
    settings = tmp_path / 'scheduler.json'
    settings.write_text(json.dumps({'python': sys.executable, 'combine_resources': {
        'cpus': 2, 'pbs': ['select=1:ncpus=2:mem=64gb', 'walltime=11:30:00']}}))
    monkeypatch.setenv('SCRATCH', pipeline_env['SCRATCH'])
    args = argparse.Namespace(control=str(tmp_path / 'control'), ckpt=pipeline_env['CHECKPOINT'],
        site_name='Treachery', img_dir=pipeline_env['IMG_DIR'], scheduler_settings=str(settings),
        imgsz=448, conf=.01, iou=.45, device='0', predict_mode='per-image', job_batch_size=2,
        combine_execution='pbs-deferred')
    request = helper.build_submission_plan(args)
    for batch in request['batches']:
        records = {Path(item['path']).stem: {'task': 'segment', 'orig_shape': {'height': 6, 'width': 8},
            'predictions': []} for item in request['images'][batch['start']:batch['end']]}
        helper.atomic(Path(request['output']) / batch['node'] / 'predictions.json', {'count': len(records), 'predictions': records})
    job = helper.new_attempt(request, 'combine', 1)
    job.update(job_id='42.server', status='accepted', released=True)
    helper.atomic(Path(args.control) / 'request.json', request)
    helper.atomic(Path(args.control) / 'state.json', {'jobs': {'combine': [job]}})
    setup = tmp_path / 'environment.sh'
    if not broken_setup:
        setup.write_text('export SCRATCH=' + str(tmp_path / 'wrong-scratch') + '\n')
    mamba = tmp_path / 'micromamba'
    mamba.write_text('#!/bin/bash\nset -eu\n[[ "$1 $2 $3" == "run -n test-environment" ]]\nshift 3\nexec "$@"\n')
    mamba.chmod(0o755)
    env = {**pipeline_env, 'MAMBA_EXE': str(mamba), 'SITE_PREDICT_CONTROL': args.control,
        'SITE_PREDICT_PYTHON': sys.executable, 'SITE_PREDICT_ENVIRONMENT': 'test-environment',
        'SITE_PREDICT_ENV_SETUP': str(setup), 'SITE_PIPELINE_JOB_LOGGING': str(logging),
        'SITE_PREDICT_OUTPUT_DIR': request['output'], 'SITE_PREDICT_NODE': 'combine',
        'SITE_PREDICT_ATTEMPT': '1', 'SITE_PREDICT_TOKEN': job['token'], 'SITE_PREDICT_REQUEST': request['hash'],
        'PBS_JOBID': job['job_id'], 'EXPECTED_BATCHES': str(len(request['batches'])), 'EXPECTED_COUNT': '5',
        'PYTHONDONTWRITEBYTECODE': '1'}
    command = ['bash', str(REPO_ROOT / 'jobs/infer/hpc/combine_yolo_site_predict_json_batches.pbs')]
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    logs = Path(args.control) / 'logs/executions/combine'
    exit_log = next(logs.glob('*.exit'))
    assert int(exit_log.read_text()) == result.returncode
    assert 'host=' in next(logs.glob('*.out')).read_text()
    if broken_setup:
        assert result.returncode != 0
        assert 'Environment setup is unreadable' in next(logs.glob('*.err')).read_text()
        assert not (Path(request['output']) / 'predictions.json').exists()
        return
    assert result.returncode == 0, result.stderr
    output = Path(request['output']) / 'predictions.json'
    receipt_path = Path(args.control) / 'receipts' / (job['token'] + '.json')
    receipt = helper.read(receipt_path)
    assert receipt['exit_status'] == 0 and receipt['state'] == 'finished'
    assert receipt['validation'] == {'count': 5, 'sha256': helper.file_hash(output)}
    assert len(helper.read(output)['predictions']) == 5
    assert not (tmp_path / 'wrong-scratch').exists()
    repeated = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert helper.read(receipt_path) == receipt
    assert len(list(logs.glob('*.exit'))) == 2
