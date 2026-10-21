#!/usr/bin/env python3
"""Optional durable control for the existing site prediction PBS graph.

Standalone submitter behavior is unchanged unless SITE_PREDICT_CONTROL is set.
This module uses only the standard library and does not import model code.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import struct
import sys
import uuid

SCHEMA = 'site-predict-control-v1'
ROOT = Path(__file__).resolve().parents[2]
JOB_ID = re.compile(r'^\d+(?:\.[A-Za-z0-9_.-]+)?(?:@[A-Za-z0-9_.-]+)?$')
TERMINAL = {'F', 'C'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def lock(control):
    control = Path(control)
    control.mkdir(parents=True, exist_ok=True)
    with (control / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def load_request(control):
    request = read(Path(control) / 'request.json')
    if request.get('schema') != SCHEMA or request['control'] != str(Path(control).resolve()) or request['hash'] != digest({k: v for k, v in request.items() if k != 'hash'}):
        raise ValueError('Prediction request identity/hash mismatch')
    return request


def verify_sources(request):
    for path, expected in request['sources'].items():
        if file_hash(path) != expected:
            raise ValueError('Prediction source changed: ' + path)
    if file_hash(request['checkpoint']) != request['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')


def verify_images(request, images=None):
    current = sorted(str(p) for p in Path(request['image_root']).iterdir() if p.is_file() and p.suffix.lower() == '.png')
    if current != [r['path'] for r in request['images']]:
        raise ValueError('Prediction tile membership changed')
    for record in request['images'] if images is None else images:
        if file_hash(record['path']) != record['sha256']:
            raise ValueError('Prediction tile changed: ' + record['path'])


def build_submission_plan(args):
    control, checkpoint = Path(args.control).resolve(), Path(args.ckpt).resolve()
    scratch = Path(os.environ['SCRATCH']).resolve()
    if checkpoint.name != 'best.pt' or checkpoint.parent.name != 'weights' or 'segment' not in checkpoint.parts:
        raise ValueError('Managed prediction requires .../segment/.../<run>/weights/best.pt')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.site_name):
        raise ValueError('Unsafe site name')
    img_dir = Path(args.img_dir)
    if img_dir.is_absolute() or '..' in img_dir.parts or not img_dir.parts:
        raise ValueError('img-dir must be relative under PSScene')
    image_root = scratch / 'data_processed' / args.site_name / 'PSScene' / img_dir
    images = [{'path': str(p), 'sha256': file_hash(p)} for p in sorted(image_root.iterdir())
              if p.is_file() and p.suffix.lower() == '.png']
    if not images or args.job_batch_size < 1:
        raise ValueError('Nonempty images and positive job batch size required')
    output = scratch / 'data_processed' / args.site_name / 'predictions/segment' / (
        checkpoint.parent.parent.name + '__' + '__'.join(img_dir.parts))
    scripts = ROOT / 'jobs/infer'
    source_names = ['site_predict_job_control.py', 'yolo_site_predict_json.py', 'combine_yolo_site_predict_json_batches.py',
                    'hpc/submit_yolo_site_predict_json.sh', 'hpc/yolo_site_predict_json.pbs', 'hpc/combine_yolo_site_predict_json_batches.pbs']
    login_combine = getattr(args, 'combine_execution', 'pbs') == 'login'
    if login_combine:
        source_names.remove('hpc/combine_yolo_site_predict_json_batches.pbs')
    request = {'schema': SCHEMA, 'control': str(control), 'checkpoint': str(checkpoint),
               'checkpoint_sha256': file_hash(checkpoint), 'scratch': str(scratch), 'site': args.site_name,
               'img_dir': str(img_dir), 'image_root': str(image_root), 'images': images, 'output': str(output),
               'options': {k: getattr(args, k) for k in ('imgsz', 'conf', 'iou', 'device', 'predict_mode', 'job_batch_size')},
               'settings': read(args.scheduler_settings), 'owner': getpass.getuser(),
               'sources': {str(scripts / name): file_hash(scripts / name) for name in source_names},
               'batches': [{'node': f'batch_{i // args.job_batch_size + 1}', 'start': i,
                            'end': min(i + args.job_batch_size, len(images)), 'index': i // args.job_batch_size + 1}
                           for i in range(0, len(images), args.job_batch_size)]}
    # This is a quota guard, not a second scheduler. Normal sites use fewer than 50 jobs.
    if len(request['batches']) + 1 > 550:
        raise ValueError('Prediction graph exceeds the single-site quota; increase job batch size')
    if login_combine:
        request.update(combine_execution='login', max_attempts=1)
    request['hash'] = digest(request)
    return request


def new_attempt(request, node, number):
    token = digest([request['hash'], node, number])[:24]
    return {'node': node, 'attempt': number, 'token': token, 'job_name': 'sp' + token[:12],
            'job_id': None, 'status': 'pending', 'owner': request['owner']}


def run_pbs(request, command):
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)


def query(request):
    result = run_pbs(request, [request['settings'].get('qstat', 'qstat'), '-xf', '-F', 'json'])
    if result.returncode:
        raise RuntimeError('PBS query failed: ' + result.stderr)
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError('Incomplete PBS query; no absence inferred')
    jobs = payload.get('Jobs')
    if (jobs is None and set(payload) == {'timestamp', 'pbs_version', 'pbs_server'}
            and type(payload['timestamp']) is int and payload['timestamp'] > 0
            and all(isinstance(payload[k], str) and payload[k] for k in ('pbs_version', 'pbs_server'))
            and not result.stderr.strip()):
        jobs = {}
    if not isinstance(jobs, dict) or any(not isinstance(j, dict) for j in jobs.values()):
        raise ValueError('Incomplete PBS query; no absence inferred')
    return jobs


def variables(history):
    value = history.get('Variable_List', {})
    return value if isinstance(value, dict) else dict(x.split('=', 1) for x in value.split(',') if '=' in x)


def matching(request, job, jobs):
    matches = []
    for identifier, history in jobs.items():
        values = variables(history)
        if values.get('SITE_PREDICT_TOKEN') == job['token']:
            if (history.get('Job_Name') != job['job_name'] or str(history.get('Job_Owner', '')).split('@')[0] != job['owner']
                    or values.get('SITE_PREDICT_REQUEST') != request['hash']):
                raise ValueError('Prediction PBS identity mismatch')
            matches.append(identifier)
    if len(matches) > 1:
        raise ValueError('Duplicate PBS submission token')
    return matches


def observe(request, job, jobs):
    matches = matching(request, job, jobs)
    if matches and matches != [job['job_id']]:
        raise ValueError('PBS job ID/token disagreement')
    history = jobs.get(job['job_id'])
    if history is not None:
        if matches != [job['job_id']]:
            raise ValueError('PBS job ID belongs to a different submission')
        if history.get('job_state') not in TERMINAL:
            return None
    path = Path(request['control']) / 'receipts' / (job['token'] + '.json')
    receipt = read(path) if path.exists() else None
    if receipt and any(receipt.get(k) != v for k, v in identity(request, job).items()):
        raise ValueError('Prediction receipt identity mismatch')
    if job.get('terminal'):
        return job['terminal']
    if history is not None and history.get('Exit_status') is not None:
        code = int(history['Exit_status'])
    elif receipt and receipt.get('state') == 'finished' and type(receipt.get('exit_status')) is int:
        code = receipt['exit_status']
    else:
        raise RuntimeError('Prediction terminal state unknown; reconcile PBS/accounting: ' + str(job['job_id']))
    return {'exit_status': code, 'receipt': receipt, 'scheduler': history, 'at': now()}


def identity(request, job):
    return {'request_hash': request['hash'], **{k: job[k] for k in ('token', 'node', 'attempt', 'job_id')}}


def command_for(request, job, state):
    settings = request['settings']
    options = request['options']
    values = {'SCRATCH': request['scratch'], 'CHECKPOINT': request['checkpoint'], 'SITE_NAME': request['site'],
              'IMG_DIR': request['img_dir'], 'SITE_PREDICT_CONTROL': request['control'],
              'SITE_PREDICT_TOKEN': job['token'], 'SITE_PREDICT_REQUEST': request['hash'],
              'SITE_PREDICT_NODE': job['node'], 'SITE_PREDICT_ATTEMPT': str(job['attempt']),
              'SITE_PREDICT_PYTHON': settings.get('python', sys.executable),
              'SITE_PREDICT_ENVIRONMENT': settings.get('environment_name', 'ultralytics_contrib'),
              'SITE_PREDICT_ENV_SETUP': settings.get('environment_setup') or '',
              'SITE_PIPELINE_JOB_LOGGING': settings.get('site_pipeline_job_logging') or '',
              'MAMBA_EXE': settings.get('micromamba') or '', 'MAMBA_ROOT_PREFIX': settings.get('mamba_root_prefix') or ''}
    for key, value in options.items():
        if key != 'job_batch_size':
            values[key.upper()] = str(value)
    values['WANDB'] = 'false'
    combine = job['node'] == 'combine'
    if combine:
        values.update(EXPECTED_BATCHES=str(len(request['batches'])), EXPECTED_COUNT=str(len(request['images'])))
    else:
        batch = next(b for b in request['batches'] if b['node'] == job['node'])
        values.update(JOB_BATCH_INDEX=str(batch['index']), JOB_BATCH_START=str(batch['start']), JOB_BATCH_END=str(batch['end']))
    if any(any(c in value for c in ',\n\r') for value in values.values()):
        raise ValueError('PBS environment values must not contain commas or newlines')
    logs = Path(request['control']) / 'logs'
    logs.mkdir(exist_ok=True)
    prefix = logs / f'{job["node"]}-{job["attempt"]}'
    command = [settings.get('qsub', 'qsub'), '-h', '-r', 'n', '-N', job['job_name'],
               '-o', str(prefix) + '.out', '-e', str(prefix) + '.err', '-v', ','.join(k + '=' + v for k, v in values.items())]
    if combine:
        parents = [state['jobs'][b['node']][-1]['job_id'] for b in request['batches']]
        command += ['-W', 'depend=afterok:' + ':'.join(parents)]
    if settings.get('queue'):
        command += ['-q', settings['queue']]
    if settings.get('account'):
        command += [settings.get('account_flag', '-A'), settings['account']]
    script = 'combine_yolo_site_predict_json_batches.pbs' if combine else 'yolo_site_predict_json.pbs'
    return command + [str(ROOT / 'jobs/infer/hpc' / script)]


def submit_job(request, state, job):
    persist = lambda: atomic(Path(request['control']) / 'state.json', state)
    if job['status'] in {'submitting', 'ambiguous'}:
        matches = matching(request, job, query(request))
        if len(matches) != 1:
            raise RuntimeError('Submission remains ambiguous; use reconcile')
        job.update(job_id=matches[0], status='accepted')
        persist()
    if not job['job_id']:
        job.update(command=command_for(request, job, state), status='submitting')
        persist()
        try:
            result = run_pbs(request, job['command'])
        except (FileNotFoundError, PermissionError):
            job['status'] = 'pending'
            persist()
            raise
        except (OSError, subprocess.TimeoutExpired):
            job['status'] = 'ambiguous'
            persist()
            raise
        job['qsub_result'] = {'code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
        identifier = result.stdout.strip()
        if (job['node'] == 'combine' and result.returncode and not identifier
                and re.fullmatch(r'qsub: Unknown Job Id(?: Error)?[^\n]*', result.stderr.strip(), re.IGNORECASE)):
            # A specific server rejection can race with expiration after our pre-submit
            # query. Unknown acknowledgements and other errors still require reconciliation.
            jobs = query(request)
            parents = [state['jobs'][b['node']][-1] for b in request['batches']]
            if not matching(request, job, jobs) and any(p['job_id'] not in jobs for p in parents):
                for parent in parents:
                    if parent['job_id'] not in jobs:
                        parent['terminal'] = observe(request, parent, jobs)
                job.update(status='not_submitted', rejection='expired_dependency')
                state['combine_fallback'] = True
                persist()
                return
        if result.returncode or not JOB_ID.fullmatch(identifier):
            job['status'] = 'ambiguous'
            persist()
            raise RuntimeError('qsub did not unambiguously acknowledge submission')
        job.update(job_id=identifier, status='accepted')
        persist()
    if job['job_id'].isdigit():
        matches = matching(request, job, query(request))
        if len(matches) != 1 or matches[0].split('.')[0] != job['job_id']:
            raise RuntimeError('Cannot canonicalize PBS job ID')
        job['job_id'] = matches[0]
        persist()
    if job.get('released'):
        return
    jobs = query(request)
    matches = matching(request, job, jobs)
    history = jobs.get(job['job_id'])
    if history is None:
        observe(request, job, jobs)
    elif matches != [job['job_id']]:
        raise ValueError('Job identity mismatch before release')
    elif history.get('job_state') not in TERMINAL and 'u' in str(history.get('Hold_Types', '')):
        job['release_intent'] = True
        persist()
        result = run_pbs(request, [request['settings'].get('qrls', 'qrls'), job['job_id']])
        if result.returncode:
            raise RuntimeError('qrls outcome needs reconciliation: ' + result.stderr)
    job['released'] = True
    persist()


def submit_or_resume(args):
    control = Path(args.control).resolve()
    with lock(control):
        proposed = build_submission_plan(args)
        if (control / 'request.json').exists():
            request = load_request(control)
            if request != proposed:
                raise ValueError('Prediction submission options or frozen inputs changed')
        else:
            request = proposed
            output = Path(request['output'])
            if output.exists() and any(output.iterdir()):
                raise ValueError('Prediction output is nonempty and not owned by this run: ' + str(output))
            atomic(control / 'request.json', request)
        if not (control / 'state.json').exists():
            state = {'jobs': {name: [new_attempt(request, name, 1)] for name in
                             [b['node'] for b in request['batches']] + ([] if request.get('combine_execution') == 'login' else ['combine'])}, 'combine_fallback': False}
            atomic(control / 'state.json', state)
        state = read(control / 'state.json')
        for batch in request['batches']:
            submit_job(request, state, state['jobs'][batch['node']][-1])
        if request.get('combine_execution') == 'login':
            return {'control': str(control), 'batches': len(request['batches']), 'combine_execution': 'login'}
        combine = state['jobs']['combine'][-1]
        if not state['combine_fallback'] and not combine['job_id'] and combine['status'] == 'pending':
            jobs = query(request)
            parents = [state['jobs'][b['node']][-1] for b in request['batches']]
            if any(j['job_id'] not in jobs for j in parents):
                # Dependencies may have expired while the login controller was stopped.
                for job in parents:
                    if job['job_id'] not in jobs:
                        job['terminal'] = observe(request, job, jobs)
                state['combine_fallback'] = True
                atomic(control / 'state.json', state)
        if not state['combine_fallback']:
            submit_job(request, state, combine)
    return {'control': str(control), 'batches': len(request['batches'])}


def status(control):
    with lock(control):
        request, state = load_request(control), read(Path(control) / 'state.json')
        jobs = query(request)
        for attempts in state['jobs'].values():
            for job in attempts:
                if job['job_id']:
                    history = jobs.get(job['job_id'])
                    if request.get('combine_execution') == 'login' and history:
                        if matching(request, job, jobs) != [job['job_id']]:
                            raise ValueError('Prediction PBS ownership mismatch')
                        if int(history.get('run_count', 0) or 0) > 1 or (history.get('job_state') == 'H' and 's' in str(history.get('Hold_Types', ''))):
                            job['execution_failure'] = {'reason': 'PBS reexecution or system hold', 'scheduler': history}
                    result = observe(request, job, jobs)
                    if result is not None:
                        job['terminal'] = result
        atomic(Path(control) / 'state.json', state)
        return state


def abandon_combine(control):
    with lock(control):
        request, state = load_request(control), read(Path(control) / 'state.json')
        job = state['jobs']['combine'][-1]
        if job['status'] in {'submitting', 'ambiguous'}:
            submit_job(request, state, job)  # reconcile first; never assume an unacknowledged job is absent
        state['combine_fallback'] = True
        atomic(Path(control) / 'state.json', state)
        if not job['job_id']:
            return True
        jobs = query(request)
        evidence = observe(request, job, jobs)
        if evidence is not None:
            job['terminal'] = evidence
            atomic(Path(control) / 'state.json', state)
            return True
        result = run_pbs(request, [request['settings'].get('qdel', 'qdel'), job['job_id']])
        if result.returncode:
            raise RuntimeError('Combine cancellation unconfirmed: ' + result.stderr)
        return False


def retry_batch(control, node):
    with lock(control):
        request, state = load_request(control), read(Path(control) / 'state.json')
        if request.get('max_attempts') == 1:
            raise ValueError('Single-attempt managed prediction cannot be retried')
        if node not in {b['node'] for b in request['batches']}:
            raise ValueError('Only prediction batches can be retried')
        attempts = state['jobs'][node]
        if len(attempts) == 2:
            if attempts[-1].get('terminal', {}).get('exit_status', 0) != 0:
                raise RuntimeError('Batch exhausted its retry allowance')
            submit_job(request, state, attempts[-1])  # recover interruption during retry submission
            return
        prior = attempts[-1]
        evidence = observe(request, prior, query(request))
        if evidence is None or evidence['exit_status'] == 0:
            raise ValueError('Only a confirmed failed terminal batch can be retried')
        combine = state['jobs']['combine'][-1]
        if combine['job_id'] and observe(request, combine, query(request)) is None:
            raise RuntimeError('Wait for combine cancellation before retrying a batch')
        if not state['combine_fallback']:
            raise ValueError('Abandon the original combine dependency before retry')
        output = Path(request['output']) / node
        preserved = Path(control) / 'failed-outputs' / (node + '-attempt-1')
        if output.exists():
            preserved.parent.mkdir(exist_ok=True)
            if preserved.exists():
                raise ValueError('Both failed output and preserved output exist; inspect before retry')
            output.rename(preserved)
        attempts.append(new_attempt(request, node, 2))
        atomic(Path(control) / 'state.json', state)
        submit_job(request, state, attempts[-1])


def unique_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def validate_batch_output(request, node):
    batch = next(b for b in request['batches'] if b['node'] == node)
    expected = request['images'][batch['start']:batch['end']]
    verify_images(request, expected)
    payload = unique_json(Path(request['output']) / node / 'predictions.json')
    records = payload.get('predictions') if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        raise ValueError('Batch output lacks predictions')
    if payload.get('scientific_use') is False or payload.get('fixture_type') or payload.get('workflow_test'):
        raise ValueError('Synthetic prediction fixtures are forbidden')
    if any(not isinstance(value, dict) for value in records.values()):
        raise ValueError('Batch prediction records must be objects')
    stems = [Path(str(value.get('image_stem') or value.get('image_name') or key)).stem for key, value in records.items()]
    if len(set(stems)) != len(stems) or set(stems) != {Path(r['path']).stem for r in expected}:
        raise ValueError('Batch prediction coverage mismatch; inference skipped or duplicated tiles')
    if payload.get('count', len(records)) != len(records):
        raise ValueError('Batch count mismatch')
    expected_paths = {Path(r['path']).stem: Path(r['path']) for r in expected}
    for stem, record in zip(stems, records.values()):
        # Tiles were fully decoded before submission and are hash-checked above. Read
        # their PNG dimensions without importing the model or its scientific stack.
        with expected_paths[stem].open('rb') as stream:
            header = stream.read(24)
        if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
            raise ValueError('Unreadable PNG tile: ' + stem)
        width, height = struct.unpack('>II', header[16:24])
        if record.get('orig_shape') != {'height': height, 'width': width}:
            raise ValueError('Batch tile dimensions mismatch: ' + stem)
        values = record.get('predictions')
        task = str(record.get('task', 'segment')).lower()
        if not isinstance(values, list) or (task != 'segment' and not (task == 'detect' and values == [])):
            raise ValueError('Invalid segmentation record: ' + stem)
        for detection in values:
            try:
                segments = detection['segments']
                xs, ys = segments['x'], segments['y']
                if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys) or len(xs) < 3:
                    raise ValueError('Invalid polygon')
                xs, ys = list(map(float, xs)), list(map(float, ys))
                confidence, class_id = float(detection['confidence']), int(detection['class'])
                if (not math.isfinite(confidence) or not 0 <= confidence <= 1 or class_id < 0
                        or not all(math.isfinite(v) for v in xs + ys)
                        or any(v < 0 or v > width for v in xs) or any(v < 0 or v > height for v in ys)):
                    raise ValueError('Invalid polygon values')
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Invalid segmentation detection: ' + stem) from exc
    return {'count': len(records), 'sha256': file_hash(Path(request['output']) / node / 'predictions.json')}


def run_with_receipt(control, node, attempt, command):
    request, state = load_request(control), read(Path(control) / 'state.json')
    job = state['jobs'][node][attempt - 1]
    if (os.environ.get('PBS_JOBID') != job['job_id'] or os.environ.get('SITE_PREDICT_TOKEN') != job['token']
            or os.environ.get('SITE_PREDICT_REQUEST') != request['hash']):
        raise ValueError('Prediction worker identity mismatch')
    path = Path(control) / 'receipts' / (job['token'] + '.json')
    if path.exists():
        print('Duplicate execution rejected; preserving existing receipt: ' + str(path), file=sys.stderr, flush=True)
        raise RuntimeError('This prediction attempt already started')
    record = {**identity(request, job), 'state': 'started', 'started_at': now()}
    atomic(path, record)
    code, validation, error = 1, None, None
    try:
        verify_sources(request)
        if node != 'combine':
            batch = next(b for b in request['batches'] if b['node'] == node)
            verify_images(request, request['images'][batch['start']:batch['end']])
        # Activation hooks / PBS startup may replace SCRATCH after submission.
        # The frozen request owns both input and output paths for every worker,
        # including the combiner. Restore it at the child-process boundary.
        worker_env = os.environ.copy()
        worker_env['SCRATCH'] = request['scratch']
        print(f"Prediction {node}: SCRATCH={request['scratch']} "
              f"(inherited {os.environ.get('SCRATCH', '<unset>')})", flush=True)
        code = subprocess.run(command, cwd=ROOT, env=worker_env).returncode
        if code == 0 and node != 'combine':
            code = 1
            validation = validate_batch_output(request, node)
            code = 0
        return code
    except BaseException as exc:
        code, error = 1, str(exc)
        import traceback
        record['traceback'] = traceback.format_exc()
        raise
    finally:
        atomic(path, {**record, 'state': 'finished', 'exit_status': code,
                      'finished_at': now(), 'validation': validation, 'error': error})


def reconcile_submission(control, node, job_id=None, confirmed_not_submitted=False, reason=None):
    with lock(control):
        request, state = load_request(control), read(Path(control) / 'state.json')
        job = state['jobs'][node][-1]
        matches = matching(request, job, query(request))
        if job['job_id'] or job['status'] not in {'submitting', 'ambiguous'}:
            raise ValueError('Only an unacknowledged submission can be reconciled')
        if confirmed_not_submitted:
            if matches or not reason:
                raise ValueError('Explicit reason and no matching PBS job required')
            job.update(status='pending', reconciliation=reason)
        else:
            if len(matches) != 1 or job_id != matches[0]:
                raise ValueError('Supplied job does not match the submission token')
            job.update(job_id=job_id, status='accepted')
        atomic(Path(control) / 'state.json', state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    submit = subs.add_parser('submit')
    submit.add_argument('--control', required=True)
    for name in ('ckpt', 'site-name', 'img-dir', 'scheduler-settings'):
        submit.add_argument('--' + name, required=True)
    for name, default, kind in (('imgsz', 448, int), ('conf', .01, float), ('iou', .45, float), ('job-batch-size', 5000, int)):
        submit.add_argument('--' + name, type=kind, default=default)
    submit.add_argument('--combine-execution', choices=['pbs', 'login'], default='pbs')
    submit.add_argument('--device', default='0')
    submit.add_argument('--predict-mode', default='per-image', choices=['per-image', 'directory'])
    for name in ('status', 'abandon-combine', 'retry', 'run-worker', 'reconcile'):
        p = subs.add_parser(name)
        p.add_argument('control')
        if name in ('retry', 'run-worker', 'reconcile'):
            p.add_argument('node')
        if name == 'run-worker':
            p.add_argument('attempt', type=int)
            p.add_argument('worker_command', nargs=argparse.REMAINDER)
        if name == 'reconcile':
            choices = p.add_mutually_exclusive_group(required=True)
            choices.add_argument('--job-id')
            choices.add_argument('--confirmed-not-submitted', action='store_true')
            p.add_argument('--reason')
    args = parser.parse_args(argv)
    if args.command == 'submit':
        result = submit_or_resume(args)
    elif args.command == 'status':
        result = status(args.control)
    elif args.command == 'retry':
        result = retry_batch(args.control, args.node)
    elif args.command == 'abandon-combine':
        result = abandon_combine(args.control)
    elif args.command == 'reconcile':
        result = reconcile_submission(args.control, args.node, args.job_id, args.confirmed_not_submitted, args.reason)
    else:
        command = args.worker_command
        if command and command[0] == '--':
            command = command[1:]
        if not command:
            raise ValueError('Worker command required')
        return run_with_receipt(args.control, args.node, args.attempt, command)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print('Error: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
