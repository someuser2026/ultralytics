"""Prepared prediction contract v2; no model imports or tile-content access."""
import hashlib
import json
import math
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def unique_payload(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


def is_v2(request):
    return request.get('settings', {}).get('prepared_site_validation', {}).get('version') == 2


def load_inventory(marker):
    path = Path(marker['inventory'])
    if path.is_symlink() or path.resolve() != path.absolute() or not path.is_file():
        raise ValueError('Invalid inventory location')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != marker['inventory_sha256']:
        raise ValueError('Prepared tile inventory checksum changed')
    value = unique_payload(data)
    if (value.get('schema_version') != 'prepared-tile-inventory-v2'
            or value.get('run') != marker['run']
            or value.get('content_verified') is not marker['verify_tile_contents']
            or value.get('evidence_sha256') != digest({k: v for k, v in value.items() if k != 'evidence_sha256'})
            or value.get('sha256') != digest(value.get('tiles'))
            or type(value.get('count')) is not int or value['count'] != len(value['tiles']) or not value['count']):
        raise ValueError('Invalid prepared tile inventory evidence')
    names = []
    for record in value['tiles']:
        name = record['path']; names.append(name)
        match = re.fullmatch(r'(.+)_x(0|[1-9][0-9]*)_y(0|[1-9][0-9]*)\.png', name)
        if (not match or Path(name).name != name or match[1] != record.get('source_scene')
                or any(type(record.get(k)) is not int or record[k] < 0 for k in ('width', 'height', 'x0', 'y0', 'bytes'))
                or min(record['width'], record['height'], record['bytes']) <= 0
                or (record['x0'], record['y0']) != (int(match[2]), int(match[3]))
                or (marker['verify_tile_contents'] and not re.fullmatch('[0-9a-f]{64}', str(record.get('sha256', ''))))):
            raise ValueError('Invalid prepared tile geometry')
    if names != sorted(set(names)):
        raise ValueError('Duplicate/unsorted tile evidence')
    return value


def image_records(marker, image_root):
    return [{**r, 'path': str(Path(image_root) / r['path'])} for r in load_inventory(marker)['tiles']]


def validate_records(payload, expected, policy=None):
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
    expected_records = {Path(r['path']).stem: r for r in expected}
    for stem, record in zip(stems, records.values()):
        width, height = expected_records[stem]['width'], expected_records[stem]['height']
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
        # Geometry filtering belongs in export; validators must never rewrite results.
        if policy is not None:
            _, counts = policy.filter_segmentation_record(record)
            if counts['removed']:
                raise ValueError('Unfiltered degenerate segmentation predictions: ' + stem)
    return len(records)


def build_batch_manifest(request, state, receipt_validator):
    files = []
    for batch in request['batches']:
        attempts = state['jobs'][batch['node']]
        job = attempts[-1]
        if (len(attempts) != 1 or job.get('attempt') != 1 or job.get('status') != 'accepted'
                or (job.get('terminal') or {}).get('exit_status') != 0):
            raise ValueError('Batch lacks accepted successful terminal evidence')
        result = receipt_validator(request, job)
        files.append({'node': batch['node'], 'index': batch['index'], 'sha256': result['sha256'],
            'count': result['count'], 'images': request['images'][batch['start']:batch['end']]})
    value = {'schema_version': 'prepared-prediction-batches-v2', 'request_hash': request['hash'],
             'control': request['control'], 'output': request['output'], 'batches': files}
    value['sha256'] = digest(value)
    return value


def load_batch_manifest(path, output):
    value = unique_payload(Path(path).read_bytes())
    request = unique_payload((Path(value['control']) / 'request.json').read_bytes())
    if (value.get('schema_version') != 'prepared-prediction-batches-v2'
            or value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'})
            or request.get('hash') != digest({k: v for k, v in request.items() if k != 'hash'})
            or value.get('request_hash') != request['hash'] or value.get('output') != str(output)
            or not is_v2(request)):
        raise ValueError('Invalid combination manifest ownership')
    # Bind evidence to frozen assignments and actual successful receipts again,
    # without opening batch predictions or image files.
    from site_predict_job_control import validate_success_receipt
    state = unique_payload((Path(value['control']) / 'state.json').read_bytes())
    if build_batch_manifest(request, state, validate_success_receipt) != value:
        raise ValueError('Combination manifest differs from receipts')
    return value, request
