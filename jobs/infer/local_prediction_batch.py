#!/usr/bin/env python3
"""Explicit-file-list, receipt-producing inference; no scheduler or W&B calls."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

PACKAGE = 'Code.time_series_analysis.scripts.analysis.prepared_site_pipeline.prediction_runs'


class LimitedText:
    """Bound entries before the existing writer creates its second final copy."""
    def __init__(self, stream, limit):
        self.stream, self.limit, self.written = stream, limit, 0

    def write(self, text):
        size = len(text.replace('\n', os.linesep).encode('utf-8'))
        if self.written + size > self.limit:
            raise OSError('Prediction output exceeds reserved workspace budget')
        value = self.stream.write(text)
        self.written += size
        return value

    def __getattr__(self, name):
        return getattr(self.stream, name)


def require_cuda_backend(torch):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; CPU fallback is disabled')
    if os.name == 'nt':
        from selective_scan_windows.diagnostics import require_backend
        require_backend()
        from selective_scan_windows import fwd
        from ultralytics.nn.modules import common_utils_mbyolo
        if common_utils_mbyolo.selective_scan_cuda_core.fwd is not fwd:
            raise RuntimeError('Model is not using the qualified Windows backend')


def execute(request_path, output):
    from Code.time_series_analysis.scripts.analysis.prepared_site_pipeline.prediction_runs import contract as c
    from Code.time_series_analysis.scripts.analysis.prepared_site_pipeline.prediction_runs import state, validation
    request = c.unseal(c.read_json(request_path), 'worker')
    if request['inference'] != c.INFERENCE:
        raise ValueError('Unsupported inference settings')
    if type(request.get('entry_limit_bytes')) is not int or request['entry_limit_bytes'] <= 0:
        raise ValueError('A positive output byte budget is required')
    checkpoint = Path(request['checkpoint']).resolve(strict=True)
    if c.file_hash(checkpoint) != request['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum changed')
    validation.policy(request['filter_path'], request['filter_sha256'])
    images, tiles = [], request['tiles']
    root = Path(request['image_root']).resolve(strict=True)
    for tile in tiles:
        c.relative(tile['path'])
        path = root / tile['path']
        if path.parent != root or path.is_symlink():
            raise ValueError('Invalid input tile path')
        c.verify_file(path, tile)
        images.append(path)
    if not images or len(set(images)) != len(images):
        raise ValueError('Empty/duplicate batch')
    # Do not inherit HPC integration or tracking controls.
    for key in ('SITE_PREDICT_CONTROL', 'SITE_PREDICT_REQUEST', 'WANDB_RUN_ID'):
        os.environ.pop(key, None)
    os.environ.update(WANDB_MODE='disabled', SITE_PREDICT_SEGMENT_FILTER=request['filter_path'])
    # Resolve the checkpoint with the checkout whose worker was fingerprinted.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    require_cuda_backend(torch)
    from ultralytics import YOLO
    from yolo_site_predict_json import PredictionJsonWriter, build_predict_kwargs, run_per_image_predictions
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Worker output directory must be empty')
    model = YOLO(str(checkpoint), task='segment')
    args = argparse.Namespace(**request['inference'])
    kwargs = build_predict_kwargs(args, output_dir=output, run_name=request['model_name'],
                                  use_half=False, max_det=100, batch=1)
    kwargs.update(save=False, save_txt=False, save_crop=False, show=False, plots=False)
    writer = PredictionJsonWriter(output, source_root=root)
    writer._entries_handle = LimitedText(writer._entries_handle, request['entry_limit_bytes'])
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        failed = run_per_image_predictions(model, images, writer, kwargs)
        if failed:
            raise RuntimeError('Inference failed tiles: ' + ', '.join(failed[:10]))
        path = writer.close()
        result = validation.predictions(path, tiles, request['filter_path'])
        torch.cuda.synchronize(0)
        versions = {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'Pillow')}
        receipt = c.seal('worker-complete', request_sha256=request['sha256'],
                         run_sha256=request['run_sha256'], batch_sha256=request['batch_sha256'],
                         predictions=c.descriptor(path), validation=result, versions=versions,
                         elapsed_seconds=time.monotonic() - started,
                         gpu_peak_allocated=torch.cuda.max_memory_allocated(0),
                         gpu_peak_reserved=torch.cuda.max_memory_reserved(0))
        state.atomic_json(output / 'worker-complete.json', receipt, immutable=True)
        return receipt
    except BaseException:
        writer.abort()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    execute(args.request, args.output_dir)


if __name__ == '__main__':
    main()
