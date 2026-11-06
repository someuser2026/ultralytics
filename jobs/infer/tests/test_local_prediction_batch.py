"""Isolated worker tests. Run with rip-detection on PYTHONPATH; no CUDA needed."""
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from PIL import Image
from Code.time_series_analysis.scripts.analysis.prepared_site_pipeline.prediction_runs import contract as c, settings, state, validation


@pytest.fixture
def worker():
    path=Path(os.environ.get('LOCAL_PREDICTION_WORKER', str(Path(__file__).resolve().parents[1]/'local_prediction_batch.py')))
    spec=importlib.util.spec_from_file_location('local_batch_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def batch(tmp_path):
    ckpt=tmp_path/'arbitrary.pt';ckpt.write_bytes(b'test')
    root=tmp_path/'tiles';root.mkdir()
    image=root/'scene_x0_y0.png';Image.new('RGB',(10,10)).save(image)
    tile={'path':image.name,'width':10,'height':10,**c.descriptor(image)}
    request=c.seal('worker',checkpoint=str(ckpt),checkpoint_sha256=c.file_hash(ckpt),inference=c.INFERENCE,
        filter_path=str(settings.FILTER),filter_sha256=c.file_hash(settings.FILTER),image_root=str(root),tiles=[tile],model_name='model',entry_limit_bytes=1024**2,
        run_sha256='a'*64,batch_sha256='b'*64)
    path=state.atomic_json(tmp_path/'request.json',request)
    return path,request,tmp_path/'output'


def test_reject_settings_before_model_import(worker,batch):
    path,r,out=batch;r['inference']={**c.INFERENCE,'imgsz':640}
    state.atomic_json(path,c.seal('worker',**{k:v for k,v in r.items() if k not in {'sha256','schema_version'}}))
    with pytest.raises(ValueError,match='settings'):worker.execute(path,out)


def test_reject_changed_tile(worker,batch):
    path,r,out=batch
    (Path(r['image_root'])/r['tiles'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):worker.execute(path,out)


@pytest.mark.parametrize('failure',[None,'skipped','oom','omitted','partial'])
def test_worker_receipt_and_failures(worker,batch,monkeypatch,failure):
    path,r,out=batch
    cuda=SimpleNamespace(is_available=lambda:True,reset_peak_memory_stats=lambda n:None,synchronize=lambda n:None,
                         max_memory_allocated=lambda n:123,max_memory_reserved=lambda n:456)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(worker,'require_cuda_backend',lambda torch:None)
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda *a,**k:object()))
    class Writer:
        def __init__(self,output,source_root):self.output=output;self._entries_handle=io.StringIO()
        def close(self):
            if failure=='partial':
                path=self.output/'predictions.json';path.write_text('{"count":1,');return path
            if failure=='omitted':return state.atomic_json(self.output/'predictions.json',{'count':0,'predictions':{}})
            return state.atomic_json(self.output/'predictions.json',{'count':1,'predictions':{'scene_x0_y0':{'task':'segment','orig_shape':{'height':10,'width':10},'predictions':[]}}})
        def abort(self):pass
    def run(*args):
        if failure=='oom':raise RuntimeError('CUDA out of memory')
        return ['scene_x0_y0.png'] if failure=='skipped' else []
    monkeypatch.setitem(sys.modules,'yolo_site_predict_json',SimpleNamespace(PredictionJsonWriter=Writer,build_predict_kwargs=lambda *a,**k:{},run_per_image_predictions=run))
    monkeypatch.setenv('SITE_PREDICT_CONTROL','bad')
    if failure:
        with pytest.raises((RuntimeError,ValueError)):worker.execute(path,out)
        assert not (out/'worker-complete.json').exists()
    else:
        result=worker.execute(path,out)
        assert result['validation']['tiles']==1
        assert result['gpu_peak_allocated']==123
        assert os.environ['WANDB_MODE']=='disabled'
        assert 'SITE_PREDICT_CONTROL' not in os.environ


def test_output_budget(worker):
    target=io.StringIO(); limited=worker.LimitedText(target,4)
    limited.write('abcd')
    with pytest.raises(OSError,match='budget'):limited.write('e')
    assert target.getvalue()=='abcd'


def test_no_cpu_fallback(worker):
    with pytest.raises(RuntimeError,match='CUDA'):
        worker.require_cuda_backend(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
