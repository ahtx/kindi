"""Sticky GPU failures must preserve their cause and turn health red."""
import json,types
from kindi.lightning import Engine
from kindi import lightning_server as server

primary=RuntimeError('original asynchronous CUDA failure')
def broken_sync():raise primary
class BrokenArray(object):
    def fill(self,value):raise RuntimeError('secondary reset failure')
eng=Engine.__new__(Engine);eng.failed=False;eng.position=17;eng.tokens=[1]*17;eng.last_logits=object()
eng.states={0:BrokenArray()};eng.histories={}
eng.cp=types.SimpleNamespace(cuda=types.SimpleNamespace(Stream=types.SimpleNamespace(null=types.SimpleNamespace(synchronize=broken_sync))))
try:eng.synchronize();raise AssertionError('GPU failure swallowed')
except RuntimeError as exc:assert exc is primary
assert eng.failed and eng.position==0 and eng.tokens==[] and eng.last_logits is None
for operation in (lambda:eng.forward(1),lambda:eng.prefill([1]),eng.snapshot,lambda:eng.restore({}),eng.reset):
    try:operation();raise AssertionError('failed engine accepted work')
    except RuntimeError as exc:assert 'restart this backend process' in str(exc)

eng.ctx=32768;eng.kv_dtype='fp16';eng.memory={}
runtime=types.SimpleNamespace(engine=eng,busy=False,phase='failed',requests=2,prefill_chunk=128,
    cache=types.SimpleNamespace(used=0,entries={}))
original=server.Runtime
app=None
try:
    server.Runtime=lambda cfg,log:runtime
    app=server.create_app({'model_name':'test','api_key':''},log=lambda text:None)
    endpoint=next(route.endpoint for route in app.routes if route.path=='/health')
    response=endpoint();assert response.status_code==503 and not json.loads(response.body)['ok']
    eng.failed=False;runtime.phase='idle';response=endpoint()
    assert response.status_code==200 and json.loads(response.body)['ok']
finally:
    server.Runtime=original
    if app is not None:
        for handler in app.router.on_shutdown:handler()
print(json.dumps({'passed':True,'original_failure_preserved':True,'failed_engine_rejects_work':True,'health_status':[503,200]}))
