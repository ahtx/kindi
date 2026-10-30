"""CPU-only metadata validation and isolated legacy conversion regression."""
import argparse
import copy
import json
import os
import tempfile
import types
from unittest.mock import patch
from kindi.packed import Tensor,PackedFile
from kindi.lightning import Engine,ModelSpec
from kindi.convert import selftest
from kindi import config
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output');a=p.parse_args();checks=[]
def rejected(name,fn):
    try:fn()
    except (ValueError,MemoryError):checks.append(name);return
    raise AssertionError('accepted '+name)
selftest();checks.append('unchanged PTQ1 trit packing round trip')
source=PackedFile(a.model);spec=ModelSpec(source.gguf);eng=Engine.__new__(Engine);eng.spec=spec;eng.file=source;eng._validate_tensors()
assert len(source.tensors)==401 and spec.blocks.count('mamba')==23 and spec.blocks.count('moe')==23 and spec.blocks.count('attention')==6
assert eng.memory_plan(16384)['kv']==201326592;checks.append('checkpoint shapes and memory plan')
for key,value in [('general.architecture','gemma4'),('nemotron_h_moe.expert_weights_norm',False),
                  ('nemotron_h_moe.expert_group_count',2),('nemotron_h_moe.expert_used_count',129),
                  ('nemotron_h_moe.ssm.inner_size',0),('nemotron_h_moe.attention.key_length',127),
                  ('nemotron_h_moe.attention.layer_norm_rms_epsilon',float('nan'))]:
    bad=dict(source.gguf.kv);bad[key]=value;rejected('metadata '+key,lambda:ModelSpec(types.SimpleNamespace(kv=bad)))
name='blk.1.ffn_up_exps.weight'
name=next(k for k in source.tensors if k.endswith('ffn_up_exps.weight'))
original=source.tensors[name];bad=copy.copy(original);bad.shape=(spec.experts,spec.routed_width+64,spec.width);source.tensors[name]=bad
rejected('incorrect routed expert width',eng._validate_tensors);source.tensors[name]=original
rejected('unsupported quantization',lambda:Tensor('x',[64,1],10,0,0))
rejected('misaligned quantization row',lambda:Tensor('x',[63,1],42,0,0))
with tempfile.TemporaryDirectory() as directory:
    path=os.path.join(directory,'tiny.gguf');open(path,'wb').write(bytes(256))
    cfgpath=os.path.join(directory,'test.conf');open(cfgpath,'w').write('[kindi]\n')
    for label,tensors in [('overlap',[('a',[64],42,0),('b',[64],42,0)]),('duplicate',[('a',[64],42,0),('a',[64],42,32)]),
        ('truncated',[('a',[64,100],42,0)]),('overflow',[('a',[64,2**63],42,0)])]:
        fake=types.SimpleNamespace(tensors=tensors,data_start=0)
        with patch('kindi.packed.GGUF',return_value=fake):rejected(label,lambda:PackedFile(path))
    for backend,model_id in [('kindi_lightning','nemotron-lightning-30b-a3b-ternary'),('kindi_gemma4','nemotron-bolt-e4b')]:
        with open(os.path.join(directory,'kindi_pack.json'),'w') as f:json.dump({'backend':backend,'model_id':model_id},f)
        with patch.dict(os.environ,{k:v for k,v in os.environ.items() if not k.startswith('KINDI_')},clear=True):
            result=config.load(cfgpath,{'pack':directory})
        assert result['model_name']==model_id
    checks.append('Lightning and legacy Bolt config identity')
result={'passed':True,'checks':checks}
if a.output:
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
