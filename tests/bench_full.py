import argparse
import atexit
import csv
import json
import os
import statistics
import subprocess
import threading
import time
from kindi.lightning import Engine
from kindi.lightning_tokenizer import Tok
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--prompt',type=int,default=128);p.add_argument('--generate',type=int,default=32);p.add_argument('--repeats',type=int,default=3)
p.add_argument('--modes',default='baseline,optimized,chunk64,chunk128');p.add_argument('--ctx',type=int,default=16384)
p.add_argument('--prompts',help='comma-separated matrix of prompt lengths');p.add_argument('--gpu',default='');a=p.parse_args()
class Telemetry(object):
    def __init__(self):
        self.rows=[];self.error=None;self.process=None
        self.fields='timestamp,name,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.gr,clocks.mem'.split(',')
        try:
            cmd=['nvidia-smi','--query-gpu='+','.join(self.fields),'--format=csv,noheader,nounits','-l','1']
            if a.gpu:cmd+=['-i',a.gpu]
            self.process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,universal_newlines=True)
            def read():
                for line in self.process.stdout:
                    values=next(csv.reader([line]));self.rows.append(dict(zip(self.fields,[v.strip() for v in values])))
            thread=threading.Thread(target=read);thread.daemon=True;thread.start()
        except OSError as exc:self.error=str(exc)
    def stop(self):
        if self.process and self.process.poll() is None:self.process.terminate();self.process.wait()
telemetry=Telemetry();atexit.register(telemetry.stop)
start=time.perf_counter();eng=Engine(a.model,ctx=a.ctx);load_s=time.perf_counter()-start;tok=Tok(a.model)
body=('Explain how arithmetic, language, computer memory, and astronomy relate to scientific reasoning. '
      'Use precise examples and check calculations. Приведи пример вычисления, затем проверь результат.\n')
results=[]
def save(complete):
    with open(a.output,'w') as f:json.dump({'complete':complete,'results':results,'memory':eng.memory,'model_load_s':load_s,
        'load_policy':'OS file cache not evicted; existing model reused','numeric_profile':eng.numeric_profile,
        'source_sha256':eng.source_sha256,'telemetry':list(telemetry.rows),'telemetry_error':telemetry.error},f,indent=2)
for prompt_length in map(int,(a.prompts or str(a.prompt)).split(',')):
    tokens=tok.encode(tok.chat([{'role':'user','content':body*(prompt_length//20+10)}]))[:prompt_length]
    assert len(tokens)==prompt_length and len(tokens)+a.generate<=eng.ctx
    for mode in a.modes.split(','):
        eng.ops.profile='baseline' if mode=='baseline' else 'optimized'
        chunk=int(mode[5:]) if mode.startswith('chunk') else 0
        eng.reset();logits=eng.prefill(tokens[:8],chunk_size=chunk);eng.forward(int(logits.argmax()));eng.cp.cuda.Stream.null.synchronize()
        for repeat in range(a.repeats):
            eng.reset();start=time.perf_counter();logits=eng.prefill(tokens,chunk_size=chunk);eng.cp.cuda.Stream.null.synchronize();prefill=time.perf_counter()-start
            latencies=[];ids=[];start=time.perf_counter();first_sample=None
            for i in range(a.generate):
                tick=time.perf_counter();token=int(logits.argmax())
                if first_sample is None:first_sample=time.perf_counter()-start
                ids.append(token);logits=eng.forward(token);eng.cp.cuda.Stream.null.synchronize();latencies.append(time.perf_counter()-tick)
            decode=time.perf_counter()-start;free,total=eng.cp.cuda.runtime.memGetInfo()
            try:
                import resource
                peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            except ImportError:peak_rss=None
            result={'mode':mode,'repeat':repeat,'prompt_tokens':len(tokens),'generated_tokens':len(ids),'eos_policy':'ignored for fixed-length benchmark',
                'prefill_s':prefill,'prefill_tps':len(tokens)/prefill,'decode_s':decode,'decode_tps':len(ids)/decode,
                'ttft_s':prefill+first_sample,'latency_p50_ms':statistics.median(latencies)*1000,'latency_p95_ms':sorted(latencies)[min(len(latencies)-1,int(len(latencies)*.95))]*1000,
                'gpu_used_bytes':total-free,'process_peak_rss_bytes':peak_rss,'ids':ids}
            results.append(result);save(False)
            print(json.dumps({k:v for k,v in result.items() if k!='ids'}),flush=True)
telemetry.stop();save(True)
