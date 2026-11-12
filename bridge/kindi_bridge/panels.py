# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
from __future__ import annotations

from collections import deque
import csv
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request

import psutil

from .store import Store
from .visual import INK,LIME,TEXT,MUTED,GREY,EDGE,CYAN,RED,RESET,BOLD,bg,fg,bar,spark,graph,pair,panel,fit


def number(value):
    try:return float(value)
    except (ValueError,TypeError):return None


def fmt(value,suffix=''):
    return '—' if value is None else f'{value:,.0f}{suffix}'


def gpu_data():
    fields=['index','name','memory.used','memory.total','utilization.gpu','temperature.gpu','power.draw','fan.speed']
    p=subprocess.run(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
    if p.returncode:return []
    rows=[]
    for values in csv.reader(p.stdout.splitlines()):
        if len(values)!=len(fields):continue
        d=dict(zip(fields,[v.strip() for v in values]))
        for key in fields[2:]:d[key]=number(d[key])
        rows.append(d)
    return rows


class Vitals:
    def __init__(self):
        self.cpu=deque(maxlen=120);self.gpu={};self.disk=None;self.net=None

    def frame(self,width,height):
        inner=max(10,width-4);bw=max(5,min(18,inner-25))
        cpu=psutil.cpu_percent();self.cpu.append(cpu)
        mem=psutil.virtual_memory();rows=[]
        rows.append(pair(fg(TEXT)+BOLD+socket.gethostname().upper(),fg(GREY)+'LOCAL HARDWARE',inner))
        rows.append(pair(fg(MUTED)+'CPU   '+bar(cpu/100,bw),fg(TEXT)+fmt(cpu,'%'),inner))
        rows+=graph(self.cpu,inner,2 if height<=24 else 3)
        rows.append(pair(fg(MUTED)+'RAM   '+bar(mem.percent/100,bw),fg(TEXT)+f'{mem.used/2**30:.1f}/{mem.total/2**30:.0f} GiB',inner))
        rows.append(fg(EDGE)+'─'*inner)
        try:cards=gpu_data()
        except (OSError,subprocess.TimeoutExpired):cards=[]
        for card in cards:
            history=self.gpu.setdefault(card['index'],deque(maxlen=120));history.append(card['utilization.gpu'] or 0)
            rows.append(fg(LIME)+BOLD+'GPU '+card['index']+'  '+fg(TEXT)+card['name'].replace('NVIDIA ',''))
            used,total=card['memory.used'],card['memory.total'];frac=used/total if used is not None and total else None
            rows.append(pair(fg(MUTED)+'VRAM  '+bar(frac,bw,RED if frac and frac>.95 else LIME),fg(TEXT)+f'{fmt(used)}/{fmt(total)} MiB',inner))
            rows.append(pair(fg(MUTED)+'LOAD  '+bar(card['utilization.gpu']/100 if card['utilization.gpu'] is not None else None,bw),spark(history,min(12,max(1,inner-bw-8))),inner))
            rows.append(pair(fg(MUTED)+'TEMP  '+fg(TEXT)+fmt(card['temperature.gpu'],' °C'),fg(GREY)+'POWER '+fmt(card['power.draw'],' W'),inner))
            rows.append(fg(EDGE)+'─'*inner)
        if not cards:rows.append(fg(GREY)+'GPU telemetry unavailable')
        disk=psutil.disk_usage(os.path.expanduser('~'))
        rows.append(pair(fg(MUTED)+'DISK  '+bar(disk.percent/100,bw),fg(TEXT)+fmt(disk.percent,'%'),inner))
        now=time.monotonic();net=psutil.net_io_counters()
        if self.net:
            stamp,old=self.net;elapsed=max(.01,now-stamp)
            rows.append(pair(fg(MUTED)+f'NET ↓ {(net.bytes_recv-old.bytes_recv)/elapsed/1024:.0f} KiB/s',fg(MUTED)+f'↑ {(net.bytes_sent-old.bytes_sent)/elapsed/1024:.0f} KiB/s',inner))
        self.net=(now,net)
        return panel('HARDWARE VITALS',rows,width,height,'live · local host · 1s')


class Inference:
    def __init__(self,store=None):
        self.store=store or Store();self.last=None;self.speeds=deque(maxlen=60);self.last_speed=None

    def frame(self,width,height):
        inner=max(10,width-4);url=self.store.setting('endpoint') or 'http://127.0.0.1:8093/health'
        try:
            from .workbench import refresh_agent_header
            refresh_agent_header(self.store)
        except Exception:pass
        try:
            with urllib.request.urlopen(url,timeout=2) as r:h=json.load(r)
        except Exception:
            return panel('INFERENCE TELEMETRY',[fg(RED)+'ENDPOINT OFFLINE','',fg(GREY)+'Waiting for /health',fg(GREY)+url],width,height,'source unavailable')
        busy=h.get('busy') or {};stats=h.get('stats') or {};phase=busy.get('phase','idle')
        label='GENERATING' if 'generat' in phase else 'READING PROMPT' if busy else 'IDLE'
        color=LIME if 'generat' in phase else CYAN if busy else GREY
        rows=[pair(fg(TEXT)+BOLD+h.get('model','—'),fg(color)+'● '+label,inner)]
        origin=self.store.setting('endpoint_label') or url.split('/')[2]
        extras=[f'ctx {fmt(h.get("ctx"))}']
        if h.get('head'):extras.append('head '+str(h['head']))
        if h.get('dspark') or h.get('draft'):extras.append('drafting')
        rows.append(pair(fg(GREY)+origin+(f' · Kindi {h["kindi"]}' if h.get('kindi') else ''),fg(GREY)+' · '.join(extras),inner))
        rows.append(fg(EDGE)+'─'*inner)
        now=time.monotonic();speed=None
        if busy:
            parts=str(busy.get('prompt_read','0/0')).split('/')
            read,total=(int(x) for x in parts) if len(parts)==2 and all(x.isdigit() for x in parts) else (0,0)
            generated=busy.get('generated',0)
            if self.last and self.last['prompt']==busy.get('prompt_tokens') and generated>=self.last['generated'] and now-self.last['at']>=.4:
                speed=(generated-self.last['generated'])/(now-self.last['at'])
            if speed is not None and speed>0:self.speeds.append(speed);self.last_speed=speed
            self.last={'prompt':busy.get('prompt_tokens'),'generated':generated,'at':now}
            rows.append(pair(fg(MUTED)+'PROMPT '+bar(read/total if total else 1,max(5,min(18,inner-24)),CYAN),fg(TEXT)+f'{fmt(read)}/{fmt(total)}',inner))
            rows.append(pair(fg(MUTED)+'ANSWER '+fg(TEXT)+fmt(generated,' tokens'),fg(LIME)+(f'{speed:.1f} tok/s' if speed is not None else '—'),inner))
            rows.append(pair(fg(MUTED)+'REUSED '+fg(TEXT)+fmt(busy.get('reused')),fg(GREY)+fmt(busy.get('elapsed_s'),'s'),inner))
        else:
            self.last=None
            rows += [fg(MUTED)+'PROMPT  '+fg(GREY)+'waiting for next request',fg(MUTED)+'ANSWER  '+fg(GREY)+'—',fg(MUTED)+'REUSED  '+fg(GREY)+'—']
        rows.append(fg(EDGE)+'─'*inner)
        rows.append(fg(MUTED)+'GENERATION · OBSERVED TOKENS / SECOND')
        rows+=graph(self.speeds,inner,2,max([1,*self.speeds])*1.1)
        rows.append(pair(fg(MUTED)+'REQUESTS',fg(TEXT)+fmt(stats.get('requests')),inner))
        rows.append(pair(fg(MUTED)+'GENERATED',fg(TEXT)+fmt(stats.get('completion_tokens'),' tokens'),inner))
        rows.append(pair(fg(MUTED)+'CACHE REUSED',fg(TEXT)+fmt(stats.get('cached_tokens'),' tokens'),inner))
        jobs=self.store.rows("SELECT count(*) n FROM jobs WHERE state='scheduled'")[0]['n']
        rows.append(pair(fg(MUTED)+'SCHEDULED PROMPTS',fg(LIME)+str(jobs),inner))
        rows.append(fg(EDGE)+'─'*inner)
        agents=self.store.rows("SELECT state,count(*) n FROM agents GROUP BY state")
        total=sum(a['n'] for a in agents);active=sum(a['n'] for a in agents if a['state'] not in {'offline','failed'})
        rows.append(pair(fg(MUTED)+'AGENTS ONLINE',fg(TEXT)+f'{active} / {total}',inner))
        next_job=self.store.rows("SELECT due FROM jobs WHERE state='scheduled' ORDER BY due LIMIT 1")
        when=time.strftime('%H:%M:%S',time.localtime(next_job[0]['due'])) if next_job else '—'
        rows.append(pair(fg(MUTED)+'NEXT PROMPT',fg(TEXT)+when,inner))
        return panel('INFERENCE TELEMETRY',rows,width,height,'/health · 0.5s · observed rates')


def main(kind):
    obj=Vitals() if kind=='vitals' else Inference()
    print(bg(INK)+'\033[2J\033[?25l',end='',flush=True)
    try:
        while True:
            width,height=shutil.get_terminal_size((52,22))
            lines=obj.frame(width,height)
            frame=bg(INK)+''.join(f'\033[{i+1};1H'+fit(line,width)+'\033[K' for i,line in enumerate(lines[:height]))+'\033[J'
            os.write(1,frame.encode('utf-8'))
            time.sleep(1 if kind=='vitals' else .5)
    except KeyboardInterrupt:pass
    finally:print(RESET+'\033[?25h',end='',flush=True)
