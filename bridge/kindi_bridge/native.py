# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""Durable scheduler sidecar for the native, themed pi terminal UI."""
from __future__ import annotations

import concurrent.futures
import contextlib
import json
import os
import queue
import sys
import threading
import time

from .runtime import prompt_message,settle_job
from .store import Store,state_dir


class NativeSession:
    def __init__(self,store,ident):
        self.store=store;self.ident=ident;self.current=None;self.answer='';self.error=None;self.idle=False

    def event(self,e):
        kind=e.get('type')
        if kind=='ready':self.idle=bool(e.get('idle')) and not e.get('editing')
        elif kind=='deferred' and self.current and self.current['id']==e.get('id'):
            self.store.execute("UPDATE jobs SET state='scheduled' WHERE id=? AND state='sending'",(self.current['id'],));self.current=None;self.idle=False
        elif kind=='agent_start':
            self.idle=False;self.answer='';self.error=None
            if self.current and e.get('job')==self.current['id']:
                self.store.execute("UPDATE jobs SET state='accepted',attempts=attempts+1,accepted=? WHERE id=? AND state='sending'",(time.time(),self.current['id']))
                self.current=self.store.rows('SELECT * FROM jobs WHERE id=?',(self.current['id'],))[0]
            self.store.execute("UPDATE agents SET state='generating' WHERE id=?",(self.ident,))
            self.store.event('agent_start','Native pi turn',self.ident,e.get('job'))
        elif kind=='message_end':
            message=e.get('message',{})
            if message.get('role')=='assistant':
                content=message.get('content',[])
                self.answer=content if isinstance(content,str) else '\n'.join(p.get('text','') for p in content if p.get('type')=='text')
                if message.get('stopReason') in {'error','aborted'}:self.error=message.get('errorMessage') or message['stopReason']
        elif kind=='delivery_error' and self.current:
            settle_job(self.store,self.current,'',str(e.get('error','Delivery failed')));self.current=None
        elif kind=='agent_settled':
            if self.current and e.get('job')==self.current['id']:
                saved=self.store.rows('SELECT * FROM jobs WHERE id=?',(self.current['id'],))[0]
                if saved['state']=='sending':
                    # If no run started, don't count an unaccepted message as successful.
                    settle_job(self.store,saved,self.answer,'No native run acceptance observed')
                else:settle_job(self.store,saved,self.answer,self.error)
                self.current=None
            else:self.store.event('assistant',self.answer or self.error or 'Native pi settled',self.ident)
            self.store.execute("UPDATE agents SET state='ready' WHERE id=?",(self.ident,))
            self.idle=False  # Wait for a fresh editor/idle observation before next delivery.
        elif kind in {'tool_execution_start','tool_execution_end','manual_prompt'}:
            if kind=='tool_execution_start':self.store.execute("UPDATE agents SET state='running tool' WHERE id=?",(self.ident,))
            elif kind=='tool_execution_end':self.store.execute("UPDATE agents SET state='waiting for inference' WHERE id=?",(self.ident,))
            elif kind=='manual_prompt' and not self.store.setting('mission:'+self.ident):self.store.setting('mission:'+self.ident,e.get('text',''))
            self.store.event(kind,e.get('text') or json.dumps(e),self.ident,self.current['id'] if self.current else None)

    def claim(self,now=None):
        now=now or time.time()
        if not self.idle or self.current:return None
        with self.store.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute("SELECT * FROM jobs WHERE agent=? AND state='scheduled' AND due<=? AND (deadline IS NULL OR deadline>=?) ORDER BY due,created LIMIT 1",(self.ident,now,now)).fetchone()
            if not row:return None
            self.current=dict(row)
            db.execute("UPDATE jobs SET state='sending',error=NULL WHERE id=?",(self.current['id'],))
        self.idle=False;self.answer='';self.error=None
        self.store.event('sending',self.current['prompt'],self.ident,self.current['id'])
        return {'type':'prompt','id':self.current['id'],'text':prompt_message(self.store,self.store.agent(self.ident),self.current)}

    def close(self):
        if self.current:self.store.execute("UPDATE jobs SET state='uncertain',error='Native pi closed before confirmed completion' WHERE id=? AND state IN ('sending','accepted')",(self.current['id'],))
        self.store.execute("UPDATE agents SET state='offline' WHERE id=?",(self.ident,))


def main(ident):
    import fcntl
    with open(state_dir()/f'agent-{ident}.lock','a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        store=Store();session=NativeSession(store,ident)
        store.execute("UPDATE jobs SET state='uncertain',error='Native session restarted; reconcile delivery' WHERE agent=? AND state IN ('sending','accepted')",(ident,))
        store.execute("UPDATE agents SET state='ready',pid=?,heartbeat=? WHERE id=?",(os.getppid(),time.time(),ident))
        incoming=queue.Queue();output_lock=threading.Lock();pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
        def write(record):
            with output_lock:
                sys.stdout.write(json.dumps(record)+'\n');sys.stdout.flush()
        def reader():
            try:
                for line in sys.stdin.buffer:
                    try:incoming.put(json.loads(line))
                    except ValueError:continue
            finally:incoming.put({'type':'shutdown'})
        threading.Thread(target=reader,daemon=True).start();beat=0;sent=0
        try:
            while True:
                now=time.time()
                if now-beat>=1:
                    store.execute('UPDATE agents SET heartbeat=? WHERE id=?',(now,ident));beat=now
                if session.current and now-sent>3600:
                    store.execute("UPDATE jobs SET state='uncertain',error='Native turn exceeded one hour' WHERE id=?",(session.current['id'],))
                    session.current=None
                    store.event('turn_timeout','Inspect native pi before retrying',ident)
                try:e=incoming.get(timeout=.1)
                except queue.Empty:e=None
                if e:
                    if e.get('type')=='shutdown':break
                    if e.get('type')=='retrieve':
                        def retrieve(request=e):
                            try:
                                from .knowledge import context
                                block=context(store,request.get('query',''),store.agent(ident)['collection'])
                                if block:store.event('retrieved_context',block,ident)
                                write({'type':'retrieval','id':request['id'],'text':block})
                            except Exception as error:
                                store.event('retrieval_error',error,ident);write({'type':'retrieval','id':request['id'],'text':''})
                        pool.submit(retrieve)
                    else:session.event(e)
                command=session.claim()
                if command:sent=now;write(command)
        except (BrokenPipeError,KeyboardInterrupt):pass
        finally:
            session.close();pool.shutdown(wait=False,cancel_futures=True)
