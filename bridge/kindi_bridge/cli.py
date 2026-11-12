# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

from .store import Store


def duration(value):
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd])\s*", value)
    if not m:
        raise ValueError("Use a duration such as 20m, 2h, or 3d")
    seconds = float(m[1]) * {"s":1,"m":60,"h":3600,"d":86400}[m[2]]
    if seconds <= 0:
        raise ValueError("Duration must be positive")
    return seconds


def parse_time(value, now=None):
    now = time.time() if now is None else now
    value = value.strip()
    if value == 'now':
        return now
    if value.startswith('in '):
        return now+duration(value[3:])
    if re.fullmatch(r'\d{1,2}:\d{2}', value):
        hour,minute = map(int,value.split(':'))
        base = dt.datetime.fromtimestamp(now).replace(hour=hour,minute=minute,second=0,microsecond=0)
        if base.timestamp()<=now:
            base+=dt.timedelta(days=1)
        return base.timestamp()
    try:
        return dt.datetime.fromisoformat(value).timestamp()
    except ValueError as e:
        raise ValueError('Use now, in 20m, HH:MM, or an ISO date/time with optional timezone offset') from e


def parser():
    p=argparse.ArgumentParser(prog='kindi-bridge',description='Persistent terminal workspace. Run without arguments for the UI.')
    p.add_argument('--state',help='Private runtime directory (otherwise BRIDGE_STATE or ~/.local/state/kindi-bridge)')
    sub=p.add_subparsers(dest='command')
    ui=sub.add_parser('ui');ui.add_argument('--dir',default=os.getcwd());ui.add_argument('--screen',default='Deck')
    screen=sub.add_parser('screen');screen.add_argument('page')
    panel=sub.add_parser('panel');panel.add_argument('kind',choices=['vitals','inference'])
    native=sub.add_parser('native-adapter');native.add_argument('agent')
    focus=sub.add_parser('focus');focus.add_argument('role',choices=['pi','shell'])
    sub.add_parser('layout')
    menu=sub.add_parser('menu');menu.add_argument('--category',default='Agents');menu.add_argument('--x',type=int);menu.add_argument('--y',type=int)
    menu_ui=sub.add_parser('menu-ui');menu_ui.add_argument('--category',default='Agents')
    sub.add_parser('serve',help='Run the durable service in foreground')
    worker=sub.add_parser('worker',help=argparse.SUPPRESS);worker.add_argument('agent')
    agents=sub.add_parser('agent').add_subparsers(dest='action',required=True)
    add=agents.add_parser('add');add.add_argument('name');add.add_argument('--dir',default=os.getcwd());add.add_argument('--exec',dest='exec_command',help='Pi RPC command; default: discovered pi --mode rpc');add.add_argument('--start',action='store_true')
    agents.add_parser('list')
    for action in ('start','default'):
        x=agents.add_parser(action);x.add_argument('name')
    jobs=sub.add_parser('jobs').add_subparsers(dest='action',required=True)
    for action in ('add','check'):
        x=jobs.add_parser(action);x.add_argument('prompt',nargs='?' if action=='check' else None);x.add_argument('--agent');x.add_argument('--at',default='now');x.add_argument('--every',default='30m' if action=='check' else None);x.add_argument('--limit',type=int,default=6 if action=='check' else 1);x.add_argument('--task',required=action=='check');x.add_argument('--deadline')
    jobs.add_parser('list');jobs.add_parser('history')
    for action in ('delete','retry','pause'):
        x=jobs.add_parser(action);x.add_argument('id')
    knowledge=sub.add_parser('knowledge').add_subparsers(dest='action',required=True)
    add=knowledge.add_parser('add');add.add_argument('folder');add.add_argument('--keyword-only',action='store_true');add.add_argument('--agent')
    knowledge.add_parser('list')
    index=knowledge.add_parser('index');index.add_argument('id')
    search=knowledge.add_parser('search');search.add_argument('query');search.add_argument('--collection')
    sub.add_parser('processes');sub.add_parser('ports');sub.add_parser('schedules')
    sub.add_parser('tiles',help='Attach to live agent terminals; Ctrl-b d returns')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.state:
        os.environ['BRIDGE_STATE']=str(Path(args.state).expanduser().resolve())
    try:
        store=Store()
        from . import runtime
        result=None
        if args.command is None:
            from .workbench import main as workbench
            raise SystemExit(workbench())
        if args.command=='ui':
            from .tui import main as ui
            ui(args.dir,args.screen);return
        if args.command in ('screen','focus','layout'):
            from . import workbench
            if args.command=='screen':workbench.screen(args.page)
            elif args.command=='focus':workbench.focus(args.role)
            else:workbench.layout()
            return
        if args.command=='panel':
            from .panels import main as panel
            panel(args.kind);return
        if args.command=='menu':
            from .workbench import menu
            menu(args.category,args.x,args.y);return
        if args.command=='menu-ui':
            from .menus import run
            run(args.category);return
        if args.command=='native-adapter':
            from .native import main as native
            native(args.agent);return
        if args.command=='serve':runtime.serve();return
        if args.command=='worker':runtime.worker(args.agent);return
        if args.command=='agent':
            if args.action=='list':result=store.rows('SELECT * FROM agents ORDER BY name')
            elif args.action=='add':
                cmd=shlex.split(args.exec_command) if args.exec_command else runtime.pi_command()
                agent=store.add_agent(args.name,args.dir,cmd)
                if args.start:runtime.ensure_service();runtime.launch_agent(store,agent['id'])
                result=agent
            elif args.action=='start':
                runtime.ensure_service();result=runtime.launch_agent(store,args.name)
            else:result=store.setting('default_agent',store.agent(args.name)['id'])
        elif args.command=='jobs':
            if args.action in ('add','check'):
                result=store.schedule(args.prompt or runtime.CHECK_PROMPT,parse_time(args.at),args.agent,duration(args.every) if args.every else 0,args.limit,args.task,parse_time(args.deadline) if args.deadline else None,'check' if args.action=='check' else 'prompt')
            elif args.action in ('list','history'):
                result=store.rows('SELECT j.*,a.name FROM jobs j JOIN agents a ON a.id=j.agent ORDER BY due')
            elif args.action=='delete':result=store.delete_job(args.id)
            elif args.action=='pause':result=store.execute("UPDATE jobs SET state='paused' WHERE id=? AND state='scheduled'",(args.id,))
            else:
                # An explicit user retry reconciles uncertainty; it is never an automatic resend.
                result=store.execute("UPDATE jobs SET state='scheduled',error=NULL,due=?,attempts=0 WHERE id=? AND state IN ('uncertain','failed','exhausted','paused')",(time.time(),args.id))
        elif args.command=='knowledge':
            from .knowledge import add_collection,index_collection,search
            if args.action=='list':result=store.rows('SELECT * FROM collections')
            elif args.action=='add':
                result=add_collection(store,args.folder,not args.keyword_only)
                if args.agent:store.execute('UPDATE agents SET collection=? WHERE id=?',(result,store.agent(args.agent)['id']))
            elif args.action=='index':
                rows=store.rows('SELECT * FROM collections WHERE id=?',(args.id,))
                if not rows:raise ValueError('Unknown collection')
                index_collection(store,rows[0]);result=store.rows('SELECT * FROM collections WHERE id=?',(args.id,))[0]
            else:result=search(store,args.query,args.collection)
        elif args.command in ('processes','ports','schedules'):
            from .system import processes,listeners,schedule_inventory
            result={'processes':processes,'ports':listeners,'schedules':schedule_inventory}[args.command]()
        elif args.command=='tiles':
            import subprocess
            raise SystemExit(subprocess.call(['tmux','-L',os.environ.get('BRIDGE_TMUX_SOCKET','kindi-bridge'),'attach-session','-t','agents']))
        if result is not None:print(json.dumps(result,indent=2,default=str))
    except (ValueError,RuntimeError,OSError) as e:
        p.exit(1,f'Bridge: {e}\n')


if __name__=='__main__':main()
