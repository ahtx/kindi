# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""User-facing agent operations; configuration stays behind these actions."""
from __future__ import annotations

import json
from pathlib import Path
import time

from .store import Store


def current_agent(store):
    ident=store.setting('active_agent') or store.setting('default_agent')
    return store.agent(ident)


def status(agent,now=None):
    now=now or time.time()
    if agent['heartbeat'] and now-agent['heartbeat']>15:return 'Offline'
    return {'ready':'Ready','starting':'Starting','generating':'Working','running tool':'Running a tool',
            'waiting for inference':'Waiting for model','needs input':'Needs you','offline':'Offline','failed':'Failed'}.get(agent['state'],agent['state'].capitalize())


def roster(store):
    active=store.setting('active_agent') or store.setting('default_agent')
    rows=store.rows('SELECT * FROM agents ORDER BY name COLLATE NOCASE')
    for a in rows:
        mission=store.setting('mission:'+a['id'])
        if not mission:
            previous=store.rows("SELECT detail FROM events WHERE agent=? AND kind='manual_prompt' ORDER BY id LIMIT 1",(a['id'],))
            mission=previous[0]['detail'] if previous else None
        a['mission']=mission or 'No mission yet'
        a['status']=status(a);a['current']=a['id']==active
        a['pending']=store.rows("SELECT count(*) n FROM jobs WHERE agent=? AND state='scheduled' AND due<=?",(a['id'],time.time()))[0]['n']
    return rows


def create(store,name,mission,folder=None):
    from .runtime import launch_agent,pi_command
    name=name.strip();mission=mission.strip()
    if not name:raise ValueError('Give the agent a name.')
    if not mission:raise ValueError('Tell the agent what you want it to do.')
    if store.rows('SELECT id FROM agents WHERE name=?',(name,)):raise ValueError('An agent with that name already exists.')
    try:template=current_agent(store)
    except ValueError:template=None
    cwd=Path(folder or (template['cwd'] if template else Path.home()/'kindi-bridge-work')).expanduser().resolve()
    if not cwd.is_dir():raise ValueError('Choose an existing working folder.')
    command=json.loads(template['command']) if template else pi_command()
    # A new agent has its own conversation, even when it inherits the model and tools.
    for flag in ('--session','--session-id','--session-dir'):
        if flag in command:
            i=command.index(flag);del command[i:i+2]
    command=[x for x in command if x not in ('--continue','-c','--resume','-r')]
    # Kindi > Use a server for agents: a new agent is pinned to the chosen server, so it keeps it after a restart
    provider,model=store.setting('kindi_provider'),store.setting('kindi_model')
    if provider and model:
        for flag in ('--provider','--model'):
            while flag in command:
                i=command.index(flag);del command[i:i+2]
        command+=['--provider',provider,'--model',model]
    agent=store.add_agent(name,str(cwd),command)
    store.setting('mission:'+agent['id'],mission)
    job=store.schedule(mission,time.time(),agent['id'])
    try:launch_agent(store,agent['id'])
    except Exception as error:
        store.event('agent_start_error',str(error),agent['id'],job)
        raise ValueError(f'Agent saved with its mission, but could not start: {error}') from error
    return store.agent(agent['id'])


def assign(store,ident,mission):
    mission=mission.strip()
    if not mission:raise ValueError('Enter a mission or prompt.')
    agent=store.agent(ident)
    store.setting('mission:'+agent['id'],mission)
    return store.schedule(mission,time.time(),agent['id'])
