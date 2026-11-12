# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""The persistent Kindi workbench: native pi, command pane, and shared side panels."""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from .runtime import child_env,ensure_service,pi_command,tmux
from .store import Store,state_dir

ASSETS=Path(__file__).parent/'assets'
# F1..F9 (and Ctrl-b 1..9) open the first nine pages; Flow (the activity log) is in Schedule > Activity log.
PAGES=['Workspace','Agents','Queue','Kindi','Files','Knowledge','System','Processes','Ports','Flow']
HEADER_FORMAT='#[align=left,bg=#caf31c,fg=#080808,bold] KINDI // BRIDGE #[bg=#080808,fg=#6b6b6b,nobold]  #{host_short} #[fg=#f3f3f1] | Agent: #{@bridge_agent} | #{@bridge_agent_state} | #{@bridge_screen}#[align=right,fg=#9a9a9a] Menu: click a title / Alt-M / F10 '


def env_command(args,extras=None):
    env={'BRIDGE_STATE':str(state_dir()),'BRIDGE_WORKBENCH':'1','BRIDGE_TMUX_SOCKET':os.environ.get('BRIDGE_TMUX_SOCKET','kindi-bridge')}
    for key in ['PI_CODING_AGENT_DIR','PATH','COLORTERM','PYTHONPATH']:
        if key in os.environ:env[key]=os.environ[key]
    env.update(extras or {})
    return shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],*args])


def pane_exists(pane):
    if not pane:return False
    try:return tmux('display-message','-p','-t',pane,'#{pane_dead}').strip()=='0'
    except RuntimeError:return False


def native_command(agent):
    command=json.loads(agent['command'])
    if '--mode' in command:
        i=command.index('--mode');command=command[:i]+command[i+2:]
    if not any(flag in command for flag in ['--no-session','--session','--session-id','--continue','--resume']):
        sessions=state_dir()/'pi-sessions';sessions.mkdir(exist_ok=True)
        command+=['--session-dir',str(sessions),'--session-id',str(uuid.UUID(agent['id']))]
    return command+['--theme',str(ASSETS/'kindi.json'),'--use-theme','kindi','-e',str(ASSETS/'bridge.ts')]


def navigation(width,page):
    from .menus import bar
    targets=json.loads(Store().setting('menu_targets') or '{}')
    return bar(page,targets)


def ensure_view(store,page):
    target=store.setting('screen_pane:'+page)
    if not pane_exists(target):
        target=tmux('new-window','-d','-P','-F','#{pane_id}','-t','bridge','-n','_'+page.lower(),env_command([sys.executable,'-m','kindi_bridge','ui','--screen',page,'--dir',store.agent()['cwd']])).strip()
        store.setting('screen_pane:'+page,target)
    return target


def prepare_menu_targets(store,window):
    # tmux 3.2's mouse coordinates are pane-relative and absent on status lines.
    # Documented status ranges carry a real window target instead, so clicking a
    # title routes directly to its category. These are the app's warm table views.
    categories=[('Agents','Agents'),('Schedule','Queue'),('Kindi','Kindi'),('Files','Files'),('System','System'),('Help','Flow')]
    for _,page in categories:ensure_view(store,page)
    windows={}
    for line in tmux('list-windows','-t','bridge','-F','#{window_name}\t#{window_index}').splitlines():
        name,index=line.rsplit('\t',1);windows[name]=int(index)
    targets={'View':int(tmux('display-message','-p','-t',window,'#{window_index}'))}
    for category,page in categories:targets[category]=windows['_'+page.lower()]
    store.setting('menu_targets',json.dumps(targets))


def refresh_agent_header(store=None):
    from .agents import current_agent,status
    store=store or Store();window=store.setting('workbench_window')
    if not window:return
    try:
        agent=current_agent(store)
        pending=store.rows("SELECT count(*) n FROM jobs WHERE agent=? AND state='scheduled' AND due<=?",(agent['id'],time.time()))[0]['n']
        label=status(agent)+(f' / {pending} queued' if pending else '')
        for key,value in [('agent',agent['name']),('agent_state',label)]:
            old=tmux('show-options','-w','-q','-v','-t',window,'@bridge_'+key).strip()
            if old!=value:set_window(window,key,value)
    except (ValueError,RuntimeError):pass


def launch_native(store,agent):
    mode=store.setting('agent_mode:'+agent['id'])
    if mode=='native' and agent['heartbeat'] and time.time()-agent['heartbeat']<10 and pane_exists(agent['pane']):return agent
    if store.rows("SELECT id FROM jobs WHERE agent=? AND state IN ('sending','accepted')",(agent['id'],)):
        raise ValueError('This agent has an active turn. Finish it before switching to native pi.')
    if agent['pid'] and agent['heartbeat'] and time.time()-agent['heartbeat']<15:
        import psutil
        with contextlib.suppress(psutil.NoSuchProcess):
            p=psutil.Process(agent['pid']);cmd=p.cmdline()
            if 'worker' not in cmd or agent['id'] not in cmd:
                raise ValueError('Existing native session needs inspection; refusing to replace its process')
            p.terminate();p.wait(timeout=10)
        if agent['pane']:
            with contextlib.suppress(RuntimeError):tmux('kill-pane','-t',agent['pane'])
    command=env_command(native_command(agent),{'BRIDGE_NATIVE_AGENT':agent['id'],'BRIDGE_NATIVE_NAME':agent['name'],'BRIDGE_PYTHON':sys.executable,'PI_OFFLINE':'1'})
    if mode=='native' and agent['pane']:
        try:dead=tmux('display-message','-p','-t',agent['pane'],'#{pane_dead}').strip()=='1'
        except RuntimeError:dead=False
        if dead:
            tmux('respawn-pane','-t',agent['pane'],'-c',agent['cwd'],command)
            store.execute("UPDATE agents SET state='starting' WHERE id=?",(agent['id'],))
            return store.agent(agent['id'])
    try:tmux('has-session','-t','bridge');exists=True
    except RuntimeError:exists=False
    if exists:
        owner=tmux('show-options','-v','-t','bridge','@bridge_state').strip()
        if owner!=str(state_dir()):raise ValueError('tmux session belongs to another Bridge state directory')
        pane=tmux('new-window','-d','-P','-F','#{pane_id}','-t','bridge','-n','agent-'+agent['id'][:6],'-c',agent['cwd'],command).strip()
    else:
        pane=tmux('new-session','-d','-s','bridge','-n','workbench','-x','160','-y','46','-P','-F','#{pane_id}','-c',agent['cwd'],command).strip()
        tmux('set-option','-t','bridge','@bridge_state',str(state_dir()))
    tmux('set-window-option','-t',pane,'remain-on-exit','on')
    store.setting('agent_mode:'+agent['id'],'native')
    store.execute("UPDATE agents SET pane=?,state='starting' WHERE id=?",(pane,agent['id']))
    return store.agent(agent['id'])


def encode_layout(width,height,primary,shell,vitals,telemetry):
    """A checked tmux layout: left pi+shell, right vitals+inference."""
    if width<60 or height<16:raise ValueError('Window too small for four panes')
    lw=max(38,round(width*.62));rw=width-lw-1
    sh=max(5,round(height*.25));ph=height-sh-1
    vh=max(7,round((height-1)*.50));th=height-vh-1
    def leaf(w,h,x,y,p):return f'{w}x{h},{x},{y},{str(p).lstrip("%")}'
    left=f'{lw}x{height},0,0['+leaf(lw,ph,0,0,primary)+','+leaf(lw,sh,0,ph+1,shell)+']'
    right=f'{rw}x{height},{lw+1},0['+leaf(rw,vh,lw+1,0,vitals)+','+leaf(rw,th,lw+1,vh+1,telemetry)+']'
    body=f'{width}x{height},0,0'+'{'+left+','+right+'}'
    checksum=0
    for char in body:checksum=((checksum>>1)+((checksum&1)<<15)+ord(char))&0xffff
    return f'{checksum:04x},'+body


def layout(window=None):
    store=Store();window=window or store.setting('workbench_window')
    if not window:return
    opts={}
    for line in tmux('show-options','-w','-t',window).splitlines():
        if line.startswith('@bridge_'):
            k,_,v=line.partition(' ');opts[k]=v.strip('"')
    primary=opts.get('@bridge_primary')
    if not primary:return
    width,height,zoom=tmux('display-message','-p','-t',window,'#{window_width} #{window_height} #{window_zoomed_flag}').split()
    width,height=int(width),int(height)
    tmux('set-option','-t','bridge','status-format[1]',navigation(width,opts.get('@bridge_screen','Deck')))
    if width<100 or height<28:
        if zoom!='1':
            tmux('resize-pane','-Z','-t',primary);tmux('set-option','-w','-t',window,'@bridge_autozoom','1')
        return
    if zoom=='1':
        if opts.get('@bridge_autozoom')=='1':tmux('resize-pane','-Z','-t',primary)
        else:return
    tmux('set-option','-w','-t',window,'@bridge_autozoom','0')
    # tmux applies custom layout leaves in pane order, rather than honoring the encoded IDs.
    # Reconcile the order first, including after recovering or swapping a cached page.
    desired=[primary,opts['@bridge_shell'],opts['@bridge_vitals'],opts['@bridge_telemetry']]
    actual=tmux('list-panes','-t',window,'-F','#{pane_id}').splitlines()
    if set(actual)!=set(desired):raise ValueError('Workbench pane set changed; inspect before refitting')
    for index,pane in enumerate(desired):
        if actual[index]!=pane:
            other=actual.index(pane)
            tmux('swap-pane','-d','-s',pane,'-t',actual[index])
            actual[index],actual[other]=actual[other],actual[index]
    value=encode_layout(width,height,primary,opts['@bridge_shell'],opts['@bridge_vitals'],opts['@bridge_telemetry'])
    tmux('select-layout','-t',window,value)


def set_window(window,key,value):
    tmux('set-option','-w','-t',window,'@bridge_'+key,str(value))


def configure(store,window):
    current=tmux('show-options','-w','-q','-v','-t',window,'@bridge_screen').strip()
    if current in ('Deck','Console',''):set_window(window,'screen','Workspace')
    prepare_menu_targets(store,window)
    tmux('set-option','-t','bridge','default-terminal','tmux-256color')
    tmux('set-option','-s','escape-time','30')
    # pi reads modified Enter keys (Shift-Enter, Ctrl-Enter) only with extended keys (tmux 3.2+); this is Bridge's own
    # tmux server, so the user's ~/.tmux.conf does not change
    tmux('set-option','-s','extended-keys','on')
    tmux('set-option','-a','terminal-overrides',',*:Tc')
    for key,value in [('status','2'),('status-position','top'),('status-style','bg=#080808,fg=#9a9a9a'),('message-style','bg=#141414,fg=#caf31c'),('mouse','on'),('display-time','2000')]:tmux('set-option','-t','bridge',key,value)
    tmux('set-option','-t','bridge','status-format[0]',HEADER_FORMAT)
    tmux('set-option','-t','bridge','status-format[1]',navigation(160,'Workspace'))
    for key,value in [('pane-border-style','fg=#232323,bg=#080808'),('pane-active-border-style','fg=#caf31c,bg=#080808'),('window-style','bg=#080808,fg=#f3f3f1'),('window-active-style','bg=#080808,fg=#f3f3f1'),('window-size','latest')]:tmux('set-window-option','-t',window,key,value)
    for i,page in enumerate(PAGES[:9],1):
        cmd=env_command([sys.executable,'-m','kindi_bridge','screen',page])
        tmux('bind-key','-n',f'F{i}','run-shell','-b',cmd)
        # Prefix + digit is an equally discoverable fallback for SSH terminals without F keys.
        tmux('bind-key',str(i%10),'run-shell','-b',cmd)
    popup=env_command([sys.executable,'-m','kindi_bridge','menu'],{'BRIDGE_CLIENT':'#{client_name}'})
    for key in ['F10','M-m']:tmux('bind-key','-n',key,'run-shell','-b',popup)
    tmux('bind-key','?','run-shell','-b',popup)
    from .menus import SHORTCUTS
    for category,key in SHORTCUTS.items():
        tmux('bind-key','-n',key,'run-shell','-b',env_command([sys.executable,'-m','kindi_bridge','menu','--category',category],{'BRIDGE_CLIENT':'#{client_name}'}))
    tmux('bind-key','-n','M-w','run-shell','-b',env_command([sys.executable,'-m','kindi_bridge','screen','Workspace']))
    # A custom menu bar is not tmux's window list: depending on the version/format,
    # clicks are classified as StatusLeft, StatusRight or StatusDefault instead.
    for area in ['Status','StatusLeft','StatusRight','StatusDefault']:
        for button in ['MouseDown1','MouseDown3']:
            mouse=env_command([sys.executable,'-m','kindi_bridge','menu','--category','#{window_name}'],{'BRIDGE_CLIENT':'#{client_name}'}) if area=='Status' else popup
            args=['-b','-t','='] if area=='Status' else ['-b']
            tmux('bind-key','-n',button+area,'run-shell',*args,mouse)
    for key in ['WheelUpStatus','WheelDownStatus']:tmux('unbind-key','-q','-n',key)
    tmux('bind-key','0','run-shell','-b',env_command([sys.executable,'-m','kindi_bridge','screen','Workspace']))
    for key,role in [('e','pi'),('c','shell')]:
        cmd=env_command([sys.executable,'-m','kindi_bridge','focus',role])
        tmux('bind-key',key,'run-shell','-b',cmd)
    tmux('bind-key','z','resize-pane','-Z')
    tmux('bind-key','r','run-shell','-b',env_command([sys.executable,'-m','kindi_bridge','layout']))
    for key,direction in [('Left','L'),('Right','R'),('Up','U'),('Down','D')]:tmux('bind-key','-r',key,'resize-pane','-'+direction,'3')
    hook=env_command([sys.executable,'-m','kindi_bridge','layout'])
    tmux('set-hook','-t','bridge','client-resized','run-shell -b '+shlex.quote(hook))
    tmux('set-hook','-t','bridge','client-attached','run-shell -b '+shlex.quote(hook))
    refresh_agent_header(store)


def build(store=None):
    store=store or Store();ensure_service()
    try:agent=store.agent()
    except ValueError:
        root=Path.home()/'kindi-bridge-work';root.mkdir(exist_ok=True)
        agent=store.add_agent('default',root,pi_command())
    existing=store.setting('workbench_window')
    if existing:
        try:
            tmux('display-message','-p','-t',existing,'#{window_id}')
            native_pane=store.setting('native_pi_pane')
            if not pane_exists(native_pane):
                recovered=launch_native(store,agent)
                if tmux('display-message','-p','-t',recovered['pane'],'#{window_id}').strip()!=existing:
                    tmux('join-pane','-d','-s',recovered['pane'],'-t',existing)
                if store.setting('workbench_primary')==native_pane:
                    store.setting('workbench_primary',recovered['pane']);set_window(existing,'primary',recovered['pane'])
                store.setting('native_pi_pane',recovered['pane']);set_window(existing,'pi',recovered['pane'])
            configure(store,existing);layout(existing);return existing
        except RuntimeError:pass
    agent=launch_native(store,agent)
    window=tmux('display-message','-p','-t',agent['pane'],'#{window_id}').strip()
    shell=tmux('split-window','-d','-v','-l','25%','-P','-F','#{pane_id}','-t',agent['pane'],'-c',agent['cwd'],env_command(['bash','--noprofile','--rcfile',str(ASSETS/'shell.bash'),'-i'])).strip()
    vitals=tmux('split-window','-d','-h','-f','-l','37%','-P','-F','#{pane_id}','-t',window,env_command([sys.executable,'-m','kindi_bridge','panel','vitals'])).strip()
    telemetry=tmux('split-window','-d','-v','-l','50%','-P','-F','#{pane_id}','-t',vitals,env_command([sys.executable,'-m','kindi_bridge','panel','inference'])).strip()
    for key,value in [('primary',agent['pane']),('pi',agent['pane']),('shell',shell),('vitals',vitals),('telemetry',telemetry),('screen','Workspace')]:set_window(window,key,value)
    for key,value in [('workbench_window',window),('workbench_primary',agent['pane']),('native_pi_pane',agent['pane']),('shell_pane',shell)]:store.setting(key,value)
    configure(store,window);layout(window);tmux('select-pane','-t',agent['pane'])
    store.setting('native_workbench','1')
    return window


def screen(page):
    page={'Deck':'Workspace','Console':'Workspace'}.get(page,page)
    if page not in PAGES:raise ValueError('Unknown screen')
    store=Store();window=store.setting('workbench_window')
    if not window:window=build(store)
    current=store.setting('workbench_primary')
    if page=='Workspace':target=store.setting('native_pi_pane')
    else:
        target=ensure_view(store,page)
    # Two navigation commands arriving together must not swap the wrong parked pane.
    import fcntl
    with open(state_dir()/'navigation.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        current=store.setting('workbench_primary')
        if current!=target:
            tmux('swap-pane','-d','-s',target,'-t',current)
            store.setting('workbench_primary',target);set_window(window,'primary',target)
        set_window(window,'screen',page);tmux('select-window','-t',window)
        layout(window);tmux('select-pane','-t',target)


def focus(role):
    store=Store()
    if role=='pi':screen('Workspace')
    else:tmux('select-pane','-t',store.setting('shell_pane'))


def show_agent(ident):
    store=Store();agent=launch_native(store,store.agent(ident))
    store.setting('native_pi_pane',agent['pane'])
    store.setting('active_agent',agent['id'])
    refresh_agent_header(store);screen('Workspace')


def menu(category='Agents',x=None,y=None):
    from .menus import category_at
    category={'workbench':'View','Workspace':'View','_agents':'Agents','_queue':'Schedule','_kindi':'Kindi','_files':'Files','_knowledge':'Files','_system':'System','_flow':'Help','Knowledge':'Files'}.get(category,category)
    if x is not None:category=category_at(x) if y in (None,1) else 'Agents'
    store=Store();window=store.setting('workbench_window')
    width,height=map(int,tmux('display-message','-p','-t',window,'#{window_width} #{window_height}').split())
    command=env_command([sys.executable,'-m','kindi_bridge','menu-ui','--category',category],{'BRIDGE_CLIENT':os.environ.get('BRIDGE_CLIENT','')})
    popup_height=min(26,height)
    args=['display-popup','-E','-x','0','-y',str(popup_height+2),'-w',str(min(100,width)),'-h',str(popup_height),command]
    client=os.environ.get('BRIDGE_CLIENT')
    if client:args[1:1]=['-c',client]
    # A menu may stay open while the user writes a mission; don't apply the collector timeout.
    result=subprocess.run(['tmux','-L',os.environ.get('BRIDGE_TMUX_SOCKET','kindi-bridge'),*args],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr.strip())


def check_pi():
    """Bridge's agents run pi. A pi that cannot start would close the first tmux window at once, and the user would
    see only "no server running". Check it first and say what is wrong."""
    from .runtime import pi_command
    pi=pi_command()[0]
    try:result=subprocess.run([pi,'--version'],capture_output=True,text=True,timeout=60)
    except FileNotFoundError:
        raise SystemExit('Bridge: the pi coding agent is not installed. Install it (needs Node.js 22.19 or later):\n'
                         '  npm install -g --prefix ~/.local @earendil-works/pi-coding-agent@1.3.0')
    except subprocess.TimeoutExpired:raise SystemExit(f'Bridge: {pi} --version did not answer in 60 s')
    if result.returncode:
        try:node=subprocess.run(['node','--version'],capture_output=True,text=True,timeout=10).stdout.strip()
        except (OSError,subprocess.TimeoutExpired):node='not found'
        detail=(result.stderr or result.stdout).strip().splitlines()
        raise SystemExit(f'Bridge: pi does not start ({pi}; Node.js {node or "?"}). pi 1.3.0 needs Node.js 22.19 or later.\n'
                         +('  '+detail[-1] if detail else ''))


def main():
    check_pi()
    window=build()
    socket=os.environ.get('BRIDGE_TMUX_SOCKET','kindi-bridge')
    # Suppress nested-session refusal; this app owns a separate tmux server.
    env=dict(os.environ);env.pop('TMUX',None)
    return subprocess.call(['tmux','-L',socket,'attach-session','-t','bridge'],env=env)
