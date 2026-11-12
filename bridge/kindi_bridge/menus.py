# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""Borland-style menu bar, pull-downs, and task-oriented dialogs."""
from __future__ import annotations

from pathlib import Path
import time

from .agents import current_agent,roster,create,assign,status
from .store import Store

CATEGORIES=['Agents','Schedule','Kindi','Files','System','View','Help']
PAGE_CATEGORY={'Workspace':'Agents','Deck':'Agents','Console':'Agents','Agents':'Agents','Queue':'Schedule','Flow':'Schedule','Kindi':'Kindi','Files':'Files','Knowledge':'Files','Processes':'System','Ports':'System','System':'System'}
# Alt-key shortcuts for the menus (Alt-W returns to the conversation)
SHORTCUTS={'Agents':'M-a','Schedule':'M-s','Kindi':'M-k','Files':'M-f','System':'M-y','View':'M-v','Help':'M-h'}


def category_at(x):
    pos=1
    for name in CATEGORIES:
        if pos<=x<pos+len(name)+2:return name
        pos+=len(name)+3
    return 'Agents'


def bar(page,targets=None):
    active=PAGE_CATEGORY.get(page,'Agents');parts=['#[bg=#080808] ']
    for name in CATEGORIES:
        style='#[bg=#caf31c,fg=#080808,bold]' if name==active else '#[bg=#080808,fg=#f3f3f1,nobold]'
        marker=f',range=window|{targets[name]}' if targets and name in targets else ''
        parts.append(style+('#['+marker[1:]+']' if marker else '')+' '+name+' #[norange,bg=#080808] ')
    parts.append('#[align=right,fg=#9a9a9a,nobold] Click a menu | Alt-M | F10 ')
    return ''.join(parts)


def items(category,store):
    if category=='Agents':
        rows=[('Open conversation','workspace'),('New agent…','new'),('Send an instruction…','mission'),('All agents','agents')]
        rows += [(f'{"▸" if a["current"] else " "} {a["name"]:<18} {a["status"]}'+(f' · {a["pending"]} queued' if a['pending'] else ''), 'agent:'+a['id']) for a in roster(store)]
        return rows
    if category=='Schedule':return [('Schedule a prompt…','schedule'),('Check until done…','check'),('Scheduled prompts','queue'),('Activity log','activity')]
    if category=='Kindi':return [('Servers & features','kindi'),('Use a server for agents…','kindi-use'),('Connect to an address…','kindi-connect'),('Full report','kindi-report')]
    if category=='Files':return [('Project files','files'),('Add a knowledge folder…','knowledge-add'),('Knowledge & search','knowledge')]
    if category=='System':return [('Performance','system'),('Processes','processes'),('Listening ports','ports'),('Cron & timers','cron')]
    if category=='View':return [('Command pane','command'),('Zoom / restore','zoom'),('Refit panels','refit'),('Detach (agents keep running)','detach')]
    return [('Getting started','help'),('Keys and mouse','keys'),('About Bridge','about')]


def use_server(dialog,store):
    from . import kindi
    found=kindi.servers(store)
    choices=[(base,f'{"▸" if source=="current" else " "} {label[:24]:<24} {base:<26} {kindi.summary(kindi.probe(base,key,1.5))}') for label,base,key,source in found]
    base=dialog.select('Use which Kindi server for new agents?',choices,0,'No Kindi server found. Use Kindi > Connect to an address.')
    if not base:return None
    label=next((l for l,b,_,_ in found if b==base),None);key=next((k for _,b,k,_ in found if b==base),None)
    return connect(dialog,store,base,label,key)


def connect(dialog,store,base=None,label=None,key=None):
    from . import kindi
    if base is None:
        result=dialog.form('Connect to a Kindi server',[('address','Address (host:port)','127.0.0.1:8093'),('key','API key (empty if none)','')],'Connect')
        if result is None:return None
        base,key=result['address'],result['key'] or None
    try:outcome=kindi.use(store,base,label,key)
    except ValueError as e:
        dialog.ui.details('Could not connect',str(e));return None
    text=f'New agents use {outcome["model"]} at {outcome["base"]} ({outcome["ctx"]:,} tokens of context).'
    text+=f'\n\n{outcome["note"]}' if outcome.get('note') else '\n\nAgents that are running keep their session. Start a new agent to use this server.'
    dialog.ui.details('Kindi server',text);return outcome


def new_agent(dialog,store):
    try:folder=current_agent(store)['cwd']
    except ValueError:folder=str(Path.home()/'kindi-bridge-work')
    result=dialog.form('New agent',[('name','Name',''),('mission','Mission — what should this agent do?',''),('folder','Working folder',folder)],'Create & Start')
    if result is None:return None
    try:agent=create(store,result['name'],result['mission'],result['folder'])
    except ValueError as e:
        dialog.ui.details('Could not create agent',str(e));return None
    from .workbench import show_agent
    show_agent(agent['id']);return agent


def mission(dialog,store,ident=None):
    ident=ident or dialog.agent(store,'Who should receive the mission?')
    if not ident:return None
    agent=store.agent(ident)
    result=dialog.form('Mission for '+agent['name'],[('mission','Mission or prompt','')],'Send mission')
    if result is None:return None
    try:job=assign(store,ident,result['mission'])
    except ValueError as e:dialog.ui.details('Mission',str(e));return None
    from .workbench import show_agent
    show_agent(ident);return job


def schedule(dialog,store,check=False):
    from .cli import parse_time,duration
    from .runtime import CHECK_PROMPT
    ident=dialog.agent(store,'Schedule for which agent?')
    if not ident:return
    agent=store.agent(ident)
    fields=[('prompt','Prompt',CHECK_PROMPT if check else ''),('when','When (now / in 20m / 16:30)','in 20m')]
    if check:fields += [('task','Job to check',store.setting('mission:'+ident) or ''),('interval','Check every','30m'),('limit','Maximum checks','6')]
    result=dialog.form(('Check until done: ' if check else 'Schedule prompt: ')+agent['name'],fields,'Schedule')
    if result is None:return
    try:
        store.schedule(result['prompt'],parse_time(result['when']),ident,interval=duration(result['interval']) if check else 0,
                       limit=int(result['limit']) if check else 1,task=result.get('task'),kind='check' if check else 'prompt')
    except ValueError as e:dialog.ui.details('Could not schedule prompt',str(e));return
    from .workbench import screen
    screen('Queue')


def execute(action,dialog,store):
    from . import workbench
    pages={'workspace':'Workspace','agents':'Agents','queue':'Queue','activity':'Flow','kindi':'Kindi','files':'Files','knowledge':'Knowledge','processes':'Processes','ports':'Ports','system':'System'}
    if action in pages:workbench.screen(pages[action])
    elif action.startswith('agent:'):workbench.show_agent(action[6:])
    elif action=='new':new_agent(dialog,store)
    elif action=='mission':mission(dialog,store)
    elif action=='kindi-use':use_server(dialog,store)
    elif action=='kindi-connect':connect(dialog,store)
    elif action=='kindi-report':
        from .kindi import report
        dialog.ui.details('Kindi status & features',report(store))
    elif action=='keys':dialog.ui.details('Keys and mouse',KEYS)
    elif action=='about':
        from . import __version__
        from .kindi import KINDI_VERSION
        dialog.ui.details('About Kindi Bridge',f'Kindi Bridge {__version__}\nFor Kindi {KINDI_VERSION}\n\n'
            'A persistent terminal workspace for agents that run on Kindi: conversations, scheduled prompts, '
            'completion checks, local knowledge folders, and the machine they run on.\n\n'
            'State: '+str(__import__('kindi_bridge.store',fromlist=['state_dir']).state_dir()))
    elif action=='schedule':schedule(dialog,store)
    elif action=='check':schedule(dialog,store,True)
    elif action=='command':workbench.focus('shell')
    elif action=='zoom':workbench.tmux('resize-pane','-Z','-t',store.setting('workbench_primary'))
    elif action=='refit':workbench.layout()
    elif action=='detach':
        import os
        client=os.environ.get('BRIDGE_CLIENT')
        workbench.tmux('detach-client',*(['-t',client] if client else []))
    elif action=='cron':
        from .system import schedule_inventory
        dialog.ui.details('Cron & timers',schedule_inventory())
    elif action=='knowledge-add':
        result=dialog.form('Add knowledge folder',[('folder','Folder on this machine','')],'Index folder')
        if result:
            from .knowledge import add_collection
            collection=add_collection(store,result['folder'])
            store.execute('UPDATE agents SET collection=? WHERE id=?',(collection,current_agent(store)['id']))
            workbench.screen('Knowledge')
    elif action=='help':dialog.ui.details('Getting started',HELP)


HELP=('1. Connect to Kindi\nKindi > Servers & features lists the Kindi servers that Bridge found. Select one and press u, '
      'or use Kindi > Use a server for agents. New agents then use that server.\n\n'
      '2. Start an agent\nAgents > New agent. Enter a name and a mission, then choose Create & Start. The work starts at once.\n\n'
      '3. Talk to it\nThe conversation is the main pane. Type the next instruction into it. Agents > Send an instruction '
      'sends to any agent; a busy agent gets it at its next safe turn boundary.\n\n'
      '4. Switch agents\nThe Agents menu lists every agent with its status. Select a name to open its conversation.\n\n'
      '5. Work later\nSchedule > Schedule a prompt sends a prompt at a time. Schedule > Check until done repeats a completion '
      'check until the agent confirms the job, a limit, or a deadline.\n\n'
      '6. Files and knowledge\nFiles > Project files browses and edits the agent\'s folder. Files > Add a knowledge folder '
      'indexes reference documents for the agent.\n\n'
      '7. Leave\nView > Detach. The agents keep running. Start Bridge again to come back.')

KEYS=('Menus\n  Click a menu title, or press Alt-M or F10. Left/Right: menus. Up/Down: items. Enter: choose. Esc: close.\n'
      '  Alt-A Agents  Alt-S Schedule  Alt-K Kindi  Alt-F Files  Alt-Y System  Alt-V View  Alt-H Help\n'
      '  Alt-W returns to the conversation.\n\n'
      'Screens\n  F1 Conversation  F2 Agents  F3 Scheduled prompts  F4 Kindi  F5 Files  F6 Knowledge  F7 Performance\n'
      '  F8 Processes  F9 Ports. Over SSH without F-keys: Ctrl-b then 1-9; Ctrl-b 0 returns to the conversation.\n\n'
      'Tables\n  Up/Down, PgUp/PgDn, Home/End, mouse click and wheel. / filters. Esc or q returns to the conversation.\n\n'
      'Panes\n  Ctrl-b e: conversation  Ctrl-b c: command pane  Ctrl-b z: zoom  Ctrl-b r: refit  Ctrl-b arrows: resize.\n\n'
      'Forms\n  Tab or Enter: next field. Ctrl-S: submit. Ctrl-E: edit a long text in the full editor. Esc: cancel.')


def run(category='Agents',action=None):
    import curses
    from .dialogs import Dialogs
    if category not in CATEGORIES:category='Agents'
    def app(s):
        dialog=Dialogs(s);store=Store();index=CATEGORIES.index(category);selected=0;top=0
        def perform(chosen):
            try:execute(chosen,dialog,store)
            except Exception as error:dialog.ui.details('Could not complete action',str(error))
        try:
            if action:perform(action);return
            s.timeout(200)
            while True:
                h,w=s.getmaxyx();name=CATEGORIES[index];choices=items(name,store);selected=min(selected,len(choices)-1)
                height=max(1,h-5)
                if selected<top:top=selected
                if selected>=top+height:top=selected-height+1
                s.erase();dialog.ui.frame('KINDI // MENU')
                x=2;spans=[]
                for i,title in enumerate(CATEGORIES):
                    attr=dialog.ui.colors['selected'] if i==index else dialog.ui.colors['lime']
                    dialog.ui.put(1,x,' '+title+' ',attr);spans.append((x,x+len(title)+2,i));x+=len(title)+3
                for i,(title,_) in enumerate(choices[top:top+height]):
                    attr=dialog.ui.colors['selected'] if top+i==selected else dialog.ui.colors['text']
                    dialog.ui.put(i+3,3,title[:max(1,w-6)].ljust(max(1,w-6)),attr)
                dialog.ui.put(h-2,2,'Left/Right Menu   Up/Down Choose   Enter Open   Esc Close',dialog.ui.colors['muted']);s.refresh()
                key=s.getch()
                if key==27:return
                if key in (curses.KEY_LEFT,curses.KEY_RIGHT,9):index=(index+(-1 if key==curses.KEY_LEFT else 1))%len(CATEGORIES);selected=top=0
                elif key in (curses.KEY_UP,curses.KEY_DOWN):selected=(selected+(-1 if key==curses.KEY_UP else 1))%len(choices)
                elif key in (10,13,curses.KEY_ENTER):perform(choices[selected][1]);return
                elif key==curses.KEY_MOUSE:
                    try:
                        _,x,y,_,buttons=curses.getmouse()
                        if buttons&(curses.BUTTON1_CLICKED|curses.BUTTON1_PRESSED):
                            if y==1:
                                for lo,hi,which in spans:
                                    if lo<=x<hi:index=which;selected=top=0;break
                            elif 3<=y<h-2 and top+y-3<len(choices):perform(choices[top+y-3][1]);return
                    except curses.error:pass
        finally:dialog.close()
    curses.wrapper(app)
