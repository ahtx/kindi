# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
from __future__ import annotations

import concurrent.futures
import curses
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import time

from . import __version__
from .runtime import CHECK_PROMPT, ensure_service, launch_agent, tmux
from .store import Store
from .system import listeners, metrics, processes, schedule_inventory, gpu_metrics
from .visual import curses_palette

SCREENS = ["Deck", "Agents", "Queue", "Kindi", "Flow", "Files", "Knowledge", "Processes", "Ports", "System", "Console"]
# The visible screen titles (the menu items use the same words)
TITLES = {"Deck": "Agents", "Queue": "Scheduled prompts", "Kindi": "Servers & features", "Flow": "Activity log",
          "Files": "Project files", "Knowledge": "Knowledge & search", "Ports": "Listening ports", "System": "Performance"}


def safe(text):
    return "".join(c if c.isprintable() else " " for c in str(text))


class UI:
    def __init__(self, screen, root=None, page='Deck'):
        self.s = screen
        self.store = Store()
        self.root = Path(root or os.getcwd()).resolve()
        self.folder = self.root
        self.view = SCREENS.index(page)
        self.embedded = os.environ.get('BRIDGE_WORKBENCH')=='1'
        self.selected = {x: None for x in SCREENS}
        self.position = {x: 0 for x in SCREENS}
        self.offset = 0
        self.message = "Use the menu bar to create agents, send missions, and switch conversations."
        self.filter = ""
        self.undo = None
        self.cache = {}
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.pending = {}
        self.fresh = {}
        self.rows = []
        self.s.timeout(50)
        curses.curs_set(0)
        with __import__('contextlib').suppress(curses.error):curses.set_escdelay(25)
        with __import__('contextlib').suppress(curses.error):
            curses.mousemask(curses.ALL_MOUSE_EVENTS)
        with __import__('contextlib').suppress(curses.error):
            curses.use_default_colors()
        self.colors=curses_palette()
        self.s.bkgd(' ',self.colors['text'])
        self.last_refresh = 0

    def put(self, y, x, text, attr=0):
        h, w = self.s.getmaxyx()
        if 0 <= y < h and x < w:
            try:
                self.s.addnstr(y, max(0, x), safe(text), max(0, w-max(0,x)), attr or self.colors['text'])
            except curses.error:
                pass

    def frame(self,title,subtitle=''):
        h,w=self.s.getmaxyx();edge=self.colors['edge']
        if w<6 or h<4:return
        self.put(0,0,'╭'+'─'*(w-2)+'╮',edge)
        self.put(0,2,' '+title.upper()+' ',self.colors['lime']|curses.A_BOLD)
        for y in range(1,h-1):
            self.put(y,0,'│',edge);self.put(y,w-1,'│',edge)
        self.put(h-1,0,'╰'+'─'*(w-2)+'╯',edge)
        if subtitle:self.put(1,2,subtitle[:max(0,w-4)],self.colors['muted'])

    def ask(self, title, default=""):
        value = default
        self.s.timeout(-1)
        curses.curs_set(1)
        try:
            while True:
                h, w = self.s.getmaxyx()
                y = max(0, h-3)
                self.put(y,2,' '*max(0,w-4),self.colors['selected'])
                self.put(y,2,(title+': '+value[-max(1,w-len(title)-7):])[:max(1,w-4)],self.colors['selected'])
                self.s.refresh()
                c = self.s.get_wch()
                if c in ("\n", "\r"):
                    return value
                if c == "\x1b":
                    return None
                if c in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                    value = value[:-1]
                elif c == "\x15":
                    value = ""
                elif isinstance(c, str) and c.isprintable():
                    value += c
        finally:
            curses.curs_set(0)
            self.s.timeout(50)

    def details(self, title, body):
        pos = 0
        self.s.timeout(100)
        try:
            while True:
                h, w = self.s.getmaxyx()
                lines = []
                for line in str(body).splitlines():
                    lines += textwrap.wrap(safe(line), max(1,w-5), replace_whitespace=False) or [""]
                self.s.erase()
                self.frame('KINDI // '+title)
                for i,line in enumerate(lines[pos:pos+max(1,h-4)]):self.put(i+1,2,line)
                self.put(h-2,2,'↑↓ PgUp/PgDn Scroll   Home/End   Esc/Enter Back',self.colors['lime'])
                self.s.refresh()
                key = self.s.getch()
                if key in (27,10,13,ord('q')):
                    return
                delta = {curses.KEY_DOWN:1,curses.KEY_UP:-1,curses.KEY_NPAGE:max(1,h-4),curses.KEY_PPAGE:-max(1,h-4)}.get(key,0)
                pos = max(0, min(max(0,len(lines)-h+3), pos+delta))
                if key == curses.KEY_HOME:
                    pos = 0
                if key == curses.KEY_END:
                    pos = max(0,len(lines)-h+3)
        finally:
            self.s.timeout(50)

    def background(self, name, fn, interval=2):
        if name in self.pending and self.pending[name].done():
            try:
                self.cache[name] = self.pending[name].result()
            except Exception as e:
                self.cache[name] = [] if name in {"Processes", "Ports"} else str(e)
                self.message = f"{name}: {e}"
            del self.pending[name]
            self.fresh[name] = time.time()
        if name not in self.pending and time.time()-self.fresh.get(name,0)>interval:
            self.pending[name] = self.pool.submit(fn)
        return self.cache.get(name, [])

    def data(self):
        name = SCREENS[self.view]
        if name == "Console":
            try:
                agent = self.store.agent()
            except ValueError:
                return 'PI CONSOLE', [{'id':'empty','label':'No agent yet. Use Agents > New agent.'}]
            if agent['pane']:
                raw = self.background('console-'+agent['id'],lambda:tmux('capture-pane','-p','-S','-120','-t',agent['pane']),0.3)
            else:
                raw = 'The agent is not running. Open it from the Agents menu.'
            rows = [{'id':str(i),'label':line} for i,line in enumerate(str(raw or 'Waiting for output...').splitlines())]
            return f'{agent["name"]} | {agent["state"]} | n Prompt | End Latest output', rows
        if name in {"Deck", "Agents"}:
            from .agents import roster
            rows = roster(self.store)
            for r in rows:
                r["label"] = f'{">" if r["current"] else " "} {r["name"]:<18} {r["status"]:<18} {r["pending"]:>3}   {r["mission"]}'
            return "  AGENT              STATUS             QUEUED MISSION", rows
        if name == "Queue":
            rows = self.store.rows("SELECT j.*,a.name FROM jobs j JOIN agents a ON a.id=j.agent WHERE j.state NOT IN ('deleted','completed','confirmed') ORDER BY j.due,j.created")
            for r in rows:
                date = dt.datetime.fromtimestamp(r["due"]).astimezone().strftime("%m-%d %H:%M:%S %Z")
                r["label"] = f'{date:<26} {r["name"]:<13} {r["state"]:<11} {r["attempts"]}/{r["max_checks"]} {r["prompt"]}'
            return "TIME                       TARGET        STATE       RUNS PROMPT", rows
        if name == "Kindi":
            from . import kindi
            def scan():
                return [(label,source,kindi.probe(base,key)) for label,base,key,source in kindi.servers(self.store)]
            probes = self.background("kindi", scan, 5) or []
            # A placeholder keeps the first row a server row, so the selection does not move when the scan lands.
            rows = [] if probes else [{"id":"server:pending","kind":"pending","label":"  Looking for Kindi servers…"}]
            for label,source,info in probes:
                rows.append({"id":"server:"+info["base"],"kind":"server","label_text":label,"base":info["base"],"info":info,
                             "label":f'{">" if source=="current" else " "} {label[:22]:<22} {info["base"][7:][:21]:<21} {kindi.summary(info)}'})
            current = next((i for _,s,i in probes if s=="current"), probes[0][2] if probes else {})
            for f in kindi.FEATURES:
                value = kindi.live(f,current)
                rows.append({"id":"feature:"+f[1],"kind":"feature","feature":f,
                             "label":f'  {f[0]:<7} {f[1][:40]:<40} {"preview" if f[2]=="preview" else "":<8} {value or ""}'})
            return "  SERVER / AREA          ADDRESS / FEATURE        STATE", rows
        if name == "Flow":
            rows = self.store.rows("SELECT e.*,a.name FROM events e LEFT JOIN agents a ON a.id=e.agent ORDER BY e.id DESC LIMIT 300")
            for r in rows:
                r["label"] = f'{dt.datetime.fromtimestamp(r["at"]).strftime("%H:%M:%S")} {r["name"] or "bridge":<15} {r["kind"]:<22} {r["detail"]}'
            return "TIME     AGENT           EVENT                  DETAIL", rows
        if name == "Files":
            if self.embedded:
                from .agents import current_agent
                agent=current_agent(self.store)
                if getattr(self,'file_agent',None)!=agent['id']:
                    self.root=Path(agent['cwd']);self.folder=self.root;self.file_agent=agent['id']
            rows = []
            if self.folder != self.root:
                rows.append({"id": str(self.folder.parent), "path": str(self.folder.parent), "label": "[..] parent folder"})
            for p in sorted(self.folder.iterdir(), key=lambda p:(not p.is_dir(),p.name.lower())):
                if p.name in {".git", ".venv", "node_modules", "__pycache__"} or p.is_symlink():
                    continue
                rows.append({"id":str(p),"path":str(p),"label":("[dir] " if p.is_dir() else "      ")+p.name})
            return str(self.folder), rows
        if name == "Knowledge":
            rows = self.store.rows("SELECT c.*, (SELECT COUNT(*) FROM documents d WHERE d.collection=c.id) AS files,(SELECT COUNT(*) FROM documents d WHERE d.collection=c.id AND d.error IS NOT NULL) AS errors FROM collections c ORDER BY root")
            for r in rows:
                r["label"] = f'{r["state"]:<14} {r["files"]:>5} files {r["errors"]:>3} issues  {r["root"]}'
            return "STATE          FILES       ERRORS     FOLDER", rows
        if name == "Processes":
            rows = self.background(name, processes) or []
            for r in rows:
                r["label"] = f'{r["pid"]:>7} {r["ppid"]:>7} {r["user"][:14]:<14} {r["cpu"]:>5.1f} {r["mb"]:>8.1f} {r["command"]}'
            return "    PID    PPID USER             CPU%   RAM MiB COMMAND", rows
        if name == "Ports":
            rows = self.background(name, listeners) or []
            for r in rows:
                r["label"] = f'{r["proto"]:<4} {r["host"]:<25} {r["port"]:>5} {str(r["pid"] or "?"):>7} {r["exposure"]:<9} {r["name"]}'
            return "TYPE LOCAL ADDRESS              PORT     PID BIND      OWNER", rows
        rows = [{"id":"host", "label": self.background("metrics",metrics) or "Loading host metrics..."}]
        gpu = self.background('gpu', gpu_metrics, 3)
        rows.append({'id':'gpu-header','label':'GPU: index, name, utilization, used/total MiB, temperature, power'})
        for i,line in enumerate(str(gpu or 'GPU data pending/unavailable').splitlines()):
            rows.append({'id':f'gpu{i}','label':line})
        def health():
            import urllib.request
            with urllib.request.urlopen(self.store.setting("endpoint") or "http://127.0.0.1:8093/health",timeout=2) as response:
                return json.dumps(json.load(response),indent=2)
        raw = self.background("health", health)
        for i,line in enumerate(str(raw or "Loading inference health...").splitlines()):
            rows.append({"id":f"health{i}","label":line})
        return "LIVE SYSTEM / INFERENCE", rows

    def table(self):
        try:
            header, rows = self.data()
        except Exception as e:
            header, rows = SCREENS[self.view], []
            self.message = str(e)
        name = SCREENS[self.view]
        if self.filter:
            rows = [r for r in rows if self.filter.lower() in r["label"].lower()]
        selected = self.selected[name]
        pos = next((i for i,r in enumerate(rows) if r["id"] == selected), min(self.position[name],max(0,len(rows)-1)))
        self.position[name] = pos
        self.selected[name] = rows[pos]["id"] if rows else None
        self.rows = rows
        h,w = self.s.getmaxyx()
        page = max(1,h-8)
        self.offset = min(self.offset,max(0,len(rows)-page))
        if pos < self.offset:
            self.offset = pos
        if pos >= self.offset+page:
            self.offset = pos-page+1
        self.s.erase()
        loading=name in self.pending and name not in self.cache
        subtitle='Connecting to live data…' if loading else f'{len(rows)} items'+(' · filter: '+self.filter if self.filter else ' · live')
        self.frame('KINDI // '+TITLES.get(name,name),subtitle)
        self.put(2,2,header[:max(0,w-4)],self.colors['muted']|curses.A_BOLD)
        for i,r in enumerate(rows[self.offset:self.offset+page]):
            y=i+3
            attr=self.colors['selected'] if self.offset+i==pos else self.colors['text']
            self.put(y,2,r['label'][:max(0,w-4)].ljust(max(0,w-4)),attr)
        if not rows:
            self.put(5,3,'Collecting live data…' if loading else 'Nothing here yet.',self.colors['muted'])
            hint='r Refresh   / Filter' if name in {'Processes','Ports'} else 'a Add   r Refresh   / Filter'
            self.put(6,3,hint,self.colors['lime'])
        self.put(h-4,2,self.message[:max(0,w-4)],self.colors['muted'])
        specific={"Processes":"k Terminate  K Force kill", "Ports":"k Stop owner  K Force kill owner", "Queue":"a Schedule  c Check-until-done  d Delete  u Undo  space Pause", "Knowledge":"a Add folder  b Attach  s Search  r Reindex", "Files":"Enter Open  e Edit  g Git diff", "Agents":"Enter Open agent  n New agent  m Mission  i Details", "Kindi":"Enter Details  u Use for agents  c Connect to an address  i Full report  r Refresh"}.get(name,"n Send prompt  s Cron/timers")
        self.put(h-3,2,specific[:max(0,w-4)],self.colors['lime'])
        legend='↑↓ Select  Enter Open  / Filter  Alt-M Menu  Esc Workspace' if self.embedded else '1-9/0 Screens   ↑↓ Select   Enter Open   / Filter   q Exit'
        self.put(h-2,2,legend[:max(0,w-4)],self.colors['muted'])
        self.s.refresh()

    def row(self):
        return self.rows[self.position[SCREENS[self.view]]] if self.rows else None

    def target(self):
        r = self.row()
        if SCREENS[self.view] in {"Deck","Agents"} and r:return self.store.agent(r['id'])
        from .agents import current_agent
        return current_agent(self.store)

    def external(self, args, cwd=None):
        curses.def_prog_mode()
        curses.endwin()
        try:
            return subprocess.run(args,cwd=cwd).returncode
        finally:
            curses.reset_prog_mode()
            self.s.clear()
            self.s.refresh()

    def edit(self, path):
        path=Path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError("Select a regular source file")
        if path.stat().st_size>2*1024*1024:
            raise ValueError("Quick editor limit is 2 MiB")
        original=path.read_bytes()
        original.decode('utf-8')
        if b'\0' in original:
            raise ValueError("Binary file; quick editor handles UTF-8 text")
        # Edit a same-directory copy; compare the original again immediately before replacing.
        fd,tmp=tempfile.mkstemp(prefix='.bridge-edit-',suffix=path.suffix,dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(original)
            os.chmod(tmp,path.stat().st_mode)
            if os.environ.get('EDITOR'):
                self.external(shlex.split(os.environ['EDITOR'])+[tmp])
            else:
                edited=self.edit_buffer(path,original.decode('utf-8'))
                if edited is None:
                    self.message='Edit cancelled';return
                Path(tmp).write_text(edited,encoding='utf-8',newline='')
            edited=Path(tmp).read_bytes()
            if edited == original:
                self.message='No edits'
                return
            if path.is_symlink() or path.read_bytes()!=original:
                recovery=Path(tmp).with_name(Path(tmp).name+'.conflict')
                os.replace(tmp,recovery)
                raise ValueError(f'File changed while editing; your edits are preserved at {recovery}')
            os.replace(tmp,path)
            self.store.event('file_saved',str(path))
            self.message=f'Saved {path.name}'
        finally:
            Path(tmp).unlink(missing_ok=True)

    def edit_buffer(self,path,text):
        lines=text.split('\n');row=col=top=left=0
        self.s.timeout(-1);curses.raw();curses.curs_set(1)
        try:
            while True:
                h,w=self.s.getmaxyx();height=max(1,h-5);width=max(1,w-11)
                if row<top:top=row
                if row>=top+height:top=row-height+1
                visual_col=len(lines[row][:col].expandtabs(4))
                if visual_col<left:left=visual_col
                if visual_col>=left+width:left=visual_col-width+1
                self.s.erase();self.frame('KINDI // EDIT · '+path.name,'Ctrl-S Save   Esc Cancel   Arrows Navigate   PgUp/PgDn')
                for i,line in enumerate(lines[top:top+height]):
                    y=i+2;self.put(y,2,f'{top+i+1:>4} ',self.colors['muted'])
                    trimmed=line.lstrip()
                    color=self.colors['muted'] if trimmed.startswith(('#','//')) else self.colors['lime'] if trimmed.startswith(('def ','class ','import ','from ','function ','const ','fn ','pub ')) else self.colors['text']
                    self.put(y,8,line.expandtabs(4)[left:left+width],color)
                modified='modified' if '\n'.join(lines)!=text else 'saved'
                self.put(h-2,2,f'{modified}  ·  line {row+1}, column {col+1}',self.colors['lime'])
                self.s.move(min(h-3,2+row-top),min(w-2,8+visual_col-left));self.s.refresh()
                key=self.s.get_wch()
                if key=='\x13':return '\n'.join(lines)
                if key=='\x1b':
                    if '\n'.join(lines)==text or self.ask('Discard changes? y to discard')=='y':return None
                    self.s.timeout(-1);curses.curs_set(1);continue
                if key in (curses.KEY_UP,curses.KEY_DOWN,curses.KEY_PPAGE,curses.KEY_NPAGE):
                    delta={curses.KEY_UP:-1,curses.KEY_DOWN:1,curses.KEY_PPAGE:-height,curses.KEY_NPAGE:height}[key]
                    row=max(0,min(len(lines)-1,row+delta));col=min(col,len(lines[row]))
                elif key==curses.KEY_LEFT:
                    if col:col-=1
                    elif row:row-=1;col=len(lines[row])
                elif key==curses.KEY_RIGHT:
                    if col<len(lines[row]):col+=1
                    elif row<len(lines)-1:row+=1;col=0
                elif key==curses.KEY_HOME:col=0
                elif key==curses.KEY_END:col=len(lines[row])
                elif key in ('\x7f','\b',curses.KEY_BACKSPACE):
                    if col:lines[row]=lines[row][:col-1]+lines[row][col:];col-=1
                    elif row:
                        col=len(lines[row-1]);lines[row-1]+=lines.pop(row);row-=1
                elif key==curses.KEY_DC:
                    if col<len(lines[row]):lines[row]=lines[row][:col]+lines[row][col+1:]
                    elif row<len(lines)-1:lines[row]+=lines.pop(row+1)
                elif key in ('\n','\r'):
                    suffix=lines[row][col:];lines[row]=lines[row][:col];lines.insert(row+1,suffix);row+=1;col=0
                elif isinstance(key,str) and (key.isprintable() or key=='\t'):
                    insert='    ' if key=='\t' else key
                    lines[row]=lines[row][:col]+insert+lines[row][col:];col+=len(insert)
        finally:
            curses.noraw();curses.cbreak();curses.curs_set(0);self.s.timeout(50)

    def add_prompt(self, check=False, immediate=False):
        from .dialogs import Dialogs
        from .menus import mission,schedule
        dialog=Dialogs(self.s)
        try:
            if immediate:mission(dialog,self.store,self.target()['id'])
            else:schedule(dialog,self.store,check)
        finally:dialog.close()

    def action(self,key):
        name=SCREENS[self.view]
        row=self.row()
        if key == ord('?'):
            from .dialogs import Dialogs
            from .menus import execute
            dialog=Dialogs(self.s)
            try:execute('help',dialog,self.store)
            finally:dialog.close()
        elif name=='Kindi' and key in (ord('u'),ord('c'),ord('i'),10,13,curses.KEY_ENTER):
            from . import kindi
            if key==ord('c'):
                url=self.ask('Kindi server address (host:port)','127.0.0.1:8093')
                if url:
                    result=kindi.use(self.store,url)
                    self.message=f'Agents now use {result["model"]} at {result["base"]}'+(' — '+result['note'] if result.get('note') else '')
                    self.fresh.clear()
            elif key==ord('i'):
                self.details('Kindi status & features',kindi.report(self.store))
            elif row and row['kind']=='server' and key in (ord('u'),10,13,curses.KEY_ENTER):
                info=row['info']
                if key!=ord('u'):
                    self.details(row['label_text'],kindi.summary(info)+'\n\n'+json.dumps(info.get('health') or {'error':info.get('error')},indent=2)+
                                 '\n\nPress u on this row to use this server for new agents.')
                elif self.ask(f'Use {info["base"]} ({info.get("model") or "?"}) for new agents? y to confirm')=='y':
                    result=kindi.use(self.store,info['base'],row['label_text'])
                    self.message=f'Agents now use {result["model"]} at {result["base"]}'+(' — '+result['note'] if result.get('note') else '')
                    self.fresh.clear()
            elif row and row['kind']=='feature':
                f=row['feature'];self.details(f[1],f'{f[0]}\n\n{f[4]}\n\nStatus: {"in Kindi "+kindi.KINDI_VERSION if f[2]=="release" else "preview, in development for a later release"}')
        elif name=='Agents' and key in (ord('a'),ord('n'),ord('m')):
            from .dialogs import Dialogs
            from .menus import new_agent,mission
            dialog=Dialogs(self.s)
            try:
                if key==ord('m'):mission(dialog,self.store,self.target()['id'])
                else:new_agent(dialog,self.store)
            finally:dialog.close()
        elif name=='Agents' and key==ord('i') and row:
            self.details(row['name'],f'Name: {row["name"]}\nStatus: {row["status"]}\n\nMission\n{row["mission"]}\n\nWorking folder\n{row["cwd"]}\n\nEnter on the agent list opens the conversation. Use m to send a mission.')
        elif key==ord('n'):
            self.add_prompt(immediate=True)
        elif key==ord('a'):
            if name in {'Queue','Deck'}:
                self.add_prompt()
            elif name=='Knowledge':
                root=self.ask('Knowledge folder (on this host)',str(self.root))
                if root:
                    from .knowledge import add_collection
                    ident=add_collection(self.store,root)
                    self.store.execute('UPDATE agents SET collection=? WHERE id=?',(ident,self.store.agent()['id']))
                    self.message='Indexing in background; attached to default agent'
        elif key==ord('c') and name=='Queue':
            self.add_prompt(check=True)
        elif key==ord('d') and name=='Queue' and row:
            if self.store.delete_job(row['id']):
                self.undo=(row['id'],time.time())
                self.message='Deleted; u to undo within 10 seconds'
            else:
                self.message='Already accepted/claimed; cannot delete a running turn'
        elif key==ord('u') and self.undo and time.time()-self.undo[1]<10:
            self.store.undo_delete(self.undo[0]); self.undo=None
        elif key==ord(' ') and name=='Queue' and row:
            state='paused' if row['state']=='scheduled' else 'scheduled'
            self.store.execute("UPDATE jobs SET state=? WHERE id=? AND state IN ('scheduled','paused')",(state,row['id']))
        elif key==ord('D') and name=='Agents' and row:
            self.store.setting('default_agent',row['id']); self.message=f'Default: {row["name"]}'
        elif key==ord('l') and name=='Agents' and row:
            launch_agent(self.store,row['id']); self.message='Worker launched'
        elif key in (ord('k'),ord('K')) and name in {'Processes','Ports'} and row:
            from .system import terminate
            force=key==ord('K')
            title=f'{"FORCE KILL" if force else "Terminate"} PID {row["pid"]} ({row["name"]}); all its sockets close. y to confirm'
            if self.ask(title)=='y':
                terminate(row['pid'],row['born'],force)
                self.store.event('process_killed' if force else 'process_terminated',json.dumps(row))
                self.message=f'Signal sent to PID {row["pid"]}; supervised services may restart'
                self.fresh[name]=0
        elif key==ord('v'):
            if self.embedded:
                from .workbench import show_agent
                show_agent(self.target()['id'])
            else:
                tmux('has-session','-t','agents')
                self.external(['tmux','-L',os.environ.get('BRIDGE_TMUX_SOCKET','kindi-bridge'),'attach-session','-t','agents'])
        elif key==ord('e') and name=='Files' and row:
            self.edit(row['path'])
        elif key==ord('g') and name=='Files':
            from .system import run
            self.details('Workspace git diff',run(['git','-C',str(self.root),'diff','--no-ext-diff','--stat'])+'\n'+run(['git','-C',str(self.root),'diff','--no-ext-diff']))
        elif key==ord('r'):
            self.fresh.clear()
            if name=='Knowledge' and row:
                self.store.execute("UPDATE collections SET state='pending' WHERE id=?",(row['id'],))
        elif key==ord('b') and name=='Knowledge' and row:
            self.store.execute('UPDATE agents SET collection=? WHERE id=?',(row['id'],self.store.agent()['id']))
            self.message='Bound collection to default agent'
        elif key==ord('s'):
            if name=='Knowledge' and row:
                query=self.ask('Search query')
                if query:
                    from .knowledge import search
                    # Search can load the model: run it outside the drawing loop and show progress.
                    self.message='Searching knowledge...'; self.table()
                    future=self.pool.submit(search,self.store,query,row['id'])
                    while not future.done():
                        if self.s.getch()==27:
                            return
                    hits=future.result()
                    self.details('Retrieved passages','\n\n'.join(x['source']+'\n'+x['text'] for x in hits) or 'No passages found')
            else:
                self.details('External schedules (read-only)',schedule_inventory())
        elif key in (10,13,curses.KEY_ENTER) and row:
            if name=='Files':
                path=Path(row['path'])
                if path.is_dir():
                    self.folder=path; self.position[name]=0; self.selected[name]=None; self.offset=0
                else:
                    if path.stat().st_size>2*1024*1024:
                        raise ValueError('Preview limit 2 MiB; use an external viewer')
                    self.details(path.name,path.read_text(encoding='utf-8'))
            elif name in {'Deck','Agents'}:
                from .workbench import show_agent
                show_agent(row['id'])
            elif name=='Knowledge':
                issues=self.store.rows('SELECT path,error FROM documents WHERE collection=? AND error IS NOT NULL',(row['id'],))
                self.details('Collection details',json.dumps(row,indent=2)+'\n'+json.dumps(issues,indent=2))
            elif name=='Queue':
                when=dt.datetime.fromtimestamp(row['due']).astimezone().isoformat(timespec='seconds')
                self.details('PROMPT',f'TARGET   {row["name"]}\nWHEN     {when}\nSTATE    {row["state"]}\nCHECKS   {row["attempts"]} / {row["max_checks"]}\n\nPROMPT\n{row["prompt"]}\n\nOBJECTIVE\n{row["task"] or "—"}\n\nLAST RESPONSE\n{row["response"] or "—"}\n\n{row["error"] or ""}')
            elif name in {'Processes','Ports'}:
                fields=[f'{k.upper():<12} {v}' for k,v in row.items() if k not in {'id','label','born'}]
                self.details(name,'\n\n'.join(fields))
            else:self.details(name+' details',json.dumps(row,indent=2,default=str))

    def loop(self):
        try:
            while True:
                if time.time()-self.last_refresh>0.2:
                    self.table(); self.last_refresh=time.time()
                key=self.s.getch()
                if key==ord('q'):
                    if self.embedded:
                        from .workbench import screen
                        screen('Deck');continue
                    else:return
                name=SCREENS[self.view]
                if key in (9,curses.KEY_RIGHT,curses.KEY_BTAB,curses.KEY_LEFT) or ord('0')<=key<=ord('9'):
                    view=(int(chr(key))-1)%len(SCREENS) if ord('0')<=key<=ord('9') else (self.view+(-1 if key in (curses.KEY_LEFT,curses.KEY_BTAB) else 1))%len(SCREENS)
                    if self.embedded:
                        from .workbench import screen
                        screen(SCREENS[view])
                    else:self.view=view
                    self.offset=0; self.filter=''
                elif key==curses.KEY_MOUSE:
                    with __import__('contextlib').suppress(curses.error):
                        _,x,y,_,buttons=curses.getmouse()
                        if buttons & curses.BUTTON4_PRESSED:
                            self.position[name]=max(0,self.position[name]-3);self.selected[name]=None
                        elif buttons & getattr(curses,'BUTTON5_PRESSED',0):
                            self.position[name]=min(max(0,len(self.rows)-1),self.position[name]+3);self.selected[name]=None
                        elif buttons & (curses.BUTTON1_CLICKED|curses.BUTTON1_PRESSED):
                            index=self.offset+y-3
                            if 3<=y<self.s.getmaxyx()[0]-4 and 0<=index<len(self.rows):
                                self.position[name]=index;self.selected[name]=None
                elif key in (curses.KEY_UP,curses.KEY_DOWN,curses.KEY_NPAGE,curses.KEY_PPAGE,curses.KEY_HOME,curses.KEY_END):
                    delta={curses.KEY_UP:-1,curses.KEY_DOWN:1,curses.KEY_NPAGE:max(1,self.s.getmaxyx()[0]-7),curses.KEY_PPAGE:-max(1,self.s.getmaxyx()[0]-7)}.get(key,0)
                    p=self.position[name]+delta
                    if key==curses.KEY_HOME:p=0
                    if key==curses.KEY_END:p=len(self.rows)-1
                    self.position[name]=max(0,min(len(self.rows)-1,p)); self.selected[name]=None
                elif key==ord('/'):
                    self.filter=self.ask('Filter',self.filter) or ''; self.offset=0
                elif key==27:
                    if self.embedded and not self.filter:
                        from .workbench import screen
                        screen('Deck')
                    else:self.filter=''
                elif key>=0 and key!=curses.KEY_RESIZE:
                    try:self.action(key)
                    except Exception as e:self.message=str(e)
                if key>=0:self.last_refresh=0
        finally:
            self.pool.shutdown(wait=False,cancel_futures=True)


def main(root=None,page='Deck'):
    ensure_service()
    curses.wrapper(lambda s:UI(s,root,page).loop())
