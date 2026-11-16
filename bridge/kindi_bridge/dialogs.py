# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""Small DOS-style forms and selectors, using the same renderer as the main tables."""
from __future__ import annotations

import curses
import textwrap


class Dialogs:
    def __init__(self,screen):
        from .tui import UI
        self.ui=UI(screen);self.s=screen

    def close(self):
        self.ui.pool.shutdown(wait=False,cancel_futures=True)

    def select(self,title,items,initial=0,empty='No agents yet. Use Agents > New agent to create one.'):
        if not items:
            self.ui.details(title,empty);return None
        selected=min(initial,len(items)-1);top=0
        self.s.timeout(-1)
        try:
            while True:
                h,w=self.s.getmaxyx();height=max(1,h-5)
                top=min(top,selected)
                if selected>=top+height:top=selected-height+1
                self.s.erase();self.ui.frame(title,'Choose with arrows or mouse. Enter opens; Esc cancels.')
                for i,item in enumerate(items[top:top+height]):
                    attr=self.ui.colors['selected'] if top+i==selected else self.ui.colors['text']
                    self.ui.put(i+2,2,item[1][:max(1,w-4)].ljust(max(1,w-4)),attr)
                self.ui.put(h-2,2,'Enter Select    Esc Cancel',self.ui.colors['lime']);self.s.refresh()
                key=self.s.getch()
                if key==27:return None
                if key in (10,13,curses.KEY_ENTER):return items[selected][0]
                if key in (curses.KEY_UP,curses.KEY_DOWN):selected=(selected+(-1 if key==curses.KEY_UP else 1))%len(items)
                elif key==curses.KEY_HOME:selected=0
                elif key==curses.KEY_END:selected=len(items)-1
                elif key==curses.KEY_MOUSE:
                    try:
                        _,x,y,_,buttons=curses.getmouse();index=top+y-2
                        if 2<=y<h-3 and 0<=index<len(items) and buttons&(curses.BUTTON1_CLICKED|curses.BUTTON1_PRESSED):return items[index][0]
                    except curses.error:pass
        finally:self.s.timeout(50)

    def agent(self,store,title='Choose an agent'):
        from .agents import roster
        rows=roster(store)
        return self.select(title,[(a['id'],f'{">" if a["current"] else " "} {a["name"]:<18} {a["status"]:<17} {a["mission"]}') for a in rows],next((i for i,a in enumerate(rows) if a['current']),0))

    def form(self,title,fields,button='Save'):
        """fields: [(key, visible label, default)], with a visible primary action."""
        values=[str(default) for _,_,default in fields];cursors=[len(v) for v in values];focus=0
        self.s.timeout(-1);curses.raw();curses.curs_set(1)
        try:
            while True:
                h,w=self.s.getmaxyx();self.s.erase()
                self.ui.frame(title,'Tab moves between fields. Enter continues. Ctrl-S '+button.lower()+'.')
                positions=[];y=3
                for index,(_,label,_) in enumerate(fields):
                    self.ui.put(y,3,label,self.ui.colors['lime']);y+=1
                    attr=self.ui.colors['selected'] if focus==index else self.ui.colors['text']
                    width=max(1,w-8)
                    start=max(0,cursors[index]-width+1)
                    self.ui.put(y,3,values[index][start:start+width].ljust(width),attr);positions.append(y)
                    y+=2
                buttons_y=min(h-4,y+1)
                self.ui.put(buttons_y,3,'[ '+button+' ]',self.ui.colors['selected'] if focus==len(fields) else self.ui.colors['lime'])
                self.ui.put(buttons_y,len(button)+12,'[ Cancel ]',self.ui.colors['selected'] if focus==len(fields)+1 else self.ui.colors['muted'])
                self.ui.put(h-2,2,'Tab Next  Enter Choose  Ctrl-S Submit  Ctrl-E Expand text  Esc Cancel',self.ui.colors['muted'])
                if focus<len(fields):
                    start=max(0,cursors[focus]-max(1,w-8)+1)
                    curses.curs_set(1);self.s.move(min(h-3,positions[focus]),min(w-3,3+cursors[focus]-start))
                else:curses.curs_set(0)
                self.s.refresh();key=self.s.get_wch()
                if key=='\x1b':return None
                if key==curses.KEY_MOUSE:
                    try:
                        _,x,y,_,buttons=curses.getmouse()
                        if buttons&(curses.BUTTON1_CLICKED|curses.BUTTON1_PRESSED):
                            if y in positions:focus=positions.index(y)
                            elif y==buttons_y:
                                if 3<=x<8+len(button):return {fields[i][0]:value.strip() for i,value in enumerate(values)}
                                elif len(button)+12<=x<len(button)+23:return None
                    except curses.error:pass
                    continue
                if key=='\x13' or (key in ('\n','\r') and focus==len(fields)):
                    return {fields[i][0]:value.strip() for i,value in enumerate(values)}
                if key in ('\n','\r') and focus==len(fields)+1:return None
                if key in ('\t','\n','\r',curses.KEY_DOWN):focus=(focus+1)%(len(fields)+2)
                elif key in (curses.KEY_BTAB,curses.KEY_UP):focus=(focus-1)%(len(fields)+2)
                elif focus<len(fields):
                    col=cursors[focus]
                    if key in ('\b','\x7f',curses.KEY_BACKSPACE) and col:
                        values[focus]=values[focus][:col-1]+values[focus][col:];cursors[focus]-=1
                    elif key==curses.KEY_LEFT:cursors[focus]=max(0,col-1)
                    elif key==curses.KEY_RIGHT:cursors[focus]=min(len(values[focus]),col+1)
                    elif key==curses.KEY_HOME:cursors[focus]=0
                    elif key==curses.KEY_END:cursors[focus]=len(values[focus])
                    elif key==curses.KEY_DC:values[focus]=values[focus][:col]+values[focus][col+1:]
                    elif key=='\x15':values[focus]='';cursors[focus]=0
                    elif key=='\x05':
                        from pathlib import Path
                        text=self.ui.edit_buffer(Path(fields[focus][1]+'.txt'),values[focus])
                        if text is not None:values[focus]=text;cursors[focus]=len(text)
                        self.s.timeout(-1);curses.raw();curses.curs_set(1)
                    elif isinstance(key,str) and key.isprintable():
                        values[focus]=values[focus][:col]+key+values[focus][col:];cursors[focus]+=len(key)
        finally:
            curses.noraw();curses.cbreak();curses.curs_set(0);self.s.timeout(50)
