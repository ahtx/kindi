# Copyright (c) 2026 Amir Husain. SPDX-License-Identifier: AGPL-3.0-only. See LICENSE and COMMERCIAL-LICENSE.md.
"""One visual vocabulary, taken from the user's Kindi teaser palette."""
from __future__ import annotations

import re

INK = '#080808'
LIME = '#caf31c'
EDGE = '#232323'
TEXT = '#f3f3f1'
MUTED = '#9a9a9a'
GREY = '#6b6b6b'
CYAN = '#38bdf8'
RED = '#ff6b5a'
RESET = '\033[0m'
BOLD = '\033[1m'
ANSI = re.compile(r'\033\[[0-9;?]*[A-Za-z]')


def fg(color):
    r,g,b = (int(color[i:i+2],16) for i in (1,3,5))
    return f'\033[38;2;{r};{g};{b}m'


def bg(color):
    return fg(color).replace('[38;', '[48;')


def clean(text):
    return ''.join(c if c.isprintable() else ' ' for c in str(text))


def fit(text, width):
    out=[]; n=0; i=0
    while i<len(text) and n<max(0,width):
        m=ANSI.match(text,i)
        if m:out.append(m[0]);i=m.end();continue
        out.append(text[i]);i+=1;n+=1
    return ''.join(out)+RESET+bg(INK)+' '*max(0,width-n)


def bar(fraction,width=16,color=LIME):
    if fraction is None:return fg(EDGE)+'░'*width+RESET
    n=max(0,min(width,round(width*fraction)))
    return fg(color)+'█'*n+fg(EDGE)+'░'*(width-n)+RESET


def spark(values,width=24,ceiling=100,color=LIME):
    chars=' ▁▂▃▄▅▆▇█'
    s=''.join(chars[max(0,min(8,round((v or 0)/max(ceiling,1)*8)))] for v in list(values)[-width:])
    return fg(color)+s.rjust(width)+RESET


def graph(values,width,height=3,ceiling=100):
    vals=([0]*width+list(values))[-width:]
    chars=' ▁▂▃▄▅▆▇█'
    return [fg(LIME)+''.join(chars[max(0,min(8,round(v/max(ceiling,1)*height*8)-(height-row-1)*8))] for v in vals)+RESET for row in range(height)]


def pair(left,right,width):
    gap=width-len(ANSI.sub('',left))-len(ANSI.sub('',right))
    return left+' '*max(1,gap)+right if gap>=1 else fit(left,width)


def panel(title,rows,width,height,subtitle=''):
    if width<6 or height<3:return [fit(row,width) for row in rows[:height]]
    title=' '+title+' '
    top=fg(EDGE)+'╭─'+fg(LIME)+BOLD+title+RESET+bg(INK)+fg(EDGE)+'─'*max(0,width-len(title)-3)+'╮'
    out=[fit(top,width)]
    for row in (rows+['']*height)[:height-2]:
        out.append(fg(EDGE)+'│ '+fit(row,width-4)+fg(EDGE)+' │'+RESET)
    foot=' '+subtitle+' ' if subtitle else ''
    out.append(fg(EDGE)+'╰'+fg(GREY)+foot+fg(EDGE)+'─'*max(0,width-len(foot)-2)+'╯'+RESET)
    return out


def curses_palette():
    import curses
    attrs={'text':0,'lime':curses.A_BOLD,'muted':curses.A_DIM,'edge':curses.A_DIM,'selected':curses.A_REVERSE,'danger':curses.A_BOLD,'cyan':curses.A_BOLD}
    if not curses.has_colors():return attrs
    curses.start_color()
    specs=[('text',TEXT,255),('lime',LIME,190),('muted',MUTED,247),('edge',EDGE,236),('danger',RED,203),('cyan',CYAN,81)]
    def color(hexcode,fallback,index):
        if curses.COLORS>=256 and curses.can_change_color() and index<curses.COLORS:
            rgb=[int(hexcode[i:i+2],16) for i in (1,3,5)]
            curses.init_color(index,*(round(v*1000/255) for v in rgb));return index
        return fallback if curses.COLORS>=256 else (curses.COLOR_GREEN if hexcode==LIME else curses.COLOR_WHITE)
    ink=color(INK,232,240)
    for i,(name,hexcode,fallback) in enumerate(specs,1):
        curses.init_pair(i,color(hexcode,fallback,240+i),ink)
        attrs[name]=curses.color_pair(i)
    curses.init_pair(7,ink,color(LIME,190,242))
    attrs['selected']=curses.color_pair(7)|curses.A_BOLD
    return attrs
