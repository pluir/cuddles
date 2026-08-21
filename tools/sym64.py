#!/usr/bin/env python
"""Resolve hex addresses to 64-bit kernel symbols.

Usage: python tools/sym64.py <addr> ...     (reads images/alpine.syms)
"""
import sys, bisect, os
path = os.environ.get('X86EMU_SYMS', 'images/alpine.syms')
S = []
for line in open(path):
    a, t, n = line.split(None, 2)
    S.append((int(a, 16), n.strip()))
S.sort()
addrs = [a for a, _ in S]
def look(v):
    i = bisect.bisect_right(addrs, v) - 1
    if i < 0:
        return '?'
    a, n = S[i]
    return '%s+0x%X' % (n, v - a) if v != a else n
for arg in sys.argv[1:]:
    v = int(arg.replace('0x', ''), 16)
    print('%016X  %s' % (v, look(v)))
