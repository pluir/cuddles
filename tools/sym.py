#!/usr/bin/env python
"""Resolve hex addresses to kernel symbols: python tools/sym.py <addr> ..."""
import sys, bisect
S = []
for line in open('images/kernel.syms'):
    a, t, n = line.split(None, 2)
    S.append((int(a, 16), n.strip()))
S.sort()
addrs = [a for a, _ in S]
def look(v):
    i = bisect.bisect_right(addrs, v) - 1
    if i < 0: return '?'
    a, n = S[i]
    return '%s+0x%X' % (n, v - a) if v != a else n
for arg in sys.argv[1:]:
    v = int(arg, 16)
    print('%08X  %s' % (v, look(v)))
