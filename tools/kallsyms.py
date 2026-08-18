#!/usr/bin/env python
"""Extract kallsyms from a stripped 32-bit x86 Linux kernel ELF.

Scans .rodata for kallsyms_addresses (a long ascending run of u32 values in
the kernel text range), then decodes the compressed kallsyms_names table.

Usage: python tools/kallsyms.py <kernel.elf> > kernel.syms
"""
import struct, sys

def sections(d):
    e_shoff, = struct.unpack('<I', d[0x20:0x24])
    e_shentsize, = struct.unpack('<H', d[0x2E:0x30])
    e_shnum, = struct.unpack('<H', d[0x30:0x32])
    e_shstrndx, = struct.unpack('<H', d[0x32:0x34])
    out = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        vals = struct.unpack('<10I', d[o:o+40])
        out.append(dict(zip(
            'name type flags addr off size link info align entsize'.split(), vals)))
    base = out[e_shstrndx]['off']
    for s in out:
        end = d.index(b'\0', base + s['name'])
        s['sname'] = d[base + s['name']:end].decode('utf-8', 'replace')
    return out

def main():
    d = open(sys.argv[1], 'rb').read()
    secs = sections(d)
    ro = [s for s in secs if s['sname'] == '.rodata'][0]
    lo, hi = 0xC0000000, 0xC1000000

    off, size = ro['off'], ro['size']
    best = (0, 0)
    i = off
    end = off + size - 4
    while i < end:
        v = struct.unpack('<I', d[i:i+4])[0]
        if not (lo <= v < hi):
            i += 4
            continue
        j, prev, n = i + 4, v, 1
        while j < end:
            w = struct.unpack('<I', d[j:j+4])[0]
            if not (lo <= w < hi) or w < prev:
                break
            prev = w
            n += 1
            j += 4
        if n > best[1]:
            best = (i, n)
        i = max(j, i + 4)

    addr_off, count = best
    if count < 1000:
        sys.exit('no plausible kallsyms_addresses found (best run %d)' % count)

    p = addr_off + count * 4
    p = (p + 3) & ~3
    num = struct.unpack('<I', d[p:p+4])[0]
    if not (0 < num <= count):
        for q in range(p, p + 64, 4):
            n2 = struct.unpack('<I', d[q:q+4])[0]
            if 0 < n2 <= count and count - n2 < 32:
                p, num = q, n2
                break
        else:
            sys.exit('kallsyms_num_syms not found near %x (saw %x)' % (p, num))
    addrs = [struct.unpack('<I', d[addr_off + 4*k:addr_off + 4*k + 4])[0]
             for k in range(num)]

    names_off = (p + 4 + 3) & ~3
    q = names_off
    raw = []
    for _ in range(num):
        ln = d[q]
        raw.append(d[q+1:q+1+ln])
        q += 1 + ln
    names_end = q

    m = (names_end + 3) & ~3
    nmark = (num >> 8) + 1
    tok_off = (m + nmark * 4 + 3) & ~3

    def try_tokens(t):
        toks, o = [], t
        try:
            for _ in range(256):
                e = d.index(b'\0', o)
                if e - o > 16:
                    return None
                toks.append(d[o:e])
                o = e + 1
        except ValueError:
            return None
        return toks, o

    tokens = None
    for cand in (tok_off, (tok_off + num * 3 + 3) & ~3, (tok_off + num * 4 + 3) & ~3):
        r = try_tokens(cand)
        if r:
            tokens, _ = r
            break
    if tokens is None:
        sys.exit('kallsyms_token_table not found near %x' % tok_off)

    for a, r in zip(addrs, raw):
        if not r:
            continue
        s = b''.join(tokens[c] for c in r)
        typ = chr(s[0])
        name = s[1:].decode('utf-8', 'replace')
        print('%08X %s %s' % (a, typ, name))

main()
