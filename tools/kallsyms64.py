#!/usr/bin/env python
"""Extract kallsyms from a stripped 64-bit x86 Linux kernel ELF.

A distro kernel has no .symtab, but it carries the kallsyms tables that
/proc/kallsyms is built from.  Rather than assume the order the pieces are
emitted in -- which moves between kernel versions; on Alpine's 6.18 the
offsets array follows the token index rather than preceding the names --
each piece is found by its own signature and cross-checked against the
others:

  kallsyms_token_index   256 u16, first 0, strictly ascending.  Unique.
  kallsyms_token_table   256 NUL-terminated tokens, located by requiring
                         that every token_index entry lands just past a NUL.
  kallsyms_num_syms      a u64 whose following bytes decode, through the
                         token table, into `num` well-formed "Tname" entries.
                         Since 6.1 a length byte with bit 7 set introduces a
                         two-byte "big symbol" length.
  kallsyms_offsets       an ascending u32 run of exactly `num` entries, each
                         an offset from the start of .text.

Usage: python tools/kallsyms64.py <kernel.elf> > kernel.syms
"""
import struct, sys

TYPES = set(b'AaBbDdGgNnPpRrSsTtUuVvWw?')
NAME = set(b'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$')


def sections(d):
    e_shoff, = struct.unpack_from('<Q', d, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from('<HHH', d, 0x3A)
    out = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        name, typ, flags, addr, off, size = struct.unpack_from('<IIQQQQ', d, o)
        out.append(dict(name=name, addr=addr, off=off, size=size))
    base = out[e_shstrndx]['off']
    for s in out:
        end = d.index(b'\0', base + s['name'])
        s['sname'] = d[base + s['name']:end].decode('utf-8', 'replace')
    return out


def find_token_index(d, lo, hi):
    for p in range(lo, hi - 512, 2):
        v0, v1 = struct.unpack_from('<HH', d, p)
        if v0 != 0 or not (2 <= v1 <= 33):
            continue
        v = struct.unpack_from('<256H', d, p)
        prev = -1
        for x in v:
            if x <= prev or x > 8192:
                break
            prev = x
        else:
            return p, v
    sys.exit('kallsyms_token_index not found')


def find_token_table(d, p, v):
    for t in range(p - 4096, p):
        if all(d[t + v[i] - 1] == 0 for i in range(1, 256)):
            toks, o = [], t
            for _ in range(256):
                e = d.index(b'\0', o)
                if e - o > 32:
                    break
                toks.append(d[o:e])
                o = e + 1
            if len(toks) == 256 and o <= p:
                return toks
    sys.exit('kallsyms_token_table not found before %x' % p)


def decode(d, p, n, toks):
    out = []
    for _ in range(n):
        b = d[p]
        if b & 0x80:                       # two-byte "big symbol" length
            ln = (b & 0x7F) | (d[p + 1] << 7)
            p += 2
        else:
            ln = b
            p += 1
        out.append(b''.join(toks[c] for c in d[p:p + ln]))
        p += ln
    return out, p


def well_formed(s):
    return 2 < len(s) < 128 and s[0] in TYPES and all(c in NAME for c in s[1:])


def find_names(d, lo, hi, toks):
    for q in range(lo, hi - 16, 8):
        n, = struct.unpack_from('<Q', d, q)
        if not (10000 < n < 400000):
            continue
        try:
            probe, _ = decode(d, q + 8, 24, toks)
        except Exception:
            continue
        if all(well_formed(s) for s in probe):
            return q + 8, n
    sys.exit('kallsyms_num_syms not found')


def find_offsets(d, lo, hi, num):
    i = lo
    while i < hi - 4:
        v, = struct.unpack_from('<I', d, i)
        if v > 0x4000000:
            i += 4
            continue
        j, prev, n = i + 4, v, 1
        while j < hi - 4:
            w, = struct.unpack_from('<I', d, j)
            if w < prev or w > 0x4000000:
                break
            prev, n, j = w, n + 1, j + 4
        if n == num:
            return i
        i = max(j, i + 4)
    sys.exit('kallsyms_offsets run of %d not found' % num)


def main():
    d = open(sys.argv[1], 'rb').read()
    secs = sections(d)
    ro = [s for s in secs if s['sname'] == '.rodata'][0]
    text = [s for s in secs if s['sname'] == '.text'][0]
    lo, hi = ro['off'], ro['off'] + ro['size']

    p, v = find_token_index(d, lo, hi)
    toks = find_token_table(d, p, v)
    names_off, num = find_names(d, lo, hi, toks)
    names, _ = decode(d, names_off, num, toks)
    off_arr = find_offsets(d, lo, hi, num)

    base = text['addr']
    for k, s in enumerate(names):
        if not s:
            continue
        o, = struct.unpack_from('<I', d, off_arr + 4 * k)
        print('%016X %s %s' % (base + o, chr(s[0]), s[1:].decode('utf-8', 'replace')))


main()
