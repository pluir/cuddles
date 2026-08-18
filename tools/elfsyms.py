#!/usr/bin/env python
"""Extract a sorted symbol table from a 32-bit ELF kernel.

Usage: python tools/elfsyms.py <kernel.elf> > syms.txt
Output: one "ADDR SIZE NAME" line per function/object symbol, hex addresses.
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
        name, typ, flags, addr, off, size, link, info, align, entsize = \
            struct.unpack('<10I', d[o:o+40])
        out.append(dict(name=name, type=typ, addr=addr, off=off, size=size,
                        link=link, entsize=entsize))
    # resolve names
    strtab = out[e_shstrndx]
    base = strtab['off']
    for s in out:
        end = d.index(b'\0', base + s['name'])
        s['sname'] = d[base + s['name']:end].decode('utf-8', 'replace')
    return out

def main():
    d = open(sys.argv[1], 'rb').read()
    secs = sections(d)
    syms = []
    for s in secs:
        if s['type'] not in (2, 11):  # SYMTAB, DYNSYM
            continue
        strs = secs[s['link']]
        n = s['size'] // 16
        for i in range(n):
            o = s['off'] + i * 16
            st_name, st_value, st_size, st_info, st_other, st_shndx = \
                struct.unpack('<IIIBBH', d[o:o+16])
            if st_name == 0:
                continue
            base = strs['off'] + st_name
            end = d.index(b'\0', base)
            name = d[base:end].decode('utf-8', 'replace')
            typ = st_info & 0xF
            if typ not in (1, 2):  # OBJECT, FUNC
                continue
            syms.append((st_value, st_size, name))
    syms.sort()
    for addr, size, name in syms:
        print('%08X %08X %s' % (addr, size, name))

main()
