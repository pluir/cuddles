#!/usr/bin/env python
"""Pull files out of the ISO9660 image the boot effort uses.

    python tools/extract_iso.py images/linux.iso            # list the root
    python tools/extract_iso.py images/linux.iso ROOT.BIN images/root.bin

The ISO carries the kernel (BZIMAGE) and an ext2 root filesystem (ROOT.BIN)
that isolinux hands to the kernel as an initrd. `images/root.bin` is what
`--initrd` wants, so extracting it is the one step between a downloaded image
and a boot that reaches userspace.

Only what is needed to reach those two files is implemented: the primary
volume descriptor and one level of directory records. No Rock Ridge, no
Joliet, no multi-extent files.
"""
import struct
import sys

SECTOR = 2048


def listdir(data, lba, size):
    """Yield (name, extent_lba, length, is_dir) for one directory extent."""
    off = lba * SECTOR
    end = off + size
    i = off
    while i < end:
        rec_len = data[i]
        if rec_len == 0:
            # Records do not straddle sector boundaries; skip to the next one.
            i = (i // SECTOR + 1) * SECTOR
            if i >= end:
                break
            continue
        extent = struct.unpack('<I', data[i + 2:i + 6])[0]
        length = struct.unpack('<I', data[i + 10:i + 14])[0]
        flags = data[i + 25]
        name_len = data[i + 32]
        name = data[i + 33:i + 33 + name_len].decode('latin-1')
        yield name, extent, length, bool(flags & 2)
        i += rec_len


def root_of(data):
    """The root directory record's (lba, size)."""
    pvd = data[0x8000:0x8000 + SECTOR]
    if pvd[1:6] != b'CD001':
        sys.exit('not an ISO9660 image (no CD001 signature)')
    rec = pvd[156:156 + 34]
    return struct.unpack('<I', rec[2:6])[0], struct.unpack('<I', rec[10:14])[0]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data = open(sys.argv[1], 'rb').read()
    lba, size = root_of(data)
    entries = list(listdir(data, lba, size))

    if len(sys.argv) == 2:
        for name, _, length, is_dir in entries:
            # The first two records are "." and ".." with one-byte names.
            if len(name) == 1 and name in ('\0', '\1'):
                continue
            print('%-24s %10d  %s' % (name, length, 'dir' if is_dir else 'file'))
        return

    if len(sys.argv) != 4:
        sys.exit('usage: extract_iso.py <iso> <NAME> <outfile>')
    want = sys.argv[2].upper()
    for name, extent, length, is_dir in entries:
        # ISO9660 names carry a ";1" version suffix, and a trailing dot when
        # the name has no extension ("BZIMAGE.;1").
        bare = name.split(';')[0].upper().rstrip('.')
        if is_dir or (bare != want and name.upper() != want):
            continue
        blob = data[extent * SECTOR:extent * SECTOR + length]
        open(sys.argv[3], 'wb').write(blob)
        print('wrote %s (%d bytes) from %s' % (sys.argv[3], len(blob), name))
        return
    sys.exit('no such file in the image: %s' % sys.argv[2])


main()
