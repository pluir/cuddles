#!/usr/bin/env python
"""Pull files out of the ISO9660 image the boot effort uses.

    python tools/extract_iso.py images/linux.iso            # list the root
    python tools/extract_iso.py images/linux.iso BOOT        # list a directory
    python tools/extract_iso.py images/linux.iso ROOT.BIN images/root.bin
    python tools/extract_iso.py alpine.iso BOOT/VMLINUZ-VIRT images/vmlinuz-virt

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


def entries_of(data, lba, size):
    """Directory entries, without the "." and ".." records."""
    for name, extent, length, is_dir in listdir(data, lba, size):
        if len(name) == 1 and name in (chr(0), chr(1)):
            continue
        yield name, extent, length, is_dir


def bare_name(name):
    """An ISO9660 name without its ";1" version suffix and trailing dot."""
    return name.split(';')[0].upper().rstrip('.')


def resolve(data, path):
    """Walk a slash-separated path from the root.

    Alpine keeps its kernel and initramfs under /boot, so one level of
    directory records is not enough; the 32-bit image happened to have both
    of its files at the root, which is why this went unnoticed.
    """
    lba, size = root_of(data)
    parts = [p for p in path.upper().split('/') if p]
    for depth, want in enumerate(parts):
        for name, extent, length, is_dir in entries_of(data, lba, size):
            if bare_name(name) != want and name.upper() != want:
                continue
            if depth == len(parts) - 1:
                return extent, length, is_dir
            if not is_dir:
                sys.exit('%s is not a directory' % want)
            lba, size = extent, length
            break
        else:
            sys.exit('no such entry in the image: %s' % want)
    return lba, size, True


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data = open(sys.argv[1], 'rb').read()

    # With no output file this lists a directory: the root by default, or
    # whatever path is named.
    if len(sys.argv) < 4:
        if len(sys.argv) == 2:
            lba, size = root_of(data)
        else:
            lba, size, is_dir = resolve(data, sys.argv[2])
            if not is_dir:
                sys.exit('%s is a file; give an output path to extract it'
                         % sys.argv[2])
        for name, _, length, is_dir in entries_of(data, lba, size):
            print('%-24s %10d  %s' % (name, length, 'dir' if is_dir else 'file'))
        return

    if len(sys.argv) != 4:
        sys.exit('usage: extract_iso.py <iso> <PATH> <outfile>')
    extent, length, is_dir = resolve(data, sys.argv[2])
    if is_dir:
        sys.exit('%s is a directory' % sys.argv[2])
    blob = data[extent * SECTOR:extent * SECTOR + length]
    open(sys.argv[3], 'wb').write(blob)
    print('wrote %s (%d bytes)' % (sys.argv[3], len(blob)))


main()
