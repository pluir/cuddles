#!/usr/bin/env python
"""Pull the decompressed kernel ELF out of a bzImage.

    python tools/unpack_bzimage.py images/bzImage images/golden_kernel.bin

A bzImage is a small real-mode setup stub followed by a self-extracting
payload: a decompressor with the compressed kernel appended. The emulator's
`--kernel-elf` mode wants that inner ELF, so this does the decompressor's job
on the host — the in-kernel decompressor does not yet run correctly under
emulation, and unpacking here is the difference between debugging a kernel and
debugging a decompressor.

The setup header at file offset 0x1F1 says how long the stub is
(`setup_sects`) and how big the payload is (`syssize`, in 16-byte units).
Inside the payload, the compressed stream is found by looking for its magic
rather than by a fixed offset, which varies between kernel versions.
"""
import struct
import sys
import zlib

# (magic, human name, decompressor) for the formats a bzImage may use. Only
# gzip is handled without an external module; the rest are named so the error
# says what the image actually is instead of "not found".
GZIP_MAGIC = b'\x1f\x8b\x08'
OTHER_MAGICS = [
    (b'\xfd7zXZ\x00', 'xz'),
    (b'\x5d\x00\x00', 'lzma'),
    (b'BZh', 'bzip2'),
    (b'\x04\x22\x4d\x18', 'lz4'),
    (b'\x89LZO', 'lzo'),
]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data = open(sys.argv[1], 'rb').read()
    if data[0x1FE:0x200] != b'\x55\xaa':
        sys.exit('not a bzImage: no 0xAA55 boot flag')

    setup_sects = data[0x1F1] or 4  # 0 means the historical default of 4
    syssize = struct.unpack_from('<I', data, 0x1F4)[0]
    payload = data[(setup_sects + 1) * 512:][:syssize * 16]

    start = payload.find(GZIP_MAGIC)
    if start < 0:
        for magic, name in OTHER_MAGICS:
            if payload.find(magic) >= 0:
                sys.exit('payload is %s-compressed; this script only does gzip'
                         % name)
        sys.exit('no compressed stream found in the payload')

    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = d.decompress(payload[start:])
    if out[:4] != b'\x7fELF':
        sys.exit('decompressed payload is not an ELF (got %s)' % out[:4].hex())

    open(sys.argv[2], 'wb').write(out)
    print('wrote %s (%d bytes), compressed stream at payload offset %d'
          % (sys.argv[2], len(out), start))


main()
