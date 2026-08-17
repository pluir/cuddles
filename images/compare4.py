import struct

emu = open('images/emu_full.bin', 'rb').read()
golden = open('images/golden_kernel.bin', 'rb').read()

# Search for the exact golden ELF header
hdr = golden[:16]
print("searching for golden ELF header:", hdr.hex())
idxs = []
start = 0
while True:
    i = emu.find(hdr, start)
    if i < 0:
        break
    idxs.append(i)
    start = i + 1
print("occurrences:", [hex(i) for i in idxs])

# Also search for ELF magic 0x7f 'ELF' with valid 32-bit class (01)
print("\nAll ELF magic occurrences:")
start = 0
count = 0
while True:
    i = emu.find(b'\x7fELF', start)
    if i < 0:
        break
    # check class byte
    cls = emu[i+4] if i+4 < len(emu) else -1
    print("  0x%X class=%d" % (i, cls))
    start = i + 1
    count += 1
    if count > 30:
        print("  ...")
        break

# Check the region around 0x200000 (where decompressor may write)
print("\nbytes at 0x200000:", emu[0x200000:0x200000+32].hex())
print("bytes at 0x300000:", emu[0x300000:0x300000+32].hex())
