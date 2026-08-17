import sys

emu = open('images/emu_dump.bin', 'rb').read()
golden = open('images/golden_kernel.bin', 'rb').read()

# Exact golden ELF header (first 16 bytes)
hdr = golden[:16]
print("searching for exact golden header:", hdr.hex())

# Find all occurrences
idxs = []
start = 0
while True:
    i = emu.find(hdr, start)
    if i < 0:
        break
    idxs.append(i)
    start = i + 1
print("occurrences in emu dump:", [hex(0x100000+i) for i in idxs])

# Also search for just the ELF magic + valid class (01 = 32-bit)
# and check what's at various candidate decompression addresses
# The buildroot kernel usually decompresses to 0x100000 (in place) or to a
# higher address. Let's check common addresses.
for addr in [0x100000, 0x200000, 0x300000, 0x400000, 0x1000000]:
    off = addr - 0x100000
    if off >= 0 and off < len(emu):
        print("bytes at 0x%X:" % addr, emu[off:off+16].hex())

# Let's also check: maybe the decompressor writes the ELF and then the ELF
# loader relocates it. Let's look at what's at the very start of the dump (0x100000)
print("start of dump (0x100000):", emu[:32].hex())
