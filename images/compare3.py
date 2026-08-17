import struct

emu = open('images/emu_dump.bin', 'rb').read()
golden = open('images/golden_kernel.bin', 'rb').read()

# Segment 0: paddr 0x100000, filesz 0x1BE000, ELF offset 0x1000
seg0_off = 0x1000
seg0_paddr = 0x100000
seg0_filesz = 0x1BE000

# Emulator memory at 0x100000 (dump starts at 0x100000)
emu_seg0 = emu[0:seg0_filesz]
golden_seg0 = golden[seg0_off:seg0_off+seg0_filesz]

print("Comparing emulator[0x100000:0x%X] vs golden[0x%X:0x%X]" % (0x100000+seg0_filesz, seg0_off, seg0_off+seg0_filesz))

# Find first mismatch
mismatches = []
for i in range(seg0_filesz):
    if emu_seg0[i] != golden_seg0[i]:
        mismatches.append(i)
        if len(mismatches) > 10:
            break

if not mismatches:
    print("MATCH: emulator's kernel at 0x100000 matches golden segment 0 fully")
else:
    print("MISMATCH: first mismatches at offsets:", [hex(m) for m in mismatches])
    for m in mismatches[:5]:
        print("  off 0x%X (addr 0x%X): emu=%02X golden=%02X" % (m, 0x100000+m, emu_seg0[m], golden_seg0[m]))

# Also check: is the emulator's 0x100000 still the stub? Show first 32 bytes
print("\nemu[0x100000:0x100020]:", emu[0:32].hex())
print("golden seg0[0:32]:", golden_seg0[0:32].hex())
