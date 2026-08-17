import sys

# Read the emulator's memory dump (from 0x100000)
emu = open('images/emu_dump.bin', 'rb').read()
print("emu dump size:", len(emu))

# Find ELF magic in the dump
idx = emu.find(b'\x7fELF')
print("ELF magic in emu dump at offset:", idx)
if idx >= 0:
    print("  (physical addr: 0x%X)" % (0x100000 + idx))
    print("  first 16 bytes:", emu[idx:idx+16].hex())

# Read the golden decompressed kernel
golden = open('images/golden_kernel.bin', 'rb').read()
print("golden size:", len(golden))
print("golden first 16:", golden[:16].hex())

# Compare the golden ELF against the emulator's dump at the found offset
if idx >= 0:
    # Compare as much as we have
    n = min(len(golden), len(emu) - idx)
    mismatches = []
    for i in range(n):
        if golden[i] != emu[idx+i]:
            mismatches.append(i)
            if len(mismatches) > 20:
                break
    if not mismatches:
        print("MATCH: emulator decompressed output matches golden for first", n, "bytes")
    else:
        print("MISMATCH: first", len(mismatches), "mismatches at golden offsets:", mismatches[:20])
        # Show context around first mismatch
        fm = mismatches[0]
        print("  golden[%d:%d]:" % (fm, fm+16), golden[fm:fm+16].hex())
        print("  emu[%d:%d]:" % (fm, fm+16), emu[idx+fm:idx+fm+16].hex())
