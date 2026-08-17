import struct

golden = open('images/golden_kernel.bin', 'rb').read()

e_type, e_machine = struct.unpack_from('<HH', golden, 16)
e_entry = struct.unpack_from('<I', golden, 24)[0]
e_phoff = struct.unpack_from('<I', golden, 28)[0]
e_phentsize = struct.unpack_from('<H', golden, 42)[0]
e_phnum = struct.unpack_from('<H', golden, 44)[0]

print("ELF type:", e_type, "machine:", e_machine)
print("entry: 0x%X" % e_entry)
print("phoff:", e_phoff, "phentsize:", e_phentsize, "phnum:", e_phnum)

print("\nProgram headers:")
for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    p_type, p_offset, p_vaddr, p_paddr = struct.unpack_from('<IIII', golden, off)
    p_filesz, p_memsz, p_flags, p_align = struct.unpack_from('<IIII', golden, off+16)
    print("  [%d] type=%d off=0x%X vaddr=0x%X paddr=0x%X filesz=0x%X memsz=0x%X flags=%d align=0x%X" %
          (i, p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align))
