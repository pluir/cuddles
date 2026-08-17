import os
b = bytearray(512)
# mov si, 0x7C1E
b[0:3] = bytes([0xBE, 0x1E, 0x7C])
# loop: lodsb
b[3] = 0xAC
# cmp al, 0
b[4:6] = bytes([0x3C, 0x00])
# jz +6 -> 0x7C0E
b[6:8] = bytes([0x74, 0x06])
# mov ah, 0x0E
b[8:10] = bytes([0xB4, 0x0E])
# int 0x10
b[10:12] = bytes([0xCD, 0x10])
# jmp -11 -> 0x7C03
b[12:14] = bytes([0xEB, 0xF5])
# done: hlt
b[14] = 0xF4
# jmp -3 -> 0x7C0E
b[15:17] = bytes([0xEB, 0xFD])
# message at 0x7C1E
msg = b'Hello from x86emu!'
b[0x1E:0x1E+len(msg)] = msg
b[0x1E+len(msg)] = 0
# boot signature
b[510] = 0x55
b[511] = 0xAA
os.makedirs('examples', exist_ok=True)
with open('examples/boot.bin', 'wb') as f:
    f.write(bytes(b))
print('wrote', len(b), 'bytes')
