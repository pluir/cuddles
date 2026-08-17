import struct

data = open('images/bzImage', 'rb').read()
setup_sects = data[0x1F1]
syssize = struct.unpack_from('<I', data, 0x1F4)[0]
kernel_offset = (setup_sects + 1) * 512
kernel_len = syssize * 16
print("kernel_offset:", kernel_offset, "kernel_len:", kernel_len)
print("kernel ends at file offset:", kernel_offset + kernel_len)
print("file size:", len(data))

# The decompressor copies from esi = 0x219EFC to edi = 0x313DCC, count 0x119F00 bytes.
# esi = ebp + 0x119efc where ebp = 0x100000 (the load address).
# So source physical = 0x219EFC, which corresponds to file offset:
src_phys = 0x219EFC
src_file = kernel_offset + (src_phys - 0x100000)
print("\nsource 0x219EFC -> file offset:", src_file)
print("  bytes at src_file:", data[src_file:src_file+16].hex())

# The gzip stream we found at payload offset 98 -> file 13922 -> phys 0x103662
gz_phys = 0x100000 + 98
gz_file = kernel_offset + 98
print("\ngzip at phys 0x%X -> file offset %d" % (gz_phys, gz_file))
print("  bytes:", data[gz_file:gz_file+16].hex())

# Check the END of the file - maybe compressed data is there
print("\nlast 16 bytes of file:", data[-16:].hex())

# Check what's at the end of the loaded kernel region
end_file = kernel_offset + kernel_len
print("bytes at end of kernel region (file %d):" % end_file, data[end_file:end_file+16].hex())

# The copy count is 0x119F00 bytes. Source region 0x219EFC..0x333DFC
# dest region 0x313DCC..0x42DCCC. Check if source has real data in file.
# Source spans file offsets src_file .. src_file + 0x119F00
src_end_file = src_file + 0x119F00
print("\nsource region file offsets: %d .. %d (len %d)" % (src_file, src_end_file, 0x119F00))
print("  is within file?", src_end_file <= len(data))

# Check a few bytes in the source region
for off in [0, 0x100, 0x1000, 0x10000, 0x100000]:
    fo = src_file + off
    if fo < len(data):
        print("  src+0x%X (file %d):" % (off, fo), data[fo:fo+8].hex())
