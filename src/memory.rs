//! Flat memory for real and protected mode.
//!
//! Real-mode physical addresses are 20 bits wide; protected mode uses 32-bit
//! physical addresses. The backing store is 256 MiB, and 32-bit addresses are
//! masked to that size. `Memory::SIZE` is the single source of truth for the
//! RAM size — the BIOS E820/E801/0x88 map and the boot loader's `boot_params`
//! derive their values from it, so scaling the RAM is a one-line change.

pub struct Memory {
    pub data: Vec<u8>,
    /// Memory-mapped VGA text window at physical 0xB8000 (80x25 cells, each
    /// `char | (attr << 8)`). Reads/writes in the 0xB8000-0xB8FFF range are
    /// routed here so the CPU can write the text screen directly (as Linux
    /// does) rather than only through the BIOS.
    pub vga_text: Vec<u16>,
}

/// Physical address of the VGA text window.
pub const VGA_TEXT_BASE: usize = 0xB8000;
/// Size of the VGA text window in bytes (80x25 cells x 2 bytes).
pub const VGA_TEXT_SIZE: usize = 80 * 25 * 2;

impl Memory {
    /// Backing store size (128 MiB). This is the single source of truth for
    /// the machine's RAM: the BIOS E820/E801/0x88 memory map and the boot
    /// loader's `boot_params` derive their values from this, so scaling the
    /// RAM is a one-line change here. Reduced from 256 MiB to 128 MiB for
    /// better cache locality during development.
    pub const SIZE: usize = 128 << 20;

    pub fn new() -> Self {
        Memory {
            data: vec![0; Self::SIZE],
            vga_text: vec![0x0720; 80 * 25],
        }
    }

    /// True if `addr` falls inside the VGA text window.
    #[inline]
    pub fn in_vga_text(&self, addr: usize) -> bool {
        addr >= VGA_TEXT_BASE && addr < VGA_TEXT_BASE + VGA_TEXT_SIZE
    }

    /// Translate a real-mode `segment:offset` pair to a 20-bit physical address.
    #[inline]
    pub fn phys(segment: u16, offset: u16) -> usize {
        (((segment as u32) << 4) + offset as u32) as usize & 0xFFFFF
    }

    /// Mask a 32-bit physical address to the backing store.
    #[inline]
    pub fn phys32(addr: u32) -> usize {
        (addr as usize) & (Self::SIZE - 1)
    }

    /// Read a byte at a physical address.
    #[inline]
    pub fn read_u8(&self, addr: usize) -> u8 {
        if addr >= VGA_TEXT_BASE && addr < VGA_TEXT_BASE + VGA_TEXT_SIZE {
            let cell = self.vga_text[(addr - VGA_TEXT_BASE) / 2];
            return if (addr - VGA_TEXT_BASE) % 2 == 0 { (cell & 0xFF) as u8 } else { (cell >> 8) as u8 };
        }
        // SAFETY: addr is masked to SIZE, which is within the data Vec.
        unsafe { *self.data.get_unchecked(addr & (Self::SIZE - 1)) }
    }

    /// Read a byte at a physical address, skipping the VGA text window check.
    /// Used for instruction fetch (code is never in VGA memory).
    #[inline]
    pub fn read_u8_raw(&self, addr: usize) -> u8 {
        // SAFETY: addr is masked to SIZE, which is within the data Vec.
        unsafe { *self.data.get_unchecked(addr & (Self::SIZE - 1)) }
    }

    /// Read a little-endian u16 at a physical address.
    #[inline]
    pub fn read_u16(&self, addr: usize) -> u16 {
        if addr >= VGA_TEXT_BASE && addr + 1 < VGA_TEXT_BASE + VGA_TEXT_SIZE {
            return self.vga_text[(addr - VGA_TEXT_BASE) / 2];
        }
        // Fast path: read directly from the backing store.
        let a = addr & (Self::SIZE - 1);
        if a + 1 < Self::SIZE {
            // SAFETY: a and a+1 are within bounds.
            unsafe {
                let lo = *self.data.get_unchecked(a) as u16;
                let hi = *self.data.get_unchecked(a + 1) as u16;
                return lo | (hi << 8);
            }
        }
        // Wraparound fallback.
        let lo = self.read_u8(addr) as u16;
        let hi = self.read_u8(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    /// Read a little-endian u32 at a physical address.
    #[inline]
    pub fn read_u32(&self, addr: usize) -> u32 {
        let a = addr & (Self::SIZE - 1);
        if a + 3 < Self::SIZE {
            // Fast path: read directly from the backing store.
            // SAFETY: a..a+3 are within bounds.
            unsafe {
                let b0 = *self.data.get_unchecked(a) as u32;
                let b1 = *self.data.get_unchecked(a + 1) as u32;
                let b2 = *self.data.get_unchecked(a + 2) as u32;
                let b3 = *self.data.get_unchecked(a + 3) as u32;
                return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            }
        }
        // Wraparound fallback.
        let b0 = self.read_u8(addr) as u32;
        let b1 = self.read_u8(addr.wrapping_add(1)) as u32;
        let b2 = self.read_u8(addr.wrapping_add(2)) as u32;
        let b3 = self.read_u8(addr.wrapping_add(3)) as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    /// Read a little-endian u64 at a physical address (used for GDT/IDT
    /// descriptors).
    #[inline]
    pub fn read_u64(&self, addr: usize) -> u64 {
        let b0 = self.read_u8(addr) as u64;
        let b1 = self.read_u8(addr.wrapping_add(1)) as u64;
        let b2 = self.read_u8(addr.wrapping_add(2)) as u64;
        let b3 = self.read_u8(addr.wrapping_add(3)) as u64;
        let b4 = self.read_u8(addr.wrapping_add(4)) as u64;
        let b5 = self.read_u8(addr.wrapping_add(5)) as u64;
        let b6 = self.read_u8(addr.wrapping_add(6)) as u64;
        let b7 = self.read_u8(addr.wrapping_add(7)) as u64;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
    }

    /// Write a byte to a physical address.
    #[inline]
    pub fn write_u8(&mut self, addr: usize, val: u8) {
        if addr >= VGA_TEXT_BASE && addr < VGA_TEXT_BASE + VGA_TEXT_SIZE {
            let idx = (addr - VGA_TEXT_BASE) / 2;
            let cell = self.vga_text[idx];
            let new = if (addr - VGA_TEXT_BASE) % 2 == 0 {
                (cell & 0xFF00) | val as u16
            } else {
                (cell & 0x00FF) | ((val as u16) << 8)
            };
            self.vga_text[idx] = new;
            return;
        }
        let a = addr & (Self::SIZE - 1);
        // SAFETY: a is within bounds.
        unsafe { *self.data.get_unchecked_mut(a) = val; }
    }

    /// Write a little-endian u16 to a physical address.
    #[inline]
    pub fn write_u16(&mut self, addr: usize, val: u16) {
        if addr >= VGA_TEXT_BASE && addr + 1 < VGA_TEXT_BASE + VGA_TEXT_SIZE {
            self.vga_text[(addr - VGA_TEXT_BASE) / 2] = val;
            return;
        }
        let a = addr & (Self::SIZE - 1);
        if a + 1 < Self::SIZE {
            // Fast path: write directly to the backing store.
            // SAFETY: a and a+1 are within bounds.
            unsafe {
                *self.data.get_unchecked_mut(a) = (val & 0xFF) as u8;
                *self.data.get_unchecked_mut(a + 1) = (val >> 8) as u8;
            }
            return;
        }
        // Wraparound fallback.
        self.write_u8(addr, (val & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8);
    }

    /// Write a little-endian u32 to a physical address.
    #[inline]
    pub fn write_u32(&mut self, addr: usize, val: u32) {
        let a = addr & (Self::SIZE - 1);
        if a + 3 < Self::SIZE {
            // Fast path: write directly to the backing store.
            // SAFETY: a..a+3 are within bounds.
            unsafe {
                *self.data.get_unchecked_mut(a) = (val & 0xFF) as u8;
                *self.data.get_unchecked_mut(a + 1) = ((val >> 8) & 0xFF) as u8;
                *self.data.get_unchecked_mut(a + 2) = ((val >> 16) & 0xFF) as u8;
                *self.data.get_unchecked_mut(a + 3) = ((val >> 24) & 0xFF) as u8;
            }
            return;
        }
        // Wraparound fallback.
        self.write_u8(addr, (val & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(1), ((val >> 8) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(2), ((val >> 16) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(3), ((val >> 24) & 0xFF) as u8);
    }

    /// Write a little-endian u64 to a physical address.
    #[inline]
    pub fn write_u64(&mut self, addr: usize, val: u64) {
        self.write_u8(addr, (val & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(1), ((val >> 8) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(2), ((val >> 16) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(3), ((val >> 24) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(4), ((val >> 32) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(5), ((val >> 40) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(6), ((val >> 48) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(7), ((val >> 56) & 0xFF) as u8);
    }

    /// Read an f64 (8 bytes, little-endian) at a physical address.
    #[inline]
    pub fn read_f64(&self, addr: usize) -> f64 {
        f64::from_bits(self.read_u64(addr))
    }

    /// Write an f64 (8 bytes, little-endian) to a physical address.
    #[inline]
    pub fn write_f64(&mut self, addr: usize, val: f64) {
        self.write_u64(addr, val.to_bits());
    }

    /// Load a block of bytes at a physical address (used to install code/data).
    pub fn load(&mut self, addr: usize, bytes: &[u8]) {
        for (i, b) in bytes.iter().enumerate() {
            self.write_u8(addr.wrapping_add(i), *b);
        }
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_offset_translation() {
        assert_eq!(Memory::phys(0x0000, 0x7C00), 0x07C00);
        assert_eq!(Memory::phys(0x07C0, 0x0000), 0x07C00);
        // Wraparound at 1 MiB.
        assert_eq!(Memory::phys(0xFFFF, 0x0010), 0x00000);
    }

    #[test]
    fn read_write_roundtrip() {
        let mut m = Memory::new();
        m.write_u16(0x100, 0x1234);
        assert_eq!(m.read_u8(0x100), 0x34);
        assert_eq!(m.read_u8(0x101), 0x12);
        assert_eq!(m.read_u16(0x100), 0x1234);
    }

    #[test]
    fn read_write_u32() {
        let mut m = Memory::new();
        m.write_u32(0x200, 0xDEADBEEF);
        assert_eq!(m.read_u32(0x200), 0xDEADBEEF);
        assert_eq!(m.read_u8(0x200), 0xEF);
        assert_eq!(m.read_u8(0x203), 0xDE);
    }

    #[test]
    fn read_write_u64() {
        let mut m = Memory::new();
        m.write_u64(0x300, 0x0123456789ABCDEF);
        assert_eq!(m.read_u64(0x300), 0x0123456789ABCDEF);
    }

    #[test]
    fn vga_text_window_is_memory_mapped() {
        let mut m = Memory::new();
        // Write a text cell (char 'A', attr 0x07) at 0xB8000.
        m.write_u16(VGA_TEXT_BASE, (0x07 << 8) | b'A' as u16);
        assert_eq!(m.vga_text[0] & 0xFF, b'A' as u16);
        assert_eq!(m.vga_text[0] >> 8, 0x07);
        // Reading it back through the physical address works too.
        assert_eq!(m.read_u8(VGA_TEXT_BASE), b'A');
        assert_eq!(m.read_u8(VGA_TEXT_BASE + 1), 0x07);
        // Byte writes update the cell.
        m.write_u8(VGA_TEXT_BASE + 2, b'B');
        assert_eq!(m.vga_text[1] & 0xFF, b'B' as u16);
        // The backing store is not clobbered by VGA writes.
        assert_eq!(m.data[VGA_TEXT_BASE], 0);
    }
}
