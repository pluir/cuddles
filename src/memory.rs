//! Flat memory for real, protected and long mode.
//!
//! Real-mode physical addresses are 20 bits wide; protected mode uses 32-bit
//! physical addresses; long mode uses up to 52. The backing store is sized at
//! construction — `Memory::DEFAULT_SIZE` is only the default, and
//! `Memory::with_size` builds a machine with any amount of RAM the host can
//! allocate. `Memory::ram_size` is the single source of truth for how much
//! RAM the machine has: the BIOS E820/E801/0x88 map and the boot loader's
//! `boot_params` are all derived from `e820_map()` below, so scaling the RAM
//! is one argument rather than an edit in four places.
//!
//! ## Where the RAM lives
//!
//! A PC does not have one contiguous run of RAM. The top of the 32-bit
//! address space belongs to devices, so a machine with more RAM than fits
//! underneath that window has the remainder wired *above* 4 GiB. This models
//! the same thing: RAM occupies `0 .. min(ram, MMIO_HOLE_START)` and then
//! `4 GiB .. 4 GiB + (ram - MMIO_HOLE_START)`, with the gap between reading
//! as an unclaimed bus. `slot()` is the one place that knows the layout, and
//! it is a single comparison for every address on a machine whose RAM fits
//! below the hole.
//!
//! Addresses with no RAM behind them read as `0xFF` and swallow writes, which
//! is what an unclaimed bus does — not an alias back into low memory. The
//! aliasing this replaced was load-bearing by accident: masking every address
//! into a 128 MiB store was quietly standing in for the kernel's direct map
//! each time a descriptor table was read at its *linear* address. That is now
//! done properly, in `Cpu::linear_to_phys_ro`.

pub struct Memory {
    pub data: Vec<u8>,
    /// Bytes of RAM below the MMIO hole; also the index at which the
    /// above-4 GiB region continues in `data`.
    low_size: usize,
    /// Total RAM in bytes (the length of `data`).
    ram_size: usize,
    /// Memory-mapped VGA text window at physical 0xB8000, one `u16` per cell
    /// (`char | (attr << 8)`). Reads and writes anywhere in the aperture are
    /// routed here so the CPU can drive the text screen directly, as Linux
    /// does, rather than only through the BIOS.
    pub vga_text: Vec<u16>,
    /// X86EMU_WATCH_STORE: physical address to log every store to, recorded
    /// where the store actually happens rather than where it was translated.
    pub watch_store: Option<usize>,
    /// (physical address, value, width in bytes, RIP) for each logged store.
    pub store_log: Vec<(usize, u64, u8, u64)>,
    /// RIP of the instruction currently executing, kept in step by `Cpu::step`
    /// only while a store watch is armed.
    pub cur_eip: u64,
}

/// Physical address of the VGA text window.
pub const VGA_TEXT_BASE: usize = 0xB8000;
/// Size of the VGA text window in bytes: the whole 32 KiB colour-text
/// aperture, not just the 4000 bytes one screenful occupies.
///
/// The extra space is not slack. A VGA text console scrolls by moving the
/// CRTC's *start address* through this buffer rather than copying characters,
/// so the lines that have scrolled off the top are still sitting in it. With
/// only one screenful of storage, every line after the twenty-fifth is either
/// lost or wraps back over the first -- which looks exactly like a machine
/// that stopped booting.
pub const VGA_TEXT_SIZE: usize = 0x8000;
/// Number of character cells in that window.
pub const VGA_TEXT_CELLS: usize = VGA_TEXT_SIZE / 2;

/// Where the 32-bit MMIO window begins. RAM that would otherwise land between
/// here and 4 GiB is wired above 4 GiB instead, which is what a real chipset
/// does and what every guest's E820 parser already expects.
pub const MMIO_HOLE_START: u64 = 0xC000_0000;
/// The 4 GiB mark, where RAM resumes on a machine with more than
/// `MMIO_HOLE_START` bytes of it.
pub const HIGH_RAM_BASE: u64 = 0x1_0000_0000;

/// The physical address handed back for a linear address with no mapping.
/// It is deliberately far above any real machine's RAM, so a read through it
/// answers `0xFF` (an open bus) instead of landing on whatever lives at zero.
pub const UNBACKED: usize = usize::MAX / 2;

/// One entry of the E820 memory map: (base, length, type). Type 1 is usable
/// RAM, 2 is reserved.
pub type E820Entry = (u64, u64, u32);

/// The machine's memory map for `ram` bytes of RAM, in the form the BIOS
/// `INT 0x15/E820` service and the Linux `boot_params` table both want.
///
/// This is the single description of the layout: the BIOS handler and the
/// boot loader both call it, so a machine cannot describe itself two
/// different ways to two different consumers.
pub fn e820_map(ram: u64) -> Vec<E820Entry> {
    let low = ram.min(MMIO_HOLE_START);
    let mut map: Vec<E820Entry> = vec![
        (0x0000_0000, 0x0009_FC00, 1), // conventional memory (640K - 1K)
        (0x0009_FC00, 0x0000_0400, 2), // EBDA
        (0x000A_0000, 0x0006_0000, 2), // VGA / ROM area
    ];
    if low > 0x0010_0000 {
        map.push((0x0010_0000, low - 0x0010_0000, 1)); // extended memory
    }
    if ram > MMIO_HOLE_START {
        // The device window, then the rest of the RAM above 4 GiB.
        map.push((MMIO_HOLE_START, HIGH_RAM_BASE - MMIO_HOLE_START, 2));
        map.push((HIGH_RAM_BASE, ram - MMIO_HOLE_START, 1));
    }
    map
}

impl Memory {
    /// Default backing store size (128 MiB) when no size is given. Enough for
    /// the 32-bit Linux boot without making every test allocate a gigabyte.
    pub const DEFAULT_SIZE: usize = 128 << 20;

    pub fn new() -> Self {
        Self::with_size(Self::DEFAULT_SIZE)
    }

    /// Build a machine with `bytes` of RAM, rounded up to a whole 4 KiB page
    /// and never smaller than 1 MiB.
    pub fn with_size(bytes: usize) -> Self {
        let size = (bytes.max(1 << 20) + 0xFFF) & !0xFFF;
        let low_size = size.min(MMIO_HOLE_START as usize);
        Memory {
            data: vec![0; size],
            low_size,
            ram_size: size,
            vga_text: vec![0x0720; VGA_TEXT_CELLS],
            watch_store: std::env::var("X86EMU_WATCH_STORE").ok()
                .and_then(|v| usize::from_str_radix(v.trim_start_matches("0x"), 16).ok()),
            store_log: Vec::new(),
            cur_eip: 0,
        }
    }

    /// Total RAM in bytes.
    #[inline]
    pub fn ram_size(&self) -> u64 { self.ram_size as u64 }

    /// The highest physical address that has RAM behind it, exclusive. On a
    /// machine larger than the MMIO hole this is past the 4 GiB mark.
    #[inline]
    pub fn top_of_ram(&self) -> u64 {
        if self.ram_size as u64 > MMIO_HOLE_START {
            HIGH_RAM_BASE + (self.ram_size as u64 - MMIO_HOLE_START)
        } else {
            self.ram_size as u64
        }
    }

    /// This machine's E820 memory map.
    pub fn e820(&self) -> Vec<E820Entry> { e820_map(self.ram_size()) }

    /// Map a physical address onto an index into the backing store, or `None`
    /// when no RAM answers at that address (the MMIO hole, or past the top).
    ///
    /// The first comparison covers every address on a machine whose RAM fits
    /// below the hole, which is the case the fast paths are tuned for.
    #[inline]
    pub fn slot(&self, addr: usize) -> Option<usize> {
        if addr < self.low_size {
            return Some(addr);
        }
        let high_base = HIGH_RAM_BASE as usize;
        if addr >= high_base {
            let off = (addr - high_base).wrapping_add(self.low_size);
            if off < self.ram_size {
                return Some(off);
            }
        }
        None
    }

    /// Log a store that overlaps the watched physical address.
    fn note_store(&mut self, addr: usize, val: u64, width: u8) {
        if let Some(w) = self.watch_store {
            if addr <= w + 7 && w < addr + width as usize {
                if self.store_log.len() >= 64 { self.store_log.remove(0); }
                let eip = self.cur_eip;
                self.store_log.push((addr, val, width, eip));
            }
        }
    }

    /// True if `addr` falls inside the VGA text window.
    pub fn in_vga_text(&self, addr: usize) -> bool {
        addr >= VGA_TEXT_BASE && addr < VGA_TEXT_BASE + VGA_TEXT_SIZE
    }

    /// Translate a real-mode `segment:offset` pair to a 20-bit physical address.
    #[inline]
    pub fn phys(segment: u16, offset: u16) -> usize {
        (((segment as u32) << 4) + offset as u32) as usize & 0xFFFFF
    }

    /// Widen a 32-bit physical address to a backing-store address.
    ///
    /// This used to mask the address down into the backing store, which
    /// aliased anything above the RAM size back into low memory. It no longer
    /// does: an address with no RAM behind it now reads as an open bus, which
    /// is both what hardware does and what makes a machine with more than
    /// 4 GiB describable at all.
    #[inline]
    pub fn phys32(addr: u32) -> usize {
        addr as usize
    }

    /// Read a byte at a physical address.
    #[inline]
    pub fn read_u8(&self, addr: usize) -> u8 {
        if addr >= VGA_TEXT_BASE && addr < VGA_TEXT_BASE + VGA_TEXT_SIZE {
            let cell = self.vga_text[(addr - VGA_TEXT_BASE) / 2];
            return if (addr - VGA_TEXT_BASE) % 2 == 0 { (cell & 0xFF) as u8 } else { (cell >> 8) as u8 };
        }
        match self.slot(addr) {
            // SAFETY: slot() returned an index inside data.
            Some(a) => unsafe { *self.data.get_unchecked(a) },
            None => 0xFF,
        }
    }

    /// Read a byte at a physical address, skipping the VGA text window check.
    /// Used for instruction fetch (code is never in VGA memory).
    #[inline]
    pub fn read_u8_raw(&self, addr: usize) -> u8 {
        match self.slot(addr) {
            // SAFETY: slot() returned an index inside data.
            Some(a) => unsafe { *self.data.get_unchecked(a) },
            None => 0xFF,
        }
    }

    /// Read a little-endian u16 at a physical address.
    #[inline]
    pub fn read_u16(&self, addr: usize) -> u16 {
        if addr >= VGA_TEXT_BASE && addr + 1 < VGA_TEXT_BASE + VGA_TEXT_SIZE {
            return self.vga_text[(addr - VGA_TEXT_BASE) / 2];
        }
        if let Some(a) = self.slot(addr) {
            if a + 1 < self.ram_size {
                // SAFETY: a and a+1 are within bounds.
                unsafe {
                    let lo = *self.data.get_unchecked(a) as u16;
                    let hi = *self.data.get_unchecked(a + 1) as u16;
                    return lo | (hi << 8);
                }
            }
        }
        // Straddles the end of a region, or is unbacked: byte at a time.
        let lo = self.read_u8(addr) as u16;
        let hi = self.read_u8(addr.wrapping_add(1)) as u16;
        lo | (hi << 8)
    }

    /// Read a little-endian u32 at a physical address.
    #[inline]
    pub fn read_u32(&self, addr: usize) -> u32 {
        if addr + 3 < VGA_TEXT_BASE + VGA_TEXT_SIZE && addr + 3 >= VGA_TEXT_BASE {
            return self.read_u16(addr) as u32 | ((self.read_u16(addr + 2) as u32) << 16);
        }
        if let Some(a) = self.slot(addr) {
            if a + 3 < self.ram_size {
                // SAFETY: a..a+3 are within bounds.
                unsafe {
                    let b0 = *self.data.get_unchecked(a) as u32;
                    let b1 = *self.data.get_unchecked(a + 1) as u32;
                    let b2 = *self.data.get_unchecked(a + 2) as u32;
                    let b3 = *self.data.get_unchecked(a + 3) as u32;
                    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                }
            }
        }
        let b0 = self.read_u8(addr) as u32;
        let b1 = self.read_u8(addr.wrapping_add(1)) as u32;
        let b2 = self.read_u8(addr.wrapping_add(2)) as u32;
        let b3 = self.read_u8(addr.wrapping_add(3)) as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    /// Read a little-endian u64 at a physical address (page-table entries,
    /// GDT/IDT descriptors, and every 64-bit operand).
    #[inline]
    pub fn read_u64(&self, addr: usize) -> u64 {
        if addr + 7 >= VGA_TEXT_BASE && addr < VGA_TEXT_BASE + VGA_TEXT_SIZE {
            let lo = self.read_u32(addr) as u64;
            let hi = self.read_u32(addr.wrapping_add(4)) as u64;
            return lo | (hi << 32);
        }
        if let Some(a) = self.slot(addr) {
            if a + 7 < self.ram_size {
                // SAFETY: a..a+7 are within bounds, and the read is unaligned-safe.
                unsafe {
                    let p = self.data.as_ptr().add(a) as *const [u8; 8];
                    return u64::from_le_bytes(std::ptr::read_unaligned(p));
                }
            }
        }
        let lo = self.read_u32(addr) as u64;
        let hi = self.read_u32(addr.wrapping_add(4)) as u64;
        lo | (hi << 32)
    }

    /// Write a byte to a physical address.
    #[inline]
    pub fn write_u8(&mut self, addr: usize, val: u8) {
        if self.watch_store.is_some() { self.note_store(addr, val as u64, 1); }
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
        if let Some(a) = self.slot(addr) {
            // SAFETY: slot() returned an index inside data.
            unsafe { *self.data.get_unchecked_mut(a) = val; }
        }
    }

    /// Write a little-endian u16 to a physical address.
    #[inline]
    pub fn write_u16(&mut self, addr: usize, val: u16) {
        if self.watch_store.is_some() { self.note_store(addr, val as u64, 2); }
        if addr >= VGA_TEXT_BASE && addr + 1 < VGA_TEXT_BASE + VGA_TEXT_SIZE {
            self.vga_text[(addr - VGA_TEXT_BASE) / 2] = val;
            return;
        }
        if let Some(a) = self.slot(addr) {
            if a + 1 < self.ram_size {
                // SAFETY: a and a+1 are within bounds.
                unsafe {
                    *self.data.get_unchecked_mut(a) = (val & 0xFF) as u8;
                    *self.data.get_unchecked_mut(a + 1) = (val >> 8) as u8;
                }
                return;
            }
        }
        self.write_u8(addr, (val & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(1), (val >> 8) as u8);
    }

    /// Write a little-endian u32 to a physical address.
    #[inline]
    pub fn write_u32(&mut self, addr: usize, val: u32) {
        if self.watch_store.is_some() { self.note_store(addr, val as u64, 4); }
        // The VGA text window has to be checked here too, not only in the 8-
        // and 16-bit paths: the console writes whole cells and cell pairs, so
        // a dword store is the common case, and letting it reach RAM makes
        // every screen update invisible.
        if addr + 3 < VGA_TEXT_BASE + VGA_TEXT_SIZE && addr + 3 >= VGA_TEXT_BASE {
            self.write_u16(addr, (val & 0xFFFF) as u16);
            self.write_u16(addr + 2, (val >> 16) as u16);
            return;
        }
        if let Some(a) = self.slot(addr) {
            if a + 3 < self.ram_size {
                // SAFETY: a..a+3 are within bounds.
                unsafe {
                    *self.data.get_unchecked_mut(a) = (val & 0xFF) as u8;
                    *self.data.get_unchecked_mut(a + 1) = ((val >> 8) & 0xFF) as u8;
                    *self.data.get_unchecked_mut(a + 2) = ((val >> 16) & 0xFF) as u8;
                    *self.data.get_unchecked_mut(a + 3) = ((val >> 24) & 0xFF) as u8;
                }
                return;
            }
        }
        self.write_u8(addr, (val & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(1), ((val >> 8) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(2), ((val >> 16) & 0xFF) as u8);
        self.write_u8(addr.wrapping_add(3), ((val >> 24) & 0xFF) as u8);
    }

    /// Write a little-endian u64 to a physical address.
    #[inline]
    pub fn write_u64(&mut self, addr: usize, val: u64) {
        if self.watch_store.is_some() { self.note_store(addr, val, 8); }
        if addr + 7 >= VGA_TEXT_BASE && addr < VGA_TEXT_BASE + VGA_TEXT_SIZE {
            self.write_u32(addr, val as u32);
            self.write_u32(addr.wrapping_add(4), (val >> 32) as u32);
            return;
        }
        if let Some(a) = self.slot(addr) {
            if a + 7 < self.ram_size {
                // SAFETY: a..a+7 are within bounds, and the write is
                // unaligned-safe.
                unsafe {
                    let p = self.data.as_mut_ptr().add(a) as *mut [u8; 8];
                    std::ptr::write_unaligned(p, val.to_le_bytes());
                }
                return;
            }
        }
        for i in 0..8usize {
            self.write_u8(addr.wrapping_add(i), (val >> (i * 8)) as u8);
        }
    }

    /// Read a 32-bit IEEE single from a physical address.
    pub fn read_f32(&self, addr: usize) -> f32 {
        f32::from_bits(self.read_u32(addr))
    }

    /// Write a 32-bit IEEE single to a physical address.
    pub fn write_f32(&mut self, addr: usize, val: f32) {
        self.write_u32(addr, val.to_bits());
    }

    /// Read an f64 (8 bytes, little-endian) at a physical address.
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
        assert_eq!(m.read_u32(0x300), 0x89ABCDEF);
        assert_eq!(m.read_u32(0x304), 0x01234567);
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

    #[test]
    fn unbacked_addresses_read_as_open_bus_and_swallow_writes() {
        // Past the top of RAM there is nothing: reads are 0xFF and writes go
        // nowhere. They used to alias back into low memory, which quietly
        // corrupted whatever happened to live at the folded address.
        let mut m = Memory::with_size(16 << 20);
        let past = 32 << 20;
        m.write_u32(past, 0xDEAD_BEEF);
        assert_eq!(m.read_u32(past), 0xFFFF_FFFF);
        // The address it would have aliased to is untouched.
        assert_eq!(m.read_u32(past & ((16 << 20) - 1)), 0);
    }

    #[test]
    fn ram_above_four_gib_is_reachable() {
        // A machine with more RAM than fits below the MMIO hole wires the
        // remainder above 4 GiB, and it must be addressable there.
        let ram = MMIO_HOLE_START as usize + (64 << 20);
        let mut m = Memory::with_size(ram);
        assert_eq!(m.top_of_ram(), HIGH_RAM_BASE + (64 << 20));
        let high = HIGH_RAM_BASE as usize + 0x1234;
        m.write_u64(high, 0x0123_4567_89AB_CDEF);
        assert_eq!(m.read_u64(high), 0x0123_4567_89AB_CDEF);
        // The hole between the top of low RAM and 4 GiB is not RAM.
        m.write_u32(MMIO_HOLE_START as usize + 0x1000, 1);
        assert_eq!(m.read_u32(MMIO_HOLE_START as usize + 0x1000), 0xFFFF_FFFF);
        // And the last byte of RAM is the last byte, with nothing past it.
        assert_eq!(m.read_u8(m.top_of_ram() as usize), 0xFF);
    }

    #[test]
    fn e820_describes_a_small_machine_as_four_regions() {
        let map = e820_map(128 << 20);
        assert_eq!(map.len(), 4);
        assert_eq!(map[3], (0x0010_0000, (128 << 20) - 0x0010_0000, 1));
    }

    #[test]
    fn e820_splits_a_large_machine_around_the_mmio_hole() {
        let ram: u64 = 8 << 30; // 8 GiB
        let map = e820_map(ram);
        // Low RAM stops at the hole...
        assert_eq!(map[3], (0x0010_0000, MMIO_HOLE_START - 0x0010_0000, 1));
        // ...the hole itself is reserved...
        assert_eq!(map[4], (MMIO_HOLE_START, HIGH_RAM_BASE - MMIO_HOLE_START, 2));
        // ...and the rest is above 4 GiB.
        assert_eq!(map[5], (HIGH_RAM_BASE, ram - MMIO_HOLE_START, 1));
        // Every usable byte is accounted for exactly once.
        let usable: u64 = map.iter().filter(|e| e.2 == 1).map(|e| e.1).sum();
        assert_eq!(usable, ram - 0x400 - 0x6_0000);
    }
}
