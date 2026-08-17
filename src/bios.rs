//! Minimal BIOS layer: emulated `INT 0x10` (video), `INT 0x16` (keyboard)
//! and `INT 0x13` (disk) services, handled natively in Rust.
//!
//! The BIOS is not real machine code; it is a set of host-side service
//! routines dispatched from the `INT` instruction. This keeps the handlers
//! simple and testable while presenting the same register interface a real
//! BIOS would. The interrupt machinery (IVT + INT/IRET) is already in the
//! CPU core; the BIOS simply intercepts the vectors it owns before the IVT
//! is consulted.
//!
//! Video output is routed through the `Vga` device (which owns the text and
//! graphics framebuffers); keyboard and disk state live in the `Kbd`/`Ide`
//! devices. The BIOS keeps a small key queue for `INT 0x16`.

use std::collections::VecDeque;

use crate::cpu::{Cpu, Reg8, flags};
use crate::memory::Memory;

/// Text screen width (standard 80x25 VGA text mode).
pub const SCREEN_COLS: usize = 80;
/// Text screen height.
pub const SCREEN_ROWS: usize = 25;

/// A minimal BIOS with emulated video, keyboard and disk state.
pub struct Bios {
    // Keyboard
    pub key_buffer: VecDeque<(u8, u8)>, // (ascii, scancode)
}

impl Bios {
    pub fn new() -> Self {
        Bios {
            key_buffer: VecDeque::new(),
        }
    }

    /// Queue a keypress (ascii, scancode) into the keyboard buffer.
    pub fn push_key(&mut self, ascii: u8, scancode: u8) {
        self.key_buffer.push_back((ascii, scancode));
    }

    /// Handle an interrupt. Returns `true` if `vector` was a BIOS service
    /// (and was handled), `false` if the caller should vector through the IVT.
    pub fn handle(&mut self, cpu: &mut Cpu, vector: u8) -> bool {
        match vector {
            0x10 => { self.video(cpu); true }
            0x15 => { self.memory_map(cpu); true }
            0x16 => { self.keyboard(cpu); true }
            0x13 => { self.disk(cpu); true }
            _ => false,
        }
    }

    // ---- INT 0x10: video services (AH = function) ----

    fn video(&mut self, cpu: &mut Cpu) {
        match cpu.reg8(Reg8::Ah) {
            0x00 => { // set video mode: AL = mode
                cpu.vga.set_mode(cpu.reg8(Reg8::Al));
            }
            0x02 => { // set cursor position: DH = row, DL = col
                cpu.vga.cursor_row = cpu.reg8(Reg8::Dh);
                cpu.vga.cursor_col = cpu.reg8(Reg8::Dl);
            }
            0x03 => { // get cursor position -> DH = row, DL = col
                cpu.set_reg8(Reg8::Dh, cpu.vga.cursor_row);
                cpu.set_reg8(Reg8::Dl, cpu.vga.cursor_col);
            }
            0x0E => { // teletype output: AL = char
                let ch = cpu.reg8(Reg8::Al);
                self.put_char(cpu, ch);
            }
            0x13 => { // write string: ES:BP = string, CX = length, DH/DL = row/col
                let row = cpu.reg8(Reg8::Dh) as usize;
                let col = cpu.reg8(Reg8::Dl) as usize;
                let len = cpu.cx as usize;
                let base = Memory::phys(cpu.es, cpu.bp);
                for i in 0..len {
                    let ch = cpu.mem.read_u8(base + i);
                    self.put_char_at(cpu, row, col + i, ch);
                }
            }
            _ => {}
        }
    }

    fn put_char(&mut self, cpu: &mut Cpu, ch: u8) {
        let row = cpu.vga.cursor_row as usize;
        let col = cpu.vga.cursor_col as usize;
        self.put_char_at(cpu, row, col, ch);
        cpu.vga.cursor_col += 1;
        if cpu.vga.cursor_col as usize >= SCREEN_COLS {
            cpu.vga.cursor_col = 0;
            cpu.vga.cursor_row += 1;
            if cpu.vga.cursor_row as usize >= SCREEN_ROWS {
                self.scroll(cpu);
                cpu.vga.cursor_row = (SCREEN_ROWS - 1) as u8;
            }
        }
    }

    fn put_char_at(&mut self, cpu: &mut Cpu, row: usize, col: usize, ch: u8) {
        if row < SCREEN_ROWS && col < SCREEN_COLS {
            let idx = row * SCREEN_COLS + col;
            cpu.mem.vga_text[idx] = (0x07 << 8) | ch as u16;
        }
    }

    /// Scroll the memory-mapped text window up one line.
    fn scroll(&mut self, cpu: &mut Cpu) {
        cpu.mem.vga_text.copy_within(SCREEN_COLS.., 0);
        let last = SCREEN_COLS * (SCREEN_ROWS - 1);
        for i in last..SCREEN_COLS * SCREEN_ROWS {
            cpu.mem.vga_text[i] = 0x0720;
        }
    }

    // ---- INT 0x15: memory map services (AH = function) ----
    // E820 (AH=0xE820), E801 (AH=0xE801) and 0x88 report the physical RAM
    // layout. Linux queries these very early in boot to learn how much memory
    // it can use. The backing store is 128 MiB, so extended memory (above the
    // 1 MiB mark) is 127 MiB.

    fn memory_map(&mut self, cpu: &mut Cpu) {
        match cpu.reg8(Reg8::Ah) {
            0xE8 => { // E820 (AL=0x20) or E801 (AL=0x01)
                if cpu.reg8(Reg8::Al) == 0x20 {
                    self.e820(cpu);
                } else {
                    self.e801(cpu);
                }
            }
            0x88 => { // extended memory size in KB
                self.mem88(cpu);
            }
            _ => {
                cpu.set_flag(flags::CF, true); // unsupported function
            }
        }
    }

    fn e820(&mut self, cpu: &mut Cpu) {
        // E820 entry layout (20 bytes): base (u64), length (u64), type (u32).
        // Types: 1 = usable, 2 = reserved, 3 = ACPI reclaimable, 4 = ACPI NVS.
        const ENTRIES: [(u64, u64, u32); 5] = [
            (0x00000000, 0x0009FC00, 1), // conventional memory (640K - 1K)
            (0x0009FC00, 0x00000400, 2), // EBDA
            (0x000A0000, 0x00060000, 2), // VGA / ROM area
            (0x00100000, 0x07F00000, 1), // extended memory up to 128 MiB
            (0x08000000, 0x00000000, 2), // end marker (reserved)
        ];
        // E820 protocol: EAX=0x0000E820, EDX='SMAP' (0x534D4150),
        // EBX=continuation (0 first), ECX=buffer size, ES:DI=buffer.
        // Returns EBX=next continuation (0=done).
        if cpu.eax != 0x0000_E820 || cpu.edx != 0x534D_4150 {
            cpu.set_flag(flags::CF, true); // unsupported
            return;
        }
        let cont = cpu.ebx as usize;
        if cont >= ENTRIES.len() {
            cpu.set_flag(flags::CF, true); // out of entries
            return;
        }
        let (base, len, typ) = ENTRIES[cont];
        let buf = Memory::phys(cpu.es, cpu.di);
        cpu.mem.write_u64(buf, base);
        cpu.mem.write_u64(buf + 8, len);
        cpu.mem.write_u32(buf + 16, typ);
        // Next continuation (0 means done).
        cpu.ebx = if cont + 1 < ENTRIES.len() { (cont + 1) as u32 } else { 0 };
        cpu.eax = 0x0000_E820;
        cpu.ecx = 20; // bytes written
        cpu.set_flag(flags::CF, false);
    }

    fn e801(&mut self, cpu: &mut Cpu) {
        // Extended memory below 16 MiB in KB (15 MiB = 15360 KB), and above
        // 16 MiB in 64 KiB units (128 MiB total = 112 * 64 KiB units above 16 MiB).
        let ext_kb: u16 = 15 * 1024;
        cpu.ax = ext_kb;
        cpu.cx = ext_kb;
        // (128 MiB - 16 MiB) / 64 KiB = 112 MiB / 64 KiB = 1792 units.
        cpu.bx = 1792;
        cpu.dx = 1792;
    }

    fn mem88(&mut self, cpu: &mut Cpu) {
        // Extended memory in KB above the 1 MiB mark. The 0x88 return is a
        // 16-bit value, so it saturates at 65535 KB (~64 MiB); real BIOSes
        // report 0xFFFF when extended memory exceeds that. Linux uses E820 /
        // E801 for the full map.
        cpu.ax = 0xFFFF;
    }

    // ---- INT 0x16: keyboard services (AH = function) ----

    fn keyboard(&mut self, cpu: &mut Cpu) {
        match cpu.reg8(Reg8::Ah) {
            0x00 => { // read character: AL = ascii, AH = scancode
                if let Some((ascii, scancode)) = self.key_buffer.pop_front() {
                    cpu.set_reg8(Reg8::Al, ascii);
                    cpu.set_reg8(Reg8::Ah, scancode);
                } else {
                    cpu.set_reg8(Reg8::Al, 0);
                    cpu.set_reg8(Reg8::Ah, 0);
                }
            }
            0x01 => { // check buffer: ZF = 1 if empty
                cpu.set_flag(flags::ZF, self.key_buffer.is_empty());
            }
            _ => {}
        }
    }

    // ---- INT 0x13: disk services (AH = function) ----
    // CHS geometry: 2 heads, 18 sectors/track (standard 1.44 MB floppy).
    // LBA = ((cylinder * 2 + head) * 18 + (sector - 1)) * 512.
    // Backed by the IDE device's disk image.

    fn disk(&mut self, cpu: &mut Cpu) {
        match cpu.reg8(Reg8::Ah) {
            0x00 => { // reset disk: CF = 0 on success
                cpu.set_flag(flags::CF, false);
            }
            0x02 => { // read sectors: AL=count, CH=cyl, CL=sector, DH=head, ES:BX=buf
                let count = cpu.reg8(Reg8::Al) as usize;
                let cyl = cpu.reg8(Reg8::Ch) as usize;
                let sector = cpu.reg8(Reg8::Cl) as usize; // 1-based
                let head = cpu.reg8(Reg8::Dh) as usize;
                let lba = ((cyl * 2 + head) * 18 + (sector - 1)) * 512;
                let buf = Memory::phys(cpu.es, cpu.bx);
                if cpu.ide.present && lba + count * 512 <= cpu.ide.disk.len() {
                    for i in 0..count * 512 {
                        cpu.mem.write_u8(buf + i, cpu.ide.disk[lba + i]);
                    }
                    cpu.set_flag(flags::CF, false);
                } else {
                    cpu.set_flag(flags::CF, true); // error
                }
            }
            0x03 => { // write sectors: same addressing, source at ES:BX
                let count = cpu.reg8(Reg8::Al) as usize;
                let cyl = cpu.reg8(Reg8::Ch) as usize;
                let sector = cpu.reg8(Reg8::Cl) as usize;
                let head = cpu.reg8(Reg8::Dh) as usize;
                let lba = ((cyl * 2 + head) * 18 + (sector - 1)) * 512;
                let buf = Memory::phys(cpu.es, cpu.bx);
                if cpu.ide.present && lba + count * 512 <= cpu.ide.disk.len() {
                    for i in 0..count * 512 {
                        cpu.ide.disk[lba + i] = cpu.mem.read_u8(buf + i);
                    }
                    cpu.set_flag(flags::CF, false);
                } else {
                    cpu.set_flag(flags::CF, true);
                }
            }
            _ => {}
        }
    }
}

impl Default for Bios {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{Cpu, Reg8, flags};

    fn load(cpu: &mut Cpu, bytes: &[u8]) {
        cpu.mem.load(Memory::phys(cpu.cs, cpu.ip), bytes);
    }

    #[test]
    fn teletype_prints_to_screen() {
        let mut cpu = Cpu::new();
        // mov ah, 0x0E ; mov al, 'A' ; int 0x10 ; hlt
        load(&mut cpu, &[
            0xB4, 0x0E,         // mov ah, 0x0E
            0xB0, 0x41,         // mov al, 'A'
            0xCD, 0x10,         // int 0x10
            0xF4,               // hlt
        ]);
        cpu.run(16);
        // 'A' should be at the top-left of the screen (memory-mapped VGA window).
        assert_eq!(cpu.mem.vga_text[0] & 0xFF, b'A' as u16);
        // Cursor advanced to column 1.
        assert_eq!(cpu.vga.cursor_col, 1);
    }

    #[test]
    fn set_and_get_cursor() {
        let mut cpu = Cpu::new();
        // mov ah, 0x02 ; mov dh, 5 ; mov dl, 10 ; int 0x10
        // mov ah, 0x03 ; int 0x10 ; hlt
        load(&mut cpu, &[
            0xB4, 0x02, 0xB6, 0x05, 0xB2, 0x0A, 0xCD, 0x10,
            0xB4, 0x03, 0xCD, 0x10,
            0xF4,
        ]);
        cpu.run(32);
        assert_eq!(cpu.reg8(Reg8::Dh), 5);
        assert_eq!(cpu.reg8(Reg8::Dl), 10);
    }

    #[test]
    fn disk_read_sector() {
        let mut cpu = Cpu::new();
        cpu.es = 0;
        cpu.bx = 0x1000;
        // Disk: sector 2 (offset 512) holds a marker.
        let mut disk = vec![0u8; 512 * 3];
        disk[512..512 + 4].copy_from_slice(b"DATA");
        cpu.ide.load_disk(disk);
        // mov ah, 0x02 ; mov al, 1 ; mov ch, 0 ; mov cl, 2 ; mov dh, 0 ; int 0x13 ; hlt
        load(&mut cpu, &[
            0xB4, 0x02, 0xB0, 0x01, 0xB5, 0x00, 0xB1, 0x02, 0xB6, 0x00,
            0xCD, 0x13,
            0xF4,
        ]);
        cpu.run(32);
        assert!(!cpu.get_flag(flags::CF)); // success
        assert_eq!(&cpu.mem.data[0x1000..0x1004], b"DATA");
    }

    #[test]
    fn e820_reports_memory_map() {
        let mut cpu = Cpu::new();
        cpu.es = 0;
        cpu.di = 0x2000;
        // E820 call: eax=0x0000E820, edx='SMAP', ebx=0 (first entry),
        // ecx=24 (buffer size), es:di = buffer. The program sets eax/edx
        // via movs; ebx/ecx are pre-set here.
        cpu.edx = 0x534D_4150;
        cpu.ebx = 0;
        cpu.ecx = 24;
        // mov ah, 0xE8 ; mov al, 0x20 ; int 0x15 ; hlt
        load(&mut cpu, &[
            0xB4, 0xE8, 0xB0, 0x20, 0xCD, 0x15,
            0xF4,
        ]);
        cpu.run(16);
        assert!(!cpu.get_flag(flags::CF)); // success
        // First entry: base 0, length 0x9FC00, type 1 (usable).
        let base = cpu.mem.read_u64(0x2000);
        let len = cpu.mem.read_u64(0x2008);
        let typ = cpu.mem.read_u32(0x2010);
        assert_eq!(base, 0);
        assert_eq!(len, 0x9FC00);
        assert_eq!(typ, 1);
        // Continuation advanced to 1.
        assert_eq!(cpu.ebx, 1);
        // eax preserved as 0x0000E820.
        assert_eq!(cpu.eax, 0x0000_E820);
    }

    #[test]
    fn e820_iterates_entries() {
        let mut cpu = Cpu::new();
        cpu.es = 0;
        cpu.di = 0x2000;
        cpu.edx = 0x534D_4150;
        cpu.ebx = 3; // fourth entry (extended memory)
        cpu.ecx = 24;
        load(&mut cpu, &[
            0xB4, 0xE8, 0xB0, 0x20, 0xCD, 0x15,
            0xF4,
        ]);
        cpu.run(16);
        assert!(!cpu.get_flag(flags::CF));
        let base = cpu.mem.read_u64(0x2000);
        let len = cpu.mem.read_u64(0x2008);
        let typ = cpu.mem.read_u32(0x2010);
        assert_eq!(base, 0x0010_0000); // 1 MiB
        assert_eq!(len, 0x07F0_0000); // 127 MiB
        assert_eq!(typ, 1);
        // Continuation advanced to 4 (index 4 is the end marker).
        assert_eq!(cpu.ebx, 4);
    }

    #[test]
    fn e801_reports_extended_memory() {
        let mut cpu = Cpu::new();
        // mov ah, 0xE8 ; mov al, 0x01 ; int 0x15 ; hlt
        load(&mut cpu, &[
            0xB4, 0xE8, 0xB0, 0x01, 0xCD, 0x15,
            0xF4,
        ]);
        cpu.run(16);
        // 15 MiB of extended memory below 16 MiB = 15360 KB in both AX and CX.
        assert_eq!(cpu.ax, 15 * 1024);
        assert_eq!(cpu.cx, 15 * 1024);
        // Memory above 16 MiB: (128 MiB - 16 MiB) / 64 KiB = 1792 units.
        assert_eq!(cpu.bx, 1792);
        assert_eq!(cpu.dx, 1792);
    }

    #[test]
    fn int15_88_reports_extended_memory_kb() {
        let mut cpu = Cpu::new();
        // mov ah, 0x88 ; int 0x15 ; hlt
        load(&mut cpu, &[
            0xB4, 0x88, 0xCD, 0x15,
            0xF4,
        ]);
        cpu.run(16);
        // 0x88 saturates at 0xFFFF KB (16-bit return) for >64 MiB extended.
        assert_eq!(cpu.ax, 0xFFFF);
    }

    #[test]
    fn keyboard_read_queued_key() {
        let mut cpu = Cpu::new();
        cpu.bios.push_key(b'x', 0x2D);
        // mov ah, 0x00 ; int 0x16 ; hlt
        load(&mut cpu, &[
            0xB4, 0x00, 0xCD, 0x16,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.reg8(Reg8::Al), b'x');
        assert_eq!(cpu.reg8(Reg8::Ah), 0x2D);
    }
}
