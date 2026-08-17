//! The x86 CPU core: registers, flags, and the fetch-decode-execute loop.
//!
//! Supports both 16-bit real mode and 32-bit protected mode. Registers are
//! kept as individual fields plus a packed flags word. The 16-bit general
//! registers map to byte registers (AL/AH, BL/BH, ...), and the 32-bit
//! registers are kept in sync with their 16-bit halves.

use crate::memory::Memory;
use crate::modrm::{ModRm, Reg};
use crate::instructions::Inst;
use crate::instructions::protected_int;
use crate::protected::Descriptor;

/// 16-bit general-purpose register indices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reg16 {
    Ax = 0,
    Cx = 1,
    Dx = 2,
    Bx = 3,
    Sp = 4,
    Bp = 5,
    Si = 6,
    Di = 7,
}

/// 32-bit general-purpose register indices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reg32 {
    Eax = 0,
    Ecx = 1,
    Edx = 2,
    Ebx = 3,
    Esp = 4,
    Ebp = 5,
    Esi = 6,
    Edi = 7,
}

/// 8-bit register indices (AL, CL, DL, BL, AH, CH, DH, BH).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reg8 {
    Al = 0,
    Cl = 1,
    Dl = 2,
    Bl = 3,
    Ah = 4,
    Ch = 5,
    Dh = 6,
    Bh = 7,
}

/// Segment register indices.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SegReg {
    Es = 0,
    Cs = 1,
    Ss = 2,
    Ds = 3,
    Fs = 4,
    Gs = 5,
}

/// Flag bit positions in the FLAGS register.
pub mod flags {
    pub const CF: u16 = 0x0001;
    pub const PF: u16 = 0x0004;
    pub const AF: u16 = 0x0010;
    pub const ZF: u16 = 0x0040;
    pub const SF: u16 = 0x0080;
    pub const TF: u16 = 0x0100;
    pub const IF: u16 = 0x0200;
    pub const DF: u16 = 0x0400;
    pub const OF: u16 = 0x0800;
    pub const IOPL: u16 = 0x3000;
    pub const NT: u16 = 0x4000;
}

pub struct Cpu {
    // 16-bit general registers (low halves of the 32-bit registers).
    pub ax: u16,
    pub cx: u16,
    pub dx: u16,
    pub bx: u16,
    pub sp: u16,
    pub bp: u16,
    pub si: u16,
    pub di: u16,
    // 32-bit general registers.
    pub eax: u32,
    pub ecx: u32,
    pub edx: u32,
    pub ebx: u32,
    pub esp: u32,
    pub ebp: u32,
    pub esi: u32,
    pub edi: u32,
    // Segment registers.
    pub es: u16,
    pub cs: u16,
    pub ss: u16,
    pub ds: u16,
    pub fs: u16,
    pub gs: u16,
    // Instruction pointer (offset within CS).
    pub ip: u16,
    pub eip: u32,
    // FLAGS register.
    pub flags: u16,
    // Memory.
    pub mem: Memory,
    // BIOS (emulated INT 0x10/0x16/0x13 services).
    pub bios: crate::bios::Bios,
    // Number of instructions executed (for tests/diagnostics).
    pub instructions_executed: u64,
    // Time-stamp counter (RDTSC source), incremented each step.
    pub tsc: u64,
    // Set true to stop the run loop (HLT, or a test sentinel).
    pub halted: bool,
    /// Set true when the CPU triple-faults (an exception fired with no IDT
    /// installed, or an exception fired while handling another). A real CPU
    /// resets here; we halt with this flag set.
    pub triple_fault: bool,

    // ---- Prefix / size state for the current instruction ----
    /// True when the 0x66 operand-size override is active (32-bit operands).
    pub opsize: bool,
    /// True when the 0x67 address-size override is active (32-bit addressing).
    pub addrsize: bool,
    /// Segment override for the current instruction, if any.
    pub seg_override: Option<SegReg>,

    // ---- Protected-mode state ----
    /// True when protected mode is enabled (CR0.PE).
    pub pe: bool,
    /// GDT base and limit.
    pub gdt_base: u32,
    pub gdt_limit: u16,
    /// IDT base and limit.
    pub idt_base: u32,
    pub idt_limit: u16,
    /// Cached descriptors for ES, CS, SS, DS, FS, GS.
    pub seg_desc: [Descriptor; 6],

    // ---- Paging state ----
    /// Control registers. CR0 bit 31 = PG (paging enabled), bit 0 = PE
    /// (protected mode enabled). CR3 = page-directory base register.
    pub cr0: u32,
    pub cr2: u32,
    pub cr3: u32,
    pub cr4: u32,

    // ---- Devices ----
    /// 8254 Programmable Interval Timer.
    pub pit: crate::pit::Pit,
    /// 8259 Programmable Interrupt Controller (master + slave).
    pub pic: crate::pic::Pic,
    /// True when a hardware interrupt is currently being serviced (so we
    /// don't re-enter on the same instruction).
    pub servicing_irq: bool,

    // ---- Exceptions ----
    /// A pending exception raised during decode/execute, dispatched at the
    /// top of the next `step`. `(vector, error_code)` — the error code is
    /// `None` for exceptions that don't push one (e.g. #DE, #UD).
    pub pending_exception: Option<(u8, Option<u32>)>,

    // ---- Devices part 2 ----
    /// VGA display (text + graphics framebuffer).
    pub vga: crate::vga::Vga,
    /// 8042 keyboard controller.
    pub kbd: crate::kbd::Kbd,
    /// 8237 DMA controller.
    pub dma: crate::dma::Dma,
    /// IDE/ATA disk controller.
    pub ide: crate::ide::Ide,
}

impl Cpu {
    pub fn new() -> Self {
        Cpu {
            ax: 0, cx: 0, dx: 0, bx: 0,
            sp: 0, bp: 0, si: 0, di: 0,
            eax: 0, ecx: 0, edx: 0, ebx: 0,
            esp: 0, ebp: 0, esi: 0, edi: 0,
            es: 0, cs: 0, ss: 0, ds: 0, fs: 0, gs: 0,
            ip: 0, eip: 0,
            flags: 0x0002, // bit 1 is always 1
            mem: Memory::new(),
            bios: crate::bios::Bios::new(),
            instructions_executed: 0,
            halted: false,
            triple_fault: false,
            tsc: 0,
            opsize: false,
            addrsize: false,
            seg_override: None,
            pe: false,
            gdt_base: 0,
            gdt_limit: 0,
            idt_base: 0,
            idt_limit: 0,
            seg_desc: [Descriptor::default(); 6],
            cr0: 0,
            cr2: 0,
            cr3: 0,
            cr4: 0,
            pit: crate::pit::Pit::new(),
            pic: crate::pic::Pic::new(),
            servicing_irq: false,
            vga: crate::vga::Vga::new(),
            kbd: crate::kbd::Kbd::new(),
            dma: crate::dma::Dma::new(),
            ide: crate::ide::Ide::new(),
            pending_exception: None,
        }
    }

    /// Physical address of the current instruction stream.
    #[inline]
    pub fn phys_ip(&mut self) -> usize {
        let linear = if self.pe {
            self.seg_desc[SegReg::Cs as usize].base.wrapping_add(self.eip)
        } else {
            ((self.cs as u32) << 4) + self.ip as u32
        };
        self.apply_paging(linear)
    }

    // ---- 16-bit register access ----

    pub fn reg16(&self, r: Reg16) -> u16 {
        match r {
            Reg16::Ax => self.ax, Reg16::Cx => self.cx, Reg16::Dx => self.dx,
            Reg16::Bx => self.bx, Reg16::Sp => self.sp, Reg16::Bp => self.bp,
            Reg16::Si => self.si, Reg16::Di => self.di,
        }
    }

    pub fn set_reg16(&mut self, r: Reg16, v: u16) {
        match r {
            Reg16::Ax => { self.ax = v; self.eax = (self.eax & 0xFFFF0000) | v as u32; }
            Reg16::Cx => { self.cx = v; self.ecx = (self.ecx & 0xFFFF0000) | v as u32; }
            Reg16::Dx => { self.dx = v; self.edx = (self.edx & 0xFFFF0000) | v as u32; }
            Reg16::Bx => { self.bx = v; self.ebx = (self.ebx & 0xFFFF0000) | v as u32; }
            Reg16::Sp => { self.sp = v; self.esp = (self.esp & 0xFFFF0000) | v as u32; }
            Reg16::Bp => { self.bp = v; self.ebp = (self.ebp & 0xFFFF0000) | v as u32; }
            Reg16::Si => { self.si = v; self.esi = (self.esi & 0xFFFF0000) | v as u32; }
            Reg16::Di => { self.di = v; self.edi = (self.edi & 0xFFFF0000) | v as u32; }
        }
    }

    // ---- 32-bit register access ----

    pub fn reg32(&self, r: Reg32) -> u32 {
        match r {
            Reg32::Eax => self.eax, Reg32::Ecx => self.ecx, Reg32::Edx => self.edx,
            Reg32::Ebx => self.ebx, Reg32::Esp => self.esp, Reg32::Ebp => self.ebp,
            Reg32::Esi => self.esi, Reg32::Edi => self.edi,
        }
    }

    pub fn set_reg32(&mut self, r: Reg32, v: u32) {
        match r {
            Reg32::Eax => { self.eax = v; self.ax = v as u16; }
            Reg32::Ecx => { self.ecx = v; self.cx = v as u16; }
            Reg32::Edx => { self.edx = v; self.dx = v as u16; }
            Reg32::Ebx => { self.ebx = v; self.bx = v as u16; }
            Reg32::Esp => { self.esp = v; self.sp = v as u16; }
            Reg32::Ebp => { self.ebp = v; self.bp = v as u16; }
            Reg32::Esi => { self.esi = v; self.si = v as u16; }
            Reg32::Edi => { self.edi = v; self.di = v as u16; }
        }
    }

    // ---- 8-bit register access ----

    pub fn reg8(&self, r: Reg8) -> u8 {
        let w = match r {
            Reg8::Al | Reg8::Ah => self.ax,
            Reg8::Cl | Reg8::Ch => self.cx,
            Reg8::Dl | Reg8::Dh => self.dx,
            Reg8::Bl | Reg8::Bh => self.bx,
        };
        if (r as u8) < 4 { (w & 0xFF) as u8 } else { (w >> 8) as u8 }
    }

    pub fn set_reg8(&mut self, r: Reg8, v: u8) {
        let lo = (r as u8) < 4;
        let idx = (r as usize) & 3;
        let cur = match idx { 0 => self.ax, 1 => self.cx, 2 => self.dx, _ => self.bx };
        let new = if lo { (cur & 0xFF00) | v as u16 } else { (cur & 0x00FF) | ((v as u16) << 8) };
        match idx { 0 => self.set_reg16(Reg16::Ax, new), 1 => self.set_reg16(Reg16::Cx, new), 2 => self.set_reg16(Reg16::Dx, new), _ => self.set_reg16(Reg16::Bx, new) };
    }

    // ---- Segment register access ----

    pub fn seg(&self, s: SegReg) -> u16 {
        match s {
            SegReg::Es => self.es, SegReg::Cs => self.cs,
            SegReg::Ss => self.ss, SegReg::Ds => self.ds,
            SegReg::Fs => self.fs, SegReg::Gs => self.gs,
        }
    }

    pub fn set_seg(&mut self, s: SegReg, v: u16) {
        match s {
            SegReg::Es => self.es = v, SegReg::Cs => self.cs = v,
            SegReg::Ss => self.ss = v, SegReg::Ds => self.ds = v,
            SegReg::Fs => self.fs = v, SegReg::Gs => self.gs = v,
        }
    }

    /// Load a segment register with a selector. In protected mode this
    /// resolves the descriptor from the GDT and caches it; in real mode it
    /// just stores the value.
    pub fn load_seg(&mut self, s: SegReg, selector: u16) {
        if self.pe {
            let idx = (selector >> 3) & 0x1FFF;
            let desc = crate::protected::read_descriptor(&self.mem, self.gdt_base, idx);
            self.seg_desc[s as usize] = desc;
        }
        self.set_seg(s, selector);
    }

    /// Translate a logical address through a segment to a physical address.
    /// Records a #PF (page fault) in `pending_exception` if paging is enabled
    /// and the page is not present.
    pub fn translate(&mut self, s: SegReg, offset: u32) -> usize {
        let linear = if self.pe {
            self.seg_desc[s as usize].base.wrapping_add(offset)
        } else {
            ((self.seg(s) as u32) << 4) + offset
        };
        self.apply_paging(linear)
    }

    /// Apply paging to a linear address if CR0.PG is set. If the page is not
    /// present, raise a #PF (vector 0x0E) with the faulting linear address in
    /// CR2 and an error code, and return 0.
    #[inline]
    pub fn apply_paging(&mut self, linear: u32) -> usize {
        if self.cr0 & 0x8000_0000 != 0 {
            match crate::paging::translate(&self.mem, self.cr3, linear) {
                Some(phys) => phys,
                None => {
                    self.cr2 = linear;
                    self.pending_exception = Some((0x0E, Some(0)));
                    0
                }
            }
        } else {
            Memory::phys32(linear)
        }
    }

    // ---- Flag helpers ----

    pub fn set_flag(&mut self, f: u16, on: bool) {
        if on { self.flags |= f; } else { self.flags &= !f; }
    }
    pub fn get_flag(&self, f: u16) -> bool { (self.flags & f) != 0 }

    /// Read the time-stamp counter (used by RDTSC).
    pub fn rdtsc(&self) -> u64 { self.tsc }

    // ---- Instruction stream fetch ----

    #[inline]
    pub fn fetch_u8(&mut self) -> u8 {
        let addr = self.phys_ip();
        let b = self.mem.read_u8(addr);
        if self.pe {
            self.eip = self.eip.wrapping_add(1);
        } else {
            self.ip = self.ip.wrapping_add(1);
        }
        b
    }

    #[inline]
    pub fn fetch_u16(&mut self) -> u16 {
        let lo = self.fetch_u8() as u16;
        let hi = self.fetch_u8() as u16;
        lo | (hi << 8)
    }

    #[inline]
    pub fn fetch_u32(&mut self) -> u32 {
        let b0 = self.fetch_u8() as u32;
        let b1 = self.fetch_u8() as u32;
        let b2 = self.fetch_u8() as u32;
        let b3 = self.fetch_u8() as u32;
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    }

    /// Read the ModR/M byte and decode it into a `ModRm` descriptor, fetching
    /// any SIB byte and displacement bytes it implies (based on `addrsize`).
    pub fn fetch_modrm(&mut self) -> ModRm {
        let byte = self.fetch_u8();
        let mut modrm = ModRm::from_byte(byte);
        if self.addrsize {
            // 32-bit addressing.
            if modrm.mod_field != 3 && modrm.rm == 4 {
                modrm.sib = Some(self.fetch_u8());
            }
            match modrm.mod_field {
                0 => {
                    // mod=00, rm=101 -> disp32; SIB base=101 -> disp32.
                    let sib_disp32 = modrm.sib.map(|s| s & 7 == 5).unwrap_or(false);
                    if modrm.rm == 5 || sib_disp32 {
                        modrm.disp32 = Some(self.fetch_u32());
                    }
                }
                1 => {
                    let disp = self.fetch_u8() as i8 as i32 as u32;
                    modrm.disp32 = Some(disp);
                }
                2 => {
                    modrm.disp32 = Some(self.fetch_u32());
                }
                _ => {}
            }
        } else {
            // 16-bit addressing (existing behaviour).
            match modrm.mod_field {
                0 => {
                    if modrm.rm == 6 {
                        let disp = self.fetch_u16();
                        modrm.disp16 = Some(disp);
                    }
                }
                1 => {
                    let disp = self.fetch_u8() as i8 as i16 as u16;
                    modrm.disp8 = Some(disp);
                }
                2 => {
                    let disp = self.fetch_u16();
                    modrm.disp16 = Some(disp);
                }
                _ => {}
            }
        }
        modrm
    }

    /// The segment register used for a memory operand, honouring any override.
    fn operand_seg(&self, default: SegReg) -> SegReg {
        self.seg_override.unwrap_or(default)
    }

    /// Compute the physical address of a 16-bit-addressed memory operand.
    pub fn modrm_addr(&mut self, m: &ModRm) -> usize {
        let (base, default_seg) = match m.rm {
            0 => (self.bx + self.si, SegReg::Ds),
            1 => (self.bx + self.di, SegReg::Ds),
            2 => (self.bp + self.si, SegReg::Ss),
            3 => (self.bp + self.di, SegReg::Ss),
            4 => (self.si, SegReg::Ds),
            5 => (self.di, SegReg::Ds),
            6 => (self.bp, SegReg::Ss),
            _ => (self.bx, SegReg::Ds),
        };
        let mut ea = base as u32;
        if let Some(d8) = m.disp8 { ea = ea.wrapping_add(d8 as u32); }
        if let Some(d16) = m.disp16 { ea = ea.wrapping_add(d16 as u32); }
        self.translate(self.operand_seg(default_seg), ea)
    }

    /// Compute the physical address of a 32-bit-addressed memory operand.
    pub fn modrm_addr32(&mut self, m: &ModRm) -> usize {
        let mut ea: u32 = 0;
        let mut default_seg = SegReg::Ds;
        if let Some(sib) = m.sib {
            let scale = 1u32 << ((sib >> 6) & 3);
            let index = (sib >> 3) & 7;
            let base = sib & 7;
            if index != 4 {
                let idx_reg = Reg::reg32(index);
                ea = ea.wrapping_add(self.reg32(idx_reg).wrapping_mul(scale));
            }
            if !(m.mod_field == 0 && base == 5) {
                let base_reg = Reg::reg32(base);
                ea = ea.wrapping_add(self.reg32(base_reg));
                if base == 4 || base == 5 { default_seg = SegReg::Ss; }
            }
        } else {
            // mod=00, rm=101 means disp32 with NO base register (only EBP/ESP
            // use SS as the default segment, and only when they are a base).
            if !(m.mod_field == 0 && m.rm == 5) {
                let base_reg = Reg::reg32(m.rm);
                ea = ea.wrapping_add(self.reg32(base_reg));
                if m.rm == 4 || m.rm == 5 { default_seg = SegReg::Ss; }
            }
        }
        if let Some(d32) = m.disp32 { ea = ea.wrapping_add(d32); }
        self.translate(self.operand_seg(default_seg), ea)
    }

    /// Read an 8-bit ModR/M operand.
    pub fn read_rm8(&mut self, m: &ModRm) -> u8 {
        if m.is_reg() {
            self.reg8(Reg::reg8(m.rm))
        } else if self.addrsize {
            let addr = self.modrm_addr32(m);
            self.mem.read_u8(addr)
        } else {
            let addr = self.modrm_addr(m);
            self.mem.read_u8(addr)
        }
    }

    /// Write an 8-bit ModR/M operand.
    pub fn write_rm8(&mut self, m: &ModRm, val: u8) {
        if m.is_reg() {
            self.set_reg8(Reg::reg8(m.rm), val);
        } else if self.addrsize {
            let addr = self.modrm_addr32(m);
            self.mem.write_u8(addr, val);
        } else {
            let addr = self.modrm_addr(m);
            self.mem.write_u8(addr, val);
        }
    }

    /// Read a 16-bit ModR/M operand.
    pub fn read_rm16(&mut self, m: &ModRm) -> u16 {
        if m.is_reg() {
            self.reg16(Reg::reg16(m.rm))
        } else if self.addrsize {
            let addr = self.modrm_addr32(m);
            self.mem.read_u16(addr)
        } else {
            let addr = self.modrm_addr(m);
            self.mem.read_u16(addr)
        }
    }

    /// Write a 16-bit ModR/M operand.
    pub fn write_rm16(&mut self, m: &ModRm, val: u16) {
        if m.is_reg() {
            self.set_reg16(Reg::reg16(m.rm), val);
        } else if self.addrsize {
            let addr = self.modrm_addr32(m);
            self.mem.write_u16(addr, val);
        } else {
            let addr = self.modrm_addr(m);
            self.mem.write_u16(addr, val);
        }
    }

    /// Read a 32-bit ModR/M operand.
    pub fn read_rm32(&mut self, m: &ModRm) -> u32 {
        if m.is_reg() {
            self.reg32(Reg::reg32(m.rm))
        } else if self.addrsize {
            let addr = self.modrm_addr32(m);
            self.mem.read_u32(addr)
        } else {
            let addr = self.modrm_addr(m);
            self.mem.read_u32(addr)
        }
    }

    /// Write a 32-bit ModR/M operand.
    pub fn write_rm32(&mut self, m: &ModRm, val: u32) {
        if m.is_reg() {
            self.set_reg32(Reg::reg32(m.rm), val);
        } else if self.addrsize {
            let addr = self.modrm_addr32(m);
            self.mem.write_u32(addr, val);
        } else {
            let addr = self.modrm_addr(m);
            self.mem.write_u32(addr, val);
        }
    }

    // ---- Stack ----

    pub fn push16(&mut self, val: u16) {
        if self.pe {
            self.esp = self.esp.wrapping_sub(2);
            let addr = self.translate(SegReg::Ss, self.esp);
            self.mem.write_u16(addr, val);
        } else {
            self.sp = self.sp.wrapping_sub(2);
            self.mem.write_u16(Memory::phys(self.ss, self.sp), val);
        }
    }

    pub fn pop16(&mut self) -> u16 {
        if self.pe {
            let addr = self.translate(SegReg::Ss, self.esp);
            let v = self.mem.read_u16(addr);
            self.esp = self.esp.wrapping_add(2);
            v
        } else {
            let v = self.mem.read_u16(Memory::phys(self.ss, self.sp));
            self.sp = self.sp.wrapping_add(2);
            v
        }
    }

    pub fn push32(&mut self, val: u32) {
        if self.pe {
            self.esp = self.esp.wrapping_sub(4);
            let addr = self.translate(SegReg::Ss, self.esp);
            self.mem.write_u32(addr, val);
        } else {
            self.sp = self.sp.wrapping_sub(4);
            self.mem.write_u32(Memory::phys(self.ss, self.sp), val);
        }
    }

    pub fn pop32(&mut self) -> u32 {
        if self.pe {
            let addr = self.translate(SegReg::Ss, self.esp);
            let v = self.mem.read_u32(addr);
            self.esp = self.esp.wrapping_add(4);
            v
        } else {
            let v = self.mem.read_u32(Memory::phys(self.ss, self.sp));
            self.sp = self.sp.wrapping_add(4);
            v
        }
    }

    // ---- Port I/O (devices) ----

    /// Read a byte from an I/O port.
    pub fn port_in(&mut self, port: u8) -> u8 {
        match port {
            0x20 => self.pic.read_command(0x20),
            0x21 => self.pic.read_data(0x21),
            0xA0 => self.pic.read_command(0xA0),
            0xA1 => self.pic.read_data(0xA1),
            0x40 | 0x41 | 0x42 => self.pit.read_data(port - 0x40),
            // 8042 keyboard controller.
            0x60 => self.kbd.read_data(),
            0x64 => self.kbd.read_status(),
            // 8237 DMA status.
            0x08 => self.dma.read_status(),
            _ => 0xFF,
        }
    }

    /// Read a 16-bit value from an I/O port (two byte reads).
    pub fn port_in16(&mut self, port: u16) -> u16 {
        match port {
            // IDE/ATA: data register (0x1F0) is 16-bit; the command block
            // registers (0x1F1-0x1F7) are byte registers read through the
            // 16-bit port path.
            0x1F0 => self.ide.read_data(),
            0x1F1 => self.ide.error as u16,
            0x1F7 => self.ide.read_status() as u16,
            0x3F6 => self.ide.read_status() as u16,
            _ => {
                let lo = self.port_in(port as u8);
                let hi = self.port_in((port + 1) as u8);
                lo as u16 | ((hi as u16) << 8)
            }
        }
    }

    /// Write a byte to an I/O port.
    pub fn port_out(&mut self, port: u8, val: u8) {
        match port {
            0x20 => self.pic.write_command(0x20, val),
            0x21 => self.pic.write_data(0x21, val),
            0xA0 => self.pic.write_command(0xA0, val),
            0xA1 => self.pic.write_data(0xA1, val),
            0x40 | 0x41 | 0x42 => self.pit.write_data(val),
            0x43 => self.pit.write_control(val),
            // 8042 keyboard controller.
            0x60 => self.kbd.write_data(val),
            0x64 => self.kbd.write_command(val),
            // 8237 DMA.
            0x08 => self.dma.write_command(0x08, val),
            0x0A => self.dma.write_command(0x0A, val),
            0x81..=0x8F => self.dma.write_page(port, val),
            _ => {}
        }
    }

    /// Write a 16-bit value to an I/O port (two byte writes).
    pub fn port_out16(&mut self, port: u16, val: u16) {
        match port {
            // IDE/ATA command block (ports 0x1F0-0x1F7 exceed u8 range, so
            // they are reached through the 16-bit port path).
            0x1F0 => self.ide.write_data(val),
            0x1F1 => self.ide.write_reg(0x1F1, (val & 0xFF) as u8),
            0x1F2 => self.ide.write_reg(0x1F2, (val & 0xFF) as u8),
            0x1F3 => self.ide.write_reg(0x1F3, (val & 0xFF) as u8),
            0x1F4 => self.ide.write_reg(0x1F4, (val & 0xFF) as u8),
            0x1F5 => self.ide.write_reg(0x1F5, (val & 0xFF) as u8),
            0x1F6 => self.ide.write_reg(0x1F6, (val & 0xFF) as u8),
            0x1F7 => self.ide.write_command((val & 0xFF) as u8),
            0x3F6 => {} // device control: ignored
            _ => {
                self.port_out(port as u8, (val & 0xFF) as u8);
                self.port_out((port + 1) as u8, (val >> 8) as u8);
            }
        }
    }

    /// Deliver a pending hardware interrupt, if any. Returns true if an
    /// interrupt was dispatched.
    pub fn deliver_hardware_interrupt(&mut self) -> bool {
        if self.servicing_irq {
            return false;
        }
        // Tick the PIT (channel 0 drives IRQ0).
        self.pit.tick(1);
        if self.pit.irq0 {
            self.pit.irq0 = false;
            self.pic.raise_irq(0);
        }
        // Keyboard (8042) drives IRQ1.
        if self.kbd.irq1 {
            self.kbd.irq1 = false;
            self.pic.raise_irq(1);
        }
        // IDE drives IRQ14 when a transfer completes.
        if self.ide.busy {
            self.ide.busy = false;
            self.pic.raise_irq(14);
        }
        if let Some(vector) = self.pic.acknowledge() {
            self.servicing_irq = true;
            if self.pe {
                protected_int(self, vector);
            } else {
                // Real-mode interrupt through the IVT.
                let ip = self.ip;
                let cs = self.cs;
                let flags = self.flags;
                self.push16(flags);
                self.push16(cs);
                self.push16(ip);
                self.set_flag(crate::cpu::flags::IF, false);
                self.set_flag(crate::cpu::flags::TF, false);
                let entry = (vector as usize) * 4;
                let off = self.mem.read_u16(entry);
                let seg = self.mem.read_u16(entry + 2);
                self.cs = seg;
                self.ip = off;
            }
            true
        } else {
            false
        }
    }

    // ---- The main loop ----

    /// Decode and execute a single instruction. Returns the decoded instruction
    /// for diagnostics/tests.
    pub fn step(&mut self) -> Inst {
        // Deliver a pending hardware interrupt before the next instruction.
        self.deliver_hardware_interrupt();
        // Dispatch any exception raised by the previous instruction (e.g. a
        // page fault recorded during address translation).
        if let Some((vector, error_code)) = self.pending_exception.take() {
            self.dispatch_exception(vector, error_code);
        }
        let inst = crate::instructions::decode(self);
        crate::instructions::execute(self, &inst);
        self.instructions_executed += 1;
        self.tsc = self.tsc.wrapping_add(1);
        inst
    }

    /// Dispatch an exception through the IDT (protected mode) or IVT
    /// (real mode), pushing an error code first if the exception has one.
    ///
    /// If the IDT/IVT is not set up for this vector (e.g. an exception fires
    /// before the kernel installs its IDT), the CPU triple-faults: it halts
    /// with `triple_fault = true` instead of dispatching to a garbage entry
    /// and looping forever (as a real CPU would reset).
    pub fn dispatch_exception(&mut self, vector: u8, error_code: Option<u32>) {
        // Check the IDT/IVT covers this vector before dispatching.
        if self.pe {
            let entry = (vector as u32) * 8;
            if (entry + 7) as u16 > self.idt_limit {
                self.triple_fault = true;
                self.halted = true;
                return;
            }
        } else {
            let entry = (vector as usize) * 4;
            if entry + 3 > 0x3FF {
                self.triple_fault = true;
                self.halted = true;
                return;
            }
        }
        if let Some(code) = error_code {
            if self.pe {
                self.push32(code);
            } else {
                self.push16(code as u16);
            }
        }
        if self.pe {
            crate::instructions::protected_int(self, vector);
        } else {
            // Real-mode exception through the IVT.
            let ip = self.ip;
            let cs = self.cs;
            let flags = self.flags;
            self.push16(flags);
            self.push16(cs);
            self.push16(ip);
            self.set_flag(crate::cpu::flags::IF, false);
            self.set_flag(crate::cpu::flags::TF, false);
            let entry = (vector as usize) * 4;
            let off = self.mem.read_u16(entry);
            let seg = self.mem.read_u16(entry + 2);
            self.cs = seg;
            self.ip = off;
        }
    }

    /// Run until halted or `max` instructions have executed.
    pub fn run(&mut self, max: u64) -> u64 {
        let mut n = 0u64;
        while !self.halted && n < max {
            self.step();
            n += 1;
        }
        n
    }
}

impl Default for Cpu {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(cpu: &mut Cpu, bytes: &[u8]) {
        cpu.mem.load(Memory::phys(cpu.cs, cpu.ip), bytes);
    }

    #[test]
    fn reg8_byte_access() {
        let mut cpu = Cpu::new();
        cpu.set_reg16(Reg16::Ax, 0x1234);
        assert_eq!(cpu.reg8(Reg8::Al), 0x34);
        assert_eq!(cpu.reg8(Reg8::Ah), 0x12);
        cpu.set_reg8(Reg8::Al, 0xAB);
        assert_eq!(cpu.ax, 0x12AB);
        cpu.set_reg8(Reg8::Ah, 0xCD);
        assert_eq!(cpu.ax, 0xCDAB);
    }

    #[test]
    fn stack_push_pop() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.sp = 0x0100;
        cpu.push16(0xCAFE);
        assert_eq!(cpu.sp, 0x00FE);
        assert_eq!(cpu.pop16(), 0xCAFE);
        assert_eq!(cpu.sp, 0x0100);
    }

    #[test]
    fn reg32_syncs_with_16bit() {
        let mut cpu = Cpu::new();
        cpu.set_reg32(Reg32::Eax, 0x12345678);
        assert_eq!(cpu.ax, 0x5678);
        assert_eq!(cpu.reg16(Reg16::Ax), 0x5678);
        cpu.set_reg16(Reg16::Ax, 0xABCD);
        assert_eq!(cpu.eax, 0x1234ABCD);
    }

    #[test]
    fn hardware_interrupt_pit_to_pic_to_cpu() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.sp = 0x0100;
        // Program the PIT: channel 0, word access, mode 3, count = 1.
        cpu.port_out(0x43, 0x36);
        cpu.port_out(0x40, 1);
        cpu.port_out(0x40, 0);
        // Program the PIC: ICW1, ICW2 base = 0x08 (so IRQ0 -> vector 0x08).
        cpu.port_out(0x20, 0x11);
        cpu.port_out(0x21, 0x08);
        // Install an IVT entry for vector 0x08 -> handler at 0x0000:0x0200.
        cpu.mem.write_u16(0x08 * 4, 0x0200);
        cpu.mem.write_u16(0x08 * 4 + 2, 0x0000);
        // Handler: mov al, 0x01 ; out 0x21, al (mask IRQ0) ; mov ax, 0x77 ; iret
        cpu.mem.load(0x0200, &[
            0xB0, 0x01,       // mov al, 0x01
            0xE6, 0x21,       // out 0x21, al
            0xB8, 0x77, 0x00, // mov ax, 0x77
            0xCF,             // iret
        ]);
        // Main loop: hlt (interrupt should still fire before it).
        cpu.mem.load(0x0100, &[0xF4]);
        cpu.cs = 0;
        cpu.ip = 0x0100;
        // Run enough instructions for the PIT to wrap (count 1 -> 1 tick).
        cpu.run(8);
        // The handler ran: AX = 0x77.
        assert_eq!(cpu.ax, 0x77);
        // The main hlt was reached (timer masked, so no re-entry).
        assert!(cpu.halted);
        // Stack restored after the interrupt frame.
        assert_eq!(cpu.sp, 0x0100);
    }

    #[test]
    fn keyboard_irq1_delivered_through_pic() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.sp = 0x0100;
        // Program the PIC: ICW1, ICW2 base = 0x08 (so IRQ1 -> vector 0x09).
        cpu.port_out(0x20, 0x11);
        cpu.port_out(0x21, 0x08);
        // Install an IVT entry for vector 0x09 -> handler at 0x0000:0x0300.
        cpu.mem.write_u16(0x09 * 4, 0x0300);
        cpu.mem.write_u16(0x09 * 4 + 2, 0x0000);
        // Handler: mov ax, 0x55 ; iret
        cpu.mem.load(0x0300, &[
            0xB8, 0x55, 0x00,
            0xCF,
        ]);
        // Queue a scancode on the 8042 (raises IRQ1).
        cpu.kbd.push_scancode(0x1E);
        // Main loop: hlt.
        cpu.mem.load(0x0100, &[0xF4]);
        cpu.cs = 0;
        cpu.ip = 0x0100;
        cpu.run(8);
        assert_eq!(cpu.ax, 0x55);
        assert!(cpu.halted);
    }

    #[test]
    fn ide_read_via_port_io() {
        let mut cpu = Cpu::new();
        // Load a disk image with a marker in sector 0.
        let mut disk = vec![0u8; 512 * 2];
        disk[0..4].copy_from_slice(b"IDED");
        cpu.ide.load_disk(disk);
        // Program the IDE command block: count=1, LBA=0, drive 0, then read.
        cpu.port_out16(0x1F2, 1);
        cpu.port_out16(0x1F3, 0);
        cpu.port_out16(0x1F4, 0);
        cpu.port_out16(0x1F5, 0);
        cpu.port_out16(0x1F6, 0xE0);
        cpu.port_out16(0x1F7, 0x20); // read sectors
        // Read two words from the data register.
        let w0 = cpu.port_in16(0x1F0);
        let w1 = cpu.port_in16(0x1F0);
        // Disk holds "IDED": word 0 = 'I' | 'D'<<8, word 1 = 'E' | 'D'<<8.
        assert_eq!(w0, b'I' as u16 | ((b'D' as u16) << 8));
        assert_eq!(w1, b'E' as u16 | ((b'D' as u16) << 8));
    }

    #[test]
    fn vga_graphics_mode_via_bios() {
        let mut cpu = Cpu::new();
        // mov ah, 0x00 ; mov al, 0x13 ; int 0x10 ; hlt
        load(&mut cpu, &[
            0xB4, 0x00, 0xB0, 0x13, 0xCD, 0x10,
            0xF4,
        ]);
        cpu.run(16);
        assert!(cpu.vga.is_graphics());
        assert_eq!(cpu.vga.framebuffer.len(), 320 * 200);
    }

    #[test]
    fn exception_without_idt_triple_faults() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.ss = 0;
        cpu.esp = 0x0100;
        // No IDT installed (idt_base=0, idt_limit=0). A #DE fires.
        // mov ax, 1 ; mov bx, 0 ; div bx ; hlt
        cpu.mem.load(0x1000, &[
            0x66, 0xB8, 0x01, 0x00, 0x00, 0x00,
            0x66, 0xBB, 0x00, 0x00, 0x00, 0x00,
            0x66, 0xF7, 0xF3,
            0xF4,
        ]);
        cpu.cs = 0x08;
        cpu.eip = 0x1000;
        cpu.run(32);
        // The #DE fired but there's no IDT -> triple fault -> halt.
        assert!(cpu.triple_fault);
        assert!(cpu.halted);
    }
}
