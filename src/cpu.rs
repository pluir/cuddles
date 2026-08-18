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

/// Number of TLB entries (must be a power of two).
const TLB_SIZE: usize = 256;
/// Size of the debug EIP ring buffer (see `Cpu::eip_ring`).
pub const EIP_RING: usize = 4096;
/// Cap on the number of exceptions recorded in `Cpu::exc_log`.
pub const EXC_LOG_MAX: usize = 512;
/// TLB index mask.
const TLB_MASK: usize = TLB_SIZE - 1;

/// A single TLB entry: caches a 4 KiB page mapping (linear page → physical
/// page). `valid` is the valid bit; the entry is invalidated on MOV CR3,
/// INVLPG, or any write that changes page tables.
#[derive(Clone, Copy)]
pub struct TlbEntry {
    pub valid: bool,
    /// High 20 bits of the linear address (the virtual page number).
    pub vpage: u32,
    /// High 20 bits of the physical address (the physical page number).
    pub ppage: u32,
    /// Whether the mapping permits writes. Cached alongside the translation
    /// because a permission check that only ran on a TLB *miss* would let the
    /// second write to a read-only page through.
    pub writable: bool,
    /// Whether the mapping is reachable from user mode.
    pub user: bool,
    /// True once the accessed (and, for a write, dirty) bits have been set in
    /// the page tables for this entry, so the common case does not re-write
    /// them on every access.
    pub dirtied: bool,
}

impl Default for TlbEntry {
    fn default() -> Self {
        TlbEntry {
            valid: false, vpage: 0, ppage: 0,
            writable: false, user: false, dirtied: false,
        }
    }
}

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

/// Flag bit positions in the EFLAGS register.
///
/// EFLAGS is 32 bits wide, not 16: a 32-bit kernel probes the high half to
/// identify the CPU. Linux toggles `AC` (bit 18) to tell a 386 from a 486 and
/// `ID` (bit 21) to decide whether `CPUID` exists at all, both through
/// `PUSHFD`/`POPFD`. With a 16-bit flags word those bits read back as zero and
/// the probes silently reach the wrong conclusion.
pub mod flags {
    pub const CF: u32 = 0x0000_0001;
    pub const PF: u32 = 0x0000_0004;
    pub const AF: u32 = 0x0000_0010;
    pub const ZF: u32 = 0x0000_0040;
    pub const SF: u32 = 0x0000_0080;
    pub const TF: u32 = 0x0000_0100;
    pub const IF: u32 = 0x0000_0200;
    pub const DF: u32 = 0x0000_0400;
    pub const OF: u32 = 0x0000_0800;
    pub const IOPL: u32 = 0x0000_3000;
    pub const NT: u32 = 0x0000_4000;
    pub const RF: u32 = 0x0001_0000;
    pub const VM: u32 = 0x0002_0000;
    pub const AC: u32 = 0x0004_0000;
    pub const VIF: u32 = 0x0008_0000;
    pub const VIP: u32 = 0x0010_0000;
    pub const ID: u32 = 0x0020_0000;

    /// Bits that software may write through POPF/POPFD/IRET. Bit 1 is always
    /// set, bits 3/5/15 are always clear, and bits 22-31 are reserved.
    pub const WRITABLE: u32 = 0x003F_7FD5;
    /// The value the reserved bits must always read back as.
    pub const ALWAYS_SET: u32 = 0x0000_0002;
}

pub struct Cpu {
    // The general registers are stored once, 32 bits wide. AX, CX, ... are
    // *views* of the low half of EAX, ECX, ... reached through the `ax()` /
    // `set_ax()` accessors below, never a second copy: keeping both meant
    // every 32-bit write had to remember to refresh the 16-bit half, and the
    // ones that forgot produced register corruption a long way from the
    // instruction that caused it.
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
    pub flags: u32,
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
    /// Task register: the selector loaded by LTR and the base/limit of the
    /// TSS it names. The TSS is what makes ring 3 workable at all -- an
    /// interrupt taken while the CPU is in user mode switches to the ring-0
    /// stack recorded in it (SS0 at offset 8, ESP0 at offset 4).
    pub tr_selector: u16,
    pub tr_base: u32,
    pub tr_limit: u32,
    /// Local descriptor table, loaded by LLDT. A selector with its TI bit set
    /// is resolved here rather than in the GDT.
    pub ldt_selector: u16,
    pub ldt_base: u32,
    pub ldt_limit: u32,
    /// Cached descriptors for ES, CS, SS, DS, FS, GS.
    pub seg_desc: [Descriptor; 6],

    // ---- Paging state ----
    /// Control registers. CR0 bit 31 = PG (paging enabled), bit 0 = PE
    /// (protected mode enabled). CR3 = page-directory base register.
    pub cr0: u32,
    pub cr2: u32,
    pub cr3: u32,
    pub cr4: u32,
    /// Debug registers DR0-DR7. Hardware breakpoints are not implemented;
    /// these exist so the kernel's startup writes and read-backs agree.
    pub dr: [u32; 8],
    /// Translation Lookaside Buffer: caches linear→physical page mappings
    /// so we don't walk the page tables on every byte fetch.
    pub tlb: [TlbEntry; TLB_SIZE],

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
    /// MC146818 CMOS RTC (ports 0x70/0x71).
    pub cmos: crate::cmos::Cmos,
    /// IDE/ATA disk controller.
    pub ide: crate::ide::Ide,
    /// x87 FPU.
    pub fpu: crate::fpu::Fpu,

    // ---- Debug tracing ----
    /// Cached trace file handle (opened once when X86EMU_TRACE is set).
    /// None when tracing is disabled — the common case.
    pub trace_file: Option<std::fs::File>,
    /// True if X86EMU_TRACE was set at startup (checked once, not per-instruction).
    pub trace_enabled: bool,
    /// Instruction number at which tracing starts (X86EMU_TRACE_FROM).
    /// Tracing every instruction of a boot is billions of lines; a window is
    /// what makes a trace usable that far in.
    pub trace_from: u64,

    // ---- Instruction fetch cache (#1) ----
    /// Cached physical address of the instruction stream. Valid between
    /// successive `fetch_u8` calls within a single instruction; invalidated
    /// at the start of each `step()` and whenever EIP/CS/page-mapping changes.
    pub phys_ip_cache: usize,
    /// Linear address corresponding to `phys_ip_cache` (for page-boundary checks).
    pub phys_ip_linear: u32,
    /// True when `phys_ip_cache` holds a valid mapping for the current EIP.
    pub phys_ip_valid: bool,

    /// EIP (or IP, in real mode) at the start of the instruction being
    /// executed. A *fault* reports the address of the instruction that
    /// faulted, not the one after it, so that the handler can fix things up
    /// and restart it -- which is exactly what demand paging and the kernel's
    /// exception table both rely on.
    pub eip_start: u32,
    pub ip_start: u16,

    // ---- Debug instrumentation (off unless X86EMU_DEBUG is set) ----
    /// True if X86EMU_DEBUG was set at startup. Gates the ring buffer and
    /// the exception log so a normal run pays nothing for them.
    pub debug_enabled: bool,
    /// Ring buffer of the most recent instruction pointers (linear EIP).
    /// Written every instruction when `debug_enabled`; dumped on demand.
    pub eip_ring: Vec<u32>,
    /// Write cursor into `eip_ring`.
    pub eip_ring_pos: usize,
    /// Log of dispatched exceptions: (instruction count, vector, error code,
    /// faulting EIP, CR2). Capped at `EXC_LOG_MAX` entries.
    pub exc_log: Vec<(u64, u8, Option<u32>, u32, u32)>,
    /// Count of every exception dispatched, by vector (not capped).
    pub exc_counts: [u64; 32],
    /// Count of hardware interrupts actually delivered to the CPU.
    pub irq_count: u64,
    /// Per-vector delivery counts, for diagnostics.
    pub irq_vectors: [u64; 256],
    /// Count of instructions executed at CPL 3, for diagnostics.
    pub user_instructions: u64,
    /// Count of privilege-level switches into the kernel via a gate.
    pub ring_switches: u64,
    /// PIT input cycles accumulated towards the next RTC second.
    pub pit_subsecond: u64,
    /// Vertical-retrace bit of the VGA input-status register, flipped per read.
    pub vga_retrace: bool,
    /// When set (X86EMU_TRAP_EIP=<hex>), the run halts the moment execution
    /// reaches this address. The EIP ring buffer then holds the instructions
    /// that led there — the way to find who jumped to a bad address.
    pub trap_eip: Option<u32>,
    /// Set once `trap_eip` has been reached, to end the run outright rather
    /// than parking as HLT does.
    pub trapped: bool,
    /// Stop before the n'th user-mode instruction (X86EMU_TRAP_USER).
    pub trap_user: Option<u64>,
    /// X86EMU_NO_TLB: bypass the TLB and walk the page tables on every
    /// access. Slow, but it settles whether a bug is a stale translation.
    pub no_tlb: bool,
    /// Linear address to watch for writes (X86EMU_WATCH), reported with the
    /// EIP that wrote it. Finding *who* wrote a wrong value is otherwise a
    /// matter of reading a million lines of trace.
    pub watch_linear: Option<u32>,
    /// Physical address to watch for writes (X86EMU_WATCH_PHYS). Catches
    /// stores made through *any* linear alias, including the kernel's direct
    /// map -- which a linear watch cannot see.
    pub watch_phys: Option<u32>,
    /// Log of writes to `watch_linear`: (instruction count, EIP, value).
    pub watch_log: Vec<(u64, u32, u32)>,
    /// Log of system calls made from user mode: (instruction count, EAX, EBX,
    /// ECX, EDX). Recorded when X86EMU_DEBUG is set.
    pub syscall_log: Vec<(u64, u32, u32, u32, u32)>,
    /// Opcodes the decoder did not recognise, with a hit count and the EIP of
    /// the first sighting. Keyed by opcode (`0x0Fxx` for two-byte opcodes).
    /// Always recorded — an unimplemented instruction is a bug worth naming
    /// even in a release run, and the map only grows once per distinct opcode.
    pub unknown_ops: std::collections::BTreeMap<u16, (u64, u32)>,
}

impl Cpu {
    pub fn new() -> Self {
        let trace_enabled = std::env::var("X86EMU_TRACE").is_ok();
        let debug_enabled = std::env::var("X86EMU_DEBUG").is_ok();
        let trace_file = if trace_enabled {
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("trace.txt")
                .ok()
        } else {
            None
        };
        Cpu {
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
            tr_selector: 0,
            tr_base: 0,
            tr_limit: 0,
            ldt_selector: 0,
            ldt_base: 0,
            ldt_limit: 0,
            seg_desc: [Descriptor::default(); 6],
            cr0: 0,
            cr2: 0,
            cr3: 0,
            cr4: 0,
            dr: [0; 8],
            pit: crate::pit::Pit::new(),
            pic: crate::pic::Pic::new(),
            servicing_irq: false,
            vga: crate::vga::Vga::new(),
            kbd: crate::kbd::Kbd::new(),
            dma: crate::dma::Dma::new(),
            cmos: crate::cmos::Cmos::new(),
            ide: crate::ide::Ide::new(),
            fpu: crate::fpu::Fpu::new(),
            pending_exception: None,
            tlb: [TlbEntry::default(); TLB_SIZE],
            trace_file,
            trace_enabled,
            trace_from: std::env::var("X86EMU_TRACE_FROM").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(0),
            phys_ip_cache: 0,
            phys_ip_linear: 0,
            phys_ip_valid: false,
            eip_start: 0,
            ip_start: 0,
            debug_enabled,
            eip_ring: if debug_enabled { vec![0; EIP_RING] } else { Vec::new() },
            eip_ring_pos: 0,
            exc_log: Vec::new(),
            exc_counts: [0; 32],
            irq_count: 0,
            irq_vectors: [0; 256],
            user_instructions: 0,
            ring_switches: 0,
            pit_subsecond: 0,
            vga_retrace: false,
            unknown_ops: std::collections::BTreeMap::new(),
            trapped: false,
            trap_user: std::env::var("X86EMU_TRAP_USER").ok()
                .and_then(|v| v.parse().ok()),
            no_tlb: std::env::var("X86EMU_NO_TLB").is_ok(),
            watch_linear: std::env::var("X86EMU_WATCH").ok()
                .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()),
            watch_phys: std::env::var("X86EMU_WATCH_PHYS").ok()
                .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()),
            watch_log: Vec::new(),
            syscall_log: Vec::new(),
            trap_eip: std::env::var("X86EMU_TRAP_EIP").ok()
                .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()),
        }
    }

    /// Record an opcode the decoder could not handle, for diagnostics.
    /// The EIP stored is that of the *first* sighting (the instruction has
    /// already been fetched, so it points just past the opcode byte).
    pub fn note_unknown_opcode(&mut self, opcode: u16) {
        let eip = if self.pe { self.eip } else { self.ip as u32 };
        let e = self.unknown_ops.entry(opcode).or_insert((0, eip));
        e.0 += 1;
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

    /// Invalidate the cached instruction-fetch physical address. Must be
    /// called whenever EIP, CS, or the page mapping changes (jumps, calls,
    /// returns, interrupts, exceptions, paging enable/disable, MOV CR3).
    #[inline]
    pub fn invalidate_phys_ip(&mut self) {
        self.phys_ip_valid = false;
    }

    /// Compute the linear address of the instruction stream (for page-boundary
    /// checks in the fetch cache).
    #[inline]
    fn ip_linear(&self) -> u32 {
        if self.pe {
            self.seg_desc[SegReg::Cs as usize].base.wrapping_add(self.eip)
        } else {
            ((self.cs as u32) << 4) + self.ip as u32
        }
    }

    // ---- 16-bit register access ----

    /// The 16-bit view of `eax`.
    #[inline]
    pub fn ax(&self) -> u16 { self.eax as u16 }
    /// Write the 16-bit view of `eax`, preserving its high half.
    #[inline]
    pub fn set_ax(&mut self, v: u16) { self.eax = (self.eax & 0xFFFF_0000) | v as u32; }

    /// The 16-bit view of `ecx`.
    #[inline]
    pub fn cx(&self) -> u16 { self.ecx as u16 }
    /// Write the 16-bit view of `ecx`, preserving its high half.
    #[inline]
    pub fn set_cx(&mut self, v: u16) { self.ecx = (self.ecx & 0xFFFF_0000) | v as u32; }

    /// The 16-bit view of `edx`.
    #[inline]
    pub fn dx(&self) -> u16 { self.edx as u16 }
    /// Write the 16-bit view of `edx`, preserving its high half.
    #[inline]
    pub fn set_dx(&mut self, v: u16) { self.edx = (self.edx & 0xFFFF_0000) | v as u32; }

    /// The 16-bit view of `ebx`.
    #[inline]
    pub fn bx(&self) -> u16 { self.ebx as u16 }
    /// Write the 16-bit view of `ebx`, preserving its high half.
    #[inline]
    pub fn set_bx(&mut self, v: u16) { self.ebx = (self.ebx & 0xFFFF_0000) | v as u32; }

    /// The 16-bit view of `esp`.
    #[inline]
    pub fn sp(&self) -> u16 { self.esp as u16 }
    /// Write the 16-bit view of `esp`, preserving its high half.
    #[inline]
    pub fn set_sp(&mut self, v: u16) { self.esp = (self.esp & 0xFFFF_0000) | v as u32; }

    /// The 16-bit view of `ebp`.
    #[inline]
    pub fn bp(&self) -> u16 { self.ebp as u16 }
    /// Write the 16-bit view of `ebp`, preserving its high half.
    #[inline]
    pub fn set_bp(&mut self, v: u16) { self.ebp = (self.ebp & 0xFFFF_0000) | v as u32; }

    /// The 16-bit view of `esi`.
    #[inline]
    pub fn si(&self) -> u16 { self.esi as u16 }
    /// Write the 16-bit view of `esi`, preserving its high half.
    #[inline]
    pub fn set_si(&mut self, v: u16) { self.esi = (self.esi & 0xFFFF_0000) | v as u32; }

    /// The 16-bit view of `edi`.
    #[inline]
    pub fn di(&self) -> u16 { self.edi as u16 }
    /// Write the 16-bit view of `edi`, preserving its high half.
    #[inline]
    pub fn set_di(&mut self, v: u16) { self.edi = (self.edi & 0xFFFF_0000) | v as u32; }

    pub fn reg16(&self, r: Reg16) -> u16 {
        match r as u8 {
            0 => self.ax(), 1 => self.cx(), 2 => self.dx(), 3 => self.bx(),
            4 => self.sp(), 5 => self.bp(), 6 => self.si(), _ => self.di(),
        }
    }

    #[inline]
    pub fn set_reg16(&mut self, r: Reg16, v: u16) {
        if self.pending_exception.is_some() { return; }
        self.set_reg16_raw(r, v);
    }

    /// `set_reg16` without the fault suppression. See `set_reg32_raw`.
    #[inline]
    pub fn set_reg16_raw(&mut self, r: Reg16, v: u16) {
        match r as u8 {
            0 => self.set_ax(v), 1 => self.set_cx(v),
            2 => self.set_dx(v), 3 => self.set_bx(v),
            4 => self.set_sp(v), 5 => self.set_bp(v),
            6 => self.set_si(v), _ => self.set_di(v),
        }
    }

    // ---- 32-bit register access ----

    #[inline]
    pub fn reg32(&self, r: Reg32) -> u32 {
        match r as u8 {
            0 => self.eax, 1 => self.ecx, 2 => self.edx, 3 => self.ebx,
            4 => self.esp, 5 => self.ebp, 6 => self.esi, _ => self.edi,
        }
    }

    /// Write a 32-bit register as an instruction *result*.
    ///
    /// Suppressed once the instruction has raised a fault. A fault means the
    /// instruction did not happen: it will be restarted after the handler
    /// returns, and committing a half-computed result first makes the retry
    /// operate on corrupted state. `add (%edi),%edx` whose load page-faults
    /// must leave EDX alone, or the second attempt adds to a value that has
    /// already been added to.
    #[inline]
    pub fn set_reg32(&mut self, r: Reg32, v: u32) {
        if self.pending_exception.is_some() { return; }
        self.set_reg32_raw(r, v);
    }

    /// Write a 32-bit register unconditionally. For state that must be
    /// recorded *because* of a fault -- the string instructions' index and
    /// count registers, which have to point at the element that faulted so
    /// the restart resumes there.
    #[inline]
    pub fn set_reg32_raw(&mut self, r: Reg32, v: u32) {
        match r as u8 {
            0 => self.eax = v, 1 => self.ecx = v,
            2 => self.edx = v, 3 => self.ebx = v,
            4 => self.esp = v, 5 => self.ebp = v,
            6 => self.esi = v, _ => self.edi = v,
        }
    }

    // ---- 8-bit register access ----

    pub fn reg8(&self, r: Reg8) -> u8 {
        let w = match r {
            Reg8::Al | Reg8::Ah => self.ax(),
            Reg8::Cl | Reg8::Ch => self.cx(),
            Reg8::Dl | Reg8::Dh => self.dx(),
            Reg8::Bl | Reg8::Bh => self.bx(),
        };
        if (r as u8) < 4 { (w & 0xFF) as u8 } else { (w >> 8) as u8 }
    }

    pub fn set_reg8(&mut self, r: Reg8, v: u8) {
        if self.pending_exception.is_some() { return; }
        let lo = (r as u8) < 4;
        let idx = (r as usize) & 3;
        let cur = match idx { 0 => self.ax(), 1 => self.cx(), 2 => self.dx(), _ => self.bx() };
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
            self.seg_desc[s as usize] = self.descriptor_for(selector);
        }
        self.set_seg(s, selector);
        // Loading CS changes the instruction-stream base.
        if s == SegReg::Cs {
            self.invalidate_phys_ip();
        }
    }

    /// Resolve a selector to its descriptor, taking the table-indicator bit
    /// (bit 2) into account: set means the LDT, clear means the GDT.
    pub fn descriptor_for(&self, selector: u16) -> crate::protected::Descriptor {
        let idx = (selector >> 3) & 0x1FFF;
        let base = if selector & 4 != 0 { self.ldt_base } else { self.gdt_base };
        crate::protected::read_descriptor(&self.mem, base, idx)
    }

    /// Load the task register from a selector, caching the TSS base/limit.
    pub fn load_tr(&mut self, selector: u16) {
        self.tr_selector = selector;
        let d = self.descriptor_for(selector);
        self.tr_base = d.base;
        self.tr_limit = d.limit;
    }

    /// Load the local descriptor table register. A null selector clears it.
    pub fn load_ldt(&mut self, selector: u16) {
        self.ldt_selector = selector;
        if selector & 0xFFF8 == 0 {
            self.ldt_base = 0;
            self.ldt_limit = 0;
            return;
        }
        // The LDT descriptor always lives in the GDT.
        let idx = (selector >> 3) & 0x1FFF;
        let d = crate::protected::read_descriptor(&self.mem, self.gdt_base, idx);
        self.ldt_base = d.base;
        self.ldt_limit = d.limit;
    }

    /// The ring-0 stack recorded in the TSS: (SS0, ESP0).
    pub fn tss_stack0(&self) -> (u16, u32) {
        let esp0 = self.mem.read_u32(Memory::phys32(self.tr_base.wrapping_add(4)));
        let ss0 = self.mem.read_u16(Memory::phys32(self.tr_base.wrapping_add(8)));
        (ss0, esp0)
    }

    /// Translate a logical address through a segment to a physical address.
    /// Records a #PF (page fault) in `pending_exception` if paging is enabled
    /// and the page is not present.
    pub fn translate(&mut self, s: SegReg, offset: u32) -> usize {
        self.translate_access(s, offset, false)
    }

    /// `translate` for a store. Separate from the read form so paging can tell
    /// a load from a store, which is what CR0.WP and the page-fault error code
    /// both turn on.
    pub fn translate_write(&mut self, s: SegReg, offset: u32) -> usize {
        self.translate_access(s, offset, true)
    }

    fn translate_access(&mut self, s: SegReg, offset: u32, write: bool) -> usize {
        let linear = if self.pe {
            self.seg_desc[s as usize].base.wrapping_add(offset)
        } else {
            ((self.seg(s) as u32) << 4) + offset
        };
        self.apply_paging_access(linear, write)
    }

    /// Flush the entire TLB (called on MOV CR3, or when paging is toggled).
    #[inline]
    pub fn flush_tlb(&mut self) {
        for e in self.tlb.iter_mut() {
            e.valid = false;
        }
        self.invalidate_phys_ip();
    }

    /// Invalidate a single TLB entry for the given linear address (INVLPG).
    #[inline]
    pub fn invlpg(&mut self, linear: u32) {
        let vpage = linear >> 12;
        let idx = (vpage as usize) & TLB_MASK;
        let e = &self.tlb[idx];
        if e.valid && e.vpage == vpage {
            self.tlb[idx].valid = false;
        }
    }

    /// Apply paging to a linear address if CR0.PG is set. If the page is not
    /// present, raise a #PF (vector 0x0E) with the faulting linear address in
    /// CR2 and an error code, and return 0.
    ///
    /// Uses the TLB to avoid walking the page tables on every access. On a
    /// TLB miss, the page table walk fills the TLB entry.
    #[inline]
    pub fn apply_paging(&mut self, linear: u32) -> usize {
        self.apply_paging_access(linear, false)
    }

    /// Translate a linear address, checking it against the access being made.
    ///
    /// `write` distinguishes a store from a load, which matters twice: a
    /// supervisor store to a read-only page faults when CR0.WP is set, and the
    /// page-fault error code has to say which kind of access faulted. The
    /// current privilege level comes from CS's RPL.
    pub fn apply_paging_access(&mut self, linear: u32, write: bool) -> usize {
        let phys = self.apply_paging_inner(linear, write);
        if write {
            if let Some(w) = self.watch_linear {
                // A store of any width that covers the watched address counts.
                // Any store within a few bytes either side: a 16-bit store
                // just above the watched address still changes the dword read
                // from it, and a window that only looked forwards missed
                // exactly that.
                if linear >= w.wrapping_sub(4) && linear <= w.wrapping_add(4) {
                    let eip = if self.pe { self.eip_start } else { self.ip_start as u32 };
                    let n = self.instructions_executed;
                    // Keep the most recent writes: the one that left the bad
                    // value is the last, not the first.
                    if self.watch_log.len() >= 64 {
                        self.watch_log.remove(0);
                    }
                    self.watch_log.push((n, eip, phys as u32));
                }
            }
            if let Some(w) = self.watch_phys {
                let p = phys as u32;
                if p >= w.wrapping_sub(4) && p <= w.wrapping_add(4) {
                    let eip = if self.pe { self.eip_start } else { self.ip_start as u32 };
                    let n = self.instructions_executed;
                    if self.watch_log.len() >= 64 { self.watch_log.remove(0); }
                    self.watch_log.push((n, eip, p));
                }
            }
        }
        phys
    }

    fn apply_paging_inner(&mut self, linear: u32, write: bool) -> usize {
        if self.cr0 & 0x8000_0000 == 0 {
            return Memory::phys32(linear);
        }
        let user = self.cpl() == 3;
        // Fast path: check the TLB.
        let vpage = linear >> 12;
        let idx = (vpage as usize) & TLB_MASK;
        let entry = self.tlb[idx];
        if entry.valid && entry.vpage == vpage && !self.no_tlb {
            if self.access_allowed(entry.writable, entry.user, write, user) {
                if write && !entry.dirtied {
                    self.mark_accessed(linear, true);
                    self.tlb[idx].dirtied = true;
                }
                let offset = linear & 0xFFF;
                return Memory::phys32((entry.ppage << 12) | offset);
            }
            // Present, but the access is not permitted: a protection fault.
            self.raise_page_fault(linear, true, write, user);
            return 0;
        }
        // TLB miss: walk the page tables.
        match crate::paging::translate(&self.mem, self.cr3, linear) {
            Some(map) => {
                if !self.access_allowed(map.writable, map.user, write, user) {
                    self.raise_page_fault(linear, true, write, user);
                    return 0;
                }
                // Fill the TLB entry. For 4 MiB pages the walk already folded
                // the offset in, so caching the page number works for both
                // 4K and 4M mappings.
                let ppage = (map.phys >> 12) as u32;
                self.tlb[idx] = TlbEntry {
                    valid: true, vpage, ppage,
                    writable: map.writable, user: map.user, dirtied: write,
                };
                self.set_accessed_bits(&map, write);
                map.phys
            }
            None => {
                self.raise_page_fault(linear, false, write, user);
                0
            }
        }
    }

    /// Offset an IDT entry points at, for diagnostics.
    pub fn idt_target(&self, vector: u8) -> u32 {
        let entry = self.idt_base.wrapping_add((vector as u32) * 8);
        let addr = Memory::phys32(entry);
        let lo = self.mem.read_u16(addr) as u32;
        let hi = self.mem.read_u16(addr + 6) as u32;
        lo | (hi << 16)
    }

    /// Current privilege level: the RPL of the code segment selector.
    #[inline]
    pub fn cpl(&self) -> u8 {
        if self.pe { (self.cs & 3) as u8 } else { 0 }
    }

    /// Is an access permitted against a mapping's permission bits?
    ///
    /// A supervisor read reaches anything. A supervisor *write* to a
    /// read-only page is allowed only while CR0.WP is clear — that switch is
    /// the whole point of `test_wp_bit` in the kernel, and of copy-on-write
    /// working at all once user pages exist.
    #[inline]
    fn access_allowed(&self, writable: bool, page_user: bool, write: bool, user: bool) -> bool {
        if user {
            if !page_user { return false; }
            if write && !writable { return false; }
            return true;
        }
        const CR0_WP: u32 = 1 << 16;
        !(write && !writable && self.cr0 & CR0_WP != 0)
    }

    /// Record a page fault: CR2 takes the faulting linear address and the
    /// error code says whether the page was present, whether the access was a
    /// write, and whether it came from user mode.
    fn raise_page_fault(&mut self, linear: u32, present: bool, write: bool, user: bool) {
        // Keep the first fault: CR2 and the error code describe the access
        // that actually failed, and a second translation later in the same
        // instruction would otherwise rewrite them.
        if self.pending_exception.is_some() {
            return;
        }
        self.cr2 = linear;
        let code = (present as u32) | ((write as u32) << 1) | ((user as u32) << 2);
        self.pending_exception = Some((0x0E, Some(code)));
    }

    /// Set the accessed (and for a store, dirty) bits of a mapping.
    fn set_accessed_bits(&mut self, map: &crate::paging::Mapping, write: bool) {
        use crate::paging::pte;
        match map.pte_addr {
            Some(addr) => {
                let pde = self.mem.read_u32(map.pde_addr);
                self.mem.write_u32(map.pde_addr, pde | pte::A);
                let e = self.mem.read_u32(addr);
                self.mem.write_u32(addr, e | pte::A | if write { pte::D } else { 0 });
            }
            None => {
                // 4 MiB page: the PDE carries both bits.
                let pde = self.mem.read_u32(map.pde_addr);
                self.mem.write_u32(map.pde_addr, pde | pte::A | if write { pte::D } else { 0 });
            }
        }
    }

    /// Set the accessed/dirty bits for a linear address whose translation was
    /// served from the TLB (so the walk's entry addresses are not to hand).
    fn mark_accessed(&mut self, linear: u32, write: bool) {
        if let Some(map) = crate::paging::translate(&self.mem, self.cr3, linear) {
            self.set_accessed_bits(&map, write);
        }
    }

    // ---- Flag helpers ----

    pub fn set_flag(&mut self, f: u32, on: bool) {
        if on { self.flags |= f; } else { self.flags &= !f; }
    }
    pub fn get_flag(&self, f: u32) -> bool { (self.flags & f) != 0 }

    /// Read the time-stamp counter (used by RDTSC).
    pub fn rdtsc(&self) -> u64 { self.tsc }

    // ---- Instruction stream fetch ----

    /// Ensure the phys_ip cache is valid for the current EIP. Called once
    /// at the start of an instruction (or after a page-boundary crossing).
    #[inline]
    fn ensure_phys_ip(&mut self) {
        if !self.phys_ip_valid {
            let linear = self.ip_linear();
            self.phys_ip_linear = linear;
            self.phys_ip_cache = self.apply_paging(linear);
            self.phys_ip_valid = true;
        }
    }

    /// Peek at the next instruction byte without advancing EIP. Uses the
    /// fetch cache for speed. Used by the decoder's prefix loop.
    #[inline]
    pub fn peek_u8(&mut self) -> u8 {
        self.ensure_phys_ip();
        self.mem.read_u8_raw(self.phys_ip_cache)
    }

    #[inline]
    pub fn fetch_u8(&mut self) -> u8 {
        self.ensure_phys_ip();
        let b = self.mem.read_u8_raw(self.phys_ip_cache);
        if self.pe {
            self.eip = self.eip.wrapping_add(1);
        } else {
            self.ip = self.ip.wrapping_add(1);
        }
        // Advance the cached physical + linear addresses. If we just crossed
        // a page boundary, invalidate so the next fetch re-translates.
        self.phys_ip_cache = self.phys_ip_cache.wrapping_add(1);
        let crossed = (self.phys_ip_linear & 0xFFF) == 0xFFF;
        self.phys_ip_linear = self.phys_ip_linear.wrapping_add(1);
        if crossed {
            self.phys_ip_valid = false;
        }
        b
    }

    #[inline]
    pub fn fetch_u16(&mut self) -> u16 {
        // Fast path: if both bytes are on the same page, read them without
        // any re-translation. Only re-translate if we straddle a page boundary.
        self.ensure_phys_ip();
        let addr = self.phys_ip_cache;
        // Check if the 2 bytes straddle a page boundary.
        let offset = self.phys_ip_linear & 0xFFF;
        if offset <= 0xFFE {
            // Same page: read both bytes directly.
            let lo = self.mem.read_u8_raw(addr) as u16;
            let hi = self.mem.read_u8_raw(addr + 1) as u16;
            // Advance EIP by 2.
            if self.pe {
                self.eip = self.eip.wrapping_add(2);
            } else {
                self.ip = self.ip.wrapping_add(2);
            }
            self.phys_ip_cache = self.phys_ip_cache.wrapping_add(2);
            self.phys_ip_linear = self.phys_ip_linear.wrapping_add(2);
            // Both bytes were on this page, but the *next* byte may not be:
            // an operand ending exactly at 0xFFF leaves the cursor at offset 0
            // of the following page, whose physical address is unrelated.
            // Without this the next fetch reads from the wrong page entirely,
            // which corrupts the tail of any immediate or displacement that
            // happens to end on a page boundary.
            if offset == 0xFFE {
                self.phys_ip_valid = false;
            }
            return lo | (hi << 8);
        }
        // Page boundary: fall back to two single-byte fetches.
        let lo = self.fetch_u8() as u16;
        let hi = self.fetch_u8() as u16;
        lo | (hi << 8)
    }

    #[inline]
    pub fn fetch_u32(&mut self) -> u32 {
        // Reuse fetch_u16 for the two halves. The fetch cache keeps this fast.
        let lo = self.fetch_u16() as u32;
        let hi = self.fetch_u16() as u32;
        lo | (hi << 16)
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

    /// Same as `operand_seg` but public (for use by INVLPG in instructions.rs).
    pub fn operand_seg_for_exec(&self, default: SegReg) -> SegReg {
        self.seg_override.unwrap_or(default)
    }

    /// Compute the offset (without segment translation) of a 16-bit-addressed
    /// memory operand. Used by INVLPG which needs the linear address.
    pub fn modrm_offset(&self, m: &ModRm) -> u32 {
        let base = match m.rm {
            0 => self.bx().wrapping_add(self.si()),
            1 => self.bx().wrapping_add(self.di()),
            2 => self.bp().wrapping_add(self.si()),
            3 => self.bp().wrapping_add(self.di()),
            4 => self.si(),
            5 => self.di(),
            6 => self.bp(),
            _ => self.bx(),
        };
        let mut ea = base as u32;
        if let Some(d8) = m.disp8 { ea = ea.wrapping_add(d8 as u32); }
        if let Some(d16) = m.disp16 { ea = ea.wrapping_add(d16 as u32); }
        ea
    }

    /// Compute the offset (without segment translation) of a 32-bit-addressed
    /// memory operand. Used by INVLPG which needs the linear address.
    pub fn modrm_offset32(&self, m: &ModRm) -> u32 {
        let mut ea: u32 = 0;
        if let Some(sib) = m.sib {
            let scale = 1u32 << ((sib >> 6) & 3);
            let index = (sib >> 3) & 7;
            let base = sib & 7;
            if index != 4 {
                ea = ea.wrapping_add(self.reg32(Reg::reg32(index)).wrapping_mul(scale));
            }
            if !(m.mod_field == 0 && base == 5) {
                ea = ea.wrapping_add(self.reg32(Reg::reg32(base)));
            }
        } else if !(m.mod_field == 0 && m.rm == 5) {
            ea = ea.wrapping_add(self.reg32(Reg::reg32(m.rm)));
        }
        if let Some(d32) = m.disp32 { ea = ea.wrapping_add(d32); }
        ea
    }

    /// Compute the physical address of a 16-bit-addressed memory operand.
    pub fn modrm_addr(&mut self, m: &ModRm) -> usize {
        self.modrm_addr_access(m, false)
    }

    /// `modrm_addr` for a store.
    pub fn modrm_addr_write(&mut self, m: &ModRm) -> usize {
        self.modrm_addr_access(m, true)
    }

    /// `modrm_addr` with the access type chosen by the caller. Bit-string
    /// instructions need this: their effective address is the ModR/M address
    /// plus a signed operand-sized displacement.
    pub fn modrm_addr_access_pub(&mut self, m: &ModRm, write: bool) -> usize {
        self.modrm_addr_access(m, write)
    }

    /// 32-bit-addressing counterpart of `modrm_addr_access_pub`.
    pub fn modrm_addr32_access_pub(&mut self, m: &ModRm, write: bool) -> usize {
        self.modrm_addr32_access(m, write)
    }

    fn modrm_addr_access(&mut self, m: &ModRm, write: bool) -> usize {
        let (base, default_seg) = match m.rm {
            0 => (self.bx() + self.si(), SegReg::Ds),
            1 => (self.bx() + self.di(), SegReg::Ds),
            2 => (self.bp() + self.si(), SegReg::Ss),
            3 => (self.bp() + self.di(), SegReg::Ss),
            4 => (self.si(), SegReg::Ds),
            5 => (self.di(), SegReg::Ds),
            6 => (self.bp(), SegReg::Ss),
            _ => (self.bx(), SegReg::Ds),
        };
        let mut ea = base as u32;
        if let Some(d8) = m.disp8 { ea = ea.wrapping_add(d8 as u32); }
        if let Some(d16) = m.disp16 { ea = ea.wrapping_add(d16 as u32); }
        self.translate_access(self.operand_seg(default_seg), ea, write)
    }

    /// Compute the physical address of a 32-bit-addressed memory operand.
    pub fn modrm_addr32(&mut self, m: &ModRm) -> usize {
        self.modrm_addr32_access(m, false)
    }

    /// `modrm_addr32` for a store.
    pub fn modrm_addr32_write(&mut self, m: &ModRm) -> usize {
        self.modrm_addr32_access(m, true)
    }

    fn modrm_addr32_access(&mut self, m: &ModRm, write: bool) -> usize {
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
        self.translate_access(self.operand_seg(default_seg), ea, write)
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
        if self.pending_exception.is_some() { return; }
        if m.is_reg() {
            self.set_reg8(Reg::reg8(m.rm), val);
        } else if self.addrsize {
            let addr = self.modrm_addr32_write(m);
            self.mem.write_u8(addr, val);
        } else {
            let addr = self.modrm_addr_write(m);
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
        if self.pending_exception.is_some() { return; }
        if m.is_reg() {
            self.set_reg16(Reg::reg16(m.rm), val);
        } else if self.addrsize {
            let addr = self.modrm_addr32_write(m);
            self.mem.write_u16(addr, val);
        } else {
            let addr = self.modrm_addr_write(m);
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
        if self.pending_exception.is_some() { return; }
        if m.is_reg() {
            self.set_reg32(Reg::reg32(m.rm), val);
        } else if self.addrsize {
            let addr = self.modrm_addr32_write(m);
            self.mem.write_u32(addr, val);
        } else {
            let addr = self.modrm_addr_write(m);
            self.mem.write_u32(addr, val);
        }
    }

    // ---- Stack ----

    pub fn push16(&mut self, val: u16) {
        if self.pe {
            // Translate before moving ESP: a push whose stack page is not
            // present must be restartable, and a decrement that survived the
            // fault would push twice as far on the retry.
            let new_esp = self.esp.wrapping_sub(2);
            let addr = self.translate_write(SegReg::Ss, new_esp);
            if self.pending_exception.is_some() { return; }
            self.esp = new_esp;
            self.mem.write_u16(addr, val);
        } else {
            self.set_sp(self.sp().wrapping_sub(2));
            self.mem.write_u16(Memory::phys(self.ss, self.sp()), val);
        }
    }

    pub fn pop16(&mut self) -> u16 {
        if self.pe {
            let addr = self.translate(SegReg::Ss, self.esp);
            if self.pending_exception.is_some() { return 0; }
            let v = self.mem.read_u16(addr);
            self.esp = self.esp.wrapping_add(2);
            v
        } else {
            let v = self.mem.read_u16(Memory::phys(self.ss, self.sp()));
            self.set_sp(self.sp().wrapping_add(2));
            v
        }
    }

    pub fn push32(&mut self, val: u32) {
        if self.pe {
            let new_esp = self.esp.wrapping_sub(4);
            let addr = self.translate_write(SegReg::Ss, new_esp);
            if self.pending_exception.is_some() { return; }
            self.esp = new_esp;
            self.mem.write_u32(addr, val);
        } else {
            self.set_sp(self.sp().wrapping_sub(4));
            self.mem.write_u32(Memory::phys(self.ss, self.sp()), val);
        }
    }

    pub fn pop32(&mut self) -> u32 {
        if self.pe {
            let addr = self.translate(SegReg::Ss, self.esp);
            if self.pending_exception.is_some() { return 0; }
            let v = self.mem.read_u32(addr);
            self.esp = self.esp.wrapping_add(4);
            v
        } else {
            let v = self.mem.read_u32(Memory::phys(self.ss, self.sp()));
            self.set_sp(self.sp().wrapping_add(4));
            v
        }
    }

    // ---- Port I/O (devices) ----

    /// Read a byte from an I/O port.
    pub fn port_in(&mut self, port: u16) -> u8 {
        match port {
            0x20 => self.pic.read_command(0x20),
            0x21 => self.pic.read_data(0x21),
            0xA0 => self.pic.read_command(0xA0),
            0xA1 => self.pic.read_data(0xA1),
            0x40 | 0x41 | 0x42 => self.pit.read_data((port - 0x40) as u8),
            // 8042 keyboard controller.
            0x60 => self.kbd.read_data(),
            0x64 => self.kbd.read_status(),
            // 8237 DMA status.
            0x08 => self.dma.read_status(),
            // CMOS RTC. 0x70 is write-only on real hardware.
            0x71 => self.cmos.read_data(),
            // IDE command block (byte registers).
            0x1F1 => self.ide.error,
            0x1F7 | 0x3F6 => self.ide.read_status(),
            // VGA CRTC (colour addressing).
            0x3D4 => self.vga.crtc_index,
            0x3D5 => self.vga.read_crtc_data(),
            // Input status 1: bit 3 is the vertical-retrace flag and bit 0
            // the display-enable flag. Both are toggled so a driver polling
            // for retrace sees it happen instead of spinning.
            0x3DA => {
                self.vga_retrace = !self.vga_retrace;
                if self.vga_retrace { 0x09 } else { 0x00 }
            }
            // Port 0x61 (system control port B): speaker/gate control, the
            // refresh toggle, and PIT channel 2's output.
            0x61 => self.pit.read_port61(),
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
                let lo = self.port_in(port);
                let hi = self.port_in(port.wrapping_add(1));
                lo as u16 | ((hi as u16) << 8)
            }
        }
    }

    /// Write a byte to an I/O port.
    pub fn port_out(&mut self, port: u16, val: u8) {
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
            0x81..=0x8F => self.dma.write_page(port as u8, val),
            0x61 => self.pit.write_port61(val),
            // IDE command block (byte registers).
            0x1F1..=0x1F6 => self.ide.write_reg(port, val),
            0x1F7 => self.ide.write_command(val),
            0x3F6 => {} // device control: ignored
            // VGA CRTC.
            0x3D4 => self.vga.write_crtc_index(val),
            0x3D5 => self.vga.write_crtc_data(val),
            // CMOS RTC index/data.
            0x70 => self.cmos.write_index(val),
            0x71 => self.cmos.write_data(val),
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
                self.port_out(port, (val & 0xFF) as u8);
                self.port_out(port.wrapping_add(1), (val >> 8) as u8);
            }
        }
    }

    /// Deliver a pending hardware interrupt, if any. Returns true if an
    /// interrupt was dispatched.
    ///
    /// The PIT is ticked every call (so timing stays accurate), but the
    /// keyboard/IDE IRQ checks and PIC acknowledge are only done every
    /// `IRQ_CHECK_INTERVAL` instructions to reduce per-instruction overhead.
    pub fn deliver_hardware_interrupt(&mut self) -> bool {
        if self.servicing_irq {
            return false;
        }
        // Tick the PIT (channel 0 drives IRQ0) in batches to reduce
        // per-instruction overhead. We tick every IRQ_CHECK_INTERVAL
        // instructions, passing the accumulated count.
        const IRQ_CHECK_INTERVAL: u64 = 64;
        if self.instructions_executed % IRQ_CHECK_INTERVAL != 0 {
            return false;
        }
        self.pit.tick(IRQ_CHECK_INTERVAL);
        // Advance the wall clock alongside the PIT. One emulated second is
        // one PIT input period (1.193182 MHz), so the RTC keeps step with
        // whatever rate the guest programs the timer at.
        const PIT_HZ: u64 = 1_193_182;
        self.pit_subsecond += IRQ_CHECK_INTERVAL;
        while self.pit_subsecond >= PIT_HZ {
            self.pit_subsecond -= PIT_HZ;
            self.cmos.tick_second();
        }
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
        // A maskable interrupt is only *delivered* when IF is set. Without
        // this check the CPU would vector into a handler in the middle of a
        // kernel critical section -- or before the IDT is even installed.
        if !self.get_flag(crate::cpu::flags::IF) {
            return false;
        }
        if let Some(vector) = self.pic.acknowledge() {
            self.servicing_irq = true;
            self.irq_count += 1;
            self.irq_vectors[vector as usize] += 1;
            if self.pe {
                protected_int(self, vector);
            } else {
                // Real-mode interrupt through the IVT.
                let ip = self.ip;
                let cs = self.cs;
                let flags = self.flags as u16;
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
            self.invalidate_phys_ip();
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
        // (Batched: we only check for device IRQs every IRQ_CHECK_INTERVAL
        // instructions to avoid per-instruction overhead. The PIT is still
        // ticked each time so timing stays accurate.)
        // Dispatch any exception raised by the previous instruction (e.g. a
        // page fault recorded during address translation) BEFORE considering a
        // hardware interrupt. Taking the interrupt first would leave the fault
        // pending and then deliver it one instruction into the interrupt
        // handler, blaming the wrong address entirely.
        if let Some((vector, error_code)) = self.pending_exception.take() {
            self.dispatch_exception(vector, error_code);
        } else {
            self.deliver_hardware_interrupt();
        }
        // Invalidate the instruction-fetch cache at the start of each step.
        // The decoder's fetch calls will re-establish it.
        self.invalidate_phys_ip();
        self.eip_start = self.eip;
        self.ip_start = self.ip;
        if self.mem.watch_store.is_some() {
            self.mem.cur_eip = if self.pe { self.eip } else { self.ip as u32 };
        }
        if self.cpl() == 3 {
            self.user_instructions += 1;
            // X86EMU_TRAP_USER=<n>: stop just before the n'th user-mode
            // instruction, so the state at the ring-0 -> ring-3 handover can
            // be inspected.
            if let Some(n) = self.trap_user {
                if self.user_instructions == n {
                    self.halted = true;
                    self.trapped = true;
                }
            }
        }
        let inst = crate::instructions::decode(self);
        // A fault raised while *fetching* the instruction means there is no
        // instruction to run: the bytes the decoder saw are whatever happened
        // to sit at physical zero. Executing them anyway both corrupts state
        // and overwrites the real fault with a bogus one, which is precisely
        // what stops demand paging from working -- the first instruction of a
        // freshly exec'd program is always on a not-yet-present page.
        if self.pending_exception.is_none() {
            crate::instructions::execute(self, &inst);
        }
        self.instructions_executed += 1;
        self.tsc = self.tsc.wrapping_add(1);
        if self.debug_enabled {
            let eip = if self.pe { self.eip } else { self.ip as u32 };
            let pos = self.eip_ring_pos;
            self.eip_ring[pos] = eip;
            self.eip_ring_pos = (pos + 1) % EIP_RING;
        }
        if let Some(trap) = self.trap_eip {
            let eip = if self.pe { self.eip } else { self.ip as u32 };
            if eip == trap {
                self.halted = true;
                self.trapped = true;
            }
        }
        // Debug tracing: only when X86EMU_TRACE is set at startup. Uses a
        // cached file handle instead of opening/closing the file per
        // instruction.
        if self.trace_enabled && self.instructions_executed >= self.trace_from {
            use std::io::Write;
            let eip = if self.pe { self.eip } else { self.ip as u32 };
            let phys = self.phys_ip();
            let b0 = self.mem.read_u8(phys);
            let b1 = self.mem.read_u8((phys + 1) & (crate::memory::Memory::SIZE - 1));
            let b2 = self.mem.read_u8((phys + 2) & (crate::memory::Memory::SIZE - 1));
            let line = format!("[{}] cpl={} eip={:08X} bytes={:02X} {:02X} {:02X} eax={:08X} ecx={:08X} edx={:08X} ebx={:08X} esp={:08X} ebp={:08X} esi={:08X} edi={:08X}
",
                self.instructions_executed, self.cpl(), eip, b0, b1, b2,
                self.eax, self.ecx, self.edx, self.ebx, self.esp, self.ebp,
                self.esi, self.edi);
            if let Some(ref mut f) = self.trace_file {
                let _ = f.write_all(line.as_bytes());
            }
        }
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
        // Faults report the faulting instruction; traps report the next one.
        // #BP (INT3) and #OF (INTO) are traps -- they are raised *after* the
        // instruction completed and must not re-run it. Everything else here
        // is a fault, and the saved EIP has to point back at the instruction
        // so the handler can restart it (or, for a kernel exception-table
        // fixup, recognise the address at all).
        if !matches!(vector, 0x03 | 0x04) {
            self.eip = self.eip_start;
            self.ip = self.ip_start;
        }
        if (vector as usize) < 32 {
            self.exc_counts[vector as usize] += 1;
        }
        if self.debug_enabled && self.exc_log.len() < EXC_LOG_MAX {
            let eip = if self.pe { self.eip } else { self.ip as u32 };
            self.exc_log.push(
                (self.instructions_executed, vector, error_code, eip, self.cr2));
        }
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
        if self.pe {
            // The error code rides *inside* the frame builder, pushed after
            // EIP so it lands on top of the stack where the handler expects.
            crate::instructions::protected_int_err(self, vector, error_code);
        } else {
            // Real mode has no error code in the frame at all.
            let _ = error_code;
            // Real-mode exception through the IVT.
            let ip = self.ip;
            let cs = self.cs;
            let flags = self.flags as u16;
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
        self.invalidate_phys_ip();
    }

    /// Run until halted or `max` instructions have executed.
    pub fn run(&mut self, max: u64) -> u64 {
        let mut n = 0u64;
        while n < max {
            if self.halted {
                // HLT parks the CPU until an interrupt arrives. With IF clear
                // nothing can ever wake it (short of an NMI or reset, neither
                // of which this emulator raises), so that is a real, permanent
                // halt and the run ends. With IF set the idle loop is simply
                // waiting for the timer, so keep ticking the devices: this is
                // how the kernel's idle task gets its next tick.
                if self.triple_fault || self.trapped
                    || !self.get_flag(crate::cpu::flags::IF) {
                    break;
                }
                self.instructions_executed += 1;
                self.tsc = self.tsc.wrapping_add(1);
                if self.deliver_hardware_interrupt() {
                    self.halted = false;
                }
                n += 1;
                continue;
            }
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
        assert_eq!(cpu.ax(), 0x12AB);
        cpu.set_reg8(Reg8::Ah, 0xCD);
        assert_eq!(cpu.ax(), 0xCDAB);
    }

    #[test]
    fn stack_push_pop() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
        cpu.push16(0xCAFE);
        assert_eq!(cpu.sp(), 0x00FE);
        assert_eq!(cpu.pop16(), 0xCAFE);
        assert_eq!(cpu.sp(), 0x0100);
    }

    #[test]
    fn reg32_syncs_with_16bit() {
        let mut cpu = Cpu::new();
        cpu.set_reg32(Reg32::Eax, 0x12345678);
        assert_eq!(cpu.ax(), 0x5678);
        assert_eq!(cpu.reg16(Reg16::Ax), 0x5678);
        cpu.set_reg16(Reg16::Ax, 0xABCD);
        assert_eq!(cpu.eax, 0x1234ABCD);
    }

    #[test]
    fn hardware_interrupt_pit_to_pic_to_cpu() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
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
        // A maskable interrupt is only delivered with IF set.
        cpu.set_flag(flags::IF, true);
        // Run enough instructions for the PIT to wrap (count 1 -> 1 tick).
        cpu.run(8);
        // The handler ran: AX = 0x77.
        assert_eq!(cpu.ax(), 0x77);
        // The main hlt was reached (timer masked, so no re-entry).
        assert!(cpu.halted);
        // Stack restored after the interrupt frame.
        assert_eq!(cpu.sp(), 0x0100);
    }

    #[test]
    fn keyboard_irq1_delivered_through_pic() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
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
        cpu.set_flag(flags::IF, true);
        cpu.run(8);
        assert_eq!(cpu.ax(), 0x55);
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
