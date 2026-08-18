//! The x86 CPU core: registers, flags, and the fetch-decode-execute loop.
//!
//! Supports both 16-bit real mode and 32-bit protected mode. Registers are
//! kept as individual fields plus a packed flags word. The 16-bit general
//! registers map to byte registers (AL/AH, BL/BH, ...), and the 32-bit
//! registers are kept in sync with their 16-bit halves.

use crate::memory::Memory;
use crate::modrm::ModRm;
use crate::instructions::Inst;
use crate::instructions::protected_int;
use crate::protected::Descriptor;

/// Number of TLB entries (must be a power of two).
const TLB_SIZE: usize = 256;
/// Size of the debug EIP ring buffer (see `Cpu::eip_ring`).
pub const EIP_RING: usize = 4096;
/// X86EMU_NO_SPLIT (debug): treat a page-straddling access as if the next
/// physical page followed, the pre-fix behaviour, for A/B comparison.
static NO_SPLIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Cap on the number of exceptions recorded in `Cpu::exc_log`.
pub const EXC_LOG_MAX: usize = 512;
/// TLB index mask.
const TLB_MASK: usize = TLB_SIZE - 1;

/// A single TLB entry: caches a 4 KiB page mapping (linear page → physical
/// page). `valid` is the valid bit; the entry is invalidated on MOV CR3,
/// INVLPG, or any write that changes page tables.
///
/// The page numbers are 64-bit because both halves of the translation grew:
/// a long-mode linear address is 48 bits and a physical address may be 52.
#[derive(Clone, Copy)]
pub struct TlbEntry {
    pub valid: bool,
    /// The linear address with its low 12 bits dropped (virtual page number).
    pub vpage: u64,
    /// The physical address with its low 12 bits dropped.
    pub ppage: u64,
    /// Whether the mapping permits writes. Cached alongside the translation
    /// because a permission check that only ran on a TLB *miss* would let the
    /// second write to a read-only page through.
    pub writable: bool,
    /// Whether the mapping is reachable from user mode.
    pub user: bool,
    /// Whether instructions may be fetched from it (NX, when EFER.NXE is on).
    /// Cached for the same reason as `writable`.
    pub exec: bool,
    /// True once the accessed (and, for a write, dirty) bits have been set in
    /// the page tables for this entry, so the common case does not re-write
    /// them on every access.
    pub dirtied: bool,
}

impl Default for TlbEntry {
    fn default() -> Self {
        TlbEntry {
            valid: false, vpage: !0, ppage: 0,
            writable: false, user: false, exec: true, dirtied: false,
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

/// Bits of the EFER MSR.
pub mod efer {
    /// SYSCALL/SYSRET enable.
    pub const SCE: u64 = 1 << 0;
    /// Long Mode Enable: software asking for long mode.
    pub const LME: u64 = 1 << 8;
    /// Long Mode Active: the CPU saying it is in it. Set by the hardware when
    /// paging is enabled with LME set, cleared when paging goes away.
    pub const LMA: u64 = 1 << 10;
    /// No-Execute Enable: makes bit 63 of a page-table entry mean something.
    pub const NXE: u64 = 1 << 11;
}

/// The MSR numbers this CPU answers to.
pub mod msr {
    pub const SYSENTER_CS: u32 = 0x174;
    pub const SYSENTER_ESP: u32 = 0x175;
    pub const SYSENTER_EIP: u32 = 0x176;
    pub const EFER: u32 = 0xC000_0080;
    pub const STAR: u32 = 0xC000_0081;
    pub const LSTAR: u32 = 0xC000_0082;
    pub const CSTAR: u32 = 0xC000_0083;
    pub const SFMASK: u32 = 0xC000_0084;
    pub const FS_BASE: u32 = 0xC000_0100;
    pub const GS_BASE: u32 = 0xC000_0101;
    pub const KERNEL_GS_BASE: u32 = 0xC000_0102;
}

/// CR0.PE — protected mode enable.
pub const CR0_PE: u32 = 1 << 0;
/// CR0.WP — supervisor writes obey the read-only bit.
pub const CR0_WP: u32 = 1 << 16;
/// CR0.PG — paging enable.
pub const CR0_PG: u32 = 1 << 31;
/// CR4.PAE — physical address extension: 8-byte page-table entries.
pub const CR4_PAE: u32 = 1 << 5;
/// CR4.OSFXSR — the OS saves SSE state with FXSAVE; SSE instructions #UD
/// until it is set.
pub const CR4_OSFXSR: u32 = 1 << 9;
/// CR4.OSXMMEXCPT — the OS handles unmasked SSE exceptions (#XM).
pub const CR4_OSXMMEXCPT: u32 = 1 << 10;
/// CR4.VMXE — VMX enabled: VMXON is legal.
pub const CR4_VMXE: u32 = 1 << 13;
/// CR4.OSXSAVE — the OS uses XSAVE; XGETBV/XSETBV #UD until it is set.
pub const CR4_OSXSAVE: u32 = 1 << 18;

/// What the processor is running as. The four are not a spectrum: real and
/// protected mode are the legacy pair, and long mode has two sub-modes
/// selected per code segment by the descriptor L bit — 64-bit code, and
/// "compatibility" mode, which runs an unmodified 32-bit binary underneath a
/// 64-bit kernel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Real,
    Protected,
    Compat,
    Long,
}

pub struct Cpu {
    /// The sixteen general registers, stored once, 64 bits wide, in the order
    /// the ModR/M byte names them: RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    /// then the eight REX adds.
    ///
    /// EAX, AX, AL and AH are all *views* of `regs[0]`, reached through the
    /// accessors below -- never a second copy. Keeping both a 16- and a
    /// 32-bit set meant every wide write had to remember to refresh the
    /// narrow half, and the ones that forgot produced register corruption a
    /// long way from the instruction that caused it. The same argument
    /// applies again one width up, so the sixteen entries here are the whole
    /// register file.
    ///
    /// The width rules are x86-64's, and they are not symmetrical: a 32-bit
    /// write **zero-extends** into the full 64-bit register, while 16- and
    /// 8-bit writes leave the bits above them alone. In legacy modes nothing
    /// can observe the upper half, so the same rule serves both.
    pub regs: [u64; 16],
    // Segment registers.
    pub es: u16,
    pub cs: u16,
    pub ss: u16,
    pub ds: u16,
    pub fs: u16,
    pub gs: u16,
    // Instruction pointer (offset within CS).
    //
    // Real mode uses `ip`; protected and long mode use `rip`, of which `eip`
    // is the low half. They are alternatives, not copies -- only one of the
    // two is live in any given mode.
    pub ip: u16,
    pub rip: u64,
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
    /// True when the operand size for this instruction is 32 bits rather than
    /// 16 (the 0x66 prefix toggles it; the code segment decides the default).
    pub opsize: bool,
    /// True when addressing for this instruction is 32 bits rather than 16.
    /// In 64-bit mode this is also true, because 64-bit addressing uses the
    /// same ModR/M and SIB encoding -- `addr64` is what distinguishes them.
    pub addrsize: bool,
    /// True when addressing is 64-bit: long mode, unless a 0x67 prefix cut it
    /// back to 32.
    pub addr64: bool,
    /// Segment override for the current instruction, if any.
    pub seg_override: Option<SegReg>,

    // ---- REX prefix state for the current instruction ----
    //
    // Like the size prefixes, these are set during decode and stay readable
    // through execute, and are cleared at the *start* of the next decode.
    /// A REX prefix (0x40-0x4F) was present. On its own that changes nothing
    /// about operand size, but it does rename the byte registers: indices 4-7
    /// become SPL/BPL/SIL/DIL instead of AH/CH/DH/BH.
    pub rex_present: bool,
    /// REX.W: the operand size is 64 bits, overriding both the default and a
    /// 0x66 prefix.
    pub rex_w: bool,
    /// REX.R: the high bit of the ModR/M `reg` field.
    pub rex_r: bool,
    /// REX.X: the high bit of the SIB index field.
    pub rex_x: bool,
    /// REX.B: the high bit of the ModR/M `rm` field, of the SIB base, and of
    /// the register an opcode embeds in its low three bits.
    pub rex_b: bool,
    /// SSE mandatory prefix: which of 0x66/0xF3/0xF2 was the last legacy
    /// prefix before the opcode (after REX). SSE uses these as part of the
    /// opcode encoding (ps/pd/dq/ss/sd), not as operand-size or REP. `None`
    /// means no mandatory prefix was present.
    pub sse_pfx: Option<u8>,

    // ---- Protected-mode state ----
    /// True when protected mode is enabled (CR0.PE).
    pub pe: bool,
    /// GDT base and limit. The base is a 64-bit *linear* address: long
    /// mode's `LGDT` takes eight bytes of it, and a 64-bit kernel puts its
    /// tables in the high half of the address space.
    pub gdt_base: u64,
    pub gdt_limit: u16,
    /// IDT base and limit.
    pub idt_base: u64,
    pub idt_limit: u16,
    /// Task register: the selector loaded by LTR and the base/limit of the
    /// TSS it names. The TSS is what makes ring 3 workable at all -- an
    /// interrupt taken while the CPU is in user mode switches to the ring-0
    /// stack recorded in it (SS0 at offset 8, ESP0 at offset 4).
    pub tr_selector: u16,
    pub tr_base: u64,
    pub tr_limit: u32,
    /// Local descriptor table, loaded by LLDT. A selector with its TI bit set
    /// is resolved here rather than in the GDT.
    pub ldt_selector: u16,
    pub ldt_base: u64,
    pub ldt_limit: u32,
    /// Cached descriptors for ES, CS, SS, DS, FS, GS.
    pub seg_desc: [Descriptor; 6],

    // ---- Paging state ----
    /// Control registers. CR0 bit 31 = PG (paging enabled), bit 0 = PE
    /// (protected mode enabled). CR3 roots the paging structures, and is 64
    /// bits wide because PAE and long mode both put a 52-bit physical address
    /// in it. CR2 holds the faulting *linear* address, which in long mode is
    /// 64 bits.
    pub cr0: u32,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u32,
    /// CR8, the task-priority register. It exists only in long mode and only
    /// means something alongside a local APIC; with no APIC here it simply
    /// remembers what was written so a kernel reads back what it set.
    pub cr8: u64,
    /// The Extended Feature Enable Register (MSR 0xC0000080). LME asks for
    /// long mode, LMA says it is active, NXE turns the no-execute bit on and
    /// SCE enables SYSCALL/SYSRET. Long mode is *requested* by setting LME
    /// and *entered* by turning paging on, which is what sets LMA.
    pub efer: u64,
    /// SYSCALL/SYSRET configuration (MSRs 0xC0000081-84): the segment
    /// selectors, the 64-bit entry point, the compatibility-mode entry point,
    /// and the mask of RFLAGS bits SYSCALL clears.
    pub star: u64,
    pub lstar: u64,
    pub cstar: u64,
    pub sfmask: u64,
    /// FS and GS segment bases (MSRs 0xC0000100/0101). In 64-bit mode these
    /// are the only segment bases that still do anything, and they come from
    /// MSRs rather than descriptors so that a base can exceed 32 bits.
    pub fs_base: u64,
    pub gs_base: u64,
    /// The GS base SWAPGS parks (MSR 0xC0000102). A kernel entry swaps this
    /// with `gs_base` to reach its per-CPU data without trusting any user
    /// register.
    pub kernel_gs_base: u64,
    /// SYSENTER/SYSEXIT configuration (MSRs 0x174-0x176).
    pub sysenter_cs: u32,
    pub sysenter_esp: u32,
    pub sysenter_eip: u32,
    /// Debug registers DR0-DR7. Hardware breakpoints are not implemented;
    /// these exist so the kernel's startup writes and read-backs agree. They
    /// are 64 bits wide in long mode, where a breakpoint address can be.
    pub dr: [u64; 8],
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
    /// SSE/SSE2: the sixteen 128-bit XMM registers (XMM0–XMM15).
    pub xmm: [u128; 16],
    /// SSE/SSE2: MXCSR control/status register. Default: all exceptions
    /// masked, round-to-nearest, flush-to-zero off, denormals-are-zero off.
    pub mxcsr: u32,
    /// XCR0: which register state XSAVE manages. Bit 0 (x87) is always set;
    /// bit 1 (SSE) is the only other one this CPU has.
    pub xcr0: u64,
    /// VT-x state: VMXON, the current VMCS, and whether a guest is running.
    pub vmx: crate::vmx::Vmx,

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
    /// X86EMU_TRACE_USER: trace only ring-3 instructions, with R8-R15 too.
    /// A userspace bug is thousands of user instructions spread over
    /// millions of kernel ones; this is the trace that fits.
    pub trace_user: bool,
    /// X86EMU_TRACE_SYSCALLS: with `trace_user`, write only the `syscall`
    /// instructions and the instruction after each -- the arguments going in
    /// and RAX coming back. A userspace that fails is usually a syscall
    /// answering wrongly, and this is a few thousand lines where the full
    /// ring-3 trace is millions.
    pub trace_syscalls: bool,
    /// The last user line written was a `syscall`; write the next one too.
    trace_after_syscall: bool,

    // ---- Instruction fetch cache (#1) ----
    /// Cached physical address of the instruction stream. Valid between
    /// successive `fetch_u8` calls within a single instruction; invalidated
    /// at the start of each `step()` and whenever EIP/CS/page-mapping changes.
    pub phys_ip_cache: usize,
    /// Linear address corresponding to `phys_ip_cache` (for page-boundary checks).
    pub phys_ip_linear: u64,
    /// Mask applied to RIP as it advances: all ones in 64-bit mode, 32 bits
    /// of ones in every legacy mode. Recomputed once per instruction in
    /// `step`, so a fetch costs an AND instead of a test of the mode.
    pub rip_mask: u64,
    /// True when `phys_ip_cache` holds a valid mapping for the current EIP.
    pub phys_ip_valid: bool,

    /// EIP (or IP, in real mode) at the start of the instruction being
    /// executed. A *fault* reports the address of the instruction that
    /// faulted, not the one after it, so that the handler can fix things up
    /// and restart it -- which is exactly what demand paging and the kernel's
    /// exception table both rely on.
    pub rip_start: u64,
    pub ip_start: u16,

    // ---- Debug instrumentation (off unless X86EMU_DEBUG is set) ----
    /// True if X86EMU_DEBUG was set at startup. Gates the ring buffer and
    /// the exception log so a normal run pays nothing for them.
    pub debug_enabled: bool,
    /// Ring buffer of the most recent instruction pointers (linear RIP).
    /// Written every instruction when `debug_enabled`; dumped on demand.
    pub eip_ring: Vec<u64>,
    /// Write cursor into `eip_ring`.
    pub eip_ring_pos: usize,
    /// Log of dispatched exceptions: (instruction count, vector, error code,
    /// faulting RIP, CR2). Capped at `EXC_LOG_MAX` entries.
    pub exc_log: Vec<(u64, u8, Option<u32>, u64, u64)>,
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
    /// reaches this address. The RIP ring buffer then holds the instructions
    /// that led there — the way to find who jumped to a bad address.
    pub trap_eip: Option<u64>,
    /// Set once `trap_eip` has been reached, to end the run outright rather
    /// than parking as HLT does.
    pub trapped: bool,
    /// Stop before the n'th user-mode instruction (X86EMU_TRAP_USER).
    pub trap_user: Option<u64>,
    /// X86EMU_NO_TLB: bypass the TLB and walk the page tables on every
    /// access. Slow, but it settles whether a bug is a stale translation.
    pub no_tlb: bool,
    /// Linear address to watch for writes (X86EMU_WATCH), reported with the
    /// RIP that wrote it. Finding *who* wrote a wrong value is otherwise a
    /// matter of reading a million lines of trace.
    pub watch_linear: Option<u64>,
    /// Physical address to watch for writes (X86EMU_WATCH_PHYS). Catches
    /// stores made through *any* linear alias, including the kernel's direct
    /// map -- which a linear watch cannot see.
    pub watch_phys: Option<u64>,
    /// Log of writes to `watch_linear`: (instruction count, RIP, address).
    pub watch_log: Vec<(u64, u64, u64)>,
    /// Log of system calls made from user mode: (instruction count, RAX, RDI,
    /// RSI, RDX -- or EAX/EBX/ECX/EDX on a 32-bit guest). Recorded when
    /// X86EMU_DEBUG is set.
    pub syscall_log: Vec<(u64, u64, u64, u64, u64)>,
    /// Opcodes the decoder did not recognise, with a hit count and the RIP of
    /// the first sighting. Keyed by opcode (`0x0Fxx` for two-byte opcodes).
    /// Always recorded — an unimplemented instruction is a bug worth naming
    /// even in a release run, and the map only grows once per distinct opcode.
    pub unknown_ops: std::collections::BTreeMap<u16, (u64, u64)>,
}

impl Cpu {
    /// A machine with the default amount of RAM.
    pub fn new() -> Self {
        Self::with_ram(Memory::DEFAULT_SIZE)
    }

    /// A machine with `ram` bytes of RAM. Everything that reports the memory
    /// layout to the guest -- the BIOS map, `boot_params` -- reads it back off
    /// the `Memory`, so this is the only place a size is chosen.
    pub fn with_ram(ram: usize) -> Self {
        let trace_enabled = std::env::var("X86EMU_TRACE").is_ok();
        NO_SPLIT.store(std::env::var_os("X86EMU_NO_SPLIT").is_some(), std::sync::atomic::Ordering::Relaxed);
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
            regs: [0; 16],
            es: 0, cs: 0, ss: 0, ds: 0, fs: 0, gs: 0,
            ip: 0, rip: 0,
            flags: 0x0002, // bit 1 is always 1
            mem: Memory::with_size(ram),
            bios: crate::bios::Bios::new(),
            instructions_executed: 0,
            halted: false,
            triple_fault: false,
            tsc: 0,
            opsize: false,
            addrsize: false,
            addr64: false,
            seg_override: None,
            rex_present: false,
            rex_w: false,
            rex_r: false,
            rex_x: false,
            rex_b: false,
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
            cr8: 0,
            efer: 0,
            star: 0,
            lstar: 0,
            cstar: 0,
            sfmask: 0,
            fs_base: 0,
            gs_base: 0,
            kernel_gs_base: 0,
            sysenter_cs: 0,
            sysenter_esp: 0,
            sysenter_eip: 0,
            dr: [0; 8],
            pit: crate::pit::Pit::new(),
            pic: crate::pic::Pic::new(),
            vga: crate::vga::Vga::new(),
            kbd: crate::kbd::Kbd::new(),
            dma: crate::dma::Dma::new(),
            cmos: crate::cmos::Cmos::new(),
            ide: crate::ide::Ide::new(),
            fpu: crate::fpu::Fpu::new(),
            xmm: [0; 16],
            mxcsr: 0x1F80, // all exceptions masked, round-to-nearest
            xcr0: 1,
            vmx: crate::vmx::Vmx::new(),
            sse_pfx: None,
            pending_exception: None,
            tlb: [TlbEntry::default(); TLB_SIZE],
            trace_file,
            trace_enabled,
            trace_from: std::env::var("X86EMU_TRACE_FROM").ok()
                .and_then(|v| v.parse().ok()).unwrap_or(0),
            trace_user: std::env::var("X86EMU_TRACE_USER").is_ok(),
            trace_syscalls: std::env::var("X86EMU_TRACE_SYSCALLS").is_ok(),
            trace_after_syscall: false,
            phys_ip_cache: 0,
            phys_ip_linear: 0,
            rip_mask: 0xFFFF_FFFF,
            phys_ip_valid: false,
            rip_start: 0,
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
                .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()),
            watch_phys: std::env::var("X86EMU_WATCH_PHYS").ok()
                .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()),
            watch_log: Vec::new(),
            syscall_log: Vec::new(),
            trap_eip: std::env::var("X86EMU_TRAP_EIP").ok()
                .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()),
        }
    }

    /// Record an opcode the decoder could not handle, for diagnostics.
    /// The EIP stored is that of the *first* sighting (the instruction has
    /// already been fetched, so it points just past the opcode byte).
    pub fn note_unknown_opcode(&mut self, opcode: u16) {
        let eip = if self.pe { self.rip } else { self.ip as u64 };
        let e = self.unknown_ops.entry(opcode).or_insert((0, eip));
        e.0 += 1;
    }

    /// Physical address of the current instruction stream.
    #[inline]
    pub fn phys_ip(&mut self) -> usize {
        let linear = self.ip_linear();
        self.apply_paging((linear) as u64)
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
    ///
    /// In 64-bit mode CS has no base, so RIP *is* the linear address -- which
    /// is also why a 64-bit kernel can be linked at an address with the top
    /// sixteen bits set and simply run there.
    #[inline]
    fn ip_linear(&self) -> u64 {
        if self.long64() {
            self.rip
        } else if self.pe {
            (self.seg_desc[SegReg::Cs as usize].base as u64)
                .wrapping_add(self.rip & 0xFFFF_FFFF) & 0xFFFF_FFFF
        } else {
            (((self.cs as u64) << 4) + self.ip as u64) & 0xFFFF_FFFF
        }
    }

    // ---- The register file: one store, many widths ----
    //
    // Every accessor below reads or writes `regs`. There is no second copy of
    // anything, at any width.

    /// Read a general register as a full 64-bit value.
    #[inline]
    pub fn reg64(&self, i: u8) -> u64 { self.regs[(i & 15) as usize] }

    /// Write a 64-bit register as an instruction *result* (suppressed once
    /// the instruction has faulted -- see `set_reg32`).
    #[inline]
    pub fn set_reg64(&mut self, i: u8, v: u64) {
        if self.pending_exception.is_some() { return; }
        self.regs[(i & 15) as usize] = v;
    }

    /// Write a 64-bit register unconditionally.
    #[inline]
    pub fn set_reg64_raw(&mut self, i: u8, v: u64) {
        self.regs[(i & 15) as usize] = v;
    }

    /// Read a general register at `width` bits (16, 32 or 64), zero-extended.
    /// Byte registers go through `reg8_idx`, which has its own naming rule.
    #[inline]
    pub fn reg_w(&self, i: u8, width: u32) -> u64 {
        let v = self.regs[(i & 15) as usize];
        match width {
            64 => v,
            32 => v & 0xFFFF_FFFF,
            16 => v & 0xFFFF,
            _ => v & 0xFF,
        }
    }

    /// Write a general register at `width` bits.
    ///
    /// A 32-bit write **zero-extends** into the full register; 16- and 8-bit
    /// writes preserve everything above them. That asymmetry is x86-64s own,
    /// not a simplification: `mov $0,%eax` clears the whole of RAX while
    /// `mov $0,%ax` does not, and code generated for 64-bit relies on it.
    #[inline]
    pub fn set_reg_w(&mut self, i: u8, width: u32, v: u64) {
        if self.pending_exception.is_some() { return; }
        self.set_reg_w_raw(i, width, v);
    }

    /// `set_reg_w` without the fault suppression. For state that must be
    /// recorded *because* of a fault -- the string instructions index and
    /// count registers, which have to point at the element that faulted so
    /// the restart resumes there.
    #[inline]
    pub fn set_reg_w_raw(&mut self, i: u8, width: u32, v: u64) {
        let r = &mut self.regs[(i & 15) as usize];
        match width {
            64 => *r = v,
            32 => *r = v & 0xFFFF_FFFF,
            16 => *r = (*r & !0xFFFF) | (v & 0xFFFF),
            _ => *r = (*r & !0xFF) | (v & 0xFF),
        }
    }

    // ---- 16-bit register access ----

    /// The 16-bit view of `rax`.
    #[inline]
    pub fn ax(&self) -> u16 { self.regs[0] as u16 }
    /// Write the 16-bit view of `rax`, preserving everything above it.
    #[inline]
    pub fn set_ax(&mut self, v: u16) { self.set_reg_w_raw(0, 16, v as u64); }

    /// The 16-bit view of `rcx`.
    #[inline]
    pub fn cx(&self) -> u16 { self.regs[1] as u16 }
    /// Write the 16-bit view of `rcx`, preserving everything above it.
    #[inline]
    pub fn set_cx(&mut self, v: u16) { self.set_reg_w_raw(1, 16, v as u64); }

    /// The 16-bit view of `rdx`.
    #[inline]
    pub fn dx(&self) -> u16 { self.regs[2] as u16 }
    /// Write the 16-bit view of `rdx`, preserving everything above it.
    #[inline]
    pub fn set_dx(&mut self, v: u16) { self.set_reg_w_raw(2, 16, v as u64); }

    /// The 16-bit view of `rbx`.
    #[inline]
    pub fn bx(&self) -> u16 { self.regs[3] as u16 }
    /// Write the 16-bit view of `rbx`, preserving everything above it.
    #[inline]
    pub fn set_bx(&mut self, v: u16) { self.set_reg_w_raw(3, 16, v as u64); }

    /// The 16-bit view of `rsp`.
    #[inline]
    pub fn sp(&self) -> u16 { self.regs[4] as u16 }
    /// Write the 16-bit view of `rsp`, preserving everything above it.
    #[inline]
    pub fn set_sp(&mut self, v: u16) { self.set_reg_w_raw(4, 16, v as u64); }

    /// The 16-bit view of `rbp`.
    #[inline]
    pub fn bp(&self) -> u16 { self.regs[5] as u16 }
    /// Write the 16-bit view of `rbp`, preserving everything above it.
    #[inline]
    pub fn set_bp(&mut self, v: u16) { self.set_reg_w_raw(5, 16, v as u64); }

    /// The 16-bit view of `rsi`.
    #[inline]
    pub fn si(&self) -> u16 { self.regs[6] as u16 }
    /// Write the 16-bit view of `rsi`, preserving everything above it.
    #[inline]
    pub fn set_si(&mut self, v: u16) { self.set_reg_w_raw(6, 16, v as u64); }

    /// The 16-bit view of `rdi`.
    #[inline]
    pub fn di(&self) -> u16 { self.regs[7] as u16 }
    /// Write the 16-bit view of `rdi`, preserving everything above it.
    #[inline]
    pub fn set_di(&mut self, v: u16) { self.set_reg_w_raw(7, 16, v as u64); }

    /// The 16-bit register named by a ModR/M index (0-15).
    #[inline]
    pub fn reg16_idx(&self, i: u8) -> u16 { self.regs[(i & 15) as usize] as u16 }
    /// Write the 16-bit register named by a ModR/M index.
    #[inline]
    pub fn set_reg16_idx(&mut self, i: u8, v: u16) { self.set_reg_w(i, 16, v as u64); }

    pub fn reg16(&self, r: Reg16) -> u16 { self.reg16_idx(r as u8) }

    #[inline]
    pub fn set_reg16(&mut self, r: Reg16, v: u16) {
        self.set_reg_w(r as u8, 16, v as u64);
    }

    /// `set_reg16` without the fault suppression. See `set_reg_w_raw`.
    #[inline]
    pub fn set_reg16_raw(&mut self, r: Reg16, v: u16) {
        self.set_reg_w_raw(r as u8, 16, v as u64);
    }

    // ---- 32-bit register access ----

    /// The 32-bit register named by a ModR/M index (0-15).
    #[inline]
    pub fn reg32_idx(&self, i: u8) -> u32 { self.regs[(i & 15) as usize] as u32 }

    /// Write the 32-bit register named by a ModR/M index. Zero-extends.
    #[inline]
    pub fn set_reg32_idx(&mut self, i: u8, v: u32) { self.set_reg_w(i, 32, v as u64); }

    #[inline]
    pub fn reg32(&self, r: Reg32) -> u32 { self.reg32_idx(r as u8) }

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
        self.set_reg_w(r as u8, 32, v as u64);
    }

    /// Write a 32-bit register unconditionally. For state that must be
    /// recorded *because* of a fault -- the string instructions index and
    /// count registers, which have to point at the element that faulted so
    /// the restart resumes there.
    #[inline]
    pub fn set_reg32_raw(&mut self, r: Reg32, v: u32) {
        self.set_reg_w_raw(r as u8, 32, v as u64);
    }

    // ---- Accumulator and index shorthands ----
    //
    // These name the low 32 bits of the first eight registers, which is what
    // the BIOS handlers, the boot loader and the diagnostics all talk in.

    #[inline] pub fn eax(&self) -> u32 { self.regs[0] as u32 }
    #[inline] pub fn ecx(&self) -> u32 { self.regs[1] as u32 }
    #[inline] pub fn edx(&self) -> u32 { self.regs[2] as u32 }
    #[inline] pub fn ebx(&self) -> u32 { self.regs[3] as u32 }
    #[inline] pub fn esp(&self) -> u32 { self.regs[4] as u32 }
    #[inline] pub fn ebp(&self) -> u32 { self.regs[5] as u32 }
    #[inline] pub fn esi(&self) -> u32 { self.regs[6] as u32 }
    #[inline] pub fn edi(&self) -> u32 { self.regs[7] as u32 }

    #[inline] pub fn set_eax(&mut self, v: u32) { self.regs[0] = v as u64; }
    #[inline] pub fn set_ecx(&mut self, v: u32) { self.regs[1] = v as u64; }
    #[inline] pub fn set_edx(&mut self, v: u32) { self.regs[2] = v as u64; }
    #[inline] pub fn set_ebx(&mut self, v: u32) { self.regs[3] = v as u64; }
    #[inline] pub fn set_esp(&mut self, v: u32) { self.regs[4] = v as u64; }
    #[inline] pub fn set_ebp(&mut self, v: u32) { self.regs[5] = v as u64; }
    #[inline] pub fn set_esi(&mut self, v: u32) { self.regs[6] = v as u64; }
    #[inline] pub fn set_edi(&mut self, v: u32) { self.regs[7] = v as u64; }

    /// The stack pointer, full width.
    #[inline]
    pub fn rsp(&self) -> u64 { self.regs[4] }
    #[inline]
    pub fn set_rsp(&mut self, v: u64) { self.regs[4] = v; }

    /// The low 32 bits of RIP. Real mode uses `ip` instead.
    #[inline]
    pub fn eip(&self) -> u32 { self.rip as u32 }
    /// Set RIP from a 32-bit value, zero-extending as a 32-bit mode does.
    #[inline]
    pub fn set_eip(&mut self, v: u32) { self.rip = v as u64; }

    /// The low 32 bits of the address a faulting instruction reports.
    #[inline]
    pub fn eip_start(&self) -> u32 { self.rip_start as u32 }

    // ---- 8-bit register access ----

    /// Read the 8-bit register named by a ModR/M index.
    ///
    /// Without a REX prefix, indices 4-7 name AH/CH/DH/BH -- the *high* byte
    /// of the first four registers. With any REX prefix present they name
    /// SPL/BPL/SIL/DIL instead, and 8-15 name R8B-R15B. The switch is a
    /// property of the instruction, not of the register file, which is why it
    /// reads `rex_present` rather than the mode.
    #[inline]
    pub fn reg8_idx(&self, i: u8) -> u8 {
        if !self.rex_present && (4..8).contains(&i) {
            (self.regs[(i - 4) as usize] >> 8) as u8
        } else {
            self.regs[(i & 15) as usize] as u8
        }
    }

    /// Write the 8-bit register named by a ModR/M index.
    #[inline]
    pub fn set_reg8_idx(&mut self, i: u8, v: u8) {
        if self.pending_exception.is_some() { return; }
        if !self.rex_present && (4..8).contains(&i) {
            let r = &mut self.regs[(i - 4) as usize];
            *r = (*r & !0xFF00) | ((v as u64) << 8);
        } else {
            self.set_reg_w_raw(i, 8, v as u64);
        }
    }

    /// Read one of the eight legacy byte registers by name. Unlike
    /// `reg8_idx`, `Reg8::Ah` means AH whatever prefixes are in play -- the
    /// BIOS handlers name registers this way and never see a REX prefix.
    pub fn reg8(&self, r: Reg8) -> u8 {
        let i = r as u8;
        if i >= 4 {
            (self.regs[(i - 4) as usize] >> 8) as u8
        } else {
            self.regs[i as usize] as u8
        }
    }

    pub fn set_reg8(&mut self, r: Reg8, v: u8) {
        if self.pending_exception.is_some() { return; }
        let i = r as u8;
        if i >= 4 {
            let reg = &mut self.regs[(i - 4) as usize];
            *reg = (*reg & !0xFF00) | ((v as u64) << 8);
        } else {
            self.set_reg_w_raw(i, 8, v as u64);
        }
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

    /// Translate a linear address to a physical one *without* side effects:
    /// no TLB fill, no accessed/dirty bits, no page fault.
    ///
    /// The descriptor tables are named by **linear** address -- `LGDT` takes
    /// one -- and a kernel running in the higher half names them up there. So
    /// reading a descriptor means walking the same page tables the CPU walks,
    /// from a `&self` context where recording a fault is not possible anyway.
    /// This used to work by accident: masking every physical address into a
    /// 128 MiB store folded 0xC02C1000 onto 0x002C1000, which is exactly what
    /// the kernel's direct map maps it to. That coincidence dies the moment
    /// the machine has more RAM than the mask, which is the whole point here.
    ///
    /// An unmapped linear address yields `memory::UNBACKED`, which reads as an
    /// open bus rather than as whatever happens to live at address zero.
    pub fn linear_to_phys_ro(&self, linear: u64) -> usize {
        if self.cr0 & CR0_PG == 0 {
            return linear as usize;
        }
        let nxe = self.efer & efer::NXE != 0;
        match crate::paging::translate_mode(
            &self.mem, self.cr3, linear, self.paging_mode(), nxe,
        ) {
            Some(map) => map.phys as usize,
            None => crate::memory::UNBACKED,
        }
    }

    /// Resolve a selector to its descriptor, taking the table-indicator bit
    /// (bit 2) into account: set means the LDT, clear means the GDT.
    pub fn descriptor_for(&self, selector: u16) -> crate::protected::Descriptor {
        let idx = (selector >> 3) & 0x1FFF;
        let base = if selector & 4 != 0 { self.ldt_base } else { self.gdt_base };
        let entry = self.linear_to_phys_ro(base.wrapping_add((idx as u64) * 8));
        crate::protected::read_descriptor(&self.mem, entry)
    }

    /// Load the task register from a selector, caching the TSS base/limit.
    ///
    /// A long-mode TSS descriptor is *sixteen* bytes, not eight: the extra
    /// half carries the top 32 bits of the base. Reading only the first half
    /// gives a base that looks plausible and points nowhere, which shows up
    /// as a triple fault on the first interrupt taken from user mode.
    pub fn load_tr(&mut self, selector: u16) {
        self.tr_selector = selector;
        let d = self.descriptor_for(selector);
        self.tr_base = d.base as u64;
        self.tr_limit = d.limit;
        if self.long_mode() {
            let idx = (selector >> 3) & 0x1FFF;
            let hi_entry = self.gdt_base.wrapping_add((idx as u64) * 8 + 8);
            let hi = self.mem.read_u64(self.linear_to_phys_ro(hi_entry));
            self.tr_base |= (hi & 0xFFFF_FFFF) << 32;
        }
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
        let entry = self.linear_to_phys_ro(self.gdt_base.wrapping_add((idx as u64) * 8));
        let d = crate::protected::read_descriptor(&self.mem, entry);
        self.ldt_base = d.base as u64;
        self.ldt_limit = d.limit;
    }

    /// The ring-0 stack recorded in the TSS: (SS0, ESP0). The TSS base, like
    /// every descriptor-table base, is a linear address.
    pub fn tss_stack0(&self) -> (u16, u32) {
        let esp0 = self.mem.read_u32(self.linear_to_phys_ro(self.tr_base.wrapping_add(4)));
        let ss0 = self.mem.read_u16(self.linear_to_phys_ro(self.tr_base.wrapping_add(8)));
        (ss0, esp0)
    }

    /// The ring-0 stack pointer from a **64-bit** TSS (RSP0, at offset 4).
    ///
    /// The 64-bit TSS is a different structure from the 32-bit one, and it
    /// has no SS0 at all: a stack switch in long mode loads a null SS, and
    /// the stack pointer is the whole of the answer.
    pub fn tss_rsp0(&self) -> u64 {
        self.mem.read_u64(self.linear_to_phys_ro(self.tr_base.wrapping_add(4)))
    }

    /// One of the seven interrupt-stack-table pointers of a 64-bit TSS
    /// (`ist` is 1-7; 0 means "no IST", handled by the caller).
    ///
    /// The IST is what makes a fault that arrives on a broken stack
    /// survivable: the gate names a stack unconditionally rather than
    /// switching only on a privilege change, which is how a double fault or
    /// an NMI gets somewhere safe to land.
    pub fn tss_ist(&self, ist: u8) -> u64 {
        let off = 0x24 + (ist as u64 - 1) * 8;
        self.mem.read_u64(self.linear_to_phys_ro(self.tr_base.wrapping_add(off)))
    }

    // ---- Mode ----

    /// True once long mode is active (EFER.LMA). Compatibility-mode code runs
    /// with this set too — it is the *machine* that is in long mode.
    #[inline]
    pub fn long_mode(&self) -> bool { self.efer & efer::LMA != 0 }

    /// True when the current code segment is a 64-bit one: long mode active
    /// and the descriptor L bit set. This is the flag that decides operand
    /// and address sizes, register widths and stack width — not `long_mode`,
    /// because a 64-bit kernel runs 32-bit processes in compatibility mode
    /// with LMA still set.
    #[inline]
    pub fn long64(&self) -> bool {
        self.long_mode() && self.seg_desc[SegReg::Cs as usize].l
    }

    /// What the processor is running as.
    pub fn mode(&self) -> Mode {
        if !self.pe {
            Mode::Real
        } else if !self.long_mode() {
            Mode::Protected
        } else if self.seg_desc[SegReg::Cs as usize].l {
            Mode::Long
        } else {
            Mode::Compat
        }
    }

    /// Operand size in bits for the instruction being decoded or executed:
    /// 16, 32 or 64. REX.W wins over the 0x66 prefix, which is why it is
    /// tested first.
    #[inline]
    pub fn osize(&self) -> u32 {
        if self.rex_w { 64 } else if self.opsize { 32 } else { 16 }
    }

    /// Address size in bits for the current instruction: 16, 32 or 64.
    #[inline]
    pub fn asize(&self) -> u32 {
        if self.addr64 { 64 } else if self.addrsize { 32 } else { 16 }
    }

    /// The default stack width: 64 bits in 64-bit mode (and not overridable),
    /// otherwise the operand size.
    #[inline]
    pub fn stack_width(&self) -> u32 {
        if self.long64() { 64 } else if self.opsize { 32 } else { 16 }
    }

    /// Which paging structure CR3 currently roots.
    #[inline]
    pub fn paging_mode(&self) -> crate::paging::PagingMode {
        use crate::paging::PagingMode;
        if self.cr0 & CR0_PG == 0 {
            PagingMode::Off
        } else if self.long_mode() {
            PagingMode::Long
        } else if self.cr4 & CR4_PAE != 0 {
            PagingMode::Pae
        } else {
            PagingMode::Legacy
        }
    }

    /// Recompute EFER.LMA from CR0.PG and EFER.LME.
    ///
    /// Long mode is *asked for* by setting LME and *entered* by turning
    /// paging on — the CPU sets LMA itself, and every 64-bit boot sequence in
    /// existence depends on that handshake: set LME, load CR3, set PG, and
    /// the very next far jump lands in 64-bit code.
    pub fn update_long_mode(&mut self) {
        let want = self.cr0 & CR0_PG != 0 && self.efer & efer::LME != 0;
        let have = self.efer & efer::LMA != 0;
        if want != have {
            if want { self.efer |= efer::LMA; } else { self.efer &= !efer::LMA; }
            self.flush_tlb();
        }
    }

    // ---- Address translation ----

    /// The linear address a logical `seg:offset` names.
    ///
    /// In 64-bit mode segmentation is gone: CS, DS, ES and SS have a base of
    /// zero and no limit, and the offset *is* the linear address. FS and GS
    /// keep a base, because thread-local storage needs one — and it comes
    /// from an MSR rather than a descriptor, so that it can exceed 32 bits.
    #[inline]
    pub fn linear_addr(&self, s: SegReg, offset: u64) -> u64 {
        if self.long64() {
            match s {
                SegReg::Fs => self.fs_base.wrapping_add(offset),
                SegReg::Gs => self.gs_base.wrapping_add(offset),
                _ => offset,
            }
        } else if self.pe {
            (self.seg_desc[s as usize].base as u64).wrapping_add(offset) & 0xFFFF_FFFF
        } else {
            (((self.seg(s) as u64) << 4) + offset) & 0xFFFF_FFFF
        }
    }

    /// Translate a logical address through a segment to a physical address.
    /// Records a #PF (page fault) in `pending_exception` if paging is enabled
    /// and the page is not present.
    pub fn translate(&mut self, s: SegReg, offset: u64) -> usize {
        self.translate_access(s, (offset) as u64, false)
    }

    /// `translate` for a store. Separate from the read form so paging can tell
    /// a load from a store, which is what CR0.WP and the page-fault error code
    /// both turn on.
    pub fn translate_write(&mut self, s: SegReg, offset: u64) -> usize {
        self.translate_access(s, (offset) as u64, true)
    }

    fn translate_access(&mut self, s: SegReg, offset: u64, write: bool) -> usize {
        let linear = self.linear_addr(s, offset);
        self.apply_paging_access((linear) as u64, write)
    }

    /// True when an access of `bytes` starting at physical address `phys`
    /// crosses a page boundary. The offset within the page is the same for
    /// the physical and the linear address, so the physical one answers it.
    #[inline]
    pub fn straddles(phys: usize, bytes: u32) -> bool {
        if NO_SPLIT.load(std::sync::atomic::Ordering::Relaxed) { return false; }
        (phys & 0xFFF) as u32 + bytes > 0x1000
    }

    /// Read `bytes` (up to 16) of an operand that straddles a page: `phys`
    /// is the translation of `lin`, the first byte; the tail lives on the
    /// next page, which is translated here on its own. Two physically
    /// unrelated pages are the normal case in a vmalloc'd region, and a
    /// read that took the tail from `phys + n` was reading the wrong page --
    /// invisible in the direct map, where the pages ARE adjacent, and fatal
    /// in a module image, whose sections are copied with unaligned tails.
    pub fn read_split(&mut self, phys: usize, lin: u64, bytes: u32) -> u128 {
        let first = 0x1000 - (phys & 0xFFF);
        let phys2 = self.apply_paging_access((lin & !0xFFF).wrapping_add(0x1000), false);
        if self.pending_exception.is_some() { return 0; }
        let mut v = 0u128;
        for i in 0..bytes as usize {
            let b = if i < first { self.mem.read_u8(phys + i) } else { self.mem.read_u8(phys2 + i - first) };
            v |= (b as u128) << (8 * i);
        }
        v
    }

    /// The store side of `read_split`. Both pages are translated (and both
    /// checked for write permission) before a byte is written, so a fault on
    /// the second page commits nothing on the first.
    pub fn write_split(&mut self, phys: usize, lin: u64, bytes: u32, v: u128) {
        let first = 0x1000 - (phys & 0xFFF);
        let phys2 = self.apply_paging_access((lin & !0xFFF).wrapping_add(0x1000), true);
        if self.pending_exception.is_some() { return; }
        for i in 0..bytes as usize {
            let b = (v >> (8 * i)) as u8;
            if i < first { self.mem.write_u8(phys + i, b); } else { self.mem.write_u8(phys2 + i - first, b); }
        }
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
    pub fn invlpg(&mut self, linear: u64) {
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
    pub fn apply_paging(&mut self, linear: u64) -> usize {
        self.apply_paging_access((linear) as u64, false)
    }

    /// Translate a linear address, checking it against the access being made.
    ///
    /// `write` distinguishes a store from a load, which matters twice: a
    /// supervisor store to a read-only page faults when CR0.WP is set, and the
    /// page-fault error code has to say which kind of access faulted. The
    /// current privilege level comes from CS's RPL.
    pub fn apply_paging_access(&mut self, linear: u64, write: bool) -> usize {
        let phys = self.apply_paging_inner(linear, write, false);
        if write {
            if let Some(w) = self.watch_linear {
                // A store of any width that covers the watched address counts.
                // Any store within a few bytes either side: a 16-bit store
                // just above the watched address still changes the dword read
                // from it, and a window that only looked forwards missed
                // exactly that.
                let w = w as u64;
                if linear >= w.wrapping_sub(4) && linear <= w.wrapping_add(4) {
                    let eip = if self.pe { self.rip_start } else { self.ip_start as u64 };
                    let n = self.instructions_executed;
                    // Keep the most recent writes: the one that left the bad
                    // value is the last, not the first.
                    if self.watch_log.len() >= 64 {
                        self.watch_log.remove(0);
                    }
                    self.watch_log.push((n, eip, phys as u64));
                }
            }
            if let Some(w) = self.watch_phys {
                let p = phys as u64;
                let w = w as u64;
                if p >= w.wrapping_sub(4) && p <= w.wrapping_add(4) {
                    let eip = if self.pe { self.rip_start } else { self.ip_start as u64 };
                    let n = self.instructions_executed;
                    if self.watch_log.len() >= 64 { self.watch_log.remove(0); }
                    self.watch_log.push((n, eip, p));
                }
            }
        }
        phys
    }

    /// Translate a linear address for an instruction *fetch*. Separate from
    /// the read path because of NX: a page a kernel marked no-execute must
    /// still be readable, so only this path may fault on it.
    #[inline]
    pub fn apply_paging_fetch(&mut self, linear: u64) -> usize {
        self.apply_paging_inner(linear, false, true)
    }

    fn apply_paging_inner(&mut self, linear: u64, write: bool, fetch: bool) -> usize {
        // A 64-bit linear address that is not canonical never reaches the page
        // tables: it is a #GP, and the unused middle of the address space is
        // a hole rather than an alias of something.
        if self.long_mode() && !crate::paging::canonical(linear) {
            self.raise_gp(0);
            return crate::memory::UNBACKED;
        }
        if self.cr0 & CR0_PG == 0 {
            return linear as usize;
        }
        let user = self.cpl() == 3;
        // Fast path: check the TLB.
        let vpage = linear >> 12;
        let idx = (vpage as usize) & TLB_MASK;
        let entry = self.tlb[idx];
        if entry.valid && entry.vpage == vpage && !self.no_tlb {
            if self.access_allowed(entry.writable, entry.user, entry.exec, write, user, fetch) {
                if write && !entry.dirtied {
                    self.mark_accessed(linear, true);
                    self.tlb[idx].dirtied = true;
                }
                let offset = linear & 0xFFF;
                return ((entry.ppage << 12) | offset) as usize;
            }
            // Present, but the access is not permitted: a protection fault.
            self.raise_page_fault(linear, true, write, user, fetch);
            return crate::memory::UNBACKED;
        }
        // TLB miss: walk the page tables.
        let nxe = self.efer & efer::NXE != 0;
        let mode = self.paging_mode();
        match crate::paging::translate_mode(&self.mem, self.cr3, linear, mode, nxe) {
            Some(map) => {
                if !self.access_allowed(map.writable, map.user, map.exec, write, user, fetch) {
                    self.raise_page_fault(linear, true, write, user, fetch);
                    return crate::memory::UNBACKED;
                }
                // Fill the TLB entry. A large page already had its offset
                // folded in by the walk, so caching the page number works for
                // every page size.
                let ppage = map.phys >> 12;
                self.tlb[idx] = TlbEntry {
                    valid: true, vpage, ppage,
                    writable: map.writable, user: map.user, exec: map.exec,
                    dirtied: write,
                };
                self.set_accessed_bits(&map, write);
                map.phys as usize
            }
            None => {
                self.raise_page_fault(linear, false, write, user, fetch);
                crate::memory::UNBACKED
            }
        }
    }

    /// Offset an IDT entry points at, for diagnostics. Long mode's gates are
    /// sixteen bytes wide with the offset in three pieces.
    pub fn idt_target(&self, vector: u8) -> u64 {
        if self.long_mode() {
            let entry = self.idt_base.wrapping_add((vector as u64) * 16);
            let addr = self.linear_to_phys_ro(entry as u64);
            let lo = self.mem.read_u16(addr) as u64;
            let mid = self.mem.read_u16(addr + 6) as u64;
            let hi = self.mem.read_u32(addr + 8) as u64;
            lo | (mid << 16) | (hi << 32)
        } else {
            let entry = self.idt_base.wrapping_add((vector as u64) * 8);
            let addr = self.linear_to_phys_ro(entry as u64);
            let lo = self.mem.read_u16(addr) as u64;
            let hi = self.mem.read_u16(addr + 6) as u64;
            lo | (hi << 16)
        }
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
    /// working at all once user pages exist. An instruction *fetch* from a
    /// no-execute page fails for everyone, which is the point of NX.
    #[inline]
    fn access_allowed(
        &self, writable: bool, page_user: bool, page_exec: bool,
        write: bool, user: bool, fetch: bool,
    ) -> bool {
        if fetch && !page_exec { return false; }
        if user {
            if !page_user { return false; }
            if write && !writable { return false; }
            return true;
        }
        !(write && !writable && self.cr0 & CR0_WP != 0)
    }

    /// Raise a #GP with the given error code, unless the instruction has
    /// already faulted (the first fault is the one that describes what went
    /// wrong).
    pub fn raise_gp(&mut self, code: u32) {
        if self.pending_exception.is_some() { return; }
        self.pending_exception = Some((0x0D, Some(code)));
    }

    /// Record an invalid-opcode fault (`#UD`, no error code).
    pub fn raise_ud(&mut self) {
        if self.pending_exception.is_some() { return; }
        self.pending_exception = Some((0x06, None));
    }

    /// Write CR0 the way `MOV CR0` does: refresh `pe`, flush the TLB when
    /// PG toggles, and let the machine enter or leave long mode. `LMSW` and
    /// a VM entry/exit go through here too, so the side effects cannot be
    /// forgotten.
    pub fn write_cr0(&mut self, v: u32) {
        let old_pg = self.cr0 & CR0_PG != 0;
        self.cr0 = v;
        self.pe = v & CR0_PE != 0;
        if old_pg != (v & CR0_PG != 0) {
            self.flush_tlb();
        }
        self.update_long_mode();
    }

    /// Record a page fault: CR2 takes the faulting linear address and the
    /// error code says whether the page was present, whether the access was a
    /// write, whether it came from user mode, and whether it was an
    /// instruction fetch.
    fn raise_page_fault(
        &mut self, linear: u64, present: bool, write: bool, user: bool, fetch: bool,
    ) {
        // Keep the first fault: CR2 and the error code describe the access
        // that actually failed, and a second translation later in the same
        // instruction would otherwise rewrite them.
        if self.pending_exception.is_some() {
            return;
        }
        self.cr2 = linear;
        let mut code = (present as u32) | ((write as u32) << 1) | ((user as u32) << 2);
        // Bit 4 (I/D) only exists when NX does; without EFER.NXE a fetch
        // fault is reported as an ordinary read fault, as the hardware does.
        if fetch && self.efer & efer::NXE != 0 {
            code |= 1 << 4;
        }
        self.pending_exception = Some((0x0E, Some(code)));
    }

    /// Set the accessed (and for a store, dirty) bits of a mapping.
    ///
    /// Every level walked gets its accessed bit; only the leaf gets dirty,
    /// and only for a write. The entry width comes from the walk because the
    /// three paging structures do not agree on it.
    fn set_accessed_bits(&mut self, map: &crate::paging::Mapping, write: bool) {
        use crate::paging::pte;
        let n = map.walk_len as usize;
        for i in 0..n {
            let addr = map.walk[i] as usize;
            let leaf = i + 1 == n;
            let bits = pte::A | if leaf && write { pte::D } else { 0 };
            if map.entry_bytes == 4 {
                let e = self.mem.read_u32(addr);
                self.mem.write_u32(addr, e | bits as u32);
            } else {
                let e = self.mem.read_u64(addr);
                self.mem.write_u64(addr, e | bits);
            }
        }
    }

    /// Set the accessed/dirty bits for a linear address whose translation was
    /// served from the TLB (so the walk's entry addresses are not to hand).
    fn mark_accessed(&mut self, linear: u64, write: bool) {
        let nxe = self.efer & efer::NXE != 0;
        let mode = self.paging_mode();
        if let Some(map) = crate::paging::translate_mode(&self.mem, self.cr3, linear, mode, nxe) {
            self.set_accessed_bits(&map, write);
        }
    }


    // ---- Model-specific registers ----

    /// Read an MSR. Unknown ones read as zero rather than faulting: a kernel
    /// probes for features by reading them, and a #GP for every register this
    /// CPU has not heard of would turn feature detection into a crash.
    pub fn read_msr(&self, index: u32) -> u64 {
        match index {
            msr::EFER => self.efer,
            msr::STAR => self.star,
            msr::LSTAR => self.lstar,
            msr::CSTAR => self.cstar,
            msr::SFMASK => self.sfmask,
            msr::FS_BASE => self.fs_base,
            msr::GS_BASE => self.gs_base,
            msr::KERNEL_GS_BASE => self.kernel_gs_base,
            msr::SYSENTER_CS => self.sysenter_cs as u64,
            msr::SYSENTER_ESP => self.sysenter_esp as u64,
            msr::SYSENTER_EIP => self.sysenter_eip as u64,
            crate::vmx::msr::FEATURE_CONTROL => self.vmx.feature_control,
            _ => crate::vmx::read_capability_msr(index).unwrap_or(0),
        }
    }

    /// Write an MSR. Writes to registers this CPU does not model are dropped.
    pub fn write_msr(&mut self, index: u32, value: u64) {
        match index {
            msr::EFER => {
                // LMA is the CPU's to set, not software's: it is the answer to
                // "is long mode active", and it becomes true only when paging
                // is switched on with LME already set.
                self.efer = (value & !efer::LMA) | (self.efer & efer::LMA);
                self.update_long_mode();
            }
            msr::STAR => self.star = value,
            msr::LSTAR => self.lstar = value,
            msr::CSTAR => self.cstar = value,
            msr::SFMASK => self.sfmask = value,
            // Writing a segment base through an MSR is the only way to give
            // FS or GS a base above 4 GiB, and it takes effect immediately --
            // there is no descriptor to reload.
            msr::FS_BASE => self.fs_base = value,
            msr::GS_BASE => self.gs_base = value,
            msr::KERNEL_GS_BASE => self.kernel_gs_base = value,
            msr::SYSENTER_CS => self.sysenter_cs = value as u32,
            msr::SYSENTER_ESP => self.sysenter_esp = value as u32,
            msr::SYSENTER_EIP => self.sysenter_eip = value as u32,
            // IA32_FEATURE_CONTROL: writable until its lock bit is set,
            // which is what firmware does and what a kernel does in its
            // absence (Linux locks it with VMX enabled on every boot).
            crate::vmx::msr::FEATURE_CONTROL => {
                if self.vmx.feature_control & crate::vmx::FEAT_LOCKED != 0 {
                    self.raise_gp(0);
                } else {
                    self.vmx.feature_control = value & 0x7;
                }
            }
            _ => {}
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

    /// Ensure the phys_ip cache is valid for the current RIP. Called once
    /// at the start of an instruction (or after a page-boundary crossing).
    #[inline]
    fn ensure_phys_ip(&mut self) {
        if !self.phys_ip_valid {
            let linear = self.ip_linear();
            self.phys_ip_linear = linear;
            // A fetch, not a read: this is the path NX has to fault on.
            self.phys_ip_cache = self.apply_paging_fetch((linear) as u64);
            self.phys_ip_valid = true;
        }
    }

    /// Advance the instruction pointer by `n` bytes.
    ///
    /// `rip_mask` is recomputed once per instruction in `step`, so this costs
    /// one AND rather than a test of the mode on every byte fetched. In a
    /// legacy mode it wraps RIP at 4 GiB; in 64-bit mode it is all ones.
    #[inline]
    fn advance_ip(&mut self, n: u64) {
        if self.pe {
            self.rip = self.rip.wrapping_add(n) & self.rip_mask;
        } else {
            self.ip = self.ip.wrapping_add(n as u16);
        }
    }

    /// Peek at the next instruction byte without advancing RIP. Uses the
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
        self.advance_ip(1);
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
            self.advance_ip(2);
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

    /// Fetch a 64-bit immediate. Only `MOV r64, imm64` has one — every other
    /// 64-bit form takes an imm32 and sign-extends it.
    #[inline]
    pub fn fetch_u64(&mut self) -> u64 {
        let lo = self.fetch_u32() as u64;
        let hi = self.fetch_u32() as u64;
        lo | (hi << 32)
    }

    /// Read the ModR/M byte and decode it into a `ModRm` descriptor, fetching
    /// any SIB byte and displacement bytes it implies (based on `addrsize`).
    ///
    /// REX is folded in here: `reg` and (for a register operand) `rm` come out
    /// as 4-bit indices, while `rm_raw` keeps the 3-bit field that every
    /// *addressing* decision is made on. Extending `rm` before those tests
    /// would turn R12 into a SIB escape and R13 into a RIP-relative operand.
    pub fn fetch_modrm(&mut self) -> ModRm {
        let byte = self.fetch_u8();
        let mut modrm = ModRm::from_byte(byte);
        modrm.rm_raw = modrm.rm;
        modrm.rex_b = self.rex_b;
        modrm.rex_x = self.rex_x;
        modrm.reg |= (self.rex_r as u8) << 3;
        if modrm.mod_field == 3 {
            modrm.rm |= (self.rex_b as u8) << 3;
        }
        if self.addrsize {
            // 32-bit (and 64-bit) addressing: same ModR/M and SIB encoding.
            if modrm.mod_field != 3 && modrm.rm_raw == 4 {
                modrm.sib = Some(self.fetch_u8());
            }
            match modrm.mod_field {
                0 => {
                    // mod=00, rm=101 -> disp32; SIB base=101 -> disp32.
                    let sib_disp32 = modrm.sib.map(|s| s & 7 == 5).unwrap_or(false);
                    if modrm.rm_raw == 5 || sib_disp32 {
                        // In 64-bit mode the no-SIB form of this is not an
                        // absolute address at all: it is RIP-relative, which
                        // is how position-independent 64-bit code reaches its
                        // own data without a base register.
                        if modrm.rm_raw == 5 && self.addr64 {
                            modrm.rip_rel = true;
                        }
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
                    if modrm.rm_raw == 6 {
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
    /// memory operand.
    pub fn modrm_offset(&self, m: &ModRm) -> u32 {
        let base = match m.rm_raw {
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

    /// Compute the effective address of a memory operand encoded with the
    /// 32/64-bit ModR/M form, and the segment it defaults to.
    ///
    /// The address size decides the register width and whether the result is
    /// truncated: with a 0x67 prefix in 64-bit mode the whole computation is
    /// done and then cut to 32 bits, which is what makes `mov (%eax),%rbx`
    /// address the low 4 GiB rather than sign-extending into the high half.
    pub fn modrm_ea32(&self, m: &ModRm) -> (u64, SegReg) {
        // RIP-relative: the displacement is measured from the *end* of the
        // instruction, which is exactly where RIP already points by the time
        // the operand is used -- decode consumed every byte, immediates too.
        if m.rip_rel {
            let disp = m.disp32.unwrap_or(0) as i32 as i64 as u64;
            return (self.rip.wrapping_add(disp), SegReg::Ds);
        }
        let a64 = self.addr64;
        let mut ea: u64 = 0;
        let mut default_seg = SegReg::Ds;
        if let Some(sib) = m.sib {
            let scale = 1u64 << ((sib >> 6) & 3);
            let index_raw = (sib >> 3) & 7;
            let base_raw = sib & 7;
            // Index 100b means "no index", tested on the raw bits: with
            // REX.X the same encoding names R12, which is a real index.
            if index_raw != 4 || m.rex_x {
                let idx = index_raw | ((m.rex_x as u8) << 3);
                let v = if a64 { self.reg64(idx) } else { self.reg32_idx(idx) as u64 };
                ea = ea.wrapping_add(v.wrapping_mul(scale));
            }
            if !(m.mod_field == 0 && base_raw == 5) {
                let base = base_raw | ((m.rex_b as u8) << 3);
                let v = if a64 { self.reg64(base) } else { self.reg32_idx(base) as u64 };
                ea = ea.wrapping_add(v);
                // Only RSP and RBP as a *base* default to the stack segment.
                if base_raw == 4 || base_raw == 5 { default_seg = SegReg::Ss; }
            }
        } else if !(m.mod_field == 0 && m.rm_raw == 5) {
            // mod=00, rm=101 means disp32 with NO base register.
            let base = m.rm_raw | ((m.rex_b as u8) << 3);
            let v = if a64 { self.reg64(base) } else { self.reg32_idx(base) as u64 };
            ea = ea.wrapping_add(v);
            if m.rm_raw == 4 || m.rm_raw == 5 { default_seg = SegReg::Ss; }
        }
        if let Some(d32) = m.disp32 {
            ea = ea.wrapping_add(d32 as i32 as i64 as u64);
        }
        if !a64 { ea &= 0xFFFF_FFFF; }
        (ea, default_seg)
    }

    /// The linear address a ModR/M memory operand names, for INVLPG (which
    /// works on linear addresses, not physical ones).
    pub fn modrm_linear(&self, m: &ModRm) -> u64 {
        let (ea, default_seg) = if self.addrsize {
            self.modrm_ea32(m)
        } else {
            // 16-bit addressing: a BP-based operand (rm 2, 3, 6) defaults to
            // SS, exactly as `modrm_addr_access` decides it.
            let ss = matches!(m.rm_raw, 2 | 3 | 6);
            (self.modrm_offset(m) as u64, if ss { SegReg::Ss } else { SegReg::Ds })
        };
        let seg = self.operand_seg_for_exec(default_seg);
        self.linear_addr(seg, ea)
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

    /// The physical address of a ModR/M memory operand, at whatever address
    /// size the instruction uses. This is the one place that chooses between
    /// the 16-bit form and the 32/64-bit one.
    #[inline]
    pub fn rm_addr(&mut self, m: &ModRm, write: bool) -> usize {
        if self.addrsize {
            self.modrm_addr32_access(m, write)
        } else {
            self.modrm_addr_access(m, write)
        }
    }

    fn modrm_addr_access(&mut self, m: &ModRm, write: bool) -> usize {
        let (base, default_seg) = match m.rm_raw {
            0 => (self.bx().wrapping_add(self.si()), SegReg::Ds),
            1 => (self.bx().wrapping_add(self.di()), SegReg::Ds),
            2 => (self.bp().wrapping_add(self.si()), SegReg::Ss),
            3 => (self.bp().wrapping_add(self.di()), SegReg::Ss),
            4 => (self.si(), SegReg::Ds),
            5 => (self.di(), SegReg::Ds),
            6 => (self.bp(), SegReg::Ss),
            _ => (self.bx(), SegReg::Ds),
        };
        let mut ea = base as u32;
        if let Some(d8) = m.disp8 { ea = ea.wrapping_add(d8 as u32); }
        if let Some(d16) = m.disp16 { ea = ea.wrapping_add(d16 as u32); }
        let seg = self.operand_seg(default_seg);
        self.translate_access(seg, ea as u64, write)
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
        let (ea, default_seg) = self.modrm_ea32(m);
        let seg = self.operand_seg(default_seg);
        self.translate_access(seg, (ea) as u64, write)
    }

    // ---- ModR/M operands ----
    //
    // `read_rm_w` / `write_rm_w` are the implementation; the fixed-width
    // wrappers below exist because most call sites know their width outright.

    /// Read a ModR/M operand at `width` bits (8, 16, 32 or 64).
    #[inline]
    pub fn read_rm_w(&mut self, m: &ModRm, width: u32) -> u64 {
        if m.is_reg() {
            if width == 8 { self.reg8_idx(m.rm) as u64 } else { self.reg_w(m.rm, width) }
        } else {
            let addr = self.rm_addr(m, false);
            if width > 8 && Self::straddles(addr, width / 8) {
                if self.pending_exception.is_some() { return 0; }
                let lin = self.modrm_linear(m);
                return self.read_split(addr, lin, width / 8) as u64;
            }
            match width {
                64 => self.mem.read_u64(addr),
                32 => self.mem.read_u32(addr) as u64,
                16 => self.mem.read_u16(addr) as u64,
                _ => self.mem.read_u8(addr) as u64,
            }
        }
    }

    /// Write a ModR/M operand at `width` bits.
    #[inline]
    pub fn write_rm_w(&mut self, m: &ModRm, width: u32, val: u64) {
        if self.pending_exception.is_some() { return; }
        if m.is_reg() {
            if width == 8 { self.set_reg8_idx(m.rm, val as u8); }
            else { self.set_reg_w(m.rm, width, val); }
            return;
        }
        let addr = self.rm_addr(m, true);
        if self.pending_exception.is_some() { return; }
        if width > 8 && Self::straddles(addr, width / 8) {
            let lin = self.modrm_linear(m);
            self.write_split(addr, lin, width / 8, val as u128);
            return;
        }
        match width {
            64 => self.mem.write_u64(addr, val),
            32 => self.mem.write_u32(addr, val as u32),
            16 => self.mem.write_u16(addr, val as u16),
            _ => self.mem.write_u8(addr, val as u8),
        }
    }

    /// Read an 8-bit ModR/M operand.
    pub fn read_rm8(&mut self, m: &ModRm) -> u8 { self.read_rm_w(m, 8) as u8 }

    /// Write an 8-bit ModR/M operand.
    pub fn write_rm8(&mut self, m: &ModRm, val: u8) { self.write_rm_w(m, 8, val as u64) }

    /// Read a 16-bit ModR/M operand.
    pub fn read_rm16(&mut self, m: &ModRm) -> u16 { self.read_rm_w(m, 16) as u16 }

    /// Write a 16-bit ModR/M operand.
    pub fn write_rm16(&mut self, m: &ModRm, val: u16) { self.write_rm_w(m, 16, val as u64) }

    /// Read a 32-bit ModR/M operand.
    pub fn read_rm32(&mut self, m: &ModRm) -> u32 { self.read_rm_w(m, 32) as u32 }

    /// Write a 32-bit ModR/M operand.
    pub fn write_rm32(&mut self, m: &ModRm, val: u32) { self.write_rm_w(m, 32, val as u64) }

    /// Read a 64-bit ModR/M operand.
    pub fn read_rm64(&mut self, m: &ModRm) -> u64 { self.read_rm_w(m, 64) }

    /// Write a 64-bit ModR/M operand.
    pub fn write_rm64(&mut self, m: &ModRm, val: u64) { self.write_rm_w(m, 64, val) }

    // ---- Stack ----
    //
    // The stack pointer is as wide as the mode: SP in real mode, ESP in
    // protected mode, RSP in 64-bit mode -- where the width is also *not*
    // overridable, which is why 64-bit code has no way to push four bytes.

    /// Push `width` bits (16, 32 or 64) onto the stack.
    ///
    /// The address is translated *before* the stack pointer moves: a push
    /// whose stack page is not present must be restartable, and a decrement
    /// that survived the fault would push twice as far on the retry.
    pub fn push_w(&mut self, width: u32, val: u64) {
        let n = (width / 8) as u64;
        if self.pe {
            let wide = self.long64();
            let new_sp = if wide {
                self.rsp().wrapping_sub(n)
            } else {
                self.esp().wrapping_sub(n as u32) as u64
            };
            let addr = self.translate_write(SegReg::Ss, (new_sp) as u64);
            if self.pending_exception.is_some() { return; }
            if wide { self.set_rsp(new_sp); } else { self.set_esp(new_sp as u32); }
            match width {
                64 => self.mem.write_u64(addr, val),
                32 => self.mem.write_u32(addr, val as u32),
                _ => self.mem.write_u16(addr, val as u16),
            }
        } else {
            self.set_sp(self.sp().wrapping_sub(n as u16));
            let addr = Memory::phys(self.ss, self.sp());
            match width {
                64 => self.mem.write_u64(addr, val),
                32 => self.mem.write_u32(addr, val as u32),
                _ => self.mem.write_u16(addr, val as u16),
            }
        }
    }

    /// Pop `width` bits (16, 32 or 64) off the stack.
    pub fn pop_w(&mut self, width: u32) -> u64 {
        let n = (width / 8) as u64;
        if self.pe {
            let wide = self.long64();
            let sp = if wide { self.rsp() } else { self.esp() as u64 };
            let addr = self.translate(SegReg::Ss, (sp) as u64);
            if self.pending_exception.is_some() { return 0; }
            let v = match width {
                64 => self.mem.read_u64(addr),
                32 => self.mem.read_u32(addr) as u64,
                _ => self.mem.read_u16(addr) as u64,
            };
            if wide {
                self.set_rsp(sp.wrapping_add(n));
            } else {
                self.set_esp((sp as u32).wrapping_add(n as u32));
            }
            v
        } else {
            let addr = Memory::phys(self.ss, self.sp());
            let v = match width {
                64 => self.mem.read_u64(addr),
                32 => self.mem.read_u32(addr) as u64,
                _ => self.mem.read_u16(addr) as u64,
            };
            self.set_sp(self.sp().wrapping_add(n as u16));
            v
        }
    }

    pub fn push16(&mut self, val: u16) { self.push_w(16, val as u64) }

    pub fn pop16(&mut self) -> u16 { self.pop_w(16) as u16 }

    pub fn push32(&mut self, val: u32) { self.push_w(32, val as u64) }

    pub fn pop32(&mut self) -> u32 { self.pop_w(32) as u32 }

    pub fn push64(&mut self, val: u64) { self.push_w(64, val) }

    pub fn pop64(&mut self) -> u64 { self.pop_w(64) }


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
        // Tick the PIT (channel 0 drives IRQ0) in batches to reduce
        // per-instruction overhead. We tick every IRQ_CHECK_INTERVAL
        // instructions, passing the accumulated count.
        const IRQ_CHECK_INTERVAL: u64 = 64;
        if self.instructions_executed % IRQ_CHECK_INTERVAL != 0 {
            return false;
        }
        // The PIT's 1.193182 MHz input runs at INSTRUCTIONS_PER_PIT_CLOCK
        // emulated instructions per clock. The ratio is the machine's speed
        // as the guest experiences it: at 1:1 the CPU is a 1.2 MIPS machine,
        // and a kernel with a 250 Hz tick gets 4.8k instructions between
        // timer interrupts -- fewer than its timer handler and softirqs cost
        // it, so it lives inside the timer interrupt (BogoMIPS 0.01, jiffies
        // running away, a boot that never gets to userspace). At 16:1 it is
        // a 19 MIPS machine with ~76k instructions per tick, and every
        // time-based wait (msleep, calibration, RTC and device polls) costs
        // sixteen times the instructions it did.
        const INSTRUCTIONS_PER_PIT_CLOCK: u64 = 16;
        const PIT_CLOCKS: u64 = IRQ_CHECK_INTERVAL / INSTRUCTIONS_PER_PIT_CLOCK;
        self.pit.tick(PIT_CLOCKS);
        // Advance the wall clock alongside the PIT. One emulated second is
        // one PIT input period (1.193182 MHz), so the RTC keeps step with
        // whatever rate the guest programs the timer at.
        const PIT_HZ: u64 = 1_193_182;
        self.pit_subsecond += PIT_CLOCKS;
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
        // A guest whose hypervisor asked for external-interrupt exiting does
        // not take the interrupt: the hypervisor does.
        if self.vmx.in_guest && self.pic.has_pending() && crate::vmx::interrupt_exit(self) {
            return true;
        }
        if let Some(vector) = self.pic.acknowledge() {
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
        if self.vmx.in_guest {
            crate::vmx::pre_step(self);
        }
        if let Some((vector, error_code)) = self.pending_exception.take() {
            self.dispatch_exception(vector, error_code);
        } else {
            self.deliver_hardware_interrupt();
        }
        // Invalidate the instruction-fetch cache at the start of each step.
        // The decoder's fetch calls will re-establish it.
        self.invalidate_phys_ip();
        self.rip_start = self.rip;
        self.ip_start = self.ip;
        // The width RIP wraps at, decided once here rather than per byte
        // fetched. It cannot change under the instruction: only a far jump or
        // an IRET changes CS, and both end the instruction.
        self.rip_mask = if self.long64() { u64::MAX } else { 0xFFFF_FFFF };
        if self.mem.watch_store.is_some() {
            self.mem.cur_eip = if self.pe { self.rip } else { self.ip as u64 };
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
            // In a guest, an instruction the hypervisor asked to see exits
            // instead of executing (see `vmx::intercept`).
            if !(self.vmx.in_guest && crate::vmx::intercept(self, &inst)) {
                crate::instructions::execute(self, &inst);
            }
        }
        self.instructions_executed += 1;
        self.tsc = self.tsc.wrapping_add(1);
        if self.debug_enabled {
            let eip = if self.pe { self.rip } else { self.ip as u64 };
            let pos = self.eip_ring_pos;
            self.eip_ring[pos] = eip;
            self.eip_ring_pos = (pos + 1) % EIP_RING;
        }
        if let Some(trap) = self.trap_eip {
            let eip = if self.pe { self.rip } else { self.ip as u64 };
            if eip == trap {
                self.halted = true;
                self.trapped = true;
            }
        }
        // Debug tracing: only when X86EMU_TRACE is set at startup. Uses a
        // cached file handle instead of opening/closing the file per
        // instruction.
        if self.trace_enabled && self.instructions_executed >= self.trace_from
            && (!self.trace_user || self.cpl() == 3) {
            use std::io::Write;
            let eip = if self.pe { self.rip } else { self.ip as u64 };
            // Peek at the next instruction's bytes WITHOUT disturbing the
            // machine: the translation can fault (the next page is not
            // present, or a user CS is now looking at kernel text), and a
            // fault raised by the trace is a fault the untraced run never
            // took -- the two diverge, and the trace lies about the crash it
            // was meant to explain. Save and restore the fault state and CR2.
            let saved = (self.pending_exception, self.cr2);
            let phys = self.phys_ip();
            let (b0, b1, b2) = if self.pending_exception == saved.0 {
                (self.mem.read_u8(phys), self.mem.read_u8(phys.wrapping_add(1)),
                 self.mem.read_u8(phys.wrapping_add(2)))
            } else {
                (0xFF, 0xFF, 0xFF)
            };
            self.pending_exception = saved.0;
            self.cr2 = saved.1;
            if self.trace_user {
                if self.trace_syscalls {
                    let is_syscall = b0 == 0x0F && b1 == 0x05;
                    let want = is_syscall || self.trace_after_syscall;
                    self.trace_after_syscall = is_syscall;
                    if !want { return inst; }
                }
                let line = format!("[{}] rip={:016X} bytes={:02X} {:02X} {:02X} rax={:016X} rcx={:016X} rdx={:016X} rbx={:016X} rsp={:016X} rbp={:016X} rsi={:016X} rdi={:016X} r8={:016X} r9={:016X} r10={:016X} r11={:016X} r12={:016X} r13={:016X} r14={:016X} r15={:016X} fl={:08X}
",
                    self.instructions_executed, eip, b0, b1, b2,
                    self.regs[0], self.regs[1], self.regs[2], self.regs[3],
                    self.regs[4], self.regs[5], self.regs[6], self.regs[7],
                    self.regs[8], self.regs[9], self.regs[10], self.regs[11],
                    self.regs[12], self.regs[13], self.regs[14], self.regs[15],
                    self.flags);
                if let Some(ref mut f) = self.trace_file {
                    let _ = f.write_all(line.as_bytes());
                }
                return inst;
            }
            let line = format!("[{}] cpl={} rip={:016X} bytes={:02X} {:02X} {:02X} rax={:016X} rcx={:016X} rdx={:016X} rbx={:016X} rsp={:016X} rbp={:016X} rsi={:016X} rdi={:016X}
",
                self.instructions_executed, self.cpl(), eip, b0, b1, b2,
                self.regs[0], self.regs[1], self.regs[2], self.regs[3],
                self.regs[4], self.regs[5], self.regs[6], self.regs[7]);
            if let Some(ref mut f) = self.trace_file {
                let _ = f.write_all(line.as_bytes());
            }
        }
        inst
    }

    /// A triple fault: the machine resets. Here it halts, flagged -- unless a
    /// guest is running, in which case it is the hypervisor's to see.
    pub fn triple_fault(&mut self) {
        if self.vmx.in_guest {
            crate::vmx::triple_fault_exit(self);
            return;
        }
        self.triple_fault = true;
        self.halted = true;
    }

    /// Dispatch an exception through the IDT (protected mode) or IVT
    /// (real mode), pushing an error code first if the exception has one.
    ///
    /// If the IDT/IVT is not set up for this vector (e.g. an exception fires
    /// before the kernel installs its IDT), the CPU triple-faults: it halts
    /// with `triple_fault = true` instead of dispatching to a garbage entry
    /// and looping forever (as a real CPU would reset).
    pub fn dispatch_exception(&mut self, vector: u8, error_code: Option<u32>) {
        // In a guest, an exception the hypervisor's bitmap names is a VM
        // exit, not a delivery through the guest's IDT.
        if self.vmx.in_guest && vector < 32 {
            // The exit reports the faulting instruction, as the guest's
            // own handler would have seen it.
            if !matches!(vector, 0x03 | 0x04) {
                self.rip = self.rip_start;
            }
            if crate::vmx::exception_exit(self, vector, error_code) {
                return;
            }
        }
        self.dispatch_exception_raw(vector, error_code);
    }

    /// Deliver an exception or interrupt through the IDT/IVT with no VMX
    /// intercept: what `dispatch_exception` does once it has decided the
    /// event is the guest's own, and what event injection at VM entry uses.
    pub fn dispatch_exception_raw(&mut self, vector: u8, error_code: Option<u32>) {
        // Faults report the faulting instruction; traps report the next one.
        // #BP (INT3) and #OF (INTO) are traps -- they are raised *after* the
        // instruction completed and must not re-run it. Everything else here
        // is a fault, and the saved EIP has to point back at the instruction
        // so the handler can restart it (or, for a kernel exception-table
        // fixup, recognise the address at all).
        if !matches!(vector, 0x03 | 0x04) {
            self.rip = self.rip_start;
            self.ip = self.ip_start;
        }
        if (vector as usize) < 32 {
            self.exc_counts[vector as usize] += 1;
        }
        if self.debug_enabled && self.exc_log.len() < EXC_LOG_MAX {
            let eip = if self.pe { self.rip } else { self.ip as u64 };
            self.exc_log.push(
                (self.instructions_executed, vector, error_code, eip, self.cr2));
        }
        // Check the IDT/IVT covers this vector before dispatching. Long mode
        // doubles the gate size, so the same vector needs twice the table.
        if self.pe {
            let gate = if self.long_mode() { 16u32 } else { 8u32 };
            let entry = (vector as u32) * gate;
            if (entry + gate - 1) as u16 > self.idt_limit {
                self.triple_fault();
                return;
            }
        } else {
            let entry = (vector as usize) * 4;
            if entry + 3 > 0x3FF {
                self.triple_fault();
                return;
            }
        }
        if self.pe {
            // The error code rides *inside* the frame builder, pushed after
            // EIP so it lands on top of the stack where the handler expects.
            crate::instructions::protected_int_err(self, vector, error_code);
            // A fault raised while *delivering* a fault is not simply the
            // next fault. If both are contributory (or page faults) the CPU
            // escalates to a double fault, and a fault while delivering #DF
            // is a triple fault -- the machine resets. Without this a stack
            // that faults on the very first push re-delivers the same #PF
            // forever, one instruction of progress per attempt and none of
            // it the kernel's. A benign second exception (a #PF taken while
            // delivering an external interrupt, say) is left pending and
            // dispatches normally on its own.
            if let Some((second, _)) = self.pending_exception {
                if vector == 0x08 {
                    self.pending_exception = None;
                    self.triple_fault();
                    return;
                }
                let contributory = |v: u8| matches!(v, 0x00 | 0x0A | 0x0B | 0x0C | 0x0D);
                let escalate = (vector == 0x0E && (contributory(second) || second == 0x0E))
                    || (contributory(vector) && contributory(second));
                if escalate {
                    self.pending_exception = None;
                    self.dispatch_exception(0x08, Some(0));
                    return;
                }
            }
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
        assert_eq!(cpu.eax(), 0x1234ABCD);
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
        cpu.set_esp(0x0100);
        // No IDT installed (idt_base=0, idt_limit=0). A #DE fires.
        // mov ax, 1 ; mov bx, 0 ; div bx ; hlt
        cpu.mem.load(0x1000, &[
            0x66, 0xB8, 0x01, 0x00, 0x00, 0x00,
            0x66, 0xBB, 0x00, 0x00, 0x00, 0x00,
            0x66, 0xF7, 0xF3,
            0xF4,
        ]);
        cpu.cs = 0x08;
        cpu.set_eip(0x1000);
        cpu.run(32);
        // The #DE fired but there's no IDT -> triple fault -> halt.
        assert!(cpu.triple_fault);
        assert!(cpu.halted);
    }
}
