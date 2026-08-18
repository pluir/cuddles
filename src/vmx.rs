//! Intel VT-x (VMX): hardware virtualization.
//!
//! A hypervisor running on this CPU can put it into *VMX non-root
//! operation* -- run a guest -- and get control back on the events it asked
//! for. The mechanism is the same shape as on hardware, because a
//! hypervisor is written to that shape:
//!
//! - **VMXON** turns the feature on (CR4.VMXE, ring 0, IA32_FEATURE_CONTROL
//!   locked with VMX enabled) and names a 4 KiB region the CPU may use.
//! - The **VMCS** is a 4 KiB region in guest-physical memory holding the
//!   guest state, the host state, the execution controls and the exit
//!   information, addressed by 16-bit *field encodings* through VMREAD /
//!   VMWRITE. Its layout is opaque to software: hardware caches the *current*
//!   VMCS (VMPTRLD) and only VMCLEAR guarantees the memory copy is coherent.
//!   This CPU does exactly that: `Vmx::vmcs` is the working copy, written
//!   back on VMCLEAR and when another VMCS becomes current.
//! - **VMLAUNCH / VMRESUME** load the guest state from the current VMCS and
//!   run it. **A VM exit** saves the guest state back, records why, loads
//!   the host state, and continues at HOST_RIP -- as if VMLAUNCH/VMRESUME
//!   had just returned.
//! - Every VMX instruction reports through RFLAGS: all six arithmetic flags
//!   clear is *VMsucceed*; CF alone is *VMfailInvalid* (no current VMCS);
//!   ZF alone is *VMfailValid*, with the reason in the VM_INSTRUCTION_ERROR
//!   field.
//!
//! **What exits.** Unconditionally: CPUID, VMCALL, every VMX instruction,
//! XSETBV, INVD, a triple fault. Under a control bit: HLT, INVLPG, RDTSC,
//! MWAIT/MONITOR/PAUSE, MOV to/from CR3 and CR8, MOV DR, I/O (unconditional
//! or through the I/O bitmaps), RDMSR/WRMSR (unconditional or through the
//! MSR bitmaps), external interrupts, the interrupt window, and the
//! exceptions named in the exception bitmap. CR0 and CR4 have the guest/host
//! mask and read shadow: a guest write that touches a bit the host owns
//! exits, a read sees the shadow for those bits.
//!
//! **What is not here.** EPT (a guest's page tables are the CPU's page
//! tables: the hypervisor shadows them, which is how KVM ran before EPT
//! existed), unrestricted guest (the guest starts in protected mode --
//! CR0.PE and PG are fixed to 1, as `IA32_VMX_CR0_FIXED0` says), the
//! preemption timer, posted interrupts, APIC virtualization, SMM, nested
//! VMX. The capability MSRs advertise exactly what is implemented, so a
//! hypervisor that reads them (they all do) will not reach for what is
//! missing.
//!
//! **Where it hooks in.** `Cpu::step` calls `pre_step` (interrupt-window
//! exits) and `intercept` (instruction exits) when `in_guest`;
//! `dispatch_exception` calls `exception_exit` and `triple_fault_exit`;
//! `deliver_hardware_interrupt` calls `interrupt_exit`. None of them cost
//! anything when the CPU is not running a guest: each is one `bool` test.

use crate::cpu::{Cpu, SegReg, flags, CR4_VMXE, efer};
use crate::instructions::Inst;
use crate::modrm::ModRm;
use crate::protected::Descriptor;

// ---------------------------------------------------------------------------
// MSRs and CPUID
// ---------------------------------------------------------------------------

pub mod msr {
    pub const FEATURE_CONTROL: u32 = 0x3A;
    pub const VMX_BASIC: u32 = 0x480;
    pub const VMX_PINBASED_CTLS: u32 = 0x481;
    pub const VMX_PROCBASED_CTLS: u32 = 0x482;
    pub const VMX_EXIT_CTLS: u32 = 0x483;
    pub const VMX_ENTRY_CTLS: u32 = 0x484;
    pub const VMX_MISC: u32 = 0x485;
    pub const VMX_CR0_FIXED0: u32 = 0x486;
    pub const VMX_CR0_FIXED1: u32 = 0x487;
    pub const VMX_CR4_FIXED0: u32 = 0x488;
    pub const VMX_CR4_FIXED1: u32 = 0x489;
    pub const VMX_VMCS_ENUM: u32 = 0x48A;
    pub const VMX_PROCBASED_CTLS2: u32 = 0x48B;
    pub const VMX_EPT_VPID_CAP: u32 = 0x48C;
    pub const VMX_TRUE_PINBASED_CTLS: u32 = 0x48D;
    pub const VMX_TRUE_PROCBASED_CTLS: u32 = 0x48E;
    pub const VMX_TRUE_EXIT_CTLS: u32 = 0x48F;
    pub const VMX_TRUE_ENTRY_CTLS: u32 = 0x490;
    pub const VMX_VMFUNC: u32 = 0x491;
}

/// IA32_FEATURE_CONTROL bits.
pub const FEAT_LOCKED: u64 = 1 << 0;
pub const FEAT_VMX_OUTSIDE_SMX: u64 = 1 << 2;

/// The VMCS revision identifier this CPU writes and expects.
pub const VMCS_REVISION: u32 = 0x0001_2345;

/// Pin-based execution controls.
pub const PIN_EXT_INT_EXITING: u32 = 1 << 0;
pub const PIN_NMI_EXITING: u32 = 1 << 3;
/// Reserved-to-1 pin-based bits.
const PIN_DEFAULT1: u32 = 0x16;
const PIN_ALLOWED1: u32 = PIN_DEFAULT1 | PIN_EXT_INT_EXITING | PIN_NMI_EXITING;

/// Primary processor-based execution controls.
pub const CPU_INT_WINDOW_EXITING: u32 = 1 << 2;
pub const CPU_TSC_OFFSETTING: u32 = 1 << 3;
pub const CPU_HLT_EXITING: u32 = 1 << 7;
pub const CPU_INVLPG_EXITING: u32 = 1 << 9;
pub const CPU_MWAIT_EXITING: u32 = 1 << 10;
pub const CPU_RDPMC_EXITING: u32 = 1 << 11;
pub const CPU_RDTSC_EXITING: u32 = 1 << 12;
pub const CPU_CR3_LOAD_EXITING: u32 = 1 << 15;
pub const CPU_CR3_STORE_EXITING: u32 = 1 << 16;
pub const CPU_CR8_LOAD_EXITING: u32 = 1 << 19;
pub const CPU_CR8_STORE_EXITING: u32 = 1 << 20;
pub const CPU_MOV_DR_EXITING: u32 = 1 << 23;
pub const CPU_UNCOND_IO_EXITING: u32 = 1 << 24;
pub const CPU_USE_IO_BITMAPS: u32 = 1 << 25;
pub const CPU_USE_MSR_BITMAPS: u32 = 1 << 28;
pub const CPU_MONITOR_EXITING: u32 = 1 << 29;
pub const CPU_PAUSE_EXITING: u32 = 1 << 30;
pub const CPU_SECONDARY_CONTROLS: u32 = 1 << 31;
const CPU_DEFAULT1: u32 = 0x0401_E172;
const CPU_ALLOWED1: u32 = CPU_DEFAULT1 | CPU_INT_WINDOW_EXITING | CPU_TSC_OFFSETTING
    | CPU_HLT_EXITING | CPU_INVLPG_EXITING | CPU_MWAIT_EXITING | CPU_RDPMC_EXITING
    | CPU_RDTSC_EXITING | CPU_CR3_LOAD_EXITING | CPU_CR3_STORE_EXITING | CPU_CR8_LOAD_EXITING
    | CPU_CR8_STORE_EXITING | CPU_MOV_DR_EXITING | CPU_UNCOND_IO_EXITING | CPU_USE_IO_BITMAPS
    | CPU_USE_MSR_BITMAPS | CPU_MONITOR_EXITING | CPU_PAUSE_EXITING | CPU_SECONDARY_CONTROLS;

/// Secondary processor-based controls.
pub const CPU2_RDTSCP: u32 = 1 << 3;
pub const CPU2_VPID: u32 = 1 << 5;
pub const CPU2_WBINVD_EXITING: u32 = 1 << 6;
const CPU2_ALLOWED1: u32 = CPU2_RDTSCP | CPU2_VPID | CPU2_WBINVD_EXITING;

/// VM-exit controls.
pub const EXIT_HOST_ADDR_SPACE_SIZE: u32 = 1 << 9;
pub const EXIT_ACK_INT_ON_EXIT: u32 = 1 << 15;
pub const EXIT_SAVE_PAT: u32 = 1 << 18;
pub const EXIT_LOAD_PAT: u32 = 1 << 19;
pub const EXIT_SAVE_EFER: u32 = 1 << 20;
pub const EXIT_LOAD_EFER: u32 = 1 << 21;
const EXIT_DEFAULT1: u32 = 0x0003_6DFF;
const EXIT_ALLOWED1: u32 = EXIT_DEFAULT1 | EXIT_HOST_ADDR_SPACE_SIZE | EXIT_ACK_INT_ON_EXIT
    | EXIT_SAVE_PAT | EXIT_LOAD_PAT | EXIT_SAVE_EFER | EXIT_LOAD_EFER;

/// VM-entry controls.
pub const ENTRY_IA32E_GUEST: u32 = 1 << 9;
pub const ENTRY_LOAD_PAT: u32 = 1 << 14;
pub const ENTRY_LOAD_EFER: u32 = 1 << 15;
const ENTRY_DEFAULT1: u32 = 0x0000_11FF;
const ENTRY_ALLOWED1: u32 = ENTRY_DEFAULT1 | ENTRY_IA32E_GUEST | ENTRY_LOAD_PAT | ENTRY_LOAD_EFER;

/// CR0 bits a guest must have set (no unrestricted guest: PE and PG) and
/// CR4 bits likewise (VMXE).
pub const CR0_FIXED0: u64 = 0x8000_0021;
pub const CR0_FIXED1: u64 = 0xFFFF_FFFF;
pub const CR4_FIXED0: u64 = CR4_VMXE as u64;
pub const CR4_FIXED1: u64 = 0xFFFF_FFFF;

/// The capability MSRs, read-only. `allowed0` in the low half means "these
/// bits must be 1"; `allowed1` in the high half means "these bits may be 1".
pub fn read_capability_msr(index: u32) -> Option<u64> {
    let ctl = |allowed0: u32, allowed1: u32| ((allowed1 as u64) << 32) | allowed0 as u64;
    Some(match index {
        // Revision, 4 KiB VMCS, write-back memory type, INS/OUTS info, TRUE
        // controls available.
        msr::VMX_BASIC => VMCS_REVISION as u64 | (4096u64 << 32) | (6u64 << 50) | (1u64 << 54) | (1u64 << 55),
        msr::VMX_PINBASED_CTLS => ctl(PIN_DEFAULT1, PIN_ALLOWED1),
        msr::VMX_PROCBASED_CTLS => ctl(CPU_DEFAULT1, CPU_ALLOWED1),
        msr::VMX_EXIT_CTLS => ctl(EXIT_DEFAULT1, EXIT_ALLOWED1),
        msr::VMX_ENTRY_CTLS => ctl(ENTRY_DEFAULT1, ENTRY_ALLOWED1),
        // The TRUE MSRs let software clear the default-1 bits; this CPU has
        // no meaning for any of them, so they may all be 0 or 1.
        msr::VMX_TRUE_PINBASED_CTLS => ctl(0, PIN_ALLOWED1),
        msr::VMX_TRUE_PROCBASED_CTLS => ctl(0, CPU_ALLOWED1),
        msr::VMX_TRUE_EXIT_CTLS => ctl(0, EXIT_ALLOWED1),
        msr::VMX_TRUE_ENTRY_CTLS => ctl(0, ENTRY_ALLOWED1),
        // Preemption-timer rate 5, EFER.LMA stored on exit, HLT activity
        // state supported, 4 CR3-target values, 512-entry MSR lists.
        msr::VMX_MISC => 5 | (1 << 5) | (1 << 6) | (4 << 16),
        msr::VMX_CR0_FIXED0 => CR0_FIXED0,
        msr::VMX_CR0_FIXED1 => CR0_FIXED1,
        msr::VMX_CR4_FIXED0 => CR4_FIXED0,
        msr::VMX_CR4_FIXED1 => CR4_FIXED1,
        // Highest field index in use (bits 9:1).
        msr::VMX_VMCS_ENUM => (HIGHEST_INDEX as u64) << 1,
        msr::VMX_PROCBASED_CTLS2 => ctl(0, CPU2_ALLOWED1),
        // INVVPID: all four types.
        msr::VMX_EPT_VPID_CAP => (1u64 << 32) | (0xFu64 << 40),
        msr::VMX_VMFUNC => 0,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// VMCS field encodings
// ---------------------------------------------------------------------------

/// The fields this CPU knows. An encoding is: bit 0 = access type (1 = the
/// high half of a 64-bit field), bits 9:1 = index, bits 11:10 = type
/// (control, exit-info, guest, host), bits 14:13 = width (16, 64, 32,
/// natural).
pub mod field {
    // 16-bit control
    pub const VPID: u32 = 0x0000;
    // 16-bit guest state
    pub const GUEST_ES_SEL: u32 = 0x0800;
    pub const GUEST_CS_SEL: u32 = 0x0802;
    pub const GUEST_SS_SEL: u32 = 0x0804;
    pub const GUEST_DS_SEL: u32 = 0x0806;
    pub const GUEST_FS_SEL: u32 = 0x0808;
    pub const GUEST_GS_SEL: u32 = 0x080A;
    pub const GUEST_LDTR_SEL: u32 = 0x080C;
    pub const GUEST_TR_SEL: u32 = 0x080E;
    // 16-bit host state
    pub const HOST_ES_SEL: u32 = 0x0C00;
    pub const HOST_CS_SEL: u32 = 0x0C02;
    pub const HOST_SS_SEL: u32 = 0x0C04;
    pub const HOST_DS_SEL: u32 = 0x0C06;
    pub const HOST_FS_SEL: u32 = 0x0C08;
    pub const HOST_GS_SEL: u32 = 0x0C0A;
    pub const HOST_TR_SEL: u32 = 0x0C0C;
    // 64-bit control
    pub const IO_BITMAP_A: u32 = 0x2000;
    pub const IO_BITMAP_B: u32 = 0x2002;
    pub const MSR_BITMAP: u32 = 0x2004;
    pub const EXIT_MSR_STORE_ADDR: u32 = 0x2006;
    pub const EXIT_MSR_LOAD_ADDR: u32 = 0x2008;
    pub const ENTRY_MSR_LOAD_ADDR: u32 = 0x200A;
    pub const TSC_OFFSET: u32 = 0x2010;
    pub const GUEST_LINK_POINTER: u32 = 0x2800;
    pub const GUEST_DEBUGCTL: u32 = 0x2802;
    pub const GUEST_PAT: u32 = 0x2804;
    pub const GUEST_EFER: u32 = 0x2806;
    pub const HOST_PAT: u32 = 0x2C00;
    pub const HOST_EFER: u32 = 0x2C02;
    // 32-bit control
    pub const PIN_CTLS: u32 = 0x4000;
    pub const CPU_CTLS: u32 = 0x4002;
    pub const EXCEPTION_BITMAP: u32 = 0x4004;
    pub const PF_ERROR_MASK: u32 = 0x4006;
    pub const PF_ERROR_MATCH: u32 = 0x4008;
    pub const CR3_TARGET_COUNT: u32 = 0x400A;
    pub const EXIT_CTLS: u32 = 0x400C;
    pub const EXIT_MSR_STORE_COUNT: u32 = 0x400E;
    pub const EXIT_MSR_LOAD_COUNT: u32 = 0x4010;
    pub const ENTRY_CTLS: u32 = 0x4012;
    pub const ENTRY_MSR_LOAD_COUNT: u32 = 0x4014;
    pub const ENTRY_INTR_INFO: u32 = 0x4016;
    pub const ENTRY_EXCEPTION_ERROR: u32 = 0x4018;
    pub const ENTRY_INSTR_LEN: u32 = 0x401A;
    pub const CPU_CTLS2: u32 = 0x401E;
    // 32-bit read-only exit information
    pub const INSTRUCTION_ERROR: u32 = 0x4400;
    pub const EXIT_REASON: u32 = 0x4402;
    pub const EXIT_INTR_INFO: u32 = 0x4404;
    pub const EXIT_INTR_ERROR: u32 = 0x4406;
    pub const IDT_VECTORING_INFO: u32 = 0x4408;
    pub const IDT_VECTORING_ERROR: u32 = 0x440A;
    pub const EXIT_INSTR_LEN: u32 = 0x440C;
    pub const EXIT_INSTR_INFO: u32 = 0x440E;
    // 32-bit guest state
    pub const GUEST_ES_LIMIT: u32 = 0x4800;
    pub const GUEST_CS_LIMIT: u32 = 0x4802;
    pub const GUEST_SS_LIMIT: u32 = 0x4804;
    pub const GUEST_DS_LIMIT: u32 = 0x4806;
    pub const GUEST_FS_LIMIT: u32 = 0x4808;
    pub const GUEST_GS_LIMIT: u32 = 0x480A;
    pub const GUEST_LDTR_LIMIT: u32 = 0x480C;
    pub const GUEST_TR_LIMIT: u32 = 0x480E;
    pub const GUEST_GDTR_LIMIT: u32 = 0x4810;
    pub const GUEST_IDTR_LIMIT: u32 = 0x4812;
    pub const GUEST_ES_AR: u32 = 0x4814;
    pub const GUEST_CS_AR: u32 = 0x4816;
    pub const GUEST_SS_AR: u32 = 0x4818;
    pub const GUEST_DS_AR: u32 = 0x481A;
    pub const GUEST_FS_AR: u32 = 0x481C;
    pub const GUEST_GS_AR: u32 = 0x481E;
    pub const GUEST_LDTR_AR: u32 = 0x4820;
    pub const GUEST_TR_AR: u32 = 0x4822;
    pub const GUEST_INTERRUPTIBILITY: u32 = 0x4824;
    pub const GUEST_ACTIVITY: u32 = 0x4826;
    pub const GUEST_SYSENTER_CS: u32 = 0x482A;
    // 32-bit host state
    pub const HOST_SYSENTER_CS: u32 = 0x4C00;
    // natural-width control
    pub const CR0_GUEST_HOST_MASK: u32 = 0x6000;
    pub const CR4_GUEST_HOST_MASK: u32 = 0x6002;
    pub const CR0_READ_SHADOW: u32 = 0x6004;
    pub const CR4_READ_SHADOW: u32 = 0x6006;
    pub const CR3_TARGET0: u32 = 0x6008;
    // natural-width read-only exit information
    pub const EXIT_QUALIFICATION: u32 = 0x6400;
    pub const GUEST_LINEAR_ADDRESS: u32 = 0x640A;
    // natural-width guest state
    pub const GUEST_CR0: u32 = 0x6800;
    pub const GUEST_CR3: u32 = 0x6802;
    pub const GUEST_CR4: u32 = 0x6804;
    pub const GUEST_ES_BASE: u32 = 0x6806;
    pub const GUEST_CS_BASE: u32 = 0x6808;
    pub const GUEST_SS_BASE: u32 = 0x680A;
    pub const GUEST_DS_BASE: u32 = 0x680C;
    pub const GUEST_FS_BASE: u32 = 0x680E;
    pub const GUEST_GS_BASE: u32 = 0x6810;
    pub const GUEST_LDTR_BASE: u32 = 0x6812;
    pub const GUEST_TR_BASE: u32 = 0x6814;
    pub const GUEST_GDTR_BASE: u32 = 0x6816;
    pub const GUEST_IDTR_BASE: u32 = 0x6818;
    pub const GUEST_DR7: u32 = 0x681A;
    pub const GUEST_RSP: u32 = 0x681C;
    pub const GUEST_RIP: u32 = 0x681E;
    pub const GUEST_RFLAGS: u32 = 0x6820;
    pub const GUEST_PENDING_DBG: u32 = 0x6822;
    pub const GUEST_SYSENTER_ESP: u32 = 0x6824;
    pub const GUEST_SYSENTER_EIP: u32 = 0x6826;
    // natural-width host state
    pub const HOST_CR0: u32 = 0x6C00;
    pub const HOST_CR3: u32 = 0x6C02;
    pub const HOST_CR4: u32 = 0x6C04;
    pub const HOST_FS_BASE: u32 = 0x6C06;
    pub const HOST_GS_BASE: u32 = 0x6C08;
    pub const HOST_TR_BASE: u32 = 0x6C0A;
    pub const HOST_GDTR_BASE: u32 = 0x6C0C;
    pub const HOST_IDTR_BASE: u32 = 0x6C0E;
    pub const HOST_SYSENTER_ESP: u32 = 0x6C10;
    pub const HOST_SYSENTER_EIP: u32 = 0x6C12;
    pub const HOST_RSP: u32 = 0x6C14;
    pub const HOST_RIP: u32 = 0x6C16;
}

/// The highest field index (bits 9:1 of an encoding) this CPU accepts.
const HIGHEST_INDEX: u32 = 0x1A;

/// Slot in the working VMCS for an encoding: 4 widths x 4 types x 32
/// indices, one 64-bit slot each. `None` for an encoding this CPU does not
/// know (an index past `HIGHEST_INDEX`).
fn slot(enc: u32) -> Option<usize> {
    let index = (enc >> 1) & 0x1FF;
    if index > HIGHEST_INDEX || enc & !0x7FFF != 0 { return None; }
    let ty = (enc >> 10) & 3;
    let width = (enc >> 13) & 3;
    Some(((width * 4 + ty) * 32 + index) as usize)
}

/// Field width in bits, from the encoding.
fn width_of(enc: u32) -> u32 {
    match (enc >> 13) & 3 { 0 => 16, 1 => 64, 2 => 32, _ => 64 }
}

/// A read-only (exit information) field: type 1.
fn is_read_only(enc: u32) -> bool { (enc >> 10) & 3 == 1 }

/// The working-copy slot that records the launch state. It is not
/// addressable through any encoding (index 31 of the 16-bit control type).
const LAUNCH_SLOT: usize = 31;
/// The size of the working copy, in slots.
const SLOTS: usize = 512;

/// The VMX instruction error codes (VM_INSTRUCTION_ERROR on VMfailValid).
pub mod err {
    pub const VMCALL_IN_ROOT: u32 = 1;
    pub const VMCLEAR_BAD_ADDR: u32 = 2;
    pub const VMCLEAR_VMXON_PTR: u32 = 3;
    pub const VMLAUNCH_NOT_CLEAR: u32 = 4;
    pub const VMRESUME_NOT_LAUNCHED: u32 = 5;
    pub const ENTRY_INVALID_CONTROLS: u32 = 7;
    pub const ENTRY_INVALID_HOST_STATE: u32 = 8;
    pub const VMPTRLD_BAD_ADDR: u32 = 9;
    pub const VMPTRLD_VMXON_PTR: u32 = 10;
    pub const VMPTRLD_BAD_REVISION: u32 = 11;
    pub const VMREAD_WRITE_BAD_FIELD: u32 = 12;
    pub const VMWRITE_READ_ONLY: u32 = 13;
    pub const VMXON_IN_ROOT: u32 = 15;
    pub const INVALID_OPERAND: u32 = 28;
}

/// VM-exit basic reasons.
pub mod reason {
    pub const EXCEPTION_NMI: u32 = 0;
    pub const EXTERNAL_INTERRUPT: u32 = 1;
    pub const TRIPLE_FAULT: u32 = 2;
    pub const INTERRUPT_WINDOW: u32 = 7;
    pub const CPUID: u32 = 10;
    pub const HLT: u32 = 12;
    pub const INVD: u32 = 13;
    pub const INVLPG: u32 = 14;
    pub const RDPMC: u32 = 15;
    pub const RDTSC: u32 = 16;
    pub const VMCALL: u32 = 18;
    pub const VMCLEAR: u32 = 19;
    pub const VMLAUNCH: u32 = 20;
    pub const VMPTRLD: u32 = 21;
    pub const VMPTRST: u32 = 22;
    pub const VMREAD: u32 = 23;
    pub const VMRESUME: u32 = 24;
    pub const VMWRITE: u32 = 25;
    pub const VMXOFF: u32 = 26;
    pub const VMXON: u32 = 27;
    pub const CR_ACCESS: u32 = 28;
    pub const DR_ACCESS: u32 = 29;
    pub const IO_INSTRUCTION: u32 = 30;
    pub const RDMSR: u32 = 31;
    pub const WRMSR: u32 = 32;
    pub const ENTRY_INVALID_GUEST_STATE: u32 = 33;
    pub const MWAIT: u32 = 36;
    pub const MONITOR: u32 = 39;
    pub const PAUSE: u32 = 40;
    pub const INVEPT: u32 = 50;
    pub const RDTSCP: u32 = 51;
    pub const INVVPID: u32 = 53;
    pub const WBINVD: u32 = 54;
    pub const XSETBV: u32 = 55;
}

/// The VMX state on the CPU.
pub struct Vmx {
    /// IA32_FEATURE_CONTROL: bit 0 locks it, bit 2 permits VMXON.
    pub feature_control: u64,
    /// VMXON has been executed (VMX root operation, at least).
    pub on: bool,
    /// The VMXON region's physical address.
    pub vmxon_ptr: u64,
    /// The current VMCS's physical address, or `None` (VMPTRST reads
    /// `!0` then).
    pub current: Option<u64>,
    /// The working copy of the current VMCS.
    pub vmcs: Vec<u64>,
    /// The CPU is running a guest (VMX non-root operation).
    pub in_guest: bool,
    /// The address-space width of the host at the last VM entry (from the
    /// VM-exit "host address-space size" control), for the exit path.
    pub host_long: bool,
}

impl Vmx {
    pub fn new() -> Self {
        Vmx {
            feature_control: 0,
            on: false,
            vmxon_ptr: 0,
            current: None,
            vmcs: vec![0; SLOTS],
            in_guest: false,
            host_long: false,
        }
    }

    /// Read a field of the current VMCS (the caller checks there is one).
    pub fn read(&self, enc: u32) -> u64 {
        match slot(enc) {
            Some(s) => {
                let v = self.vmcs[s];
                if enc & 1 == 1 { v >> 32 } else { v & mask_w(width_of(enc)) }
            }
            None => 0,
        }
    }

    /// Write a field of the current VMCS.
    pub fn write(&mut self, enc: u32, v: u64) {
        if let Some(s) = slot(enc) {
            if enc & 1 == 1 {
                self.vmcs[s] = (self.vmcs[s] & 0xFFFF_FFFF) | (v << 32);
            } else {
                let w = width_of(enc);
                self.vmcs[s] = (self.vmcs[s] & !mask_w(w)) | (v & mask_w(w));
            }
        }
    }

    fn launched(&self) -> bool { self.vmcs[LAUNCH_SLOT] != 0 }
    fn set_launched(&mut self, on: bool) { self.vmcs[LAUNCH_SLOT] = on as u64; }
}

impl Default for Vmx {
    fn default() -> Self { Self::new() }
}

fn mask_w(width: u32) -> u64 {
    if width >= 64 { u64::MAX } else { (1u64 << width) - 1 }
}

// ---------------------------------------------------------------------------
// The instructions
// ---------------------------------------------------------------------------

/// The VMX instruction being executed. The memory operand (`m`) is a
/// 64-bit physical address for VMXON/VMCLEAR/VMPTRLD/VMPTRST; VMREAD and
/// VMWRITE take a field encoding in `reg` and a value in `m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmxOp {
    Vmxon,
    Vmxoff,
    Vmclear,
    Vmptrld,
    Vmptrst,
    Vmread,
    Vmwrite,
    Vmlaunch,
    Vmresume,
    Vmcall,
    Invept,
    Invvpid,
}

#[derive(Clone, Copy, Debug)]
pub struct VmxInst {
    pub op: VmxOp,
    pub m: ModRm,
    pub reg: u8,
}

/// Decode the `0F C7 /6` and `/7` memory forms: VMPTRLD (no prefix),
/// VMCLEAR (66), VMXON (F3), VMPTRST (/7).
pub fn decode_0f_c7(cpu: &Cpu, m: ModRm) -> Option<Inst> {
    if m.is_reg() { return None; }
    let op = match (m.reg & 7, cpu.sse_pfx) {
        (6, None) => VmxOp::Vmptrld,
        (6, Some(0x66)) => VmxOp::Vmclear,
        (6, Some(0xF3)) => VmxOp::Vmxon,
        (7, None) => VmxOp::Vmptrst,
        _ => return None,
    };
    Some(Inst::Vmx(VmxInst { op, m, reg: m.reg }))
}

/// Set RFLAGS for VMsucceed / VMfailInvalid / VMfailValid.
fn vm_succeed(cpu: &mut Cpu) {
    cpu.flags &= !(flags::CF | flags::PF | flags::AF | flags::ZF | flags::SF | flags::OF);
}
fn vm_fail_invalid(cpu: &mut Cpu) {
    vm_succeed(cpu);
    cpu.flags |= flags::CF;
}
fn vm_fail_valid(cpu: &mut Cpu, error: u32) {
    vm_succeed(cpu);
    cpu.flags |= flags::ZF;
    cpu.vmx.write(field::INSTRUCTION_ERROR, error as u64);
}

/// The 4 KiB region a pointer operand names must be 4 KiB aligned and
/// within the physical address width.
fn bad_region_ptr(p: u64) -> bool { p & 0xFFF != 0 || p >> 52 != 0 }

/// Read the 64-bit memory operand of VMXON/VMCLEAR/VMPTRLD.
fn read_ptr_operand(cpu: &mut Cpu, m: &ModRm) -> Option<u64> {
    if m.is_reg() { cpu.raise_ud(); return None; }
    let a = cpu.rm_addr(m, false);
    if cpu.pending_exception.is_some() { return None; }
    Some(cpu.mem.read_u64(a))
}

/// Write the working copy back to its region in memory.
fn flush_vmcs(cpu: &mut Cpu) {
    if let Some(p) = cpu.vmx.current {
        for (i, v) in cpu.vmx.vmcs.iter().enumerate() {
            cpu.mem.write_u64(p as usize + 8 + i * 8, *v);
        }
    }
}

/// Load a region into the working copy (revision id already checked).
fn load_vmcs(cpu: &mut Cpu, p: u64) {
    for i in 0..SLOTS {
        cpu.vmx.vmcs[i] = cpu.mem.read_u64(p as usize + 8 + i * 8);
    }
    cpu.vmx.current = Some(p);
}

/// Execute a VMX instruction in root operation. (In non-root operation
/// every one of them exits instead; see `intercept`.)
pub fn execute_vmx(cpu: &mut Cpu, v: &VmxInst) {
    // VMXON is legal without VMX on; everything else needs it, and #UDs
    // otherwise -- the same as an unknown opcode, which to a CPU without
    // the feature is what these are.
    if v.op != VmxOp::Vmxon && !cpu.vmx.on {
        cpu.raise_ud();
        return;
    }
    if cpu.cpl() != 0 { cpu.raise_gp(0); return; }
    match v.op {
        VmxOp::Vmxon => {
            if cpu.cr4 & CR4_VMXE == 0 { cpu.raise_ud(); return; }
            if cpu.vmx.on { vm_fail_valid(cpu, err::VMXON_IN_ROOT); return; }
            let fc = cpu.vmx.feature_control;
            if fc & FEAT_LOCKED == 0 || fc & FEAT_VMX_OUTSIDE_SMX == 0 { cpu.raise_gp(0); return; }
            // CR0 and CR4 must be within their fixed ranges to enter.
            if (cpu.cr0 as u64) & CR0_FIXED0 != CR0_FIXED0 { cpu.raise_gp(0); return; }
            let Some(p) = read_ptr_operand(cpu, &v.m) else { return };
            if bad_region_ptr(p) { vm_fail_invalid(cpu); return; }
            if cpu.mem.read_u32(p as usize) != VMCS_REVISION { vm_fail_invalid(cpu); return; }
            cpu.vmx.on = true;
            cpu.vmx.vmxon_ptr = p;
            cpu.vmx.current = None;
            vm_succeed(cpu);
        }
        VmxOp::Vmxoff => {
            flush_vmcs(cpu);
            cpu.vmx.on = false;
            cpu.vmx.current = None;
            vm_succeed(cpu);
        }
        VmxOp::Vmclear => {
            let Some(p) = read_ptr_operand(cpu, &v.m) else { return };
            if bad_region_ptr(p) { vm_fail_valid(cpu, err::VMCLEAR_BAD_ADDR); return; }
            if p == cpu.vmx.vmxon_ptr { vm_fail_valid(cpu, err::VMCLEAR_VMXON_PTR); return; }
            if cpu.vmx.current == Some(p) {
                cpu.vmx.set_launched(false);
                flush_vmcs(cpu);
                cpu.vmx.current = None;
            } else {
                // Clear the launch state in the memory copy directly.
                cpu.mem.write_u64(p as usize + 8 + LAUNCH_SLOT * 8, 0);
            }
            vm_succeed(cpu);
        }
        VmxOp::Vmptrld => {
            let Some(p) = read_ptr_operand(cpu, &v.m) else { return };
            if bad_region_ptr(p) { vm_fail_valid(cpu, err::VMPTRLD_BAD_ADDR); return; }
            if p == cpu.vmx.vmxon_ptr { vm_fail_valid(cpu, err::VMPTRLD_VMXON_PTR); return; }
            if cpu.mem.read_u32(p as usize) != VMCS_REVISION { vm_fail_valid(cpu, err::VMPTRLD_BAD_REVISION); return; }
            if cpu.vmx.current != Some(p) {
                flush_vmcs(cpu);
                load_vmcs(cpu, p);
            }
            vm_succeed(cpu);
        }
        VmxOp::Vmptrst => {
            if v.m.is_reg() { cpu.raise_ud(); return; }
            let a = cpu.rm_addr(&v.m, true);
            if cpu.pending_exception.is_some() { return; }
            let p = cpu.vmx.current.unwrap_or(u64::MAX);
            cpu.mem.write_u64(a, p);
            vm_succeed(cpu);
        }
        VmxOp::Vmread => {
            if cpu.vmx.current.is_none() { vm_fail_invalid(cpu); return; }
            let enc = cpu.reg_w(v.reg, 64) as u32;
            if slot(enc).is_none() { vm_fail_valid(cpu, err::VMREAD_WRITE_BAD_FIELD); return; }
            let val = cpu.vmx.read(enc);
            let w = if cpu.long64() { 64 } else { 32 };
            cpu.write_rm_w(&v.m, w, val);
            if cpu.pending_exception.is_some() { return; }
            vm_succeed(cpu);
        }
        VmxOp::Vmwrite => {
            if cpu.vmx.current.is_none() { vm_fail_invalid(cpu); return; }
            let enc = cpu.reg_w(v.reg, 64) as u32;
            if slot(enc).is_none() { vm_fail_valid(cpu, err::VMREAD_WRITE_BAD_FIELD); return; }
            if is_read_only(enc) { vm_fail_valid(cpu, err::VMWRITE_READ_ONLY); return; }
            let w = if cpu.long64() { 64 } else { 32 };
            let val = cpu.read_rm_w(&v.m, w);
            if cpu.pending_exception.is_some() { return; }
            cpu.vmx.write(enc, val);
            vm_succeed(cpu);
        }
        VmxOp::Vmlaunch | VmxOp::Vmresume => {
            if cpu.vmx.current.is_none() { vm_fail_invalid(cpu); return; }
            let launched = cpu.vmx.launched();
            if v.op == VmxOp::Vmlaunch && launched { vm_fail_valid(cpu, err::VMLAUNCH_NOT_CLEAR); return; }
            if v.op == VmxOp::Vmresume && !launched { vm_fail_valid(cpu, err::VMRESUME_NOT_LAUNCHED); return; }
            if let Err(e) = check_controls(cpu) { vm_fail_valid(cpu, e); return; }
            vm_entry(cpu);
        }
        VmxOp::Vmcall => {
            // In root operation VMCALL is for an SMM monitor; there is none.
            vm_fail_valid(cpu, err::VMCALL_IN_ROOT);
        }
        VmxOp::Invept | VmxOp::Invvpid => {
            // No EPT and no tagged TLB: every VM entry and exit flushes, so
            // there is nothing left to invalidate. Type-check the operand and
            // succeed.
            if v.m.is_reg() { cpu.raise_ud(); return; }
            let ty = cpu.reg_w(v.reg, 64);
            if v.op == VmxOp::Invept || ty > 3 { vm_fail_valid(cpu, err::INVALID_OPERAND); return; }
            cpu.flush_tlb();
            vm_succeed(cpu);
        }
    }
}

/// The checks a VM entry makes on the controls: every reserved-1 bit set,
/// no unsupported bit set, and a host state that can be entered.
fn check_controls(cpu: &Cpu) -> Result<(), u32> {
    let vm = &cpu.vmx;
    let pin = vm.read(field::PIN_CTLS) as u32;
    let cpu_ctl = vm.read(field::CPU_CTLS) as u32;
    let exit = vm.read(field::EXIT_CTLS) as u32;
    let entry = vm.read(field::ENTRY_CTLS) as u32;
    if pin & !PIN_ALLOWED1 != 0 || cpu_ctl & !CPU_ALLOWED1 != 0
        || exit & !EXIT_ALLOWED1 != 0 || entry & !ENTRY_ALLOWED1 != 0 {
        return Err(err::ENTRY_INVALID_CONTROLS);
    }
    if cpu_ctl & CPU_SECONDARY_CONTROLS != 0 && vm.read(field::CPU_CTLS2) as u32 & !CPU2_ALLOWED1 != 0 {
        return Err(err::ENTRY_INVALID_CONTROLS);
    }
    let host_cr0 = vm.read(field::HOST_CR0);
    let host_cr4 = vm.read(field::HOST_CR4);
    if host_cr0 & CR0_FIXED0 != CR0_FIXED0 || host_cr4 & CR4_FIXED0 != CR4_FIXED0 {
        return Err(err::ENTRY_INVALID_HOST_STATE);
    }
    // The host must be able to run at HOST_RIP: a 64-bit host needs a
    // 64-bit address space, and vice versa.
    let host_long = exit & EXIT_HOST_ADDR_SPACE_SIZE != 0;
    if !host_long && vm.read(field::HOST_RIP) >> 32 != 0 {
        return Err(err::ENTRY_INVALID_HOST_STATE);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Segment state in VMCS form
// ---------------------------------------------------------------------------

/// A segment as the VMCS describes it.
struct SegState { sel: u16, base: u64, limit: u32, ar: u32 }

/// Pack a cached descriptor into VMCS access-rights form: type/S/DPL/P in
/// bits 7:0, AVL 12, L 13, D/B 14, G 15, unusable 16.
fn ar_of(d: &Descriptor, usable: bool) -> u32 {
    let mut ar = d.attr as u32;
    if d.l { ar |= 1 << 13; }
    if d.d_b { ar |= 1 << 14; }
    if d.g { ar |= 1 << 15; }
    if !usable { ar |= 1 << 16; }
    ar
}

fn desc_of(base: u64, limit: u32, ar: u32) -> Descriptor {
    Descriptor {
        base: base as u32,
        limit,
        attr: (ar & 0xFF) as u8,
        g: ar & (1 << 15) != 0,
        d_b: ar & (1 << 14) != 0,
        l: ar & (1 << 13) != 0,
    }
}

const SEGS: [(SegReg, u32, u32, u32, u32); 6] = [
    (SegReg::Es, field::GUEST_ES_SEL, field::GUEST_ES_BASE, field::GUEST_ES_LIMIT, field::GUEST_ES_AR),
    (SegReg::Cs, field::GUEST_CS_SEL, field::GUEST_CS_BASE, field::GUEST_CS_LIMIT, field::GUEST_CS_AR),
    (SegReg::Ss, field::GUEST_SS_SEL, field::GUEST_SS_BASE, field::GUEST_SS_LIMIT, field::GUEST_SS_AR),
    (SegReg::Ds, field::GUEST_DS_SEL, field::GUEST_DS_BASE, field::GUEST_DS_LIMIT, field::GUEST_DS_AR),
    (SegReg::Fs, field::GUEST_FS_SEL, field::GUEST_FS_BASE, field::GUEST_FS_LIMIT, field::GUEST_FS_AR),
    (SegReg::Gs, field::GUEST_GS_SEL, field::GUEST_GS_BASE, field::GUEST_GS_LIMIT, field::GUEST_GS_AR),
];

/// Save the CPU's state into the guest-state area.
fn save_guest_state(cpu: &mut Cpu) {
    let mut w = |f: u32, v: u64| cpu.vmx.write(f, v);
    w(field::GUEST_CR0, cpu.cr0 as u64);
    w(field::GUEST_CR3, cpu.cr3);
    w(field::GUEST_CR4, cpu.cr4 as u64);
    w(field::GUEST_RIP, cpu.rip);
    w(field::GUEST_RSP, cpu.regs[4]);
    w(field::GUEST_RFLAGS, cpu.flags as u64);
    w(field::GUEST_DR7, cpu.dr[7]);
    w(field::GUEST_EFER, cpu.efer);
    w(field::GUEST_SYSENTER_CS, cpu.sysenter_cs as u64);
    w(field::GUEST_SYSENTER_ESP, cpu.sysenter_esp as u64);
    w(field::GUEST_SYSENTER_EIP, cpu.sysenter_eip as u64);
    w(field::GUEST_GDTR_BASE, cpu.gdt_base);
    w(field::GUEST_GDTR_LIMIT, cpu.gdt_limit as u64);
    w(field::GUEST_IDTR_BASE, cpu.idt_base);
    w(field::GUEST_IDTR_LIMIT, cpu.idt_limit as u64);
    w(field::GUEST_LDTR_SEL, cpu.ldt_selector as u64);
    w(field::GUEST_LDTR_BASE, cpu.ldt_base);
    w(field::GUEST_LDTR_LIMIT, cpu.ldt_limit as u64);
    w(field::GUEST_LDTR_AR, if cpu.ldt_selector & !3 == 0 { 1 << 16 } else { 0x82 });
    w(field::GUEST_TR_SEL, cpu.tr_selector as u64);
    w(field::GUEST_TR_BASE, cpu.tr_base);
    w(field::GUEST_TR_LIMIT, cpu.tr_limit as u64);
    w(field::GUEST_TR_AR, 0x8B);
    w(field::GUEST_ACTIVITY, if cpu.halted { 1 } else { 0 });
    w(field::GUEST_INTERRUPTIBILITY, 0);
    for (s, fsel, fbase, flimit, far) in SEGS {
        let d = cpu.seg_desc[s as usize];
        let base = match s {
            SegReg::Fs if cpu.long_mode() => cpu.fs_base,
            SegReg::Gs if cpu.long_mode() => cpu.gs_base,
            _ => d.base as u64,
        };
        let sel = cpu.seg(s);
        // A null selector outside CS/SS/TR is an unusable segment.
        let usable = s == SegReg::Cs || s == SegReg::Ss || sel & !3 != 0 || d.present();
        cpu.vmx.write(fsel, sel as u64);
        cpu.vmx.write(fbase, base);
        cpu.vmx.write(flimit, d.limit as u64);
        cpu.vmx.write(far, ar_of(&d, usable) as u64);
    }
}

/// Load the guest-state area into the CPU.
fn load_guest_state(cpu: &mut Cpu) {
    let r = |cpu: &Cpu, f: u32| cpu.vmx.read(f);
    let entry = r(cpu, field::ENTRY_CTLS) as u32;
    // CR0/CR4: the guest sees the read shadow for masked bits, but the
    // register itself holds the VMCS's GUEST_CR0 (the hypervisor sets both
    // consistently). CR0.PG/PE are fixed on.
    let cr0 = r(cpu, field::GUEST_CR0) | CR0_FIXED0;
    let cr4 = r(cpu, field::GUEST_CR4) | CR4_FIXED0;
    // EFER first, so update_long_mode sees LME when PG lands.
    let efer_v = if entry & ENTRY_LOAD_EFER != 0 {
        r(cpu, field::GUEST_EFER)
    } else if entry & ENTRY_IA32E_GUEST != 0 {
        cpu.efer | efer::LME
    } else {
        cpu.efer & !(efer::LME | efer::LMA)
    };
    cpu.efer = efer_v & !efer::LMA;
    cpu.cr4 = cr4 as u32;
    cpu.cr3 = r(cpu, field::GUEST_CR3);
    cpu.write_cr0(cr0 as u32);
    cpu.flush_tlb();
    cpu.rip = r(cpu, field::GUEST_RIP);
    cpu.regs[4] = r(cpu, field::GUEST_RSP);
    cpu.flags = (r(cpu, field::GUEST_RFLAGS) as u32 & flags::WRITABLE) | flags::ALWAYS_SET;
    cpu.dr[7] = r(cpu, field::GUEST_DR7);
    cpu.sysenter_cs = r(cpu, field::GUEST_SYSENTER_CS) as u32;
    cpu.sysenter_esp = r(cpu, field::GUEST_SYSENTER_ESP) as u32;
    cpu.sysenter_eip = r(cpu, field::GUEST_SYSENTER_EIP) as u32;
    cpu.gdt_base = r(cpu, field::GUEST_GDTR_BASE);
    cpu.gdt_limit = r(cpu, field::GUEST_GDTR_LIMIT) as u16;
    cpu.idt_base = r(cpu, field::GUEST_IDTR_BASE);
    cpu.idt_limit = r(cpu, field::GUEST_IDTR_LIMIT) as u16;
    cpu.ldt_selector = r(cpu, field::GUEST_LDTR_SEL) as u16;
    cpu.ldt_base = r(cpu, field::GUEST_LDTR_BASE);
    cpu.ldt_limit = r(cpu, field::GUEST_LDTR_LIMIT) as u32;
    cpu.tr_selector = r(cpu, field::GUEST_TR_SEL) as u16;
    cpu.tr_base = r(cpu, field::GUEST_TR_BASE);
    cpu.tr_limit = r(cpu, field::GUEST_TR_LIMIT) as u32;
    for (s, fsel, fbase, flimit, far) in SEGS {
        let st = SegState { sel: r(cpu, fsel) as u16, base: r(cpu, fbase), limit: r(cpu, flimit) as u32, ar: r(cpu, far) as u32 };
        cpu.set_seg(s, st.sel);
        cpu.seg_desc[s as usize] = desc_of(st.base, st.limit, st.ar);
        match s {
            SegReg::Fs => cpu.fs_base = st.base,
            SegReg::Gs => cpu.gs_base = st.base,
            _ => {}
        }
    }
    cpu.halted = r(cpu, field::GUEST_ACTIVITY) == 1;
    cpu.invalidate_phys_ip();
}

/// Load the host-state area into the CPU (a VM exit).
fn load_host_state(cpu: &mut Cpu) {
    let r = |cpu: &Cpu, f: u32| cpu.vmx.read(f);
    let exit = r(cpu, field::EXIT_CTLS) as u32;
    let host_long = exit & EXIT_HOST_ADDR_SPACE_SIZE != 0;
    let efer_v = if exit & EXIT_LOAD_EFER != 0 {
        r(cpu, field::HOST_EFER)
    } else if host_long {
        cpu.efer | efer::LME
    } else {
        cpu.efer & !(efer::LME | efer::LMA)
    };
    cpu.efer = efer_v & !efer::LMA;
    cpu.cr4 = r(cpu, field::HOST_CR4) as u32;
    cpu.cr3 = r(cpu, field::HOST_CR3);
    cpu.write_cr0(r(cpu, field::HOST_CR0) as u32);
    cpu.flush_tlb();
    cpu.rip = r(cpu, field::HOST_RIP);
    cpu.regs[4] = r(cpu, field::HOST_RSP);
    cpu.flags = flags::ALWAYS_SET;
    cpu.dr[7] = 0x400;
    cpu.sysenter_cs = r(cpu, field::HOST_SYSENTER_CS) as u32;
    cpu.sysenter_esp = r(cpu, field::HOST_SYSENTER_ESP) as u32;
    cpu.sysenter_eip = r(cpu, field::HOST_SYSENTER_EIP) as u32;
    cpu.gdt_base = r(cpu, field::HOST_GDTR_BASE);
    cpu.gdt_limit = 0xFFFF;
    cpu.idt_base = r(cpu, field::HOST_IDTR_BASE);
    cpu.idt_limit = 0xFFFF;
    cpu.ldt_selector = 0;
    cpu.ldt_base = 0;
    cpu.ldt_limit = 0;
    cpu.tr_selector = r(cpu, field::HOST_TR_SEL) as u16;
    cpu.tr_base = r(cpu, field::HOST_TR_BASE);
    cpu.tr_limit = 0x67;
    // The host segments are flat: CS is code, present, DPL 0, with L or D
    // per the host address-space size; the data segments are writable,
    // present, DPL 0, 4 GiB.
    let code = Descriptor { base: 0, limit: 0xFFFF_FFFF, attr: 0x9B, g: true, d_b: !host_long, l: host_long };
    let data = Descriptor { base: 0, limit: 0xFFFF_FFFF, attr: 0x93, g: true, d_b: true, l: false };
    let host_segs = [
        (SegReg::Cs, field::HOST_CS_SEL), (SegReg::Ss, field::HOST_SS_SEL), (SegReg::Ds, field::HOST_DS_SEL),
        (SegReg::Es, field::HOST_ES_SEL), (SegReg::Fs, field::HOST_FS_SEL), (SegReg::Gs, field::HOST_GS_SEL),
    ];
    for (s, fsel) in host_segs {
        cpu.set_seg(s, r(cpu, fsel) as u16);
        cpu.seg_desc[s as usize] = if s == SegReg::Cs { code } else { data };
    }
    cpu.fs_base = r(cpu, field::HOST_FS_BASE);
    cpu.gs_base = r(cpu, field::HOST_GS_BASE);
    cpu.halted = false;
    cpu.pending_exception = None;
    cpu.invalidate_phys_ip();
}

/// Enter the guest: load its state and, if the entry controls say so,
/// inject an event as the first thing it sees.
fn vm_entry(cpu: &mut Cpu) {
    let exit = cpu.vmx.read(field::EXIT_CTLS) as u32;
    cpu.vmx.host_long = exit & EXIT_HOST_ADDR_SPACE_SIZE != 0;
    load_guest_state(cpu);
    cpu.vmx.set_launched(true);
    cpu.vmx.in_guest = true;
    // Event injection: bit 31 valid, bits 7:0 vector, bits 10:8 type
    // (0 external interrupt, 2 NMI, 3 hardware exception, 4 software
    // interrupt, 6 software exception), bit 11 error code valid.
    let info = cpu.vmx.read(field::ENTRY_INTR_INFO) as u32;
    if info & (1 << 31) != 0 {
        let vector = (info & 0xFF) as u8;
        let ty = (info >> 8) & 7;
        let code = if info & (1 << 11) != 0 { Some(cpu.vmx.read(field::ENTRY_EXCEPTION_ERROR) as u32) } else { None };
        // The field is consumed by the entry.
        cpu.vmx.write(field::ENTRY_INTR_INFO, 0);
        // A software interrupt/exception's RIP is past the instruction.
        if ty == 4 || ty == 6 {
            let len = cpu.vmx.read(field::ENTRY_INSTR_LEN);
            cpu.rip = cpu.rip.wrapping_add(len);
        }
        cpu.rip_start = cpu.rip;
        cpu.ip_start = cpu.rip as u16;
        cpu.halted = false;
        // Deliver through the guest's IDT. Trap-class vectors report the
        // next instruction, faults the current one; both are RIP here.
        cpu.dispatch_exception_raw(vector, code);
    }
}

/// Leave the guest: record the reason, save its state, load the host's.
/// `qualification` is the exit qualification; `intr_info` is for exception
/// and interrupt exits (0 when none).
pub fn vm_exit(cpu: &mut Cpu, reason: u32, qualification: u64, intr_info: u32, intr_error: u32) {
    // The instruction length is what decode advanced RIP by; the guest RIP
    // saved is the start of that instruction, so the hypervisor's "skip
    // this instruction" is RIP + length. For an interrupt arriving between
    // instructions the two are equal and the length is 0.
    let len = cpu.rip.wrapping_sub(cpu.rip_start);
    let (guest_rip, len) = if reason == reason::EXTERNAL_INTERRUPT || reason == reason::INTERRUPT_WINDOW
        || reason == reason::TRIPLE_FAULT {
        (cpu.rip, 0)
    } else {
        (cpu.rip_start, len)
    };
    let saved_rip = cpu.rip;
    cpu.rip = guest_rip;
    save_guest_state(cpu);
    cpu.rip = saved_rip;
    cpu.vmx.write(field::EXIT_REASON, reason as u64);
    cpu.vmx.write(field::EXIT_QUALIFICATION, qualification);
    cpu.vmx.write(field::EXIT_INSTR_LEN, len);
    cpu.vmx.write(field::EXIT_INTR_INFO, intr_info as u64);
    cpu.vmx.write(field::EXIT_INTR_ERROR, intr_error as u64);
    cpu.vmx.write(field::IDT_VECTORING_INFO, 0);
    if reason == reason::EXCEPTION_NMI && (intr_info & 0xFF) == 0x0E {
        cpu.vmx.write(field::GUEST_LINEAR_ADDRESS, cpu.cr2);
    }
    // MSR save/load lists are not implemented (the counts are 0 for the
    // hypervisors this CPU targets); EFER is saved when asked.
    if cpu.vmx.read(field::EXIT_CTLS) as u32 & EXIT_SAVE_EFER != 0 {
        cpu.vmx.write(field::GUEST_EFER, cpu.efer);
    }
    cpu.vmx.in_guest = false;
    load_host_state(cpu);
}

// ---------------------------------------------------------------------------
// The hooks
// ---------------------------------------------------------------------------

/// Called by `Cpu::step` before dispatching anything, in the guest: the
/// interrupt-window exit fires as soon as the guest can take an interrupt.
pub fn pre_step(cpu: &mut Cpu) {
    let ctl = cpu.vmx.read(field::CPU_CTLS) as u32;
    if ctl & CPU_INT_WINDOW_EXITING != 0 && cpu.get_flag(flags::IF) && cpu.pending_exception.is_none() {
        vm_exit(cpu, reason::INTERRUPT_WINDOW, 0, 0, 0);
    }
}

/// Called by `deliver_hardware_interrupt` in the guest with an unmasked
/// interrupt pending. Returns true when it exited (and, with "acknowledge
/// interrupt on exit", the vector is in the exit interruption info).
pub fn interrupt_exit(cpu: &mut Cpu) -> bool {
    let pin = cpu.vmx.read(field::PIN_CTLS) as u32;
    if pin & PIN_EXT_INT_EXITING == 0 { return false; }
    let exit = cpu.vmx.read(field::EXIT_CTLS) as u32;
    let info = if exit & EXIT_ACK_INT_ON_EXIT != 0 {
        match cpu.pic.acknowledge() {
            Some(v) => (1u32 << 31) | v as u32,
            None => 0,
        }
    } else { 0 };
    vm_exit(cpu, reason::EXTERNAL_INTERRUPT, 0, info, 0);
    true
}

/// Called by `dispatch_exception` in the guest. Returns true when the
/// exception bitmap routes this vector to the hypervisor.
pub fn exception_exit(cpu: &mut Cpu, vector: u8, error_code: Option<u32>) -> bool {
    let bitmap = cpu.vmx.read(field::EXCEPTION_BITMAP) as u32;
    if bitmap & (1 << vector) == 0 { return false; }
    if vector == 0x0E {
        // #PF is filtered further: exit if (code & mask) == match, XOR'd
        // with the bitmap bit -- with the bit set, exit when they match.
        let mask = cpu.vmx.read(field::PF_ERROR_MASK) as u32;
        let mtch = cpu.vmx.read(field::PF_ERROR_MATCH) as u32;
        if (error_code.unwrap_or(0) & mask) != mtch { return false; }
    }
    // Interruption info: vector, type 3 (hardware exception; #BP/#OF are
    // software exceptions, type 6), error code valid, valid.
    let ty = if vector == 3 || vector == 4 { 6 } else { 3 };
    let mut info = (1u32 << 31) | (ty << 8) | vector as u32;
    if error_code.is_some() { info |= 1 << 11; }
    let qual = if vector == 0x0E { cpu.cr2 } else { 0 };
    // A fault reports the faulting instruction (rip_start); dispatch has
    // already put RIP there.
    let saved = cpu.rip;
    cpu.rip = cpu.rip_start;
    vm_exit(cpu, reason::EXCEPTION_NMI, qual, info, error_code.unwrap_or(0));
    let _ = saved;
    true
}

/// A triple fault in the guest is a VM exit, not a machine reset.
pub fn triple_fault_exit(cpu: &mut Cpu) {
    vm_exit(cpu, reason::TRIPLE_FAULT, 0, 0, 0);
}

/// Exit qualification for a CR access: CR number, access type (0 MOV to,
/// 1 MOV from, 2 CLTS, 3 LMSW), LMSW operand type, register.
fn cr_qual(cr: u8, ty: u32, reg: u8, lmsw: u32) -> u64 {
    cr as u64 | (ty as u64) << 4 | (reg as u64) << 8 | (lmsw as u64) << 16
}

/// Called by `Cpu::step` before executing an instruction in the guest.
/// Returns true when the instruction was handled here -- it exited, or was
/// executed with virtualization semantics (CR0/CR4 shadowing) -- and must
/// not be executed normally.
pub fn intercept(cpu: &mut Cpu, inst: &Inst) -> bool {
    let ctl = cpu.vmx.read(field::CPU_CTLS) as u32;
    let ctl2 = if ctl & CPU_SECONDARY_CONTROLS != 0 { cpu.vmx.read(field::CPU_CTLS2) as u32 } else { 0 };
    let exit = |cpu: &mut Cpu, r: u32, q: u64| { vm_exit(cpu, r, q, 0, 0); true };
    match inst {
        Inst::Cpuid => exit(cpu, reason::CPUID, 0),
        Inst::Vmx(v) => {
            // A VMX instruction in the guest exits with its own reason
            // (this is how nested hypervisors are built).
            let r = match v.op {
                VmxOp::Vmcall => reason::VMCALL,
                VmxOp::Vmclear => reason::VMCLEAR,
                VmxOp::Vmlaunch => reason::VMLAUNCH,
                VmxOp::Vmptrld => reason::VMPTRLD,
                VmxOp::Vmptrst => reason::VMPTRST,
                VmxOp::Vmread => reason::VMREAD,
                VmxOp::Vmresume => reason::VMRESUME,
                VmxOp::Vmwrite => reason::VMWRITE,
                VmxOp::Vmxoff => reason::VMXOFF,
                VmxOp::Vmxon => reason::VMXON,
                VmxOp::Invept => reason::INVEPT,
                VmxOp::Invvpid => reason::INVVPID,
            };
            exit(cpu, r, 0)
        }
        Inst::Xsetbv => exit(cpu, reason::XSETBV, 0),
        Inst::Hlt if ctl & CPU_HLT_EXITING != 0 => exit(cpu, reason::HLT, 0),
        Inst::Invlpg { m } if ctl & CPU_INVLPG_EXITING != 0 => {
            let lin = cpu.modrm_linear(m);
            exit(cpu, reason::INVLPG, lin)
        }
        Inst::Rdtsc if ctl & CPU_RDTSC_EXITING != 0 => exit(cpu, reason::RDTSC, 0),
        Inst::Rdtscp if ctl & CPU_RDTSC_EXITING != 0 => exit(cpu, reason::RDTSCP, 0),
        Inst::Rdtscp if ctl2 & CPU2_RDTSCP == 0 => { cpu.raise_ud(); true }
        Inst::Rdmsr | Inst::Wrmsr => {
            let idx = cpu.regs[1] as u32;
            let exits = if ctl & CPU_USE_MSR_BITMAPS != 0 {
                msr_bitmap_hit(cpu, idx, matches!(inst, Inst::Wrmsr))
            } else { true };
            if exits {
                let r = if matches!(inst, Inst::Rdmsr) { reason::RDMSR } else { reason::WRMSR };
                exit(cpu, r, 0)
            } else { false }
        }
        Inst::MovToCr { cr, reg } => cr_write_intercept(cpu, *cr, *reg, ctl),
        Inst::MovCr { cr, reg } => {
            match cr {
                3 if ctl & CPU_CR3_STORE_EXITING != 0 => exit(cpu, reason::CR_ACCESS, cr_qual(3, 1, *reg, 0)),
                8 if ctl & CPU_CR8_STORE_EXITING != 0 => exit(cpu, reason::CR_ACCESS, cr_qual(8, 1, *reg, 0)),
                0 | 4 => {
                    // The read sees the shadow for the host-owned bits.
                    let (val, mask, shadow) = if *cr == 0 {
                        (cpu.cr0 as u64, cpu.vmx.read(field::CR0_GUEST_HOST_MASK), cpu.vmx.read(field::CR0_READ_SHADOW))
                    } else {
                        (cpu.cr4 as u64, cpu.vmx.read(field::CR4_GUEST_HOST_MASK), cpu.vmx.read(field::CR4_READ_SHADOW))
                    };
                    let v = (val & !mask) | (shadow & mask);
                    let w = if cpu.long_mode() { 64 } else { 32 };
                    cpu.set_reg_w(*reg, w, v);
                    true
                }
                _ => false,
            }
        }
        Inst::Clts => {
            let mask = cpu.vmx.read(field::CR0_GUEST_HOST_MASK);
            let shadow = cpu.vmx.read(field::CR0_READ_SHADOW);
            // CLTS clears TS (bit 3); if the host owns TS and the shadow has
            // it set, that is a change the host wants to see.
            if mask & 0x8 != 0 && shadow & 0x8 != 0 {
                exit(cpu, reason::CR_ACCESS, cr_qual(0, 2, 0, 0))
            } else { false }
        }
        Inst::Lmsw { m } => {
            let mask = cpu.vmx.read(field::CR0_GUEST_HOST_MASK);
            let shadow = cpu.vmx.read(field::CR0_READ_SHADOW);
            let v = cpu.read_rm16(m) as u64;
            if cpu.pending_exception.is_some() { return true; }
            // LMSW touches PE, MP, EM, TS; PE can only be set.
            let new = (v & 0xF) | (cpu.cr0 as u64 & 1);
            let touched = (new ^ shadow) & mask & 0xF;
            if touched != 0 {
                let q = cr_qual(0, 3, 0, if m.is_reg() { 0 } else { 1 }) | (v & 0xFFFF) << 16;
                exit(cpu, reason::CR_ACCESS, q)
            } else { false }
        }
        Inst::MovDr { .. } | Inst::MovToDr { .. } if ctl & CPU_MOV_DR_EXITING != 0 => {
            let (dr, reg, dir) = match inst {
                Inst::MovDr { dr, reg } => (*dr, *reg, 1u64),
                Inst::MovToDr { dr, reg } => (*dr, *reg, 0u64),
                _ => unreachable!(),
            };
            exit(cpu, reason::DR_ACCESS, dr as u64 | dir << 4 | (reg as u64) << 8)
        }
        Inst::InAlImm { port } | Inst::InAxImm { port } | Inst::OutImmAl { port } | Inst::OutImmAx { port } => {
            let (input, size) = match inst {
                Inst::InAlImm { .. } => (true, 1u64), Inst::InAxImm { .. } => (true, if cpu.osize() == 16 { 2 } else { 4 }),
                Inst::OutImmAl { .. } => (false, 1), _ => (false, if cpu.osize() == 16 { 2 } else { 4 }),
            };
            io_intercept(cpu, *port as u16, size, input, true, ctl)
        }
        Inst::InAlDx | Inst::InAxDx | Inst::OutDxAl | Inst::OutDxAx => {
            let (input, size) = match inst {
                Inst::InAlDx => (true, 1u64), Inst::InAxDx => (true, if cpu.osize() == 16 { 2 } else { 4 }),
                Inst::OutDxAl => (false, 1), _ => (false, if cpu.osize() == 16 { 2 } else { 4 }),
            };
            let port = cpu.regs[2] as u16;
            io_intercept(cpu, port, size, input, false, ctl)
        }
        Inst::Invd => exit(cpu, reason::INVD, 0),
        Inst::Wbinvd if ctl2 & CPU2_WBINVD_EXITING != 0 => exit(cpu, reason::WBINVD, 0),
        Inst::Mwait if ctl & CPU_MWAIT_EXITING != 0 => exit(cpu, reason::MWAIT, 0),
        Inst::Monitor if ctl & CPU_MONITOR_EXITING != 0 => exit(cpu, reason::MONITOR, 0),
        Inst::Pause if ctl & CPU_PAUSE_EXITING != 0 => exit(cpu, reason::PAUSE, 0),
        _ => false,
    }
}

/// MOV to CR in the guest.
fn cr_write_intercept(cpu: &mut Cpu, cr: u8, reg: u8, ctl: u32) -> bool {
    let w = if cpu.long_mode() { 64 } else { 32 };
    let v = cpu.reg_w(reg, w);
    match cr {
        3 => {
            if ctl & CPU_CR3_LOAD_EXITING != 0 {
                // A value in the CR3-target list does not exit.
                let n = cpu.vmx.read(field::CR3_TARGET_COUNT) as usize;
                for i in 0..n.min(4) {
                    if cpu.vmx.read(field::CR3_TARGET0 + 2 * i as u32) == v { return false; }
                }
                vm_exit(cpu, reason::CR_ACCESS, cr_qual(3, 0, reg, 0), 0, 0);
                return true;
            }
            false
        }
        8 => {
            if ctl & CPU_CR8_LOAD_EXITING != 0 {
                vm_exit(cpu, reason::CR_ACCESS, cr_qual(8, 0, reg, 0), 0, 0);
                return true;
            }
            false
        }
        0 | 4 => {
            let (cur, mask, shadow) = if cr == 0 {
                (cpu.cr0 as u64, cpu.vmx.read(field::CR0_GUEST_HOST_MASK), cpu.vmx.read(field::CR0_READ_SHADOW))
            } else {
                (cpu.cr4 as u64, cpu.vmx.read(field::CR4_GUEST_HOST_MASK), cpu.vmx.read(field::CR4_READ_SHADOW))
            };
            // Writing a host-owned bit to a value other than the shadow's
            // exits; otherwise the guest-owned bits are written and the
            // host-owned ones keep the CPU's value.
            if (v ^ shadow) & mask != 0 {
                vm_exit(cpu, reason::CR_ACCESS, cr_qual(cr, 0, reg, 0), 0, 0);
                return true;
            }
            let new = (v & !mask) | (cur & mask);
            if cr == 0 { cpu.write_cr0(new as u32); } else { cpu.cr4 = new as u32; cpu.flush_tlb(); }
            true
        }
        _ => false,
    }
}

/// I/O in the guest: unconditional exiting, or the port's bit in the I/O
/// bitmaps (A covers 0-0x7FFF, B 0x8000-0xFFFF).
fn io_intercept(cpu: &mut Cpu, port: u16, size: u64, input: bool, imm: bool, ctl: u32) -> bool {
    let exits = if ctl & CPU_UNCOND_IO_EXITING != 0 {
        true
    } else if ctl & CPU_USE_IO_BITMAPS != 0 {
        let (bm, p) = if port < 0x8000 { (cpu.vmx.read(field::IO_BITMAP_A), port) } else { (cpu.vmx.read(field::IO_BITMAP_B), port - 0x8000) };
        let byte = cpu.mem.read_u8(bm as usize + (p / 8) as usize);
        byte >> (p % 8) & 1 == 1
    } else { false };
    if !exits { return false; }
    // Qualification: bits 2:0 size-1, 3 direction (1 = IN), 4 string,
    // 5 REP, 6 immediate operand, 31:16 port.
    let q = (size - 1) | (input as u64) << 3 | (imm as u64) << 6 | (port as u64) << 16;
    vm_exit(cpu, reason::IO_INSTRUCTION, q, 0, 0);
    true
}

/// The MSR bitmap: four 1 KiB maps -- read low (0-0x1FFF), read high
/// (0xC0000000-0xC0001FFF), write low, write high. A bit set exits; an
/// MSR outside both ranges always exits.
fn msr_bitmap_hit(cpu: &Cpu, idx: u32, write: bool) -> bool {
    let bm = cpu.vmx.read(field::MSR_BITMAP) as usize;
    let (base, off) = if idx < 0x2000 { (0usize, idx) }
        else if (0xC000_0000..0xC000_2000).contains(&idx) { (1024, idx - 0xC000_0000) }
        else { return true; };
    let base = base + if write { 2048 } else { 0 };
    let byte = cpu.mem.read_u8(bm + base + (off / 8) as usize);
    byte >> (off % 8) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{Cpu, flags};

    // The hypervisor and its guest live in one flat 64-bit address space
    // (identity mapped by `long_cpu`), so a test is one program that runs
    // VMXON, fills a VMCS, VMLAUNCHes, and lands at HOST_RIP on the exit.

    const VMXON_REGION: u64 = 0x30_0000;
    const VMCS_REGION: u64 = 0x30_1000;
    const HOST_STACK: u64 = 0x30_3000;
    const GUEST_STACK: u64 = 0x30_5000;
    /// Where the guest code sits (`long_cpu` loads the program at 0x10_0000;
    /// tests place the guest inside it).
    const CODE: u64 = 0x10_0000;

    fn cpu64(code: &[u8]) -> Cpu {
        let mut cpu = crate::instructions::tests::long_cpu(code);
        cpu.cr4 |= CR4_VMXE;
        // VMXON needs CR0 inside the fixed range: PE, PG and NE.
        cpu.cr0 |= 0x20;
        cpu.vmx.feature_control = FEAT_LOCKED | FEAT_VMX_OUTSIDE_SMX;
        cpu.mem.write_u32(VMXON_REGION as usize, VMCS_REVISION);
        cpu.mem.write_u32(VMCS_REGION as usize, VMCS_REVISION);
        cpu
    }

    /// `mov rax, imm64`
    fn movabs(reg: u8, v: u64) -> Vec<u8> {
        let mut c = vec![0x48 | (reg >> 3), 0xB8 | (reg & 7)];
        c.extend_from_slice(&v.to_le_bytes());
        c
    }
    /// `vmwrite rax, rbx` with rax = field, rbx = value.
    fn vmwrite(field: u32, value: u64) -> Vec<u8> {
        let mut c = movabs(0, field as u64);
        c.extend(movabs(3, value));
        c.extend_from_slice(&[0x0F, 0x79, 0xC3]); // vmwrite %rbx,%rax
        c
    }
    /// A hypervisor prologue: vmxon [rip+..] via a pointer in memory,
    /// vmclear, vmptrld, then a full set of VMCS fields for a 64-bit host
    /// (the current CPU state) and a 64-bit guest at `guest_rip`.
    fn prologue(cpu: &Cpu, guest_rip: u64, host_rip: u64, extra: &[(u32, u64)]) -> Vec<u8> {
        let mut c = Vec::new();
        // The pointer operands are 64-bit memory operands: put them at
        // VMXON_REGION+0x800 / +0x808 (inside our own region, past the
        // fields -- harmless for the test).
        let ptrs = VMXON_REGION + 0xF00;
        c.extend(movabs(0, ptrs));
        c.extend_from_slice(&[0xF3, 0x0F, 0xC7, 0x30]);        // vmxon (%rax)
        c.extend_from_slice(&[0x66, 0x0F, 0xC7, 0x70, 0x08]);  // vmclear 8(%rax)
        c.extend_from_slice(&[0x0F, 0xC7, 0x70, 0x08]);        // vmptrld 8(%rax)
        // Controls: reserved-1 bits + HLT exiting; 64-bit host and guest.
        c.extend(vmwrite(field::PIN_CTLS, PIN_DEFAULT1 as u64));
        c.extend(vmwrite(field::CPU_CTLS, (CPU_DEFAULT1 | CPU_HLT_EXITING) as u64));
        c.extend(vmwrite(field::EXIT_CTLS, (EXIT_DEFAULT1 | EXIT_HOST_ADDR_SPACE_SIZE) as u64));
        c.extend(vmwrite(field::ENTRY_CTLS, (ENTRY_DEFAULT1 | ENTRY_IA32E_GUEST) as u64));
        // Host state = this CPU's state.
        c.extend(vmwrite(field::HOST_CR0, cpu.cr0 as u64));
        c.extend(vmwrite(field::HOST_CR3, cpu.cr3));
        c.extend(vmwrite(field::HOST_CR4, cpu.cr4 as u64));
        c.extend(vmwrite(field::HOST_CS_SEL, cpu.cs as u64));
        c.extend(vmwrite(field::HOST_SS_SEL, cpu.ss as u64));
        c.extend(vmwrite(field::HOST_DS_SEL, cpu.ds as u64));
        c.extend(vmwrite(field::HOST_ES_SEL, cpu.es as u64));
        c.extend(vmwrite(field::HOST_TR_SEL, 0x40));
        c.extend(vmwrite(field::HOST_GDTR_BASE, cpu.gdt_base));
        c.extend(vmwrite(field::HOST_IDTR_BASE, cpu.idt_base));
        c.extend(vmwrite(field::HOST_RSP, HOST_STACK));
        c.extend(vmwrite(field::HOST_RIP, host_rip));
        // Guest state: same flat segments, its own stack and RIP.
        c.extend(vmwrite(field::GUEST_CR0, cpu.cr0 as u64));
        c.extend(vmwrite(field::GUEST_CR3, cpu.cr3));
        c.extend(vmwrite(field::GUEST_CR4, cpu.cr4 as u64));
        c.extend(vmwrite(field::GUEST_CS_SEL, cpu.cs as u64));
        c.extend(vmwrite(field::GUEST_CS_AR, 0xA09B)); // code, present, L, G
        c.extend(vmwrite(field::GUEST_CS_LIMIT, 0xFFFF_FFFF));
        for (sel, ar, lim) in [(field::GUEST_SS_SEL, field::GUEST_SS_AR, field::GUEST_SS_LIMIT),
                               (field::GUEST_DS_SEL, field::GUEST_DS_AR, field::GUEST_DS_LIMIT),
                               (field::GUEST_ES_SEL, field::GUEST_ES_AR, field::GUEST_ES_LIMIT)] {
            c.extend(vmwrite(sel, cpu.ds as u64));
            c.extend(vmwrite(ar, 0xC093));
            c.extend(vmwrite(lim, 0xFFFF_FFFF));
        }
        c.extend(vmwrite(field::GUEST_FS_AR, 1 << 16));
        c.extend(vmwrite(field::GUEST_GS_AR, 1 << 16));
        c.extend(vmwrite(field::GUEST_LDTR_AR, 1 << 16));
        c.extend(vmwrite(field::GUEST_TR_AR, 0x8B));
        c.extend(vmwrite(field::GUEST_GDTR_BASE, cpu.gdt_base));
        c.extend(vmwrite(field::GUEST_GDTR_LIMIT, cpu.gdt_limit as u64));
        c.extend(vmwrite(field::GUEST_IDTR_BASE, cpu.idt_base));
        c.extend(vmwrite(field::GUEST_IDTR_LIMIT, cpu.idt_limit as u64));
        c.extend(vmwrite(field::GUEST_RFLAGS, 0x2));
        c.extend(vmwrite(field::GUEST_RSP, GUEST_STACK));
        c.extend(vmwrite(field::GUEST_RIP, guest_rip));
        c.extend(vmwrite(field::GUEST_LINK_POINTER, u64::MAX));
        for (f, v) in extra { c.extend(vmwrite(*f, *v)); }
        c
    }

    /// `vmread rax -> rbx` : mov rax, field ; vmread %rax,%rbx
    fn vmread_to_rbx(field: u32) -> Vec<u8> {
        let mut c = movabs(0, field as u64);
        c.extend_from_slice(&[0x0F, 0x78, 0xC3]);
        c
    }

    /// Assemble a whole test program: prologue at 0, then `hv` (which must
    /// end by launching), then the guest, then the host exit handler; the
    /// guest and host addresses are patched into the VMCS writes.
    fn program(cpu: &Cpu, hv_tail: &[u8], guest: &[u8], host: &[u8], extra: &[(u32, u64)]) -> Vec<u8> {
        // Two passes: the prologue's length depends only on `extra`.
        let p0 = prologue(cpu, 0, 0, extra);
        let guest_at = CODE + (p0.len() + hv_tail.len()) as u64;
        let host_at = guest_at + guest.len() as u64;
        let mut c = prologue(cpu, guest_at, host_at, extra);
        c.extend_from_slice(hv_tail);
        c.extend_from_slice(guest);
        c.extend_from_slice(host);
        c
    }

    fn setup(cpu: &mut Cpu, code: &[u8]) {
        // Pointer operands for vmxon/vmclear/vmptrld.
        cpu.mem.write_u64((VMXON_REGION + 0xF00) as usize, VMXON_REGION);
        cpu.mem.write_u64((VMXON_REGION + 0xF08) as usize, VMCS_REGION);
        for (i, b) in code.iter().enumerate() { cpu.mem.write_u8(CODE as usize + i, *b); }
    }

    fn run(cpu: &mut Cpu, max: u64) {
        cpu.run(max);
        assert!(cpu.halted, "did not halt (rip={:016X}, in_guest={})", cpu.rip, cpu.vmx.in_guest);
        assert!(!cpu.triple_fault, "triple fault at {:016X}", cpu.rip);
    }

    #[test]
    fn vmxon_needs_the_feature_control_lock_and_cr4_vmxe() {
        let mut c = movabs(0, VMXON_REGION + 0xF00);
        c.extend_from_slice(&[0xF3, 0x0F, 0xC7, 0x30, 0xF4]);
        // Without CR4.VMXE: #UD.
        let mut cpu = cpu64(&c);
        setup(&mut cpu, &c);
        cpu.cr4 &= !CR4_VMXE;
        cpu.step(); cpu.step();
        assert_eq!(cpu.pending_exception, Some((0x06, None)));
        // With VMXE but the feature control unlocked: #GP.
        let mut cpu = cpu64(&c);
        setup(&mut cpu, &c);
        cpu.vmx.feature_control = 0;
        cpu.step(); cpu.step();
        assert_eq!(cpu.pending_exception, Some((0x0D, Some(0))));
        // Properly: VMsucceed and VMX on.
        let mut cpu = cpu64(&c);
        setup(&mut cpu, &c);
        run(&mut cpu, 100);
        assert!(cpu.vmx.on);
        assert_eq!(cpu.flags & 0x8D5, 0);
        // A wrong revision id: VMfailInvalid (CF).
        let mut cpu = cpu64(&c);
        setup(&mut cpu, &c);
        cpu.mem.write_u32(VMXON_REGION as usize, 0xBAD);
        run(&mut cpu, 100);
        assert!(!cpu.vmx.on);
        assert!(cpu.get_flag(flags::CF));
    }

    #[test]
    fn vmread_and_vmwrite_round_trip_and_fail_on_bad_fields() {
        // vmxon; vmptrld; vmwrite GUEST_RIP=0x1234; vmread -> rbx; hlt
        let mut c = movabs(0, VMXON_REGION + 0xF00);
        c.extend_from_slice(&[0xF3, 0x0F, 0xC7, 0x30, 0x0F, 0xC7, 0x70, 0x08]);
        c.extend(vmwrite(field::GUEST_RIP, 0x1234));
        c.extend(vmread_to_rbx(field::GUEST_RIP));
        // A read-only field: VMfailValid with error 13.
        c.extend(vmwrite(field::EXIT_REASON, 1));
        c.extend_from_slice(&[0x9C, 0x59]); // pushfq ; pop rcx
        c.extend(vmread_to_rbx(field::INSTRUCTION_ERROR));
        c.extend_from_slice(&[0x48, 0x89, 0xDA]); // mov rdx, rbx
        // An unknown field: error 12.
        c.extend(vmwrite(0x7FFE, 1));
        c.extend(vmread_to_rbx(field::INSTRUCTION_ERROR));
        c.extend_from_slice(&[0x48, 0x89, 0xDE]); // mov rsi, rbx
        // vmptrst -> memory
        c.extend(movabs(0, VMXON_REGION + 0xF10));
        c.extend_from_slice(&[0x0F, 0xC7, 0x38]); // vmptrst (%rax)
        c.push(0xF4);
        let mut cpu = cpu64(&c);
        setup(&mut cpu, &c);
        run(&mut cpu, 200);
        assert_eq!(cpu.regs[2], err::VMWRITE_READ_ONLY as u64);
        assert!(cpu.regs[1] & flags::ZF as u64 != 0 && cpu.regs[1] & flags::CF as u64 == 0);
        assert_eq!(cpu.regs[6], err::VMREAD_WRITE_BAD_FIELD as u64);
        assert_eq!(cpu.mem.read_u64((VMXON_REGION + 0xF10) as usize), VMCS_REGION);
        // The write landed in the working copy, and the field reads back.
        assert_eq!(cpu.vmx.read(field::GUEST_RIP), 0x1234);
    }

    #[test]
    fn vmlaunch_runs_the_guest_and_hlt_exits_to_the_host() {
        // Guest: mov rcx, 0x77 ; hlt   -> HLT exiting brings us to host.
        let mut guest = movabs(1, 0x77);
        guest.push(0xF4);
        // Host handler: vmread EXIT_REASON -> rbx ; vmread GUEST_RIP -> r8 ;
        // vmread EXIT_INSTR_LEN -> r9 ; hlt
        let mut host = vmread_to_rbx(field::EXIT_REASON);
        host.extend_from_slice(&[0x49, 0x89, 0xDA]); // mov r10, rbx
        host.extend(vmread_to_rbx(field::GUEST_RIP));
        host.extend_from_slice(&[0x49, 0x89, 0xD8]); // mov r8, rbx
        host.extend(vmread_to_rbx(field::EXIT_INSTR_LEN));
        host.extend_from_slice(&[0x49, 0x89, 0xD9]); // mov r9, rbx
        host.push(0xF4);
        let cpu0 = cpu64(&[0xF4]);
        let code = program(&cpu0, &[0x0F, 0x01, 0xC2], &guest, &host, &[]);  // vmlaunch
        let mut cpu = cpu64(&code);
        setup(&mut cpu, &code);
        run(&mut cpu, 2000);
        assert!(!cpu.vmx.in_guest, "should be back in the host");
        assert_eq!(cpu.regs[1], 0x77, "the guest ran (rcx)");
        assert_eq!(cpu.regs[10], reason::HLT as u64);
        // The exit points at the HLT itself, one byte long.
        let guest_at = CODE + (code.len() - guest.len() - host.len()) as u64;
        assert_eq!(cpu.regs[8], guest_at + guest.len() as u64 - 1);
        assert_eq!(cpu.regs[9], 1);
        // Host state was restored: RSP is the host stack, RFLAGS cleared.
        assert_eq!(cpu.regs[4], HOST_STACK);
        assert!(cpu.vmx.launched());
    }

    #[test]
    fn cpuid_and_vmcall_exit_and_vmresume_continues() {
        // Guest: cpuid ; vmcall ; mov rcx,0x99 ; hlt
        let mut guest = vec![0x0F, 0xA2, 0x0F, 0x01, 0xC1];
        guest.extend(movabs(1, 0x99));
        guest.push(0xF4);
        // Host: read reason into r10, count exits in r11, then skip the
        // instruction (GUEST_RIP += EXIT_INSTR_LEN) and vmresume -- unless
        // the reason is HLT, then hlt.
        let mut host = vmread_to_rbx(field::EXIT_REASON);
        host.extend_from_slice(&[0x49, 0x89, 0xDA]);             // mov r10, rbx
        host.extend_from_slice(&[0x49, 0xFF, 0xC3]);             // inc r11
        host.extend_from_slice(&[0x48, 0x83, 0xFB, 0x0C, 0x74, 0x1D]); // cmp rbx,12 ; je +0x1D (to hlt)
        host.extend(vmread_to_rbx(field::EXIT_INSTR_LEN));       // 13 bytes
        host.extend_from_slice(&[0x49, 0x89, 0xDC]);             // mov r12, rbx  (3)
        host.extend(vmread_to_rbx(field::GUEST_RIP));            // 13 bytes
        host.extend_from_slice(&[0x4C, 0x01, 0xE3]);             // add rbx, r12  (3)
        host.extend_from_slice(&[0x0F, 0x79, 0xC3]);             // vmwrite %rbx,%rax (rax still GUEST_RIP) (3)
        host.extend_from_slice(&[0x0F, 0x01, 0xC3]);             // vmresume (3)
        host.push(0xF4);
        // Fix the jump: from after `je` to the final hlt = 13+3+13+3+3+3 = 38 = 0x26.
        let je_pos = host.iter().position(|&b| b == 0x74).unwrap();
        host[je_pos + 1] = 0x26;
        let cpu0 = cpu64(&[0xF4]);
        let code = program(&cpu0, &[0x0F, 0x01, 0xC2], &guest, &host, &[]);
        let mut cpu = cpu64(&code);
        setup(&mut cpu, &code);
        run(&mut cpu, 5000);
        assert!(!cpu.vmx.in_guest);
        assert_eq!(cpu.regs[11], 3, "cpuid, vmcall, hlt: three exits");
        assert_eq!(cpu.regs[10], reason::HLT as u64);
        assert_eq!(cpu.regs[1], 0x99, "the guest resumed after each exit");
    }

    #[test]
    fn exception_bitmap_routes_a_guest_fault_to_the_host() {
        // Guest: mov rax,[0xFFFF_8000_0000_0000] (unmapped -> #PF) ; hlt
        let mut guest = movabs(0, 0xFFFF_8000_0000_0000);
        guest.extend_from_slice(&[0x48, 0x8B, 0x00, 0xF4]);
        let mut host = vmread_to_rbx(field::EXIT_REASON);
        host.extend_from_slice(&[0x49, 0x89, 0xDA]);   // r10 = reason
        host.extend(vmread_to_rbx(field::EXIT_INTR_INFO));
        host.extend_from_slice(&[0x49, 0x89, 0xDB]);   // r11 = info
        host.extend(vmread_to_rbx(field::EXIT_QUALIFICATION));
        host.extend_from_slice(&[0x49, 0x89, 0xDC]);   // r12 = qual (CR2)
        host.extend(vmread_to_rbx(field::EXIT_INTR_ERROR));
        host.extend_from_slice(&[0x49, 0x89, 0xDD]);   // r13 = error code
        host.push(0xF4);
        let cpu0 = cpu64(&[0xF4]);
        let extra = [(field::EXCEPTION_BITMAP, 1u64 << 14)];
        let code = program(&cpu0, &[0x0F, 0x01, 0xC2], &guest, &host, &extra);
        let mut cpu = cpu64(&code);
        setup(&mut cpu, &code);
        run(&mut cpu, 2000);
        assert_eq!(cpu.regs[10], reason::EXCEPTION_NMI as u64);
        assert_eq!(cpu.regs[11], (1 << 31) | (3 << 8) | (1 << 11) | 14);
        assert_eq!(cpu.regs[12], 0xFFFF_8000_0000_0000);
        assert_eq!(cpu.regs[13], 0); // not-present, read, supervisor
    }

    #[test]
    fn cr0_guest_host_mask_and_read_shadow() {
        // Host owns CR0.TS (bit 3) with a shadow of TS=1. Guest reads CR0
        // (sees TS set although the CPU's is clear), writes CR0 with the
        // same value (no exit), then clts (exits: host wants to see it).
        let mut guest = vec![0x0F, 0x20, 0xC1];             // mov rcx, cr0
        guest.extend_from_slice(&[0x0F, 0x22, 0xC1]);       // mov cr0, rcx  (shadow-consistent: no exit)
        guest.extend_from_slice(&[0x0F, 0x06]);             // clts -> exit
        guest.push(0xF4);
        let mut host = vmread_to_rbx(field::EXIT_REASON);
        host.extend_from_slice(&[0x49, 0x89, 0xDA]);
        host.extend(vmread_to_rbx(field::EXIT_QUALIFICATION));
        host.extend_from_slice(&[0x49, 0x89, 0xDB]);
        host.push(0xF4);
        let cpu0 = cpu64(&[0xF4]);
        let extra = [(field::CR0_GUEST_HOST_MASK, 0x8u64), (field::CR0_READ_SHADOW, cpu0.cr0 as u64 | 0x8)];
        let code = program(&cpu0, &[0x0F, 0x01, 0xC2], &guest, &host, &extra);
        let mut cpu = cpu64(&code);
        setup(&mut cpu, &code);
        run(&mut cpu, 2000);
        assert_eq!(cpu.regs[10], reason::CR_ACCESS as u64);
        assert_eq!(cpu.regs[11], 2 << 4, "CLTS access type");
        assert_eq!(cpu.regs[1] & 0x8, 0x8, "the guest read the shadow's TS");
    }

    #[test]
    fn io_and_msr_exits_carry_their_qualifications() {
        // Guest: in al, 0x60 ; rdmsr(ecx=0xC0000080) ; hlt  with
        // unconditional I/O exiting; the host resumes after each.
        let mut guest = vec![0xE4, 0x60];
        guest.extend_from_slice(&[0xB9, 0x80, 0x00, 0x00, 0xC0, 0x0F, 0x32, 0xF4]);
        // Host: shift each reason into r10, store each qualification at
        // TABLE[reason], then skip the instruction and resume -- until HLT.
        let table = VMXON_REGION + 0xF20;
        let mut host = Vec::new();
        host.extend(vmread_to_rbx(field::EXIT_REASON));
        host.extend_from_slice(&[0x49, 0xC1, 0xE2, 0x08, 0x49, 0x09, 0xDA]); // shl r10,8 ; or r10,rbx
        host.extend_from_slice(&[0x49, 0x89, 0xDC]);                         // mov r12, rbx
        host.extend(vmread_to_rbx(field::EXIT_QUALIFICATION));
        host.extend(movabs(0, table));
        host.extend_from_slice(&[0x4A, 0x89, 0x1C, 0xE0]);                   // mov [rax+r12*8], rbx
        host.extend_from_slice(&[0x49, 0x83, 0xFC, 0x0C, 0x74, 0x26]);       // cmp r12,12 ; je hlt
        host.extend(vmread_to_rbx(field::EXIT_INSTR_LEN));
        host.extend_from_slice(&[0x49, 0x89, 0xDD]);                         // mov r13, rbx
        host.extend(vmread_to_rbx(field::GUEST_RIP));
        host.extend_from_slice(&[0x4C, 0x01, 0xEB, 0x0F, 0x79, 0xC3, 0x0F, 0x01, 0xC3, 0xF4]); // add rbx,r13 ; vmwrite ; vmresume ; hlt
        let cpu0 = cpu64(&[0xF4]);
        let extra = [(field::CPU_CTLS, (CPU_DEFAULT1 | CPU_HLT_EXITING | CPU_UNCOND_IO_EXITING) as u64)];
        let code = program(&cpu0, &[0x0F, 0x01, 0xC2], &guest, &host, &extra);
        let mut cpu = cpu64(&code);
        setup(&mut cpu, &code);
        run(&mut cpu, 5000);
        assert_eq!(cpu.regs[10], (reason::IO_INSTRUCTION as u64) << 16 | (reason::RDMSR as u64) << 8 | reason::HLT as u64);
        // The I/O qualification: IN, byte, immediate port 0x60.
        let io_qual = cpu.mem.read_u64((table + 8 * reason::IO_INSTRUCTION as u64) as usize);
        assert_eq!(io_qual, (1 << 3) | (1 << 6) | (0x60 << 16));
    }

    #[test]
    fn external_interrupt_exits_when_asked() {
        // Guest: sti ; loop forever (jmp $). The PIT fires IRQ0 -> the
        // pin-based control routes it to the host with the vector.
        let guest = vec![0xFB, 0xEB, 0xFE];
        let mut host = vmread_to_rbx(field::EXIT_REASON);
        host.extend_from_slice(&[0x49, 0x89, 0xDA]);
        host.extend(vmread_to_rbx(field::EXIT_INTR_INFO));
        host.extend_from_slice(&[0x49, 0x89, 0xDB]);
        host.push(0xF4);
        let cpu0 = cpu64(&[0xF4]);
        let extra = [(field::PIN_CTLS, (PIN_DEFAULT1 | PIN_EXT_INT_EXITING) as u64),
                     (field::EXIT_CTLS, (EXIT_DEFAULT1 | EXIT_HOST_ADDR_SPACE_SIZE | EXIT_ACK_INT_ON_EXIT) as u64)];
        let code = program(&cpu0, &[0x0F, 0x01, 0xC2], &guest, &host, &extra);
        let mut cpu = cpu64(&code);
        setup(&mut cpu, &code);
        // Program the PIT for a short period and unmask IRQ0.
        cpu.pit.write_control(0x36);
        cpu.pit.write_data(0x40); cpu.pit.write_data(0x00);
        cpu.pic.master_imr = 0xFE;
        run(&mut cpu, 200_000);
        assert_eq!(cpu.regs[10], reason::EXTERNAL_INTERRUPT as u64);
        assert_eq!(cpu.regs[11], (1 << 31) | 0x08, "IRQ0 at the PIC's base vector 8");
    }

    #[test]
    fn event_injection_delivers_through_the_guest_idt() {
        // Guest IDT entry 0x21 -> handler that sets rcx and halts. Entry
        // injects vector 0x21 (external interrupt type 0). Guest RIP is a
        // separate `hlt` the handler never returns to.
        let guest_at_guess = 0; // computed below via program()
        let _ = guest_at_guess;
        let mut guest = movabs(1, 0xABC);
        guest.push(0xF4);                        // the handler: mov rcx,0xABC ; hlt
        let mut host = vmread_to_rbx(field::EXIT_REASON);
        host.extend_from_slice(&[0x49, 0x89, 0xDA]);
        host.push(0xF4);
        let cpu0 = cpu64(&[0xF4]);
        // A 16-byte long-mode gate at guest IDT[0x21]: use a fresh IDT page.
        let idt = 0x30_7000u64;
        // Build once (same number of extra fields, so the same layout) to
        // learn the guest address, then set the IDT gate and GUEST_RIP.
        let mut extra = [(field::GUEST_IDTR_BASE, idt), (field::GUEST_IDTR_LIMIT, 0xFFF),
                         (field::ENTRY_INTR_INFO, (1u64 << 31) | 0x21),
                         (field::GUEST_RIP, 0)];
        let code = program(&cpu0, &[0x0F, 0x01, 0xC2], &guest, &host, &extra);
        let handler = CODE + (code.len() - guest.len() - host.len()) as u64;
        // Guest RIP: the halt at the end of the handler (never reached: the
        // injected interrupt lands first and the handler halts -> HLT exit).
        extra[3].1 = handler + guest.len() as u64 - 1;
        let code = program(&cpu0, &[0x0F, 0x01, 0xC2], &guest, &host, &extra);
        let mut cpu = cpu64(&code);
        setup(&mut cpu, &code);
        let g = idt as usize + 0x21 * 16;
        cpu.mem.write_u16(g, handler as u16);
        cpu.mem.write_u16(g + 2, cpu.cs);
        cpu.mem.write_u8(g + 4, 0);
        cpu.mem.write_u8(g + 5, 0x8E);
        cpu.mem.write_u16(g + 6, (handler >> 16) as u16);
        cpu.mem.write_u32(g + 8, (handler >> 32) as u32);
        run(&mut cpu, 3000);
        assert_eq!(cpu.regs[1], 0xABC, "the injected interrupt's handler ran");
        assert_eq!(cpu.regs[10], reason::HLT as u64);
        // The injection field was consumed.
        assert_eq!(cpu.vmx.read(field::ENTRY_INTR_INFO), 0);
    }

    #[test]
    fn capability_msrs_are_consistent_with_the_controls() {
        for (m, allowed1) in [(msr::VMX_PINBASED_CTLS, PIN_ALLOWED1), (msr::VMX_PROCBASED_CTLS, CPU_ALLOWED1),
                              (msr::VMX_EXIT_CTLS, EXIT_ALLOWED1), (msr::VMX_ENTRY_CTLS, ENTRY_ALLOWED1)] {
            let v = read_capability_msr(m).unwrap();
            let (a0, a1) = (v as u32, (v >> 32) as u32);
            assert_eq!(a0 & !a1, 0, "must-be-1 bits are allowed-1 for {m:X}");
            assert_eq!(a1, allowed1);
        }
        let basic = read_capability_msr(msr::VMX_BASIC).unwrap();
        assert_eq!(basic as u32, VMCS_REVISION);
        assert_eq!((basic >> 32) & 0x1FFF, 4096);
        // Through the CPU's MSR path too, and CPUID says VMX.
        let cpu = Cpu::new();
        assert_eq!(cpu.read_msr(msr::VMX_CR0_FIXED0), CR0_FIXED0);
    }

    #[test]
    fn slots_are_distinct_and_cover_every_field() {
        let mut seen = std::collections::HashSet::new();
        for f in [field::VPID, field::GUEST_CS_SEL, field::HOST_TR_SEL, field::IO_BITMAP_A, field::MSR_BITMAP,
                  field::GUEST_EFER, field::HOST_EFER, field::PIN_CTLS, field::CPU_CTLS2, field::EXIT_REASON,
                  field::GUEST_CS_AR, field::GUEST_SYSENTER_CS, field::HOST_SYSENTER_CS, field::CR0_GUEST_HOST_MASK,
                  field::EXIT_QUALIFICATION, field::GUEST_CR0, field::GUEST_RIP, field::HOST_RIP, field::GUEST_SYSENTER_EIP] {
            let s = slot(f).unwrap();
            assert!(seen.insert(s), "slot collision for {f:X}");
            assert!(s < SLOTS && s != LAUNCH_SLOT);
        }
        // The high half of a 64-bit field shares its slot.
        assert_eq!(slot(field::MSR_BITMAP), slot(field::MSR_BITMAP | 1));
        let mut v = Vmx::new();
        v.write(field::MSR_BITMAP, 0x1122_3344_5566_7788);
        assert_eq!(v.read(field::MSR_BITMAP | 1), 0x1122_3344);
        v.write(field::MSR_BITMAP | 1, 0xAAAA_BBBB);
        assert_eq!(v.read(field::MSR_BITMAP), 0xAAAA_BBBB_5566_7788);
        // A 16-bit field is masked.
        v.write(field::GUEST_CS_SEL, 0x1_0010);
        assert_eq!(v.read(field::GUEST_CS_SEL), 0x10);
    }
}
