//! Instruction decoder and executor.
//!
//! Supports the 8086 real-mode instruction set plus 32-bit protected-mode
//! extensions: the 0x66/0x67 size-override prefixes, 32-bit register and
//! addressing forms, LGDT/LIDT, and protected-mode interrupt dispatch
//! through the IDT.

use crate::cpu::{Cpu, Reg8, Reg16, Reg32, SegReg, flags, CR4_OSXSAVE};
use crate::modrm::ModRm;

/// A decoded instruction, kept simple for diagnostics and tests.
#[derive(Clone, Debug)]
pub enum Inst {
    Nop,
    Hlt,
    MovRm8Reg { m: ModRm, src: u8 },
    MovRm16Reg { m: ModRm, src: u8 },
    MovRm32Reg { m: ModRm, src: u8 },
    MovRegRm8 { m: ModRm, dst: u8 },
    MovRegRm16 { m: ModRm, dst: u8 },
    MovRegRm32 { m: ModRm, dst: u8 },
    MovRm8Imm { m: ModRm, imm: u8 },
    MovRm16Imm { m: ModRm, imm: u16 },
    MovRm32Imm { m: ModRm, imm: u32 },
    MovReg8Imm { dst: u8, imm: u8 },
    MovReg16Imm { dst: u8, imm: u16 },
    MovReg32Imm { dst: u8, imm: u32 },
    /// B8+r with REX.W: the only instruction carrying a 64-bit immediate.
    MovReg64Imm { dst: u8, imm: u64 },
    MovAccMem8 { addr: u16 },
    MovMem8Acc { addr: u16 },
    MovAccMem8Addr32 { addr: u32 },
    MovMem8AccAddr32 { addr: u32 },
    MovAccMem16 { addr: u16 },
    MovMem16Acc { addr: u16 },
    MovAccMem16Addr32 { addr: u32 },
    MovMem16AccAddr32 { addr: u32 },
    // The moffs forms take an address as wide as the address size, which
    // in 64-bit mode is a full 64 bits — `movabs` to and from an absolute
    // address, the only way to reach one without a base register.
    MovAccMem32 { addr: u64 },
    MovMem32Acc { addr: u64 },
    MovRmSeg { m: ModRm, seg: SegReg },
    MovSegRm { seg: SegReg, m: ModRm },
    // Load segment with pointer: LDS (0xC5) / LES (0xC4) / LSS (0F B2) /
    // LFS (0F B4) / LGS (0F B5)
    Lds { m: ModRm },
    Les { m: ModRm },
    Lss { m: ModRm },
    Lfs { m: ModRm },
    Lgs { m: ModRm },
    AluRm8Reg { op: AluOp, m: ModRm, reg: u8, dir: Dir },
    AluRm16Reg { op: AluOp, m: ModRm, reg: u8, dir: Dir },
    AluRm32Reg { op: AluOp, m: ModRm, reg: u8, dir: Dir },
    AluRm8Imm { op: AluOp, m: ModRm, imm: u8 },
    AluRm16Imm { op: AluOp, m: ModRm, imm: u16, imm_is8: bool },
    AluRm32Imm { op: AluOp, m: ModRm, imm: u32, imm_is8: bool },
    AluAccImm8 { op: AluOp, imm: u8 },
    AluAccImm16 { op: AluOp, imm: u16 },
    AluAccImm32 { op: AluOp, imm: u32 },
    IncReg16 { dst: u8 },
    DecReg16 { dst: u8 },
    IncReg32 { dst: u8 },
    DecReg32 { dst: u8 },
    PushReg16 { src: u8 },
    PopReg16 { dst: u8 },
    PushReg32 { src: u8 },
    PopReg32 { dst: u8 },
    // Two- and three-operand IMUL: 0F AF (r <- r * r/m) and 69/6B
    // (r <- r/m * imm). Both set CF/OF when the full product does not fit
    // in the destination; the stored result is the truncated low half.
    ImulRegRm16 { m: ModRm, dst: u8 },
    ImulRegRm32 { m: ModRm, dst: u8 },
    ImulRegRmImm16 { m: ModRm, dst: u8, imm: i16 },
    ImulRegRmImm32 { m: ModRm, dst: u8, imm: i32 },
    // Double-precision shifts: SHLD (0F A4/A5) and SHRD (0F AC/AD).
    Shld { m: ModRm, reg: u8, count: ShiftCount, w32: bool },
    Shrd { m: ModRm, reg: u8, count: ShiftCount, w32: bool },
    // SETcc r/m8 (0F 90-9F): store 1 if the condition holds, else 0.
    Setcc { cond: Cond, m: ModRm },
    // LEAVE (0xC9): tear down a stack frame — ESP = EBP, then pop EBP.
    // Bit scan forward/reverse (0F BC / 0F BD).
    Bsf { m: ModRm, dst: u8, w32: bool },
    Bsr { m: ModRm, dst: u8, w32: bool },
    // XCHG r/m, r (0x86 byte / 0x87 word-dword).
    XchgRmReg { m: ModRm, reg: u8, width: u32 },
    // CMPXCHG r/m, r (0F B0 byte / 0F B1 word-dword).
    Cmpxchg { m: ModRm, reg: u8, width: u32 },
    // XADD r/m, r (0F C0 byte / 0F C1 word-dword).
    Xadd { m: ModRm, reg: u8, width: u32 },
    // CMPXCHG8B m64 (0F C7 /1).
    Cmpxchg8b { m: ModRm },
    // BSWAP r32 (0F C8+r).
    Bswap { reg: u8 },
    // CMOVcc r, r/m (0F 40-4F).
    Cmovcc { cond: Cond, m: ModRm, dst: u8, w32: bool },
    // PUSH/POP a segment register. The one-byte forms cover ES/CS/SS/DS
    // (0x06/0x0E/0x16/0x1E push, 0x07/0x17/0x1F pop -- there is no POP CS);
    // FS/GS use the two-byte forms 0F A0/A1 and 0F A8/A9.
    PushSeg { seg: SegReg },
    PopSeg { seg: SegReg },
    // MOV r32, DRx (0F 21) / MOV DRx, r32 (0F 23).
    MovDr { dr: u8, reg: u8 },
    MovToDr { dr: u8, reg: u8 },
    // 0F 00 group: LLDT (/2), LTR (/3), SLDT (/0), STR (/1).
    Lldt { m: ModRm },
    Ltr { m: ModRm },
    Sldt { m: ModRm },
    Str { m: ModRm },
    Leave { w32: bool },
    // POP r/m16 / r/m32 (0x8F /0).
    PopRm16 { m: ModRm },
    PopRm32 { m: ModRm },
    // PUSHA/PUSHAD (0x60) and POPA/POPAD (0x61).
    Pusha { w32: bool },
    Popa { w32: bool },
    PushImm16 { imm: u16 },
    PushImm32 { imm: u32 },
    JmpRel8 { rel: i8 },
    JmpRel16 { rel: i16 },
    JmpRel32 { rel: i32 },
    Jcc { cond: Cond, rel: i8 },
    // 32-bit conditional jump (0F 80-8F): Jcc rel32.
    Jcc32 { cond: Cond, rel: i32 },
    // MOVZX r16/32, r/m8 (0F B6) / MOVZX r32, r/m16 (0F B7)
    Movzx8 { m: ModRm, dst: u8 },
    Movzx16 { m: ModRm, dst: u8 },
    // MOVSX r16/32, r/m8 (0F BE) / MOVSX r32, r/m16 (0F BF)
    Movsx8 { m: ModRm, dst: u8 },
    Movsx16 { m: ModRm, dst: u8 },
    /// MOVSXD r64, r/m32 (0x63 in 64-bit mode, where it displaces ARPL).
    Movsxd { m: ModRm, dst: u8 },
    CallRel16 { rel: i16 },
    CallRel32 { rel: i32 },
    Ret,
    Ret32,
    // RET imm16 (0xC2): return, then drop `imm` bytes of arguments.
    RetImm { imm: u16, w32: bool },
    XchgAxReg { reg: u8 },
    XchgEaxReg { reg: u8 },
    Int { vector: u8 },
    Int3,
    Into,
    Iret,
    Iret32,
    Pushf,
    Popf,
    // Shifts / rotates (group 2, 0xD0-0xD3). `width` is 8, 16 or 32 — a
    // bool here is what let the 32-bit form silently run as an 8-bit shift.
    Shift { op: ShiftOp, m: ModRm, width: u32, count: ShiftCount },
    // Shifts / rotates with imm8 count (group 2, 0xC0-0xC1)
    ShiftImm { op: ShiftOp, m: ModRm, width: u32, imm: u8 },
    // Group 3 (0xF6/0xF7): TEST / NOT / NEG / MUL / IMUL / DIV / IDIV
    TestRm8Imm { m: ModRm, imm: u8 },
    TestRm16Imm { m: ModRm, imm: u16 },
    TestRm32Imm { m: ModRm, imm: u32 },
    TestRm8Reg { m: ModRm, reg: u8 },
    TestRm16Reg { m: ModRm, reg: u8 },
    TestRm32Reg { m: ModRm, reg: u8 },
    TestAccImm8 { imm: u8 },
    TestAccImm16 { imm: u16 },
    TestAccImm32 { imm: u32 },
    NotRm8 { m: ModRm },
    NotRm16 { m: ModRm },
    NotRm32 { m: ModRm },
    NegRm8 { m: ModRm },
    NegRm16 { m: ModRm },
    NegRm32 { m: ModRm },
    MulRm8 { m: ModRm },
    MulRm16 { m: ModRm },
    MulRm32 { m: ModRm },
    ImulRm8 { m: ModRm },
    ImulRm16 { m: ModRm },
    ImulRm32 { m: ModRm },
    DivRm8 { m: ModRm },
    DivRm16 { m: ModRm },
    DivRm32 { m: ModRm },
    IdivRm8 { m: ModRm },
    IdivRm16 { m: ModRm },
    IdivRm32 { m: ModRm },
    // LEA (0x8D)
    Lea { m: ModRm, dst: u8 },
    // CBW (0x98) / CWD (0x99) / CWDE (0x98 w/ 66) / CDQ (0x99 w/ 66)
    Cbw,
    Cwd,
    Cwde,
    Cdq,
    // LOOP / LOOPZ / LOOPNZ / JCXZ (0xE0-0xE3)
    Loop { cond: LoopCond, rel: i8 },
    // Far control flow
    JmpFar { off: u16, seg: u16 },
    CallFar { off: u16, seg: u16 },
    JmpFar32 { off: u32, seg: u16 },
    CallFar32 { off: u32, seg: u16 },
    Retf,
    Retf32,
    // Group 5 (0xFF): INC / DEC / CALL / JMP / PUSH r/m
    // INC/DEC r/m8 (group 4, 0xFE /0 and /1).
    IncRm8 { m: ModRm },
    DecRm8 { m: ModRm },
    IncRm16 { m: ModRm },
    IncRm32 { m: ModRm },
    DecRm16 { m: ModRm },
    DecRm32 { m: ModRm },
    CallRm16 { m: ModRm },
    CallRm32 { m: ModRm },
    JmpRm16 { m: ModRm },
    JmpRm32 { m: ModRm },
    PushRm16 { m: ModRm },
    PushRm32 { m: ModRm },
    // String ops (with optional REP prefix)
    Movs { rep: Rep, w: bool },
    Stos { rep: Rep, w: bool },
    Lods { rep: Rep, w: bool },
    Cmps { rep: Rep, w: bool },
    Scas { rep: Rep, w: bool },
    // LGDT / LIDT (0x0F 0x01 /2 and /3), and their store forms SGDT / SIDT
    // (/0 and /1). The kernel reads its own GDTR back to verify it
    // (`native_store_gdt`), so a CPU that loads but cannot store hangs
    // `cpu_init` on a #UD.
    Lgdt { m: ModRm },
    Lidt { m: ModRm },
    Sgdt { m: ModRm },
    Sidt { m: ModRm },
    /// SMSW (0F 01 /4) / LMSW (0F 01 /6): the 286-era views of CR0's low
    /// 16 bits. LMSW can set PE and never clear it.
    Smsw { m: ModRm },
    Lmsw { m: ModRm },
    /// XGETBV / XSETBV (0F 01 D0 / D1): the extended control register XCR0.
    Xgetbv,
    Xsetbv,
    /// CLAC / STAC (0F 01 CA / CB): clear or set RFLAGS.AC (SMAP).
    Clac,
    Stac,
    // INVLPG (0x0F 0x01 /7): invalidate TLB entry for a linear address.
    Invlpg { m: ModRm },
    // MOV r32, cr (0x0F 0x20) / MOV cr, r32 (0x0F 0x22)
    MovCr { cr: u8, reg: u8 },
    MovToCr { cr: u8, reg: u8 },
    // CLTS (0x0F 0x06): clear CR0.TS (task-switched flag).
    Clts,
    /// SYSCALL (0F 05) and SYSRET (0F 07): the fast system-call pair long
    /// mode replaces `int 0x80` with.
    Syscall,
    Sysret,
    /// SWAPGS (0F 01 F8): exchange GS.base with KERNEL_GS_BASE.
    Swapgs,
    /// RDTSCP (0F 01 F9): RDTSC plus the processor id in ECX.
    Rdtscp,
    /// Instructions with no architectural effect here: memory fences,
    /// prefetch hints, cache management, and the multi-byte NOP.
    NopHint,
    /// INVD / WBINVD (0F 08 / 0F 09): no cache to invalidate, but a
    /// hypervisor may ask to see them.
    Invd,
    Wbinvd,
    /// MONITOR / MWAIT (0F 01 C8 / C9) and PAUSE (F3 90). One core and no
    /// caches: MONITOR arms nothing, MWAIT is a HLT that also wakes on the
    /// interrupt it would have waited for, PAUSE is a NOP -- but each is its
    /// own instruction so a hypervisor can intercept it.
    Monitor,
    Mwait,
    Pause,
    // CPUID (0x0F 0xA2) / RDTSC (0x0F 0x31)
    Cpuid,
    Rdtsc,
    // RDMSR (0x0F 0x32) / WRMSR (0x0F 0x30)
    Rdmsr,
    Wrmsr,
    // Bit tests: BT/BTS/BTR/BTC (0F A3/AB/B3/BB, and group 8 0F BA /4-/7)
    Bt { m: ModRm, bit: BitOffset },
    Bts { m: ModRm, bit: BitOffset },
    Btr { m: ModRm, bit: BitOffset },
    Btc { m: ModRm, bit: BitOffset },
    // IN / OUT (0xE4-0xE7, 0xEC-0xEF)
    InAlImm { port: u8 },
    InAxImm { port: u8 },
    InAlDx,
    InAxDx,
    OutImmAl { port: u8 },
    OutImmAx { port: u8 },
    OutDxAl,
    OutDxAx,
    // Flag-control instructions: CLC/STC/CLI/STI/CLD/STD/CMC
    Clc,
    Stc,
    Cli,
    Sti,
    Cld,
    Std,
    Cmc,
    // ---- x87 FPU (D8-DF) ----
    // FNINIT (DB E3)
    Fninit,
    // FSTCW m16 (D9 /7) / FLDCW m16 (D9 /5)
    Fstcw { m: ModRm },
    Fldcw { m: ModRm },
    // FSTSW AX (DF E0) / FSTSW m16 (DD /7)
    FstswAx,
    Fstsw { m: ModRm },
    // FST/FSTP ST(i) (DD /0, /1)
    Fst { m: ModRm, w64: bool },
    Fstp { m: ModRm, w64: bool },
    // FLD m32/m64 (D9 /0, DD /0)
    Fld { m: ModRm, w64: bool },
    // FILD m16/m32 (DF /0, DB /0)
    Fild { m: ModRm },
    // FISTP m16/m32 (DF /3, DB /3)
    Fistp { m: ModRm },
    // FADD/FSUB/FMUL/FDIV (D8/DC groups) — simplified: operate on ST0.
    Fop { op: FpuOp, m: ModRm },
    /// An SSE/SSE2/SSE3 instruction, decoded and executed in `sse.rs`
    /// (which also owns FXSAVE/FXRSTOR and MXCSR).
    Sse(crate::sse::SseInst),
    /// A VT-x instruction (VMXON ... VMCALL), decoded and executed in
    /// `vmx.rs`.
    Vmx(crate::vmx::VmxInst),
    Unknown { opcode: u16 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FpuOp { Add, Sub, Mul, Div }

/// Where a bit-test instruction's bit offset comes from.
///
/// The register form (0F A3/AB/B3/BB) takes the *value* of the register named
/// by the ModR/M reg field, not the field itself; reading the index instead
/// makes every `test_bit()` in the kernel answer from the wrong bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BitOffset {
    Imm(u8),
    Reg(u8),
}

/// Shift/rotate operation selected by the `reg` field of group 2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShiftOp { Rol, Ror, Rcl, Rcr, Shl, Shr, Sar }

impl ShiftOp {
    pub fn from_index(i: u8) -> Self {
        match i & 7 {
            0 => ShiftOp::Rol, 1 => ShiftOp::Ror, 2 => ShiftOp::Rcl, 3 => ShiftOp::Rcr,
            4 => ShiftOp::Shl, 5 => ShiftOp::Shr, _ => ShiftOp::Sar,
        }
    }
}

/// Shift count source: immediate 1, or the CL register.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShiftCount { One, Cl, Imm(u8) }

/// REP prefix state for string instructions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rep { None, Repe, Repne }

/// Loop condition for LOOP / LOOPZ / LOOPNZ / JCXZ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoopCond { Loop, Loopz, Loopnz, Jcxz }

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AluOp { Add, Or, Adc, Sbb, And, Sub, Xor, Cmp }

impl AluOp {
    /// The ALU operation index encoded in bits 3-5 of opcodes 0x00-0x3F.
    pub fn from_index(i: u8) -> Self {
        match i {
            0 => AluOp::Add, 1 => AluOp::Or, 2 => AluOp::Adc, 3 => AluOp::Sbb,
            4 => AluOp::And, 5 => AluOp::Sub, 6 => AluOp::Xor, _ => AluOp::Cmp,
        }
    }
}

/// Direction of a two-operand r/m,reg instruction.
/// `RmReg`: dest = r/m, src = reg.  `RegRm`: dest = reg, src = r/m.
#[derive(Clone, Copy, Debug)]
pub enum Dir { RmReg, RegRm }

#[derive(Clone, Copy, Debug)]
pub enum Cond {
    O, No, B, Nb, E, Ne, Be, Nbe, S, Ns, P, Np, L, Nl, Le, Nle,
}

impl Cond {
    /// Map the low nibble of a 0x70-0x7F Jcc opcode to a condition.
    pub fn from_jcc(i: u8) -> Self {
        match i & 0xF {
            0 => Cond::O, 1 => Cond::No, 2 => Cond::B, 3 => Cond::Nb,
            4 => Cond::E, 5 => Cond::Ne, 6 => Cond::Be, 7 => Cond::Nbe,
            8 => Cond::S, 9 => Cond::Ns, 10 => Cond::P, 11 => Cond::Np,
            12 => Cond::L, 13 => Cond::Nl, 14 => Cond::Le, _ => Cond::Nle,
        }
    }

    pub fn test(&self, cpu: &Cpu) -> bool {
        use flags::*;
        match self {
            Cond::O => cpu.get_flag(OF),
            Cond::No => !cpu.get_flag(OF),
            Cond::B => cpu.get_flag(CF),
            Cond::Nb => !cpu.get_flag(CF),
            Cond::E => cpu.get_flag(ZF),
            Cond::Ne => !cpu.get_flag(ZF),
            Cond::Be => cpu.get_flag(CF) || cpu.get_flag(ZF),
            Cond::Nbe => !cpu.get_flag(CF) && !cpu.get_flag(ZF),
            Cond::S => cpu.get_flag(SF),
            Cond::Ns => !cpu.get_flag(SF),
            Cond::P => cpu.get_flag(PF),
            Cond::Np => !cpu.get_flag(PF),
            // signed comparisons use SF != OF
            Cond::L => cpu.get_flag(SF) != cpu.get_flag(OF),
            Cond::Nl => cpu.get_flag(SF) == cpu.get_flag(OF),
            Cond::Le => cpu.get_flag(ZF) || cpu.get_flag(SF) != cpu.get_flag(OF),
            Cond::Nle => !cpu.get_flag(ZF) && cpu.get_flag(SF) == cpu.get_flag(OF),
        }
    }
}

// ---- Decoder ----

pub fn decode(cpu: &mut Cpu) -> Inst {
    // Reset per-instruction prefix state (persists through execute, which
    // runs after decode returns).
    //
    // The defaults come from the mode. In a legacy mode they come from the
    // code segment: a D=1 code segment defaults to 32-bit operands and
    // addressing, otherwise 16-bit, and 0x66/0x67 *toggle* the size. In
    // 64-bit mode they do not come from the segment at all -- D must be 0
    // when L is set -- and are fixed at 32-bit operands and 64-bit
    // addressing, with 0x66 selecting 16-bit operands, 0x67 selecting 32-bit
    // addressing, and REX.W selecting 64-bit operands.
    let long64 = cpu.long64();
    if long64 {
        cpu.opsize = true;
        cpu.addrsize = true;
        cpu.addr64 = true;
    } else {
        let d32 = cpu.pe && cpu.seg_desc[SegReg::Cs as usize].d_b;
        cpu.opsize = d32;
        cpu.addrsize = d32;
        cpu.addr64 = false;
    }
    cpu.seg_override = None;
    cpu.rex_present = false;
    cpu.rex_w = false;
    cpu.rex_r = false;
    cpu.rex_x = false;
    cpu.rex_b = false;
    cpu.sse_pfx = None;
    // Handle prefixes: REP/REPNE (0xF3/0xF2), operand-size (0x66),
    // address-size (0x67), segment overrides, and — in 64-bit mode — REX.
    // The 0x66/0xF3/0xF2 bytes also serve as SSE mandatory prefixes; the
    // *last* one before the opcode (or before REX) is what SSE uses, so we
    // record it in `sse_pfx` and let the 0F-escape decoder read it.
    let mut rep = Rep::None;
    loop {
        let peek = cpu.peek_u8();
        // REX (0x40-0x4F) exists only in 64-bit mode, where it displaces
        // the one-byte INC/DEC forms entirely, and it must be the **last**
        // prefix before the opcode: a legacy prefix after one cancels it.
        if long64 && (0x40..=0x4F).contains(&peek) {
            let b = cpu.fetch_u8();
            cpu.rex_present = true;
            cpu.rex_w = b & 8 != 0;
            cpu.rex_r = b & 4 != 0;
            cpu.rex_x = b & 2 != 0;
            cpu.rex_b = b & 1 != 0;
            continue;
        }
        if matches!(peek, 0xF0 | 0xF3 | 0xF2 | 0x66 | 0x67
                        | 0x2E | 0x36 | 0x3E | 0x26 | 0x64 | 0x65) {
            cpu.rex_present = false;
            cpu.rex_w = false;
            cpu.rex_r = false;
            cpu.rex_x = false;
            cpu.rex_b = false;
        }
        match peek {
            // LOCK. This is a uniprocessor emulator with no concurrent bus
            // master, so the prefix is consumed and the instruction runs
            // normally -- but it must be consumed, or `lock cmpxchg` decodes
            // 0xF0 as an opcode and faults.
            0xF0 => { cpu.fetch_u8(); }
            0xF3 => { rep = Rep::Repe; cpu.sse_pfx = Some(0xF3); cpu.fetch_u8(); }
            0xF2 => { rep = Rep::Repne; cpu.sse_pfx = Some(0xF2); cpu.fetch_u8(); }
            0x66 => {
                if long64 { cpu.opsize = false; } else { cpu.opsize = !cpu.opsize; }
                cpu.sse_pfx = Some(0x66);
                cpu.fetch_u8();
            }
            0x67 => {
                if long64 { cpu.addr64 = false; } else { cpu.addrsize = !cpu.addrsize; }
                cpu.fetch_u8();
            }
            // In 64-bit mode CS, DS, ES and SS overrides are ignored -- those
            // segments have no base to override. FS and GS keep theirs.
            0x2E => { if !long64 { cpu.seg_override = Some(SegReg::Cs); } cpu.fetch_u8(); }
            0x36 => { if !long64 { cpu.seg_override = Some(SegReg::Ss); } cpu.fetch_u8(); }
            0x3E => { if !long64 { cpu.seg_override = Some(SegReg::Ds); } cpu.fetch_u8(); }
            0x26 => { if !long64 { cpu.seg_override = Some(SegReg::Es); } cpu.fetch_u8(); }
            0x64 => { cpu.seg_override = Some(SegReg::Fs); cpu.fetch_u8(); }
            0x65 => { cpu.seg_override = Some(SegReg::Gs); cpu.fetch_u8(); }
            _ => break,
        }
    }
    // REX.W wins over everything: it is the one prefix that can *widen* the
    // operand size, and a 0x66 seen earlier does not undo it.
    if cpu.rex_w { cpu.opsize = true; }
    let op = cpu.fetch_u8();
    decode_op(cpu, op, rep)
}

/// REX.B as the high bit of an opcode-embedded register number.
#[inline]
fn rb(cpu: &Cpu) -> u8 { (cpu.rex_b as u8) << 3 }

fn decode_op(cpu: &mut Cpu, op: u8, rep: Rep) -> Inst {
    // `w32` selects between the 16-bit and the wider instruction *shapes*.
    // The wider one covers both 32 and 64 bits; `cpu.osize()` is what says
    // which, and instructions that carry an explicit width take it from
    // there so REX.W actually widens them.
    let w32 = cpu.opsize;
    match op {
        0x00 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Add, m, reg: m.reg, dir: Dir::RmReg } }
        0x01 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Add, m, reg: m.reg, dir: Dir::RmReg } } else { Inst::AluRm16Reg { op: AluOp::Add, m, reg: m.reg, dir: Dir::RmReg } } }
        0x02 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Add, m, reg: m.reg, dir: Dir::RegRm } }
        0x03 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Add, m, reg: m.reg, dir: Dir::RegRm } } else { Inst::AluRm16Reg { op: AluOp::Add, m, reg: m.reg, dir: Dir::RegRm } } }
        0x04 => Inst::AluAccImm8 { op: AluOp::Add, imm: cpu.fetch_u8() },
        0x05 => { if w32 { Inst::AluAccImm32 { op: AluOp::Add, imm: cpu.fetch_u32() } } else { Inst::AluAccImm16 { op: AluOp::Add, imm: cpu.fetch_u16() } } }

        0x08 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Or, m, reg: m.reg, dir: Dir::RmReg } }
        0x09 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Or, m, reg: m.reg, dir: Dir::RmReg } } else { Inst::AluRm16Reg { op: AluOp::Or, m, reg: m.reg, dir: Dir::RmReg } } }
        0x0A => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Or, m, reg: m.reg, dir: Dir::RegRm } }
        0x0B => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Or, m, reg: m.reg, dir: Dir::RegRm } } else { Inst::AluRm16Reg { op: AluOp::Or, m, reg: m.reg, dir: Dir::RegRm } } }
        0x0C => Inst::AluAccImm8 { op: AluOp::Or, imm: cpu.fetch_u8() },
        0x0D => { if w32 { Inst::AluAccImm32 { op: AluOp::Or, imm: cpu.fetch_u32() } } else { Inst::AluAccImm16 { op: AluOp::Or, imm: cpu.fetch_u16() } } }

        0x10 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Adc, m, reg: m.reg, dir: Dir::RmReg } }
        0x11 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Adc, m, reg: m.reg, dir: Dir::RmReg } } else { Inst::AluRm16Reg { op: AluOp::Adc, m, reg: m.reg, dir: Dir::RmReg } } }
        0x12 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Adc, m, reg: m.reg, dir: Dir::RegRm } }
        0x13 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Adc, m, reg: m.reg, dir: Dir::RegRm } } else { Inst::AluRm16Reg { op: AluOp::Adc, m, reg: m.reg, dir: Dir::RegRm } } }
        0x14 => Inst::AluAccImm8 { op: AluOp::Adc, imm: cpu.fetch_u8() },
        0x15 => { if w32 { Inst::AluAccImm32 { op: AluOp::Adc, imm: cpu.fetch_u32() } } else { Inst::AluAccImm16 { op: AluOp::Adc, imm: cpu.fetch_u16() } } }

        0x18 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Sbb, m, reg: m.reg, dir: Dir::RmReg } }
        0x19 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Sbb, m, reg: m.reg, dir: Dir::RmReg } } else { Inst::AluRm16Reg { op: AluOp::Sbb, m, reg: m.reg, dir: Dir::RmReg } } }
        0x1A => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Sbb, m, reg: m.reg, dir: Dir::RegRm } }
        0x1B => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Sbb, m, reg: m.reg, dir: Dir::RegRm } } else { Inst::AluRm16Reg { op: AluOp::Sbb, m, reg: m.reg, dir: Dir::RegRm } } }
        0x1C => Inst::AluAccImm8 { op: AluOp::Sbb, imm: cpu.fetch_u8() },
        0x1D => { if w32 { Inst::AluAccImm32 { op: AluOp::Sbb, imm: cpu.fetch_u32() } } else { Inst::AluAccImm16 { op: AluOp::Sbb, imm: cpu.fetch_u16() } } }

        0x20 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::And, m, reg: m.reg, dir: Dir::RmReg } }
        0x21 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::And, m, reg: m.reg, dir: Dir::RmReg } } else { Inst::AluRm16Reg { op: AluOp::And, m, reg: m.reg, dir: Dir::RmReg } } }
        0x22 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::And, m, reg: m.reg, dir: Dir::RegRm } }
        0x23 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::And, m, reg: m.reg, dir: Dir::RegRm } } else { Inst::AluRm16Reg { op: AluOp::And, m, reg: m.reg, dir: Dir::RegRm } } }
        0x24 => Inst::AluAccImm8 { op: AluOp::And, imm: cpu.fetch_u8() },
        0x25 => { if w32 { Inst::AluAccImm32 { op: AluOp::And, imm: cpu.fetch_u32() } } else { Inst::AluAccImm16 { op: AluOp::And, imm: cpu.fetch_u16() } } }

        0x28 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Sub, m, reg: m.reg, dir: Dir::RmReg } }
        0x29 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Sub, m, reg: m.reg, dir: Dir::RmReg } } else { Inst::AluRm16Reg { op: AluOp::Sub, m, reg: m.reg, dir: Dir::RmReg } } }
        0x2A => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Sub, m, reg: m.reg, dir: Dir::RegRm } }
        0x2B => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Sub, m, reg: m.reg, dir: Dir::RegRm } } else { Inst::AluRm16Reg { op: AluOp::Sub, m, reg: m.reg, dir: Dir::RegRm } } }
        0x2C => Inst::AluAccImm8 { op: AluOp::Sub, imm: cpu.fetch_u8() },
        0x2D => { if w32 { Inst::AluAccImm32 { op: AluOp::Sub, imm: cpu.fetch_u32() } } else { Inst::AluAccImm16 { op: AluOp::Sub, imm: cpu.fetch_u16() } } }

        0x30 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Xor, m, reg: m.reg, dir: Dir::RmReg } }
        0x31 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Xor, m, reg: m.reg, dir: Dir::RmReg } } else { Inst::AluRm16Reg { op: AluOp::Xor, m, reg: m.reg, dir: Dir::RmReg } } }
        0x32 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Xor, m, reg: m.reg, dir: Dir::RegRm } }
        0x33 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Xor, m, reg: m.reg, dir: Dir::RegRm } } else { Inst::AluRm16Reg { op: AluOp::Xor, m, reg: m.reg, dir: Dir::RegRm } } }
        0x34 => Inst::AluAccImm8 { op: AluOp::Xor, imm: cpu.fetch_u8() },
        0x35 => { if w32 { Inst::AluAccImm32 { op: AluOp::Xor, imm: cpu.fetch_u32() } } else { Inst::AluAccImm16 { op: AluOp::Xor, imm: cpu.fetch_u16() } } }

        0x38 => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Cmp, m, reg: m.reg, dir: Dir::RmReg } }
        0x39 => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Cmp, m, reg: m.reg, dir: Dir::RmReg } } else { Inst::AluRm16Reg { op: AluOp::Cmp, m, reg: m.reg, dir: Dir::RmReg } } }
        0x3A => { let m = cpu.fetch_modrm(); Inst::AluRm8Reg { op: AluOp::Cmp, m, reg: m.reg, dir: Dir::RegRm } }
        0x3B => { let m = cpu.fetch_modrm(); if w32 { Inst::AluRm32Reg { op: AluOp::Cmp, m, reg: m.reg, dir: Dir::RegRm } } else { Inst::AluRm16Reg { op: AluOp::Cmp, m, reg: m.reg, dir: Dir::RegRm } } }
        0x3C => Inst::AluAccImm8 { op: AluOp::Cmp, imm: cpu.fetch_u8() },
        0x3D => { if w32 { Inst::AluAccImm32 { op: AluOp::Cmp, imm: cpu.fetch_u32() } } else { Inst::AluAccImm16 { op: AluOp::Cmp, imm: cpu.fetch_u16() } } }

        // INC reg (0x40-0x47) / DEC reg (0x48-0x4F)
        // 0x40-0x4F are INC/DEC only in a legacy mode; in 64-bit mode the
        // prefix loop above has already eaten them as REX.
        0x40..=0x47 => { if w32 { Inst::IncReg32 { dst: op - 0x40 } } else { Inst::IncReg16 { dst: op - 0x40 } } }
        0x48..=0x4F => { if w32 { Inst::DecReg32 { dst: op - 0x48 } } else { Inst::DecReg16 { dst: op - 0x48 } } }
        // PUSH reg (0x50-0x57) / POP reg (0x58-0x5F)
        0x50..=0x57 => { let r = (op - 0x50) | rb(cpu); if w32 { Inst::PushReg32 { src: r } } else { Inst::PushReg16 { src: r } } }
        0x58..=0x5F => { let r = (op - 0x58) | rb(cpu); if w32 { Inst::PopReg32 { dst: r } } else { Inst::PopReg16 { dst: r } } }

        // IMUL r, r/m, imm (0x69 imm16/32, 0x6B imm8 sign-extended)
        0x69 => {
            let m = cpu.fetch_modrm();
            if w32 {
                let imm = cpu.fetch_u32() as i32;
                Inst::ImulRegRmImm32 { m, dst: m.reg, imm }
            } else {
                let imm = cpu.fetch_u16() as i16;
                Inst::ImulRegRmImm16 { m, dst: m.reg, imm }
            }
        }
        0x6B => {
            let m = cpu.fetch_modrm();
            let imm = cpu.fetch_u8() as i8;
            if w32 { Inst::ImulRegRmImm32 { m, dst: m.reg, imm: imm as i32 } }
            else { Inst::ImulRegRmImm16 { m, dst: m.reg, imm: imm as i16 } }
        }

        // LEAVE (0xC9)
        0xC9 => Inst::Leave { w32 },
        // RET imm16 (0xC2)
        0xC2 => { let imm = cpu.fetch_u16(); Inst::RetImm { imm, w32 } }

        // PUSH/POP segment register (one-byte forms). POP CS does not exist.
        0x06 => Inst::PushSeg { seg: SegReg::Es },
        0x07 => Inst::PopSeg { seg: SegReg::Es },
        0x0E => Inst::PushSeg { seg: SegReg::Cs },
        0x16 => Inst::PushSeg { seg: SegReg::Ss },
        0x17 => Inst::PopSeg { seg: SegReg::Ss },
        0x1E => Inst::PushSeg { seg: SegReg::Ds },
        0x1F => Inst::PopSeg { seg: SegReg::Ds },

        // XCHG r/m, r (0x86 / 0x87)
        0x86 => { let m = cpu.fetch_modrm(); Inst::XchgRmReg { m, reg: m.reg, width: 8 } }
        0x87 => { let m = cpu.fetch_modrm(); Inst::XchgRmReg { m, reg: m.reg, width: cpu.osize() } }

        // POP r/m (0x8F). Only /0 is defined; other /reg values are invalid.
        0x8F => {
            let m = cpu.fetch_modrm();
            if m.reg & 7 == 0 {
                if w32 { Inst::PopRm32 { m } } else { Inst::PopRm16 { m } }
            } else {
                Inst::Unknown { opcode: 0x008F }
            }
        }

        // PUSHA/PUSHAD (0x60), POPA/POPAD (0x61)
        0x60 => Inst::Pusha { w32 },
        0x61 => Inst::Popa { w32 },

        // PUSH imm8 (0x6A) / PUSH imm16 (0x68) / PUSH imm32 (0x68 w/ 66)
        // PUSH imm8: the immediate is sign-extended to the *operand size*,
        // so in 32-bit mode this pushes four bytes, not two.
        0x6A => {
            let imm = cpu.fetch_u8();
            if w32 { Inst::PushImm32 { imm: imm as i8 as i32 as u32 } }
            else { Inst::PushImm16 { imm: imm as i8 as i16 as u16 } }
        }
        0x68 => { if w32 { Inst::PushImm32 { imm: cpu.fetch_u32() } } else { Inst::PushImm16 { imm: cpu.fetch_u16() } } }

        // Jcc rel8 (0x70-0x7F)
        0x70..=0x7F => Inst::Jcc { cond: Cond::from_jcc(op - 0x70), rel: cpu.fetch_u8() as i8 },

        // 0x80-0x83: group 1 ALU r/m, imm
        0x80 => { let m = cpu.fetch_modrm(); let imm = cpu.fetch_u8(); Inst::AluRm8Imm { op: AluOp::from_index(m.reg), m, imm } }
        0x81 => {
            let m = cpu.fetch_modrm();
            if w32 { let imm = cpu.fetch_u32(); Inst::AluRm32Imm { op: AluOp::from_index(m.reg), m, imm, imm_is8: false } }
            else { let imm = cpu.fetch_u16(); Inst::AluRm16Imm { op: AluOp::from_index(m.reg), m, imm, imm_is8: false } }
        }
        0x83 => {
            let m = cpu.fetch_modrm();
            let imm8 = cpu.fetch_u8() as i8;
            if w32 { Inst::AluRm32Imm { op: AluOp::from_index(m.reg), m, imm: imm8 as i32 as u32, imm_is8: true } }
            else { Inst::AluRm16Imm { op: AluOp::from_index(m.reg), m, imm: imm8 as i16 as u16, imm_is8: true } }
        }

        // TEST r/m8, r8 (0x84) / TEST r/m16/32, r (0x85)
        0x84 => { let m = cpu.fetch_modrm(); Inst::TestRm8Reg { m, reg: m.reg } }
        0x85 => { let m = cpu.fetch_modrm(); if w32 { Inst::TestRm32Reg { m, reg: m.reg } } else { Inst::TestRm16Reg { m, reg: m.reg } } }

        // MOV r/m8, reg8
        0x88 => { let m = cpu.fetch_modrm(); Inst::MovRm8Reg { m, src: m.reg } }
        // MOV r/m16/32, reg
        0x89 => { let m = cpu.fetch_modrm(); if w32 { Inst::MovRm32Reg { m, src: m.reg } } else { Inst::MovRm16Reg { m, src: m.reg } } }
        // MOV reg8, r/m8
        0x8A => { let m = cpu.fetch_modrm(); Inst::MovRegRm8 { m, dst: m.reg } }
        // MOV reg16/32, r/m
        0x8B => { let m = cpu.fetch_modrm(); if w32 { Inst::MovRegRm32 { m, dst: m.reg } } else { Inst::MovRegRm16 { m, dst: m.reg } } }
        // MOV r/m16, sreg
        0x8C => { let m = cpu.fetch_modrm(); Inst::MovRmSeg { m, seg: seg_from_index(m.reg) } }
        // MOV sreg, r/m16
        0x8E => { let m = cpu.fetch_modrm(); Inst::MovSegRm { seg: seg_from_index(m.reg), m } }

        // LDS (0xC5) / LES (0xC4): load a far pointer (offset + segment).
        0xC5 => { let m = cpu.fetch_modrm(); Inst::Lds { m } }
        0xC4 => { let m = cpu.fetch_modrm(); Inst::Les { m } }

        // XCHG AX/EAX, reg (0x90 = NOP when reg is AX)
        // 0x90 is NOP -- and, with REX.B, `xchg %rax,%r8`, which shares
        // the encoding; with an F3 prefix it is PAUSE.
        0x90 if cpu.rex_b => { if w32 { Inst::XchgEaxReg { reg: 8 } } else { Inst::XchgAxReg { reg: 8 } } }
        0x90 => if rep == Rep::Repe { Inst::Pause } else { Inst::Nop },
        0x91..=0x97 => { let r = (op - 0x90) | rb(cpu); if w32 { Inst::XchgEaxReg { reg: r } } else { Inst::XchgAxReg { reg: r } } }

        // CBW (0x98) / CWD (0x99) / CWDE / CDQ
        0x98 => { if w32 { Inst::Cwde } else { Inst::Cbw } }
        0x99 => { if w32 { Inst::Cdq } else { Inst::Cwd } }

        // PUSHF (0x9C) / POPF (0x9D)
        0x9C => Inst::Pushf,
        0x9D => Inst::Popf,

        // MOV reg8, imm8 (0xB0-0xB7)
        0xB0..=0xB7 => Inst::MovReg8Imm { dst: (op - 0xB0) | rb(cpu), imm: cpu.fetch_u8() },
        // MOV reg16/32, imm (0xB8-0xBF)
        0xB8..=0xBF => {
            let r = (op - 0xB8) | rb(cpu);
            // With REX.W this is the one instruction that carries a full
            // 64-bit immediate (`movabs`).
            if cpu.rex_w { Inst::MovReg64Imm { dst: r, imm: cpu.fetch_u64() } }
            else if w32 { Inst::MovReg32Imm { dst: r, imm: cpu.fetch_u32() } }
            else { Inst::MovReg16Imm { dst: r, imm: cpu.fetch_u16() } }
        }

        // RET (near) 0xC3
        0xC3 => { if w32 { Inst::Ret32 } else { Inst::Ret } }
        // RETF (far) 0xCB
        0xCB => { if w32 { Inst::Retf32 } else { Inst::Retf } }
        // MOV r/m8, imm8 (0xC6)
        0xC6 => { let m = cpu.fetch_modrm(); let imm = cpu.fetch_u8(); Inst::MovRm8Imm { m, imm } }
        // MOV r/m16/32, imm (0xC7)
        0xC7 => {
            let m = cpu.fetch_modrm();
            if w32 { let imm = cpu.fetch_u32(); Inst::MovRm32Imm { m, imm } }
            else { let imm = cpu.fetch_u16(); Inst::MovRm16Imm { m, imm } }
        }
        // INT imm8 (0xCD)
        0xCD => Inst::Int { vector: cpu.fetch_u8() },
        // INT3 (0xCC) / INTO (0xCE)
        0xCC => Inst::Int3,
        0xCE => Inst::Into,
        // IRET (0xCF)
        0xCF => { if w32 { Inst::Iret32 } else { Inst::Iret } }

        // Group 2 shifts/rotates: 0xD0 (r/m8, 1), 0xD1 (r/m16/32, 1),
        // 0xD2 (r/m8, CL), 0xD3 (r/m16/32, CL)
        0xD0 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, width: 8, count: ShiftCount::One } }
        0xD1 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, width: cpu.osize(), count: ShiftCount::One } }
        0xD2 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, width: 8, count: ShiftCount::Cl } }
        0xD3 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, width: cpu.osize(), count: ShiftCount::Cl } }
        // Group 2 shifts/rotates with imm8 count: 0xC0 (r/m8, imm8),
        // 0xC1 (r/m16/32, imm8)
        0xC0 => { let m = cpu.fetch_modrm(); let imm = cpu.fetch_u8(); Inst::ShiftImm { op: ShiftOp::from_index(m.reg), m, width: 8, imm } }
        0xC1 => { let m = cpu.fetch_modrm(); let imm = cpu.fetch_u8(); Inst::ShiftImm { op: ShiftOp::from_index(m.reg), m, width: cpu.osize(), imm } }

        // CALL rel16/32 (0xE8)
        0xE8 => { if w32 { Inst::CallRel32 { rel: cpu.fetch_u32() as i32 } } else { Inst::CallRel16 { rel: cpu.fetch_u16() as i16 } } }
        // JMP rel16/32 (0xE9)
        0xE9 => { if w32 { Inst::JmpRel32 { rel: cpu.fetch_u32() as i32 } } else { Inst::JmpRel16 { rel: cpu.fetch_u16() as i16 } } }
        // JMP rel8 (0xEB)
        0xEB => Inst::JmpRel8 { rel: cpu.fetch_u8() as i8 },

        // LOOP / LOOPZ / LOOPNZ / JCXZ (0xE0-0xE3)
        0xE0 => Inst::Loop { cond: LoopCond::Loop, rel: cpu.fetch_u8() as i8 },
        0xE1 => Inst::Loop { cond: LoopCond::Loopz, rel: cpu.fetch_u8() as i8 },
        0xE2 => Inst::Loop { cond: LoopCond::Loopnz, rel: cpu.fetch_u8() as i8 },
        0xE3 => Inst::Loop { cond: LoopCond::Jcxz, rel: cpu.fetch_u8() as i8 },

        // WAIT / FWAIT (0x9B): wait for the FPU. No-op (we execute FPU
        // instructions synchronously).
        0x9B => Inst::Nop,
        // HLT (0xF4)
        0xF4 => Inst::Hlt,
        // CMC (0xF5)
        0xF5 => Inst::Cmc,
        // CLC (0xF8) / STC (0xF9) / CLI (0xFA) / STI (0xFB) / CLD (0xFC) / STD (0xFD)
        0xF8 => Inst::Clc,
        0xF9 => Inst::Stc,
        0xFA => Inst::Cli,
        0xFB => Inst::Sti,
        0xFC => Inst::Cld,
        0xFD => Inst::Std,

        // IN/OUT with immediate port (0xE4-0xE7)
        0xE4 => Inst::InAlImm { port: cpu.fetch_u8() },
        0xE5 => Inst::InAxImm { port: cpu.fetch_u8() },
        0xE6 => Inst::OutImmAl { port: cpu.fetch_u8() },
        0xE7 => Inst::OutImmAx { port: cpu.fetch_u8() },
        // IN/OUT with DX port (0xEC-0xEF)
        0xEC => Inst::InAlDx,
        0xED => Inst::InAxDx,
        0xEE => Inst::OutDxAl,
        0xEF => Inst::OutDxAx,

        // Group 3 (0xF6/0xF7): TEST / NOT / NEG / MUL / IMUL / DIV / IDIV
        0xF6 => {
            let m = cpu.fetch_modrm();
            match m.reg & 7 {
                0 => { let imm = cpu.fetch_u8(); Inst::TestRm8Imm { m, imm } }
                2 => Inst::NotRm8 { m },
                3 => Inst::NegRm8 { m },
                4 => Inst::MulRm8 { m },
                5 => Inst::ImulRm8 { m },
                6 => Inst::DivRm8 { m },
                _ => Inst::IdivRm8 { m },
            }
        }
        0xF7 => {
            let m = cpu.fetch_modrm();
            match m.reg & 7 {
                0 => {
                    if w32 { let imm = cpu.fetch_u32(); Inst::TestRm32Imm { m, imm } }
                    else { let imm = cpu.fetch_u16(); Inst::TestRm16Imm { m, imm } }
                }
                2 => { if w32 { Inst::NotRm32 { m } } else { Inst::NotRm16 { m } } }
                3 => { if w32 { Inst::NegRm32 { m } } else { Inst::NegRm16 { m } } }
                4 => { if w32 { Inst::MulRm32 { m } } else { Inst::MulRm16 { m } } }
                5 => { if w32 { Inst::ImulRm32 { m } } else { Inst::ImulRm16 { m } } }
                6 => { if w32 { Inst::DivRm32 { m } } else { Inst::DivRm16 { m } } }
                _ => { if w32 { Inst::IdivRm32 { m } } else { Inst::IdivRm16 { m } } }
            }
        }

        // Group 5 (0xFF): INC / DEC / CALL / JMP / PUSH r/m
        // Group 4 (0xFE): INC /0 and DEC /1 on a byte operand.
        0xFE => {
            let m = cpu.fetch_modrm();
            match m.reg & 7 {
                0 => Inst::IncRm8 { m },
                1 => Inst::DecRm8 { m },
                _ => Inst::Unknown { opcode: 0x00FE },
            }
        }

        0xFF => {
            let m = cpu.fetch_modrm();
            match m.reg & 7 {
                0 => { if w32 { Inst::IncRm32 { m } } else { Inst::IncRm16 { m } } }
                1 => { if w32 { Inst::DecRm32 { m } } else { Inst::DecRm16 { m } } }
                2 => { if w32 { Inst::CallRm32 { m } } else { Inst::CallRm16 { m } } }
                4 => { if w32 { Inst::JmpRm32 { m } } else { Inst::JmpRm16 { m } } }
                6 => { if w32 { Inst::PushRm32 { m } } else { Inst::PushRm16 { m } } }
                _ => Inst::Unknown { opcode: 0x00FF },
            }
        }

        // MOV AL, moffs8 / MOV moffs8, AL (0xA0/0xA2)
        // The moffs width follows the ADDRESS size (addrsize), not the
        // operand size. In 32-bit addressing mode the moffs is 32-bit.
        0xA0 => {
            if cpu.addrsize { Inst::MovAccMem8Addr32 { addr: cpu.fetch_u32() } }
            else { Inst::MovAccMem8 { addr: cpu.fetch_u16() } }
        }
        0xA2 => {
            if cpu.addrsize { Inst::MovMem8AccAddr32 { addr: cpu.fetch_u32() } }
            else { Inst::MovMem8Acc { addr: cpu.fetch_u16() } }
        }
        // TEST AL, imm8 (0xA8) / TEST AX/EAX, imm (0xA9)
        0xA8 => Inst::TestAccImm8 { imm: cpu.fetch_u8() },
        0xA9 => {
            if w32 { Inst::TestAccImm32 { imm: cpu.fetch_u32() } }
            else { Inst::TestAccImm16 { imm: cpu.fetch_u16() } }
        }
        // MOV AX/EAX, moffs / MOV moffs, AX/EAX (0xA1/0xA3)
        // The moffs width follows the ADDRESS size.
        0xA1 => {
            if cpu.addr64 { Inst::MovAccMem32 { addr: cpu.fetch_u64() } }
            else if cpu.addrsize {
                if w32 { Inst::MovAccMem32 { addr: cpu.fetch_u32() as u64 } }
                else { Inst::MovAccMem16Addr32 { addr: cpu.fetch_u32() } }
            } else {
                if w32 { Inst::MovAccMem32 { addr: cpu.fetch_u16() as u64 } }
                else { Inst::MovAccMem16 { addr: cpu.fetch_u16() } }
            }
        }
        0xA3 => {
            if cpu.addr64 { Inst::MovMem32Acc { addr: cpu.fetch_u64() } }
            else if cpu.addrsize {
                if w32 { Inst::MovMem32Acc { addr: cpu.fetch_u32() as u64 } }
                else { Inst::MovMem16AccAddr32 { addr: cpu.fetch_u32() } }
            } else {
                if w32 { Inst::MovMem32Acc { addr: cpu.fetch_u16() as u64 } }
                else { Inst::MovMem16Acc { addr: cpu.fetch_u16() } }
            }
        }

        // 0x63: ARPL in a legacy mode (a protection check this emulator does
        // not enforce), MOVSXD in 64-bit mode — where it is how every 32-bit
        // value becomes a 64-bit index.
        0x63 => {
            let m = cpu.fetch_modrm();
            if cpu.long64() { Inst::Movsxd { m, dst: m.reg } } else { Inst::Nop }
        }

        // LEA reg16/32, m (0x8D)
        0x8D => { let m = cpu.fetch_modrm(); Inst::Lea { m, dst: m.reg } }

        // String ops: MOVS (0xA4/0xA5), STOS (0xAA/0xAB),
        // LODS (0xAC/0xAD), CMPS (0xA6/0xA7), SCAS (0xAE/0xAF)
        // Byte forms have w=false; word forms have w=true (element size is
        // 2 in 16-bit mode, 4 in 32-bit mode, resolved at execute time).
        0xA4 => Inst::Movs { rep, w: false },
        0xA5 => Inst::Movs { rep, w: true },
        0xA6 => Inst::Cmps { rep, w: false },
        0xA7 => Inst::Cmps { rep, w: true },
        0xAA => Inst::Stos { rep, w: false },
        0xAB => Inst::Stos { rep, w: true },
        0xAC => Inst::Lods { rep, w: false },
        0xAD => Inst::Lods { rep, w: true },
        0xAE => Inst::Scas { rep, w: false },
        0xAF => Inst::Scas { rep, w: true },

        // Far JMP (0xEA) / far CALL (0x9A)
        0xEA => {
            if w32 { let off = cpu.fetch_u32(); let seg = cpu.fetch_u16(); Inst::JmpFar32 { off, seg } }
            else { let off = cpu.fetch_u16(); let seg = cpu.fetch_u16(); Inst::JmpFar { off, seg } }
        }
        0x9A => {
            if w32 { let off = cpu.fetch_u32(); let seg = cpu.fetch_u16(); Inst::CallFar32 { off, seg } }
            else { let off = cpu.fetch_u16(); let seg = cpu.fetch_u16(); Inst::CallFar { off, seg } }
        }

        // 0x0F escape: LGDT/LIDT (0x0F 0x01 /2 and /3)
        0x0F => {
            let op2 = cpu.fetch_u8();
            match op2 {
                0x01 => {
                    let m = cpu.fetch_modrm();
                    // The register forms of /7 are not INVLPG at all: F8 is
                    // SWAPGS and F9 is RDTSCP, which is why the mod field has
                    // to be looked at before the reg field is trusted.
                    if m.mod_field == 3 {
                        return match (m.reg & 7, m.rm_raw) {
                            (7, 0) => Inst::Swapgs,
                            (7, 1) => Inst::Rdtscp,
                            // /0 register forms: VMCALL (C1), VMLAUNCH (C2),
                            // VMRESUME (C3), VMXOFF (C4).
                            (0, 1) => Inst::Vmx(crate::vmx::VmxInst { op: crate::vmx::VmxOp::Vmcall, m, reg: 0 }),
                            (0, 2) => Inst::Vmx(crate::vmx::VmxInst { op: crate::vmx::VmxOp::Vmlaunch, m, reg: 0 }),
                            (0, 3) => Inst::Vmx(crate::vmx::VmxInst { op: crate::vmx::VmxOp::Vmresume, m, reg: 0 }),
                            (0, 4) => Inst::Vmx(crate::vmx::VmxInst { op: crate::vmx::VmxOp::Vmxoff, m, reg: 0 }),
                            // /1 register forms: MONITOR (C8) and MWAIT (C9).
                            // One core and no caches: MONITOR arms nothing,
                            // and MWAIT is a HLT that also wakes on the
                            // interrupt it would have waited for.
                            (1, 0) => Inst::Monitor,
                            (1, 1) => Inst::Mwait,
                            // /1: CLAC (CA) and STAC (CB).
                            (1, 2) => Inst::Clac,
                            (1, 3) => Inst::Stac,
                            // /2: XGETBV (D0) and XSETBV (D1).
                            (2, 0) => Inst::Xgetbv,
                            (2, 1) => Inst::Xsetbv,
                            // /4 and /6 have register forms too: SMSW r and
                            // LMSW r are the same instructions with a
                            // register operand.
                            (4, _) => Inst::Smsw { m },
                            (6, _) => Inst::Lmsw { m },
                            _ => Inst::Unknown { opcode: 0x0F01 },
                        };
                    }
                    match m.reg & 7 {
                        0 => Inst::Sgdt { m },
                        1 => Inst::Sidt { m },
                        2 => Inst::Lgdt { m },
                        3 => Inst::Lidt { m },
                        4 => Inst::Smsw { m },
                        6 => Inst::Lmsw { m },
                        7 => Inst::Invlpg { m },
                        _ => Inst::Unknown { opcode: 0x0F00 | op2 as u16 },
                    }
                }
                // SYSCALL / SYSRET: the fast system-call pair. They replace
                // `int 0x80` entirely on 64-bit, so a 64-bit userspace cannot
                // make a single call without them.
                0x05 => Inst::Syscall,
                0x07 => Inst::Sysret,
                // Cache and store-ordering instructions. This emulator has
                // one core, no caches and no store buffer, so they are
                // architecturally complete as no-ops -- but they must be
                // *decoded*, or a kernel's `mfence` reads as a bad opcode.
                0x08 => Inst::Invd,
                0x09 => Inst::Wbinvd,
                0x0D => { cpu.fetch_modrm(); Inst::NopHint }  // prefetch hints
                0x18 => { cpu.fetch_modrm(); Inst::NopHint }  // PREFETCHh
                // The multi-byte NOP (0F 1F /0). Compilers emit it by the
                // yard to align 64-bit branch targets.
                0x1F => { cpu.fetch_modrm(); Inst::NopHint }
                // 0F 00 group: SLDT (/0), STR (/1), LLDT (/2), LTR (/3).
                // LTR is load-bearing once user mode exists: the TSS it names
                // holds the ring-0 stack an interrupt from CPL 3 switches to.
                0x00 => {
                    let m = cpu.fetch_modrm();
                    match m.reg & 7 {
                        0 => Inst::Sldt { m },
                        1 => Inst::Str { m },
                        2 => Inst::Lldt { m },
                        3 => Inst::Ltr { m },
                        // /4 VERR and /5 VERW: segment access checks this
                        // emulator does not enforce, so they do nothing.
                        _ => Inst::Nop,
                    }
                }
                // 0F 80-8F: Jcc rel32 (32-bit conditional jumps). Same
                // conditions as the 0x70-0x7F rel8 forms.
                0x80..=0x8F => {
                    Inst::Jcc32 { cond: Cond::from_jcc(op2 - 0x80), rel: cpu.fetch_u32() as i32 }
                }
                // MOVZX r16/32, r/m8 (0F B6) / MOVZX r32, r/m16 (0F B7)
                0xB6 => {
                    let m = cpu.fetch_modrm();
                    Inst::Movzx8 { m, dst: m.reg }
                }
                0xB7 => {
                    let m = cpu.fetch_modrm();
                    Inst::Movzx16 { m, dst: m.reg }
                }
                // MOVSX r16/32, r/m8 (0F BE) / MOVSX r32, r/m16 (0F BF)
                0xBE => {
                    let m = cpu.fetch_modrm();
                    Inst::Movsx8 { m, dst: m.reg }
                }
                0xBF => {
                    let m = cpu.fetch_modrm();
                    Inst::Movsx16 { m, dst: m.reg }
                }
                // CLTS (0x0F 0x06)
                0x06 => Inst::Clts,
                // MOV r32, cr (0x0F 0x20) / MOV cr, r32 (0x0F 0x22). The
                // ModR/M mod field is ignored (the operand is always a
                // register), so REX.B is applied to the raw rm bits here
                // rather than trusting `m.rm`, which only carries it when
                // mod == 3. Masking to three bits turned `mov %cr4,%r12`
                // into a write to RSP -- inside the kernel's own crash
                // reporter, which then faulted on the stack it had just lost.
                // REX.R on the reg field is what names CR8.
                0x20 => {
                    let m = cpu.fetch_modrm();
                    Inst::MovCr { cr: m.reg, reg: m.rm_raw | rb(cpu) }
                }
                0x22 => {
                    let m = cpu.fetch_modrm();
                    Inst::MovToCr { cr: m.reg, reg: m.rm_raw | rb(cpu) }
                }
                // CPUID (0x0F 0xA2)
                0xA2 => Inst::Cpuid,
                // RDTSC (0x0F 0x31)
                0x31 => Inst::Rdtsc,
                // RDMSR (0x0F 0x32) / WRMSR (0x0F 0x30)
                0x32 => Inst::Rdmsr,
                0x30 => Inst::Wrmsr,
                // Bit tests: BT/BTS/BTR/BTC with register operand
                0xA3 => { let m = cpu.fetch_modrm(); Inst::Bt { m, bit: BitOffset::Reg(m.reg) } }
                0xAB => { let m = cpu.fetch_modrm(); Inst::Bts { m, bit: BitOffset::Reg(m.reg) } }
                0xB3 => { let m = cpu.fetch_modrm(); Inst::Btr { m, bit: BitOffset::Reg(m.reg) } }
                0xBB => { let m = cpu.fetch_modrm(); Inst::Btc { m, bit: BitOffset::Reg(m.reg) } }
                // Load segment with pointer: LSS (0F B2) / LFS (0F B4) / LGS (0F B5)
                0xB2 => { let m = cpu.fetch_modrm(); Inst::Lss { m } }
                0xB4 => { let m = cpu.fetch_modrm(); Inst::Lfs { m } }
                0xB5 => { let m = cpu.fetch_modrm(); Inst::Lgs { m } }
                // Group 8 (0F BA /4-/7): bit tests with imm8
                0xBA => {
                    let m = cpu.fetch_modrm();
                    let imm = cpu.fetch_u8();
                    match m.reg & 7 {
                        4 => Inst::Bt { m, bit: BitOffset::Imm(imm) },
                        5 => Inst::Bts { m, bit: BitOffset::Imm(imm) },
                        6 => Inst::Btr { m, bit: BitOffset::Imm(imm) },
                        _ => Inst::Btc { m, bit: BitOffset::Imm(imm) },
                    }
                }
                // IMUL r16/32, r/m (0F AF)
                0xAF => {
                    let m = cpu.fetch_modrm();
                    if w32 { Inst::ImulRegRm32 { m, dst: m.reg } }
                    else { Inst::ImulRegRm16 { m, dst: m.reg } }
                }
                // SHLD/SHRD, by imm8 (0F A4 / 0F AC) or by CL (0F A5 / 0F AD)
                0xA4 => {
                    let m = cpu.fetch_modrm();
                    let imm = cpu.fetch_u8();
                    Inst::Shld { m, reg: m.reg, count: ShiftCount::Imm(imm), w32 }
                }
                0xA5 => {
                    let m = cpu.fetch_modrm();
                    Inst::Shld { m, reg: m.reg, count: ShiftCount::Cl, w32 }
                }
                0xAC => {
                    let m = cpu.fetch_modrm();
                    let imm = cpu.fetch_u8();
                    Inst::Shrd { m, reg: m.reg, count: ShiftCount::Imm(imm), w32 }
                }
                0xAD => {
                    let m = cpu.fetch_modrm();
                    Inst::Shrd { m, reg: m.reg, count: ShiftCount::Cl, w32 }
                }
                // PUSH/POP FS and GS (0F A0/A1, 0F A8/A9)
                0xA0 => Inst::PushSeg { seg: SegReg::Fs },
                0xA1 => Inst::PopSeg { seg: SegReg::Fs },
                0xA8 => Inst::PushSeg { seg: SegReg::Gs },
                0xA9 => Inst::PopSeg { seg: SegReg::Gs },
                // MOV r32, DRx (0F 21) / MOV DRx, r32 (0F 23)
                0x21 => { let m = cpu.fetch_modrm(); Inst::MovDr { dr: m.reg & 7, reg: m.rm_raw | rb(cpu) } }
                0x23 => { let m = cpu.fetch_modrm(); Inst::MovToDr { dr: m.reg & 7, reg: m.rm_raw | rb(cpu) } }
                // Bit scan forward / reverse (0F BC / 0F BD)
                0xBC => { let m = cpu.fetch_modrm(); Inst::Bsf { m, dst: m.reg, w32 } }
                0xBD => { let m = cpu.fetch_modrm(); Inst::Bsr { m, dst: m.reg, w32 } }
                // CMPXCHG (0F B0 / 0F B1)
                0xB0 => { let m = cpu.fetch_modrm(); Inst::Cmpxchg { m, reg: m.reg, width: 8 } }
                0xB1 => { let m = cpu.fetch_modrm(); Inst::Cmpxchg { m, reg: m.reg, width: cpu.osize() } }
                // XADD (0F C0 / 0F C1)
                0xC0 => { let m = cpu.fetch_modrm(); Inst::Xadd { m, reg: m.reg, width: 8 } }
                0xC1 => { let m = cpu.fetch_modrm(); Inst::Xadd { m, reg: m.reg, width: cpu.osize() } }
                // CMPXCHG8B m64 (0F C7 /1)
                0xC7 => {
                    let m = cpu.fetch_modrm();
                    if m.reg & 7 == 1 { Inst::Cmpxchg8b { m } }
                    else {
                        // /6 and /7 memory forms are VMPTRLD/VMCLEAR/VMXON
                        // and VMPTRST; the register forms would be
                        // RDRAND/RDSEED, which this CPU does not claim.
                        crate::vmx::decode_0f_c7(cpu, m).unwrap_or(Inst::Unknown { opcode: 0x0FC7 })
                    }
                }
                // VMREAD r/m64, r64 (0F 78) and VMWRITE r64, r/m64 (0F 79):
                // the field encoding is in `reg` for both.
                0x78 => { let m = cpu.fetch_modrm(); Inst::Vmx(crate::vmx::VmxInst { op: crate::vmx::VmxOp::Vmread, m, reg: m.reg }) }
                0x79 => { let m = cpu.fetch_modrm(); Inst::Vmx(crate::vmx::VmxInst { op: crate::vmx::VmxOp::Vmwrite, m, reg: m.reg }) }
                // The 0F 38 three-byte escape: INVEPT (66 0F 38 80) and
                // INVVPID (66 0F 38 81) are the only forms here.
                0x38 => {
                    let op3 = cpu.fetch_u8();
                    let m = cpu.fetch_modrm();
                    match (cpu.sse_pfx, op3) {
                        (Some(0x66), 0x80) => Inst::Vmx(crate::vmx::VmxInst { op: crate::vmx::VmxOp::Invept, m, reg: m.reg }),
                        (Some(0x66), 0x81) => Inst::Vmx(crate::vmx::VmxInst { op: crate::vmx::VmxOp::Invvpid, m, reg: m.reg }),
                        _ => Inst::Unknown { opcode: 0x0F38 },
                    }
                }
                // BSWAP r32 (0F C8+r)
                0xC8..=0xCF => Inst::Bswap { reg: (op2 - 0xC8) | rb(cpu) },
                // CMOVcc r, r/m (0F 40-4F)
                0x40..=0x4F => {
                    let m = cpu.fetch_modrm();
                    Inst::Cmovcc { cond: Cond::from_jcc(op2 & 0xF), m, dst: m.reg, w32 }
                }
                // Multi-byte NOP (0F 1F /0) and the prefetch/NOP hints
                // (0F 18-1F): they take a ModR/M and do nothing. GCC pads
                // with these, so they turn up in ordinary kernel text.
                0x18..=0x1F => { let _ = cpu.fetch_modrm(); Inst::Nop }
                // UD2 (0F 0B): an intentional invalid opcode. The kernel's
                // BUG() macro is exactly this, so it must raise #UD.
                0x0B => Inst::Unknown { opcode: 0x0F0B },
                // SETcc r/m8 (0F 90-9F)
                0x90..=0x9F => {
                    let m = cpu.fetch_modrm();
                    Inst::Setcc { cond: Cond::from_jcc(op2 & 0xF), m }
                }
                // Group 15 (0F AE): FXSAVE /0, FXRSTOR /1, LDMXCSR /2,
                // STMXCSR /3, LFENCE /5, MFENCE /6, CLFLUSH /7. The fences
                // and the cache flush are no-ops on an in-order emulator
                // with no cache -- but they must decode, because the kernel
                // patches MFENCE in as its memory barrier.
                0xAE => {
                    let m = cpu.fetch_modrm();
                    // The *register* forms are the fences; only the memory
                    // forms are FXSAVE/FXRSTOR, and treating `mfence` as an
                    // FXSAVE with reg=6 would have it write 512 bytes of FPU
                    // state over whatever a register number resolved to.
                    if m.mod_field == 3 {
                        Inst::NopHint
                    } else if (m.reg & 7) == 7 {
                        // CLFLUSH: no cache to flush.
                        Inst::NopHint
                    } else {
                        // The memory forms are FXSAVE/FXRSTOR/LDMXCSR/STMXCSR;
                        // /4-/6 are XSAVE-family forms this CPU does not have.
                        crate::sse::decode_0f_ae(m)
                            .unwrap_or(Inst::Unknown { opcode: 0x0FAE })
                    }
                }
                // ---- SSE/SSE2/SSE3 (0F 10-17, 28-2F, 50-7F, C2-C6, D0-FE) ----
                // Everything on an XMM register lives in `sse.rs`; the
                // mandatory prefix the loop above recorded picks the form.
                0x10..=0x17 | 0x28..=0x2F | 0x50..=0x76 | 0x7C..=0x7F | 0xC2..=0xC6
                | 0xD0..=0xFE => {
                    crate::sse::decode_sse(cpu, op2)
                        .unwrap_or(Inst::Unknown { opcode: 0x0F00 | op2 as u16 })
                }
                _ => Inst::Unknown { opcode: 0x0F00 | op2 as u16 },
            }
        }

        // ---- x87 FPU (D8-DF) ----
        // D9: FLD m32 (/0), FSTCW (/7), FLDCW (/5), FST m32 (/2),
        //     FSTP m32 (/3), FNSTCW (/7), FNSTSW (/7 w/ mod=3,rm=4).
        0xD9 => {
            let m = cpu.fetch_modrm();
            match m.reg & 7 {
                0 => Inst::Fld { m, w64: false },
                2 => Inst::Fst { m, w64: false },
                3 => Inst::Fstp { m, w64: false },
                5 => Inst::Fldcw { m },
                7 => {
                    // mod=3, rm=4 is FNSTSW AX (D9 E0); otherwise FSTCW.
                    if m.is_reg() && m.rm == 4 { Inst::FstswAx }
                    else { Inst::Fstcw { m } }
                }
                _ => Inst::Fstcw { m },
            }
        }
        // DD: FLD m64 (/0), FST m64 (/2), FSTP m64 (/3), FSTSW m16 (/7).
        0xDD => {
            let m = cpu.fetch_modrm();
            match m.reg & 7 {
                0 => Inst::Fld { m, w64: true },
                2 => Inst::Fst { m, w64: true },
                3 => Inst::Fstp { m, w64: true },
                7 => Inst::Fstsw { m },
                _ => Inst::Fst { m, w64: true },
            }
        }
        // DB: FILD m32 (/0), FISTP m32 (/3), FNINIT (DB E3).
        0xDB => {
            let m = cpu.fetch_modrm();
            match m.reg & 7 {
                0 => Inst::Fild { m },
                3 => Inst::Fistp { m },
                4 => {
                    // /4 group: FNINIT (DB E3), FNCLEX (DB E2, no-op).
                    if m.is_reg() && m.rm == 3 { Inst::Fninit }
                    else { Inst::Fstcw { m } }
                }
                7 => {
                    // DB E0 = FNSTSW AX.
                    if m.is_reg() && m.rm == 0 { Inst::FstswAx }
                    else { Inst::Fstcw { m } }
                }
                _ => Inst::Fild { m },
            }
        }
        // DF: FILD m16 (/0), FISTP m16 (/3), FSTSW AX (DF E0).
        0xDF => {
            let m = cpu.fetch_modrm();
            match m.reg & 7 {
                0 => Inst::Fild { m },
                3 => Inst::Fistp { m },
                4 => {
                    // /4 group: FNSTSW AX (DF E0), FSTSW m16 (DD /7 is
                    // separate; DF E0 is the AX form).
                    if m.is_reg() && m.rm == 0 { Inst::FstswAx }
                    else { Inst::Fstsw { m } }
                }
                _ => Inst::Fild { m },
            }
        }
        // D8/DC/DA/DE: arithmetic with m32/m64/m16. Simplified to ST0 op m.
        0xD8 | 0xDC => {
            let m = cpu.fetch_modrm();
            let op = match m.reg & 7 {
                0 => FpuOp::Add,
                1 => FpuOp::Mul,
                4 => FpuOp::Sub,
                6 => FpuOp::Div,
                _ => FpuOp::Add,
            };
            Inst::Fop { op, m }
        }
        0xDA | 0xDE => {
            let m = cpu.fetch_modrm();
            let op = match m.reg & 7 {
                0 => FpuOp::Add,
                1 => FpuOp::Mul,
                4 => FpuOp::Sub,
                6 => FpuOp::Div,
                _ => FpuOp::Add,
            };
            Inst::Fop { op, m }
        }

        _ => Inst::Unknown { opcode: op as u16 },
    }
}

/// Merge a software-written EFLAGS value onto the current one: only the
/// writable bits move, and the reserved bits keep their architectural values.
fn write_flags(old: u32, new: u32) -> u32 {
    (new & flags::WRITABLE) | flags::ALWAYS_SET | (old & !flags::WRITABLE & !flags::ALWAYS_SET)
}

fn seg_from_index(i: u8) -> SegReg {
    match i & 7 {
        0 => SegReg::Es, 1 => SegReg::Cs, 2 => SegReg::Ss, 3 => SegReg::Ds,
        4 => SegReg::Fs, _ => SegReg::Gs,
    }
}

// ---- Executor ----

pub fn execute(cpu: &mut Cpu, inst: &Inst) {
    use flags::*;
    match *inst {
        Inst::Nop | Inst::Pause | Inst::Monitor | Inst::Invd | Inst::Wbinvd => {}
        Inst::Mwait => { cpu.halted = true; }
        Inst::Hlt => { cpu.halted = true; }

        // ---- MOV ----
        Inst::MovRm8Reg { m, src } => {
            let v = cpu.reg8_idx(src);
            cpu.write_rm8(&m, v);
        }
        Inst::MovRm16Reg { m, src } => {
            let v = cpu.reg16_idx(src);
            cpu.write_rm16(&m, v);
        }
        Inst::MovRm32Reg { m, src } => {
            let w = cpu.osize();
            let v = cpu.reg_w(src, w);
            cpu.write_rm_w(&m, w, v);
        }
        Inst::MovRegRm8 { m, dst } => {
            let v = cpu.read_rm8(&m);
            cpu.set_reg8_idx(dst, v);
        }
        Inst::MovRegRm16 { m, dst } => {
            let v = cpu.read_rm16(&m);
            cpu.set_reg16_idx(dst, v);
        }
        Inst::MovRegRm32 { m, dst } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w);
            cpu.set_reg_w(dst, w, v);
        }
        Inst::MovRm8Imm { m, imm } => cpu.write_rm8(&m, imm),
        Inst::MovRm16Imm { m, imm } => cpu.write_rm16(&m, imm),
        // C7 /0 takes an imm32 even at 64-bit operand size, and
        // **sign-extends** it: `movq $-1,(%rax)` is five bytes, not twelve.
        Inst::MovRm32Imm { m, imm } => {
            let w = cpu.osize();
            cpu.write_rm_w(&m, w, sext(imm as u64, 32));
        }
        Inst::MovReg8Imm { dst, imm } => cpu.set_reg8_idx(dst, imm),
        Inst::MovReg16Imm { dst, imm } => cpu.set_reg16_idx(dst, imm),
        Inst::MovReg32Imm { dst, imm } => cpu.set_reg_w(dst, 32, imm as u64),
        // B8+r with REX.W: the one instruction that carries a full 64-bit
        // immediate, and the only way to load an arbitrary 64-bit constant in
        // one go.
        Inst::MovReg64Imm { dst, imm } => cpu.set_reg_w(dst, 64, imm),
        Inst::MovAccMem8 { addr } => {
            let phys = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), (addr as u32) as u64);
            cpu.set_reg8(Reg8::Al, cpu.mem.read_u8(phys));
        }
        Inst::MovAccMem8Addr32 { addr } => {
            let phys = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), (addr) as u64);
            cpu.set_reg8(Reg8::Al, cpu.mem.read_u8(phys));
        }
        Inst::MovMem8Acc { addr } => {
            let phys = cpu.translate_write(cpu.operand_seg_for_exec(SegReg::Ds), (addr as u32) as u64);
            cpu.mem.write_u8(phys, cpu.reg8(Reg8::Al));
        }
        Inst::MovMem8AccAddr32 { addr } => {
            let phys = cpu.translate_write(SegReg::Ds, (addr) as u64);
            cpu.mem.write_u8(phys, cpu.reg8(Reg8::Al));
        }
        Inst::MovAccMem16 { addr } => {
            let phys = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), (addr as u32) as u64);
            cpu.set_reg16(Reg16::Ax, cpu.mem.read_u16(phys));
        }
        Inst::MovAccMem16Addr32 { addr } => {
            let phys = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), (addr) as u64);
            cpu.set_reg16(Reg16::Ax, cpu.mem.read_u16(phys));
        }
        Inst::MovMem16Acc { addr } => {
            let phys = cpu.translate_write(cpu.operand_seg_for_exec(SegReg::Ds), (addr as u32) as u64);
            cpu.mem.write_u16(phys, cpu.reg16(Reg16::Ax));
        }
        Inst::MovMem16AccAddr32 { addr } => {
            let phys = cpu.translate_write(SegReg::Ds, (addr) as u64);
            cpu.mem.write_u16(phys, cpu.reg16(Reg16::Ax));
        }
        Inst::MovAccMem32 { addr } => {
            let w = cpu.osize();
            let seg = cpu.operand_seg_for_exec(SegReg::Ds);
            let phys = cpu.translate(seg, addr);
            let v = mem_read_w(cpu, phys, w);
            cpu.set_reg_w(0, w, v);
        }
        Inst::MovMem32Acc { addr } => {
            let w = cpu.osize();
            let seg = cpu.operand_seg_for_exec(SegReg::Ds);
            let phys = cpu.translate_write(seg, addr);
            let v = cpu.reg_w(0, w);
            mem_write_w(cpu, phys, w, v);
        }
        Inst::MovRmSeg { m, seg } => {
            let v = cpu.seg(seg);
            cpu.write_rm16(&m, v);
        }
        Inst::MovSegRm { seg, m } => {
            let v = cpu.read_rm16(&m);
            cpu.load_seg(seg, v);
        }
        // ---- Load segment with pointer (LDS/LES/LSS/LFS/LGS) ----
        // Reads a far pointer (offset, then segment) from the memory operand,
        // loads the segment register, and stores the offset in the reg field.
        Inst::Lds { m } => { load_far_pointer(cpu, &m, SegReg::Ds); }
        Inst::Les { m } => { load_far_pointer(cpu, &m, SegReg::Es); }
        Inst::Lss { m } => { load_far_pointer(cpu, &m, SegReg::Ss); }
        Inst::Lfs { m } => { load_far_pointer(cpu, &m, SegReg::Fs); }
        Inst::Lgs { m } => { load_far_pointer(cpu, &m, SegReg::Gs); }

        // ---- ALU r/m, reg ----
        Inst::AluRm8Reg { op, m, reg, dir } => {
            let regv = cpu.reg8_idx(reg);
            let rmv = cpu.read_rm8(&m);
            let (a, b, store) = match dir {
                Dir::RmReg => (rmv, regv, true),
                Dir::RegRm => (regv, rmv, false),
            };
            let result = alu8(cpu, op, a, b);
            // CMP sets flags only — it must never write its result back.
            if op != AluOp::Cmp {
                if store { cpu.write_rm8(&m, result); } else { cpu.set_reg8_idx(reg, result); }
            }
        }
        Inst::AluRm16Reg { op, m, reg, dir } => {
            let regv = cpu.reg16_idx(reg);
            let rmv = cpu.read_rm16(&m);
            let (a, b, store) = match dir {
                Dir::RmReg => (rmv, regv, true),
                Dir::RegRm => (regv, rmv, false),
            };
            let result = alu16(cpu, op, a, b);
            // CMP sets flags only — it must never write its result back.
            if op != AluOp::Cmp {
                if store { cpu.write_rm16(&m, result); } else { cpu.set_reg16_idx(reg, result); }
            }
        }
        Inst::AluRm32Reg { op, m, reg, dir } => {
            let w = cpu.osize();
            let regv = cpu.reg_w(reg, w);
            let rmv = cpu.read_rm_w(&m, w);
            let (a, b, store) = match dir {
                Dir::RmReg => (rmv, regv, true),
                Dir::RegRm => (regv, rmv, false),
            };
            let result = alu_w(cpu, op, a, b, w);
            // CMP sets flags only — it must never write its result back.
            if op != AluOp::Cmp {
                if store { cpu.write_rm_w(&m, w, result); } else { cpu.set_reg_w(reg, w, result); }
            }
        }

        // ---- ALU r/m, imm ----
        Inst::AluRm8Imm { op, m, imm } => {
            let rmv = cpu.read_rm8(&m);
            let result = alu8(cpu, op, rmv, imm);
            if op != AluOp::Cmp { cpu.write_rm8(&m, result); }
        }
        Inst::AluRm16Imm { op, m, imm, .. } => {
            let rmv = cpu.read_rm16(&m);
            let result = alu16(cpu, op, rmv, imm);
            if op != AluOp::Cmp { cpu.write_rm16(&m, result); }
        }
        // The immediate is imm32 (or a sign-extended imm8) whatever the
        // operand size: 64-bit forms sign-extend it, they do not carry eight
        // more bytes.
        Inst::AluRm32Imm { op, m, imm, .. } => {
            let w = cpu.osize();
            let rmv = cpu.read_rm_w(&m, w);
            let result = alu_w(cpu, op, rmv, sext(imm as u64, 32), w);
            if op != AluOp::Cmp { cpu.write_rm_w(&m, w, result); }
        }
        Inst::AluAccImm8 { op, imm } => {
            let a = cpu.reg8(Reg8::Al);
            let result = alu8(cpu, op, a, imm);
            if op != AluOp::Cmp { cpu.set_reg8(Reg8::Al, result); }
        }
        Inst::AluAccImm16 { op, imm } => {
            let a = cpu.reg16(Reg16::Ax);
            let result = alu16(cpu, op, a, imm);
            if op != AluOp::Cmp { cpu.set_reg16(Reg16::Ax, result); }
        }
        Inst::AluAccImm32 { op, imm } => {
            let w = cpu.osize();
            let a = cpu.reg_w(0, w);
            let result = alu_w(cpu, op, a, sext(imm as u64, 32), w);
            if op != AluOp::Cmp { cpu.set_reg_w(0, w, result); }
        }

        // ---- INC / DEC ----
        Inst::IncReg16 { dst } => {
            let v = cpu.reg16_idx(dst);
            let result = v.wrapping_add(1);
            cpu.set_reg16_idx(dst, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags16(cpu, result);
            set_add_carry(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }
        Inst::DecReg16 { dst } => {
            let v = cpu.reg16_idx(dst);
            let result = v.wrapping_sub(1);
            cpu.set_reg16_idx(dst, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags16(cpu, result);
            set_sub_borrow(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }
        Inst::IncReg32 { dst } => {
            let v = cpu.reg32_idx(dst);
            let result = v.wrapping_add(1);
            cpu.set_reg32_idx(dst, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags32(cpu, result);
            set_add_carry32(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }
        Inst::DecReg32 { dst } => {
            let v = cpu.reg32_idx(dst);
            let result = v.wrapping_sub(1);
            cpu.set_reg32_idx(dst, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags32(cpu, result);
            set_sub_borrow32(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }

        // ---- PUSH / POP ----
        // PUSHA/PUSHAD: push all eight general registers in index order,
        // with the *original* SP/ESP (captured before the first push).
        Inst::Pusha { w32 } => {
            if w32 {
                let esp = cpu.esp();
                cpu.push32(cpu.eax()); cpu.push32(cpu.ecx());
                cpu.push32(cpu.edx()); cpu.push32(cpu.ebx());
                cpu.push32(esp);     cpu.push32(cpu.ebp());
                cpu.push32(cpu.esi()); cpu.push32(cpu.edi());
            } else {
                let sp = cpu.sp();
                cpu.push16(cpu.ax()); cpu.push16(cpu.cx());
                cpu.push16(cpu.dx()); cpu.push16(cpu.bx());
                cpu.push16(sp);     cpu.push16(cpu.bp());
                cpu.push16(cpu.si()); cpu.push16(cpu.di());
            }
        }
        // POPA/POPAD: pop in reverse order. The stored SP/ESP slot is
        // discarded (the stack pointer is restored by the pops themselves).
        Inst::Popa { w32 } => {
            if w32 {
                let edi = cpu.pop32(); let esi = cpu.pop32();
                let ebp = cpu.pop32(); let _esp = cpu.pop32();
                let ebx = cpu.pop32(); let edx = cpu.pop32();
                let ecx = cpu.pop32(); let eax = cpu.pop32();
                cpu.set_reg32(Reg32::Edi, edi); cpu.set_reg32(Reg32::Esi, esi);
                cpu.set_reg32(Reg32::Ebp, ebp);
                cpu.set_reg32(Reg32::Ebx, ebx); cpu.set_reg32(Reg32::Edx, edx);
                cpu.set_reg32(Reg32::Ecx, ecx); cpu.set_reg32(Reg32::Eax, eax);
            } else {
                let di = cpu.pop16(); let si = cpu.pop16();
                let bp = cpu.pop16(); let _sp = cpu.pop16();
                let bx = cpu.pop16(); let dx = cpu.pop16();
                let cx = cpu.pop16(); let ax = cpu.pop16();
                cpu.set_reg16(Reg16::Di, di); cpu.set_reg16(Reg16::Si, si);
                cpu.set_reg16(Reg16::Bp, bp);
                cpu.set_reg16(Reg16::Bx, bx); cpu.set_reg16(Reg16::Dx, dx);
                cpu.set_reg16(Reg16::Cx, cx); cpu.set_reg16(Reg16::Ax, ax);
            }
        }
        Inst::PushReg16 { src } => { let v = cpu.reg16_idx(src) as u64; cpu.push_w(16, v) }
        Inst::PopReg16 { dst } => { let v = cpu.pop_w(16); cpu.set_reg16_idx(dst, v as u16); }
        // In 64-bit mode PUSH and POP of a register are 64-bit and there is
        // no encoding for anything narrower -- the stack width is not the
        // operand size, which is why these go through `stack_width`.
        Inst::PushReg32 { src } => {
            let w = cpu.stack_width();
            let v = cpu.reg_w(src, w);
            cpu.push_w(w, v);
        }
        Inst::PopReg32 { dst } => {
            let w = cpu.stack_width();
            let v = cpu.pop_w(w);
            cpu.set_reg_w(dst, w, v);
        }
        Inst::PushImm16 { imm } => cpu.push_w(16, imm as u64),
        Inst::PushImm32 { imm } => {
            let w = cpu.stack_width();
            cpu.push_w(w, sext(imm as u64, 32));
        }

        // ---- Control flow ----
        Inst::JmpRel8 { rel } => branch_rel(cpu, rel as i64),
        Inst::JmpRel16 { rel } => { cpu.ip = cpu.ip.wrapping_add(rel as u16); }
        Inst::JmpRel32 { rel } => branch_rel(cpu, rel as i64),
        Inst::Jcc { cond, rel } => {
            if cond.test(cpu) { branch_rel(cpu, rel as i64); }
        }
        Inst::Jcc32 { cond, rel } => {
            // 0F 80-8F: conditional jump with rel32 displacement. In 32-bit
            // mode it branches via EIP; in 16-bit mode via IP.
            if cond.test(cpu) { branch_rel(cpu, rel as i64); }
        }
        // ---- MOVZX / MOVSX ----
        Inst::Movzx8 { m, dst } => {
            let w = cpu.osize();
            let v = cpu.read_rm8(&m) as u64;
            cpu.set_reg_w(dst, w, v);
        }
        Inst::Movzx16 { m, dst } => {
            let w = cpu.osize();
            let v = cpu.read_rm16(&m) as u64;
            cpu.set_reg_w(dst, w, v);
        }
        Inst::Movsx8 { m, dst } => {
            let w = cpu.osize();
            let v = sext(cpu.read_rm8(&m) as u64, 8);
            cpu.set_reg_w(dst, w, v);
        }
        Inst::Movsx16 { m, dst } => {
            let w = cpu.osize();
            let v = sext(cpu.read_rm16(&m) as u64, 16);
            cpu.set_reg_w(dst, w, v);
        }
        // MOVSXD (0x63) took over the ARPL opcode in 64-bit mode, and it is
        // how a 32-bit value becomes a 64-bit index: every array subscript in
        // compiled 64-bit code goes through it.
        Inst::Movsxd { m, dst } => {
            let w = cpu.osize();
            let v = sext(cpu.read_rm_w(&m, 32), 32);
            cpu.set_reg_w(dst, w, v);
        }
        Inst::CallRel16 { rel } => {
            let next = cpu.ip;
            cpu.push16(next);
            cpu.ip = cpu.ip.wrapping_add(rel as u16);
        }
        Inst::CallRel32 { rel } => {
            let w = cpu.stack_width();
            let next = if cpu.long64() { cpu.rip } else { cpu.eip() as u64 };
            cpu.push_w(w, next);
            if cpu.pending_exception.is_some() { return; }
            branch_rel(cpu, rel as i64);
        }
        Inst::Ret => { cpu.ip = cpu.pop16(); }
        Inst::RetImm { imm, w32 } => {
            // The stack adjustment happens after the return address is popped.
            if cpu.long64() {
                let target = cpu.pop_w(64);
                cpu.set_rsp(cpu.rsp().wrapping_add(imm as u64));
                cpu.rip = target;
            } else if w32 {
                let target = cpu.pop32();
                cpu.set_esp(cpu.esp().wrapping_add(imm as u32));
                cpu.set_eip(target);
            } else {
                let target = cpu.pop16();
                cpu.set_sp(cpu.sp().wrapping_add(imm));
                cpu.ip = target;
            }
        }
        Inst::Ret32 => {
            let w = cpu.stack_width();
            let t = cpu.pop_w(w);
            if cpu.long64() { cpu.rip = t; } else { cpu.set_eip(t as u32); }
        }
        Inst::XchgAxReg { reg } => {
            let ax = cpu.reg16(Reg16::Ax);
            let r = cpu.reg16_idx(reg);
            cpu.set_reg16(Reg16::Ax, r);
            cpu.set_reg16_idx(reg, ax);
        }
        Inst::XchgEaxReg { reg } => {
            let w = cpu.osize();
            let ax = cpu.reg_w(0, w);
            let r = cpu.reg_w(reg, w);
            cpu.set_reg_w(0, w, r);
            cpu.set_reg_w(reg, w, ax);
        }
        Inst::Int { vector } => {
            // Record system calls from user mode: the sequence of calls ld.so
            // makes is the quickest way to see which path it took.
            if cpu.debug_enabled && vector == 0x80 && cpu.cpl() == 3
                && cpu.syscall_log.len() < 512 {
                let n = cpu.instructions_executed;
                cpu.syscall_log.push((n, cpu.reg64(0), cpu.reg64(3), cpu.reg64(1), cpu.reg64(2)));
            }
            // BIOS services (INT 0x10/0x16/0x13) are handled natively in Rust.
            let mut bios = std::mem::take(&mut cpu.bios);
            let handled = bios.handle(cpu, vector);
            cpu.bios = bios;
            if handled { return; }
            if cpu.pe {
                // Protected-mode interrupt through the IDT.
                protected_int(cpu, vector);
            } else {
                // Real-mode interrupt through the IVT.
                let ip = cpu.ip;
                let cs = cpu.cs;
                let flags = cpu.flags as u16;
                cpu.push16(flags);
                cpu.push16(cs);
                cpu.push16(ip);
                cpu.set_flag(flags::IF, false);
                cpu.set_flag(flags::TF, false);
                let entry = (vector as usize) * 4;
                let off = cpu.mem.read_u16(entry);
                let seg = cpu.mem.read_u16(entry + 2);
                cpu.cs = seg;
                cpu.ip = off;
            }
        }
        Inst::Int3 => {
            // Breakpoint exception (#BP, vector 0x03). No error code.
            cpu.pending_exception = Some((0x03, None));
        }
        Inst::Into => {
            // Overflow exception (#OF, vector 0x04) if OF is set.
            if cpu.get_flag(flags::OF) {
                cpu.pending_exception = Some((0x04, None));
            }
        }
        Inst::Iret => {
            cpu.ip = cpu.pop16();
            cpu.cs = cpu.pop16();
            let f = cpu.pop16();
            // A 16-bit IRET restores only the low half; the high half of
            // EFLAGS is left as it was.
            cpu.flags = write_flags(cpu.flags, (cpu.flags & 0xFFFF_0000) | f as u32);
        }
        // IRETQ: in long mode the frame is always five 8-byte words --
        // SS:RSP included, whether or not the privilege level changes -- so
        // it is popped unconditionally. Getting that wrong leaves RSP eight
        // bytes off on every kernel-to-kernel return, which unwinds into
        // nonsense a few interrupts later.
        Inst::Iret32 if cpu.long_mode() => {
            // In 64-bit long mode, IRET always pops 64-bit words — REX.W
            // is redundant and the operand-size prefix is ignored. Getting
            // this wrong truncates the 64-bit RIP to 32 bits on every
            // kernel-to-kernel return, which sends the CPU to a low address
            // that the page tables do not map.
            let rip = cpu.pop_w(64);
            let cs = cpu.pop_w(64) as u16;
            let f = cpu.pop_w(64) as u32;
            let rsp = cpu.pop_w(64);
            let ss = cpu.pop_w(64) as u16;
            cpu.load_seg(SegReg::Cs, cs);
            cpu.load_seg(SegReg::Ss, ss);
            cpu.set_rsp(rsp);
            cpu.rip = rip;
            cpu.flags = write_flags(cpu.flags, f);
            cpu.invalidate_phys_ip();
        }
        Inst::Iret32 => {
            let eip = cpu.pop32();
            let cs = cpu.pop32() as u16;
            let f = cpu.pop32();
            // Returning to a less privileged level: the frame carries the
            // user stack as well, and it must be restored before CS changes
            // the CPL out from under the pops.
            if (cs & 3) as u8 > cpu.cpl() {
                let esp = cpu.pop32();
                let ss = cpu.pop32() as u16;
                cpu.load_seg(SegReg::Ss, ss);
                cpu.set_esp(esp);
            }
            cpu.set_eip(eip);
            cpu.load_seg(SegReg::Cs, cs);
            cpu.flags = write_flags(cpu.flags, f);
            cpu.invalidate_phys_ip();
        }
        // PUSHF/PUSHFD and POPF/POPFD follow the operand size. In 32-bit mode
        // these move the whole of EFLAGS — which is how Linux probes the AC
        // and ID bits — so a 16-bit-only implementation both loses those bits
        // and moves ESP by the wrong amount.
        Inst::Pushf => {
            let w = cpu.stack_width();
            if w >= 32 {
                // VM and RF read back as zero in the pushed image.
                let f = (cpu.flags & !(flags::VM | flags::RF)) as u64;
                cpu.push_w(w, f);
            } else {
                cpu.push_w(16, cpu.flags as u64);
            }
        }
        Inst::Popf => {
            let w = cpu.stack_width();
            if w >= 32 {
                let f = cpu.pop_w(w) as u32;
                cpu.flags = write_flags(cpu.flags, f);
            } else {
                let f = cpu.pop_w(16) as u32;
                cpu.flags = write_flags(cpu.flags, (cpu.flags & 0xFFFF_0000) | f);
            }
        }

        // ---- Shifts / rotates (group 2) ----
        Inst::Shift { op, m, width, count } => {
            let n = match count {
                ShiftCount::One => 1,
                ShiftCount::Cl => cpu.reg8(Reg8::Cl) as u32,
                ShiftCount::Imm(i) => i as u32,
            };
            do_shift(cpu, op, &m, width, n);
        }
        Inst::ShiftImm { op, m, width, imm } => {
            do_shift(cpu, op, &m, width, imm as u32);
        }

        // ---- Group 3: TEST / NOT / NEG / MUL / IMUL / DIV / IDIV ----
        Inst::TestRm8Imm { m, imm } => {
            let v = cpu.read_rm8(&m);
            let r = v & imm;
            set_logic_flags8(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestRm16Imm { m, imm } => {
            let v = cpu.read_rm16(&m);
            let r = v & imm;
            set_logic_flags16(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestRm32Imm { m, imm } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w);
            let r = v & sext(imm as u64, 32);
            set_logic_flags_width(cpu, r, w);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestRm8Reg { m, reg } => {
            let v = cpu.read_rm8(&m);
            let r = v & cpu.reg8_idx(reg);
            set_logic_flags8(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestRm16Reg { m, reg } => {
            let v = cpu.read_rm16(&m);
            let r = v & cpu.reg16_idx(reg);
            set_logic_flags16(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestRm32Reg { m, reg } => {
            let w = cpu.osize();
            let a = cpu.read_rm_w(&m, w);
            let b = cpu.reg_w(reg, w);
            let r = a & b;
            set_logic_flags_width(cpu, r, w);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestAccImm8 { imm } => {
            let v = cpu.reg8(Reg8::Al);
            let r = v & imm;
            set_logic_flags8(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestAccImm16 { imm } => {
            let v = cpu.reg16(Reg16::Ax);
            let r = v & imm;
            set_logic_flags16(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestAccImm32 { imm } => {
            let w = cpu.osize();
            let a = cpu.reg_w(0, w);
            let r = a & sext(imm as u64, 32);
            set_logic_flags_width(cpu, r, w);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::NotRm8 { m } => { let v = cpu.read_rm8(&m); cpu.write_rm8(&m, !v); }
        Inst::NotRm16 { m } => { let v = cpu.read_rm16(&m); cpu.write_rm16(&m, !v); }
        Inst::NotRm32 { m } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w);
            cpu.write_rm_w(&m, w, !v);
        }
        Inst::NegRm8 { m } => {
            let v = cpu.read_rm8(&m);
            let r = v.wrapping_neg();
            cpu.write_rm8(&m, r);
            set_logic_flags8(cpu, r);
            cpu.set_flag(CF, v != 0);
            cpu.set_flag(OF, v == 0x80);
            cpu.set_flag(AF, v != 0);
        }
        Inst::NegRm16 { m } => {
            let v = cpu.read_rm16(&m);
            let r = v.wrapping_neg();
            cpu.write_rm16(&m, r);
            set_logic_flags16(cpu, r);
            cpu.set_flag(CF, v != 0);
            cpu.set_flag(OF, v == 0x8000);
            cpu.set_flag(AF, v != 0);
        }
        Inst::NegRm32 { m } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w);
            let r = v.wrapping_neg() & mask_w(w);
            cpu.write_rm_w(&m, w, r);
            set_logic_flags_width(cpu, r, w);
            cpu.set_flag(CF, v != 0);
            // OF is set only for the one value that has no negation.
            cpu.set_flag(OF, v == 1u64 << (w - 1));
            cpu.set_flag(AF, v != 0);
        }
        Inst::MulRm8 { m } => {
            let v = cpu.read_rm8(&m);
            let a = cpu.reg8(Reg8::Al) as u16;
            let r = a * v as u16;
            cpu.set_reg16(Reg16::Ax, r);
            let hi = (r >> 8) as u8;
            let c = hi != 0;
            cpu.set_flag(CF, c);
            cpu.set_flag(OF, c);
        }
        Inst::MulRm16 { m } => {
            let v = cpu.read_rm16(&m);
            let a = cpu.reg16(Reg16::Ax) as u32;
            let r = a * v as u32;
            cpu.set_reg16(Reg16::Ax, r as u16);
            cpu.set_reg16(Reg16::Dx, (r >> 16) as u16);
            let c = (r >> 16) != 0;
            cpu.set_flag(CF, c);
            cpu.set_flag(OF, c);
        }
        Inst::MulRm32 { m } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w) as u128;
            let a = cpu.reg_w(0, w) as u128;
            let r = a * v;
            cpu.set_reg_w(0, w, r as u64);
            cpu.set_reg_w(2, w, (r >> w) as u64);
            let c = (r >> w) != 0;
            cpu.set_flag(CF, c);
            cpu.set_flag(OF, c);
        }
        Inst::ImulRm8 { m } => {
            let v = cpu.read_rm8(&m) as i8 as i16;
            let a = cpu.reg8(Reg8::Al) as i8 as i16;
            let r = a * v;
            cpu.set_reg16(Reg16::Ax, r as u16);
            let hi = (r >> 8) as i8;
            let lo = r as i8;
            let c = hi != lo;
            cpu.set_flag(CF, c);
            cpu.set_flag(OF, c);
        }
        Inst::ImulRm16 { m } => {
            let v = cpu.read_rm16(&m) as i16 as i32;
            let a = cpu.reg16(Reg16::Ax) as i16 as i32;
            let r = a * v;
            cpu.set_reg16(Reg16::Ax, r as u16);
            cpu.set_reg16(Reg16::Dx, (r >> 16) as u16);
            let hi = (r >> 16) as i16;
            let lo = r as i16;
            let c = hi != lo;
            cpu.set_flag(CF, c);
            cpu.set_flag(OF, c);
        }
        Inst::ImulRm32 { m } => {
            let w = cpu.osize();
            let v = sign_extend(cpu.read_rm_w(&m, w), w) as i128;
            let a = sign_extend(cpu.reg_w(0, w), w) as i128;
            let r = a * v;
            cpu.set_reg_w(0, w, r as u64);
            cpu.set_reg_w(2, w, (r >> w) as u64);
            // CF/OF say the product did not fit in the low half: the high
            // half must be nothing but copies of the low half sign bit.
            let c = r != sign_extend(r as u64, w) as i128;
            cpu.set_flag(CF, c);
            cpu.set_flag(OF, c);
        }
        Inst::DivRm8 { m } => {
            let v = cpu.read_rm8(&m);
            if v == 0 {
                cpu.pending_exception = Some((0x00, None)); // #DE
                return;
            }
            let a = cpu.reg16(Reg16::Ax);
            let q = a / v as u16;
            let rem = a % v as u16;
            cpu.set_reg8(Reg8::Al, q as u8);
            cpu.set_reg8(Reg8::Ah, rem as u8);
        }
        Inst::DivRm16 { m } => {
            let v = cpu.read_rm16(&m);
            if v == 0 {
                cpu.pending_exception = Some((0x00, None)); // #DE
                return;
            }
            let a = ((cpu.reg16(Reg16::Dx) as u32) << 16) | cpu.reg16(Reg16::Ax) as u32;
            let q = a / v as u32;
            let rem = a % v as u32;
            cpu.set_reg16(Reg16::Ax, q as u16);
            cpu.set_reg16(Reg16::Dx, rem as u16);
        }
        Inst::DivRm32 { m } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w) as u128;
            if v == 0 {
                cpu.pending_exception = Some((0x00, None)); // #DE
                return;
            }
            let a = ((cpu.reg_w(2, w) as u128) << w) | cpu.reg_w(0, w) as u128;
            let q = a / v;
            let rem = a % v;
            // A quotient too wide for the destination is also a #DE, not a
            // truncated answer.
            if q > mask_w(w) as u128 {
                cpu.pending_exception = Some((0x00, None));
                return;
            }
            cpu.set_reg_w(0, w, q as u64);
            cpu.set_reg_w(2, w, rem as u64);
        }
        Inst::IdivRm8 { m } => {
            let v = cpu.read_rm8(&m) as i8 as i16;
            if v == 0 {
                cpu.pending_exception = Some((0x00, None)); // #DE
                return;
            }
            let a = cpu.reg16(Reg16::Ax) as i16;
            let q = a / v;
            let rem = a % v;
            cpu.set_reg8(Reg8::Al, q as u8);
            cpu.set_reg8(Reg8::Ah, rem as u8);
        }
        Inst::IdivRm16 { m } => {
            let v = cpu.read_rm16(&m) as i16 as i32;
            if v == 0 {
                cpu.pending_exception = Some((0x00, None)); // #DE
                return;
            }
            let a = ((cpu.reg16(Reg16::Dx) as u32) << 16 | cpu.reg16(Reg16::Ax) as u32) as i32;
            let q = a / v;
            let rem = a % v;
            cpu.set_reg16(Reg16::Ax, q as u16);
            cpu.set_reg16(Reg16::Dx, rem as u16);
        }
        Inst::IdivRm32 { m } => {
            let w = cpu.osize();
            let v = sign_extend(cpu.read_rm_w(&m, w), w) as i128;
            if v == 0 {
                cpu.pending_exception = Some((0x00, None)); // #DE
                return;
            }
            let raw = ((cpu.reg_w(2, w) as u128) << w) | cpu.reg_w(0, w) as u128;
            // The dividend is a signed 2*w-bit value: sign-extend it from its
            // own width, not from 64.
            let a = if w == 64 { raw as i128 } else {
                let bits = 2 * w;
                ((raw << (128 - bits)) as i128) >> (128 - bits)
            };
            let q = a / v;
            let rem = a % v;
            if q != sign_extend(q as u64, w) as i128 {
                cpu.pending_exception = Some((0x00, None));
                return;
            }
            cpu.set_reg_w(0, w, q as u64);
            cpu.set_reg_w(2, w, rem as u64);
        }

        // ---- LEA ----
        Inst::Lea { m, dst } => {
            let w = cpu.osize();
            let ea = lea_offset(&m, cpu);
            cpu.set_reg_w(dst, w, ea);
        }

        // ---- CBW / CWD / CWDE / CDQ ----
        Inst::Cbw => {
            let al = cpu.reg8(Reg8::Al);
            cpu.set_reg16(Reg16::Ax, al as i8 as i16 as u16);
        }
        Inst::Cwd => {
            let ax = cpu.reg16(Reg16::Ax);
            let dx = if (ax as i16) < 0 { 0xFFFF } else { 0 };
            cpu.set_reg16(Reg16::Dx, dx);
        }
        // CWDE at 32-bit operand size; CDQE (the same opcode with REX.W)
        // sign-extends EAX into the whole of RAX.
        Inst::Cwde => {
            let w = cpu.osize();
            let v = sext(cpu.reg_w(0, w / 2), w / 2);
            cpu.set_reg_w(0, w, v);
        }
        // CDQ, and CQO with REX.W: fill the high register with the sign.
        Inst::Cdq => {
            let w = cpu.osize();
            let v = cpu.reg_w(0, w);
            let neg = (v >> (w - 1)) & 1 != 0;
            cpu.set_reg_w(2, w, if neg { u64::MAX } else { 0 });
        }

        // ---- LOOP / LOOPZ / LOOPNZ / JCXZ ----
        Inst::Loop { cond, rel } => {
            // The count register follows the *address* size, so in 64-bit
            // mode LOOP counts RCX.
            if cpu.addr64 {
                let take = match cond {
                    LoopCond::Jcxz => cpu.reg64(1) == 0,
                    _ => {
                        let c = cpu.reg64(1).wrapping_sub(1);
                        cpu.set_reg64_raw(1, c);
                        match cond {
                            LoopCond::Loop => c != 0,
                            LoopCond::Loopz => c != 0 && cpu.get_flag(ZF),
                            LoopCond::Loopnz => c != 0 && !cpu.get_flag(ZF),
                            _ => false,
                        }
                    }
                };
                if take { branch_rel(cpu, rel as i64); }
                return;
            }
            let take = if cpu.opsize {
                // 32-bit mode: LOOP uses ECX and branches via EIP.
                match cond {
                    LoopCond::Jcxz => cpu.ecx() == 0,
                    _ => {
                        cpu.set_ecx(cpu.ecx().wrapping_sub(1));
                        match cond {
                            LoopCond::Loop => cpu.ecx() != 0,
                            LoopCond::Loopz => cpu.ecx() != 0 && cpu.get_flag(ZF),
                            LoopCond::Loopnz => cpu.ecx() != 0 && !cpu.get_flag(ZF),
                            _ => false,
                        }
                    }
                }
            } else {
                // 16-bit mode: LOOP uses CX and branches via IP.
                match cond {
                    LoopCond::Jcxz => cpu.cx() == 0,
                    _ => {
                        cpu.set_cx(cpu.cx().wrapping_sub(1));
                        match cond {
                            LoopCond::Loop => cpu.cx() != 0,
                            LoopCond::Loopz => cpu.cx() != 0 && cpu.get_flag(ZF),
                            LoopCond::Loopnz => cpu.cx() != 0 && !cpu.get_flag(ZF),
                            _ => false,
                        }
                    }
                }
            };
            if take {
                if cpu.opsize { cpu.set_eip(cpu.eip().wrapping_add(rel as i32 as u32)); }
                else { cpu.ip = cpu.ip.wrapping_add(rel as i16 as u16); }
            }
        }

        // ---- Far control flow ----
        Inst::JmpFar { off, seg } => {
            cpu.load_seg(SegReg::Cs, seg);
            cpu.ip = off;
            cpu.invalidate_phys_ip();
        }
        Inst::CallFar { off, seg } => {
            let ip = cpu.ip;
            cpu.push16(cpu.cs);
            cpu.push16(ip);
            cpu.load_seg(SegReg::Cs, seg);
            cpu.ip = off;
            cpu.invalidate_phys_ip();
        }
        // The far jump is how a boot sequence *leaves* the mode it is in:
        // it is the instruction that makes a newly loaded CS take effect, and
        // in a 64-bit boot it is the step that turns compatibility-mode code
        // into 64-bit code by landing in an L=1 segment.
        Inst::JmpFar32 { off, seg } => {
            cpu.load_seg(SegReg::Cs, seg);
            cpu.set_eip(off);
            cpu.invalidate_phys_ip();
        }
        Inst::CallFar32 { off, seg } => {
            let ip = cpu.eip();
            let cs = cpu.cs as u32;
            cpu.push32(cs);
            cpu.push32(ip);
            cpu.load_seg(SegReg::Cs, seg);
            cpu.set_eip(off);
            cpu.invalidate_phys_ip();
        }
        Inst::Retf => {
            cpu.ip = cpu.pop16();
            let cs = cpu.pop16();
            cpu.load_seg(SegReg::Cs, cs);
        }
        Inst::Retf32 if cpu.long_mode() => {
            // In 64-bit long mode, RETF pops a 64-bit RIP and 16-bit CS,
            // and loading CS must update the descriptor cache so the L bit
            // is read — without that, long64() returns false and every
            // address is truncated to 32 bits.
            let rip = cpu.pop_w(64);
            let cs = cpu.pop_w(64) as u16;
            cpu.load_seg(SegReg::Cs, cs);
            cpu.rip = rip;
            cpu.invalidate_phys_ip();
        }
        Inst::Retf32 => {
            let t = cpu.pop32();
            let cs = cpu.pop32() as u16;
            cpu.load_seg(SegReg::Cs, cs);
            cpu.set_eip(t);
            cpu.invalidate_phys_ip();
        }

        // ---- Group 5 (0xFF): INC / DEC / CALL / JMP / PUSH r/m ----
        // INC/DEC preserve CF -- that is the whole reason they exist
        // alongside ADD/SUB by 1 -- so it is saved and restored around the
        // flag computation.
        Inst::IncRm8 { m } => {
            let v = cpu.read_rm8(&m);
            let result = v.wrapping_add(1);
            cpu.write_rm8(&m, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags8(cpu, result);
            set_add_carry(cpu, v as u16, 1, result as u16, false);
            cpu.set_flag(OF, v == 0x7F);
            cpu.set_flag(AF, (v & 0x0F) == 0x0F);
            cpu.set_flag(CF, cf);
        }
        Inst::DecRm8 { m } => {
            let v = cpu.read_rm8(&m);
            let result = v.wrapping_sub(1);
            cpu.write_rm8(&m, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags8(cpu, result);
            cpu.set_flag(OF, v == 0x80);
            cpu.set_flag(AF, (v & 0x0F) == 0x00);
            cpu.set_flag(CF, cf);
        }
        Inst::IncRm16 { m } => {
            let v = cpu.read_rm16(&m);
            let result = v.wrapping_add(1);
            cpu.write_rm16(&m, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags16(cpu, result);
            set_add_carry(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }
        Inst::IncRm32 { m } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w);
            let result = v.wrapping_add(1) & mask_w(w);
            cpu.write_rm_w(&m, w, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags_width(cpu, result, w);
            set_add_carry64(cpu, v, 1, result, false);
            cpu.set_flag(OF, v == mask_w(w) >> 1);
            cpu.set_flag(CF, cf);
        }
        Inst::DecRm16 { m } => {
            let v = cpu.read_rm16(&m);
            let result = v.wrapping_sub(1);
            cpu.write_rm16(&m, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags16(cpu, result);
            set_sub_borrow(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }
        Inst::DecRm32 { m } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w);
            let result = v.wrapping_sub(1) & mask_w(w);
            cpu.write_rm_w(&m, w, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags_width(cpu, result, w);
            set_sub_borrow64(cpu, v, 1, result, false);
            cpu.set_flag(OF, v == 1u64 << (w - 1));
            cpu.set_flag(CF, cf);
        }
        Inst::CallRm16 { m } => {
            let target = cpu.read_rm16(&m);
            let next = cpu.ip;
            cpu.push16(next);
            cpu.ip = target;
        }
        Inst::CallRm32 { m } => {
            // The call target follows the operand size (64-bit by default in
            // 64-bit mode), and so does the return address pushed for it.
            let w = cpu.stack_width();
            let target = cpu.read_rm_w(&m, w);
            let next = if cpu.long64() { cpu.rip } else { cpu.eip() as u64 };
            cpu.push_w(w, next);
            if cpu.pending_exception.is_some() { return; }
            if cpu.long64() { cpu.rip = target; } else { cpu.set_eip(target as u32); }
            cpu.invalidate_phys_ip();
        }
        Inst::JmpRm16 { m } => {
            cpu.ip = cpu.read_rm16(&m);
        }
        Inst::JmpRm32 { m } => {
            let w = cpu.stack_width();
            let t = cpu.read_rm_w(&m, w);
            if cpu.long64() { cpu.rip = t; } else { cpu.set_eip(t as u32); }
            cpu.invalidate_phys_ip();
        }
        Inst::PushRm16 { m } => {
            let v = cpu.read_rm16(&m);
            cpu.push16(v);
        }
        Inst::PushRm32 { m } => {
            let w = cpu.stack_width();
            let v = cpu.read_rm_w(&m, w);
            cpu.push_w(w, v);
        }
        // BSF/BSR: ZF is set when the source is zero, and the destination is
        // then architecturally undefined - we leave it alone.
        Inst::Bsf { m, dst, .. } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w);
            cpu.set_flag(flags::ZF, v == 0);
            // With a zero source the destination is left alone -- it is
            // architecturally undefined, and leaving it is what every real
            // CPU does.
            if v != 0 { cpu.set_reg_w(dst, w, v.trailing_zeros() as u64); }
        }
        Inst::Bsr { m, dst, .. } => {
            let w = cpu.osize();
            let v = cpu.read_rm_w(&m, w);
            cpu.set_flag(flags::ZF, v == 0);
            if v != 0 { cpu.set_reg_w(dst, w, (63 - v.leading_zeros()) as u64); }
        }
        // XCHG r/m, r. No flags.
        Inst::XchgRmReg { m, reg, width } => match width {
            8 => {
                let a = cpu.read_rm8(&m);
                let b = cpu.reg8_idx(reg);
                cpu.write_rm8(&m, b);
                cpu.set_reg8_idx(reg, a);
            }
            16 => {
                let a = cpu.read_rm16(&m);
                let b = cpu.reg16_idx(reg);
                cpu.write_rm16(&m, b);
                cpu.set_reg16_idx(reg, a);
            }
            _ => {
                let w = cpu.osize();
                let a = cpu.read_rm_w(&m, w);
                let b = cpu.reg_w(reg, w);
                cpu.write_rm_w(&m, w, b);
                cpu.set_reg_w(reg, w, a);
            }
        },
        // CMPXCHG: compare the accumulator with the destination. On a match
        // the source register is stored; otherwise the accumulator takes the
        // destination's value. Flags are those of `CMP acc, dest`.
        Inst::Cmpxchg { m, reg, width } => match width {
            8 => {
                let dest = cpu.read_rm8(&m);
                let acc = cpu.reg8(Reg8::Al);
                alu8(cpu, AluOp::Cmp, acc, dest);
                if acc == dest { let v = cpu.reg8_idx(reg); cpu.write_rm8(&m, v); }
                else { cpu.set_reg8(Reg8::Al, dest); }
            }
            16 => {
                let dest = cpu.read_rm16(&m);
                let acc = cpu.reg16(Reg16::Ax);
                alu16(cpu, AluOp::Cmp, acc, dest);
                if acc == dest { let v = cpu.reg16_idx(reg); cpu.write_rm16(&m, v); }
                else { cpu.set_reg16(Reg16::Ax, dest); }
            }
            _ => {
                let w = cpu.osize();
                let dest = cpu.read_rm_w(&m, w);
                let acc = cpu.reg_w(0, w);
                alu_w(cpu, AluOp::Cmp, acc, dest, w);
                if acc == dest { let v = cpu.reg_w(reg, w); cpu.write_rm_w(&m, w, v); }
                else { cpu.set_reg_w(0, w, dest); }
            }
        },
        // XADD: the destination gets the sum, the source register gets the
        // destination's old value. Flags are those of ADD.
        Inst::Xadd { m, reg, width } => match width {
            8 => {
                let dest = cpu.read_rm8(&m);
                let src = cpu.reg8_idx(reg);
                let sum = alu8(cpu, AluOp::Add, dest, src);
                cpu.set_reg8_idx(reg, dest);
                cpu.write_rm8(&m, sum);
            }
            16 => {
                let dest = cpu.read_rm16(&m);
                let src = cpu.reg16_idx(reg);
                let sum = alu16(cpu, AluOp::Add, dest, src);
                cpu.set_reg16_idx(reg, dest);
                cpu.write_rm16(&m, sum);
            }
            _ => {
                let w = cpu.osize();
                let dest = cpu.read_rm_w(&m, w);
                let src = cpu.reg_w(reg, w);
                let sum = alu_w(cpu, AluOp::Add, dest, src, w);
                cpu.set_reg_w(reg, w, dest);
                cpu.write_rm_w(&m, w, sum);
            }
        },
        // CMPXCHG8B: compare EDX:EAX with the 64-bit destination; on a
        // match store ECX:EBX, otherwise load the destination into EDX:EAX.
        // Only ZF reports the outcome.
        Inst::Cmpxchg8b { m } => {
            let addr = cpu.rm_addr(&m, true);
            let lo = cpu.mem.read_u32(addr);
            let hi = cpu.mem.read_u32(addr + 4);
            if lo == cpu.eax() && hi == cpu.edx() {
                cpu.set_flag(flags::ZF, true);
                let (bl, ch) = (cpu.ebx(), cpu.ecx());
                cpu.mem.write_u32(addr, bl);
                cpu.mem.write_u32(addr + 4, ch);
            } else {
                cpu.set_flag(flags::ZF, false);
                cpu.set_reg32(Reg32::Eax, lo);
                cpu.set_reg32(Reg32::Edx, hi);
            }
        }
        Inst::Bswap { reg } => {
            let w = cpu.osize();
            let v = cpu.reg_w(reg, w);
            let r = if w == 64 { v.swap_bytes() } else { (v as u32).swap_bytes() as u64 };
            cpu.set_reg_w(reg, w, r);
        }
        // CMOVcc: the load happens only when the condition holds. (A real CPU
        // reads the memory operand either way, but nothing observable here
        // depends on that, and skipping the read avoids a spurious fault.)
        Inst::Cmovcc { cond, m, dst, .. } => {
            if cond.test(cpu) {
                let w = cpu.osize();
                let v = cpu.read_rm_w(&m, w);
                cpu.set_reg_w(dst, w, v);
            }
        }
        // PUSH/POP a segment register. The pushed value occupies the full
        // operand size, but only the low 16 bits carry the selector.
        Inst::PushSeg { seg } => {
            let w = cpu.stack_width();
            let v = cpu.seg(seg) as u64;
            cpu.push_w(w, v);
        }
        Inst::PopSeg { seg } => {
            let w = cpu.stack_width();
            let v = cpu.pop_w(w) as u16;
            cpu.load_seg(seg, v);
        }
        // Debug registers. Nothing here implements hardware breakpoints, so
        // they are plain storage: the kernel writes DR7 = 0 and DR0-3 = 0 at
        // startup and expects the reads to agree, which this satisfies.
        Inst::MovDr { dr, reg } => {
            let w = if cpu.long_mode() { 64 } else { 32 };
            let v = cpu.dr[dr as usize];
            cpu.set_reg_w(reg, w, v);
        }
        Inst::MovToDr { dr, reg } => {
            let w = if cpu.long_mode() { 64 } else { 32 };
            cpu.dr[dr as usize] = cpu.reg_w(reg, w);
        }
        Inst::Lldt { m } => {
            let sel = cpu.read_rm16(&m);
            cpu.load_ldt(sel);
        }
        Inst::Ltr { m } => {
            let sel = cpu.read_rm16(&m);
            cpu.load_tr(sel);
        }
        Inst::Sldt { m } => { let v = cpu.ldt_selector; cpu.write_rm16(&m, v); }
        Inst::Str { m } => { let v = cpu.tr_selector; cpu.write_rm16(&m, v); }
        Inst::Leave { w32 } => {
            if cpu.long64() {
                let bp = cpu.reg64(5);
                cpu.set_rsp(bp);
                let v = cpu.pop_w(64);
                cpu.set_reg64(5, v);
            } else if w32 {
                cpu.set_reg32(Reg32::Esp, cpu.ebp());
                let v = cpu.pop32();
                cpu.set_reg32(Reg32::Ebp, v);
            } else {
                cpu.set_reg16(Reg16::Sp, cpu.bp());
                let v = cpu.pop16();
                cpu.set_reg16(Reg16::Bp, v);
            }
        }
        // POP r/m: the stack pop happens first, so a memory destination that
        // is addressed through ESP sees the *updated* stack pointer.
        Inst::PopRm16 { m } => {
            let v = cpu.pop16();
            cpu.write_rm16(&m, v);
        }
        Inst::PopRm32 { m } => {
            let w = cpu.stack_width();
            let v = cpu.pop_w(w);
            cpu.write_rm_w(&m, w, v);
        }

        // ---- String ops ----
        // Element size: byte forms (w=false) are 8-bit; word forms (w=true)
        // are 16-bit in 16-bit mode and 32-bit in 32-bit mode. The index
        // registers and REP counter follow the operand size.
        //
        // Optimization: compute the physical address once, then increment
        // it directly for each element. Only re-translate on page boundary
        // crossings (which are rare within a REP loop).
        // ---- String instructions ----
        //
        // All five share one shape, and one rule that is easy to miss and
        // expensive to get wrong: **a REP loop must be restartable.** If an
        // element faults, the instruction aborts with SI/DI/CX still pointing
        // at that element, so that when the handler returns and the
        // instruction re-runs it picks up exactly where it stopped. Running
        // the loop to completion through a fault instead writes every
        // remaining element to whatever address translation returned, and the
        // restart then copies nothing -- which is how a `clear_user` over a
        // not-yet-present page left a program's BSS full of file data.
        //
        // The index registers follow the *address* size and the element the
        // *operand* size; they are not the same thing when a 0x66 prefix is
        // in play.
        Inst::Movs { rep, w } => {
            let bits = if w { cpu.osize() } else { 8 };
            let esize = bits / 8;
            let step = string_step(cpu, esize);
            let asize = cpu.asize();
            let (mut si, mut di) = (string_si(cpu, asize), string_di(cpu, asize));
            let mut cnt = string_count(cpu, asize, rep);
            while cnt > 0 {
                let seg = cpu.operand_seg_for_exec(SegReg::Ds);
                let src = cpu.translate(seg, si);
                let dst = cpu.translate_write(SegReg::Es, di);
                if cpu.pending_exception.is_some() { break; }
                let v = str_read(cpu, src, seg, si, bits);
                str_write(cpu, dst, SegReg::Es, di, bits, v);
                if cpu.pending_exception.is_some() { break; }
                si = string_advance(si, step, asize);
                di = string_advance(di, step, asize);
                cnt -= 1;
            }
            string_set_si(cpu, asize, si);
            string_set_di(cpu, asize, di);
            if rep != Rep::None { string_set_count(cpu, asize, cnt); }
        }
        Inst::Stos { rep, w } => {
            let bits = if w { cpu.osize() } else { 8 };
            let esize = bits / 8;
            let step = string_step(cpu, esize);
            let asize = cpu.asize();
            let mut di = string_di(cpu, asize);
            let mut cnt = string_count(cpu, asize, rep);
            while cnt > 0 {
                let dst = cpu.translate_write(SegReg::Es, di);
                if cpu.pending_exception.is_some() { break; }
                let v = cpu.reg_w(0, bits);
                str_write(cpu, dst, SegReg::Es, di, bits, v);
                if cpu.pending_exception.is_some() { break; }
                di = string_advance(di, step, asize);
                cnt -= 1;
            }
            string_set_di(cpu, asize, di);
            if rep != Rep::None { string_set_count(cpu, asize, cnt); }
        }
        Inst::Lods { rep, w } => {
            let bits = if w { cpu.osize() } else { 8 };
            let esize = bits / 8;
            let step = string_step(cpu, esize);
            let asize = cpu.asize();
            let mut si = string_si(cpu, asize);
            let mut cnt = string_count(cpu, asize, rep);
            while cnt > 0 {
                let seg = cpu.operand_seg_for_exec(SegReg::Ds);
                let src = cpu.translate(seg, si);
                if cpu.pending_exception.is_some() { break; }
                let v = str_read(cpu, src, seg, si, bits);
                if cpu.pending_exception.is_some() { break; }
                cpu.set_reg_w(0, bits, v);
                si = string_advance(si, step, asize);
                cnt -= 1;
            }
            string_set_si(cpu, asize, si);
            if rep != Rep::None { string_set_count(cpu, asize, cnt); }
        }
        Inst::Cmps { rep, w } => {
            let bits = if w { cpu.osize() } else { 8 };
            let esize = bits / 8;
            let step = string_step(cpu, esize);
            let asize = cpu.asize();
            let (mut si, mut di) = (string_si(cpu, asize), string_di(cpu, asize));
            let mut cnt = string_count(cpu, asize, rep);
            while cnt > 0 {
                let seg = cpu.operand_seg_for_exec(SegReg::Ds);
                let src = cpu.translate(seg, si);
                let dst = cpu.translate(SegReg::Es, di);
                if cpu.pending_exception.is_some() { break; }
                let a = mem_read_w(cpu, src, bits);
                let b = mem_read_w(cpu, dst, bits);
                alu_w(cpu, AluOp::Cmp, a, b, bits);
                si = string_advance(si, step, asize);
                di = string_advance(di, step, asize);
                cnt -= 1;
                if !string_repeat(cpu, rep, cnt) { break; }
            }
            string_set_si(cpu, asize, si);
            string_set_di(cpu, asize, di);
            if rep != Rep::None { string_set_count(cpu, asize, cnt); }
        }
        Inst::Scas { rep, w } => {
            let bits = if w { cpu.osize() } else { 8 };
            let esize = bits / 8;
            let step = string_step(cpu, esize);
            let asize = cpu.asize();
            let mut di = string_di(cpu, asize);
            let mut cnt = string_count(cpu, asize, rep);
            while cnt > 0 {
                let dst = cpu.translate(SegReg::Es, di);
                if cpu.pending_exception.is_some() { break; }
                let a = cpu.reg_w(0, bits);
                let b = mem_read_w(cpu, dst, bits);
                alu_w(cpu, AluOp::Cmp, a, b, bits);
                di = string_advance(di, step, asize);
                cnt -= 1;
                if !string_repeat(cpu, rep, cnt) { break; }
            }
            string_set_di(cpu, asize, di);
            if rep != Rep::None { string_set_count(cpu, asize, cnt); }
        }

        // ---- LGDT / LIDT ----
        // The address size of the memory operand follows the current
        // addressing mode (addrsize): 32-bit (modrm_addr32) in a D=1
        // segment, 16-bit (modrm_addr) otherwise. The decoder already
        // fetched the ModR/M, SIB, and displacement bytes according to
        // addrsize; the executor must compute the address the same way.
        // The pseudo-descriptor is a 16-bit limit followed by the table
        // base -- four bytes of it in a legacy mode, eight in long mode,
        // where a descriptor table can live anywhere in the 64-bit space.
        Inst::Lgdt { m } => {
            let addr = cpu.rm_addr(&m, false);
            let limit = cpu.mem.read_u16(addr);
            let base = if cpu.long_mode() {
                cpu.mem.read_u64(addr + 2)
            } else {
                cpu.mem.read_u32(addr + 2) as u64
            };
            cpu.gdt_base = base;
            cpu.gdt_limit = limit;
        }
        Inst::Lidt { m } => {
            let addr = cpu.rm_addr(&m, false);
            let limit = cpu.mem.read_u16(addr);
            let base = if cpu.long_mode() {
                cpu.mem.read_u64(addr + 2)
            } else {
                cpu.mem.read_u32(addr + 2) as u64
            };
            cpu.idt_base = base;
            cpu.idt_limit = limit;
        }
        // SGDT / SIDT store the register the way LGDT / LIDT read it: a
        // 16-bit limit and then the base, 8 bytes wide in long mode and 4
        // otherwise (a 16-bit operand size does not narrow the base on any
        // CPU newer than a 286, and neither does it here).
        Inst::Sgdt { m } | Inst::Sidt { m } => {
            let (base, limit) = if matches!(inst, Inst::Sgdt { .. }) {
                (cpu.gdt_base, cpu.gdt_limit)
            } else {
                (cpu.idt_base, cpu.idt_limit)
            };
            let addr = cpu.rm_addr(&m, true);
            if cpu.pending_exception.is_some() { return; }
            cpu.mem.write_u16(addr, limit);
            if cpu.long_mode() {
                cpu.mem.write_u64(addr + 2, base);
            } else {
                cpu.mem.write_u32(addr + 2, base as u32);
            }
        }
        // SMSW: the low 16 bits of CR0 (a 32/64-bit register destination
        // gets the whole of CR0; a memory destination is always 16 bits).
        Inst::Smsw { m } => {
            if m.is_reg() {
                let w = cpu.osize();
                let v = if w == 16 { cpu.cr0 as u64 & 0xFFFF } else { cpu.cr0 as u64 };
                cpu.write_rm_w(&m, w, v);
            } else {
                let v = cpu.cr0 as u16;
                cpu.write_rm16(&m, v);
            }
        }
        // LMSW writes CR0's low four bits (PE, MP, EM, TS) and can set PE
        // but never clear it -- the 286 had no way back to real mode.
        Inst::Lmsw { m } => {
            if cpu.cpl() != 0 { cpu.raise_gp(0); return; }
            let v = cpu.read_rm16(&m) as u32;
            if cpu.pending_exception.is_some() { return; }
            let pe = cpu.cr0 & 1;
            let new = (cpu.cr0 & !0xE) | (v & 0xE) | pe | (v & 1);
            cpu.write_cr0(new);
        }
        // XGETBV / XSETBV: XCR0 says which register state XSAVE manages. Only
        // XCR0 exists (ECX must be 0), bit 0 (x87) can never be cleared, and
        // this CPU has no state past SSE, so only bits 0-1 may be set.
        Inst::Xgetbv => {
            if cpu.cr4 & CR4_OSXSAVE == 0 { cpu.raise_ud(); return; }
            if cpu.reg32(Reg32::Ecx) != 0 { cpu.raise_gp(0); return; }
            let v = cpu.xcr0;
            cpu.set_reg32(Reg32::Eax, v as u32);
            cpu.set_reg32(Reg32::Edx, (v >> 32) as u32);
        }
        Inst::Xsetbv => {
            if cpu.cr4 & CR4_OSXSAVE == 0 { cpu.raise_ud(); return; }
            if cpu.cpl() != 0 || cpu.reg32(Reg32::Ecx) != 0 { cpu.raise_gp(0); return; }
            let v = (cpu.reg32(Reg32::Edx) as u64) << 32 | cpu.reg32(Reg32::Eax) as u64;
            if v & 1 == 0 || v & !0x3 != 0 { cpu.raise_gp(0); return; }
            cpu.xcr0 = v;
        }
        // CLAC / STAC: RFLAGS.AC is the SMAP override in supervisor mode.
        // Privileged, and #UD rather than #GP outside ring 0.
        Inst::Clac => { if cpu.cpl() != 0 { cpu.raise_ud(); return; } cpu.flags &= !flags::AC; }
        Inst::Stac => { if cpu.cpl() != 0 { cpu.raise_ud(); return; } cpu.flags |= flags::AC; }

        // ---- INVLPG (0x0F 0x01 /7) ----
        // Invalidate the TLB entry for the linear address of the memory
        // operand. The linear address is computed the same way as a normal
        // memory operand (segment + offset), then we invalidate that page.
        Inst::Invlpg { m } => {
            // INVLPG names a *linear* address, not a physical one, so the
            // operand is resolved through the segment but not through the
            // page tables -- which is the whole point: the mapping it is
            // dropping may be the one that no longer works.
            let linear = cpu.modrm_linear(&m);
            cpu.invlpg(linear);
        }

        // ---- MOV to/from control registers ----
        // A control-register move is always the full width of the mode: 64
        // bits in long mode, 32 otherwise, with no way to ask for anything
        // else (REX.W is redundant and 0x66 is ignored).
        Inst::MovCr { cr, reg } => {
            let v: u64 = match cr {
                0 => cpu.cr0 as u64,
                2 => cpu.cr2,
                3 => cpu.cr3,
                4 => cpu.cr4 as u64,
                // CR8 is the task-priority register, which exists only in
                // long mode and only alongside a local APIC. There is no APIC
                // here, so it reads back as the zero it was written.
                _ => cpu.cr8,
            };
            let w = if cpu.long_mode() { 64 } else { 32 };
            cpu.set_reg_w(reg, w, v);
        }
        Inst::MovToCr { cr, reg } => {
            let w = if cpu.long_mode() { 64 } else { 32 };
            let v = cpu.reg_w(reg, w);
            match cr {
                // `write_cr0` refreshes PE, flushes the TLB when PG toggles,
                // and enters or leaves long mode (PG on with EFER.LME set is
                // what enters).
                0 => cpu.write_cr0(v as u32),
                2 => cpu.cr2 = v,
                3 => {
                    cpu.cr3 = v;
                    cpu.flush_tlb();
                }
                4 => {
                    // Writing CR4 flushes the TLB. Linux's
                    // `__flush_tlb_global()` is *literally* a CR4 write with
                    // PGE toggled off and back on -- the flush is the whole
                    // point of the sequence, and without it every global
                    // mapping keeps a stale translation. Toggling PAE changes
                    // the shape of the page tables, so the flush is not
                    // optional there either.
                    cpu.cr4 = v as u32;
                    cpu.flush_tlb();
                }
                _ => cpu.cr8 = v & 0xF,
            }
        }

        // ---- CLTS (0x0F 0x06) ----
        Inst::Clts => {
            cpu.cr0 &= !0x8; // clear CR0.TS (bit 3)
        }

        // Instructions with nothing to do on this machine: memory fences (no
        // store buffer to drain), prefetch and cache hints (no cache), and
        // the multi-byte NOP compilers use for alignment.
        Inst::NopHint => {}

        // ---- SYSCALL / SYSRET (0F 05 / 0F 07) ----
        //
        // The fast system-call pair, and the only way into a 64-bit kernel:
        // 64-bit Linux does not install an `int 0x80` path for 64-bit
        // processes at all. What makes it fast is that it consults no table
        // and touches no memory -- the entry point and the segments come out
        // of MSRs, and the return address goes in a register instead of onto
        // a stack the kernel would have to trust.
        //
        // The cost of that is spelled out in the semantics: **SYSCALL does
        // not switch stacks**. It lands in the kernel still running on the
        // user stack, with RSP under user control, which is exactly why every
        // 64-bit kernel entry stub begins with SWAPGS and a load of the real
        // stack out of per-CPU data.
        Inst::Syscall => {
            if cpu.efer & crate::cpu::efer::SCE == 0 {
                // SYSCALL is disabled: #UD, not a silent fall-through.
                cpu.pending_exception = Some((0x06, None));
                return;
            }
            // RCX takes the return address and R11 the flags -- both are
            // clobbered, which is why the 64-bit ABI lists them as
            // caller-saved and why no system call passes an argument in them.
            cpu.set_reg64_raw(1, cpu.rip);
            cpu.set_reg64_raw(11, cpu.flags as u64);
            // STAR bits 47:32 hold the kernel CS; SS is the next descriptor.
            let sel = ((cpu.star >> 32) & 0xFFFF) as u16;
            cpu.flags &= !(cpu.sfmask as u32);
            cpu.flags |= flags::ALWAYS_SET;
            cpu.load_seg(SegReg::Cs, sel & 0xFFFC);
            cpu.load_seg(SegReg::Ss, (sel & 0xFFFC).wrapping_add(8));
            cpu.rip = cpu.lstar;
            cpu.ring_switches += 1;
            cpu.invalidate_phys_ip();
        }
        Inst::Sysret => {
            // Returning: RCX back to RIP, R11 back to RFLAGS, and the user
            // segments from STAR bits 63:48. REX.W selects the 64-bit form;
            // without it the return is to compatibility mode.
            let sel = ((cpu.star >> 48) & 0xFFFF) as u16;
            let (cs, ss) = if cpu.rex_w {
                // 64-bit: CS is base+16, SS is base+8, both with RPL 3.
                ((sel + 16) | 3, (sel + 8) | 3)
            } else {
                (sel | 3, (sel + 8) | 3)
            };
            cpu.rip = cpu.reg64(1);
            cpu.flags = write_flags(cpu.flags, cpu.reg64(11) as u32);
            cpu.load_seg(SegReg::Cs, cs);
            cpu.load_seg(SegReg::Ss, ss);
            cpu.invalidate_phys_ip();
        }

        // ---- SWAPGS (0F 01 F8) ----
        //
        // Exchange the GS base with the one parked in KERNEL_GS_BASE. It
        // exists because a kernel entered by SYSCALL has no trustworthy
        // register and no stack of its own: this is the one instruction that
        // gets it a pointer to its per-CPU data without reading anything the
        // user could have arranged. It is a #UD outside 64-bit mode.
        Inst::Swapgs => {
            if !cpu.long64() {
                cpu.pending_exception = Some((0x06, None));
                return;
            }
            std::mem::swap(&mut cpu.gs_base, &mut cpu.kernel_gs_base);
        }

        // RDTSCP: RDTSC, plus the processor id (always 0 here) in ECX.
        Inst::Rdtscp => {
            let tsc = cpu.rdtsc();
            cpu.set_reg32(Reg32::Eax, tsc as u32);
            cpu.set_reg32(Reg32::Edx, (tsc >> 32) as u32);
            cpu.set_reg32(Reg32::Ecx, 0);
        }

        // ---- CPUID (0x0F 0xA2) ----
        Inst::Cpuid => {
            let leaf = cpu.reg32(Reg32::Eax);
            match leaf {
                0 => {
                    // Highest basic leaf = 1, vendor string "GenuineIntel".
                    cpu.set_reg32(Reg32::Eax, 1);
                    cpu.set_reg32(Reg32::Ebx, 0x756E6547); // "Genu"
                    cpu.set_reg32(Reg32::Edx, 0x49656E69); // "ineI"
                    cpu.set_reg32(Reg32::Ecx, 0x6C65746E); // "ntel"
                }
                1 => {
                    // Family 6, model 0, stepping 0.
                    cpu.set_reg32(Reg32::Eax, 0x00000600);
                    // EBX: brand index 0, CLFLUSH line size 8 (x8 = 64
                    // bytes), one logical processor, APIC id 0. The line
                    // size is not decoration: Linux takes it as the cache
                    // alignment, and a zero there makes ALIGN(x, 0) collapse
                    // to 0 -- which reaches reciprocal_value() as a divide
                    // by zero before the slab allocator is even up.
                    cpu.set_reg32(Reg32::Ebx, 0x0001_0800);
                    // ECX: SSE3 (0) and VMX (5). Not claimed, because they
                    // are not implemented: SSSE3, SSE4.x, CX16, POPCNT,
                    // XSAVE, AVX -- a userspace that dispatches on these
                    // would then run instructions this CPU lacks.
                    cpu.set_reg32(Reg32::Ecx, 0x0000_0021);
                    // Feature flags, deliberately only the ones implemented
                    // here: FPU(0), PSE(3), TSC(4), MSR(5), PAE(6), CX8(8),
                    // PGE(13), CMOV(15), CLFSH(19), MMX(23), FXSR(24),
                    // SSE(25), SSE2(26).
                    //
                    // Left OFF on purpose, because claiming them would make
                    // the kernel issue instructions this CPU does not have:
                    // APIC(9), SEP(11) -- so 32-bit system calls arrive as
                    // int 0x80 rather than SYSENTER -- MTRR(12), PAT(16).
                    cpu.set_reg32(Reg32::Edx, 0x0788_A179);
                }
                0x8000_0000 => {
                    // Highest extended leaf. A CPU that does not answer this
                    // one is a CPU without long mode as far as any bootloader
                    // is concerned -- the check is "is 0x80000001 reachable",
                    // long before anything looks at its feature bits.
                    cpu.set_reg32(Reg32::Eax, 0x8000_0008);
                    cpu.set_reg32(Reg32::Ebx, 0);
                    cpu.set_reg32(Reg32::Ecx, 0);
                    cpu.set_reg32(Reg32::Edx, 0);
                }
                0x8000_0001 => {
                    cpu.set_reg32(Reg32::Eax, 0);
                    cpu.set_reg32(Reg32::Ebx, 0);
                    // ECX: LAHF/SAHF valid in 64-bit mode (bit 0).
                    cpu.set_reg32(Reg32::Ecx, 0x0000_0021);
                    // EDX extended features, again only the implemented ones:
                    // FPU(0), PSE(3), TSC(4), MSR(5), PAE(6), CX8(8), PGE(13),
                    // CMOV(15), NX(20), SYSCALL(11), MMX(23), 1 GiB pages(26),
                    // SSE2(26 in the basic leaf is bit 26; here the AMD
                    // extended leaf reuses the same bit positions), RDTSCP
                    // left off, and **LM (bit 29)** -- the bit that says this
                    // CPU has long mode at all.
                    cpu.set_reg32(Reg32::Edx, 0x2490_A97B);
                }
                0x8000_0008 => {
                    // Physical and linear address sizes: 52 and 48, which is
                    // what the page-table walk actually implements.
                    cpu.set_reg32(Reg32::Eax, 0x0000_3034);
                    cpu.set_reg32(Reg32::Ebx, 0);
                    cpu.set_reg32(Reg32::Ecx, 0);
                    cpu.set_reg32(Reg32::Edx, 0);
                }
                _ => {
                    // Unknown leaf: report 0.
                    cpu.set_reg32(Reg32::Eax, 0);
                    cpu.set_reg32(Reg32::Ebx, 0);
                    cpu.set_reg32(Reg32::Ecx, 0);
                    cpu.set_reg32(Reg32::Edx, 0);
                }
            }
        }

        // ---- RDTSC (0x0F 0x31) ----
        Inst::Rdtsc => {
            let tsc = cpu.rdtsc();
            cpu.set_reg32(Reg32::Eax, tsc as u32);
            cpu.set_reg32(Reg32::Edx, (tsc >> 32) as u32);
        }

        // ---- RDMSR (0x0F 0x32) / WRMSR (0x0F 0x30) ----
        // ECX names the MSR; the value is EDX:EAX, high half first. The
        // registers this CPU actually keeps are the ones long mode needs:
        // EFER, the SYSCALL configuration, and the FS/GS bases. Everything
        // else reads back as zero and swallows writes, which is what lets a
        // kernel probe for features without faulting.
        Inst::Rdmsr => {
            let v = cpu.read_msr(cpu.reg32(Reg32::Ecx));
            cpu.set_reg32(Reg32::Eax, v as u32);
            cpu.set_reg32(Reg32::Edx, (v >> 32) as u32);
        }
        Inst::Wrmsr => {
            let v = (cpu.reg32(Reg32::Eax) as u64) | ((cpu.reg32(Reg32::Edx) as u64) << 32);
            let idx = cpu.reg32(Reg32::Ecx);
            cpu.write_msr(idx, v);
        }

        // ---- Bit tests: BT / BTS / BTR / BTC ----
        Inst::Bt { m, bit } => bit_op(cpu, &m, bit, BitOp::Test),
        Inst::Bts { m, bit } => bit_op(cpu, &m, bit, BitOp::Set),
        Inst::Btr { m, bit } => bit_op(cpu, &m, bit, BitOp::Reset),
        Inst::Btc { m, bit } => bit_op(cpu, &m, bit, BitOp::Complement),

        // ---- IN / OUT ----
        Inst::InAlImm { port } => {
            let v = cpu.port_in(port as u16);
            cpu.set_reg8(Reg8::Al, v);
        }
        Inst::InAxImm { port } => {
            let v = cpu.port_in16(port as u16);
            cpu.set_reg16(Reg16::Ax, v);
        }
        Inst::InAlDx => {
            let v = cpu.port_in(cpu.dx());
            cpu.set_reg8(Reg8::Al, v);
        }
        Inst::InAxDx => {
            let v = cpu.port_in16(cpu.dx() as u16);
            cpu.set_reg16(Reg16::Ax, v);
        }
        Inst::OutImmAl { port } => {
            let v = cpu.reg8(Reg8::Al);
            cpu.port_out(port as u16, v);
        }
        Inst::OutImmAx { port } => {
            let v = cpu.reg16(Reg16::Ax);
            cpu.port_out16(port as u16, v);
        }
        Inst::OutDxAl => {
            let v = cpu.reg8(Reg8::Al);
            cpu.port_out(cpu.dx(), v);
        }
        Inst::OutDxAx => {
            let v = cpu.reg16(Reg16::Ax);
            cpu.port_out16(cpu.dx() as u16, v);
        }

        // ---- Flag-control instructions ----
        Inst::Clc => cpu.set_flag(flags::CF, false),
        Inst::Stc => cpu.set_flag(flags::CF, true),
        Inst::Cli => cpu.set_flag(flags::IF, false),
        Inst::Sti => cpu.set_flag(flags::IF, true),
        Inst::Cld => cpu.set_flag(flags::DF, false),
        Inst::Std => cpu.set_flag(flags::DF, true),
        Inst::Cmc => cpu.set_flag(flags::CF, !cpu.get_flag(flags::CF)),

        // ---- x87 FPU ----
        Inst::Fninit => cpu.fpu.finit(),
        Inst::Fstcw { m } => {
            let v = cpu.fpu.control;
            if m.is_reg() {
                // Store to a register is meaningless; treat as no-op.
            } else if cpu.addrsize {
                let a = cpu.modrm_addr32_write(&m);
                cpu.mem.write_u16(a, v);
            } else {
                let a = cpu.modrm_addr_write(&m);
                cpu.mem.write_u16(a, v);
            }
        }
        Inst::Fldcw { m } => {
            let v = if cpu.addrsize {
                let a = cpu.modrm_addr32(&m);
                cpu.mem.read_u16(a)
            } else {
                let a = cpu.modrm_addr(&m);
                cpu.mem.read_u16(a)
            };
            cpu.fpu.control = v;
        }
        Inst::FstswAx => {
            cpu.set_reg16(Reg16::Ax, cpu.fpu.fstsw());
        }
        Inst::Fstsw { m } => {
            let v = cpu.fpu.fstsw();
            if cpu.addrsize {
                let a = cpu.modrm_addr32_write(&m);
                cpu.mem.write_u16(a, v);
            } else {
                let a = cpu.modrm_addr_write(&m);
                cpu.mem.write_u16(a, v);
            }
        }
        Inst::Fst { m, w64 } => {
            let v = cpu.fpu.st_i(0);
            if m.is_reg() {
                // FST ST(i): copy ST0 to ST(i).
                cpu.fpu.set_st_i(m.rm as usize, v);
            } else {
                let a = if cpu.addrsize { cpu.modrm_addr32_write(&m) } else { cpu.modrm_addr_write(&m) };
                if w64 { cpu.mem.write_f64(a, v); } else { cpu.mem.write_f32(a, v as f32); }
            }
        }
        Inst::Fstp { m, w64 } => {
            let v = cpu.fpu.st_i(0);
            if m.is_reg() {
                cpu.fpu.set_st_i(m.rm as usize, v);
            } else {
                let a = if cpu.addrsize { cpu.modrm_addr32_write(&m) } else { cpu.modrm_addr_write(&m) };
                if w64 { cpu.mem.write_f64(a, v); } else { cpu.mem.write_f32(a, v as f32); }
            }
            cpu.fpu.pop();
        }
        Inst::Fld { m, w64 } => {
            let v = if m.is_reg() {
                cpu.fpu.st_i(m.rm as usize)
            } else {
                let a = if cpu.addrsize { cpu.modrm_addr32(&m) } else { cpu.modrm_addr(&m) };
                if w64 { cpu.mem.read_f64(a) } else { cpu.mem.read_f32(a) as f64 }
            };
            cpu.fpu.push(v);
        }
        Inst::Fild { m } => {
            // Integer load: read a 32-bit signed int (or 16-bit) and push.
            let v = if cpu.addrsize {
                let a = cpu.modrm_addr32(&m);
                cpu.mem.read_u32(a) as i32 as f64
            } else {
                let a = cpu.modrm_addr(&m);
                cpu.mem.read_u16(a) as i16 as f64
            };
            cpu.fpu.push(v);
        }
        Inst::Fistp { m } => {
            let v = cpu.fpu.st_i(0) as i32;
            if cpu.addrsize {
                let a = cpu.modrm_addr32(&m);
                cpu.mem.write_u32(a, v as u32);
            } else {
                let a = cpu.modrm_addr(&m);
                cpu.mem.write_u16(a, v as u16);
            }
            cpu.fpu.pop();
        }
        Inst::Fop { op, m } => {
            // Simplified arithmetic: ST0 op m (m read as f64 or int).
            let rhs = if m.is_reg() {
                cpu.fpu.st_i(m.rm as usize)
            } else if cpu.addrsize {
                let a = cpu.modrm_addr32(&m);
                cpu.mem.read_f64(a)
            } else {
                let a = cpu.modrm_addr(&m);
                cpu.mem.read_f64(a)
            };
            let st0 = cpu.fpu.st_i(0);
            let result = match op {
                FpuOp::Add => st0 + rhs,
                FpuOp::Sub => st0 - rhs,
                FpuOp::Mul => st0 * rhs,
                FpuOp::Div => if rhs != 0.0 { st0 / rhs } else { st0 },
            };
            cpu.fpu.set_st_i(0, result);
        }

        // Two/three-operand IMUL. CF = OF = 1 when the truncated result
        // differs from the full signed product (i.e. it did not fit).
        Inst::ImulRegRm16 { m, dst } => {
            let a = cpu.read_rm16(&m) as i16 as i32;
            let b = cpu.reg16_idx(dst) as i16 as i32;
            imul_store16(cpu, dst, a * b);
        }
        Inst::ImulRegRm32 { m, dst } => {
            let w = cpu.osize();
            let a = sign_extend(cpu.read_rm_w(&m, w), w);
            let b = sign_extend(cpu.reg_w(dst, w), w);
            if w == 64 {
                imul_store64(cpu, dst, (a as i128) * (b as i128));
            } else {
                imul_store32(cpu, dst, a.wrapping_mul(b));
            }
        }
        Inst::ImulRegRmImm16 { m, dst, imm } => {
            let a = cpu.read_rm16(&m) as i16 as i32;
            imul_store16(cpu, dst, a * imm as i32);
        }
        Inst::ImulRegRmImm32 { m, dst, imm } => {
            let w = cpu.osize();
            let a = sign_extend(cpu.read_rm_w(&m, w), w);
            if w == 64 {
                imul_store64(cpu, dst, (a as i128) * (imm as i128));
            } else {
                imul_store32(cpu, dst, a.wrapping_mul(imm as i64));
            }
        }

        // SHLD/SHRD: shift the destination, feeding in bits from the source
        // register. A count of 0 is a no-op that leaves every flag alone; a
        // count >= the operand width is architecturally undefined, and like
        // real hardware we mask it to 5 bits and let it fall out.
        Inst::Shld { m, reg, count, w32 } => {
            let w = cpu.osize();
            let n = match count { ShiftCount::One => 1, ShiftCount::Imm(i) => i, ShiftCount::Cl => cpu.reg8(Reg8::Cl) }
                & if w == 64 { 0x3F } else { 0x1F };
            if n != 0 {
                if w == 64 {
                    let d = cpu.read_rm_w(&m, 64);
                    let src = cpu.reg_w(reg, 64);
                    let nn = n as u32;
                    let res = (d << nn) | (src >> (64 - nn));
                    let cf = (d >> (64 - nn)) & 1 != 0;
                    cpu.write_rm_w(&m, 64, res);
                    set_shift_flags64(cpu, res, cf, (d ^ res) >> 63 & 1 != 0);
                } else if w32 {
                    let d = cpu.read_rm32(&m);
                    let src = cpu.reg32_idx(reg);
                    let res = (d << n) | (src >> (32 - n));
                    let cf = (d >> (32 - n)) & 1 != 0;
                    cpu.write_rm32(&m, res);
                    set_shift_flags32(cpu, res, cf, (d ^ res) >> 31 & 1 != 0);
                } else {
                    let d = cpu.read_rm16(&m);
                    let src = cpu.reg16_idx(reg);
                    // A 16-bit SHLD with count > 16 feeds in bits that a real
                    // CPU leaves undefined; do the shift in 32 bits so the
                    // in-range cases are exact.
                    let wide = ((d as u32) << 16) | src as u32;
                    let res = ((wide << n) >> 16) as u16;
                    let cf = (d >> (16 - n.min(16))) & 1 != 0;
                    cpu.write_rm16(&m, res);
                    set_shift_flags16(cpu, res, cf, (d ^ res) >> 15 & 1 != 0);
                }
            }
        }
        Inst::Shrd { m, reg, count, w32 } => {
            let w = cpu.osize();
            let n = match count { ShiftCount::One => 1, ShiftCount::Imm(i) => i, ShiftCount::Cl => cpu.reg8(Reg8::Cl) }
                & if w == 64 { 0x3F } else { 0x1F };
            if n != 0 {
                if w == 64 {
                    let d = cpu.read_rm_w(&m, 64);
                    let src = cpu.reg_w(reg, 64);
                    let nn = n as u32;
                    let res = (d >> nn) | (src << (64 - nn));
                    let cf = (d >> (nn - 1)) & 1 != 0;
                    cpu.write_rm_w(&m, 64, res);
                    set_shift_flags64(cpu, res, cf, (d ^ res) >> 63 & 1 != 0);
                } else if w32 {
                    let d = cpu.read_rm32(&m);
                    let src = cpu.reg32_idx(reg);
                    let res = (d >> n) | (src << (32 - n));
                    let cf = (d >> (n - 1)) & 1 != 0;
                    cpu.write_rm32(&m, res);
                    set_shift_flags32(cpu, res, cf, (d ^ res) >> 31 & 1 != 0);
                } else {
                    let d = cpu.read_rm16(&m);
                    let src = cpu.reg16_idx(reg);
                    let wide = ((src as u32) << 16) | d as u32;
                    let res = (wide >> n) as u16;
                    let cf = (d >> (n - 1).min(15)) & 1 != 0;
                    cpu.write_rm16(&m, res);
                    set_shift_flags16(cpu, res, cf, (d ^ res) >> 15 & 1 != 0);
                }
            }
        }

        // SETcc: write 1 or 0 to an 8-bit destination. The operand is always
        // byte-sized regardless of the operand-size prefix.
        Inst::Setcc { cond, m } => {
            let v = if cond.test(cpu) { 1u8 } else { 0u8 };
            cpu.write_rm8(&m, v);
        }

        Inst::Sse(s) => crate::sse::execute_sse(cpu, &s),
        Inst::Vmx(v) => crate::vmx::execute_vmx(cpu, &v),

        Inst::Unknown { opcode } => {
            // Invalid opcode exception (#UD, vector 0x06). No error code.
            // Record the opcode so a debug run can list every instruction the
            // decoder is missing in one pass (see `Cpu::unknown_ops`).
            cpu.note_unknown_opcode(opcode);
            cpu.pending_exception = Some((0x06, None));
        }
    }
}

/// Dispatch a protected-mode interrupt through the IDT.
pub(crate) fn protected_int(cpu: &mut Cpu, vector: u8) {
    protected_int_err(cpu, vector, None)
}

/// Dispatch a protected-mode interrupt or exception through the IDT.
///
/// The frame order is not a detail: the CPU pushes EFLAGS, CS and EIP, and
/// then — for the exceptions that have one — the **error code last**, so it
/// ends up on top of the stack. Linux's `error_code` entry stub reads the
/// frame at fixed offsets from ESP, so pushing the error code first shifts
/// every field by one slot and the kernel reports the fault at the CS
/// selector's "address" with the real EIP as the error code.
pub(crate) fn protected_int_err(cpu: &mut Cpu, vector: u8, error_code: Option<u32>) {
    if cpu.long_mode() {
        return long_int(cpu, vector, error_code);
    }
    // IDT entry: 8 bytes. offset = (bytes 0-1) | (bytes 6-7 << 16), the
    // segment selector is bytes 2-3, and byte 5 holds the type/attributes.
    let entry = cpu.idt_base.wrapping_add((vector as u64) * 8);
    // The IDT base is a *linear* address, so a kernel that runs in the higher
    // half keeps its IDT up there too. Resolve it the way the CPU does.
    let addr = cpu.linear_to_phys_ro(entry as u64);
    let off_lo = cpu.mem.read_u16(addr) as u32;
    let off_hi = cpu.mem.read_u16(addr + 6) as u32;
    let target = off_lo | (off_hi << 16);
    let selector = cpu.mem.read_u16(addr + 2);
    let gate_type = cpu.mem.read_u8(addr + 5) & 0x0F;

    // A gate to a more privileged code segment switches stacks. The new SS
    // and ESP come from the TSS, and the *old* SS:ESP are pushed below the
    // usual frame -- without which the kernel's entry code reads a frame two
    // dwords short and IRET has nothing to return to user mode on.
    let target_dpl = cpu.descriptor_for(selector).dpl();
    let switching = target_dpl < cpu.cpl();

    // Everything pushed is the state as it was *before* the gate.
    let old_cs = cpu.cs;
    let old_eip = cpu.eip();
    let old_flags = cpu.flags;
    let (old_ss, old_esp) = (cpu.ss, cpu.esp());

    if switching {
        cpu.ring_switches += 1;
        let (ss0, esp0) = cpu.tss_stack0();
        cpu.load_seg(SegReg::Ss, ss0);
        cpu.set_esp(esp0);
    }
    // Enter the handler's privilege level BEFORE writing the frame. The
    // pushes are part of the gate transition and happen at the new CPL: done
    // the other way round, a frame written on the ring-0 stack while CS still
    // says ring 3 is a user write to supervisor memory, and paging rejects it.
    cpu.load_seg(SegReg::Cs, selector);
    if switching {
        cpu.push32(old_ss as u32);
        cpu.push32(old_esp);
    }
    cpu.push32(old_flags);
    cpu.push32(old_cs as u32);
    cpu.push32(old_eip);
    if let Some(code) = error_code {
        cpu.push32(code);
    }
    // An interrupt gate (type 6/14) clears IF on entry; a trap gate (7/15)
    // leaves it alone, which is how Linux keeps interrupts enabled inside
    // handlers like int3 and the system-call entry.
    if gate_type == 0x6 || gate_type == 0xE {
        cpu.set_flag(flags::IF, false);
    }
    cpu.set_flag(flags::TF, false);
    cpu.set_flag(flags::RF, false);
    cpu.set_eip(target);
    cpu.invalidate_phys_ip();
}

/// Dispatch an interrupt or exception through a **long-mode** IDT.
///
/// Three things differ from the 32-bit form, and all three are load-bearing:
///
/// 1. A gate is **sixteen** bytes, with the 64-bit offset in three pieces and
///    an interrupt-stack-table index sharing a byte with the reserved field.
/// 2. The frame is five 8-byte words and **SS:RSP are always pushed**, even
///    when the privilege level did not change. That is what makes `IRETQ` a
///    single shape rather than two, and a handler that pops only three words
///    returns to nonsense.
/// 3. A stack switch loads a **null** SS. There is no SS0 in a 64-bit TSS to
///    load anything else from; the segment is not used for anything but its
///    selector, and long mode is happy to run on a null one at ring 0.
fn long_int(cpu: &mut Cpu, vector: u8, error_code: Option<u32>) {
    let entry = cpu.idt_base.wrapping_add((vector as u64) * 16);
    let addr = cpu.linear_to_phys_ro(entry);
    let off_lo = cpu.mem.read_u16(addr) as u64;
    let selector = cpu.mem.read_u16(addr + 2);
    // Bits 0-2 of byte 4 are the IST index; the rest of the byte is reserved.
    let ist = cpu.mem.read_u8(addr + 4) & 7;
    let gate_type = cpu.mem.read_u8(addr + 5) & 0x0F;
    let off_mid = cpu.mem.read_u16(addr + 6) as u64;
    let off_hi = cpu.mem.read_u32(addr + 8) as u64;
    let target = off_lo | (off_mid << 16) | (off_hi << 32);

    let target_dpl = cpu.descriptor_for(selector).dpl();
    let switching = target_dpl < cpu.cpl();

    // Everything pushed is the state as it was *before* the gate.
    let old_cs = cpu.cs;
    let old_rip = cpu.rip;
    let old_flags = cpu.flags as u64;
    let old_ss = cpu.ss;
    let old_rsp = cpu.rsp();

    // Which stack to land on. An IST entry is taken *unconditionally* -- that
    // is the point of it: a double fault or an NMI must reach a stack that is
    // known good even when the one in RSP is what broke.
    if ist != 0 {
        let sp = cpu.tss_ist(ist);
        if switching { cpu.ring_switches += 1; }
        cpu.load_seg(SegReg::Ss, 0);
        cpu.set_rsp(sp);
    } else if switching {
        cpu.ring_switches += 1;
        let sp = cpu.tss_rsp0();
        cpu.load_seg(SegReg::Ss, 0);
        cpu.set_rsp(sp);
    }
    // Enter the handler's privilege level BEFORE writing the frame, for the
    // same reason as the 32-bit path: the pushes are part of the transition
    // and happen at the new CPL.
    cpu.load_seg(SegReg::Cs, selector);
    // The frame is aligned to 16 bytes before anything is pushed.
    let sp = cpu.rsp() & !0xF;
    cpu.set_rsp(sp);
    cpu.push64(old_ss as u64);
    cpu.push64(old_rsp);
    cpu.push64(old_flags);
    cpu.push64(old_cs as u64);
    cpu.push64(old_rip);
    if let Some(code) = error_code {
        cpu.push64(code as u64);
    }
    // An interrupt gate (type 14) clears IF on entry; a trap gate (15) leaves
    // it alone. Long mode has no 16-bit gates, so those are the only two.
    if gate_type == 0xE {
        cpu.set_flag(flags::IF, false);
    }
    cpu.set_flag(flags::TF, false);
    cpu.set_flag(flags::RF, false);
    cpu.rip = target;
    cpu.invalidate_phys_ip();
}

/// All-ones for the low `width` bits. `1 << 64` is undefined, so the widest
/// case cannot be written as `(1 << w) - 1`.
#[inline]
fn mask_w(width: u32) -> u64 {
    if width >= 64 { u64::MAX } else { (1u64 << width) - 1 }
}

/// Read a string element: `phys` is the translation of `(seg, off)`; a
/// straddling element is split across its two pages.
fn str_read(cpu: &mut Cpu, phys: usize, seg: SegReg, off: u64, width: u32) -> u64 {
    if width > 8 && Cpu::straddles(phys, width / 8) {
        let lin = cpu.linear_addr(seg, off);
        return cpu.read_split(phys, lin, width / 8) as u64;
    }
    mem_read_w(cpu, phys, width)
}

/// The store side of `str_read`.
fn str_write(cpu: &mut Cpu, phys: usize, seg: SegReg, off: u64, width: u32, v: u64) {
    if width > 8 && Cpu::straddles(phys, width / 8) {
        let lin = cpu.linear_addr(seg, off);
        cpu.write_split(phys, lin, width / 8, v as u128);
        return;
    }
    mem_write_w(cpu, phys, width, v)
}

/// Read `width` bits from a physical address.
fn mem_read_w(cpu: &Cpu, addr: usize, width: u32) -> u64 {
    match width {
        64 => cpu.mem.read_u64(addr),
        32 => cpu.mem.read_u32(addr) as u64,
        16 => cpu.mem.read_u16(addr) as u64,
        _ => cpu.mem.read_u8(addr) as u64,
    }
}

/// Write `width` bits to a physical address.
fn mem_write_w(cpu: &mut Cpu, addr: usize, width: u32, v: u64) {
    match width {
        64 => cpu.mem.write_u64(addr, v),
        32 => cpu.mem.write_u32(addr, v as u32),
        16 => cpu.mem.write_u16(addr, v as u16),
        _ => cpu.mem.write_u8(addr, v as u8),
    }
}

/// Sign-extend the low `width` bits of `v` to a full 64-bit value.
#[inline]
fn sext(v: u64, width: u32) -> u64 {
    sign_extend(v, width) as u64
}

/// Branch by a signed displacement, at the width of the current mode.
///
/// A near branch in 64-bit mode moves the whole of RIP; in a legacy mode it
/// wraps at 32 bits (or 16, with a 16-bit operand size). Doing it at one
/// fixed width is how a jump in the high half of a 64-bit address space lands
/// four gigabytes from where it meant to.
fn branch_rel(cpu: &mut Cpu, rel: i64) {
    if cpu.long64() {
        cpu.rip = cpu.rip.wrapping_add(rel as u64);
    } else if cpu.opsize {
        cpu.set_eip(cpu.eip().wrapping_add(rel as u32));
    } else {
        cpu.ip = cpu.ip.wrapping_add(rel as u16);
    }
}

// ---- ALU flag computation ----

fn alu8(cpu: &mut Cpu, op: AluOp, a: u8, b: u8) -> u8 {
    use flags::*;
    match op {
        AluOp::Add => {
            let (r, c) = a.overflowing_add(b);
            set_logic_flags8(cpu, r);
            set_add_carry(cpu, a as u16, b as u16, r as u16, c);
            r
        }
        AluOp::Adc => {
            let cin = cpu.get_flag(CF) as u16;
            let total = a as u16 + b as u16 + cin;
            let r = total as u8;
            let c = total > 0xFF;
            set_logic_flags8(cpu, r);
            set_add_carry(cpu, a as u16, b as u16, r as u16, c);
            r
        }
        AluOp::Sub | AluOp::Cmp => {
            let (r, c) = a.overflowing_sub(b);
            set_logic_flags8(cpu, r);
            set_sub_borrow(cpu, a as u16, b as u16, r as u16, c);
            r
        }
        AluOp::Sbb => {
            let cin = cpu.get_flag(CF) as u16;
            let total = a as u16 - b as u16 - cin;
            let r = total as u8;
            let c = (a as u16) < (b as u16 + cin);
            set_logic_flags8(cpu, r);
            set_sub_borrow(cpu, a as u16, b as u16, r as u16, c);
            r
        }
        AluOp::And => { let r = a & b; set_logic_flags8(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
        AluOp::Or  => { let r = a | b; set_logic_flags8(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
        AluOp::Xor => { let r = a ^ b; set_logic_flags8(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
    }
}

fn alu16(cpu: &mut Cpu, op: AluOp, a: u16, b: u16) -> u16 {
    use flags::*;
    match op {
        AluOp::Add => {
            let (r, c) = a.overflowing_add(b);
            set_logic_flags16(cpu, r);
            set_add_carry(cpu, a, b, r, c);
            r
        }
        AluOp::Adc => {
            let cin = cpu.get_flag(CF) as u32;
            let total = a as u32 + b as u32 + cin;
            let r = total as u16;
            let c = total > 0xFFFF;
            set_logic_flags16(cpu, r);
            set_add_carry(cpu, a, b, r, c);
            r
        }
        AluOp::Sub | AluOp::Cmp => {
            let (r, c) = a.overflowing_sub(b);
            set_logic_flags16(cpu, r);
            set_sub_borrow(cpu, a, b, r, c);
            r
        }
        AluOp::Sbb => {
            let cin = cpu.get_flag(CF) as u32;
            let total = (a as u32).wrapping_sub(b as u32).wrapping_sub(cin);
            let r = total as u16;
            let c = (a as u32) < (b as u32 + cin);
            set_logic_flags16(cpu, r);
            set_sub_borrow(cpu, a, b, r, c);
            r
        }
        AluOp::And => { let r = a & b; set_logic_flags16(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
        AluOp::Or  => { let r = a | b; set_logic_flags16(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
        AluOp::Xor => { let r = a ^ b; set_logic_flags16(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
    }
}

fn alu32(cpu: &mut Cpu, op: AluOp, a: u32, b: u32) -> u32 {
    use flags::*;
    match op {
        AluOp::Add => {
            let (r, c) = a.overflowing_add(b);
            set_logic_flags32(cpu, r);
            set_add_carry32(cpu, a, b, r, c);
            r
        }
        AluOp::Adc => {
            let cin = cpu.get_flag(CF) as u64;
            let total = a as u64 + b as u64 + cin;
            let r = total as u32;
            let c = total > 0xFFFF_FFFF;
            set_logic_flags32(cpu, r);
            set_add_carry32(cpu, a, b, r, c);
            r
        }
        AluOp::Sub | AluOp::Cmp => {
            let (r, c) = a.overflowing_sub(b);
            set_logic_flags32(cpu, r);
            set_sub_borrow32(cpu, a, b, r, c);
            r
        }
        AluOp::Sbb => {
            let cin = cpu.get_flag(CF) as u64;
            let total = (a as u64).wrapping_sub(b as u64).wrapping_sub(cin);
            let r = total as u32;
            let c = (a as u64) < (b as u64 + cin);
            set_logic_flags32(cpu, r);
            set_sub_borrow32(cpu, a, b, r, c);
            r
        }
        AluOp::And => { let r = a & b; set_logic_flags32(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
        AluOp::Or  => { let r = a | b; set_logic_flags32(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
        AluOp::Xor => { let r = a ^ b; set_logic_flags32(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
    }
}

/// Store a two/three-operand IMUL result into a 16-bit register. CF and OF
/// are set when the full signed product did not fit in 16 bits; SF/ZF/PF are
/// architecturally undefined but we set them from the truncated result, which
/// is what real CPUs do.
fn alu64(cpu: &mut Cpu, op: AluOp, a: u64, b: u64) -> u64 {
    use flags::*;
    match op {
        AluOp::Add => {
            let (r, c) = a.overflowing_add(b);
            set_logic_flags64(cpu, r);
            set_add_carry64(cpu, a, b, r, c);
            r
        }
        AluOp::Adc => {
            let cin = cpu.get_flag(CF) as u64;
            let r = a.wrapping_add(b).wrapping_add(cin);
            // Carry out of a 64-bit add cannot be found by widening -- there
            // is nothing wider to widen into -- so it is read off the result:
            // the sum wrapped if it came out below either input, and the
            // equality case is the one where the carry-in did it alone.
            let c = r < a || (cin == 1 && r == a);
            set_logic_flags64(cpu, r);
            set_add_carry64(cpu, a, b, r, c);
            r
        }
        AluOp::Sub | AluOp::Cmp => {
            let (r, c) = a.overflowing_sub(b);
            set_logic_flags64(cpu, r);
            set_sub_borrow64(cpu, a, b, r, c);
            r
        }
        AluOp::Sbb => {
            let cin = cpu.get_flag(CF) as u64;
            let r = a.wrapping_sub(b).wrapping_sub(cin);
            // Borrow iff a < b + cin, written so that b + cin cannot itself
            // overflow out of the comparison.
            let c = a < b || (a == b && cin == 1);
            set_logic_flags64(cpu, r);
            set_sub_borrow64(cpu, a, b, r, c);
            r
        }
        AluOp::And => { let r = a & b; set_logic_flags64(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
        AluOp::Or  => { let r = a | b; set_logic_flags64(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
        AluOp::Xor => { let r = a ^ b; set_logic_flags64(cpu, r); cpu.set_flag(CF, false); cpu.set_flag(OF, false); r }
    }
}

/// An ALU operation at whatever width the instruction is running: 8, 16, 32
/// or 64 bits.
///
/// Each width has its own implementation rather than one masked one, because
/// the flags are where the subtleties live and a masked version gets the
/// carry out of the widest case wrong -- there is nothing wider than u64 to
/// widen into. The dispatch is what the width-generic execute arms call.
fn alu_w(cpu: &mut Cpu, op: AluOp, a: u64, b: u64, width: u32) -> u64 {
    match width {
        32 => alu32(cpu, op, a as u32, b as u32) as u64,
        64 => alu64(cpu, op, a, b),
        16 => alu16(cpu, op, a as u16, b as u16) as u64,
        _ => alu8(cpu, op, a as u8, b as u8) as u64,
    }
}

fn imul_store16(cpu: &mut Cpu, dst: u8, full: i32) {
    use flags::*;
    let r = full as u16;
    cpu.set_reg16_idx(dst, r);
    let overflow = full != (r as i16) as i32;
    cpu.set_flag(CF, overflow);
    cpu.set_flag(OF, overflow);
    set_logic_flags16(cpu, r);
}

/// 32-bit counterpart of `imul_store16`.
fn imul_store32(cpu: &mut Cpu, dst: u8, full: i64) {
    use flags::*;
    let r = full as u32;
    cpu.set_reg32_idx(dst, r);
    let overflow = full != (r as i32) as i64;
    cpu.set_flag(CF, overflow);
    cpu.set_flag(OF, overflow);
    set_logic_flags32(cpu, r);
}

/// Flags after a 16-bit double-precision shift (SHLD/SHRD).
fn set_shift_flags16(cpu: &mut Cpu, r: u16, cf: bool, of: bool) {
    use flags::*;
    cpu.set_flag(CF, cf);
    cpu.set_flag(OF, of);
    set_logic_flags16(cpu, r);
}

/// Flags after a 32-bit double-precision shift (SHLD/SHRD).
fn set_shift_flags32(cpu: &mut Cpu, r: u32, cf: bool, of: bool) {
    use flags::*;
    cpu.set_flag(CF, cf);
    cpu.set_flag(OF, of);
    set_logic_flags32(cpu, r);
}

fn set_logic_flags8(cpu: &mut Cpu, r: u8) {
    use flags::*;
    cpu.set_flag(SF, (r as i8) < 0);
    cpu.set_flag(ZF, r == 0);
    cpu.set_flag(PF, parity(r));
}

fn set_logic_flags16(cpu: &mut Cpu, r: u16) {
    use flags::*;
    cpu.set_flag(SF, (r as i16) < 0);
    cpu.set_flag(ZF, r == 0);
    cpu.set_flag(PF, parity(r as u8));
}

fn set_logic_flags32(cpu: &mut Cpu, r: u32) {
    use flags::*;
    cpu.set_flag(SF, (r as i32) < 0);
    cpu.set_flag(ZF, r == 0);
    cpu.set_flag(PF, parity(r as u8));
}

fn set_logic_flags64(cpu: &mut Cpu, r: u64) {
    use flags::*;
    cpu.set_flag(SF, (r as i64) < 0);
    cpu.set_flag(ZF, r == 0);
    cpu.set_flag(PF, parity(r as u8));
}

fn set_add_carry64(cpu: &mut Cpu, a: u64, b: u64, r: u64, c: bool) {
    use flags::*;
    cpu.set_flag(CF, c);
    cpu.set_flag(AF, ((a ^ b ^ r) & 0x10) != 0);
    let of = ((a ^ r) & (b ^ r)) & 0x8000_0000_0000_0000 != 0;
    cpu.set_flag(OF, of);
}

fn set_sub_borrow64(cpu: &mut Cpu, a: u64, b: u64, r: u64, c: bool) {
    use flags::*;
    cpu.set_flag(CF, c);
    cpu.set_flag(AF, ((a ^ b ^ r) & 0x10) != 0);
    let of = ((a ^ b) & (a ^ r)) & 0x8000_0000_0000_0000 != 0;
    cpu.set_flag(OF, of);
}

/// Flags after a 64-bit double-precision shift (SHLD/SHRD).
fn set_shift_flags64(cpu: &mut Cpu, r: u64, cf: bool, of: bool) {
    use flags::*;
    cpu.set_flag(CF, cf);
    cpu.set_flag(OF, of);
    set_logic_flags64(cpu, r);
}

/// 64-bit counterpart of `imul_store16`.
fn imul_store64(cpu: &mut Cpu, dst: u8, full: i128) {
    use flags::*;
    let r = full as u64;
    cpu.set_reg_w(dst, 64, r);
    let overflow = full != (r as i64) as i128;
    cpu.set_flag(CF, overflow);
    cpu.set_flag(OF, overflow);
    set_logic_flags64(cpu, r);
}

fn set_add_carry(cpu: &mut Cpu, a: u16, b: u16, r: u16, c: bool) {
    use flags::*;
    cpu.set_flag(CF, c);
    cpu.set_flag(AF, ((a ^ b ^ r) & 0x10) != 0);
    let of = ((a ^ r) & (b ^ r)) & 0x8000 != 0;
    cpu.set_flag(OF, of);
}

fn set_sub_borrow(cpu: &mut Cpu, a: u16, b: u16, r: u16, c: bool) {
    use flags::*;
    cpu.set_flag(CF, c);
    cpu.set_flag(AF, ((a ^ b ^ r) & 0x10) != 0);
    let of = ((a ^ b) & (a ^ r)) & 0x8000 != 0;
    cpu.set_flag(OF, of);
}

fn set_add_carry32(cpu: &mut Cpu, a: u32, b: u32, r: u32, c: bool) {
    use flags::*;
    cpu.set_flag(CF, c);
    cpu.set_flag(AF, ((a ^ b ^ r) & 0x10) != 0);
    let of = ((a ^ r) & (b ^ r)) & 0x8000_0000 != 0;
    cpu.set_flag(OF, of);
}

fn set_sub_borrow32(cpu: &mut Cpu, a: u32, b: u32, r: u32, c: bool) {
    use flags::*;
    cpu.set_flag(CF, c);
    cpu.set_flag(AF, ((a ^ b ^ r) & 0x10) != 0);
    let of = ((a ^ b) & (a ^ r)) & 0x8000_0000 != 0;
    cpu.set_flag(OF, of);
}

fn parity(v: u8) -> bool {
    let mut v = v & 0xFF;
    v ^= v >> 4;
    v ^= v >> 2;
    v ^= v >> 1;
    (v & 1) == 0
}

/// Load a far pointer (offset + segment) from a memory operand into the
/// `reg` field's register and the given segment register. Used by
/// LDS/LES/LSS/LFS/LGS.
fn load_far_pointer(cpu: &mut Cpu, m: &ModRm, seg: SegReg) {
    let addr = if cpu.addrsize {
        cpu.modrm_addr32(m)
    } else {
        cpu.modrm_addr(m)
    };
    if cpu.opsize {
        // 32-bit offset + 16-bit segment.
        let off = cpu.mem.read_u32(addr);
        let sel = cpu.mem.read_u16(addr + 4);
        cpu.set_reg32_idx(m.reg, off);
        cpu.load_seg(seg, sel);
    } else {
        // 16-bit offset + 16-bit segment.
        let off = cpu.mem.read_u16(addr);
        let sel = cpu.mem.read_u16(addr + 2);
        cpu.set_reg16_idx(m.reg, off);
        cpu.load_seg(seg, sel);
    }
}

/// Compute the effective address (offset only, no segment) of a ModR/M
/// memory operand, for LEA. This is the same computation the addressing path
/// does, minus the segment and the page tables -- including RIP-relative,
/// which is how 64-bit code takes the address of its own data.
fn lea_offset(m: &ModRm, cpu: &Cpu) -> u64 {
    if cpu.addrsize {
        cpu.modrm_ea32(m).0
    } else {
        cpu.modrm_offset(m) as u64
    }
}

/// Perform an 8-bit shift/rotate, setting flags, and return the result.
/// Perform a shift/rotate of `width` bits (8, 16 or 32), set the flags, and
/// return the result in the low `width` bits.
///
/// One implementation for all three widths: doing it per-width is how the
/// 32-bit case came to be missing entirely, with `D1`/`D3` in 32-bit mode
/// silently shifting only the low byte.
///
/// Rules worth stating, because they are easy to get subtly wrong:
/// - The count is masked to 5 bits on 386+ for every operand size **except
///   64-bit, where it is masked to 6** -- otherwise `shl $32,%rax` would be a
///   no-op instead of clearing the low half. Rust's `wrapping_shl` masks the
///   count to the type's width, which is *not* the same thing as either — so
///   the shift is done wide and truncated.
/// - A count of 0 changes nothing, flags included.
/// - RCL/RCR rotate through a `width + 1`-bit quantity (the operand plus CF),
///   so their effective count is taken modulo `width + 1` for 8- and 16-bit
///   operands. For 32-bit the 5-bit mask already keeps it in range.
/// - OF is architecturally defined only for a count of 1; we leave it clear
///   otherwise, which is what the shape `n == 1 && ..` below expresses.
fn shift_width(cpu: &mut Cpu, op: ShiftOp, v: u64, n: u32, width: u32) -> u64 {
    use flags::*;
    let n = n & if width == 64 { 0x3F } else { 0x1F };
    if n == 0 { return v; }
    let mask: u64 = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    let msb: u64 = 1u64 << (width - 1);
    let v = v & mask;
    let v64 = v;

    match op {
        ShiftOp::Shl => {
            let wide = v64 << n.min(63);
            let r = wide & mask;
            let cf = n <= width && (v64 >> (width - n.min(width))) & 1 != 0;
            cpu.set_flag(CF, cf);
            set_logic_flags_width(cpu, r, width);
            // OF (count 1) = MSB of the result XOR the new CF.
            cpu.set_flag(OF, n == 1 && (((r & msb) != 0) != cf));
            r
        }
        ShiftOp::Shr => {
            let r = if n >= width { 0 } else { v64 >> n };
            let cf = n <= width && (v64 >> (n - 1)) & 1 != 0;
            cpu.set_flag(CF, cf);
            set_logic_flags_width(cpu, r, width);
            // OF (count 1) = MSB of the *original* operand.
            cpu.set_flag(OF, n == 1 && (v & msb) != 0);
            r
        }
        ShiftOp::Sar => {
            // Sign-extend to i64, shift, truncate. A count at or past the
            // width saturates to all-sign-bits, which the min() gives us.
            let sv = sign_extend(v, width);
            let r = (sv >> n.min(width - 1)) as u64 & mask;
            let cf = ((sv >> (n - 1).min(width - 1)) & 1) != 0;
            cpu.set_flag(CF, cf);
            set_logic_flags_width(cpu, r, width);
            cpu.set_flag(OF, false);
            r
        }
        ShiftOp::Rol => {
            let k = n % width;
            let r = if k == 0 { v } else { ((v64 << k) | (v64 >> (width - k))) & mask };
            let cf = r & 1 != 0;
            cpu.set_flag(CF, cf);
            cpu.set_flag(OF, n == 1 && (((r & msb) != 0) != cf));
            r
        }
        ShiftOp::Ror => {
            let k = n % width;
            let r = if k == 0 { v } else { ((v64 >> k) | (v64 << (width - k))) & mask };
            let cf = r & msb != 0;
            cpu.set_flag(CF, cf);
            // OF (count 1) = XOR of the two most significant result bits.
            let second = (r >> (width - 2)) & 1 != 0;
            cpu.set_flag(OF, n == 1 && (cf != second));
            r
        }
        ShiftOp::Rcl | ShiftOp::Rcr => {
            // Rotate through carry: a (width + 1)-bit quantity. At 64
            // bits that quantity is 65 wide and does not fit in a u64, so
            // the rotate is done with an explicit carry bit alongside.
            let bits = width + 1;
            let k = n % bits;
            let carry = cpu.get_flag(CF) as u64;
            if width == 64 {
                let (r, cf) = rcl_rcr64(v64, carry, k, op == ShiftOp::Rcl);
                cpu.set_flag(CF, cf);
                if op == ShiftOp::Rcl {
                    cpu.set_flag(OF, n == 1 && (((r & msb) != 0) != cf));
                } else {
                    let second = (r >> (width - 2)) & 1 != 0;
                    cpu.set_flag(OF, n == 1 && (((r & msb) != 0) != second));
                }
                return r;
            }
            let wide = v64 | (carry << width);
            let full: u64 = (1u64 << bits) - 1;
            let rot = if k == 0 {
                wide
            } else if op == ShiftOp::Rcl {
                ((wide << k) | (wide >> (bits - k))) & full
            } else {
                ((wide >> k) | (wide << (bits - k))) & full
            };
            let r = rot & mask;
            let cf = (rot >> width) & 1 != 0;
            cpu.set_flag(CF, cf);
            if op == ShiftOp::Rcl {
                cpu.set_flag(OF, n == 1 && (((r & msb) != 0) != cf));
            } else {
                let second = (r >> (width - 2)) & 1 != 0;
                cpu.set_flag(OF, n == 1 && (((r & msb) != 0) != second));
            }
            // Rotates leave SF/ZF/PF alone.
            r
        }
    }
}

/// Rotate a 65-bit quantity (a 64-bit operand plus the carry flag) left or
/// right by `k`, returning the operand and the new carry.
///
/// It gets its own function because 65 bits do not fit anywhere: the trick
/// the narrower widths use -- park the carry in bit `width` of a u64 -- has
/// nowhere to park it here.
fn rcl_rcr64(v: u64, carry: u64, k: u32, left: bool) -> (u64, bool) {
    let mut val = v;
    let mut c = carry != 0;
    if left {
        for _ in 0..k {
            let out = val >> 63 != 0;
            val = (val << 1) | (c as u64);
            c = out;
        }
    } else {
        for _ in 0..k {
            let out = val & 1 != 0;
            val = (val >> 1) | ((c as u64) << 63);
            c = out;
        }
    }
    (val, c)
}

/// Sign-extend the low `width` bits of `v` to i64.
fn sign_extend(v: u64, width: u32) -> i64 {
    match width {
        8 => v as u8 as i8 as i64,
        16 => v as u16 as i16 as i64,
        32 => v as u32 as i32 as i64,
        _ => v as i64,
    }
}

/// SF/ZF/PF for a result of the given width.
fn set_logic_flags_width(cpu: &mut Cpu, r: u64, width: u32) {
    match width {
        8 => set_logic_flags8(cpu, r as u8),
        16 => set_logic_flags16(cpu, r as u16),
        32 => set_logic_flags32(cpu, r as u32),
        _ => set_logic_flags64(cpu, r),
    }
}

/// What a bit-test instruction does to the bit it selects.
#[derive(Clone, Copy, PartialEq)]
enum BitOp { Test, Set, Reset, Complement }

/// BT/BTS/BTR/BTC. CF takes the bit's old value; the other flags are
/// architecturally undefined and left alone.
///
/// The addressing rule differs by operand kind, and the difference matters:
/// with a **register** destination the offset wraps within the operand
/// (mod 16 or 32), but with a **memory** destination it is a signed bit index
/// into an array that may run far past the addressed word -- which is exactly
/// how the kernel's bitmaps (`test_bit`, `set_bit`) are addressed.
fn bit_op(cpu: &mut Cpu, m: &ModRm, bit: BitOffset, op: BitOp) {
    let w = cpu.osize();
    let width = w as i64;
    let offset: i64 = match bit {
        BitOffset::Imm(i) => i as i64,
        BitOffset::Reg(r) => sign_extend(cpu.reg_w(r, w), w),
    };

    if m.is_reg() {
        // Register destination: the offset is taken modulo the width.
        let b = offset.rem_euclid(width) as u32;
        let v = cpu.reg_w(m.rm, w);
        let (old, new) = (v >> b & 1 != 0, apply_bit(v, b, op));
        cpu.set_flag(flags::CF, old);
        if op != BitOp::Test {
            cpu.set_reg_w(m.rm, w, new);
        }
        return;
    }

    // Memory destination: step whole operands along the bit string. The
    // displacement is signed, so a negative offset reaches backwards.
    let bytes = width / 8;
    let word = offset.div_euclid(width);
    let b = offset.rem_euclid(width) as u32;
    let disp = (word * bytes) as i64;
    let base = cpu.rm_addr(m, op != BitOp::Test);
    let addr = base.wrapping_add(disp as isize as usize);

    let v = match w {
        64 => cpu.mem.read_u64(addr),
        32 => cpu.mem.read_u32(addr) as u64,
        _ => cpu.mem.read_u16(addr) as u64,
    };
    let (old, new) = (v >> b & 1 != 0, apply_bit(v, b, op));
    cpu.set_flag(flags::CF, old);
    if op != BitOp::Test {
        match w {
            64 => cpu.mem.write_u64(addr, new),
            32 => cpu.mem.write_u32(addr, new as u32),
            _ => cpu.mem.write_u16(addr, new as u16),
        }
    }
}

fn apply_bit(v: u64, b: u32, op: BitOp) -> u64 {
    match op {
        BitOp::Test => v,
        BitOp::Set => v | (1 << b),
        BitOp::Reset => v & !(1 << b),
        BitOp::Complement => v ^ (1 << b),
    }
}

/// Signed step for a string instruction: forward, or backward when DF is set.
fn string_step(cpu: &Cpu, esize: u32) -> i32 {
    if cpu.get_flag(flags::DF) { -(esize as i32) } else { esize as i32 }
}

/// Advance a string index register, wrapping at the address size.
fn string_advance(v: u64, step: i32, asize: u32) -> u64 {
    let n = v.wrapping_add(step as i64 as u64);
    match asize {
        64 => n,
        32 => n & 0xFFFF_FFFF,
        _ => n & 0xFFFF,
    }
}

fn string_si(cpu: &Cpu, asize: u32) -> u64 { cpu.reg_w(6, asize) }
fn string_di(cpu: &Cpu, asize: u32) -> u64 { cpu.reg_w(7, asize) }
fn string_set_si(cpu: &mut Cpu, asize: u32, v: u64) {
    // `_raw`: this write records *where the fault stopped*, so it has to land
    // even though a fault is pending.
    cpu.set_reg_w_raw(6, asize, v);
}
fn string_set_di(cpu: &mut Cpu, asize: u32, v: u64) {
    cpu.set_reg_w_raw(7, asize, v);
}

/// Iteration count: the count register under a REP prefix, one without.
/// A REP with a zero count does nothing at all, which the `while cnt > 0`
/// loops express directly.
fn string_count(cpu: &Cpu, asize: u32, rep: Rep) -> u64 {
    if rep == Rep::None { 1 } else { cpu.reg_w(1, asize) }
}
fn string_set_count(cpu: &mut Cpu, asize: u32, v: u64) {
    cpu.set_reg_w_raw(1, asize, v);
}

/// Should a REPE/REPNE comparison keep going? REP alone always continues
/// (the count test is the loop condition); the conditional forms also stop
/// on ZF.
fn string_repeat(cpu: &Cpu, rep: Rep, remaining: u64) -> bool {
    match rep {
        Rep::None => false,
        _ if remaining == 0 => false,
        Rep::Repe => cpu.get_flag(flags::ZF),
        Rep::Repne => !cpu.get_flag(flags::ZF),
    }
}

/// Read the r/m operand at `width` bits, shift it, and write it back.
fn do_shift(cpu: &mut Cpu, op: ShiftOp, m: &ModRm, width: u32, n: u32) {
    let v = cpu.read_rm_w(m, width);
    let r = shift_width(cpu, op, v, n, width);
    cpu.write_rm_w(m, width, r);
}


/// Perform a 16-bit shift/rotate, setting flags, and return the result.

/// Perform a 32-bit shift/rotate, setting flags, and return the result.

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::memory::Memory;
    use crate::cpu::{Cpu, flags};
    use crate::protected::Descriptor;

    fn load(cpu: &mut Cpu, bytes: &[u8]) {
        cpu.mem.load(Memory::phys(cpu.cs, cpu.ip), bytes);
    }

    #[test]
    fn mov_reg16_imm_and_add() {
        let mut cpu = Cpu::new();
        load(&mut cpu, &[
            0xB8, 0x34, 0x12,
            0xBB, 0x02, 0x00,
            0x01, 0xD8,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x1236);
        assert!(cpu.halted);
    }

    #[test]
    fn sub_sets_borrow() {
        let mut cpu = Cpu::new();
        load(&mut cpu, &[
            0xB8, 0x01, 0x00,
            0xBB, 0x02, 0x00,
            0x29, 0xD8,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0xFFFF);
        assert!(cpu.get_flag(flags::CF));
        assert!(cpu.get_flag(flags::SF));
    }

    #[test]
    fn conditional_jump_taken() {
        let mut cpu = Cpu::new();
        load(&mut cpu, &[
            0xB8, 0x05, 0x00,
            0x3D, 0x05, 0x00,
            0x74, 0x01,
            0xF4,
            0x90,
            0xF4,
        ]);
        cpu.run(32);
        assert!(cpu.halted);
        assert_eq!(cpu.instructions_executed, 5);
    }

    #[test]
    fn call_ret_roundtrip() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
        load(&mut cpu, &[
            0xE8, 0x01, 0x00,
            0xF4,
            0xB8, 0x99, 0x00,
            0xC3,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x0099);
        assert!(cpu.halted);
    }

    #[test]
    fn memory_mov_roundtrip() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        load(&mut cpu, &[
            0xC7, 0x06, 0x00, 0x01, 0x42, 0x42,
            0xA1, 0x00, 0x01,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x4242);
    }

    #[test]
    fn xor_clears_reg() {
        let mut cpu = Cpu::new();
        load(&mut cpu, &[
            0xB8, 0xFF, 0xFF,
            0x31, 0xC0,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0);
        assert!(cpu.get_flag(flags::ZF));
        assert!(!cpu.get_flag(flags::CF));
    }

    #[test]
    fn inc_does_not_touch_carry() {
        let mut cpu = Cpu::new();
        cpu.set_flag(flags::CF, true);
        load(&mut cpu, &[
            0xB8, 0xFF, 0xFF,
            0x40,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x0000);
        assert!(cpu.get_flag(flags::ZF));
        assert!(cpu.get_flag(flags::CF));
    }

    #[test]
    fn int_iret_through_ivt() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
        cpu.mem.write_u16(0x84, 0x0100);
        cpu.mem.write_u16(0x86, 0x0000);
        cpu.mem.load(0x100, &[
            0xB8, 0x99, 0x00,
            0xCF,
        ]);
        load(&mut cpu, &[
            0xCD, 0x21,
            0xF4,
        ]);
        cpu.run(32);
        assert_eq!(cpu.ax(), 0x0099);
        assert!(cpu.halted);
        assert_eq!(cpu.sp(), 0x0100);
    }

    #[test]
    fn pushf_popf_roundtrip() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
        cpu.set_flag(flags::CF, true);
        cpu.set_flag(flags::ZF, true);
        load(&mut cpu, &[
            0x9C,
            0x58,
            0xF4,
        ]);
        cpu.run(16);
        assert!(cpu.ax() as u32 & flags::CF != 0);
        assert!(cpu.ax() as u32 & flags::ZF != 0);
    }

    #[test]
    fn shift_left_sets_carry() {
        let mut cpu = Cpu::new();
        load(&mut cpu, &[
            0xB8, 0x00, 0x80,
            0xD1, 0xE0,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x0000);
        assert!(cpu.get_flag(flags::CF));
        assert!(cpu.get_flag(flags::ZF));
    }

    #[test]
    fn shift_imm8_with_count() {
        let mut cpu = Cpu::new();
        // mov ax, 0x8000 ; shr ax, 2 (0xC1 E8 02)
        load(&mut cpu, &[
            0xB8, 0x00, 0x80,
            0xC1, 0xE8, 0x02,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x2000);
    }

    #[test]
    fn flag_control_instructions() {
        let mut cpu = Cpu::new();
        // stc ; clc ; std ; cld ; sti ; cli ; cmc
        load(&mut cpu, &[
            0xF9, // stc
            0xF8, // clc
            0xFD, // std
            0xFC, // cld
            0xFB, // sti
            0xFA, // cli
            0xF5, // cmc
            0xF4,
        ]);
        cpu.run(16);
        // After stc, clc: CF=0. After cmc: CF=1.
        assert!(cpu.get_flag(flags::CF));
        assert!(!cpu.get_flag(flags::DF));
        assert!(!cpu.get_flag(flags::IF));
    }

    #[test]
    fn jmp_rm32_indirect() {
        let mut cpu = Cpu::new();
        // mov ax, 0x1000 ; jmp ax (FF E0, 16-bit in real mode) ; hlt
        load(&mut cpu, &[
            0xB8, 0x00, 0x10,
            0xFF, 0xE0,
            0xF4,
        ]);
        cpu.mem.load(0x1000, &[0xF4]);
        cpu.run(16);
        assert!(cpu.halted);
        assert_eq!(cpu.ip, 0x1001); // jmp to 0x1000, hlt advanced past it
    }

    #[test]
    fn inc_rm16_via_ff() {
        let mut cpu = Cpu::new();
        // mov ax, 0x0001 ; inc word [bx] via FF /0 with rm=reg
        // FF C0 = inc eax (32-bit default? no — 16-bit mode here)
        load(&mut cpu, &[
            0xB8, 0x01, 0x00,
            0xFF, 0xC0,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x0002);
    }

    #[test]
    fn mul16_uses_dx_ax() {
        let mut cpu = Cpu::new();
        load(&mut cpu, &[
            0xB8, 0x00, 0x10,
            0xBB, 0x10, 0x00,
            0xF7, 0xE3,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x0000);
        assert_eq!(cpu.dx(), 0x0001);
        assert!(cpu.get_flag(flags::CF));
    }

    #[test]
    fn div16_quotient_remainder() {
        let mut cpu = Cpu::new();
        load(&mut cpu, &[
            0xB8, 0x13, 0x00,
            0xBB, 0x05, 0x00,
            0xF7, 0xF3,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x0003);
        assert_eq!(cpu.dx(), 0x0004);
    }

    #[test]
    fn rep_movsb_copies_string() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        cpu.es = 0;
        cpu.set_si(0x0100);
        cpu.set_di(0x0200);
        cpu.set_cx(3);
        cpu.mem.write_u8(0x100, 0x41);
        cpu.mem.write_u8(0x101, 0x42);
        cpu.mem.write_u8(0x102, 0x43);
        load(&mut cpu, &[
            0xF3, 0xA4,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.mem.read_u8(0x200), 0x41);
        assert_eq!(cpu.mem.read_u8(0x201), 0x42);
        assert_eq!(cpu.mem.read_u8(0x202), 0x43);
        assert_eq!(cpu.cx(), 0);
        assert_eq!(cpu.si(), 0x0103);
        assert_eq!(cpu.di(), 0x0203);
    }

    #[test]
    fn loop_decrements_cx() {
        let mut cpu = Cpu::new();
        cpu.set_cx(3);
        load(&mut cpu, &[
            0xE2, 0xFE,
            0xF4,
        ]);
        cpu.run(64);
        assert!(cpu.halted);
        assert_eq!(cpu.cx(), 0);
    }

    #[test]
    fn lea_loads_effective_address() {
        let mut cpu = Cpu::new();
        cpu.set_bx(0x0100);
        cpu.set_si(0x0020);
        load(&mut cpu, &[
            0x8D, 0x00,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x0120);
    }

    #[test]
    fn cwd_sign_extends_ax() {
        let mut cpu = Cpu::new();
        load(&mut cpu, &[
            0xB8, 0x00, 0x80,
            0x99,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.dx(), 0xFFFF);
    }

    #[test]
    fn neg_sets_carry() {
        let mut cpu = Cpu::new();
        load(&mut cpu, &[
            0xB8, 0x05, 0x00,
            0xF7, 0xD8,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0xFFFB);
        assert!(cpu.get_flag(flags::CF));
    }

    // ---- 32-bit protected-mode tests ----

    #[test]
    fn opsize_prefix_32bit_mov() {
        let mut cpu = Cpu::new();
        // 66 B8 imm32 -> mov eax, 0x12345678
        load(&mut cpu, &[
            0x66, 0xB8, 0x78, 0x56, 0x34, 0x12,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.eax(), 0x12345678);
        assert_eq!(cpu.ax(), 0x5678);
    }

    #[test]
    fn opsize_prefix_32bit_add() {
        let mut cpu = Cpu::new();
        // 66 B8 imm32 ; 66 05 imm32 -> mov eax,1 ; add eax,2
        load(&mut cpu, &[
            0x66, 0xB8, 0x01, 0x00, 0x00, 0x00,
            0x66, 0x05, 0x02, 0x00, 0x00, 0x00,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.eax(), 3);
    }

    #[test]
    fn opsize_prefix_32bit_mul() {
        let mut cpu = Cpu::new();
        // 66 B8 0x10000000 ; 66 BB 0x10 ; 66 F7 E3 -> mul ebx
        load(&mut cpu, &[
            0x66, 0xB8, 0x00, 0x00, 0x00, 0x10,
            0x66, 0xBB, 0x10, 0x00, 0x00, 0x00,
            0x66, 0xF7, 0xE3,
            0xF4,
        ]);
        cpu.run(16);
        // 0x10000000 * 0x10 = 0x100000000 -> EAX=0, EDX=1
        assert_eq!(cpu.eax(), 0);
        assert_eq!(cpu.edx(), 1);
    }

    #[test]
    fn addrsize_prefix_32bit_addressing() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        // 67 8B 04 85 disp32 -> mov eax, [eax*4 + disp32]
        // modrm = 00 000 100 (reg=EAX, rm=100 -> SIB follows)
        // sib = 10 000 101 (scale=4, index=EAX, base=101 -> disp32)
        // disp32 = 0x1000
        cpu.set_eax(0x10);
        cpu.mem.write_u32(0x1000 + 0x10 * 4, 0xDEADBEEF);
        load(&mut cpu, &[
            0x66, 0x67, 0x8B, 0x04, 0x85, 0x00, 0x10, 0x00, 0x00,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.eax(), 0xDEADBEEF);
    }

    #[test]
    fn lgdt_loads_gdt() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        // GDT at physical 0x2000: limit=0x17, base=0x3000
        cpu.mem.write_u16(0x1000, 0x17);
        cpu.mem.write_u32(0x1002, 0x3000);
        // lgdt [0x1000] = 0F 01 16 disp16
        load(&mut cpu, &[
            0x0F, 0x01, 0x16, 0x00, 0x10,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.gdt_limit, 0x17);
        assert_eq!(cpu.gdt_base, 0x3000);
    }

    #[test]
    fn protected_mode_segment_translation() {
        let mut cpu = Cpu::new();
        // Build a GDT at 0x2000 with a data descriptor at index 1:
        // base=0x10000, limit=0xFFFFF, granularity, present, data.
        // Descriptor bytes (little-endian u64):
        //   bits 0-15   limit 15:0 = 0xFFFF
        //   bits 16-31  base 15:0  = 0x0000
        //   bits 32-39  base 23:16 = 0x01
        //   bits 40-47  type/attr  = 0x92 (present, data read/write)
        //   bits 48-55  limit 19:16 = 0xF, G=1, D=1
        //   bits 56-63  base 31:24 = 0x00
        let desc: u64 = 0x00CF_9201_0000_FFFF;
        cpu.mem.write_u64(0x2008, desc);
        cpu.gdt_base = 0x2000;
        cpu.gdt_limit = 0x17;
        cpu.pe = true;
        // Load DS with selector 0x08 (index 1, RPL 0).
        cpu.load_seg(SegReg::Ds, 0x08);
        // DS base should be 0x10000.
        assert_eq!(cpu.seg_desc[SegReg::Ds as usize].base, 0x10000);
        // Translate DS:0x1234 -> 0x11234.
        assert_eq!(cpu.translate(SegReg::Ds, (0x1234) as u64), 0x11234);
    }

    #[test]
    fn protected_mode_int_through_idt() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.ss = 0;
        cpu.set_esp(0x0100);
        // IDT at 0x3000, vector 0x20 entry at 0x3000 + 0x20*8 = 0x3100.
        // offset_lo = 0x5000, selector = 0x08, offset_hi = 0x0000.
        cpu.idt_base = 0x3000;
        cpu.idt_limit = 0xFF;
        let entry = 0x3000 + 0x20 * 8;
        cpu.mem.write_u16(entry, 0x5000);
        cpu.mem.write_u16(entry + 2, 0x08);
        cpu.mem.write_u16(entry + 6, 0x0000);
        // Handler at 0x5000: mov eax, 0x99 ; iret (32-bit)
        cpu.mem.load(0x5000, &[
            0x66, 0xB8, 0x99, 0x00, 0x00, 0x00,
            0x66, 0xCF,
        ]);
        // Main: int 0x20 ; hlt
        cpu.mem.load(0x1000, &[
            0xCD, 0x20,
            0xF4,
        ]);
        cpu.cs = 0x08;
        cpu.set_eip(0x1000);
        cpu.run(32);
        assert_eq!(cpu.eax(), 0x99);
        assert!(cpu.halted);
        assert_eq!(cpu.esp(), 0x0100);
    }

    #[test]
    fn loop_uses_eip_in_32bit_mode() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        // Flat 32-bit code segment (D=1) so opsize defaults to 32-bit.
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false,
        };
        cpu.set_eip(0x1000);
        // mov ecx, 3 ; loop $ (E2 FE) ; hlt
        // (no 0x66 prefix: opsize already 32-bit in this segment)
        cpu.mem.load(0x1000, &[
            0xB9, 0x03, 0x00, 0x00, 0x00, // mov ecx, 3
            0xE2, 0xFE,                    // loop -2 (back to itself)
            0xF4,                          // hlt
        ]);
        cpu.run(64);
        assert!(cpu.halted);
        assert_eq!(cpu.ecx(), 0);
        assert_eq!(cpu.eip(), 0x1008);
    }

    #[test]
    fn jcc_uses_eip_in_32bit_mode() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false,
        };
        cpu.set_eip(0x1000);
        // mov eax, 1 ; test eax, eax ; jz +1 (74 01) ; hlt ; hlt
        cpu.mem.load(0x1000, &[
            0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
            0x85, 0xC0,                    // test eax, eax
            0x74, 0x01,                    // jz +1 (not taken, ZF=0)
            0xF4,                          // hlt
            0xF4,                          // hlt
        ]);
        cpu.run(32);
        assert!(cpu.halted);
        // mov(5) + test(2) + jz(2) = 0x1009, then hlt advances to 0x100A.
        assert_eq!(cpu.eip(), 0x100A);
    }

    #[test]
    fn jmp_rel8_uses_eip_in_32bit_mode() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false,
        };
        cpu.set_eip(0x1000);
        // jmp +1 (EB 01) over a hlt, landing on a second hlt.
        cpu.mem.load(0x1000, &[
            0xEB, 0x01, // jmp +1
            0xF4,       // hlt (skipped)
            0xF4,       // hlt (land here)
        ]);
        cpu.run(32);
        assert!(cpu.halted);
        assert_eq!(cpu.eip(), 0x1004);
    }

    #[test]
    fn lss_loads_ss_and_offset() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false,
        };
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true, l: false,
        };
        // Far pointer at 0x2000: offset=0x8000, selector=0x10.
        cpu.mem.write_u32(0x2000, 0x8000);
        cpu.mem.write_u16(0x2004, 0x10);
        // lss eax, [0x2000] = 0F B2 05 disp32
        cpu.set_eip(0x1000);
        cpu.mem.load(0x1000, &[
            0x0F, 0xB2, 0x05, 0x00, 0x20, 0x00, 0x00, // lss eax, [0x2000]
            0xF4,
        ]);
        cpu.run(32);
        assert_eq!(cpu.eax(), 0x8000);
        assert_eq!(cpu.ss, 0x10);
        assert!(cpu.halted);
    }

    #[test]
    fn rep_stosd_uses_ecx_in_32bit_mode() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false,
        };
        cpu.seg_desc[SegReg::Es as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true, l: false,
        };
        cpu.set_ecx(4);
        cpu.set_edi(0x3000);
        cpu.set_eax(0xDEADBEEF);
        // rep stosd (F3 AB) ; hlt
        cpu.set_eip(0x1000);
        cpu.mem.load(0x1000, &[
            0xF3, 0xAB,
            0xF4,
        ]);
        cpu.run(32);
        // 4 dwords written at 0x3000..0x3010.
        for i in 0..4 {
            assert_eq!(cpu.mem.read_u32(0x3000 + i * 4), 0xDEADBEEF);
        }
        assert_eq!(cpu.ecx(), 0);
        assert_eq!(cpu.edi(), 0x3010);
        assert!(cpu.halted);
    }

    // ---- x87 FPU tests ----

    #[test]
    fn fninit_resets_fpu() {
        let mut cpu = Cpu::new();
        cpu.fpu.push(3.5);
        cpu.fpu.control = 0x0000;
        // fninit (DB E3) ; hlt
        load(&mut cpu, &[
            0xDB, 0xE3,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.fpu.control, 0x037F);
        assert_eq!(cpu.fpu.tag, 0xFFFF); // all empty
    }

    #[test]
    fn fstsw_ax_reports_status() {
        let mut cpu = Cpu::new();
        cpu.fpu.status = 0x3800;
        // fnstsw ax (DF E0) ; hlt
        load(&mut cpu, &[
            0xDF, 0xE0,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax(), 0x3800);
    }

    #[test]
    fn fld_push_and_fstp_store() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        // fld qword [0x1000] (DD 06 disp16) ; fstp qword [0x2000] (DD 1E) ; hlt
        cpu.mem.write_f64(0x1000, 3.14159);
        load(&mut cpu, &[
            0xDD, 0x06, 0x00, 0x10, // fld qword [0x1000]
            0xDD, 0x1E, 0x00, 0x20, // fstp qword [0x2000]
            0xF4,
        ]);
        cpu.run(16);
        assert!((cpu.mem.read_f64(0x2000) - 3.14159).abs() < 1e-9);
        assert_eq!(cpu.fpu.tag, 0xFFFF); // popped back to empty
    }

    #[test]
    fn fild_fistp_integer_roundtrip() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        // fild dword [0x1000] (DB 06) ; fistp dword [0x2000] (DB 1E) ; hlt
        cpu.mem.write_u32(0x1000, 42);
        load(&mut cpu, &[
            0xDB, 0x06, 0x00, 0x10,
            0xDB, 0x1E, 0x00, 0x20,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.mem.read_u32(0x2000), 42);
    }

    #[test]
    fn fadd_st0_with_memory() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        cpu.fpu.push(1.5);
        cpu.mem.write_f64(0x1000, 2.5);
        // fadd qword [0x1000] (DC 06) ; hlt
        load(&mut cpu, &[
            0xDC, 0x06, 0x00, 0x10,
            0xF4,
        ]);
        cpu.run(16);
        assert!((cpu.fpu.st_i(0) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn fxsave_fxrstor_roundtrip() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        cpu.fpu.push(1.25);
        cpu.fpu.push(2.5);
        // fxsave [0x3000] (0F AE 06 disp16) ; fxrstor [0x3000] (0F AE 0E) ; hlt
        load(&mut cpu, &[
            0x0F, 0xAE, 0x06, 0x00, 0x30,
            0x0F, 0xAE, 0x0E, 0x00, 0x30,
            0xF4,
        ]);
        cpu.run(16);
        // After fxrstor, the FPU should have the saved registers back.
        // Two pushes: top=6, so st[7] and st[6] are valid (tag bits 7,6 clear).
        assert_eq!(cpu.fpu.tag, 0xFF3F);
        assert!((cpu.fpu.st_i(0) - 2.5).abs() < 1e-9);
    }

    // ---- Paging tests ----

    // ---- CPUID / RDTSC tests ----

    #[test]
    fn cpuid_leaf0_returns_vendor() {
        let mut cpu = Cpu::new();
        // mov eax, 0 ; cpuid ; hlt
        load(&mut cpu, &[
            0x66, 0xB8, 0x00, 0x00, 0x00, 0x00,
            0x0F, 0xA2,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.eax(), 1); // highest basic leaf
        assert_eq!(cpu.ebx(), 0x756E6547); // "Genu"
        assert_eq!(cpu.edx(), 0x49656E69); // "ineI"
        assert_eq!(cpu.ecx(), 0x6C65746E); // "ntel"
    }

    #[test]
    fn cpuid_leaf1_returns_features() {
        let mut cpu = Cpu::new();
        // mov eax, 1 ; cpuid ; hlt
        load(&mut cpu, &[
            0x66, 0xB8, 0x01, 0x00, 0x00, 0x00,
            0x0F, 0xA2,
            0xF4,
        ]);
        cpu.run(16);
        // Family 6, model 0, stepping 0.
        assert_eq!(cpu.eax(), 0x00000600);
        // TSC bit (bit 4) must be set in EDX.
        assert!(cpu.edx() & (1 << 4) != 0);
    }

    #[test]
    fn rdtsc_returns_timestamp() {
        let mut cpu = Cpu::new();
        // Set a known TSC, then rdtsc ; hlt
        cpu.tsc = 0x1234_5678_9ABC_DEF0;
        load(&mut cpu, &[
            0x0F, 0x31,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.eax(), 0x9ABC_DEF0);
        assert_eq!(cpu.edx(), 0x1234_5678);
    }

    #[test]
    fn test_acc_imm_sets_flags() {
        let mut cpu = Cpu::new();
        // mov al, 0x0F ; test al, 0x0F (A8) ; hlt
        load(&mut cpu, &[
            0xB0, 0x0F,
            0xA8, 0x0F,
            0xF4,
        ]);
        cpu.run(16);
        // 0x0F & 0x0F = 0x0F -> ZF clear, CF clear.
        assert!(!cpu.get_flag(flags::ZF));
        assert!(!cpu.get_flag(flags::CF));
    }

    #[test]
    fn rdmsr_returns_zero() {
        let mut cpu = Cpu::new();
        // mov ecx, 0x1B ; rdmsr ; hlt
        load(&mut cpu, &[
            0x66, 0xB9, 0x1B, 0x00, 0x00, 0x00,
            0x0F, 0x32,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.eax(), 0);
        assert_eq!(cpu.edx(), 0);
    }

    #[test]
    fn bts_sets_bit_and_carry() {
        let mut cpu = Cpu::new();
        // mov eax, 0 ; bts eax, 3 (0F BA E8 03, /5) ; hlt
        load(&mut cpu, &[
            0x66, 0xB8, 0x00, 0x00, 0x00, 0x00,
            0x0F, 0xBA, 0xE8, 0x03,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.eax(), 0x8); // bit 3 set
        assert!(!cpu.get_flag(flags::CF)); // was 0 before
    }

    #[test]
    fn mov_moffs32_uses_addrsize() {
        // 0x12345678 is past the default 128 MiB, so this also pins that a
        // machine can be built with enough RAM to reach it -- it used to
        // "work" only because every address was masked back into the store.
        let mut cpu = Cpu::with_ram(320 << 20);
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false,
        };
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true, l: false,
        };
        // mov [0x12345678], al (A2 moffs32) ; hlt
        // In 32-bit addressing mode the moffs is 32-bit.
        cpu.set_eip(0x1000);
        cpu.mem.load(0x1000, &[
            0xA2, 0x78, 0x56, 0x34, 0x12,
            0xF4,
        ]);
        cpu.set_reg8(Reg8::Al, 0xAB);
        cpu.run(32);
        assert_eq!(cpu.mem.read_u8(0x12345678), 0xAB);
        assert!(cpu.halted);
    }

    #[test]
    fn jcc32_branches_via_eip_in_32bit_mode() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false,
        };
        cpu.set_eip(0x1000);
        // mov eax, 1 ; test eax, eax ; jz rel32 (0F 84) not taken ; hlt
        cpu.mem.load(0x1000, &[
            0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
            0x85, 0xC0,                    // test eax, eax
            0x0F, 0x84, 0x01, 0x00, 0x00, 0x00, // jz +1 (not taken)
            0xF4,                          // hlt
            0xF4,
        ]);
        cpu.run(32);
        assert!(cpu.halted);
        // mov(5) + test(2) + jz(6) = 0x100D, then hlt advances to 0x100E.
        assert_eq!(cpu.eip(), 0x100E);
    }

    #[test]
    fn movzx_zero_extends_8bit() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false,
        };
        cpu.set_eip(0x1000);
        // movzx eax, al (0F B6 C0) ; hlt
        cpu.mem.load(0x1000, &[
            0x0F, 0xB6, 0xC0,
            0xF4,
        ]);
        cpu.set_reg8(Reg8::Al, 0xFF);
        cpu.run(32);
        assert_eq!(cpu.eax(), 0xFF);
        assert!(cpu.halted);
    }

    #[test]
    fn movsx_sign_extends_8bit() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false,
        };
        cpu.set_eip(0x1000);
        // movsx eax, al (0F BE C0) ; hlt
        cpu.mem.load(0x1000, &[
            0x0F, 0xBE, 0xC0,
            0xF4,
        ]);
        cpu.set_reg8(Reg8::Al, 0xFF);
        cpu.run(32);
        assert_eq!(cpu.eax(), 0xFFFF_FFFF);
        assert!(cpu.halted);
    }

    #[test]
    fn wait_and_lldt_are_noops() {
        let mut cpu = Cpu::new();
        // wait (9B) ; lldt ax (0F 00 D0) ; hlt
        load(&mut cpu, &[
            0x9B,
            0x0F, 0x00, 0xD0,
            0xF4,
        ]);
        cpu.run(16);
        assert!(cpu.halted);
    }

    #[test]
    fn mov_to_from_cr() {
        let mut cpu = Cpu::new();
        // mov eax, 0x12345000 ; mov cr3, eax ; mov ebx, cr3
        load(&mut cpu, &[
            0x66, 0xB8, 0x00, 0x50, 0x34, 0x12,
            0x0F, 0x22, 0xD8,
            0x66, 0x8B, 0xD8,
            0x0F, 0x20, 0xDB,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.cr3, 0x12345000);
        assert_eq!(cpu.ebx(), 0x12345000);
    }

    #[test]
    fn paging_translates_linear_to_physical() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        // Flat data segment: base 0, limit 4 GiB.
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true, l: false,
        };
        // Page directory at 0x1000, page table at 0x2000.
        // Map linear 0x0040_0000 (PD 1, PT 0) to physical 0x1000.
        cpu.mem.write_u32(0x1000 + 1 * 4, 0x2003);
        cpu.mem.write_u32(0x2000 + 0 * 4, 0x1003);
        cpu.cr3 = 0x1000;
        cpu.cr0 = 0x8000_0000; // PG set
        // Write a value at physical 0x1000, read it via linear 0x0040_0000.
        cpu.mem.write_u32(0x1000, 0xCAFEBABE);
        let phys = cpu.translate(SegReg::Ds, (0x0040_0000) as u64);
        assert_eq!(phys, 0x1000);
        assert_eq!(cpu.mem.read_u32(phys), 0xCAFEBABE);
    }

    #[test]
    fn paging_disabled_identity_maps() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true, l: false,
        };
        // PG clear: linear == physical.
        cpu.cr0 = 0;
        assert_eq!(cpu.translate(SegReg::Ds, (0x1234) as u64), 0x1234);
    }

    // ---- Exception tests ----

    #[test]
    fn divide_by_zero_raises_de() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
        // Install an IVT entry for vector 0x00 (#DE) -> handler at 0x0000:0x0300.
        cpu.mem.write_u16(0x00 * 4, 0x0300);
        cpu.mem.write_u16(0x00 * 4 + 2, 0x0000);
        // Handler: mov ax, 0xDEAD ; hlt. It deliberately does not IRET --
        // #DE is a fault, so the saved address is the DIV itself and
        // returning without fixing the divisor would just re-run it.
        cpu.mem.load(0x0300, &[
            0xB8, 0xAD, 0xDE,
            0xF4,
        ]);
        // mov ax, 0x0001 ; mov bx, 0x0000 ; div bx ; hlt
        cpu.ip = 0x1000;
        load(&mut cpu, &[
            0xB8, 0x01, 0x00,
            0xBB, 0x00, 0x00,
            0xF7, 0xF3,
            0xF4,
        ]);
        cpu.run(32);
        // The #DE handler ran: AX = 0xDEAD.
        assert_eq!(cpu.ax(), 0xDEAD);
        assert!(cpu.halted);
        // The frame's return address is the DIV, not the instruction after
        // it: a fault is restartable.
        assert_eq!(cpu.mem.read_u16(0x00FA), 0x1006);
    }

    #[test]
    fn int3_raises_bp() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
        // IVT entry for vector 0x03 (#BP) -> handler at 0x0000:0x0300.
        cpu.mem.write_u16(0x03 * 4, 0x0300);
        cpu.mem.write_u16(0x03 * 4 + 2, 0x0000);
        // Handler: mov ax, 0x1234 ; iret
        cpu.mem.load(0x0300, &[
            0xB8, 0x34, 0x12,
            0xCF,
        ]);
        // int3 ; hlt
        load(&mut cpu, &[
            0xCC,
            0xF4,
        ]);
        cpu.run(32);
        assert_eq!(cpu.ax(), 0x1234);
        assert!(cpu.halted);
    }

    #[test]
    fn into_raises_of_when_overflow_set() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
        // IVT entry for vector 0x04 (#OF) -> handler at 0x0000:0x0300.
        cpu.mem.write_u16(0x04 * 4, 0x0300);
        cpu.mem.write_u16(0x04 * 4 + 2, 0x0000);
        // Handler: mov ax, 0x7777 ; iret
        cpu.mem.load(0x0300, &[
            0xB8, 0x77, 0x77,
            0xCF,
        ]);
        // Set OF, then into ; hlt
        cpu.set_flag(flags::OF, true);
        load(&mut cpu, &[
            0xCE,
            0xF4,
        ]);
        cpu.run(32);
        assert_eq!(cpu.ax(), 0x7777);
        assert!(cpu.halted);
    }

    #[test]
    fn invalid_opcode_raises_ud() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
        // IVT entry for vector 0x06 (#UD) -> handler at 0x0000:0x0300.
        cpu.mem.write_u16(0x06 * 4, 0x0300);
        cpu.mem.write_u16(0x06 * 4 + 2, 0x0000);
        // Handler: mov ax, 0xBEEF ; hlt (#UD is a fault -- see the #DE test).
        cpu.mem.load(0x0300, &[
            0xB8, 0xEF, 0xBE,
            0xF4,
        ]);
        // 0x0F 0xFF is an invalid opcode (not implemented) ; hlt
        load(&mut cpu, &[
            0x0F, 0xFF,
            0xF4,
        ]);
        cpu.run(32);
        assert_eq!(cpu.ax(), 0xBEEF);
        assert!(cpu.halted);
        // The saved address is the invalid instruction itself (loaded at
        // CS:IP = 0:0 by the test helper), not the byte after it.
        assert_eq!(cpu.mem.read_u16(0x00FA), 0x0000);
    }

    #[test]
    fn page_fault_raises_pf_through_idt() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.ss = 0;
        cpu.set_esp(0x0100);
        // Flat data segment: base 0.
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true, l: false,
        };
        // IDT at 0x3000, vector 0x0E (#PF) entry at 0x3000 + 0x0E*8 = 0x3070.
        cpu.idt_base = 0x3000;
        cpu.idt_limit = 0xFF;
        let entry = 0x3000 + 0x0E * 8;
        cpu.mem.write_u16(entry, 0x5000);
        cpu.mem.write_u16(entry + 2, 0x08);
        cpu.mem.write_u16(entry + 6, 0x0000);
        // Handler at 0x5000: mov eax, 0xCAFE ; hlt. No IRET: #PF pushes an
        // error code on top of the frame, so a handler has to drop it before
        // returning -- and returning at all would re-run the faulting load.
        cpu.mem.load(0x5000, &[
            0x66, 0xB8, 0xFE, 0xCA, 0x00, 0x00,
            0xF4,
        ]);
        // Page directory at 0x2000. Identity-map the low 4 MiB (PDE[0] ->
        // page table at 0x3000, PT[0..5] -> pages 0x0000..0x5000) so the
        // code at 0x1000 and the handler at 0x5000 both fetch fine. Leave
        // PDE[1] NOT-present so linear 0x0040_0000 faults.
        cpu.mem.write_u32(0x2000 + 0 * 4, 0x3003); // PDE[0] -> PT 0x3000, present
        for i in 0..6 {
            cpu.mem.write_u32(0x3000 + i * 4, (i << 12) as u32 | 0x3); // PT[i] -> page i
        }
        cpu.mem.write_u32(0x2000 + 1 * 4, 0x2000); // PDE[1] not present
        cpu.cr3 = 0x2000;
        cpu.cr0 = 0x8000_0000; // PG set
        // mov eax, [0x0040_0000] ; hlt  (linear 0x0040_0000 -> PD index 1)
        // 66 = 32-bit operand, 67 = 32-bit addressing (so moffs is 32-bit).
        cpu.mem.load(0x1000, &[
            0x66, 0x67, 0xA1, 0x00, 0x00, 0x40, 0x00,
            0xF4,
        ]);
        cpu.cs = 0x08;
        cpu.set_eip(0x1000);
        cpu.run(32);
        // The #PF handler ran: EAX = 0xCAFE.
        assert_eq!(cpu.eax(), 0xCAFE);
        // CR2 holds the faulting linear address.
        assert_eq!(cpu.cr2, 0x0040_0000);
        assert!(cpu.halted);
        // The frame is EFLAGS, CS, EIP, error code -- error code on TOP, at
        // the lowest address, which is where a handler reads it from.
        assert_eq!(cpu.mem.read_u32(0x00F0), 0x0000_0000); // error code: not present, read
        assert_eq!(cpu.mem.read_u32(0x00F4), 0x1000);      // EIP: the faulting load
        assert_eq!(cpu.mem.read_u32(0x00F8), 0x08);        // CS
    }

    // ---- Tests for the instructions and behaviours added while getting a
    // ---- real Linux kernel to boot. Each of these pins a bug that was found
    // ---- the hard way, from a kernel that crashed thousands of instructions
    // ---- later with no obvious connection to the cause.

    /// Set up a CPU in flat 32-bit protected mode with code at 0x1000 and a
    /// stack at 0x8000, the shape almost every 32-bit test wants.
    fn flat32() -> Cpu {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.ss = 0x10;
        let code = Descriptor { base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true, l: false };
        let data = Descriptor { base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true, l: false };
        cpu.seg_desc[SegReg::Cs as usize] = code;
        for s in [SegReg::Ds, SegReg::Es, SegReg::Ss, SegReg::Fs, SegReg::Gs] {
            cpu.seg_desc[s as usize] = data;
        }
        cpu.set_eip(0x1000);
        cpu.set_esp(0x8000);
        cpu
    }

    fn run32(bytes: &[u8]) -> Cpu {
        let mut cpu = flat32();
        cpu.mem.load(0x1000, bytes);
        cpu.run(200);
        cpu
    }

    #[test]
    fn cmp_sets_flags_without_writing_the_destination() {
        // CMP is SUB that throws the result away. Writing it back corrupts a
        // register on every comparison -- and the kernel compares constantly.
        let cpu = run32(&[
            0xB8, 0x0A, 0x00, 0x00, 0x00, // mov eax, 10
            0xBB, 0x03, 0x00, 0x00, 0x00, // mov ebx, 3
            0x39, 0xD8,                   // cmp eax, ebx
            0xF4,
        ]);
        assert_eq!(cpu.eax(), 10, "CMP must not modify its destination");
        assert_eq!(cpu.ebx(), 3);
        assert!(!cpu.get_flag(flags::ZF));
        assert!(!cpu.get_flag(flags::CF));
    }

    #[test]
    fn cmp_reg_form_also_leaves_the_register_alone() {
        let cpu = run32(&[
            0xB8, 0x03, 0x00, 0x00, 0x00, // mov eax, 3
            0xBB, 0x0A, 0x00, 0x00, 0x00, // mov ebx, 10
            0x3B, 0xC3,                   // cmp eax, ebx  (Dir::RegRm)
            0xF4,
        ]);
        assert_eq!(cpu.eax(), 3);
        assert!(cpu.get_flag(flags::CF), "3 - 10 borrows");
    }

    #[test]
    fn shifts_are_32_bit_wide_in_a_32_bit_segment() {
        // D1/D3 in a D=1 segment shift the whole 32-bit register. Treating
        // them as byte shifts leaves the top 24 bits untouched, which turns
        // a `value >>= 4` loop into an infinite one.
        let cpu = run32(&[
            0xBA, 0x00, 0x00, 0x1A, 0xE1, // mov edx, 0xE11A0000
            0xB1, 0x04,                   // mov cl, 4
            0xD3, 0xEA,                   // shr edx, cl
            0xF4,
        ]);
        assert_eq!(cpu.edx(), 0x0E11_A000);
    }

    #[test]
    fn shift_by_one_and_by_immediate_agree() {
        let cpu = run32(&[
            0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
            0xD1, 0xE8,                   // shr eax, 1
            0xBB, 0x00, 0x00, 0x00, 0x80, // mov ebx, 0x80000000
            0xC1, 0xEB, 0x01,             // shr ebx, 1
            0xF4,
        ]);
        assert_eq!(cpu.eax(), 0x4000_0000);
        assert_eq!(cpu.ebx(), 0x4000_0000);
    }

    #[test]
    fn shift_count_past_the_operand_width_clears_it() {
        // The count is masked to 5 bits for every operand size, so a 16-bit
        // shift by 20 really does shift a 16-bit value 20 places.
        let mut cpu = flat32();
        cpu.mem.load(0x1000, &[
            0x66, 0xB8, 0xFF, 0xFF, // mov ax, 0xFFFF
            0xB1, 0x14,             // mov cl, 20
            0x66, 0xD3, 0xE0,       // shl ax, cl
            0xF4,
        ]);
        cpu.run(64);
        assert_eq!(cpu.ax(), 0);
    }

    #[test]
    fn sar_replicates_the_sign_bit() {
        let cpu = run32(&[
            0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
            0xC1, 0xF8, 0x04,             // sar eax, 4
            0xF4,
        ]);
        assert_eq!(cpu.eax(), 0xF800_0000);
    }

    #[test]
    fn rotate_through_carry_by_more_than_one() {
        // RCL rotates through a 33-bit quantity; doing it as a plain 32-bit
        // rotate loses the carry bit for any count above 1.
        let cpu = run32(&[
            0xF8,                         // clc
            0xB8, 0x01, 0x00, 0x00, 0x80, // mov eax, 0x80000001
            0xC1, 0xD0, 0x02,             // rcl eax, 2
            0xF4,
        ]);
        // Worked through as a 33-bit rotate: (CF:EAX) = 0_80000001 rotated
        // left twice gives 0_00000005, so EAX = 5 and CF comes back clear.
        // A plain 32-bit rotate would give 6 and lose the carry.
        assert_eq!(cpu.eax(), 0x0000_0005);
        assert!(!cpu.get_flag(flags::CF));
    }

    #[test]
    fn pusha_and_popa_round_trip() {
        let cpu = run32(&[
            0xB8, 0x11, 0x11, 0x11, 0x11, // mov eax, 0x11111111
            0xBB, 0x22, 0x22, 0x22, 0x22, // mov ebx, 0x22222222
            0xB9, 0x33, 0x33, 0x33, 0x33, // mov ecx, 0x33333333
            0x60,                         // pushad
            0x31, 0xC0,                   // xor eax, eax
            0x31, 0xDB,                   // xor ebx, ebx
            0x31, 0xC9,                   // xor ecx, ecx
            0x61,                         // popad
            0xF4,
        ]);
        assert_eq!(cpu.eax(), 0x1111_1111);
        assert_eq!(cpu.ebx(), 0x2222_2222);
        assert_eq!(cpu.ecx(), 0x3333_3333);
        assert_eq!(cpu.esp(), 0x8000, "POPAD must restore the stack pointer");
    }

    #[test]
    fn pop_rm_writes_memory() {
        let cpu = run32(&[
            0x68, 0xEF, 0xBE, 0xAD, 0xDE, // push 0xDEADBEEF
            0x8F, 0x05, 0x00, 0x30, 0x00, 0x00, // pop dword [0x3000]
            0xF4,
        ]);
        assert_eq!(cpu.mem.read_u32(0x3000), 0xDEAD_BEEF);
        assert_eq!(cpu.esp(), 0x8000);
    }

    #[test]
    fn push_imm8_pushes_four_bytes_in_32_bit_mode() {
        // A two-byte push here misaligns the stack for everything after it.
        let cpu = run32(&[
            0x6A, 0xFF, // push -1
            0xF4,
        ]);
        assert_eq!(cpu.esp(), 0x7FFC);
        assert_eq!(cpu.mem.read_u32(0x7FFC), 0xFFFF_FFFF, "sign-extended");
    }

    #[test]
    fn setcc_writes_one_or_zero() {
        let cpu = run32(&[
            0xB8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
            0x83, 0xF8, 0x05,             // cmp eax, 5
            0x0F, 0x94, 0xC3,             // sete bl
            0x0F, 0x95, 0xC7,             // setne bh
            0xF4,
        ]);
        assert_eq!(cpu.ebx() & 0xFF, 1);
        assert_eq!((cpu.ebx() >> 8) & 0xFF, 0);
    }

    #[test]
    fn imul_three_operand_and_two_operand() {
        let cpu = run32(&[
            0xB8, 0x07, 0x00, 0x00, 0x00, // mov eax, 7
            0x6B, 0xD8, 0x06,             // imul ebx, eax, 6
            0xB9, 0x03, 0x00, 0x00, 0x00, // mov ecx, 3
            0x0F, 0xAF, 0xC8,             // imul ecx, eax
            0xF4,
        ]);
        assert_eq!(cpu.ebx(), 42);
        assert_eq!(cpu.ecx(), 21);
    }

    #[test]
    fn imul_sets_carry_when_the_product_does_not_fit() {
        let cpu = run32(&[
            0xB8, 0x00, 0x00, 0x00, 0x40, // mov eax, 0x40000000
            0x6B, 0xC0, 0x04,             // imul eax, eax, 4
            0xF4,
        ]);
        assert_eq!(cpu.eax(), 0, "low half of the product");
        assert!(cpu.get_flag(flags::CF));
        assert!(cpu.get_flag(flags::OF));
    }

    #[test]
    fn shrd_and_shld_move_bits_between_registers() {
        let cpu = run32(&[
            0xB8, 0xFF, 0xFF, 0x0F, 0x00, // mov eax, 0x000FFFFF
            0xBA, 0x00, 0x00, 0x1A, 0xE1, // mov edx, 0xE11A0000
            0xB1, 0x04,                   // mov cl, 4
            0x0F, 0xAD, 0xD0,             // shrd eax, edx, cl
            0xF4,
        ]);
        // eax >> 4, with edx's low nibble shifted into the top.
        assert_eq!(cpu.eax(), 0x0000_FFFF);
    }

    #[test]
    fn bsf_and_bsr_find_the_end_bits() {
        let cpu = run32(&[
            0xB8, 0x00, 0x01, 0x00, 0x01, // mov eax, 0x01000100
            0x0F, 0xBC, 0xD8,             // bsf ebx, eax
            0x0F, 0xBD, 0xC8,             // bsr ecx, eax
            0xF4,
        ]);
        assert_eq!(cpu.ebx(), 8);
        assert_eq!(cpu.ecx(), 24);
        assert!(!cpu.get_flag(flags::ZF));
    }

    #[test]
    fn bsf_of_zero_sets_zf_and_leaves_the_destination() {
        let cpu = run32(&[
            0xBB, 0x99, 0x00, 0x00, 0x00, // mov ebx, 0x99
            0x31, 0xC0,                   // xor eax, eax
            0x0F, 0xBC, 0xD8,             // bsf ebx, eax
            0xF4,
        ]);
        assert!(cpu.get_flag(flags::ZF));
        assert_eq!(cpu.ebx(), 0x99);
    }

    #[test]
    fn bit_test_uses_the_registers_value_not_its_number() {
        // BT with a register offset takes the *value* in that register. Using
        // the ModR/M reg field instead makes every `test_bit()` answer from
        // whichever bit the register number happens to name -- which had the
        // kernel believing every interrupt vector was already claimed.
        let cpu = run32(&[
            0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
            0xBA, 0x1F, 0x00, 0x00, 0x00, // mov edx, 31
            0x0F, 0xA3, 0xD0,             // bt eax, edx
            0xF4,
        ]);
        assert!(cpu.get_flag(flags::CF), "bit 31 of 0x80000000 is set");
    }

    #[test]
    fn bit_test_on_memory_indexes_a_bit_string() {
        // With a memory operand the offset is not masked to the operand size:
        // it selects a dword further along, which is how kernel bitmaps work.
        let mut cpu = flat32();
        cpu.mem.write_u32(0x3000, 0);
        cpu.mem.write_u32(0x3004, 0x0000_0004); // bit 34 overall
        cpu.mem.load(0x1000, &[
            0xBA, 0x22, 0x00, 0x00, 0x00,             // mov edx, 34
            0x0F, 0xA3, 0x15, 0x00, 0x30, 0x00, 0x00, // bt [0x3000], edx
            0xF4,
        ]);
        cpu.run(64);
        assert!(cpu.get_flag(flags::CF));
    }

    #[test]
    fn bts_on_memory_sets_the_right_bit() {
        let mut cpu = flat32();
        cpu.mem.load(0x1000, &[
            0xBA, 0x21, 0x00, 0x00, 0x00,             // mov edx, 33
            0x0F, 0xAB, 0x15, 0x00, 0x30, 0x00, 0x00, // bts [0x3000], edx
            0xF4,
        ]);
        cpu.run(64);
        assert_eq!(cpu.mem.read_u32(0x3000), 0);
        assert_eq!(cpu.mem.read_u32(0x3004), 0x0000_0002);
    }

    #[test]
    fn xchg_swaps_register_and_memory() {
        let mut cpu = flat32();
        cpu.mem.write_u32(0x3000, 0xAAAA_AAAA);
        cpu.mem.load(0x1000, &[
            0xB8, 0xBB, 0xBB, 0xBB, 0xBB,       // mov eax, 0xBBBBBBBB
            0x87, 0x05, 0x00, 0x30, 0x00, 0x00, // xchg [0x3000], eax
            0xF4,
        ]);
        cpu.run(64);
        assert_eq!(cpu.eax(), 0xAAAA_AAAA);
        assert_eq!(cpu.mem.read_u32(0x3000), 0xBBBB_BBBB);
    }

    #[test]
    fn cmpxchg_swaps_only_on_a_match() {
        let mut cpu = flat32();
        cpu.mem.write_u32(0x3000, 5);
        cpu.mem.load(0x1000, &[
            0xB8, 0x05, 0x00, 0x00, 0x00,       // mov eax, 5   (expected)
            0xBB, 0x63, 0x00, 0x00, 0x00,       // mov ebx, 99  (new)
            0x0F, 0xB1, 0x1D, 0x00, 0x30, 0x00, 0x00, // cmpxchg [0x3000], ebx
            0xF4,
        ]);
        cpu.run(64);
        assert!(cpu.get_flag(flags::ZF));
        assert_eq!(cpu.mem.read_u32(0x3000), 99);
    }

    #[test]
    fn cmpxchg_loads_the_accumulator_on_a_mismatch() {
        let mut cpu = flat32();
        cpu.mem.write_u32(0x3000, 7);
        cpu.mem.load(0x1000, &[
            0xB8, 0x05, 0x00, 0x00, 0x00,
            0xBB, 0x63, 0x00, 0x00, 0x00,
            0x0F, 0xB1, 0x1D, 0x00, 0x30, 0x00, 0x00,
            0xF4,
        ]);
        cpu.run(64);
        assert!(!cpu.get_flag(flags::ZF));
        assert_eq!(cpu.eax(), 7, "accumulator takes the destination's value");
        assert_eq!(cpu.mem.read_u32(0x3000), 7, "destination unchanged");
    }

    #[test]
    fn xadd_exchanges_and_adds() {
        let mut cpu = flat32();
        cpu.mem.write_u32(0x3000, 10);
        cpu.mem.load(0x1000, &[
            0xBB, 0x05, 0x00, 0x00, 0x00,             // mov ebx, 5
            0x0F, 0xC1, 0x1D, 0x00, 0x30, 0x00, 0x00, // xadd [0x3000], ebx
            0xF4,
        ]);
        cpu.run(64);
        assert_eq!(cpu.ebx(), 10, "source register gets the old destination");
        assert_eq!(cpu.mem.read_u32(0x3000), 15);
    }

    #[test]
    fn cmpxchg8b_compares_and_swaps_64_bits() {
        let mut cpu = flat32();
        cpu.mem.write_u32(0x3000, 0x1111_1111);
        cpu.mem.write_u32(0x3004, 0x2222_2222);
        cpu.mem.load(0x1000, &[
            0xB8, 0x11, 0x11, 0x11, 0x11,       // mov eax, 0x11111111
            0xBA, 0x22, 0x22, 0x22, 0x22,       // mov edx, 0x22222222
            0xBB, 0x44, 0x44, 0x44, 0x44,       // mov ebx, 0x44444444
            0xB9, 0x33, 0x33, 0x33, 0x33,       // mov ecx, 0x33333333
            0x0F, 0xC7, 0x0D, 0x00, 0x30, 0x00, 0x00, // cmpxchg8b [0x3000]
            0xF4,
        ]);
        cpu.run(64);
        assert!(cpu.get_flag(flags::ZF));
        assert_eq!(cpu.mem.read_u32(0x3000), 0x4444_4444);
        assert_eq!(cpu.mem.read_u32(0x3004), 0x3333_3333);
    }

    #[test]
    fn bswap_reverses_byte_order() {
        let cpu = run32(&[
            0xB8, 0x78, 0x56, 0x34, 0x12, // mov eax, 0x12345678
            0x0F, 0xC8,                   // bswap eax
            0xF4,
        ]);
        assert_eq!(cpu.eax(), 0x7856_3412);
    }

    #[test]
    fn cmovcc_moves_only_when_the_condition_holds() {
        let cpu = run32(&[
            0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
            0xBB, 0x63, 0x00, 0x00, 0x00, // mov ebx, 99
            0x85, 0xC0,                   // test eax, eax
            0x0F, 0x44, 0xD8,             // cmove ebx, eax   (not taken)
            0x0F, 0x45, 0xD8,             // cmovne ebx, eax  (taken)
            0xF4,
        ]);
        assert_eq!(cpu.ebx(), 1);
    }

    #[test]
    fn leave_tears_down_the_frame() {
        let cpu = run32(&[
            0xBD, 0x00, 0x70, 0x00, 0x00, // mov ebp, 0x7000
            0x55,                         // push ebp
            0x89, 0xE5,                   // mov ebp, esp
            0x83, 0xEC, 0x20,             // sub esp, 32
            0xC9,                         // leave
            0xF4,
        ]);
        assert_eq!(cpu.ebp(), 0x7000);
        assert_eq!(cpu.esp(), 0x8000);
    }

    #[test]
    fn ret_imm16_drops_arguments() {
        let cpu = run32(&[
            0x68, 0x08, 0x00, 0x00, 0x00, // push 8   (a fake argument)
            0xE8, 0x01, 0x00, 0x00, 0x00, // call +1
            0xF4,                         // hlt (returned here)
            0xC2, 0x04, 0x00,             // ret 4
        ]);
        assert!(cpu.halted);
        assert_eq!(cpu.esp(), 0x8000, "the pushed argument was dropped too");
    }

    #[test]
    fn inc_and_dec_r_m8_preserve_carry() {
        let mut cpu = flat32();
        cpu.mem.write_u8(0x3000, 0x7F);
        cpu.mem.load(0x1000, &[
            0xF9,                               // stc
            0xFE, 0x05, 0x00, 0x30, 0x00, 0x00, // inc byte [0x3000]
            0xF4,
        ]);
        cpu.run(64);
        assert_eq!(cpu.mem.read_u8(0x3000), 0x80);
        assert!(cpu.get_flag(flags::CF), "INC leaves CF alone");
        assert!(cpu.get_flag(flags::OF), "0x7F + 1 overflows a signed byte");
    }

    #[test]
    fn lock_prefix_is_accepted() {
        // `lock` is a prefix, not an opcode: failing to consume it turns
        // every atomic operation in the kernel into an invalid instruction.
        let mut cpu = flat32();
        cpu.mem.write_u32(0x3000, 1);
        cpu.mem.load(0x1000, &[
            0xF0, 0xFF, 0x05, 0x00, 0x30, 0x00, 0x00, // lock inc dword [0x3000]
            0xF4,
        ]);
        cpu.run(64);
        assert!(cpu.unknown_ops.is_empty(), "LOCK must decode as a prefix");
        assert_eq!(cpu.mem.read_u32(0x3000), 2);
    }

    #[test]
    fn multi_byte_nop_decodes() {
        let cpu = run32(&[
            0x0F, 0x1F, 0x40, 0x00, // nopl 0x0(%eax)
            0xF4,
        ]);
        assert!(cpu.unknown_ops.is_empty());
        assert!(cpu.halted);
    }

    #[test]
    fn pushfd_and_popfd_move_the_whole_of_eflags() {
        // Linux toggles the ID bit (21) through PUSHFD/POPFD to decide whether
        // CPUID exists at all. A 16-bit flags word cannot carry it.
        let cpu = run32(&[
            0x9C,                               // pushfd
            0x58,                               // pop eax
            0x35, 0x00, 0x00, 0x20, 0x00,       // xor eax, 0x00200000  (ID)
            0x50,                               // push eax
            0x9D,                               // popfd
            0x9C,                               // pushfd
            0x5B,                               // pop ebx
            0xF4,
        ]);
        assert_eq!(cpu.ebx() & 0x0020_0000, 0x0020_0000, "ID bit is writable");
        assert_eq!(cpu.esp(), 0x8000);
    }

    #[test]
    fn segment_registers_push_and_pop() {
        let cpu = run32(&[
            0x0F, 0xA0, // push fs
            0x1E,       // push ds
            0x1F,       // pop ds
            0x0F, 0xA1, // pop fs
            0xF4,
        ]);
        assert!(cpu.unknown_ops.is_empty());
        assert_eq!(cpu.esp(), 0x8000);
    }

    #[test]
    fn debug_registers_read_back_what_was_written() {
        let cpu = run32(&[
            0xB8, 0x55, 0x00, 0x00, 0x00, // mov eax, 0x55
            0x0F, 0x23, 0xF8,             // mov dr7, eax
            0x0F, 0x21, 0xFB,             // mov ebx, dr7
            0xF4,
        ]);
        assert_eq!(cpu.ebx(), 0x55);
    }

    #[test]
    fn moffs_honours_a_segment_override() {
        // `mov %gs:0xC,%eax` is how i386 userspace reads thread-local storage.
        // Translating it through DS instead reads address 0xC.
        let mut cpu = flat32();
        cpu.seg_desc[SegReg::Gs as usize] = Descriptor {
            base: 0x5000, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true, l: false,
        };
        cpu.mem.write_u32(0x500C, 0xFEED_FACE);
        cpu.mem.write_u32(0x000C, 0xDEAD_0000);
        cpu.mem.load(0x1000, &[
            0x65, 0xA1, 0x0C, 0x00, 0x00, 0x00, // mov eax, gs:0xC
            0xF4,
        ]);
        cpu.run(64);
        assert_eq!(cpu.eax(), 0xFEED_FACE);
    }

    #[test]
    fn an_immediate_that_ends_on_a_page_boundary_is_fetched_whole() {
        // The instruction-fetch cache advances a physical pointer. When an
        // operand's last byte sits at offset 0xFFF the cursor lands on the
        // next *virtual* page, whose physical page is unrelated -- so the
        // cache has to be dropped. Here the two pages are deliberately not
        // contiguous in physical memory, so a stale cache reads rubbish.
        let mut cpu = flat32();
        cpu.cr3 = 0x20000;
        // Page directory entry 0 -> page table at 0x21000.
        cpu.mem.write_u32(0x20000, 0x21003);
        // Identity-map the first 16 pages so the stack and data work.
        for i in 0..16u32 {
            cpu.mem.write_u32(0x21000 + i as usize * 4, (i << 12) | 3);
        }
        // Virtual page 0x10 -> physical 0x40000, virtual 0x11 -> 0x60000:
        // adjacent virtually, far apart physically.
        cpu.mem.write_u32(0x21000 + 0x10 * 4, 0x40003);
        cpu.mem.write_u32(0x21000 + 0x11 * 4, 0x60003);
        cpu.cr0 = 0x8000_0001;
        // `mov dword [0x3000], 0xAABBCCDD` placed so its immediate straddles
        // the boundary: the instruction is 10 bytes and starts 6 before it.
        let code: [u8; 11] = [
            0xC7, 0x05, 0x00, 0x30, 0x00, 0x00, 0xDD, 0xCC, 0xBB, 0xAA,
            0xF4,
        ];
        // The first six bytes sit at the end of physical page 0x40000; the
        // rest continue at the start of physical page 0x60000, because that
        // is where the *next virtual* page actually lives.
        for (i, b) in code.iter().enumerate() {
            if i < 6 {
                cpu.mem.write_u8(0x40000 + 0x1000 - 6 + i, *b);
            } else {
                cpu.mem.write_u8(0x60000 + (i - 6), *b);
            }
        }
        // Physical 0x41000 -- where a stale fetch cache would keep reading --
        // holds a pattern that must not reach the immediate.
        for i in 0..8 {
            cpu.mem.write_u8(0x41000 + i, 0x77);
        }
        cpu.set_eip(0x10000 + 0x1000 - 6);
        cpu.run(16);
        assert_eq!(cpu.mem.read_u32(0x3000), 0xAABB_CCDD);
    }

    #[test]
    fn a_faulting_instruction_does_not_commit_its_result() {
        // `add (%edi),%edx` whose load page-faults must leave EDX untouched:
        // the instruction restarts after the handler, and a partial commit
        // makes the retry add to an already-modified value.
        let mut cpu = flat32();
        cpu.cr3 = 0x20000;
        cpu.mem.write_u32(0x20000, 0x21003);
        for i in 0..16u32 {
            cpu.mem.write_u32(0x21000 + i as usize * 4, (i << 12) | 3);
        }
        // Leave virtual page 0x30 unmapped.
        cpu.cr0 = 0x8000_0001;
        cpu.idt_base = 0x5000;
        cpu.idt_limit = 0xFF;
        // #PF handler: just halt, so the test can look at the register.
        let entry = 0x5000 + 0x0E * 8;
        cpu.mem.write_u16(entry, 0x2000);
        cpu.mem.write_u16(entry + 2, 0x08);
        cpu.mem.write_u16(entry + 4, 0x8E00);
        cpu.mem.write_u16(entry + 6, 0x0000);
        cpu.mem.write_u8(0x2000, 0xF4);
        cpu.mem.load(0x1000, &[
            0xBA, 0x0A, 0x00, 0x00, 0x00,       // mov edx, 10
            0xBF, 0x00, 0x00, 0x03, 0x00,       // mov edi, 0x30000 (unmapped)
            0x03, 0x17,                         // add edx, [edi]
            0xF4,
        ]);
        cpu.run(64);
        assert_eq!(cpu.cr2, 0x0003_0000, "the faulting address");
        assert_eq!(cpu.edx(), 10, "EDX must be untouched by the faulted add");
    }

    #[test]
    fn a_rep_string_stops_at_the_element_that_faults() {
        // The count and index registers have to point at the failing element
        // so the restart resumes there. Running the loop to completion through
        // the fault writes the rest of it into nowhere and the retry copies
        // nothing.
        let mut cpu = flat32();
        cpu.cr3 = 0x20000;
        cpu.mem.write_u32(0x20000, 0x21003);
        for i in 0..16u32 {
            cpu.mem.write_u32(0x21000 + i as usize * 4, (i << 12) | 3);
        }
        cpu.cr0 = 0x8000_0001;
        cpu.idt_base = 0x5000;
        cpu.idt_limit = 0xFF;
        let entry = 0x5000 + 0x0E * 8;
        cpu.mem.write_u16(entry, 0x2000);
        cpu.mem.write_u16(entry + 2, 0x08);
        cpu.mem.write_u16(entry + 4, 0x8E00);
        cpu.mem.write_u16(entry + 6, 0x0000);
        cpu.mem.write_u8(0x2000, 0xF4);
        // Store 0x100 dwords from 0xFF00: the last of them runs off the end
        // of the mapped region at 0x10000.
        cpu.mem.load(0x1000, &[
            0xB8, 0xFF, 0xFF, 0xFF, 0xFF,       // mov eax, 0xFFFFFFFF
            0xBF, 0x00, 0xFF, 0x00, 0x00,       // mov edi, 0xFF00
            0xB9, 0x00, 0x01, 0x00, 0x00,       // mov ecx, 0x100
            0xF3, 0xAB,                         // rep stosd
            0xF4,
        ]);
        cpu.run(64);
        assert_eq!(cpu.cr2 & !0xFFF, 0x0001_0000, "faulted on the next page");
        assert_eq!(cpu.edi(), 0x1_0000, "EDI stops at the faulting element");
        assert_eq!(cpu.ecx(), 0x100 - 0x40, "and so does the count");
        // Everything before the fault really was written.
        assert_eq!(cpu.mem.read_u32(0xFFFC), 0xFFFF_FFFF);
    }

    #[test]
    fn write_protect_faults_a_supervisor_write_to_a_read_only_page() {
        // CR0.WP. Linux checks this before it will run at all.
        let mut cpu = flat32();
        cpu.cr3 = 0x20000;
        cpu.mem.write_u32(0x20000, 0x21003);
        for i in 0..16u32 {
            cpu.mem.write_u32(0x21000 + i as usize * 4, (i << 12) | 3);
        }
        // Make virtual page 5 read-only.
        cpu.mem.write_u32(0x21000 + 5 * 4, 0x5001);
        cpu.idt_base = 0x5000;
        cpu.idt_limit = 0xFF;
        let entry = 0x5000 + 0x0E * 8;
        cpu.mem.write_u16(entry, 0x2000);
        cpu.mem.write_u16(entry + 2, 0x08);
        cpu.mem.write_u16(entry + 4, 0x8E00);
        cpu.mem.write_u16(entry + 6, 0x0000);
        cpu.mem.write_u8(0x2000, 0xF4);

        // With WP clear the write goes through.
        cpu.cr0 = 0x8000_0001;
        cpu.mem.load(0x1000, &[
            0xC7, 0x05, 0x00, 0x50, 0x00, 0x00, 0x2A, 0x00, 0x00, 0x00,
            0xF4,
        ]);
        cpu.run(32);
        assert_eq!(cpu.mem.read_u32(0x5000), 42);

        // With WP set the same write faults.
        let mut cpu2 = flat32();
        cpu2.cr3 = 0x20000;
        cpu2.mem.write_u32(0x20000, 0x21003);
        for i in 0..16u32 {
            cpu2.mem.write_u32(0x21000 + i as usize * 4, (i << 12) | 3);
        }
        cpu2.mem.write_u32(0x21000 + 5 * 4, 0x5001);
        cpu2.idt_base = 0x5000;
        cpu2.idt_limit = 0xFF;
        cpu2.mem.write_u16(entry, 0x2000);
        cpu2.mem.write_u16(entry + 2, 0x08);
        cpu2.mem.write_u16(entry + 4, 0x8E00);
        cpu2.mem.write_u16(entry + 6, 0x0000);
        cpu2.mem.write_u8(0x2000, 0xF4);
        cpu2.cr0 = 0x8001_0001; // PG | WP | PE
        cpu2.mem.load(0x1000, &[
            0xC7, 0x05, 0x00, 0x50, 0x00, 0x00, 0x63, 0x00, 0x00, 0x00,
            0xF4,
        ]);
        cpu2.run(32);
        assert_eq!(cpu2.cr2, 0x5000);
        assert_ne!(cpu2.mem.read_u32(0x5000), 99, "the write must not land");
    }

    #[test]
    fn an_exception_pushes_its_error_code_on_top_of_the_frame() {
        // EFLAGS, CS, EIP, then the error code -- error code last, so it ends
        // up at the lowest address. Any other order shifts every field the
        // handler reads by one slot.
        let mut cpu = flat32();
        cpu.cr3 = 0x20000;
        cpu.mem.write_u32(0x20000, 0x21003);
        for i in 0..16u32 {
            cpu.mem.write_u32(0x21000 + i as usize * 4, (i << 12) | 3);
        }
        cpu.cr0 = 0x8000_0001;
        cpu.idt_base = 0x5000;
        cpu.idt_limit = 0xFF;
        let entry = 0x5000 + 0x0E * 8;
        cpu.mem.write_u16(entry, 0x2000);
        cpu.mem.write_u16(entry + 2, 0x08);
        cpu.mem.write_u16(entry + 4, 0x8E00);
        cpu.mem.write_u16(entry + 6, 0x0000);
        cpu.mem.write_u8(0x2000, 0xF4);
        cpu.mem.load(0x1000, &[
            0xA1, 0x00, 0x00, 0x03, 0x00, // mov eax, [0x30000] (unmapped)
            0xF4,
        ]);
        cpu.run(32);
        let esp = cpu.esp() as usize;
        assert_eq!(cpu.mem.read_u32(esp), 0, "error code: not present, read");
        assert_eq!(cpu.mem.read_u32(esp + 4), 0x1000, "EIP: the faulting insn");
        assert_eq!(cpu.mem.read_u32(esp + 8), 0x08, "CS");
    }

    #[test]
    fn the_16_bit_view_of_a_register_is_not_a_separate_copy() {
        // Writing ECX then reading CL must see the new value. Keeping AX and
        // EAX as two fields meant any 32-bit write that forgot to refresh the
        // 16-bit half left the two disagreeing, and the next byte-register
        // write rebuilt the 32-bit register from the stale half.
        let cpu = run32(&[
            0xBF, 0x00, 0x00, 0x04, 0x00, // mov edi, 0x40000 (clear of the code)
            0xB9, 0x9C, 0x06, 0x00, 0x00, // mov ecx, 0x69C
            0xF3, 0xAB,                   // rep stosd  (drives ECX to 0)
            0xB1, 0x04,                   // mov cl, 4
            0xF4,
        ]);
        assert_eq!(cpu.ecx(), 4, "CL must write into the register ECX shares");
    }

    #[test]
    fn hlt_waits_for_an_interrupt_instead_of_stopping_the_machine() {
        // The idle loop is `hlt`; a CPU that stops there never takes another
        // timer tick and the kernel never runs again.
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.set_sp(0x0100);
        cpu.port_out(0x43, 0x36);
        cpu.port_out(0x40, 200);
        cpu.port_out(0x40, 0);
        cpu.port_out(0x20, 0x11);
        cpu.port_out(0x21, 0x08);
        cpu.mem.write_u16(0x08 * 4, 0x0200);
        cpu.mem.write_u16(0x08 * 4 + 2, 0x0000);
        // Handler: mask the timer, set AX, iret.
        cpu.mem.load(0x0200, &[
            0xB0, 0x01, 0xE6, 0x21, 0xB8, 0x42, 0x00, 0xCF,
        ]);
        cpu.mem.load(0x0100, &[0xF4, 0xF4]);
        cpu.cs = 0;
        cpu.ip = 0x0100;
        cpu.set_flag(flags::IF, true);
        cpu.run(4096);
        assert_eq!(cpu.ax(), 0x42, "the timer woke the halted CPU");
    }

    // ================================================================
    // Long mode (64-bit)
    // ================================================================
    //
    // Every test here starts from the real boot path (`boot::load_flat64`),
    // so it exercises the same long-mode entry a payload would get: PAE on,
    // a 4-level page table identity-mapping the low 4 GiB, EFER.LME set,
    // paging enabled, and a code segment with L set.

    /// Where `long_cpu` loads its code.
    const CODE64: u64 = 0x10_0000;

    pub(crate) fn long_cpu(code: &[u8]) -> Cpu {
        let mut cpu = Cpu::new();
        crate::boot::load_flat64(&mut cpu, code, CODE64).unwrap();
        cpu
    }

    /// Run an already-built machine until HLT, with a bound so a broken
    /// test fails instead of hanging. Shared with `sse.rs`'s tests.
    pub(crate) fn run64(cpu: &mut Cpu) {
        cpu.run(4096);
        assert!(cpu.halted, "did not reach HLT (rip={:016X})", cpu.rip);
        assert!(!cpu.triple_fault, "triple faulted at rip={:016X}", cpu.rip);
    }

    /// Build a long-mode machine from `code` and run it to HLT.
    fn run64_code(code: &[u8]) -> Cpu {
        let mut cpu = long_cpu(code);
        run64(&mut cpu);
        cpu
    }

    /// Map linear 0x40_0000 and 0x40_1000 to two physically UNRELATED pages
    /// (0x70_0000 and 0x90_0000), the way a vmalloc'd region is mapped, so
    /// an access that straddles the boundary has to be split.
    fn split_map(cpu: &mut Cpu) {
        use crate::paging::pte;
        // PD entry 2 of the first GiB (linear 0x40_0000-0x60_0000) becomes a
        // 4 KiB page table at 0xA000 instead of a 2 MiB page.
        let pt = 0xA000usize;
        for i in 0..512 { cpu.mem.write_u64(pt + i * 8, 0); }
        cpu.mem.write_u64(pt, 0x70_0000 | pte::P | pte::RW);
        cpu.mem.write_u64(pt + 8, 0x90_0000 | pte::P | pte::RW);
        cpu.mem.write_u64(0x4000 + 2 * 8, pt as u64 | pte::P | pte::RW);
        cpu.flush_tlb();
    }

    #[test]
    fn an_access_that_straddles_a_page_is_split_across_both_pages() {
        // movups xmm0,[0x400FF8] ; movups [0x400FF4],xmm0 ;
        // mov [0x400FFA],rbx ; mov rax,[0x400FFC] ; hlt
        let code = [
            0x0F, 0x10, 0x04, 0x25, 0xF8, 0x0F, 0x40, 0x00,          // movups xmm0,[0x400FF8]
            0x0F, 0x11, 0x04, 0x25, 0xF4, 0x0F, 0x40, 0x00,          // movups [0x400FF4],xmm0
            0x48, 0x89, 0x1C, 0x25, 0xFA, 0x0F, 0x40, 0x00,          // mov [0x400FFA],rbx
            0x48, 0x8B, 0x04, 0x25, 0xFC, 0x0F, 0x40, 0x00,          // mov rax,[0x400FFC]
            0xF4,
        ];
        let mut cpu = long_cpu(&code);
        cpu.cr4 |= crate::cpu::CR4_OSFXSR;
        split_map(&mut cpu);
        // The two physical pages: distinct fill patterns.
        for i in 0..0x1000 { cpu.mem.write_u8(0x70_0000 + i, 0xA0 | (i & 0xF) as u8); }
        for i in 0..0x1000 { cpu.mem.write_u8(0x90_0000 + i, 0xB0 | (i & 0xF) as u8); }
        cpu.regs[3] = 0x1122_3344_5566_7788;
        run64(&mut cpu);
        // The 16-byte load: 8 bytes from the end of the first page, 8 from
        // the start of the second (never from 0x70_1000, which is not
        // mapped here).
        assert_eq!(cpu.xmm[0], 0xB7B6_B5B4_B3B2_B1B0_AFAE_ADAC_ABAA_A9A8);
        // The 16-byte store, 12 + 4: bytes the later mov did not overwrite.
        assert_eq!(cpu.mem.read_u32(0x70_0FF4), 0xABAA_A9A8);
        assert_eq!(cpu.mem.read_u16(0x70_0FF8), 0xADAC);
        assert_eq!(cpu.mem.read_u16(0x90_0002), 0xB7B6);
        // The 8-byte store, 6 + 2.
        assert_eq!(cpu.mem.read_u16(0x70_0FFA), 0x7788);
        assert_eq!(cpu.mem.read_u16(0x90_0000), 0x1122);
        // The 8-byte load, 4 + 4, sees both stores.
        assert_eq!(cpu.regs[0], 0xB7B6_1122_3344_5566);
    }

    #[test]
    fn entering_long_mode_takes_four_steps_in_order() {
        let cpu = long_cpu(&[0xF4]);
        // CR4.PAE, CR3, EFER.LME, CR0.PG -- and LMA set by the hardware in
        // response to the last of them, not by software.
        assert_ne!(cpu.cr4 & crate::cpu::CR4_PAE, 0, "PAE");
        assert_ne!(cpu.cr0 & crate::cpu::CR0_PG, 0, "paging");
        assert_ne!(cpu.efer & crate::cpu::efer::LME, 0, "LME");
        assert_ne!(cpu.efer & crate::cpu::efer::LMA, 0, "LMA was not set by the CPU");
        assert!(cpu.long_mode() && cpu.long64());
        assert_eq!(cpu.mode(), crate::cpu::Mode::Long);
        assert_eq!(cpu.paging_mode(), crate::paging::PagingMode::Long);
        // The code segment is 64-bit: L set, D/B clear. Both set is illegal.
        let cs = cpu.seg_desc[SegReg::Cs as usize];
        assert!(cs.l && !cs.d_b);
    }

    #[test]
    fn clearing_paging_leaves_long_mode() {
        // LMA follows CR0.PG. `mov %rax,%cr0` with PG cleared drops the CPU
        // out of long mode, which is what a kernel does on the way to a
        // reboot or a 32-bit trampoline.
        let mut cpu = long_cpu(&[0xF4]);
        assert!(cpu.long_mode());
        cpu.cr0 &= !crate::cpu::CR0_PG;
        cpu.update_long_mode();
        assert!(!cpu.long_mode());
        assert_eq!(cpu.mode(), crate::cpu::Mode::Protected);
    }

    #[test]
    fn rex_w_makes_the_operand_64_bits() {
        // movabs $0x0123456789ABCDEF,%rax ; mov %rax,%rbx ; add %rax,%rbx ; hlt
        let cpu = run64_code(&[
            0x48, 0xB8, 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01,
            0x48, 0x89, 0xC3,
            0x48, 0x01, 0xC3,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(0), 0x0123_4567_89AB_CDEF);
        assert_eq!(cpu.reg64(3), 0x0123_4567_89AB_CDEFu64.wrapping_mul(2));
    }

    #[test]
    fn rex_b_reaches_the_registers_rex_added() {
        // mov $0x1234,%eax ; mov %rax,%r8 ; mov %r8,%r15 ; hlt
        let cpu = run64_code(&[
            0xB8, 0x34, 0x12, 0x00, 0x00,
            0x49, 0x89, 0xC0,
            0x4D, 0x89, 0xC7,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(8), 0x1234, "R8");
        assert_eq!(cpu.reg64(15), 0x1234, "R15");
    }

    #[test]
    fn a_32_bit_write_zero_extends_but_a_16_bit_one_does_not() {
        // The asymmetry is x86-64's, and code generated for it depends on
        // both halves: `mov $0,%eax` is the idiomatic way to clear RAX.
        // movabs $-1,%rax ; mov $1,%eax ; hlt
        let cpu = run64_code(&[
            0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0xB8, 0x01, 0x00, 0x00, 0x00,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(0), 1, "a 32-bit write must clear the high half");

        // movabs $-1,%rax ; mov $1,%ax ; hlt
        let cpu = run64_code(&[
            0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0x66, 0xB8, 0x01, 0x00,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(0), 0xFFFF_FFFF_FFFF_0001,
            "a 16-bit write must leave the bits above it alone");
    }

    #[test]
    fn rip_relative_addressing_measures_from_the_end_of_the_instruction() {
        // mov 0x9(%rip),%rax ; hlt ; <8 bytes of data>
        //
        // The displacement is from the *next* instruction, so it has to skip
        // the HLT: seven bytes of MOV, one of HLT, and the data follows.
        let mut code = vec![
            0x48, 0x8B, 0x05, 0x01, 0x00, 0x00, 0x00, // mov 1(%rip),%rax
            0xF4,                                     // hlt
        ];
        code.extend_from_slice(&0xDEAD_BEEF_CAFE_F00Du64.to_le_bytes());
        let cpu = run64_code(&code);
        assert_eq!(cpu.reg64(0), 0xDEAD_BEEF_CAFE_F00D);
    }

    #[test]
    fn lea_computes_a_rip_relative_address_without_reading_it() {
        // lea 0x10(%rip),%rax ; hlt
        let cpu = run64_code(&[
            0x48, 0x8D, 0x05, 0x10, 0x00, 0x00, 0x00,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(0), CODE64 + 7 + 0x10);
    }

    #[test]
    fn movsxd_sign_extends_a_32_bit_value() {
        // mov $-2,%ecx ; movslq %ecx,%rax ; hlt
        let cpu = run64_code(&[
            0xB9, 0xFE, 0xFF, 0xFF, 0xFF,
            0x48, 0x63, 0xC1,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(0), 0xFFFF_FFFF_FFFF_FFFE);
        // ECX itself was zero-extended by its own 32-bit write, so the value
        // MOVSXD widened came from the low half alone.
        assert_eq!(cpu.reg64(1), 0xFFFF_FFFE);
    }

    #[test]
    fn push_and_pop_are_eight_bytes_wide_and_not_overridable() {
        // movabs $0x1122334455667788,%rax ; push %rax ; pop %rbx ; hlt
        let mut cpu = long_cpu(&[
            0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
            0x50,
            0x5B,
            0xF4,
        ]);
        let rsp0 = cpu.rsp();
        cpu.run(4096);
        assert_eq!(cpu.reg64(3), 0x1122_3344_5566_7788);
        assert_eq!(cpu.rsp(), rsp0, "the stack came back to where it started");
    }

    #[test]
    fn a_call_pushes_a_64_bit_return_address() {
        // call +1 ; hlt ; mov $7,%eax ; ret
        let cpu = run64_code(&[
            0xE8, 0x01, 0x00, 0x00, 0x00, // call .+1 (past the hlt)
            0xF4,                         // hlt
            0xB8, 0x07, 0x00, 0x00, 0x00, // mov $7,%eax
            0xC3,                         // ret
        ]);
        assert_eq!(cpu.reg64(0), 7);
        assert_eq!(cpu.rip, CODE64 + 6, "returned to the instruction after the call");
    }

    #[test]
    fn shifts_take_a_six_bit_count_at_64_bit_width() {
        // A 5-bit mask would make `shl $32,%rax` a no-op, leaving the value
        // where it was instead of moving it into the high half.
        // mov $1,%eax ; shl $32,%rax ; hlt
        let cpu = run64_code(&[
            0xB8, 0x01, 0x00, 0x00, 0x00,
            0x48, 0xC1, 0xE0, 0x20,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(0), 1u64 << 32);
    }

    #[test]
    fn arithmetic_flags_are_computed_at_the_full_width() {
        // movabs $-1,%rax ; add $1,%rax ; hlt  -> zero, with a carry out.
        let cpu = run64_code(&[
            0x48, 0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
            0x48, 0x83, 0xC0, 0x01,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(0), 0);
        assert!(cpu.get_flag(flags::ZF));
        assert!(cpu.get_flag(flags::CF), "the carry out of bit 63");
        // The same addition at 32 bits would have carried out of bit 31 long
        // before, which is the bug a masked-to-32 implementation produces.
        assert!(!cpu.get_flag(flags::SF));
    }

    #[test]
    fn multiply_and_divide_use_the_full_128_bit_intermediate() {
        // movabs $0x100000000,%rax ; mov %rax,%rbx ; mul %rbx ; hlt
        // 2^32 * 2^32 = 2^64: the whole product lives in RDX.
        let cpu = run64_code(&[
            0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
            0x48, 0x89, 0xC3,
            0x48, 0xF7, 0xE3,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(0), 0, "low half");
        assert_eq!(cpu.reg64(2), 1, "high half");
        assert!(cpu.get_flag(flags::CF), "the product did not fit in RAX");
    }

    #[test]
    fn imul_sign_extends_across_the_whole_width() {
        // mov $-1,%eax ; movslq %eax,%rax ; imul $3,%rax,%rbx ; hlt
        let cpu = run64_code(&[
            0xB8, 0xFF, 0xFF, 0xFF, 0xFF,
            0x48, 0x63, 0xC0,
            0x48, 0x6B, 0xD8, 0x03,
            0xF4,
        ]);
        assert_eq!(cpu.reg64(3) as i64, -3);
    }

    #[test]
    fn string_instructions_step_64_bit_index_registers() {
        // lea 0x100(%rip),%rdi ; mov $4,%ecx ; movabs $0x1111...,%rax ;
        // rep stosq ; hlt
        let mut cpu = long_cpu(&[
            0x48, 0x8D, 0x3D, 0x00, 0x01, 0x00, 0x00,      // lea 0x100(%rip),%rdi
            0xB9, 0x04, 0x00, 0x00, 0x00,                  // mov $4,%ecx
            0x48, 0xB8, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
            0xF3, 0x48, 0xAB,                              // rep stos %rax,(%rdi)
            0xF4,
        ]);
        cpu.run(4096);
        assert!(cpu.halted);
        let base = CODE64 + 7 + 0x100;
        for i in 0..4u64 {
            assert_eq!(cpu.mem.read_u64((base + i * 8) as usize), 0x1111_1111_1111_1111);
        }
        assert_eq!(cpu.reg64(7), base + 32, "RDI advanced eight bytes per element");
        assert_eq!(cpu.reg64(1), 0, "RCX ran down to zero");
    }

    #[test]
    fn four_level_paging_reaches_the_high_half_of_the_address_space() {
        // Map linear 0xFFFF_8000_0000_0000 (PML4 entry 256) onto a 2 MiB page
        // at physical 0x40_0000, then write through it and read the physical
        // bytes back. This is the mapping every 64-bit kernel runs from.
        use crate::paging::pte;
        let mut cpu = long_cpu(&[0xF4]);
        let pml4 = cpu.cr3 as usize;
        // A fresh PDPT and PD above the identity map's tables.
        let pdpt = 0x9000usize;
        let pd = 0xA000usize;
        cpu.mem.write_u64(pml4 + 256 * 8, pdpt as u64 | pte::P | pte::RW);
        cpu.mem.write_u64(pdpt, pd as u64 | pte::P | pte::RW);
        cpu.mem.write_u64(pd, 0x40_0000 | pte::P | pte::RW | pte::PS);
        cpu.flush_tlb();

        let high: u64 = 0xFFFF_8000_0000_0000;
        let phys = cpu.apply_paging(high + 0x1234);
        assert!(cpu.pending_exception.is_none(), "the high half must translate");
        assert_eq!(phys, 0x40_0000 + 0x1234);

        // And it is writable through the CPU's own store path.
        let addr = cpu.translate_write(SegReg::Ds, high + 8);
        cpu.mem.write_u64(addr, 0xFEED_FACE_1234_5678);
        assert_eq!(cpu.mem.read_u64(0x40_0000 + 8), 0xFEED_FACE_1234_5678);
    }

    #[test]
    fn a_non_canonical_address_is_a_general_protection_fault() {
        // The unused middle of the 64-bit address space is a hole, not an
        // alias: reaching into it is a #GP before the page tables are even
        // consulted, which is what stops a 48-bit machine from pretending to
        // have 64 bits of address space.
        let mut cpu = long_cpu(&[0xF4]);
        let phys = cpu.apply_paging(0x0000_8000_0000_0000);
        assert_eq!(cpu.pending_exception, Some((0x0D, Some(0))));
        assert_eq!(phys, crate::memory::UNBACKED);
        // A canonical address gets past this check and on to the page tables,
        // where being unmapped is an ordinary page fault -- a different
        // exception, reported differently, and fixable by a handler.
        cpu.pending_exception = None;
        cpu.apply_paging(0xFFFF_8000_0000_0000);
        assert_eq!(cpu.pending_exception.unwrap().0, 0x0E);
        // And a canonical address that *is* mapped simply works.
        cpu.pending_exception = None;
        assert_eq!(cpu.apply_paging(0x1234), 0x1234);
        assert!(cpu.pending_exception.is_none());
    }

    #[test]
    fn no_execute_faults_on_a_fetch_and_only_on_a_fetch() {
        // NX has to be checked on the fetch path alone: a no-execute page is
        // still perfectly readable, and faulting on the read would break
        // every string constant a kernel keeps in its data segment.
        use crate::paging::pte;
        let mut cpu = long_cpu(&[0xF4]);
        // The identity map's second 2 MiB page, marked no-execute.
        let pd = 0x4000usize;
        let e = cpu.mem.read_u64(pd + 8);
        cpu.mem.write_u64(pd + 8, e | pte::NX);
        cpu.flush_tlb();

        let target = 0x20_0000u64;
        // A read is fine.
        let phys = cpu.apply_paging(target);
        assert!(cpu.pending_exception.is_none(), "a read of an NX page must work");
        assert_eq!(phys, target as usize);
        // A fetch is not.
        let _ = cpu.apply_paging_fetch(target);
        let (vector, code) = cpu.pending_exception.expect("fetch from NX must fault");
        assert_eq!(vector, 0x0E);
        assert_eq!(code.unwrap() & (1 << 4), 1 << 4, "the I/D bit says it was a fetch");
        assert_eq!(cpu.cr2, target);
    }

    #[test]
    fn cpuid_reports_long_mode() {
        // Bit 29 of leaf 0x80000001's EDX is the long-mode bit, and reaching
        // that leaf at all requires 0x80000000 to answer with a high enough
        // maximum. A bootloader checks both, in that order.
        let mut cpu = long_cpu(&[0x0F, 0xA2, 0xF4]);
        cpu.set_reg32(Reg32::Eax, 0x8000_0000);
        cpu.run(8);
        assert!(cpu.reg32(Reg32::Eax) >= 0x8000_0001);

        let mut cpu = long_cpu(&[0x0F, 0xA2, 0xF4]);
        cpu.set_reg32(Reg32::Eax, 0x8000_0001);
        cpu.run(8);
        assert_ne!(cpu.reg32(Reg32::Edx) & (1 << 29), 0, "LM");
        assert_ne!(cpu.reg32(Reg32::Edx) & (1 << 20), 0, "NX");
        assert_ne!(cpu.reg32(Reg32::Edx) & (1 << 11), 0, "SYSCALL");
        // And PAE, in the basic leaf: a 64-bit boot will not start without it.
        let mut cpu = long_cpu(&[0x0F, 0xA2, 0xF4]);
        cpu.set_reg32(Reg32::Eax, 1);
        cpu.run(8);
        assert_ne!(cpu.reg32(Reg32::Edx) & (1 << 6), 0, "PAE");
    }

    #[test]
    fn efer_round_trips_through_wrmsr_but_lma_is_the_cpus_to_set() {
        // mov $0xC0000080,%ecx ; rdmsr ; hlt
        let mut cpu = long_cpu(&[0xB9, 0x80, 0x00, 0x00, 0xC0, 0x0F, 0x32, 0xF4]);
        cpu.run(16);
        let efer = (cpu.reg32(Reg32::Eax) as u64) | ((cpu.reg32(Reg32::Edx) as u64) << 32);
        assert_eq!(efer, cpu.efer);
        assert_ne!(efer & crate::cpu::efer::LMA, 0);

        // Software cannot clear LMA by writing EFER: it follows CR0.PG.
        let mut cpu = long_cpu(&[0xF4]);
        cpu.write_msr(crate::cpu::msr::EFER, crate::cpu::efer::LME);
        assert_ne!(cpu.efer & crate::cpu::efer::LMA, 0,
            "LMA must not be clearable by a plain EFER write");
    }

    #[test]
    fn fs_and_gs_bases_come_from_msrs_and_actually_offset_an_access() {
        // In 64-bit mode FS and GS are the only segments with a base, and it
        // is set through an MSR because it no longer fits in a descriptor.
        let mut cpu = long_cpu(&[0x64, 0x48, 0x8B, 0x04, 0x25, 0x00, 0x00, 0x00, 0x00, 0xF4]);
        // ^ mov %fs:0x0,%rax
        cpu.write_msr(crate::cpu::msr::FS_BASE, 0x30_0000);
        cpu.mem.write_u64(0x30_0000, 0xABCD_1234_5678_9EF0);
        cpu.run(16);
        assert_eq!(cpu.reg64(0), 0xABCD_1234_5678_9EF0);
    }

    #[test]
    fn swapgs_exchanges_the_two_gs_bases() {
        // swapgs ; swapgs ; hlt -- one swap moves the kernel base into GS,
        // the second puts it back, which is the pattern at every kernel entry
        // and exit.
        let mut cpu = long_cpu(&[0x0F, 0x01, 0xF8, 0x0F, 0x01, 0xF8, 0xF4]);
        cpu.gs_base = 0x1111_0000;
        cpu.kernel_gs_base = 0x2222_0000;
        // Step one instruction at a time so the intermediate state is visible.
        cpu.step();
        assert_eq!(cpu.gs_base, 0x2222_0000);
        assert_eq!(cpu.kernel_gs_base, 0x1111_0000);
        cpu.step();
        assert_eq!(cpu.gs_base, 0x1111_0000);
        assert_eq!(cpu.kernel_gs_base, 0x2222_0000);
    }

    /// Add the user code/data descriptors SYSRET expects to the long-mode GDT.
    fn add_user_descriptors(cpu: &mut Cpu) {
        let g = crate::boot::GDT64_ADDR as usize;
        // 0x18: 32-bit user code (the SYSRET base; unused by the 64-bit form)
        cpu.mem.write_u64(g + 0x18, 0x00CF_FA00_0000_FFFF);
        // 0x20: user data
        cpu.mem.write_u64(g + 0x20, 0x00CF_F200_0000_FFFF);
        // 0x28: 64-bit user code (L set, DPL 3)
        cpu.mem.write_u64(g + 0x28, 0x00AF_FA00_0000_FFFF);
        cpu.gdt_limit = 0x2F;
    }

    #[test]
    fn syscall_and_sysret_round_trip_through_the_msrs() {
        // syscall ; hlt ; <handler> mov $0x99,%eax ; sysretq
        let handler = CODE64 + 0x40;
        let mut code = vec![
            0x0F, 0x05, // syscall
            0xF4,       // hlt
        ];
        code.resize(0x40, 0x90);
        code.extend_from_slice(&[
            0xB8, 0x99, 0x00, 0x00, 0x00, // mov $0x99,%eax
            0x48, 0x0F, 0x07,             // sysretq
        ]);
        let mut cpu = long_cpu(&code);
        add_user_descriptors(&mut cpu);
        cpu.write_msr(crate::cpu::msr::LSTAR, handler);
        // STAR: kernel CS in 47:32, the SYSRET selector base in 63:48.
        cpu.write_msr(crate::cpu::msr::STAR, (0x08u64 << 32) | (0x18u64 << 48));
        // SYSCALL clears IF through SFMASK, as a kernel entry must.
        cpu.write_msr(crate::cpu::msr::SFMASK, flags::IF as u64);
        cpu.set_flag(flags::IF, true);

        let expect_return = CODE64 + 2;
        cpu.step(); // syscall
        assert_eq!(cpu.rip, handler, "landed at LSTAR");
        assert_eq!(cpu.reg64(1), expect_return, "RCX carries the return address");
        assert_eq!(cpu.cs, 0x08, "kernel CS from STAR");
        assert_eq!(cpu.ss, 0x10, "kernel SS is the next descriptor");
        assert!(!cpu.get_flag(flags::IF), "SFMASK cleared IF");
        assert!(cpu.long64(), "still 64-bit code");

        cpu.step(); // mov $0x99,%eax
        cpu.step(); // sysretq
        assert_eq!(cpu.reg64(0), 0x99);
        assert_eq!(cpu.rip, expect_return, "SYSRET returned to RCX");
        assert!(cpu.get_flag(flags::IF), "R11 carried the flags back");
        assert_eq!(cpu.cs, 0x2B, "user CS with RPL 3");
        assert_eq!(cpu.ss, 0x23, "user SS with RPL 3");
        assert!(cpu.long64(), "the user segment is 64-bit too");
    }

    #[test]
    fn syscall_without_efer_sce_is_an_invalid_opcode() {
        let mut cpu = long_cpu(&[0x0F, 0x05, 0xF4]);
        cpu.efer &= !crate::cpu::efer::SCE;
        cpu.step();
        assert_eq!(cpu.pending_exception, Some((0x06, None)));
    }

    /// Install a 64-bit interrupt gate: sixteen bytes, offset in three
    /// pieces, with an IST index sharing a byte with the reserved field.
    fn install_gate64(cpu: &mut Cpu, idt: u64, vector: u8, handler: u64, ist: u8) {
        let e = idt as usize + (vector as usize) * 16;
        cpu.mem.write_u16(e, handler as u16);
        cpu.mem.write_u16(e + 2, 0x08); // kernel CS
        cpu.mem.write_u8(e + 4, ist & 7);
        cpu.mem.write_u8(e + 5, 0x8E); // present, DPL 0, interrupt gate
        cpu.mem.write_u16(e + 6, (handler >> 16) as u16);
        cpu.mem.write_u32(e + 8, (handler >> 32) as u32);
        cpu.mem.write_u32(e + 12, 0);
        cpu.idt_base = idt;
        cpu.idt_limit = 0xFFF;
    }

    #[test]
    fn an_interrupt_in_long_mode_uses_a_16_byte_gate_and_iretq_returns() {
        // int $0x80 ; hlt ; <handler> mov $0x55,%eax ; iretq
        let handler = CODE64 + 0x40;
        let mut code = vec![0xCD, 0x80, 0xF4];
        code.resize(0x40, 0x90);
        code.extend_from_slice(&[
            0xB8, 0x55, 0x00, 0x00, 0x00, // mov $0x55,%eax
            0x48, 0xCF,                   // iretq
        ]);
        let mut cpu = long_cpu(&code);
        install_gate64(&mut cpu, 0xB000, 0x80, handler, 0);
        let rsp0 = cpu.rsp();

        cpu.step(); // int $0x80
        assert_eq!(cpu.rip, handler, "vectored through the 16-byte gate");
        // The frame is five 8-byte words: RIP, CS, RFLAGS, RSP, SS -- SS and
        // RSP are pushed even though the privilege level did not change.
        let sp = cpu.rsp() as usize;
        assert_eq!(cpu.mem.read_u64(sp), CODE64 + 2, "saved RIP");
        assert_eq!(cpu.mem.read_u64(sp + 8), 0x08, "saved CS");
        assert_eq!(cpu.mem.read_u64(sp + 24), rsp0, "saved RSP");
        assert!(!cpu.get_flag(flags::IF), "an interrupt gate clears IF");

        cpu.run(16);
        assert!(cpu.halted);
        assert_eq!(cpu.reg64(0), 0x55);
        assert_eq!(cpu.rip, CODE64 + 3, "IRETQ returned past the INT");
        assert_eq!(cpu.rsp(), rsp0, "IRETQ restored the stack pointer");
    }

    #[test]
    fn a_gate_with_an_ist_index_switches_to_the_table_stack() {
        // The IST is what makes a fault on a broken stack survivable: the
        // gate names a stack unconditionally, with no privilege change
        // needed. Without it, a double fault has nowhere to land.
        let handler = CODE64 + 0x40;
        let mut code = vec![0xCD, 0x80, 0xF4];
        code.resize(0x40, 0x90);
        code.extend_from_slice(&[0xF4]);
        let mut cpu = long_cpu(&code);
        install_gate64(&mut cpu, 0xB000, 0x80, handler, 1);
        // A 64-bit TSS at 0xC000 with IST1 pointing at a known stack.
        let tss = 0xC000u64;
        cpu.tr_base = tss;
        cpu.mem.write_u64(tss as usize + 0x24, 0x7_0000); // IST1
        cpu.step();
        assert_eq!(cpu.rip, handler);
        // The frame landed on the IST stack, not the one it came in on.
        assert_eq!(cpu.rsp(), 0x7_0000 - 40);
    }

    #[test]
    fn a_page_fault_in_long_mode_records_a_64_bit_cr2() {
        // CR2 has to be 64 bits wide, or a fault in the high half of the
        // address space reports an address in the low half -- and the
        // handler fixes up the wrong page.
        let mut cpu = long_cpu(&[0xF4]);
        let bad: u64 = 0xFFFF_8800_1234_5000;
        let _ = cpu.apply_paging(bad);
        assert_eq!(cpu.pending_exception.unwrap().0, 0x0E);
        assert_eq!(cpu.cr2, bad);
    }

    #[test]
    fn the_tlb_caches_a_64_bit_translation_and_invlpg_drops_it() {
        use crate::paging::pte;
        let mut cpu = long_cpu(&[0xF4]);
        let pml4 = cpu.cr3 as usize;
        let pdpt = 0x9000usize;
        let pd = 0xA000usize;
        cpu.mem.write_u64(pml4 + 256 * 8, pdpt as u64 | pte::P | pte::RW);
        cpu.mem.write_u64(pdpt, pd as u64 | pte::P | pte::RW);
        cpu.mem.write_u64(pd, 0x40_0000 | pte::P | pte::RW | pte::PS);
        cpu.flush_tlb();

        let high: u64 = 0xFFFF_8000_0000_0000;
        assert_eq!(cpu.apply_paging(high), 0x40_0000);
        // Repoint the mapping without telling the CPU: the TLB still answers
        // with the old translation, which is the whole point of INVLPG.
        cpu.mem.write_u64(pd, 0x60_0000 | pte::P | pte::RW | pte::PS);
        assert_eq!(cpu.apply_paging(high), 0x40_0000, "served from the TLB");
        cpu.invlpg(high);
        assert_eq!(cpu.apply_paging(high), 0x60_0000, "re-walked after INVLPG");
    }

    #[test]
    fn ram_above_four_gib_is_reachable_through_the_page_tables() {
        // The point of the whole exercise: a machine with more RAM than fits
        // below the MMIO hole, addressed from 64-bit code.
        use crate::paging::pte;
        let ram = crate::memory::MMIO_HOLE_START as usize + (32 << 20);
        let mut cpu = Cpu::with_ram(ram);
        crate::boot::load_flat64(&mut cpu, &[0xF4], CODE64).unwrap();
        let high_phys = crate::memory::HIGH_RAM_BASE + 0x2_0000;

        // Map linear 0xFFFF_8000_0000_0000 onto a 2 MiB page up there.
        let pml4 = cpu.cr3 as usize;
        cpu.mem.write_u64(pml4 + 256 * 8, 0x9000u64 | pte::P | pte::RW);
        cpu.mem.write_u64(0x9000, 0xA000u64 | pte::P | pte::RW);
        cpu.mem.write_u64(0xA000, (crate::memory::HIGH_RAM_BASE) | pte::P | pte::RW | pte::PS);
        cpu.flush_tlb();

        let linear = 0xFFFF_8000_0000_0000u64 + 0x2_0000;
        let phys = cpu.translate_write(SegReg::Ds, linear);
        assert!(cpu.pending_exception.is_none());
        assert_eq!(phys as u64, high_phys);
        cpu.mem.write_u64(phys, 0x0BAD_C0DE_0BAD_C0DE);
        assert_eq!(cpu.mem.read_u64(high_phys as usize), 0x0BAD_C0DE_0BAD_C0DE);
        // And the machine says so in its memory map.
        let map = cpu.mem.e820();
        assert!(map.iter().any(|e| e.0 == crate::memory::HIGH_RAM_BASE && e.2 == 1));
    }

    #[test]
    fn compatibility_mode_runs_32_bit_code_under_a_64_bit_machine() {
        // Long mode with a code segment whose L bit is clear: the machine is
        // still in long mode (LMA set, 4-level paging) but the code runs with
        // 32-bit defaults. This is how a 64-bit kernel runs a 32-bit process.
        let mut cpu = long_cpu(&[0xF4]);
        // Install a 32-bit code descriptor and load it.
        let g = crate::boot::GDT64_ADDR as usize;
        cpu.mem.write_u64(g + 0x18, 0x00CF_9A00_0000_FFFF); // D/B set, L clear
        cpu.gdt_limit = 0x1F;
        cpu.load_seg(SegReg::Cs, 0x18);
        assert_eq!(cpu.mode(), crate::cpu::Mode::Compat);
        assert!(cpu.long_mode(), "the machine is still in long mode");
        assert!(!cpu.long64(), "but this code segment is not 64-bit");
        // Paging is still the 4-level kind, which is what makes it long mode.
        assert_eq!(cpu.paging_mode(), crate::paging::PagingMode::Long);
    }

    #[test]
    fn pae_paging_works_without_long_mode() {
        // The middle of the three paging modes: 8-byte entries and three
        // levels, but a 32-bit linear address. A 32-bit kernel with more than
        // 4 GiB of RAM runs here.
        use crate::paging::pte;
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cr4 |= crate::cpu::CR4_PAE;
        // PDPT at 0x1000 -> PD 0x2000 -> a 2 MiB page at 0x80_0000.
        cpu.mem.write_u64(0x1000, 0x2000u64 | pte::P);
        cpu.mem.write_u64(0x2000, 0x80_0000u64 | pte::P | pte::RW | pte::PS);
        cpu.cr3 = 0x1000;
        cpu.cr0 |= crate::cpu::CR0_PG;
        cpu.flush_tlb();
        assert_eq!(cpu.paging_mode(), crate::paging::PagingMode::Pae);
        assert_eq!(cpu.apply_paging(0x1234), 0x80_0000 + 0x1234);
        assert!(cpu.pending_exception.is_none());
    }

    #[test]
    fn a_halt_with_interrupts_disabled_really_stops() {
        // ... but with IF clear nothing can ever wake it, so the run ends
        // rather than spinning to the instruction limit.
        let mut cpu = Cpu::new();
        cpu.mem.load(0, &[0xFA, 0xF4]); // cli ; hlt
        let ran = cpu.run(1_000_000);
        assert!(cpu.halted);
        assert!(ran < 10, "ran {} instructions", ran);
    }
}
