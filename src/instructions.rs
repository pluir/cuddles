//! Instruction decoder and executor.
//!
//! Supports the 8086 real-mode instruction set plus 32-bit protected-mode
//! extensions: the 0x66/0x67 size-override prefixes, 32-bit register and
//! addressing forms, LGDT/LIDT, and protected-mode interrupt dispatch
//! through the IDT.

use crate::cpu::{Cpu, Reg8, Reg16, Reg32, SegReg, flags};
use crate::modrm::{ModRm, Reg};

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
    MovReg8Imm { dst: Reg8, imm: u8 },
    MovReg16Imm { dst: Reg16, imm: u16 },
    MovReg32Imm { dst: Reg32, imm: u32 },
    MovAccMem8 { addr: u16 },
    MovMem8Acc { addr: u16 },
    MovAccMem8Addr32 { addr: u32 },
    MovMem8AccAddr32 { addr: u32 },
    MovAccMem16 { addr: u16 },
    MovMem16Acc { addr: u16 },
    MovAccMem16Addr32 { addr: u32 },
    MovMem16AccAddr32 { addr: u32 },
    MovAccMem32 { addr: u32 },
    MovMem32Acc { addr: u32 },
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
    IncReg16 { dst: Reg16 },
    DecReg16 { dst: Reg16 },
    IncReg32 { dst: Reg32 },
    DecReg32 { dst: Reg32 },
    PushReg16 { src: Reg16 },
    PopReg16 { dst: Reg16 },
    PushReg32 { src: Reg32 },
    PopReg32 { dst: Reg32 },
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
    CallRel16 { rel: i16 },
    CallRel32 { rel: i32 },
    Ret,
    Ret32,
    // RET imm16 (0xC2): return, then drop `imm` bytes of arguments.
    RetImm { imm: u16, w32: bool },
    XchgAxReg { reg: Reg16 },
    XchgEaxReg { reg: Reg32 },
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
    // LGDT / LIDT (0x0F 0x01 /2 and /3)
    Lgdt { m: ModRm },
    Lidt { m: ModRm },
    // INVLPG (0x0F 0x01 /7): invalidate TLB entry for a linear address.
    Invlpg { m: ModRm },
    // MOV r32, cr (0x0F 0x20) / MOV cr, r32 (0x0F 0x22)
    MovCr { cr: u8, reg: u8 },
    MovToCr { cr: u8, reg: u8 },
    // CLTS (0x0F 0x06): clear CR0.TS (task-switched flag).
    Clts,
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
    // FXSAVE (0F AE /0) / FXRSTOR (0F AE /1)
    Fxsave { m: ModRm },
    Fxrstor { m: ModRm },
    Unknown { opcode: u16 },
}

/// x87 arithmetic operation (simplified: ST0 op m).
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
    // runs after decode returns). The default operand/address size comes
    // from the code segment: in protected mode a D=1 code segment defaults
    // to 32-bit operands and addressing; otherwise the default is 16-bit.
    // The 0x66/0x67 prefixes then *toggle* the size.
    let d32 = cpu.pe && cpu.seg_desc[SegReg::Cs as usize].d_b;
    cpu.opsize = d32;
    cpu.addrsize = d32;
    cpu.seg_override = None;
    // Handle prefixes: REP/REPNE (0xF3/0xF2), operand-size (0x66),
    // address-size (0x67), and segment overrides.
    let mut rep = Rep::None;
    loop {
        let peek = cpu.peek_u8();
        match peek {
            // LOCK. This is a uniprocessor emulator with no concurrent bus
            // master, so the prefix is consumed and the instruction runs
            // normally -- but it must be consumed, or `lock cmpxchg` decodes
            // 0xF0 as an opcode and faults.
            0xF0 => { cpu.fetch_u8(); }
            0xF3 => { rep = Rep::Repe; cpu.fetch_u8(); }
            0xF2 => { rep = Rep::Repne; cpu.fetch_u8(); }
            0x66 => { cpu.opsize = !cpu.opsize; cpu.fetch_u8(); }
            0x67 => { cpu.addrsize = !cpu.addrsize; cpu.fetch_u8(); }
            0x2E => { cpu.seg_override = Some(SegReg::Cs); cpu.fetch_u8(); }
            0x36 => { cpu.seg_override = Some(SegReg::Ss); cpu.fetch_u8(); }
            0x3E => { cpu.seg_override = Some(SegReg::Ds); cpu.fetch_u8(); }
            0x26 => { cpu.seg_override = Some(SegReg::Es); cpu.fetch_u8(); }
            0x64 => { cpu.seg_override = Some(SegReg::Fs); cpu.fetch_u8(); }
            0x65 => { cpu.seg_override = Some(SegReg::Gs); cpu.fetch_u8(); }
            _ => break,
        }
    }
    let op = cpu.fetch_u8();
    decode_op(cpu, op, rep)
}

fn decode_op(cpu: &mut Cpu, op: u8, rep: Rep) -> Inst {
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
        0x40..=0x47 => { if w32 { Inst::IncReg32 { dst: Reg::reg32(op - 0x40) } } else { Inst::IncReg16 { dst: Reg::reg16(op - 0x40) } } }
        0x48..=0x4F => { if w32 { Inst::DecReg32 { dst: Reg::reg32(op - 0x48) } } else { Inst::DecReg16 { dst: Reg::reg16(op - 0x48) } } }
        // PUSH reg (0x50-0x57) / POP reg (0x58-0x5F)
        0x50..=0x57 => { if w32 { Inst::PushReg32 { src: Reg::reg32(op - 0x50) } } else { Inst::PushReg16 { src: Reg::reg16(op - 0x50) } } }
        0x58..=0x5F => { if w32 { Inst::PopReg32 { dst: Reg::reg32(op - 0x58) } } else { Inst::PopReg16 { dst: Reg::reg16(op - 0x58) } } }

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
        0x87 => { let m = cpu.fetch_modrm(); Inst::XchgRmReg { m, reg: m.reg, width: if w32 { 32 } else { 16 } } }

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
        0x90 => Inst::Nop,
        0x91..=0x97 => { if w32 { Inst::XchgEaxReg { reg: Reg::reg32(op - 0x90) } } else { Inst::XchgAxReg { reg: Reg::reg16(op - 0x90) } } }

        // CBW (0x98) / CWD (0x99) / CWDE / CDQ
        0x98 => { if w32 { Inst::Cwde } else { Inst::Cbw } }
        0x99 => { if w32 { Inst::Cdq } else { Inst::Cwd } }

        // PUSHF (0x9C) / POPF (0x9D)
        0x9C => Inst::Pushf,
        0x9D => Inst::Popf,

        // MOV reg8, imm8 (0xB0-0xB7)
        0xB0..=0xB7 => Inst::MovReg8Imm { dst: Reg::reg8(op - 0xB0), imm: cpu.fetch_u8() },
        // MOV reg16/32, imm (0xB8-0xBF)
        0xB8..=0xBF => {
            if w32 { Inst::MovReg32Imm { dst: Reg::reg32(op - 0xB8), imm: cpu.fetch_u32() } }
            else { Inst::MovReg16Imm { dst: Reg::reg16(op - 0xB8), imm: cpu.fetch_u16() } }
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
        0xD1 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, width: if w32 { 32 } else { 16 }, count: ShiftCount::One } }
        0xD2 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, width: 8, count: ShiftCount::Cl } }
        0xD3 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, width: if w32 { 32 } else { 16 }, count: ShiftCount::Cl } }
        // Group 2 shifts/rotates with imm8 count: 0xC0 (r/m8, imm8),
        // 0xC1 (r/m16/32, imm8)
        0xC0 => { let m = cpu.fetch_modrm(); let imm = cpu.fetch_u8(); Inst::ShiftImm { op: ShiftOp::from_index(m.reg), m, width: 8, imm } }
        0xC1 => { let m = cpu.fetch_modrm(); let imm = cpu.fetch_u8(); Inst::ShiftImm { op: ShiftOp::from_index(m.reg), m, width: if w32 { 32 } else { 16 }, imm } }

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
            if cpu.addrsize {
                if w32 { Inst::MovAccMem32 { addr: cpu.fetch_u32() } }
                else { Inst::MovAccMem16Addr32 { addr: cpu.fetch_u32() } }
            } else {
                if w32 { Inst::MovAccMem32 { addr: cpu.fetch_u16() as u32 } }
                else { Inst::MovAccMem16 { addr: cpu.fetch_u16() } }
            }
        }
        0xA3 => {
            if cpu.addrsize {
                if w32 { Inst::MovMem32Acc { addr: cpu.fetch_u32() } }
                else { Inst::MovMem16AccAddr32 { addr: cpu.fetch_u32() } }
            } else {
                if w32 { Inst::MovMem32Acc { addr: cpu.fetch_u16() as u32 } }
                else { Inst::MovMem16Acc { addr: cpu.fetch_u16() } }
            }
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
                    match m.reg & 7 {
                        2 => Inst::Lgdt { m },
                        3 => Inst::Lidt { m },
                        7 => Inst::Invlpg { m },
                        _ => Inst::Unknown { opcode: 0x0F00 | op2 as u16 },
                    }
                }
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
                // MOV r32, cr (0x0F 0x20) / MOV cr, r32 (0x0F 0x22)
                0x20 => {
                    let m = cpu.fetch_modrm();
                    Inst::MovCr { cr: m.reg & 7, reg: m.rm & 7 }
                }
                0x22 => {
                    let m = cpu.fetch_modrm();
                    Inst::MovToCr { cr: m.reg & 7, reg: m.rm & 7 }
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
                0x21 => { let m = cpu.fetch_modrm(); Inst::MovDr { dr: m.reg & 7, reg: m.rm & 7 } }
                0x23 => { let m = cpu.fetch_modrm(); Inst::MovToDr { dr: m.reg & 7, reg: m.rm & 7 } }
                // Bit scan forward / reverse (0F BC / 0F BD)
                0xBC => { let m = cpu.fetch_modrm(); Inst::Bsf { m, dst: m.reg, w32 } }
                0xBD => { let m = cpu.fetch_modrm(); Inst::Bsr { m, dst: m.reg, w32 } }
                // CMPXCHG (0F B0 / 0F B1)
                0xB0 => { let m = cpu.fetch_modrm(); Inst::Cmpxchg { m, reg: m.reg, width: 8 } }
                0xB1 => { let m = cpu.fetch_modrm(); Inst::Cmpxchg { m, reg: m.reg, width: if w32 { 32 } else { 16 } } }
                // XADD (0F C0 / 0F C1)
                0xC0 => { let m = cpu.fetch_modrm(); Inst::Xadd { m, reg: m.reg, width: 8 } }
                0xC1 => { let m = cpu.fetch_modrm(); Inst::Xadd { m, reg: m.reg, width: if w32 { 32 } else { 16 } } }
                // CMPXCHG8B m64 (0F C7 /1)
                0xC7 => {
                    let m = cpu.fetch_modrm();
                    if m.reg & 7 == 1 { Inst::Cmpxchg8b { m } }
                    else { Inst::Unknown { opcode: 0x0FC7 } }
                }
                // BSWAP r32 (0F C8+r)
                0xC8..=0xCF => Inst::Bswap { reg: op2 - 0xC8 },
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
                    match m.reg & 7 {
                        0 => Inst::Fxsave { m },
                        1 => Inst::Fxrstor { m },
                        _ => Inst::Nop,
                    }
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
        Inst::Nop => {}
        Inst::Hlt => { cpu.halted = true; }

        // ---- MOV ----
        Inst::MovRm8Reg { m, src } => {
            let v = cpu.reg8(Reg::reg8(src));
            cpu.write_rm8(&m, v);
        }
        Inst::MovRm16Reg { m, src } => {
            let v = cpu.reg16(Reg::reg16(src));
            cpu.write_rm16(&m, v);
        }
        Inst::MovRm32Reg { m, src } => {
            let v = cpu.reg32(Reg::reg32(src));
            cpu.write_rm32(&m, v);
        }
        Inst::MovRegRm8 { m, dst } => {
            let v = cpu.read_rm8(&m);
            cpu.set_reg8(Reg::reg8(dst), v);
        }
        Inst::MovRegRm16 { m, dst } => {
            let v = cpu.read_rm16(&m);
            cpu.set_reg16(Reg::reg16(dst), v);
        }
        Inst::MovRegRm32 { m, dst } => {
            let v = cpu.read_rm32(&m);
            cpu.set_reg32(Reg::reg32(dst), v);
        }
        Inst::MovRm8Imm { m, imm } => cpu.write_rm8(&m, imm),
        Inst::MovRm16Imm { m, imm } => cpu.write_rm16(&m, imm),
        Inst::MovRm32Imm { m, imm } => cpu.write_rm32(&m, imm),
        Inst::MovReg8Imm { dst, imm } => cpu.set_reg8(dst, imm),
        Inst::MovReg16Imm { dst, imm } => cpu.set_reg16(dst, imm),
        Inst::MovReg32Imm { dst, imm } => cpu.set_reg32(dst, imm),
        Inst::MovAccMem8 { addr } => {
            let phys = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), addr as u32);
            cpu.set_reg8(Reg8::Al, cpu.mem.read_u8(phys));
        }
        Inst::MovAccMem8Addr32 { addr } => {
            let phys = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), addr);
            cpu.set_reg8(Reg8::Al, cpu.mem.read_u8(phys));
        }
        Inst::MovMem8Acc { addr } => {
            let phys = cpu.translate_write(cpu.operand_seg_for_exec(SegReg::Ds), addr as u32);
            cpu.mem.write_u8(phys, cpu.reg8(Reg8::Al));
        }
        Inst::MovMem8AccAddr32 { addr } => {
            let phys = cpu.translate_write(SegReg::Ds, addr);
            cpu.mem.write_u8(phys, cpu.reg8(Reg8::Al));
        }
        Inst::MovAccMem16 { addr } => {
            let phys = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), addr as u32);
            cpu.set_reg16(Reg16::Ax, cpu.mem.read_u16(phys));
        }
        Inst::MovAccMem16Addr32 { addr } => {
            let phys = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), addr);
            cpu.set_reg16(Reg16::Ax, cpu.mem.read_u16(phys));
        }
        Inst::MovMem16Acc { addr } => {
            let phys = cpu.translate_write(cpu.operand_seg_for_exec(SegReg::Ds), addr as u32);
            cpu.mem.write_u16(phys, cpu.reg16(Reg16::Ax));
        }
        Inst::MovMem16AccAddr32 { addr } => {
            let phys = cpu.translate_write(SegReg::Ds, addr);
            cpu.mem.write_u16(phys, cpu.reg16(Reg16::Ax));
        }
        Inst::MovAccMem32 { addr } => {
            let phys = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), addr);
            cpu.set_reg32(Reg32::Eax, cpu.mem.read_u32(phys));
        }
        Inst::MovMem32Acc { addr } => {
            let phys = cpu.translate_write(cpu.operand_seg_for_exec(SegReg::Ds), addr);
            cpu.mem.write_u32(phys, cpu.reg32(Reg32::Eax));
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
            let regv = cpu.reg8(Reg::reg8(reg));
            let rmv = cpu.read_rm8(&m);
            let (a, b, store) = match dir {
                Dir::RmReg => (rmv, regv, true),
                Dir::RegRm => (regv, rmv, false),
            };
            let result = alu8(cpu, op, a, b);
            // CMP sets flags only — it must never write its result back.
            if op != AluOp::Cmp {
                if store { cpu.write_rm8(&m, result); } else { cpu.set_reg8(Reg::reg8(reg), result); }
            }
        }
        Inst::AluRm16Reg { op, m, reg, dir } => {
            let regv = cpu.reg16(Reg::reg16(reg));
            let rmv = cpu.read_rm16(&m);
            let (a, b, store) = match dir {
                Dir::RmReg => (rmv, regv, true),
                Dir::RegRm => (regv, rmv, false),
            };
            let result = alu16(cpu, op, a, b);
            // CMP sets flags only — it must never write its result back.
            if op != AluOp::Cmp {
                if store { cpu.write_rm16(&m, result); } else { cpu.set_reg16(Reg::reg16(reg), result); }
            }
        }
        Inst::AluRm32Reg { op, m, reg, dir } => {
            let regv = cpu.reg32(Reg::reg32(reg));
            let rmv = cpu.read_rm32(&m);
            let (a, b, store) = match dir {
                Dir::RmReg => (rmv, regv, true),
                Dir::RegRm => (regv, rmv, false),
            };
            let result = alu32(cpu, op, a, b);
            // CMP sets flags only — it must never write its result back.
            if op != AluOp::Cmp {
                if store { cpu.write_rm32(&m, result); } else { cpu.set_reg32(Reg::reg32(reg), result); }
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
        Inst::AluRm32Imm { op, m, imm, .. } => {
            let rmv = cpu.read_rm32(&m);
            let result = alu32(cpu, op, rmv, imm);
            if op != AluOp::Cmp { cpu.write_rm32(&m, result); }
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
            let a = cpu.reg32(Reg32::Eax);
            let result = alu32(cpu, op, a, imm);
            if op != AluOp::Cmp { cpu.set_reg32(Reg32::Eax, result); }
        }

        // ---- INC / DEC ----
        Inst::IncReg16 { dst } => {
            let v = cpu.reg16(dst);
            let result = v.wrapping_add(1);
            cpu.set_reg16(dst, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags16(cpu, result);
            set_add_carry(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }
        Inst::DecReg16 { dst } => {
            let v = cpu.reg16(dst);
            let result = v.wrapping_sub(1);
            cpu.set_reg16(dst, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags16(cpu, result);
            set_sub_borrow(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }
        Inst::IncReg32 { dst } => {
            let v = cpu.reg32(dst);
            let result = v.wrapping_add(1);
            cpu.set_reg32(dst, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags32(cpu, result);
            set_add_carry32(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }
        Inst::DecReg32 { dst } => {
            let v = cpu.reg32(dst);
            let result = v.wrapping_sub(1);
            cpu.set_reg32(dst, result);
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
                let esp = cpu.esp;
                cpu.push32(cpu.eax); cpu.push32(cpu.ecx);
                cpu.push32(cpu.edx); cpu.push32(cpu.ebx);
                cpu.push32(esp);     cpu.push32(cpu.ebp);
                cpu.push32(cpu.esi); cpu.push32(cpu.edi);
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
        Inst::PushReg16 { src } => cpu.push16(cpu.reg16(src)),
        Inst::PopReg16 { dst } => { let v = cpu.pop16(); cpu.set_reg16(dst, v); }
        Inst::PushReg32 { src } => cpu.push32(cpu.reg32(src)),
        Inst::PopReg32 { dst } => { let v = cpu.pop32(); cpu.set_reg32(dst, v); }
        Inst::PushImm16 { imm } => cpu.push16(imm),
        Inst::PushImm32 { imm } => cpu.push32(imm),

        // ---- Control flow ----
        Inst::JmpRel8 { rel } => {
            if cpu.opsize { cpu.eip = cpu.eip.wrapping_add(rel as i32 as u32); }
            else { cpu.ip = cpu.ip.wrapping_add(rel as i16 as u16); }
        }
        Inst::JmpRel16 { rel } => { cpu.ip = cpu.ip.wrapping_add(rel as u16); }
        Inst::JmpRel32 { rel } => { cpu.eip = cpu.eip.wrapping_add(rel as u32); }
        Inst::Jcc { cond, rel } => {
            if cond.test(cpu) {
                if cpu.opsize { cpu.eip = cpu.eip.wrapping_add(rel as i32 as u32); }
                else { cpu.ip = cpu.ip.wrapping_add(rel as i16 as u16); }
            }
        }
        Inst::Jcc32 { cond, rel } => {
            // 0F 80-8F: conditional jump with rel32 displacement. In 32-bit
            // mode it branches via EIP; in 16-bit mode via IP.
            if cond.test(cpu) {
                if cpu.opsize { cpu.eip = cpu.eip.wrapping_add(rel as u32); }
                else { cpu.ip = cpu.ip.wrapping_add(rel as u16); }
            }
        }
        // ---- MOVZX / MOVSX ----
        Inst::Movzx8 { m, dst } => {
            let v = cpu.read_rm8(&m);
            if cpu.opsize { cpu.set_reg32(Reg::reg32(dst), v as u32); }
            else { cpu.set_reg16(Reg::reg16(dst), v as u16); }
        }
        Inst::Movzx16 { m, dst } => {
            let v = cpu.read_rm16(&m);
            if cpu.opsize { cpu.set_reg32(Reg::reg32(dst), v as u32); }
            else { cpu.set_reg16(Reg::reg16(dst), v); }
        }
        Inst::Movsx8 { m, dst } => {
            let v = cpu.read_rm8(&m) as i8;
            if cpu.opsize { cpu.set_reg32(Reg::reg32(dst), v as i32 as u32); }
            else { cpu.set_reg16(Reg::reg16(dst), v as i16 as u16); }
        }
        Inst::Movsx16 { m, dst } => {
            let v = cpu.read_rm16(&m) as i16;
            if cpu.opsize { cpu.set_reg32(Reg::reg32(dst), v as i32 as u32); }
            else { cpu.set_reg16(Reg::reg16(dst), v as u16); }
        }
        Inst::CallRel16 { rel } => {
            let next = cpu.ip;
            cpu.push16(next);
            cpu.ip = cpu.ip.wrapping_add(rel as u16);
        }
        Inst::CallRel32 { rel } => {
            let next = cpu.eip;
            cpu.push32(next);
            cpu.eip = cpu.eip.wrapping_add(rel as u32);
        }
        Inst::Ret => { cpu.ip = cpu.pop16(); }
        Inst::RetImm { imm, w32 } => {
            // The stack adjustment happens after the return address is popped.
            if w32 {
                let target = cpu.pop32();
                cpu.esp = cpu.esp.wrapping_add(imm as u32);
                cpu.eip = target;
            } else {
                let target = cpu.pop16();
                cpu.set_sp(cpu.sp().wrapping_add(imm));
                cpu.ip = target;
            }
        }
        Inst::Ret32 => { cpu.eip = cpu.pop32(); }
        Inst::XchgAxReg { reg } => {
            let ax = cpu.reg16(Reg16::Ax);
            let r = cpu.reg16(reg);
            cpu.set_reg16(Reg16::Ax, r);
            cpu.set_reg16(reg, ax);
        }
        Inst::XchgEaxReg { reg } => {
            let ax = cpu.reg32(Reg32::Eax);
            let r = cpu.reg32(reg);
            cpu.set_reg32(Reg32::Eax, r);
            cpu.set_reg32(reg, ax);
        }
        Inst::Int { vector } => {
            // Record system calls from user mode: the sequence of calls ld.so
            // makes is the quickest way to see which path it took.
            if cpu.debug_enabled && vector == 0x80 && cpu.cpl() == 3
                && cpu.syscall_log.len() < 512 {
                let n = cpu.instructions_executed;
                cpu.syscall_log.push((n, cpu.eax, cpu.ebx, cpu.ecx, cpu.edx));
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
            cpu.servicing_irq = false;
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
                cpu.esp = esp;
            }
            cpu.eip = eip;
            cpu.load_seg(SegReg::Cs, cs);
            cpu.flags = write_flags(cpu.flags, f);
            cpu.servicing_irq = false;
            cpu.invalidate_phys_ip();
        }
        // PUSHF/PUSHFD and POPF/POPFD follow the operand size. In 32-bit mode
        // these move the whole of EFLAGS — which is how Linux probes the AC
        // and ID bits — so a 16-bit-only implementation both loses those bits
        // and moves ESP by the wrong amount.
        Inst::Pushf => {
            if cpu.opsize {
                // VM and RF read back as zero in the pushed image.
                cpu.push32(cpu.flags & !(flags::VM | flags::RF));
            } else {
                cpu.push16(cpu.flags as u16);
            }
        }
        Inst::Popf => {
            if cpu.opsize {
                let f = cpu.pop32();
                cpu.flags = write_flags(cpu.flags, f);
            } else {
                let f = cpu.pop16();
                cpu.flags = write_flags(cpu.flags, (cpu.flags & 0xFFFF_0000) | f as u32);
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
            let v = cpu.read_rm32(&m);
            let r = v & imm;
            set_logic_flags32(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestRm8Reg { m, reg } => {
            let v = cpu.read_rm8(&m);
            let r = v & cpu.reg8(Reg::reg8(reg));
            set_logic_flags8(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestRm16Reg { m, reg } => {
            let v = cpu.read_rm16(&m);
            let r = v & cpu.reg16(Reg::reg16(reg));
            set_logic_flags16(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::TestRm32Reg { m, reg } => {
            let v = cpu.read_rm32(&m);
            let r = v & cpu.reg32(Reg::reg32(reg));
            set_logic_flags32(cpu, r);
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
            let v = cpu.reg32(Reg32::Eax);
            let r = v & imm;
            set_logic_flags32(cpu, r);
            cpu.set_flag(CF, false);
            cpu.set_flag(OF, false);
        }
        Inst::NotRm8 { m } => { let v = cpu.read_rm8(&m); cpu.write_rm8(&m, !v); }
        Inst::NotRm16 { m } => { let v = cpu.read_rm16(&m); cpu.write_rm16(&m, !v); }
        Inst::NotRm32 { m } => { let v = cpu.read_rm32(&m); cpu.write_rm32(&m, !v); }
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
            let v = cpu.read_rm32(&m);
            let r = v.wrapping_neg();
            cpu.write_rm32(&m, r);
            set_logic_flags32(cpu, r);
            cpu.set_flag(CF, v != 0);
            cpu.set_flag(OF, v == 0x8000_0000);
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
            let v = cpu.read_rm32(&m);
            let a = cpu.reg32(Reg32::Eax) as u64;
            let r = a * v as u64;
            cpu.set_reg32(Reg32::Eax, r as u32);
            cpu.set_reg32(Reg32::Edx, (r >> 32) as u32);
            let c = (r >> 32) != 0;
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
            let v = cpu.read_rm32(&m) as i32 as i64;
            let a = cpu.reg32(Reg32::Eax) as i32 as i64;
            let r = a * v;
            cpu.set_reg32(Reg32::Eax, r as u32);
            cpu.set_reg32(Reg32::Edx, (r >> 32) as u32);
            let hi = (r >> 32) as i32;
            let lo = r as i32;
            let c = hi != lo;
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
            let v = cpu.read_rm32(&m);
            if v == 0 {
                cpu.pending_exception = Some((0x00, None)); // #DE
                return;
            }
            let a = ((cpu.reg32(Reg32::Edx) as u64) << 32) | cpu.reg32(Reg32::Eax) as u64;
            let q = a / v as u64;
            let rem = a % v as u64;
            cpu.set_reg32(Reg32::Eax, q as u32);
            cpu.set_reg32(Reg32::Edx, rem as u32);
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
            let v = cpu.read_rm32(&m) as i32 as i64;
            if v == 0 {
                cpu.pending_exception = Some((0x00, None)); // #DE
                return;
            }
            let a = ((cpu.reg32(Reg32::Edx) as u64) << 32 | cpu.reg32(Reg32::Eax) as u64) as i64;
            let q = a / v;
            let rem = a % v;
            cpu.set_reg32(Reg32::Eax, q as u32);
            cpu.set_reg32(Reg32::Edx, rem as u32);
        }

        // ---- LEA ----
        Inst::Lea { m, dst } => {
            let ea = lea_offset(&m, cpu);
            if cpu.opsize {
                cpu.set_reg32(Reg::reg32(dst), ea as u32);
            } else {
                cpu.set_reg16(Reg::reg16(dst), ea as u16);
            }
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
        Inst::Cwde => {
            let ax = cpu.reg16(Reg16::Ax);
            cpu.set_reg32(Reg32::Eax, ax as i16 as i32 as u32);
        }
        Inst::Cdq => {
            let eax = cpu.reg32(Reg32::Eax);
            let edx = if (eax as i32) < 0 { 0xFFFF_FFFF } else { 0 };
            cpu.set_reg32(Reg32::Edx, edx);
        }

        // ---- LOOP / LOOPZ / LOOPNZ / JCXZ ----
        Inst::Loop { cond, rel } => {
            let take = if cpu.opsize {
                // 32-bit mode: LOOP uses ECX and branches via EIP.
                match cond {
                    LoopCond::Jcxz => cpu.ecx == 0,
                    _ => {
                        cpu.ecx = cpu.ecx.wrapping_sub(1);
                        match cond {
                            LoopCond::Loop => cpu.ecx != 0,
                            LoopCond::Loopz => cpu.ecx != 0 && cpu.get_flag(ZF),
                            LoopCond::Loopnz => cpu.ecx != 0 && !cpu.get_flag(ZF),
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
                if cpu.opsize { cpu.eip = cpu.eip.wrapping_add(rel as i32 as u32); }
                else { cpu.ip = cpu.ip.wrapping_add(rel as i16 as u16); }
            }
        }

        // ---- Far control flow ----
        Inst::JmpFar { off, seg } => {
            cpu.cs = seg;
            cpu.ip = off;
        }
        Inst::CallFar { off, seg } => {
            let ip = cpu.ip;
            cpu.push16(cpu.cs);
            cpu.push16(ip);
            cpu.cs = seg;
            cpu.ip = off;
        }
        Inst::JmpFar32 { off, seg } => {
            cpu.cs = seg;
            cpu.eip = off;
        }
        Inst::CallFar32 { off, seg } => {
            let ip = cpu.eip;
            cpu.push32(cpu.cs as u32);
            cpu.push32(ip);
            cpu.cs = seg;
            cpu.eip = off;
        }
        Inst::Retf => {
            cpu.ip = cpu.pop16();
            cpu.cs = cpu.pop16();
        }
        Inst::Retf32 => {
            cpu.eip = cpu.pop32();
            cpu.cs = cpu.pop16();
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
            let v = cpu.read_rm32(&m);
            let result = v.wrapping_add(1);
            cpu.write_rm32(&m, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags32(cpu, result);
            set_add_carry32(cpu, v, 1, result, false);
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
            let v = cpu.read_rm32(&m);
            let result = v.wrapping_sub(1);
            cpu.write_rm32(&m, result);
            let cf = cpu.get_flag(CF);
            set_logic_flags32(cpu, result);
            set_sub_borrow32(cpu, v, 1, result, false);
            cpu.set_flag(CF, cf);
        }
        Inst::CallRm16 { m } => {
            let target = cpu.read_rm16(&m);
            let next = cpu.ip;
            cpu.push16(next);
            cpu.ip = target;
        }
        Inst::CallRm32 { m } => {
            let target = cpu.read_rm32(&m);
            let next = cpu.eip;
            cpu.push32(next);
            cpu.eip = target;
        }
        Inst::JmpRm16 { m } => {
            cpu.ip = cpu.read_rm16(&m);
        }
        Inst::JmpRm32 { m } => {
            cpu.eip = cpu.read_rm32(&m);
        }
        Inst::PushRm16 { m } => {
            let v = cpu.read_rm16(&m);
            cpu.push16(v);
        }
        Inst::PushRm32 { m } => {
            let v = cpu.read_rm32(&m);
            cpu.push32(v);
        }
        // BSF/BSR: ZF is set when the source is zero, and the destination is
        // then architecturally undefined - we leave it alone.
        Inst::Bsf { m, dst, w32 } => {
            if w32 {
                let v = cpu.read_rm32(&m);
                cpu.set_flag(flags::ZF, v == 0);
                if v != 0 { cpu.set_reg32(Reg::reg32(dst), v.trailing_zeros()); }
            } else {
                let v = cpu.read_rm16(&m);
                cpu.set_flag(flags::ZF, v == 0);
                if v != 0 { cpu.set_reg16(Reg::reg16(dst), v.trailing_zeros() as u16); }
            }
        }
        Inst::Bsr { m, dst, w32 } => {
            if w32 {
                let v = cpu.read_rm32(&m);
                cpu.set_flag(flags::ZF, v == 0);
                if v != 0 { cpu.set_reg32(Reg::reg32(dst), 31 - v.leading_zeros()); }
            } else {
                let v = cpu.read_rm16(&m);
                cpu.set_flag(flags::ZF, v == 0);
                if v != 0 { cpu.set_reg16(Reg::reg16(dst), (15 - v.leading_zeros()) as u16); }
            }
        }
        // XCHG r/m, r. No flags.
        Inst::XchgRmReg { m, reg, width } => match width {
            8 => {
                let a = cpu.read_rm8(&m);
                let b = cpu.reg8(Reg::reg8(reg));
                cpu.write_rm8(&m, b);
                cpu.set_reg8(Reg::reg8(reg), a);
            }
            16 => {
                let a = cpu.read_rm16(&m);
                let b = cpu.reg16(Reg::reg16(reg));
                cpu.write_rm16(&m, b);
                cpu.set_reg16(Reg::reg16(reg), a);
            }
            _ => {
                let a = cpu.read_rm32(&m);
                let b = cpu.reg32(Reg::reg32(reg));
                cpu.write_rm32(&m, b);
                cpu.set_reg32(Reg::reg32(reg), a);
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
                if acc == dest { let v = cpu.reg8(Reg::reg8(reg)); cpu.write_rm8(&m, v); }
                else { cpu.set_reg8(Reg8::Al, dest); }
            }
            16 => {
                let dest = cpu.read_rm16(&m);
                let acc = cpu.reg16(Reg16::Ax);
                alu16(cpu, AluOp::Cmp, acc, dest);
                if acc == dest { let v = cpu.reg16(Reg::reg16(reg)); cpu.write_rm16(&m, v); }
                else { cpu.set_reg16(Reg16::Ax, dest); }
            }
            _ => {
                let dest = cpu.read_rm32(&m);
                let acc = cpu.reg32(Reg32::Eax);
                alu32(cpu, AluOp::Cmp, acc, dest);
                if acc == dest { let v = cpu.reg32(Reg::reg32(reg)); cpu.write_rm32(&m, v); }
                else { cpu.set_reg32(Reg32::Eax, dest); }
            }
        },
        // XADD: the destination gets the sum, the source register gets the
        // destination's old value. Flags are those of ADD.
        Inst::Xadd { m, reg, width } => match width {
            8 => {
                let dest = cpu.read_rm8(&m);
                let src = cpu.reg8(Reg::reg8(reg));
                let sum = alu8(cpu, AluOp::Add, dest, src);
                cpu.set_reg8(Reg::reg8(reg), dest);
                cpu.write_rm8(&m, sum);
            }
            16 => {
                let dest = cpu.read_rm16(&m);
                let src = cpu.reg16(Reg::reg16(reg));
                let sum = alu16(cpu, AluOp::Add, dest, src);
                cpu.set_reg16(Reg::reg16(reg), dest);
                cpu.write_rm16(&m, sum);
            }
            _ => {
                let dest = cpu.read_rm32(&m);
                let src = cpu.reg32(Reg::reg32(reg));
                let sum = alu32(cpu, AluOp::Add, dest, src);
                cpu.set_reg32(Reg::reg32(reg), dest);
                cpu.write_rm32(&m, sum);
            }
        },
        // CMPXCHG8B: compare EDX:EAX with the 64-bit destination; on a
        // match store ECX:EBX, otherwise load the destination into EDX:EAX.
        // Only ZF reports the outcome.
        Inst::Cmpxchg8b { m } => {
            let addr = if cpu.addrsize { cpu.modrm_addr32_write(&m) } else { cpu.modrm_addr_write(&m) };
            let lo = cpu.mem.read_u32(addr);
            let hi = cpu.mem.read_u32(addr + 4);
            if lo == cpu.eax && hi == cpu.edx {
                cpu.set_flag(flags::ZF, true);
                let (bl, ch) = (cpu.ebx, cpu.ecx);
                cpu.mem.write_u32(addr, bl);
                cpu.mem.write_u32(addr + 4, ch);
            } else {
                cpu.set_flag(flags::ZF, false);
                cpu.set_reg32(Reg32::Eax, lo);
                cpu.set_reg32(Reg32::Edx, hi);
            }
        }
        Inst::Bswap { reg } => {
            let v = cpu.reg32(Reg::reg32(reg));
            cpu.set_reg32(Reg::reg32(reg), v.swap_bytes());
        }
        // CMOVcc: the load happens only when the condition holds. (A real CPU
        // reads the memory operand either way, but nothing observable here
        // depends on that, and skipping the read avoids a spurious fault.)
        Inst::Cmovcc { cond, m, dst, w32 } => {
            if cond.test(cpu) {
                if w32 {
                    let v = cpu.read_rm32(&m);
                    cpu.set_reg32(Reg::reg32(dst), v);
                } else {
                    let v = cpu.read_rm16(&m);
                    cpu.set_reg16(Reg::reg16(dst), v);
                }
            }
        }
        // PUSH/POP a segment register. The pushed value occupies the full
        // operand size, but only the low 16 bits carry the selector.
        Inst::PushSeg { seg } => {
            let v = cpu.seg(seg);
            if cpu.opsize { cpu.push32(v as u32); } else { cpu.push16(v); }
        }
        Inst::PopSeg { seg } => {
            let v = if cpu.opsize { cpu.pop32() as u16 } else { cpu.pop16() };
            cpu.load_seg(seg, v);
        }
        // Debug registers. Nothing here implements hardware breakpoints, so
        // they are plain storage: the kernel writes DR7 = 0 and DR0-3 = 0 at
        // startup and expects the reads to agree, which this satisfies.
        Inst::MovDr { dr, reg } => {
            let v = cpu.dr[dr as usize];
            cpu.set_reg32(Reg::reg32(reg), v);
        }
        Inst::MovToDr { dr, reg } => {
            cpu.dr[dr as usize] = cpu.reg32(Reg::reg32(reg));
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
            if w32 {
                cpu.set_reg32(Reg32::Esp, cpu.ebp);
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
            let v = cpu.pop32();
            cpu.write_rm32(&m, v);
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
            let esize = if w { if cpu.opsize { 4u32 } else { 2 } } else { 1 };
            let step = string_step(cpu, esize);
            let a32 = cpu.addrsize;
            let (mut si, mut di) = (string_si(cpu, a32), string_di(cpu, a32));
            let mut cnt = string_count(cpu, a32, rep);
            while cnt > 0 {
                let src = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), si);
                let dst = cpu.translate_write(SegReg::Es, di);
                if cpu.pending_exception.is_some() { break; }
                match esize {
                    4 => { let v = cpu.mem.read_u32(src); cpu.mem.write_u32(dst, v); }
                    2 => { let v = cpu.mem.read_u16(src); cpu.mem.write_u16(dst, v); }
                    _ => { let v = cpu.mem.read_u8(src); cpu.mem.write_u8(dst, v); }
                }
                si = string_advance(si, step, a32);
                di = string_advance(di, step, a32);
                cnt -= 1;
            }
            string_set_si(cpu, a32, si);
            string_set_di(cpu, a32, di);
            if rep != Rep::None { string_set_count(cpu, a32, cnt); }
        }
        Inst::Stos { rep, w } => {
            let esize = if w { if cpu.opsize { 4u32 } else { 2 } } else { 1 };
            let step = string_step(cpu, esize);
            let a32 = cpu.addrsize;
            let mut di = string_di(cpu, a32);
            let mut cnt = string_count(cpu, a32, rep);
            while cnt > 0 {
                let dst = cpu.translate_write(SegReg::Es, di);
                if cpu.pending_exception.is_some() { break; }
                match esize {
                    4 => { let v = cpu.reg32(Reg32::Eax); cpu.mem.write_u32(dst, v); }
                    2 => { let v = cpu.reg16(Reg16::Ax); cpu.mem.write_u16(dst, v); }
                    _ => { let v = cpu.reg8(Reg8::Al); cpu.mem.write_u8(dst, v); }
                }
                di = string_advance(di, step, a32);
                cnt -= 1;
            }
            string_set_di(cpu, a32, di);
            if rep != Rep::None { string_set_count(cpu, a32, cnt); }
        }
        Inst::Lods { rep, w } => {
            let esize = if w { if cpu.opsize { 4u32 } else { 2 } } else { 1 };
            let step = string_step(cpu, esize);
            let a32 = cpu.addrsize;
            let mut si = string_si(cpu, a32);
            let mut cnt = string_count(cpu, a32, rep);
            while cnt > 0 {
                let src = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), si);
                if cpu.pending_exception.is_some() { break; }
                match esize {
                    4 => { let v = cpu.mem.read_u32(src); cpu.set_reg32(Reg32::Eax, v); }
                    2 => { let v = cpu.mem.read_u16(src); cpu.set_reg16(Reg16::Ax, v); }
                    _ => { let v = cpu.mem.read_u8(src); cpu.set_reg8(Reg8::Al, v); }
                }
                si = string_advance(si, step, a32);
                cnt -= 1;
            }
            string_set_si(cpu, a32, si);
            if rep != Rep::None { string_set_count(cpu, a32, cnt); }
        }
        Inst::Cmps { rep, w } => {
            let esize = if w { if cpu.opsize { 4u32 } else { 2 } } else { 1 };
            let step = string_step(cpu, esize);
            let a32 = cpu.addrsize;
            let (mut si, mut di) = (string_si(cpu, a32), string_di(cpu, a32));
            let mut cnt = string_count(cpu, a32, rep);
            while cnt > 0 {
                let src = cpu.translate(cpu.operand_seg_for_exec(SegReg::Ds), si);
                let dst = cpu.translate(SegReg::Es, di);
                if cpu.pending_exception.is_some() { break; }
                match esize {
                    4 => {
                        let (a, b) = (cpu.mem.read_u32(src), cpu.mem.read_u32(dst));
                        alu32(cpu, AluOp::Cmp, a, b);
                    }
                    2 => {
                        let (a, b) = (cpu.mem.read_u16(src), cpu.mem.read_u16(dst));
                        alu16(cpu, AluOp::Cmp, a, b);
                    }
                    _ => {
                        let (a, b) = (cpu.mem.read_u8(src), cpu.mem.read_u8(dst));
                        alu8(cpu, AluOp::Cmp, a, b);
                    }
                }
                si = string_advance(si, step, a32);
                di = string_advance(di, step, a32);
                cnt -= 1;
                if !string_repeat(cpu, rep, cnt) { break; }
            }
            string_set_si(cpu, a32, si);
            string_set_di(cpu, a32, di);
            if rep != Rep::None { string_set_count(cpu, a32, cnt); }
        }
        Inst::Scas { rep, w } => {
            let esize = if w { if cpu.opsize { 4u32 } else { 2 } } else { 1 };
            let step = string_step(cpu, esize);
            let a32 = cpu.addrsize;
            let mut di = string_di(cpu, a32);
            let mut cnt = string_count(cpu, a32, rep);
            while cnt > 0 {
                let dst = cpu.translate(SegReg::Es, di);
                if cpu.pending_exception.is_some() { break; }
                match esize {
                    4 => {
                        let (a, b) = (cpu.reg32(Reg32::Eax), cpu.mem.read_u32(dst));
                        alu32(cpu, AluOp::Cmp, a, b);
                    }
                    2 => {
                        let (a, b) = (cpu.reg16(Reg16::Ax), cpu.mem.read_u16(dst));
                        alu16(cpu, AluOp::Cmp, a, b);
                    }
                    _ => {
                        let (a, b) = (cpu.reg8(Reg8::Al), cpu.mem.read_u8(dst));
                        alu8(cpu, AluOp::Cmp, a, b);
                    }
                }
                di = string_advance(di, step, a32);
                cnt -= 1;
                if !string_repeat(cpu, rep, cnt) { break; }
            }
            string_set_di(cpu, a32, di);
            if rep != Rep::None { string_set_count(cpu, a32, cnt); }
        }

        // ---- LGDT / LIDT ----
        // The address size of the memory operand follows the current
        // addressing mode (addrsize): 32-bit (modrm_addr32) in a D=1
        // segment, 16-bit (modrm_addr) otherwise. The decoder already
        // fetched the ModR/M, SIB, and displacement bytes according to
        // addrsize; the executor must compute the address the same way.
        Inst::Lgdt { m } => {
            let base = if cpu.addrsize { cpu.modrm_addr32(&m) } else { cpu.modrm_addr(&m) };
            let limit = cpu.mem.read_u16(base);
            let base32 = cpu.mem.read_u32(base + 2);
            cpu.gdt_base = base32;
            cpu.gdt_limit = limit;
        }
        Inst::Lidt { m } => {
            let base = if cpu.addrsize { cpu.modrm_addr32(&m) } else { cpu.modrm_addr(&m) };
            let limit = cpu.mem.read_u16(base);
            let base32 = cpu.mem.read_u32(base + 2);
            cpu.idt_base = base32;
            cpu.idt_limit = limit;
        }

        // ---- INVLPG (0x0F 0x01 /7) ----
        // Invalidate the TLB entry for the linear address of the memory
        // operand. The linear address is computed the same way as a normal
        // memory operand (segment + offset), then we invalidate that page.
        Inst::Invlpg { m } => {
            // Compute the linear address (segment base + offset), not the
            // physical address, since INVLPG operates on linear addresses.
            let linear = if cpu.pe {
                let seg = cpu.operand_seg_for_exec(SegReg::Ds);
                let offset = if cpu.addrsize {
                    cpu.modrm_offset32(&m)
                } else {
                    cpu.modrm_offset(&m)
                };
                cpu.seg_desc[seg as usize].base.wrapping_add(offset)
            } else {
                let seg = cpu.operand_seg_for_exec(SegReg::Ds);
                let offset = if cpu.addrsize {
                    cpu.modrm_offset32(&m)
                } else {
                    cpu.modrm_offset(&m)
                };
                ((cpu.seg(seg) as u32) << 4).wrapping_add(offset)
            };
            cpu.invlpg(linear);
        }

        // ---- MOV to/from control registers ----
        Inst::MovCr { cr, reg } => {
            let v = match cr {
                0 => cpu.cr0,
                2 => cpu.cr2,
                3 => cpu.cr3,
                _ => cpu.cr4,
            };
            cpu.set_reg32(Reg::reg32(reg), v);
        }
        Inst::MovToCr { cr, reg } => {
            let v = cpu.reg32(Reg::reg32(reg));
            match cr {
                0 => {
                    // If paging is being toggled (PG bit changes), flush TLB.
                    let old_pg = cpu.cr0 & 0x8000_0000 != 0;
                    let new_pg = v & 0x8000_0000 != 0;
                    if old_pg != new_pg {
                        cpu.flush_tlb();
                    }
                    cpu.cr0 = v;
                }
                2 => cpu.cr2 = v,
                3 => {
                    cpu.cr3 = v;
                    cpu.flush_tlb();
                }
                _ => {
                    // Writing CR4 flushes the TLB. Linux's
                    // `__flush_tlb_global()` is *literally* a CR4 write with
                    // PGE toggled off and back on -- the flush is the whole
                    // point of the sequence, and without it every global
                    // mapping keeps a stale translation.
                    cpu.cr4 = v;
                    cpu.flush_tlb();
                }
            }
        }

        // ---- CLTS (0x0F 0x06) ----
        Inst::Clts => {
            cpu.cr0 &= !0x8; // clear CR0.TS (bit 3)
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
                    cpu.set_reg32(Reg32::Ecx, 0x0000_0000);
                    // Feature flags, deliberately only the ones implemented
                    // here: FPU(0), PSE(3), TSC(4), MSR(5), CX8(8), PGE(13),
                    // CMOV(15), CLFSH(19), FXSR(24).
                    //
                    // Left OFF on purpose, because claiming them would make
                    // the kernel issue instructions this CPU does not have:
                    // PAE(6), APIC(9), SEP(11) -- so system calls arrive as
                    // int 0x80 rather than SYSENTER -- MTRR(12), PAT(16),
                    // MMX(23), SSE(25) and SSE2(26).
                    cpu.set_reg32(Reg32::Edx, 0x0108_A139);
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
        Inst::Rdmsr => {
            // ECX = MSR index. Return 0 for all MSRs (a real CPU would
            // return specific values; 0 is enough for early boot probing).
            cpu.set_reg32(Reg32::Eax, 0);
            cpu.set_reg32(Reg32::Edx, 0);
        }
        Inst::Wrmsr => {
            // Ignore writes (no-op).
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
        Inst::Fxsave { m } => {
            let a = if cpu.addrsize { cpu.modrm_addr32_write(&m) } else { cpu.modrm_addr_write(&m) };
            cpu.fpu.fxsave(&mut cpu.mem, a);
        }
        Inst::Fxrstor { m } => {
            let a = if cpu.addrsize { cpu.modrm_addr32(&m) } else { cpu.modrm_addr(&m) };
            cpu.fpu.fxrstor(&cpu.mem, a);
        }

        // Two/three-operand IMUL. CF = OF = 1 when the truncated result
        // differs from the full signed product (i.e. it did not fit).
        Inst::ImulRegRm16 { m, dst } => {
            let a = cpu.read_rm16(&m) as i16 as i32;
            let b = cpu.reg16(Reg::reg16(dst)) as i16 as i32;
            imul_store16(cpu, dst, a * b);
        }
        Inst::ImulRegRm32 { m, dst } => {
            let a = cpu.read_rm32(&m) as i32 as i64;
            let b = cpu.reg32(Reg::reg32(dst)) as i32 as i64;
            imul_store32(cpu, dst, a * b);
        }
        Inst::ImulRegRmImm16 { m, dst, imm } => {
            let a = cpu.read_rm16(&m) as i16 as i32;
            imul_store16(cpu, dst, a * imm as i32);
        }
        Inst::ImulRegRmImm32 { m, dst, imm } => {
            let a = cpu.read_rm32(&m) as i32 as i64;
            imul_store32(cpu, dst, a * imm as i64);
        }

        // SHLD/SHRD: shift the destination, feeding in bits from the source
        // register. A count of 0 is a no-op that leaves every flag alone; a
        // count >= the operand width is architecturally undefined, and like
        // real hardware we mask it to 5 bits and let it fall out.
        Inst::Shld { m, reg, count, w32 } => {
            let n = match count { ShiftCount::One => 1, ShiftCount::Imm(i) => i, ShiftCount::Cl => cpu.reg8(Reg8::Cl) } & 0x1F;
            if n != 0 {
                if w32 {
                    let d = cpu.read_rm32(&m);
                    let src = cpu.reg32(Reg::reg32(reg));
                    let res = (d << n) | (src >> (32 - n));
                    let cf = (d >> (32 - n)) & 1 != 0;
                    cpu.write_rm32(&m, res);
                    set_shift_flags32(cpu, res, cf, (d ^ res) >> 31 & 1 != 0);
                } else {
                    let d = cpu.read_rm16(&m);
                    let src = cpu.reg16(Reg::reg16(reg));
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
            let n = match count { ShiftCount::One => 1, ShiftCount::Imm(i) => i, ShiftCount::Cl => cpu.reg8(Reg8::Cl) } & 0x1F;
            if n != 0 {
                if w32 {
                    let d = cpu.read_rm32(&m);
                    let src = cpu.reg32(Reg::reg32(reg));
                    let res = (d >> n) | (src << (32 - n));
                    let cf = (d >> (n - 1)) & 1 != 0;
                    cpu.write_rm32(&m, res);
                    set_shift_flags32(cpu, res, cf, (d ^ res) >> 31 & 1 != 0);
                } else {
                    let d = cpu.read_rm16(&m);
                    let src = cpu.reg16(Reg::reg16(reg));
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
    // IDT entry: 8 bytes. offset = (bytes 0-1) | (bytes 6-7 << 16), the
    // segment selector is bytes 2-3, and byte 5 holds the type/attributes.
    let entry = cpu.idt_base.wrapping_add((vector as u32) * 8);
    let addr = Memory::phys32(entry);
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
    let old_eip = cpu.eip;
    let old_flags = cpu.flags;
    let (old_ss, old_esp) = (cpu.ss, cpu.esp);

    if switching {
        cpu.ring_switches += 1;
        let (ss0, esp0) = cpu.tss_stack0();
        cpu.load_seg(SegReg::Ss, ss0);
        cpu.esp = esp0;
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
    cpu.eip = target;
    cpu.invalidate_phys_ip();
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
fn imul_store16(cpu: &mut Cpu, dst: u8, full: i32) {
    use flags::*;
    let r = full as u16;
    cpu.set_reg16(Reg::reg16(dst), r);
    let overflow = full != (r as i16) as i32;
    cpu.set_flag(CF, overflow);
    cpu.set_flag(OF, overflow);
    set_logic_flags16(cpu, r);
}

/// 32-bit counterpart of `imul_store16`.
fn imul_store32(cpu: &mut Cpu, dst: u8, full: i64) {
    use flags::*;
    let r = full as u32;
    cpu.set_reg32(Reg::reg32(dst), r);
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
        cpu.set_reg32(Reg::reg32(m.reg), off);
        cpu.load_seg(seg, sel);
    } else {
        // 16-bit offset + 16-bit segment.
        let off = cpu.mem.read_u16(addr);
        let sel = cpu.mem.read_u16(addr + 2);
        cpu.set_reg16(Reg::reg16(m.reg), off);
        cpu.load_seg(seg, sel);
    }
}

/// Compute the effective address (offset only, no segment) of a ModR/M
/// memory operand, for LEA.
fn lea_offset(m: &ModRm, cpu: &Cpu) -> u32 {
    if cpu.addrsize {
        let mut ea: u32 = 0;
        if let Some(sib) = m.sib {
            let scale = 1u32 << ((sib >> 6) & 3);
            let index = (sib >> 3) & 7;
            let base = sib & 7;
            if index != 4 {
                ea = ea.wrapping_add(cpu.reg32(Reg::reg32(index)).wrapping_mul(scale));
            }
            if !(m.mod_field == 0 && base == 5) {
                ea = ea.wrapping_add(cpu.reg32(Reg::reg32(base)));
            }
        } else if m.mod_field != 3 && !(m.mod_field == 0 && m.rm == 5) {
            // mod=00, rm=101 is disp32 with no base register.
            ea = ea.wrapping_add(cpu.reg32(Reg::reg32(m.rm)));
        }
        if let Some(d32) = m.disp32 { ea = ea.wrapping_add(d32); }
        ea
    } else {
        let base = match m.rm {
            0 => cpu.bx().wrapping_add(cpu.si()),
            1 => cpu.bx().wrapping_add(cpu.di()),
            2 => cpu.bp().wrapping_add(cpu.si()),
            3 => cpu.bp().wrapping_add(cpu.di()),
            4 => cpu.si(),
            5 => cpu.di(),
            6 => cpu.bp(),
            _ => cpu.bx(),
        };
        let mut ea = base as u32;
        if let Some(d8) = m.disp8 { ea = ea.wrapping_add(d8 as u32); }
        if let Some(d16) = m.disp16 { ea = ea.wrapping_add(d16 as u32); }
        ea
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
/// - The count is masked to 5 bits on 386+ **for every operand size**, so a
///   16-bit shift by 20 shifts a 16-bit value 20 places (result 0) rather
///   than wrapping the count. Rust's `wrapping_shl` masks the count to the
///   type's width, which is *not* the same thing — so the shift is done in
///   `u64` and truncated.
/// - A count of 0 changes nothing, flags included.
/// - RCL/RCR rotate through a `width + 1`-bit quantity (the operand plus CF),
///   so their effective count is taken modulo `width + 1` for 8- and 16-bit
///   operands. For 32-bit the 5-bit mask already keeps it in range.
/// - OF is architecturally defined only for a count of 1; we leave it clear
///   otherwise, which is what the shape `n == 1 && ..` below expresses.
fn shift_width(cpu: &mut Cpu, op: ShiftOp, v: u32, n: u32, width: u32) -> u32 {
    use flags::*;
    let n = n & 0x1F;
    if n == 0 { return v; }
    let mask: u64 = if width == 32 { 0xFFFF_FFFF } else { (1u64 << width) - 1 };
    let msb: u32 = 1u32 << (width - 1);
    let v64 = v as u64 & mask;

    match op {
        ShiftOp::Shl => {
            let wide = v64 << n.min(63);
            let r = (wide & mask) as u32;
            let cf = n <= width && (v64 >> (width - n.min(width))) & 1 != 0;
            cpu.set_flag(CF, cf);
            set_logic_flags_width(cpu, r, width);
            // OF (count 1) = MSB of the result XOR the new CF.
            cpu.set_flag(OF, n == 1 && (((r & msb) != 0) != cf));
            r
        }
        ShiftOp::Shr => {
            let r = if n >= width { 0 } else { (v64 >> n) as u32 };
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
            let r = ((sv >> n.min(width - 1)) as u64 & mask) as u32;
            let cf = ((sv >> (n - 1).min(width - 1)) & 1) != 0;
            cpu.set_flag(CF, cf);
            set_logic_flags_width(cpu, r, width);
            cpu.set_flag(OF, false);
            r
        }
        ShiftOp::Rol => {
            let k = n % width;
            let r = if k == 0 { v } else { (((v64 << k) | (v64 >> (width - k))) & mask) as u32 };
            let cf = r & 1 != 0;
            cpu.set_flag(CF, cf);
            cpu.set_flag(OF, n == 1 && (((r & msb) != 0) != cf));
            r
        }
        ShiftOp::Ror => {
            let k = n % width;
            let r = if k == 0 { v } else { (((v64 >> k) | (v64 << (width - k))) & mask) as u32 };
            let cf = r & msb != 0;
            cpu.set_flag(CF, cf);
            // OF (count 1) = XOR of the two most significant result bits.
            let second = (r >> (width - 2)) & 1 != 0;
            cpu.set_flag(OF, n == 1 && (cf != second));
            r
        }
        ShiftOp::Rcl | ShiftOp::Rcr => {
            // Rotate through carry: a (width + 1)-bit quantity.
            let bits = width + 1;
            let k = n % bits;
            let wide = v64 | ((cpu.get_flag(CF) as u64) << width);
            let full: u64 = (1u64 << bits) - 1;
            let rot = if k == 0 {
                wide
            } else if op == ShiftOp::Rcl {
                ((wide << k) | (wide >> (bits - k))) & full
            } else {
                ((wide >> k) | (wide << (bits - k))) & full
            };
            let r = (rot & mask) as u32;
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

/// Sign-extend the low `width` bits of `v` to i64.
fn sign_extend(v: u32, width: u32) -> i64 {
    match width {
        8 => v as u8 as i8 as i64,
        16 => v as u16 as i16 as i64,
        _ => v as i32 as i64,
    }
}

/// SF/ZF/PF for a result of the given width.
fn set_logic_flags_width(cpu: &mut Cpu, r: u32, width: u32) {
    match width {
        8 => set_logic_flags8(cpu, r as u8),
        16 => set_logic_flags16(cpu, r as u16),
        _ => set_logic_flags32(cpu, r),
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
    let width: i64 = if cpu.opsize { 32 } else { 16 };
    let offset: i64 = match bit {
        BitOffset::Imm(i) => i as i64,
        BitOffset::Reg(r) => {
            if cpu.opsize { cpu.reg32(Reg::reg32(r)) as i32 as i64 }
            else { cpu.reg16(Reg::reg16(r)) as i16 as i64 }
        }
    };

    if m.is_reg() {
        // Register destination: the offset is taken modulo the width.
        let b = offset.rem_euclid(width) as u32;
        let (old, new) = if cpu.opsize {
            let v = cpu.reg32(Reg::reg32(m.rm));
            (v >> b & 1 != 0, apply_bit(v, b, op))
        } else {
            let v = cpu.reg16(Reg::reg16(m.rm)) as u32;
            (v >> b & 1 != 0, apply_bit(v, b, op))
        };
        cpu.set_flag(flags::CF, old);
        if op != BitOp::Test {
            if cpu.opsize { cpu.set_reg32(Reg::reg32(m.rm), new); }
            else { cpu.set_reg16(Reg::reg16(m.rm), new as u16); }
        }
        return;
    }

    // Memory destination: step whole operands along the bit string. The
    // displacement is signed, so a negative offset reaches backwards.
    let bytes = width / 8;
    let word = offset.div_euclid(width);
    let b = offset.rem_euclid(width) as u32;
    let disp = (word * bytes) as i32;
    let base = if cpu.addrsize { cpu.modrm_addr32_access_pub(m, op != BitOp::Test) }
               else { cpu.modrm_addr_access_pub(m, op != BitOp::Test) };
    let addr = base.wrapping_add(disp as isize as usize);

    let (old, new) = if cpu.opsize {
        let v = cpu.mem.read_u32(addr);
        (v >> b & 1 != 0, apply_bit(v, b, op))
    } else {
        let v = cpu.mem.read_u16(addr) as u32;
        (v >> b & 1 != 0, apply_bit(v, b, op))
    };
    cpu.set_flag(flags::CF, old);
    if op != BitOp::Test {
        if cpu.opsize { cpu.mem.write_u32(addr, new); }
        else { cpu.mem.write_u16(addr, new as u16); }
    }
}

fn apply_bit(v: u32, b: u32, op: BitOp) -> u32 {
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
fn string_advance(v: u32, step: i32, a32: bool) -> u32 {
    let n = v.wrapping_add(step as u32);
    if a32 { n } else { n & 0xFFFF }
}

fn string_si(cpu: &Cpu, a32: bool) -> u32 { if a32 { cpu.esi } else { cpu.si() as u32 } }
fn string_di(cpu: &Cpu, a32: bool) -> u32 { if a32 { cpu.edi } else { cpu.di() as u32 } }
fn string_set_si(cpu: &mut Cpu, a32: bool, v: u32) {
    // `_raw`: this write records *where the fault stopped*, so it has to land
    // even though a fault is pending.
    if a32 { cpu.set_reg32_raw(Reg32::Esi, v); }
    else { cpu.set_reg16_raw(Reg16::Si, v as u16); }
}
fn string_set_di(cpu: &mut Cpu, a32: bool, v: u32) {
    if a32 { cpu.set_reg32_raw(Reg32::Edi, v); }
    else { cpu.set_reg16_raw(Reg16::Di, v as u16); }
}

/// Iteration count: the count register under a REP prefix, one without.
/// A REP with a zero count does nothing at all, which the `while cnt > 0`
/// loops express directly.
fn string_count(cpu: &Cpu, a32: bool, rep: Rep) -> u32 {
    if rep == Rep::None { 1 } else if a32 { cpu.ecx } else { cpu.cx() as u32 }
}
fn string_set_count(cpu: &mut Cpu, a32: bool, v: u32) {
    if a32 { cpu.set_reg32_raw(Reg32::Ecx, v); }
    else { cpu.set_reg16_raw(Reg16::Cx, v as u16); }
}

/// Should a REPE/REPNE comparison keep going? REP alone always continues
/// (the count test is the loop condition); the conditional forms also stop
/// on ZF.
fn string_repeat(cpu: &Cpu, rep: Rep, remaining: u32) -> bool {
    match rep {
        Rep::None => false,
        _ if remaining == 0 => false,
        Rep::Repe => cpu.get_flag(flags::ZF),
        Rep::Repne => !cpu.get_flag(flags::ZF),
    }
}

/// Read the r/m operand at `width` bits, shift it, and write it back.
fn do_shift(cpu: &mut Cpu, op: ShiftOp, m: &ModRm, width: u32, n: u32) {
    match width {
        8 => { let v = cpu.read_rm8(m); let r = shift8(cpu, op, v, n); cpu.write_rm8(m, r); }
        16 => { let v = cpu.read_rm16(m); let r = shift16(cpu, op, v, n); cpu.write_rm16(m, r); }
        _ => { let v = cpu.read_rm32(m); let r = shift32(cpu, op, v, n); cpu.write_rm32(m, r); }
    }
}

fn shift8(cpu: &mut Cpu, op: ShiftOp, v: u8, n: u32) -> u8 {
    shift_width(cpu, op, v as u32, n, 8) as u8
}

/// Perform a 16-bit shift/rotate, setting flags, and return the result.
fn shift16(cpu: &mut Cpu, op: ShiftOp, v: u16, n: u32) -> u16 {
    shift_width(cpu, op, v as u32, n, 16) as u16
}

/// Perform a 32-bit shift/rotate, setting flags, and return the result.
fn shift32(cpu: &mut Cpu, op: ShiftOp, v: u32, n: u32) -> u32 {
    shift_width(cpu, op, v, n, 32)
}

use crate::memory::Memory;

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(cpu.eax, 0x12345678);
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
        assert_eq!(cpu.eax, 3);
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
        assert_eq!(cpu.eax, 0);
        assert_eq!(cpu.edx, 1);
    }

    #[test]
    fn addrsize_prefix_32bit_addressing() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        // 67 8B 04 85 disp32 -> mov eax, [eax*4 + disp32]
        // modrm = 00 000 100 (reg=EAX, rm=100 -> SIB follows)
        // sib = 10 000 101 (scale=4, index=EAX, base=101 -> disp32)
        // disp32 = 0x1000
        cpu.eax = 0x10;
        cpu.mem.write_u32(0x1000 + 0x10 * 4, 0xDEADBEEF);
        load(&mut cpu, &[
            0x66, 0x67, 0x8B, 0x04, 0x85, 0x00, 0x10, 0x00, 0x00,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.eax, 0xDEADBEEF);
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
        assert_eq!(cpu.translate(SegReg::Ds, 0x1234), 0x11234);
    }

    #[test]
    fn protected_mode_int_through_idt() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.ss = 0;
        cpu.esp = 0x0100;
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
        cpu.eip = 0x1000;
        cpu.run(32);
        assert_eq!(cpu.eax, 0x99);
        assert!(cpu.halted);
        assert_eq!(cpu.esp, 0x0100);
    }

    #[test]
    fn loop_uses_eip_in_32bit_mode() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        // Flat 32-bit code segment (D=1) so opsize defaults to 32-bit.
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true,
        };
        cpu.eip = 0x1000;
        // mov ecx, 3 ; loop $ (E2 FE) ; hlt
        // (no 0x66 prefix: opsize already 32-bit in this segment)
        cpu.mem.load(0x1000, &[
            0xB9, 0x03, 0x00, 0x00, 0x00, // mov ecx, 3
            0xE2, 0xFE,                    // loop -2 (back to itself)
            0xF4,                          // hlt
        ]);
        cpu.run(64);
        assert!(cpu.halted);
        assert_eq!(cpu.ecx, 0);
        assert_eq!(cpu.eip, 0x1008);
    }

    #[test]
    fn jcc_uses_eip_in_32bit_mode() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true,
        };
        cpu.eip = 0x1000;
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
        assert_eq!(cpu.eip, 0x100A);
    }

    #[test]
    fn jmp_rel8_uses_eip_in_32bit_mode() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true,
        };
        cpu.eip = 0x1000;
        // jmp +1 (EB 01) over a hlt, landing on a second hlt.
        cpu.mem.load(0x1000, &[
            0xEB, 0x01, // jmp +1
            0xF4,       // hlt (skipped)
            0xF4,       // hlt (land here)
        ]);
        cpu.run(32);
        assert!(cpu.halted);
        assert_eq!(cpu.eip, 0x1004);
    }

    #[test]
    fn lss_loads_ss_and_offset() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true,
        };
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true,
        };
        // Far pointer at 0x2000: offset=0x8000, selector=0x10.
        cpu.mem.write_u32(0x2000, 0x8000);
        cpu.mem.write_u16(0x2004, 0x10);
        // lss eax, [0x2000] = 0F B2 05 disp32
        cpu.eip = 0x1000;
        cpu.mem.load(0x1000, &[
            0x0F, 0xB2, 0x05, 0x00, 0x20, 0x00, 0x00, // lss eax, [0x2000]
            0xF4,
        ]);
        cpu.run(32);
        assert_eq!(cpu.eax, 0x8000);
        assert_eq!(cpu.ss, 0x10);
        assert!(cpu.halted);
    }

    #[test]
    fn rep_stosd_uses_ecx_in_32bit_mode() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true,
        };
        cpu.seg_desc[SegReg::Es as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true,
        };
        cpu.ecx = 4;
        cpu.edi = 0x3000;
        cpu.eax = 0xDEADBEEF;
        // rep stosd (F3 AB) ; hlt
        cpu.eip = 0x1000;
        cpu.mem.load(0x1000, &[
            0xF3, 0xAB,
            0xF4,
        ]);
        cpu.run(32);
        // 4 dwords written at 0x3000..0x3010.
        for i in 0..4 {
            assert_eq!(cpu.mem.read_u32(0x3000 + i * 4), 0xDEADBEEF);
        }
        assert_eq!(cpu.ecx, 0);
        assert_eq!(cpu.edi, 0x3010);
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
        assert_eq!(cpu.eax, 1); // highest basic leaf
        assert_eq!(cpu.ebx, 0x756E6547); // "Genu"
        assert_eq!(cpu.edx, 0x49656E69); // "ineI"
        assert_eq!(cpu.ecx, 0x6C65746E); // "ntel"
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
        assert_eq!(cpu.eax, 0x00000600);
        // TSC bit (bit 4) must be set in EDX.
        assert!(cpu.edx & (1 << 4) != 0);
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
        assert_eq!(cpu.eax, 0x9ABC_DEF0);
        assert_eq!(cpu.edx, 0x1234_5678);
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
        assert_eq!(cpu.eax, 0);
        assert_eq!(cpu.edx, 0);
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
        assert_eq!(cpu.eax, 0x8); // bit 3 set
        assert!(!cpu.get_flag(flags::CF)); // was 0 before
    }

    #[test]
    fn mov_moffs32_uses_addrsize() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true,
        };
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true,
        };
        // mov [0x12345678], al (A2 moffs32) ; hlt
        // In 32-bit addressing mode the moffs is 32-bit.
        cpu.eip = 0x1000;
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
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true,
        };
        cpu.eip = 0x1000;
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
        assert_eq!(cpu.eip, 0x100E);
    }

    #[test]
    fn movzx_zero_extends_8bit() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true,
        };
        cpu.eip = 0x1000;
        // movzx eax, al (0F B6 C0) ; hlt
        cpu.mem.load(0x1000, &[
            0x0F, 0xB6, 0xC0,
            0xF4,
        ]);
        cpu.set_reg8(Reg8::Al, 0xFF);
        cpu.run(32);
        assert_eq!(cpu.eax, 0xFF);
        assert!(cpu.halted);
    }

    #[test]
    fn movsx_sign_extends_8bit() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.cs = 0x08;
        cpu.seg_desc[SegReg::Cs as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true,
        };
        cpu.eip = 0x1000;
        // movsx eax, al (0F BE C0) ; hlt
        cpu.mem.load(0x1000, &[
            0x0F, 0xBE, 0xC0,
            0xF4,
        ]);
        cpu.set_reg8(Reg8::Al, 0xFF);
        cpu.run(32);
        assert_eq!(cpu.eax, 0xFFFF_FFFF);
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
        assert_eq!(cpu.ebx, 0x12345000);
    }

    #[test]
    fn paging_translates_linear_to_physical() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        // Flat data segment: base 0, limit 4 GiB.
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true,
        };
        // Page directory at 0x1000, page table at 0x2000.
        // Map linear 0x0040_0000 (PD 1, PT 0) to physical 0x1000.
        cpu.mem.write_u32(0x1000 + 1 * 4, 0x2003);
        cpu.mem.write_u32(0x2000 + 0 * 4, 0x1003);
        cpu.cr3 = 0x1000;
        cpu.cr0 = 0x8000_0000; // PG set
        // Write a value at physical 0x1000, read it via linear 0x0040_0000.
        cpu.mem.write_u32(0x1000, 0xCAFEBABE);
        let phys = cpu.translate(SegReg::Ds, 0x0040_0000);
        assert_eq!(phys, 0x1000);
        assert_eq!(cpu.mem.read_u32(phys), 0xCAFEBABE);
    }

    #[test]
    fn paging_disabled_identity_maps() {
        let mut cpu = Cpu::new();
        cpu.pe = true;
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true,
        };
        // PG clear: linear == physical.
        cpu.cr0 = 0;
        assert_eq!(cpu.translate(SegReg::Ds, 0x1234), 0x1234);
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
        cpu.esp = 0x0100;
        // Flat data segment: base 0.
        cpu.seg_desc[SegReg::Ds as usize] = Descriptor {
            base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true,
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
        cpu.eip = 0x1000;
        cpu.run(32);
        // The #PF handler ran: EAX = 0xCAFE.
        assert_eq!(cpu.eax, 0xCAFE);
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
        let code = Descriptor { base: 0, limit: 0xFFFF_FFFF, attr: 0x9A, g: true, d_b: true };
        let data = Descriptor { base: 0, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true };
        cpu.seg_desc[SegReg::Cs as usize] = code;
        for s in [SegReg::Ds, SegReg::Es, SegReg::Ss, SegReg::Fs, SegReg::Gs] {
            cpu.seg_desc[s as usize] = data;
        }
        cpu.eip = 0x1000;
        cpu.esp = 0x8000;
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
        assert_eq!(cpu.eax, 10, "CMP must not modify its destination");
        assert_eq!(cpu.ebx, 3);
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
        assert_eq!(cpu.eax, 3);
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
        assert_eq!(cpu.edx, 0x0E11_A000);
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
        assert_eq!(cpu.eax, 0x4000_0000);
        assert_eq!(cpu.ebx, 0x4000_0000);
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
        assert_eq!(cpu.eax, 0xF800_0000);
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
        assert_eq!(cpu.eax, 0x0000_0005);
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
        assert_eq!(cpu.eax, 0x1111_1111);
        assert_eq!(cpu.ebx, 0x2222_2222);
        assert_eq!(cpu.ecx, 0x3333_3333);
        assert_eq!(cpu.esp, 0x8000, "POPAD must restore the stack pointer");
    }

    #[test]
    fn pop_rm_writes_memory() {
        let cpu = run32(&[
            0x68, 0xEF, 0xBE, 0xAD, 0xDE, // push 0xDEADBEEF
            0x8F, 0x05, 0x00, 0x30, 0x00, 0x00, // pop dword [0x3000]
            0xF4,
        ]);
        assert_eq!(cpu.mem.read_u32(0x3000), 0xDEAD_BEEF);
        assert_eq!(cpu.esp, 0x8000);
    }

    #[test]
    fn push_imm8_pushes_four_bytes_in_32_bit_mode() {
        // A two-byte push here misaligns the stack for everything after it.
        let cpu = run32(&[
            0x6A, 0xFF, // push -1
            0xF4,
        ]);
        assert_eq!(cpu.esp, 0x7FFC);
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
        assert_eq!(cpu.ebx & 0xFF, 1);
        assert_eq!((cpu.ebx >> 8) & 0xFF, 0);
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
        assert_eq!(cpu.ebx, 42);
        assert_eq!(cpu.ecx, 21);
    }

    #[test]
    fn imul_sets_carry_when_the_product_does_not_fit() {
        let cpu = run32(&[
            0xB8, 0x00, 0x00, 0x00, 0x40, // mov eax, 0x40000000
            0x6B, 0xC0, 0x04,             // imul eax, eax, 4
            0xF4,
        ]);
        assert_eq!(cpu.eax, 0, "low half of the product");
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
        assert_eq!(cpu.eax, 0x0000_FFFF);
    }

    #[test]
    fn bsf_and_bsr_find_the_end_bits() {
        let cpu = run32(&[
            0xB8, 0x00, 0x01, 0x00, 0x01, // mov eax, 0x01000100
            0x0F, 0xBC, 0xD8,             // bsf ebx, eax
            0x0F, 0xBD, 0xC8,             // bsr ecx, eax
            0xF4,
        ]);
        assert_eq!(cpu.ebx, 8);
        assert_eq!(cpu.ecx, 24);
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
        assert_eq!(cpu.ebx, 0x99);
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
        assert_eq!(cpu.eax, 0xAAAA_AAAA);
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
        assert_eq!(cpu.eax, 7, "accumulator takes the destination's value");
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
        assert_eq!(cpu.ebx, 10, "source register gets the old destination");
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
        assert_eq!(cpu.eax, 0x7856_3412);
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
        assert_eq!(cpu.ebx, 1);
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
        assert_eq!(cpu.ebp, 0x7000);
        assert_eq!(cpu.esp, 0x8000);
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
        assert_eq!(cpu.esp, 0x8000, "the pushed argument was dropped too");
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
        assert_eq!(cpu.ebx & 0x0020_0000, 0x0020_0000, "ID bit is writable");
        assert_eq!(cpu.esp, 0x8000);
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
        assert_eq!(cpu.esp, 0x8000);
    }

    #[test]
    fn debug_registers_read_back_what_was_written() {
        let cpu = run32(&[
            0xB8, 0x55, 0x00, 0x00, 0x00, // mov eax, 0x55
            0x0F, 0x23, 0xF8,             // mov dr7, eax
            0x0F, 0x21, 0xFB,             // mov ebx, dr7
            0xF4,
        ]);
        assert_eq!(cpu.ebx, 0x55);
    }

    #[test]
    fn moffs_honours_a_segment_override() {
        // `mov %gs:0xC,%eax` is how i386 userspace reads thread-local storage.
        // Translating it through DS instead reads address 0xC.
        let mut cpu = flat32();
        cpu.seg_desc[SegReg::Gs as usize] = Descriptor {
            base: 0x5000, limit: 0xFFFF_FFFF, attr: 0x92, g: true, d_b: true,
        };
        cpu.mem.write_u32(0x500C, 0xFEED_FACE);
        cpu.mem.write_u32(0x000C, 0xDEAD_0000);
        cpu.mem.load(0x1000, &[
            0x65, 0xA1, 0x0C, 0x00, 0x00, 0x00, // mov eax, gs:0xC
            0xF4,
        ]);
        cpu.run(64);
        assert_eq!(cpu.eax, 0xFEED_FACE);
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
        cpu.eip = 0x10000 + 0x1000 - 6;
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
        assert_eq!(cpu.edx, 10, "EDX must be untouched by the faulted add");
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
        assert_eq!(cpu.edi, 0x1_0000, "EDI stops at the faulting element");
        assert_eq!(cpu.ecx, 0x100 - 0x40, "and so does the count");
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
        let esp = cpu.esp as usize;
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
        assert_eq!(cpu.ecx, 4, "CL must write into the register ECX shares");
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
