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
    MovAccMem16 { addr: u16 },
    MovMem16Acc { addr: u16 },
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
    PushImm16 { imm: u16 },
    PushImm8 { imm: u8 },
    PushImm32 { imm: u32 },
    JmpRel8 { rel: i8 },
    JmpRel16 { rel: i16 },
    JmpRel32 { rel: i32 },
    Jcc { cond: Cond, rel: i8 },
    CallRel16 { rel: i16 },
    CallRel32 { rel: i32 },
    Ret,
    Ret32,
    XchgAxReg { reg: Reg16 },
    XchgEaxReg { reg: Reg32 },
    Int { vector: u8 },
    Int3,
    Into,
    Iret,
    Iret32,
    Pushf,
    Popf,
    // Shifts / rotates (group 2, 0xD0-0xD3)
    Shift { op: ShiftOp, m: ModRm, w: bool, count: ShiftCount },
    // Shifts / rotates with imm8 count (group 2, 0xC0-0xC1)
    ShiftImm { op: ShiftOp, m: ModRm, w: bool, imm: u8 },
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
    // MOV r32, cr (0x0F 0x20) / MOV cr, r32 (0x0F 0x22)
    MovCr { cr: u8, reg: u8 },
    MovToCr { cr: u8, reg: u8 },
    // CPUID (0x0F 0xA2) / RDTSC (0x0F 0x31)
    Cpuid,
    Rdtsc,
    // RDMSR (0x0F 0x32) / WRMSR (0x0F 0x30)
    Rdmsr,
    Wrmsr,
    // Bit tests: BT/BTS/BTR/BTC (0F A3/AB/B3/BB, and group 8 0F BA /4-/7)
    Bt { m: ModRm, bit: u8 },
    Bts { m: ModRm, bit: u8 },
    Btr { m: ModRm, bit: u8 },
    Btc { m: ModRm, bit: u8 },
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
    Unknown { opcode: u8 },
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
pub enum ShiftCount { One, Cl }

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
        let peek_addr = cpu.phys_ip();
        let peek = cpu.mem.read_u8(peek_addr);
        match peek {
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

        // PUSH imm8 (0x6A) / PUSH imm16 (0x68) / PUSH imm32 (0x68 w/ 66)
        0x6A => Inst::PushImm8 { imm: cpu.fetch_u8() },
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
        0xD0 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, w: false, count: ShiftCount::One } }
        0xD1 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, w: !w32, count: ShiftCount::One } }
        0xD2 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, w: false, count: ShiftCount::Cl } }
        0xD3 => { let m = cpu.fetch_modrm(); Inst::Shift { op: ShiftOp::from_index(m.reg), m, w: !w32, count: ShiftCount::Cl } }
        // Group 2 shifts/rotates with imm8 count: 0xC0 (r/m8, imm8),
        // 0xC1 (r/m16/32, imm8)
        0xC0 => { let m = cpu.fetch_modrm(); let imm = cpu.fetch_u8(); Inst::ShiftImm { op: ShiftOp::from_index(m.reg), m, w: false, imm } }
        0xC1 => { let m = cpu.fetch_modrm(); let imm = cpu.fetch_u8(); Inst::ShiftImm { op: ShiftOp::from_index(m.reg), m, w: !w32, imm } }

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
        0xFF => {
            let m = cpu.fetch_modrm();
            match m.reg & 7 {
                0 => { if w32 { Inst::IncRm32 { m } } else { Inst::IncRm16 { m } } }
                1 => { if w32 { Inst::DecRm32 { m } } else { Inst::DecRm16 { m } } }
                2 => { if w32 { Inst::CallRm32 { m } } else { Inst::CallRm16 { m } } }
                4 => { if w32 { Inst::JmpRm32 { m } } else { Inst::JmpRm16 { m } } }
                6 => { if w32 { Inst::PushRm32 { m } } else { Inst::PushRm16 { m } } }
                _ => Inst::Unknown { opcode: 0xFF },
            }
        }

        // MOV AL, moffs8 / MOV moffs8, AL (0xA0/0xA2)
        0xA0 => Inst::MovAccMem8 { addr: cpu.fetch_u16() },
        0xA2 => Inst::MovMem8Acc { addr: cpu.fetch_u16() },
        // TEST AL, imm8 (0xA8) / TEST AX/EAX, imm (0xA9)
        0xA8 => Inst::TestAccImm8 { imm: cpu.fetch_u8() },
        0xA9 => {
            if w32 { Inst::TestAccImm32 { imm: cpu.fetch_u32() } }
            else { Inst::TestAccImm16 { imm: cpu.fetch_u16() } }
        }
        // MOV AX/EAX, moffs / MOV moffs, AX/EAX (0xA1/0xA3)
        0xA1 => {
            if w32 { Inst::MovAccMem32 { addr: cpu.fetch_u32() } }
            else { Inst::MovAccMem16 { addr: cpu.fetch_u16() } }
        }
        0xA3 => {
            if w32 { Inst::MovMem32Acc { addr: cpu.fetch_u32() } }
            else { Inst::MovMem16Acc { addr: cpu.fetch_u16() } }
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
                        _ => Inst::Unknown { opcode: 0x0F },
                    }
                }
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
                0xA3 => { let m = cpu.fetch_modrm(); Inst::Bt { m, bit: m.reg } }
                0xAB => { let m = cpu.fetch_modrm(); Inst::Bts { m, bit: m.reg } }
                0xB3 => { let m = cpu.fetch_modrm(); Inst::Btr { m, bit: m.reg } }
                0xBB => { let m = cpu.fetch_modrm(); Inst::Btc { m, bit: m.reg } }
                // Load segment with pointer: LSS (0F B2) / LFS (0F B4) / LGS (0F B5)
                0xB2 => { let m = cpu.fetch_modrm(); Inst::Lss { m } }
                0xB4 => { let m = cpu.fetch_modrm(); Inst::Lfs { m } }
                0xB5 => { let m = cpu.fetch_modrm(); Inst::Lgs { m } }
                // Group 8 (0F BA /4-/7): bit tests with imm8
                0xBA => {
                    let m = cpu.fetch_modrm();
                    let imm = cpu.fetch_u8();
                    match m.reg & 7 {
                        4 => Inst::Bt { m, bit: imm },
                        5 => Inst::Bts { m, bit: imm },
                        6 => Inst::Btr { m, bit: imm },
                        _ => Inst::Btc { m, bit: imm },
                    }
                }
                _ => Inst::Unknown { opcode: 0x0F },
            }
        }

        _ => Inst::Unknown { opcode: op },
    }
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
            let phys = cpu.translate(SegReg::Ds, addr as u32);
            cpu.set_reg8(Reg8::Al, cpu.mem.read_u8(phys));
        }
        Inst::MovMem8Acc { addr } => {
            let phys = cpu.translate(SegReg::Ds, addr as u32);
            cpu.mem.write_u8(phys, cpu.reg8(Reg8::Al));
        }
        Inst::MovAccMem16 { addr } => {
            let phys = cpu.translate(SegReg::Ds, addr as u32);
            cpu.set_reg16(Reg16::Ax, cpu.mem.read_u16(phys));
        }
        Inst::MovMem16Acc { addr } => {
            let phys = cpu.translate(SegReg::Ds, addr as u32);
            cpu.mem.write_u16(phys, cpu.reg16(Reg16::Ax));
        }
        Inst::MovAccMem32 { addr } => {
            let phys = cpu.translate(SegReg::Ds, addr);
            cpu.set_reg32(Reg32::Eax, cpu.mem.read_u32(phys));
        }
        Inst::MovMem32Acc { addr } => {
            let phys = cpu.translate(SegReg::Ds, addr);
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
            if store { cpu.write_rm8(&m, result); } else { cpu.set_reg8(Reg::reg8(reg), result); }
        }
        Inst::AluRm16Reg { op, m, reg, dir } => {
            let regv = cpu.reg16(Reg::reg16(reg));
            let rmv = cpu.read_rm16(&m);
            let (a, b, store) = match dir {
                Dir::RmReg => (rmv, regv, true),
                Dir::RegRm => (regv, rmv, false),
            };
            let result = alu16(cpu, op, a, b);
            if store { cpu.write_rm16(&m, result); } else { cpu.set_reg16(Reg::reg16(reg), result); }
        }
        Inst::AluRm32Reg { op, m, reg, dir } => {
            let regv = cpu.reg32(Reg::reg32(reg));
            let rmv = cpu.read_rm32(&m);
            let (a, b, store) = match dir {
                Dir::RmReg => (rmv, regv, true),
                Dir::RegRm => (regv, rmv, false),
            };
            let result = alu32(cpu, op, a, b);
            if store { cpu.write_rm32(&m, result); } else { cpu.set_reg32(Reg::reg32(reg), result); }
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
        Inst::PushReg16 { src } => cpu.push16(cpu.reg16(src)),
        Inst::PopReg16 { dst } => { let v = cpu.pop16(); cpu.set_reg16(dst, v); }
        Inst::PushReg32 { src } => cpu.push32(cpu.reg32(src)),
        Inst::PopReg32 { dst } => { let v = cpu.pop32(); cpu.set_reg32(dst, v); }
        Inst::PushImm16 { imm } => cpu.push16(imm),
        Inst::PushImm8 { imm } => cpu.push16(imm as i8 as i16 as u16),
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
                let flags = cpu.flags;
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
            cpu.flags = cpu.pop16();
            cpu.servicing_irq = false;
        }
        Inst::Iret32 => {
            cpu.eip = cpu.pop32();
            cpu.cs = cpu.pop32() as u16;
            cpu.flags = cpu.pop32() as u16;
            cpu.servicing_irq = false;
        }
        Inst::Pushf => cpu.push16(cpu.flags),
        Inst::Popf => { cpu.flags = cpu.pop16(); }

        // ---- Shifts / rotates (group 2) ----
        Inst::Shift { op, m, w, count } => {
            let n = match count {
                ShiftCount::One => 1,
                ShiftCount::Cl => cpu.reg8(Reg8::Cl) as u32,
            };
            if w {
                let v = cpu.read_rm16(&m);
                let r = shift16(cpu, op, v, n);
                cpu.write_rm16(&m, r);
            } else {
                let v = cpu.read_rm8(&m);
                let r = shift8(cpu, op, v, n);
                cpu.write_rm8(&m, r);
            }
        }
        Inst::ShiftImm { op, m, w, imm } => {
            let n = imm as u32;
            if w {
                let v = cpu.read_rm16(&m);
                let r = shift16(cpu, op, v, n);
                cpu.write_rm16(&m, r);
            } else {
                let v = cpu.read_rm8(&m);
                let r = shift8(cpu, op, v, n);
                cpu.write_rm8(&m, r);
            }
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
                    LoopCond::Jcxz => cpu.cx == 0,
                    _ => {
                        cpu.cx = cpu.cx.wrapping_sub(1);
                        match cond {
                            LoopCond::Loop => cpu.cx != 0,
                            LoopCond::Loopz => cpu.cx != 0 && cpu.get_flag(ZF),
                            LoopCond::Loopnz => cpu.cx != 0 && !cpu.get_flag(ZF),
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

        // ---- String ops ----
        // Element size: byte forms (w=false) are 8-bit; word forms (w=true)
        // are 16-bit in 16-bit mode and 32-bit in 32-bit mode. The index
        // registers and REP counter follow the operand size.
        Inst::Movs { rep, w } => {
            let delta: i32 = if cpu.get_flag(DF) { -1 } else { 1 };
            let esize: i32 = if w { if cpu.opsize { 4 } else { 2 } } else { 1 };
            if cpu.opsize {
                let mut count = match rep { Rep::None => 1, _ => cpu.ecx as i32 };
                while count > 0 {
                    let src = cpu.translate(SegReg::Ds, cpu.esi);
                    let dst = cpu.translate(SegReg::Es, cpu.edi);
                    if esize == 4 {
                        let v = cpu.mem.read_u32(src);
                        cpu.mem.write_u32(dst, v);
                    } else {
                        let v = cpu.mem.read_u8(src);
                        cpu.mem.write_u8(dst, v);
                    }
                    cpu.esi = cpu.esi.wrapping_add((delta * esize) as u32);
                    cpu.edi = cpu.edi.wrapping_add((delta * esize) as u32);
                    count -= 1;
                }
                if rep != Rep::None { cpu.ecx = 0; }
            } else {
                let mut count = match rep { Rep::None => 1, _ => cpu.cx as i32 };
                while count > 0 {
                    let src = cpu.translate(SegReg::Ds, cpu.si as u32);
                    let dst = cpu.translate(SegReg::Es, cpu.di as u32);
                    if esize == 2 {
                        let v = cpu.mem.read_u16(src);
                        cpu.mem.write_u16(dst, v);
                    } else {
                        let v = cpu.mem.read_u8(src);
                        cpu.mem.write_u8(dst, v);
                    }
                    cpu.si = cpu.si.wrapping_add((delta * esize) as u16);
                    cpu.di = cpu.di.wrapping_add((delta * esize) as u16);
                    count -= 1;
                }
                if rep != Rep::None { cpu.cx = 0; }
            }
        }
        Inst::Stos { rep, w } => {
            let delta: i32 = if cpu.get_flag(DF) { -1 } else { 1 };
            let esize: i32 = if w { if cpu.opsize { 4 } else { 2 } } else { 1 };
            if cpu.opsize {
                let mut count = match rep { Rep::None => 1, _ => cpu.ecx as i32 };
                while count > 0 {
                    let dst = cpu.translate(SegReg::Es, cpu.edi);
                    if esize == 4 {
                        let v = cpu.reg32(Reg32::Eax);
                        cpu.mem.write_u32(dst, v);
                    } else {
                        let v = cpu.reg8(Reg8::Al);
                        cpu.mem.write_u8(dst, v);
                    }
                    cpu.edi = cpu.edi.wrapping_add((delta * esize) as u32);
                    count -= 1;
                }
                if rep != Rep::None { cpu.ecx = 0; }
            } else {
                let mut count = match rep { Rep::None => 1, _ => cpu.cx as i32 };
                while count > 0 {
                    let dst = cpu.translate(SegReg::Es, cpu.di as u32);
                    if esize == 2 {
                        let v = cpu.reg16(Reg16::Ax);
                        cpu.mem.write_u16(dst, v);
                    } else {
                        let v = cpu.reg8(Reg8::Al);
                        cpu.mem.write_u8(dst, v);
                    }
                    cpu.di = cpu.di.wrapping_add((delta * esize) as u16);
                    count -= 1;
                }
                if rep != Rep::None { cpu.cx = 0; }
            }
        }
        Inst::Lods { rep, w } => {
            let delta: i32 = if cpu.get_flag(DF) { -1 } else { 1 };
            let esize: i32 = if w { if cpu.opsize { 4 } else { 2 } } else { 1 };
            if cpu.opsize {
                let mut count = match rep { Rep::None => 1, _ => cpu.ecx as i32 };
                while count > 0 {
                    let src = cpu.translate(SegReg::Ds, cpu.esi);
                    if esize == 4 {
                        let v = cpu.mem.read_u32(src);
                        cpu.set_reg32(Reg32::Eax, v);
                    } else {
                        let v = cpu.mem.read_u8(src);
                        cpu.set_reg8(Reg8::Al, v);
                    }
                    cpu.esi = cpu.esi.wrapping_add((delta * esize) as u32);
                    count -= 1;
                }
                if rep != Rep::None { cpu.ecx = 0; }
            } else {
                let mut count = match rep { Rep::None => 1, _ => cpu.cx as i32 };
                while count > 0 {
                    let src = cpu.translate(SegReg::Ds, cpu.si as u32);
                    if esize == 2 {
                        let v = cpu.mem.read_u16(src);
                        cpu.set_reg16(Reg16::Ax, v);
                    } else {
                        let v = cpu.mem.read_u8(src);
                        cpu.set_reg8(Reg8::Al, v);
                    }
                    cpu.si = cpu.si.wrapping_add((delta * esize) as u16);
                    count -= 1;
                }
                if rep != Rep::None { cpu.cx = 0; }
            }
        }
        Inst::Cmps { rep, w } => {
            let delta: i32 = if cpu.get_flag(DF) { -1 } else { 1 };
            let esize: i32 = if w { if cpu.opsize { 4 } else { 2 } } else { 1 };
            if cpu.opsize {
                loop {
                    let src = cpu.translate(SegReg::Ds, cpu.esi);
                    let dst = cpu.translate(SegReg::Es, cpu.edi);
                    if esize == 4 {
                        let a = cpu.mem.read_u32(src);
                        let b = cpu.mem.read_u32(dst);
                        alu32(cpu, AluOp::Sub, a, b);
                    } else {
                        let a = cpu.mem.read_u8(src);
                        let b = cpu.mem.read_u8(dst);
                        alu8(cpu, AluOp::Sub, a, b);
                    }
                    cpu.esi = cpu.esi.wrapping_add((delta * esize) as u32);
                    cpu.edi = cpu.edi.wrapping_add((delta * esize) as u32);
                    match rep {
                        Rep::None => break,
                        Rep::Repe => {
                            cpu.ecx = cpu.ecx.wrapping_sub(1);
                            if cpu.ecx == 0 || !cpu.get_flag(ZF) { break; }
                        }
                        Rep::Repne => {
                            cpu.ecx = cpu.ecx.wrapping_sub(1);
                            if cpu.ecx == 0 || cpu.get_flag(ZF) { break; }
                        }
                    }
                }
            } else {
                loop {
                    let src = cpu.translate(SegReg::Ds, cpu.si as u32);
                    let dst = cpu.translate(SegReg::Es, cpu.di as u32);
                    if esize == 2 {
                        let a = cpu.mem.read_u16(src);
                        let b = cpu.mem.read_u16(dst);
                        alu16(cpu, AluOp::Sub, a, b);
                    } else {
                        let a = cpu.mem.read_u8(src);
                        let b = cpu.mem.read_u8(dst);
                        alu8(cpu, AluOp::Sub, a, b);
                    }
                    cpu.si = cpu.si.wrapping_add((delta * esize) as u16);
                    cpu.di = cpu.di.wrapping_add((delta * esize) as u16);
                    match rep {
                        Rep::None => break,
                        Rep::Repe => {
                            cpu.cx = cpu.cx.wrapping_sub(1);
                            if cpu.cx == 0 || !cpu.get_flag(ZF) { break; }
                        }
                        Rep::Repne => {
                            cpu.cx = cpu.cx.wrapping_sub(1);
                            if cpu.cx == 0 || cpu.get_flag(ZF) { break; }
                        }
                    }
                }
            }
        }
        Inst::Scas { rep, w } => {
            let delta: i32 = if cpu.get_flag(DF) { -1 } else { 1 };
            let esize: i32 = if w { if cpu.opsize { 4 } else { 2 } } else { 1 };
            if cpu.opsize {
                loop {
                    let dst = cpu.translate(SegReg::Es, cpu.edi);
                    if esize == 4 {
                        let a = cpu.reg32(Reg32::Eax);
                        let b = cpu.mem.read_u32(dst);
                        alu32(cpu, AluOp::Sub, a, b);
                    } else {
                        let a = cpu.reg8(Reg8::Al);
                        let b = cpu.mem.read_u8(dst);
                        alu8(cpu, AluOp::Sub, a, b);
                    }
                    cpu.edi = cpu.edi.wrapping_add((delta * esize) as u32);
                    match rep {
                        Rep::None => break,
                        Rep::Repe => {
                            cpu.ecx = cpu.ecx.wrapping_sub(1);
                            if cpu.ecx == 0 || !cpu.get_flag(ZF) { break; }
                        }
                        Rep::Repne => {
                            cpu.ecx = cpu.ecx.wrapping_sub(1);
                            if cpu.ecx == 0 || cpu.get_flag(ZF) { break; }
                        }
                    }
                }
            } else {
                loop {
                    let dst = cpu.translate(SegReg::Es, cpu.di as u32);
                    if esize == 2 {
                        let a = cpu.reg16(Reg16::Ax);
                        let b = cpu.mem.read_u16(dst);
                        alu16(cpu, AluOp::Sub, a, b);
                    } else {
                        let a = cpu.reg8(Reg8::Al);
                        let b = cpu.mem.read_u8(dst);
                        alu8(cpu, AluOp::Sub, a, b);
                    }
                    cpu.di = cpu.di.wrapping_add((delta * esize) as u16);
                    match rep {
                        Rep::None => break,
                        Rep::Repe => {
                            cpu.cx = cpu.cx.wrapping_sub(1);
                            if cpu.cx == 0 || !cpu.get_flag(ZF) { break; }
                        }
                        Rep::Repne => {
                            cpu.cx = cpu.cx.wrapping_sub(1);
                            if cpu.cx == 0 || cpu.get_flag(ZF) { break; }
                        }
                    }
                }
            }
        }

        // ---- LGDT / LIDT ----
        Inst::Lgdt { m } => {
            let base = cpu.modrm_addr(&m);
            let limit = cpu.mem.read_u16(base);
            let base32 = cpu.mem.read_u32(base + 2);
            cpu.gdt_base = base32;
            cpu.gdt_limit = limit;
        }
        Inst::Lidt { m } => {
            let base = cpu.modrm_addr(&m);
            let limit = cpu.mem.read_u16(base);
            let base32 = cpu.mem.read_u32(base + 2);
            cpu.idt_base = base32;
            cpu.idt_limit = limit;
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
                0 => cpu.cr0 = v,
                2 => cpu.cr2 = v,
                3 => cpu.cr3 = v,
                _ => cpu.cr4 = v,
            }
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
                    // Family 6, model 0, stepping 0. Feature flags: FPU,
                    // TSC, MSR, CX8, SEP, CMOV, PGE, MMX, FXSR, SSE, SSE2.
                    cpu.set_reg32(Reg32::Eax, 0x00000600);
                    cpu.set_reg32(Reg32::Ebx, 0x00000000);
                    cpu.set_reg32(Reg32::Ecx, 0x00000000);
                    cpu.set_reg32(Reg32::Edx, 0x178BFBFF);
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
        Inst::Bt { m, bit } => {
            if cpu.opsize {
                let v = cpu.read_rm32(&m);
                let b = (bit & 31) as u32;
                cpu.set_flag(flags::CF, (v >> b) & 1 != 0);
            } else {
                let v = cpu.read_rm16(&m);
                let b = (bit & 15) as u16;
                cpu.set_flag(flags::CF, (v >> b) & 1 != 0);
            }
        }
        Inst::Bts { m, bit } => {
            if cpu.opsize {
                let v = cpu.read_rm32(&m);
                let b = (bit & 31) as u32;
                cpu.set_flag(flags::CF, (v >> b) & 1 != 0);
                cpu.write_rm32(&m, v | (1 << b));
            } else {
                let v = cpu.read_rm16(&m);
                let b = (bit & 15) as u16;
                cpu.set_flag(flags::CF, (v >> b) & 1 != 0);
                cpu.write_rm16(&m, v | (1 << b));
            }
        }
        Inst::Btr { m, bit } => {
            if cpu.opsize {
                let v = cpu.read_rm32(&m);
                let b = (bit & 31) as u32;
                cpu.set_flag(flags::CF, (v >> b) & 1 != 0);
                cpu.write_rm32(&m, v & !(1 << b));
            } else {
                let v = cpu.read_rm16(&m);
                let b = (bit & 15) as u16;
                cpu.set_flag(flags::CF, (v >> b) & 1 != 0);
                cpu.write_rm16(&m, v & !(1 << b));
            }
        }
        Inst::Btc { m, bit } => {
            if cpu.opsize {
                let v = cpu.read_rm32(&m);
                let b = (bit & 31) as u32;
                cpu.set_flag(flags::CF, (v >> b) & 1 != 0);
                cpu.write_rm32(&m, v ^ (1 << b));
            } else {
                let v = cpu.read_rm16(&m);
                let b = (bit & 15) as u16;
                cpu.set_flag(flags::CF, (v >> b) & 1 != 0);
                cpu.write_rm16(&m, v ^ (1 << b));
            }
        }

        // ---- IN / OUT ----
        Inst::InAlImm { port } => {
            let v = cpu.port_in(port);
            cpu.set_reg8(Reg8::Al, v);
        }
        Inst::InAxImm { port } => {
            let v = cpu.port_in16(port as u16);
            cpu.set_reg16(Reg16::Ax, v);
        }
        Inst::InAlDx => {
            let v = cpu.port_in(cpu.dx as u8);
            cpu.set_reg8(Reg8::Al, v);
        }
        Inst::InAxDx => {
            let v = cpu.port_in16(cpu.dx as u16);
            cpu.set_reg16(Reg16::Ax, v);
        }
        Inst::OutImmAl { port } => {
            let v = cpu.reg8(Reg8::Al);
            cpu.port_out(port, v);
        }
        Inst::OutImmAx { port } => {
            let v = cpu.reg16(Reg16::Ax);
            cpu.port_out16(port as u16, v);
        }
        Inst::OutDxAl => {
            let v = cpu.reg8(Reg8::Al);
            cpu.port_out(cpu.dx as u8, v);
        }
        Inst::OutDxAx => {
            let v = cpu.reg16(Reg16::Ax);
            cpu.port_out16(cpu.dx as u16, v);
        }

        // ---- Flag-control instructions ----
        Inst::Clc => cpu.set_flag(flags::CF, false),
        Inst::Stc => cpu.set_flag(flags::CF, true),
        Inst::Cli => cpu.set_flag(flags::IF, false),
        Inst::Sti => cpu.set_flag(flags::IF, true),
        Inst::Cld => cpu.set_flag(flags::DF, false),
        Inst::Std => cpu.set_flag(flags::DF, true),
        Inst::Cmc => cpu.set_flag(flags::CF, !cpu.get_flag(flags::CF)),

        Inst::Unknown { opcode } => {
            // Invalid opcode exception (#UD, vector 0x06). No error code.
            cpu.pending_exception = Some((0x06, None));
            cpu.dx = opcode as u16;
        }
    }
}

/// Dispatch a protected-mode interrupt through the IDT.
pub(crate) fn protected_int(cpu: &mut Cpu, vector: u8) {
    // IDT entry: 8 bytes. offset = (bytes 0-1) | (bytes 6-7 << 16).
    let entry = cpu.idt_base.wrapping_add((vector as u32) * 8);
    let addr = Memory::phys32(entry);
    let off_lo = cpu.mem.read_u16(addr) as u32;
    let off_hi = cpu.mem.read_u16(addr + 6) as u32;
    let target = off_lo | (off_hi << 16);
    let selector = cpu.mem.read_u16(addr + 2);
    // Push a 32-bit interrupt frame: FLAGS, CS, EIP.
    cpu.push32(cpu.flags as u32);
    cpu.push32(cpu.cs as u32);
    cpu.push32(cpu.eip);
    cpu.set_flag(flags::IF, false);
    cpu.set_flag(flags::TF, false);
    cpu.cs = selector;
    cpu.eip = target;
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
        } else if m.mod_field != 3 {
            ea = ea.wrapping_add(cpu.reg32(Reg::reg32(m.rm)));
        }
        if let Some(d32) = m.disp32 { ea = ea.wrapping_add(d32); }
        ea
    } else {
        let base = match m.rm {
            0 => cpu.bx.wrapping_add(cpu.si),
            1 => cpu.bx.wrapping_add(cpu.di),
            2 => cpu.bp.wrapping_add(cpu.si),
            3 => cpu.bp.wrapping_add(cpu.di),
            4 => cpu.si,
            5 => cpu.di,
            6 => cpu.bp,
            _ => cpu.bx,
        };
        let mut ea = base as u32;
        if let Some(d8) = m.disp8 { ea = ea.wrapping_add(d8 as u32); }
        if let Some(d16) = m.disp16 { ea = ea.wrapping_add(d16 as u32); }
        ea
    }
}

/// Perform an 8-bit shift/rotate, setting flags, and return the result.
fn shift8(cpu: &mut Cpu, op: ShiftOp, v: u8, n: u32) -> u8 {
    use flags::*;
    let n = n & 0x1F;
    if n == 0 { return v; }
    match op {
        ShiftOp::Shl => {
            let r = v.wrapping_shl(n);
            let cf = if n <= 8 { (v >> (8 - n)) & 1 != 0 } else { false };
            cpu.set_flag(CF, cf);
            set_logic_flags8(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x80) != 0);
            r
        }
        ShiftOp::Shr => {
            let r = v.wrapping_shr(n);
            let cf = if n <= 8 { (v >> (n - 1)) & 1 != 0 } else { false };
            cpu.set_flag(CF, cf);
            set_logic_flags8(cpu, r);
            cpu.set_flag(OF, n == 1 && (v & 0x80) != 0);
            r
        }
        ShiftOp::Sar => {
            let r = ((v as i8) >> n) as u8;
            let cf = if n <= 8 { (v >> (n - 1)) & 1 != 0 } else { false };
            cpu.set_flag(CF, cf);
            set_logic_flags8(cpu, r);
            cpu.set_flag(OF, false);
            r
        }
        ShiftOp::Rol => {
            let r = v.rotate_left(n);
            cpu.set_flag(CF, r & 1 != 0);
            set_logic_flags8(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x80) != 0);
            r
        }
        ShiftOp::Ror => {
            let r = v.rotate_right(n);
            cpu.set_flag(CF, (r & 0x80) != 0);
            set_logic_flags8(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x80) != 0);
            r
        }
        ShiftOp::Rcl => {
            let mut wide = (v as u16) | ((cpu.get_flag(CF) as u16) << 8);
            wide = wide.rotate_left(n);
            cpu.set_flag(CF, (wide >> 8) & 1 != 0);
            let r = wide as u8;
            set_logic_flags8(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x80) != 0);
            r
        }
        ShiftOp::Rcr => {
            let mut wide = (v as u16) | ((cpu.get_flag(CF) as u16) << 8);
            wide = wide.rotate_right(n);
            cpu.set_flag(CF, (wide >> 8) & 1 != 0);
            let r = wide as u8;
            set_logic_flags8(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x80) != 0);
            r
        }
    }
}

/// Perform a 16-bit shift/rotate, setting flags, and return the result.
fn shift16(cpu: &mut Cpu, op: ShiftOp, v: u16, n: u32) -> u16 {
    use flags::*;
    let n = n & 0x1F;
    if n == 0 { return v; }
    match op {
        ShiftOp::Shl => {
            let r = v.wrapping_shl(n);
            let cf = if n <= 16 { (v >> (16 - n)) & 1 != 0 } else { false };
            cpu.set_flag(CF, cf);
            set_logic_flags16(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x8000) != 0);
            r
        }
        ShiftOp::Shr => {
            let r = v.wrapping_shr(n);
            let cf = if n <= 16 { (v >> (n - 1)) & 1 != 0 } else { false };
            cpu.set_flag(CF, cf);
            set_logic_flags16(cpu, r);
            cpu.set_flag(OF, n == 1 && (v & 0x8000) != 0);
            r
        }
        ShiftOp::Sar => {
            let r = ((v as i16) >> n) as u16;
            let cf = if n <= 16 { (v >> (n - 1)) & 1 != 0 } else { false };
            cpu.set_flag(CF, cf);
            set_logic_flags16(cpu, r);
            cpu.set_flag(OF, false);
            r
        }
        ShiftOp::Rol => {
            let r = v.rotate_left(n);
            cpu.set_flag(CF, r & 1 != 0);
            set_logic_flags16(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x8000) != 0);
            r
        }
        ShiftOp::Ror => {
            let r = v.rotate_right(n);
            cpu.set_flag(CF, (r & 0x8000) != 0);
            set_logic_flags16(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x8000) != 0);
            r
        }
        ShiftOp::Rcl => {
            let mut wide = (v as u32) | ((cpu.get_flag(CF) as u32) << 16);
            wide = wide.rotate_left(n);
            cpu.set_flag(CF, (wide >> 16) & 1 != 0);
            let r = wide as u16;
            set_logic_flags16(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x8000) != 0);
            r
        }
        ShiftOp::Rcr => {
            let mut wide = (v as u32) | ((cpu.get_flag(CF) as u32) << 16);
            wide = wide.rotate_right(n);
            cpu.set_flag(CF, (wide >> 16) & 1 != 0);
            let r = wide as u16;
            set_logic_flags16(cpu, r);
            cpu.set_flag(OF, n == 1 && ((v ^ r) & 0x8000) != 0);
            r
        }
    }
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
        assert_eq!(cpu.ax, 0x1236);
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
        assert_eq!(cpu.ax, 0xFFFF);
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
        cpu.sp = 0x0100;
        load(&mut cpu, &[
            0xE8, 0x01, 0x00,
            0xF4,
            0xB8, 0x99, 0x00,
            0xC3,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax, 0x0099);
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
        assert_eq!(cpu.ax, 0x4242);
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
        assert_eq!(cpu.ax, 0);
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
        assert_eq!(cpu.ax, 0x0000);
        assert!(cpu.get_flag(flags::ZF));
        assert!(cpu.get_flag(flags::CF));
    }

    #[test]
    fn int_iret_through_ivt() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.sp = 0x0100;
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
        assert_eq!(cpu.ax, 0x0099);
        assert!(cpu.halted);
        assert_eq!(cpu.sp, 0x0100);
    }

    #[test]
    fn pushf_popf_roundtrip() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.sp = 0x0100;
        cpu.set_flag(flags::CF, true);
        cpu.set_flag(flags::ZF, true);
        load(&mut cpu, &[
            0x9C,
            0x58,
            0xF4,
        ]);
        cpu.run(16);
        assert!(cpu.ax & flags::CF != 0);
        assert!(cpu.ax & flags::ZF != 0);
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
        assert_eq!(cpu.ax, 0x0000);
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
        assert_eq!(cpu.ax, 0x2000);
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
        assert_eq!(cpu.ax, 0x0002);
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
        assert_eq!(cpu.ax, 0x0000);
        assert_eq!(cpu.dx, 0x0001);
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
        assert_eq!(cpu.ax, 0x0003);
        assert_eq!(cpu.dx, 0x0004);
    }

    #[test]
    fn rep_movsb_copies_string() {
        let mut cpu = Cpu::new();
        cpu.ds = 0;
        cpu.es = 0;
        cpu.si = 0x0100;
        cpu.di = 0x0200;
        cpu.cx = 3;
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
        assert_eq!(cpu.cx, 0);
        assert_eq!(cpu.si, 0x0103);
        assert_eq!(cpu.di, 0x0203);
    }

    #[test]
    fn loop_decrements_cx() {
        let mut cpu = Cpu::new();
        cpu.cx = 3;
        load(&mut cpu, &[
            0xE2, 0xFE,
            0xF4,
        ]);
        cpu.run(64);
        assert!(cpu.halted);
        assert_eq!(cpu.cx, 0);
    }

    #[test]
    fn lea_loads_effective_address() {
        let mut cpu = Cpu::new();
        cpu.bx = 0x0100;
        cpu.si = 0x0020;
        load(&mut cpu, &[
            0x8D, 0x00,
            0xF4,
        ]);
        cpu.run(16);
        assert_eq!(cpu.ax, 0x0120);
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
        assert_eq!(cpu.dx, 0xFFFF);
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
        assert_eq!(cpu.ax, 0xFFFB);
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
        assert_eq!(cpu.ax, 0x5678);
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
        //   limit15:0 = 0xFFFF
        //   base23:16 = 0x00
        //   type/attr = 0x92 (present, data read/write)
        //   limit19:16 = 0xF, G=1, D=1
        //   base31:24 = 0x00
        let desc: u64 = 0x00CF_9200_0001_FFFF;
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
        cpu.sp = 0x0100;
        // Install an IVT entry for vector 0x00 (#DE) -> handler at 0x0000:0x0300.
        cpu.mem.write_u16(0x00 * 4, 0x0300);
        cpu.mem.write_u16(0x00 * 4 + 2, 0x0000);
        // Handler: mov ax, 0xDEAD ; iret
        cpu.mem.load(0x0300, &[
            0xB8, 0xAD, 0xDE,
            0xCF,
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
        assert_eq!(cpu.ax, 0xDEAD);
        assert!(cpu.halted);
        // Stack restored after the exception frame.
        assert_eq!(cpu.sp, 0x0100);
    }

    #[test]
    fn int3_raises_bp() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.sp = 0x0100;
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
        assert_eq!(cpu.ax, 0x1234);
        assert!(cpu.halted);
    }

    #[test]
    fn into_raises_of_when_overflow_set() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.sp = 0x0100;
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
        assert_eq!(cpu.ax, 0x7777);
        assert!(cpu.halted);
    }

    #[test]
    fn invalid_opcode_raises_ud() {
        let mut cpu = Cpu::new();
        cpu.ss = 0;
        cpu.sp = 0x0100;
        // IVT entry for vector 0x06 (#UD) -> handler at 0x0000:0x0300.
        cpu.mem.write_u16(0x06 * 4, 0x0300);
        cpu.mem.write_u16(0x06 * 4 + 2, 0x0000);
        // Handler: mov ax, 0xBEEF ; iret
        cpu.mem.load(0x0300, &[
            0xB8, 0xEF, 0xBE,
            0xCF,
        ]);
        // 0x0F 0xFF is an invalid opcode (not implemented) ; hlt
        load(&mut cpu, &[
            0x0F, 0xFF,
            0xF4,
        ]);
        cpu.run(32);
        assert_eq!(cpu.ax, 0xBEEF);
        assert!(cpu.halted);
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
        // Handler at 0x5000: mov eax, 0xCAFE ; iret (32-bit)
        cpu.mem.load(0x5000, &[
            0x66, 0xB8, 0xFE, 0xCA, 0x00, 0x00,
            0x66, 0xCF,
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
        cpu.mem.load(0x1000, &[
            0x66, 0xA1, 0x00, 0x00, 0x40, 0x00,
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
    }
}
