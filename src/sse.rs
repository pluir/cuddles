//! SSE / SSE2 / SSE3: the sixteen 128-bit XMM registers, MXCSR, and the
//! `0F xx` instructions that work on them.
//!
//! This is the layer a 64-bit userspace cannot run without. The x86-64 ABI
//! passes floating-point arguments in XMM registers and assumes SSE2, so a
//! 64-bit libc uses these instructions for things that have nothing to do
//! with floating point: musl's malloc initialises a bin sentinel with
//! `movq %rax,%xmm0; punpcklqdq %xmm0,%xmm0; movups %xmm0,(%rax)`. Get a
//! store wrong and the heap is corrupt before `main` runs.
//!
//! **Encoding.** One opcode byte after `0F` names a *family*, and the
//! *mandatory prefix* -- none, `66`, `F3` or `F2`, whichever legacy prefix
//! came last before the REX/opcode -- picks the data type: packed single,
//! packed double, scalar single, scalar double (for the FP families), or
//! MMX vs XMM (for the integer ones). The prefix loop in `decode` records
//! it in `Cpu::sse_pfx`; `decode_sse` reads it. Only the XMM forms are
//! implemented: the MMX ones (no prefix on an integer opcode) are `#UD`,
//! and CPUID says as much.
//!
//! **Operand width is per instruction, not per family.** A scalar op reads
//! exactly 4 or 8 bytes from memory, never 16 -- a 16-byte read of a
//! `float` at the end of a mapped page would fault where hardware would
//! not. `MOVAPS`/`MOVDQA` require 16-byte alignment (`#GP` otherwise);
//! `MOVUPS`/`MOVDQU` do not.
//!
//! **What is not modelled**, and says so rather than pretending: MXCSR's
//! exception flags are not raised or accumulated (all exceptions are
//! masked, which is how every OS runs), and the rounding-control field is
//! honoured by the conversions but not by the arithmetic (which rounds to
//! nearest, as the host does). Denormals-are-zero and flush-to-zero are
//! ignored.

use crate::cpu::{Cpu, SegReg, flags, CR4_OSFXSR};
use crate::instructions::Inst;
use crate::modrm::ModRm;

/// The mandatory prefix, as a data type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pfx {
    /// No prefix: packed single (`ps`), or MMX for the integer opcodes.
    None,
    /// `66`: packed double (`pd`), or XMM for the integer opcodes.
    P66,
    /// `F3`: scalar single (`ss`).
    F3,
    /// `F2`: scalar double (`sd`).
    F2,
}

/// Saturation mode for the packed integer add/subtract families.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sat { Wrap, Signed, Unsigned }

/// A packed shift.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShiftKind { Sll, Srl, Sra }

/// A floating-point binary op shared by the `ps`/`pd`/`ss`/`sd` families.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FpOp { Add, Sub, Mul, Div, Min, Max }

/// The operation. Element widths are in bits where a family has several.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    // ---- Moves ----
    /// 128-bit load/store: MOVUPS/MOVUPD/MOVDQU (unaligned) and
    /// MOVAPS/MOVAPD/MOVDQA (`aligned`, #GP on a misaligned address).
    Mov128 { store: bool, aligned: bool },
    /// MOVNTPS/MOVNTPD/MOVNTDQ: a 128-bit store (no cache to bypass here).
    Movnt128,
    /// MOVSS/MOVSD: scalar. From memory the upper bits are zeroed; between
    /// registers only the low element moves.
    MovScalar { store: bool, bits: u32 },
    /// MOVLPS/MOVLPD (`high: false`) and MOVHPS/MOVHPD (`high: true`): move
    /// one 64-bit half to or from memory, the other half untouched.
    MovHalf { store: bool, high: bool },
    /// MOVHLPS: high half of src to low half of dst.
    Movhlps,
    /// MOVLHPS: low half of src to high half of dst.
    Movlhps,
    /// MOVD/MOVQ between a general register or memory and an XMM register.
    /// `bits` is 32 or 64 (REX.W).
    MovGpr { store: bool, bits: u32 },
    /// MOVQ xmm, xmm/m64 (F3 0F 7E): low 64 bits, upper zeroed.
    MovqLoad,
    /// MOVQ xmm/m64, xmm (66 0F D6): low 64 bits.
    MovqStore,
    /// MOVDDUP / MOVSHDUP / MOVSLDUP (SSE3): load with duplication.
    Movddup,
    Movshdup,
    Movsldup,
    /// MOVMSKPS/MOVMSKPD/PMOVMSKB: sign bits to a general register.
    Movmsk { bits: u32 },
    /// MOVNTI (0F C3): a general-register store.
    Movnti,
    /// MASKMOVDQU: byte-masked store to [RDI].
    Maskmovdqu,

    // ---- Bitwise ----
    And, Andn, Or, Xor,

    // ---- Floating point ----
    Arith(FpOp),
    Sqrt,
    Rsqrt,
    Rcp,
    /// CMPPS/CMPPD/CMPSS/CMPSD with the predicate in imm8.
    Cmp,
    /// COMISS/COMISD (`signal: true`) and UCOMISS/UCOMISD.
    Comis { signal: bool },
    Shuf,
    Unpck { high: bool },
    /// SSE3 horizontal and add/sub ops.
    Addsub,
    Hadd,
    Hsub,

    // ---- Conversions ----
    /// CVTSI2SS/CVTSI2SD: r/m32 or r/m64 (REX.W) to scalar.
    CvtSi2 { bits: u32 },
    /// CVT(T)SS2SI / CVT(T)SD2SI: scalar to r32/r64.
    Cvt2Si { trunc: bool, bits: u32 },
    /// CVTSS2SD / CVTSD2SS / CVTPS2PD / CVTPD2PS.
    CvtSs2Sd,
    CvtSd2Ss,
    CvtPs2Pd,
    CvtPd2Ps,
    /// CVTDQ2PS / CVT(T)PS2DQ / CVTDQ2PD / CVT(T)PD2DQ.
    CvtDq2Ps,
    CvtPs2Dq { trunc: bool },
    CvtDq2Pd,
    CvtPd2Dq { trunc: bool },

    // ---- Packed integer ----
    Padd { bits: u32, sat: Sat },
    Psub { bits: u32, sat: Sat },
    Pmullw,
    Pmulhw,
    Pmulhuw,
    Pmuludq,
    Pmaddwd,
    Psadbw,
    Pavg { bits: u32 },
    Pmax { bits: u32, signed: bool },
    Pmin { bits: u32, signed: bool },
    Pcmpeq { bits: u32 },
    Pcmpgt { bits: u32 },
    Packss { bits: u32 },
    Packus,
    Punpckl { bits: u32 },
    Punpckh { bits: u32 },
    Pshufd,
    Pshufhw,
    Pshuflw,
    Pinsrw,
    Pextrw,
    /// Packed shift by imm8 (`imm: true`) or by the count in an XMM/m128.
    Pshift { kind: ShiftKind, bits: u32, imm: bool },
    /// PSLLDQ / PSRLDQ: whole-register byte shifts.
    Pshiftdq { left: bool },

    // ---- State ----
    Ldmxcsr,
    Stmxcsr,
    Fxsave,
    Fxrstor,
}

/// A decoded SSE instruction. `reg` is the REX-extended `reg` field (an XMM
/// or general register, per `op`); the other operand is `m`.
#[derive(Clone, Copy, Debug)]
pub struct SseInst {
    pub op: Op,
    pub m: ModRm,
    pub reg: u8,
    pub imm: u8,
    pub pfx: Pfx,
}

/// The mandatory prefix the prefix loop recorded.
fn pfx(cpu: &Cpu) -> Pfx {
    match cpu.sse_pfx {
        Some(0x66) => Pfx::P66,
        Some(0xF3) => Pfx::F3,
        Some(0xF2) => Pfx::F2,
        _ => Pfx::None,
    }
}

/// Decode `0F <op2>` as an SSE instruction. `None` means the opcode is not
/// one of ours (or is an MMX form, which this CPU does not have) and the
/// caller should report it as unknown.
pub fn decode_sse(cpu: &mut Cpu, op2: u8) -> Option<Inst> {
    let p = pfx(cpu);
    let mut m = cpu.fetch_modrm();
    // Every ModR/M-taking SSE instruction has an XMM (or GPR) in `reg` and
    // an XMM/memory operand in `rm`. `m.rm` carries REX.B when it names a
    // register; `m.reg` carries REX.R.
    let reg = m.reg;
    let mut imm = 0u8;
    // A width for the MOVD/MOVQ and conversion families: REX.W selects 64.
    let gbits = if cpu.rex_w { 64 } else { 32 };
    let op = match op2 {
        // ---- 0F 10-17: moves ----
        0x10 | 0x11 => {
            let store = op2 == 0x11;
            match p {
                Pfx::None | Pfx::P66 => Op::Mov128 { store, aligned: false },
                Pfx::F3 => Op::MovScalar { store, bits: 32 },
                Pfx::F2 => Op::MovScalar { store, bits: 64 },
            }
        }
        0x12 => match p {
            Pfx::F2 => Op::Movddup,
            Pfx::F3 => Op::Movsldup,
            _ if m.is_reg() => Op::Movhlps,
            _ => Op::MovHalf { store: false, high: false },
        },
        0x13 => Op::MovHalf { store: true, high: false },
        0x14 => Op::Unpck { high: false },
        0x15 => Op::Unpck { high: true },
        0x16 => match p {
            Pfx::F3 => Op::Movshdup,
            _ if m.is_reg() && p == Pfx::None => Op::Movlhps,
            _ => Op::MovHalf { store: false, high: true },
        },
        0x17 => Op::MovHalf { store: true, high: true },
        0x28 => Op::Mov128 { store: false, aligned: true },
        0x29 => Op::Mov128 { store: true, aligned: true },
        0x2A => match p {
            Pfx::F3 | Pfx::F2 => Op::CvtSi2 { bits: gbits },
            _ => return None, // CVTPI2PS/PD: MMX source
        },
        0x2B => Op::Movnt128,
        0x2C | 0x2D => match p {
            Pfx::F3 | Pfx::F2 => Op::Cvt2Si { trunc: op2 == 0x2C, bits: gbits },
            _ => return None, // CVT(T)PS2PI: MMX destination
        },
        0x2E => Op::Comis { signal: false },
        0x2F => Op::Comis { signal: true },

        // ---- 0F 50-5F: floating point ----
        0x50 => Op::Movmsk { bits: if p == Pfx::P66 { 64 } else { 32 } },
        0x51 => Op::Sqrt,
        0x52 => Op::Rsqrt,
        0x53 => Op::Rcp,
        0x54 => Op::And,
        0x55 => Op::Andn,
        0x56 => Op::Or,
        0x57 => Op::Xor,
        0x58 => Op::Arith(FpOp::Add),
        0x59 => Op::Arith(FpOp::Mul),
        0x5A => match p {
            Pfx::None => Op::CvtPs2Pd,
            Pfx::P66 => Op::CvtPd2Ps,
            Pfx::F3 => Op::CvtSs2Sd,
            Pfx::F2 => Op::CvtSd2Ss,
        },
        0x5B => match p {
            Pfx::None => Op::CvtDq2Ps,
            Pfx::P66 => Op::CvtPs2Dq { trunc: false },
            Pfx::F3 => Op::CvtPs2Dq { trunc: true },
            Pfx::F2 => return None,
        },
        0x5C => Op::Arith(FpOp::Sub),
        0x5D => Op::Arith(FpOp::Min),
        0x5E => Op::Arith(FpOp::Div),
        0x5F => Op::Arith(FpOp::Max),

        // ---- 0F 60-6F: packed integer (66 = XMM; no prefix = MMX) ----
        0x60..=0x6D | 0x6F if p == Pfx::None && op2 != 0x6E => return None,
        0x60 => Op::Punpckl { bits: 8 },
        0x61 => Op::Punpckl { bits: 16 },
        0x62 => Op::Punpckl { bits: 32 },
        0x63 => Op::Packss { bits: 16 },
        0x64 => Op::Pcmpgt { bits: 8 },
        0x65 => Op::Pcmpgt { bits: 16 },
        0x66 => Op::Pcmpgt { bits: 32 },
        0x67 => Op::Packus,
        0x68 => Op::Punpckh { bits: 8 },
        0x69 => Op::Punpckh { bits: 16 },
        0x6A => Op::Punpckh { bits: 32 },
        0x6B => Op::Packss { bits: 32 },
        0x6C => Op::Punpckl { bits: 64 },
        0x6D => Op::Punpckh { bits: 64 },
        // MOVD/MOVQ xmm, r/m32/64. The no-prefix form is the MMX one.
        0x6E => if p == Pfx::P66 { Op::MovGpr { store: false, bits: gbits } } else { return None },
        // MOVDQA (66) / MOVDQU (F3) load.
        0x6F => match p {
            Pfx::P66 => Op::Mov128 { store: false, aligned: true },
            Pfx::F3 => Op::Mov128 { store: false, aligned: false },
            _ => return None,
        },
        // PSHUFD (66) / PSHUFHW (F3) / PSHUFLW (F2); no prefix is MMX PSHUFW.
        0x70 => {
            imm = cpu.fetch_u8();
            match p {
                Pfx::P66 => Op::Pshufd,
                Pfx::F3 => Op::Pshufhw,
                Pfx::F2 => Op::Pshuflw,
                Pfx::None => return None,
            }
        }
        // 0F 71/72/73: shift groups by imm8. The XMM register is in `rm`.
        0x71 | 0x72 | 0x73 => {
            if p != Pfx::P66 { return None; }
            imm = cpu.fetch_u8();
            let bits = match op2 { 0x71 => 16, 0x72 => 32, _ => 64 };
            match (op2, m.reg & 7) {
                (_, 2) => Op::Pshift { kind: ShiftKind::Srl, bits, imm: true },
                (0x71, 4) | (0x72, 4) => Op::Pshift { kind: ShiftKind::Sra, bits, imm: true },
                (_, 6) => Op::Pshift { kind: ShiftKind::Sll, bits, imm: true },
                (0x73, 3) => Op::Pshiftdq { left: false },
                (0x73, 7) => Op::Pshiftdq { left: true },
                _ => return None,
            }
        }
        0x74 => Op::Pcmpeq { bits: 8 },
        0x75 => Op::Pcmpeq { bits: 16 },
        0x76 => Op::Pcmpeq { bits: 32 },
        // SSE3 horizontal ops.
        0x7C => match p { Pfx::P66 | Pfx::F2 => Op::Hadd, _ => return None },
        0x7D => match p { Pfx::P66 | Pfx::F2 => Op::Hsub, _ => return None },
        // MOVD/MOVQ r/m, xmm (66 0F 7E) and MOVQ xmm, xmm/m64 (F3 0F 7E).
        0x7E => match p {
            Pfx::P66 => Op::MovGpr { store: true, bits: gbits },
            Pfx::F3 => Op::MovqLoad,
            _ => return None,
        },
        // MOVDQA (66) / MOVDQU (F3) store.
        0x7F => match p {
            Pfx::P66 => Op::Mov128 { store: true, aligned: true },
            Pfx::F3 => Op::Mov128 { store: true, aligned: false },
            _ => return None,
        },

        // ---- 0F C2-C6 ----
        0xC2 => { imm = cpu.fetch_u8(); Op::Cmp }
        0xC3 => if p == Pfx::None && !m.is_reg() { Op::Movnti } else { return None },
        0xC4 => { if p != Pfx::P66 { return None; } imm = cpu.fetch_u8(); Op::Pinsrw }
        0xC5 => { if p != Pfx::P66 { return None; } imm = cpu.fetch_u8(); Op::Pextrw }
        0xC6 => { imm = cpu.fetch_u8(); Op::Shuf }

        // ---- 0F D0-FF: packed integer, and the SSE3 ADDSUB ----
        0xD0 => match p { Pfx::P66 | Pfx::F2 => Op::Addsub, _ => return None },
        0xD6 => if p == Pfx::P66 { Op::MovqStore } else { return None },
        0xF0 => if p == Pfx::F2 { Op::Mov128 { store: false, aligned: false } } else { return None }, // LDDQU
        0xE6 => match p {
            Pfx::F3 => Op::CvtDq2Pd,
            Pfx::P66 => Op::CvtPd2Dq { trunc: false },
            Pfx::F2 => Op::CvtPd2Dq { trunc: true },
            Pfx::None => return None,
        },
        0xD1..=0xFE => {
            if p != Pfx::P66 { return None; }
            match op2 {
                0xD1 => Op::Pshift { kind: ShiftKind::Srl, bits: 16, imm: false },
                0xD2 => Op::Pshift { kind: ShiftKind::Srl, bits: 32, imm: false },
                0xD3 => Op::Pshift { kind: ShiftKind::Srl, bits: 64, imm: false },
                0xD4 => Op::Padd { bits: 64, sat: Sat::Wrap },
                0xD5 => Op::Pmullw,
                0xD7 => Op::Movmsk { bits: 8 },
                0xD8 => Op::Psub { bits: 8, sat: Sat::Unsigned },
                0xD9 => Op::Psub { bits: 16, sat: Sat::Unsigned },
                0xDA => Op::Pmin { bits: 8, signed: false },
                0xDB => Op::And,
                0xDC => Op::Padd { bits: 8, sat: Sat::Unsigned },
                0xDD => Op::Padd { bits: 16, sat: Sat::Unsigned },
                0xDE => Op::Pmax { bits: 8, signed: false },
                0xDF => Op::Andn,
                0xE0 => Op::Pavg { bits: 8 },
                0xE1 => Op::Pshift { kind: ShiftKind::Sra, bits: 16, imm: false },
                0xE2 => Op::Pshift { kind: ShiftKind::Sra, bits: 32, imm: false },
                0xE3 => Op::Pavg { bits: 16 },
                0xE4 => Op::Pmulhuw,
                0xE5 => Op::Pmulhw,
                0xE7 => Op::Movnt128,
                0xE8 => Op::Psub { bits: 8, sat: Sat::Signed },
                0xE9 => Op::Psub { bits: 16, sat: Sat::Signed },
                0xEA => Op::Pmin { bits: 16, signed: true },
                0xEB => Op::Or,
                0xEC => Op::Padd { bits: 8, sat: Sat::Signed },
                0xED => Op::Padd { bits: 16, sat: Sat::Signed },
                0xEE => Op::Pmax { bits: 16, signed: true },
                0xEF => Op::Xor,
                0xF1 => Op::Pshift { kind: ShiftKind::Sll, bits: 16, imm: false },
                0xF2 => Op::Pshift { kind: ShiftKind::Sll, bits: 32, imm: false },
                0xF3 => Op::Pshift { kind: ShiftKind::Sll, bits: 64, imm: false },
                0xF4 => Op::Pmuludq,
                0xF5 => Op::Pmaddwd,
                0xF6 => Op::Psadbw,
                0xF7 => Op::Maskmovdqu,
                0xF8 => Op::Psub { bits: 8, sat: Sat::Wrap },
                0xF9 => Op::Psub { bits: 16, sat: Sat::Wrap },
                0xFA => Op::Psub { bits: 32, sat: Sat::Wrap },
                0xFB => Op::Psub { bits: 64, sat: Sat::Wrap },
                0xFC => Op::Padd { bits: 8, sat: Sat::Wrap },
                0xFD => Op::Padd { bits: 16, sat: Sat::Wrap },
                0xFE => Op::Padd { bits: 32, sat: Sat::Wrap },
                _ => return None,
            }
        }
        _ => return None,
    };
    // The shift-by-immediate groups name their XMM operand in `rm`; keep
    // `reg` pointing at it so the executor has one shape.
    let reg = match op {
        Op::Pshift { imm: true, .. } | Op::Pshiftdq { .. } => { let r = m.rm; m.mod_field = 3; r }
        _ => reg,
    };
    Some(Inst::Sse(SseInst { op, m, reg, imm, pfx: p }))
}

/// The `0F AE` group's memory forms: FXSAVE (/0), FXRSTOR (/1), LDMXCSR
/// (/2), STMXCSR (/3). The register forms are the fences and are the
/// caller's (they need no SSE state).
pub fn decode_0f_ae(m: ModRm) -> Option<Inst> {
    let op = match m.reg & 7 {
        0 => Op::Fxsave,
        1 => Op::Fxrstor,
        2 => Op::Ldmxcsr,
        3 => Op::Stmxcsr,
        _ => return None,
    };
    Some(Inst::Sse(SseInst { op, m, reg: m.reg, imm: 0, pfx: Pfx::None }))
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// The MXCSR value after reset: all six exceptions masked, round to nearest.
pub const MXCSR_DEFAULT: u32 = 0x1F80;
/// The writable MXCSR bits (reserved ones #GP when set by LDMXCSR).
pub const MXCSR_MASK: u32 = 0xFFFF;

const LO64: u128 = 0xFFFF_FFFF_FFFF_FFFF;
const LO32: u128 = 0xFFFF_FFFF;

/// Split a 128-bit value into `n` lanes of `bits` each, low lane first.
fn lanes(v: u128, bits: u32) -> Vec<u64> {
    let n = 128 / bits;
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    (0..n).map(|i| ((v >> (i * bits)) as u64) & mask).collect()
}

/// Reassemble lanes into a 128-bit value.
fn join(l: &[u64], bits: u32) -> u128 {
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    l.iter().enumerate().fold(0u128, |acc, (i, &v)| acc | (((v & mask) as u128) << (i as u32 * bits)))
}

/// Sign-extend a `bits`-wide lane.
fn sx(v: u64, bits: u32) -> i64 {
    let s = 64 - bits;
    ((v << s) as i64) >> s
}

fn f32s(v: u128) -> [f32; 4] {
    let mut r = [0f32; 4];
    for (i, x) in r.iter_mut().enumerate() { *x = f32::from_bits((v >> (i * 32)) as u32); }
    r
}
fn f64s(v: u128) -> [f64; 2] {
    [f64::from_bits(v as u64), f64::from_bits((v >> 64) as u64)]
}
fn from_f32s(a: [f32; 4]) -> u128 {
    a.iter().enumerate().fold(0u128, |acc, (i, x)| acc | ((x.to_bits() as u128) << (i * 32)))
}
fn from_f64s(a: [f64; 2]) -> u128 {
    (a[0].to_bits() as u128) | ((a[1].to_bits() as u128) << 64)
}

/// Round to an integer under MXCSR.RC (bits 13-14): nearest-even, down, up,
/// toward zero. `trunc` forces toward zero (the CVTT forms).
fn round_rc(v: f64, rc: u32, trunc: bool) -> f64 {
    if trunc { return v.trunc(); }
    match rc {
        0 => v.round_ties_even(),
        1 => v.floor(),
        2 => v.ceil(),
        _ => v.trunc(),
    }
}

/// Float to signed integer of `bits`, with the "integer indefinite" result
/// (the minimum value) for NaN and out-of-range inputs, as the hardware.
fn to_int(v: f64, rc: u32, trunc: bool, bits: u32) -> u64 {
    let r = round_rc(v, rc, trunc);
    if bits == 64 {
        if r.is_nan() || r < -9.223_372_036_854_775_808e18 || r >= 9.223_372_036_854_775_808e18 {
            0x8000_0000_0000_0000
        } else { r as i64 as u64 }
    } else if r.is_nan() || r < -2_147_483_648.0 || r >= 2_147_483_648.0 {
        0x8000_0000
    } else { (r as i32) as u32 as u64 }
}

/// Saturate a wide intermediate into a signed or unsigned lane of `bits`.
fn sat_signed(v: i64, bits: u32) -> u64 {
    let max = (1i64 << (bits - 1)) - 1;
    let min = -(1i64 << (bits - 1));
    let mask = (1u64 << bits) - 1;
    (v.clamp(min, max) as u64) & mask
}
fn sat_unsigned(v: i64, bits: u32) -> u64 {
    let max = (1i64 << bits) - 1;
    v.clamp(0, max) as u64
}

impl Cpu {
    /// Read the XMM-or-memory operand of an SSE instruction: `bytes` (4, 8
    /// or 16) from memory, or the named register, zero-extended to 128 bits.
    fn sse_src(&mut self, m: &ModRm, bytes: u32, aligned: bool) -> u128 {
        if m.is_reg() {
            return self.xmm[m.rm as usize];
        }
        let a = self.rm_addr(m, false);
        if self.pending_exception.is_some() { return 0; }
        if aligned && a & 15 != 0 { self.raise_gp(0); return 0; }
        if Self::straddles(a, bytes) {
            let lin = self.modrm_linear(m);
            return self.read_split(a, lin, bytes);
        }
        match bytes {
            16 => self.mem.read_u128(a),
            8 => self.mem.read_u64(a) as u128,
            _ => self.mem.read_u32(a) as u128,
        }
    }

    /// Store `bytes` of `v` to the memory operand, or the low bits to the
    /// register operand (the caller merges when it must).
    fn sse_store(&mut self, m: &ModRm, bytes: u32, v: u128, aligned: bool) {
        if m.is_reg() {
            match bytes {
                16 => self.set_xmm(m.rm, v),
                8 => { let old = self.xmm[m.rm as usize]; self.set_xmm(m.rm, (old & !LO64) | (v & LO64)); }
                _ => { let old = self.xmm[m.rm as usize]; self.set_xmm(m.rm, (old & !LO32) | (v & LO32)); }
            }
            return;
        }
        let a = self.rm_addr(m, true);
        if self.pending_exception.is_some() { return; }
        if aligned && a & 15 != 0 { self.raise_gp(0); return; }
        if Self::straddles(a, bytes) {
            let lin = self.modrm_linear(m);
            self.write_split(a, lin, bytes, v);
            return;
        }
        match bytes {
            16 => self.mem.write_u128(a, v),
            8 => self.mem.write_u64(a, v as u64),
            _ => self.mem.write_u32(a, v as u32),
        }
    }

    /// Write an XMM register as an instruction result: dropped once the
    /// instruction has faulted, like `set_reg_w`.
    #[inline]
    pub fn set_xmm(&mut self, i: u8, v: u128) {
        if self.pending_exception.is_some() { return; }
        self.xmm[(i & 15) as usize] = v;
    }

    /// The gate every SSE instruction passes: `#UD` when the OS has not
    /// enabled SSE (CR4.OSFXSR clear) or has emulation on (CR0.EM), `#NM`
    /// when the FPU is marked task-switched (CR0.TS), so a lazily-switching
    /// kernel gets its trap.
    fn sse_gate(&mut self) -> bool {
        if self.cr0 & 0x4 != 0 || self.cr4 & CR4_OSFXSR == 0 {
            self.raise_ud();
            return false;
        }
        if self.cr0 & 0x8 != 0 {
            if self.pending_exception.is_none() {
                self.pending_exception = Some((0x07, None));
            }
            return false;
        }
        true
    }

    /// Read `buf.len()` bytes at a linear address, translating page by
    /// page: FXSAVE's 512 bytes can straddle one.
    fn read_linear_bytes(&mut self, seg: SegReg, off: u64, buf: &mut [u8]) {
        let mut i = 0usize;
        while i < buf.len() {
            let lin = self.linear_addr(seg, off.wrapping_add(i as u64));
            let phys = self.apply_paging_access(lin, false);
            if self.pending_exception.is_some() { return; }
            let run = ((0x1000 - (lin & 0xFFF)) as usize).min(buf.len() - i);
            for k in 0..run { buf[i + k] = self.mem.read_u8(phys + k); }
            i += run;
        }
    }

    fn write_linear_bytes(&mut self, seg: SegReg, off: u64, buf: &[u8]) {
        let mut i = 0usize;
        while i < buf.len() {
            let lin = self.linear_addr(seg, off.wrapping_add(i as u64));
            let phys = self.apply_paging_access(lin, true);
            if self.pending_exception.is_some() { return; }
            let run = ((0x1000 - (lin & 0xFFF)) as usize).min(buf.len() - i);
            for k in 0..run { self.mem.write_u8(phys + k, buf[i + k]); }
            i += run;
        }
    }

    /// The (segment, offset) a memory operand names, for the page-by-page
    /// helpers above.
    fn sse_operand(&self, m: &ModRm) -> (SegReg, u64) {
        let (ea, default_seg) = if self.addrsize {
            self.modrm_ea32(m)
        } else {
            (self.modrm_offset(m) as u64, SegReg::Ds)
        };
        (self.operand_seg_for_exec(default_seg), ea)
    }
}

/// Convert an `f64` to the 80-bit extended format FXSAVE stores ST(i) in:
/// (64-bit significand with an explicit integer bit, 16-bit sign+exponent).
pub fn f64_to_f80(v: f64) -> (u64, u16) {
    let bits = v.to_bits();
    let sign = ((bits >> 63) & 1) as u16;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let mant = bits & 0x000F_FFFF_FFFF_FFFF;
    if exp == 0 && mant == 0 {
        return (0, sign << 15);
    }
    if exp == 0x7FF {
        // Infinity or NaN: integer bit set, payload shifted up.
        return ((mant << 11) | 0x8000_0000_0000_0000, (sign << 15) | 0x7FFF);
    }
    if exp == 0 {
        // A double denormal (mant * 2^-1074) is a normal number in extended
        // precision: shift the leading one up to the integer bit and take
        // the shift out of the exponent.
        let lz = mant.leading_zeros();
        let sig = mant << lz;
        let e = 15372 - lz as i32;
        return (sig, (sign << 15) | (e as u16));
    }
    ((mant << 11) | 0x8000_0000_0000_0000, (sign << 15) | (exp - 1023 + 16383) as u16)
}

/// The reverse: an 80-bit value back to `f64`, rounding the significand.
pub fn f80_to_f64(sig: u64, se: u16) -> f64 {
    let sign = ((se >> 15) & 1) as u64;
    let exp = (se & 0x7FFF) as i32;
    if exp == 0 && sig == 0 {
        return f64::from_bits(sign << 63);
    }
    if exp == 0x7FFF {
        let payload = (sig & 0x7FFF_FFFF_FFFF_FFFF) >> 11;
        return f64::from_bits((sign << 63) | (0x7FFu64 << 52) | payload);
    }
    // Normalise (the integer bit may be clear in a pseudo-denormal).
    let lz = sig.leading_zeros() as i32;
    let sig = sig << lz;
    let e = exp - lz - 16383 + 1023;
    if e >= 0x7FF {
        return f64::from_bits((sign << 63) | (0x7FFu64 << 52));
    }
    if e <= 0 {
        // Underflow into a double denormal (or zero).
        let shift = 12 - e;
        let m = if shift >= 64 { 0 } else { sig >> shift };
        return f64::from_bits((sign << 63) | m);
    }
    // Round the 63 fraction bits to 52 (nearest even).
    let frac = sig & 0x7FFF_FFFF_FFFF_FFFF;
    let mut m = frac >> 11;
    let rem = frac & 0x7FF;
    if rem > 0x400 || (rem == 0x400 && (m & 1) == 1) { m += 1; }
    let mut e = e as u64;
    if m >> 52 != 0 { m = 0; e += 1; }
    f64::from_bits((sign << 63) | (e << 52) | m)
}

/// Execute an SSE instruction.
pub fn execute_sse(cpu: &mut Cpu, s: &SseInst) {
    // LDMXCSR/STMXCSR/FXSAVE/FXRSTOR are gated on CR4.OSFXSR/CR0.EM but
    // FXSAVE/FXRSTOR are how the OS *turns SSE on* (they save state before
    // OSFXSR is set on some paths), so only the register-file instructions
    // take the full gate.
    match s.op {
        Op::Fxsave | Op::Fxrstor | Op::Ldmxcsr | Op::Stmxcsr => {}
        _ => if !cpu.sse_gate() { return; },
    }
    let m = &s.m;
    let d = s.reg;
    let rc = (cpu.mxcsr >> 13) & 3;
    match s.op {
        // ---- Moves ----
        Op::Mov128 { store, aligned } => {
            if store {
                let v = cpu.xmm[d as usize];
                cpu.sse_store(m, 16, v, aligned);
            } else {
                let v = cpu.sse_src(m, 16, aligned);
                cpu.set_xmm(d, v);
            }
        }
        Op::Movnt128 => {
            // A non-temporal store to a register operand is not encodable.
            if m.is_reg() { cpu.raise_ud(); return; }
            let v = cpu.xmm[d as usize];
            cpu.sse_store(m, 16, v, true);
        }
        Op::MovScalar { store, bits } => {
            let bytes = bits / 8;
            if store {
                let v = cpu.xmm[d as usize];
                cpu.sse_store(m, bytes, v, false);
            } else if m.is_reg() {
                // Register to register: only the low element moves.
                let src = cpu.xmm[m.rm as usize];
                let mask = if bits == 64 { LO64 } else { LO32 };
                let old = cpu.xmm[d as usize];
                cpu.set_xmm(d, (old & !mask) | (src & mask));
            } else {
                let v = cpu.sse_src(m, bytes, false);
                cpu.set_xmm(d, v);
            }
        }
        Op::MovHalf { store, high } => {
            if store {
                let v = cpu.xmm[d as usize];
                let half = if high { v >> 64 } else { v & LO64 };
                if m.is_reg() { cpu.raise_ud(); return; }
                cpu.sse_store(m, 8, half, false);
            } else {
                if m.is_reg() { cpu.raise_ud(); return; }
                let v = cpu.sse_src(m, 8, false);
                let old = cpu.xmm[d as usize];
                let new = if high { (old & LO64) | (v << 64) } else { (old & !LO64) | v };
                cpu.set_xmm(d, new);
            }
        }
        Op::Movhlps => {
            let src = cpu.xmm[m.rm as usize];
            let old = cpu.xmm[d as usize];
            cpu.set_xmm(d, (old & !LO64) | (src >> 64));
        }
        Op::Movlhps => {
            let src = cpu.xmm[m.rm as usize];
            let old = cpu.xmm[d as usize];
            cpu.set_xmm(d, (old & LO64) | ((src & LO64) << 64));
        }
        Op::MovGpr { store, bits } => {
            if store {
                let v = cpu.xmm[d as usize] as u64;
                let v = if bits == 32 { v as u32 as u64 } else { v };
                if m.is_reg() {
                    cpu.set_reg_w(m.rm, bits, v);
                } else {
                    let a = cpu.rm_addr(m, true);
                    if cpu.pending_exception.is_some() { return; }
                    if bits == 64 { cpu.mem.write_u64(a, v); } else { cpu.mem.write_u32(a, v as u32); }
                }
            } else {
                let v = if m.is_reg() {
                    cpu.reg_w(m.rm, bits)
                } else {
                    let a = cpu.rm_addr(m, false);
                    if cpu.pending_exception.is_some() { return; }
                    if bits == 64 { cpu.mem.read_u64(a) } else { cpu.mem.read_u32(a) as u64 }
                };
                cpu.set_xmm(d, v as u128);
            }
        }
        Op::MovqLoad => {
            let v = cpu.sse_src(m, 8, false) & LO64;
            cpu.set_xmm(d, v);
        }
        Op::MovqStore => {
            let v = cpu.xmm[d as usize] & LO64;
            if m.is_reg() {
                // MOVQ xmm, xmm zeroes the destination's upper half.
                cpu.set_xmm(m.rm, v);
            } else {
                cpu.sse_store(m, 8, v, false);
            }
        }
        Op::Movddup => {
            let v = cpu.sse_src(m, 8, false) & LO64;
            cpu.set_xmm(d, v | (v << 64));
        }
        Op::Movshdup | Op::Movsldup => {
            let v = f32s(cpu.sse_src(m, 16, false));
            let r = if s.op == Op::Movshdup { [v[1], v[1], v[3], v[3]] } else { [v[0], v[0], v[2], v[2]] };
            cpu.set_xmm(d, from_f32s(r));
        }
        Op::Movmsk { bits } => {
            // Sign bits of every lane, into a general register (always a
            // 32-bit result, zero-extended).
            let v = cpu.xmm[m.rm as usize];
            let mut r = 0u64;
            for (i, lane) in lanes(v, bits).iter().enumerate() {
                if lane >> (bits - 1) & 1 == 1 { r |= 1 << i; }
            }
            cpu.set_reg_w(d, 32, r);
        }
        Op::Movnti => {
            let w = cpu.osize();
            let w = if w == 16 { 32 } else { w };
            let v = cpu.reg_w(d, w);
            cpu.write_rm_w(m, w, v);
        }
        Op::Maskmovdqu => {
            let mask = cpu.xmm[m.rm as usize];
            let v = cpu.xmm[d as usize];
            let base = cpu.reg_w(7, if cpu.addrsize { 64 } else { 16 });
            let seg = cpu.operand_seg_for_exec(SegReg::Ds);
            for i in 0..16u32 {
                if (mask >> (i * 8 + 7)) & 1 == 1 {
                    let a = cpu.translate_write(seg, base.wrapping_add(i as u64));
                    if cpu.pending_exception.is_some() { return; }
                    cpu.mem.write_u8(a, (v >> (i * 8)) as u8);
                }
            }
        }

        // ---- Bitwise: identical for ps/pd/dq ----
        Op::And | Op::Andn | Op::Or | Op::Xor => {
            let src = cpu.sse_src(m, 16, true);
            let a = cpu.xmm[d as usize];
            let r = match s.op { Op::And => a & src, Op::Andn => !a & src, Op::Or => a | src, _ => a ^ src };
            cpu.set_xmm(d, r);
        }

        // ---- Floating point ----
        Op::Arith(op) => {
            let f32op = |a: f32, b: f32| -> f32 { match op {
                FpOp::Add => a + b, FpOp::Sub => a - b, FpOp::Mul => a * b, FpOp::Div => a / b,
                // MIN/MAX return the SECOND operand when either is NaN or
                // both are zero, which is what these comparisons give.
                FpOp::Min => if a < b { a } else { b },
                FpOp::Max => if a > b { a } else { b },
            } };
            let f64op = |a: f64, b: f64| -> f64 { match op {
                FpOp::Add => a + b, FpOp::Sub => a - b, FpOp::Mul => a * b, FpOp::Div => a / b,
                FpOp::Min => if a < b { a } else { b },
                FpOp::Max => if a > b { a } else { b },
            } };
            let a = cpu.xmm[d as usize];
            let r = match s.pfx {
                Pfx::None => {
                    let b = cpu.sse_src(m, 16, true);
                    let (x, y) = (f32s(a), f32s(b));
                    from_f32s([f32op(x[0], y[0]), f32op(x[1], y[1]), f32op(x[2], y[2]), f32op(x[3], y[3])])
                }
                Pfx::P66 => {
                    let b = cpu.sse_src(m, 16, true);
                    let (x, y) = (f64s(a), f64s(b));
                    from_f64s([f64op(x[0], y[0]), f64op(x[1], y[1])])
                }
                Pfx::F3 => {
                    let b = cpu.sse_src(m, 4, false);
                    let r = f32op(f32::from_bits(a as u32), f32::from_bits(b as u32));
                    (a & !LO32) | r.to_bits() as u128
                }
                Pfx::F2 => {
                    let b = cpu.sse_src(m, 8, false);
                    let r = f64op(f64::from_bits(a as u64), f64::from_bits(b as u64));
                    (a & !LO64) | r.to_bits() as u128
                }
            };
            cpu.set_xmm(d, r);
        }
        Op::Sqrt | Op::Rsqrt | Op::Rcp => {
            let f = |x: f32| -> f32 { match s.op { Op::Sqrt => x.sqrt(), Op::Rsqrt => 1.0 / x.sqrt(), _ => 1.0 / x } };
            let g = |x: f64| -> f64 { x.sqrt() };
            let a = cpu.xmm[d as usize];
            let r = match s.pfx {
                Pfx::None => { let b = f32s(cpu.sse_src(m, 16, true)); from_f32s([f(b[0]), f(b[1]), f(b[2]), f(b[3])]) }
                Pfx::F3 => { let b = cpu.sse_src(m, 4, false); (a & !LO32) | f(f32::from_bits(b as u32)).to_bits() as u128 }
                // RSQRT/RCP have no double forms; SQRTPD/SQRTSD do.
                Pfx::P66 => {
                    if s.op != Op::Sqrt { cpu.raise_ud(); return; }
                    let b = f64s(cpu.sse_src(m, 16, true)); from_f64s([g(b[0]), g(b[1])])
                }
                Pfx::F2 => {
                    if s.op != Op::Sqrt { cpu.raise_ud(); return; }
                    let b = cpu.sse_src(m, 8, false); (a & !LO64) | g(f64::from_bits(b as u64)).to_bits() as u128
                }
            };
            cpu.set_xmm(d, r);
        }
        Op::Cmp => {
            // The eight predicates, with NaN handled the way the manual
            // says: `unord` is true when either operand is NaN, and the
            // negated predicates (neq/nlt/nle) are true then too.
            let pred = s.imm & 7;
            let cmp64 = |a: f64, b: f64| -> bool { match pred {
                0 => a == b, 1 => a < b, 2 => a <= b, 3 => a.is_nan() || b.is_nan(),
                4 => a != b, 5 => !(a < b), 6 => !(a <= b), _ => !a.is_nan() && !b.is_nan(),
            } };
            let a = cpu.xmm[d as usize];
            let r = match s.pfx {
                Pfx::None => {
                    let b = cpu.sse_src(m, 16, true);
                    let (x, y) = (f32s(a), f32s(b));
                    let mut r = 0u128;
                    for i in 0..4 { if cmp64(x[i] as f64, y[i] as f64) { r |= LO32 << (i * 32); } }
                    r
                }
                Pfx::P66 => {
                    let b = cpu.sse_src(m, 16, true);
                    let (x, y) = (f64s(a), f64s(b));
                    let mut r = 0u128;
                    for i in 0..2 { if cmp64(x[i], y[i]) { r |= LO64 << (i * 64); } }
                    r
                }
                Pfx::F3 => {
                    let b = cpu.sse_src(m, 4, false);
                    let t = cmp64(f32::from_bits(a as u32) as f64, f32::from_bits(b as u32) as f64);
                    (a & !LO32) | if t { LO32 } else { 0 }
                }
                Pfx::F2 => {
                    let b = cpu.sse_src(m, 8, false);
                    let t = cmp64(f64::from_bits(a as u64), f64::from_bits(b as u64));
                    (a & !LO64) | if t { LO64 } else { 0 }
                }
            };
            cpu.set_xmm(d, r);
        }
        Op::Comis { .. } => {
            // ZF,PF,CF = 111 unordered, 000 greater, 001 less, 100 equal;
            // OF, SF and AF are cleared. (COMIS also signals on a quiet
            // NaN; nothing here raises #XM, so the two are one.)
            let (a, b) = if s.pfx == Pfx::P66 {
                let b = cpu.sse_src(m, 8, false);
                (f64::from_bits(cpu.xmm[d as usize] as u64), f64::from_bits(b as u64))
            } else {
                let b = cpu.sse_src(m, 4, false);
                (f32::from_bits(cpu.xmm[d as usize] as u32) as f64, f32::from_bits(b as u32) as f64)
            };
            if cpu.pending_exception.is_some() { return; }
            let (zf, pf, cf) = if a.is_nan() || b.is_nan() { (true, true, true) }
                else if a > b { (false, false, false) }
                else if a < b { (false, false, true) }
                else { (true, false, false) };
            cpu.set_flag(flags::ZF, zf);
            cpu.set_flag(flags::PF, pf);
            cpu.set_flag(flags::CF, cf);
            cpu.set_flag(flags::OF, false);
            cpu.set_flag(flags::SF, false);
            cpu.set_flag(flags::AF, false);
        }
        Op::Shuf => {
            let b = cpu.sse_src(m, 16, true);
            let a = cpu.xmm[d as usize];
            let imm = s.imm as usize;
            let r = if s.pfx == Pfx::P66 {
                let (x, y) = (f64s(a), f64s(b));
                from_f64s([x[imm & 1], y[(imm >> 1) & 1]])
            } else {
                let (x, y) = (f32s(a), f32s(b));
                from_f32s([x[imm & 3], x[(imm >> 2) & 3], y[(imm >> 4) & 3], y[(imm >> 6) & 3]])
            };
            cpu.set_xmm(d, r);
        }
        Op::Unpck { high } => {
            let b = cpu.sse_src(m, 16, true);
            let a = cpu.xmm[d as usize];
            let r = if s.pfx == Pfx::P66 {
                let (x, y) = (f64s(a), f64s(b));
                if high { from_f64s([x[1], y[1]]) } else { from_f64s([x[0], y[0]]) }
            } else {
                let (x, y) = (f32s(a), f32s(b));
                if high { from_f32s([x[2], y[2], x[3], y[3]]) } else { from_f32s([x[0], y[0], x[1], y[1]]) }
            };
            cpu.set_xmm(d, r);
        }
        Op::Addsub | Op::Hadd | Op::Hsub => {
            let b = cpu.sse_src(m, 16, true);
            let a = cpu.xmm[d as usize];
            let r = if s.pfx == Pfx::P66 {
                let (x, y) = (f64s(a), f64s(b));
                match s.op {
                    Op::Addsub => from_f64s([x[0] - y[0], x[1] + y[1]]),
                    Op::Hadd => from_f64s([x[0] + x[1], y[0] + y[1]]),
                    _ => from_f64s([x[0] - x[1], y[0] - y[1]]),
                }
            } else {
                let (x, y) = (f32s(a), f32s(b));
                match s.op {
                    Op::Addsub => from_f32s([x[0] - y[0], x[1] + y[1], x[2] - y[2], x[3] + y[3]]),
                    Op::Hadd => from_f32s([x[0] + x[1], x[2] + x[3], y[0] + y[1], y[2] + y[3]]),
                    _ => from_f32s([x[0] - x[1], x[2] - x[3], y[0] - y[1], y[2] - y[3]]),
                }
            };
            cpu.set_xmm(d, r);
        }

        // ---- Conversions ----
        Op::CvtSi2 { bits } => {
            let v = if m.is_reg() { cpu.reg_w(m.rm, bits) } else {
                let a = cpu.rm_addr(m, false);
                if cpu.pending_exception.is_some() { return; }
                if bits == 64 { cpu.mem.read_u64(a) } else { cpu.mem.read_u32(a) as u64 }
            };
            let i = if bits == 64 { v as i64 } else { v as i32 as i64 };
            let a = cpu.xmm[d as usize];
            let r = if s.pfx == Pfx::F3 { (a & !LO32) | (i as f32).to_bits() as u128 }
                    else { (a & !LO64) | (i as f64).to_bits() as u128 };
            cpu.set_xmm(d, r);
        }
        Op::Cvt2Si { trunc, bits } => {
            let v = if s.pfx == Pfx::F3 { f32::from_bits(cpu.sse_src(m, 4, false) as u32) as f64 }
                    else { f64::from_bits(cpu.sse_src(m, 8, false) as u64) };
            if cpu.pending_exception.is_some() { return; }
            let r = to_int(v, rc, trunc, bits);
            cpu.set_reg_w(d, bits, r);
        }
        Op::CvtSs2Sd => {
            let v = f32::from_bits(cpu.sse_src(m, 4, false) as u32) as f64;
            let a = cpu.xmm[d as usize];
            cpu.set_xmm(d, (a & !LO64) | v.to_bits() as u128);
        }
        Op::CvtSd2Ss => {
            let v = f64::from_bits(cpu.sse_src(m, 8, false) as u64) as f32;
            let a = cpu.xmm[d as usize];
            cpu.set_xmm(d, (a & !LO32) | v.to_bits() as u128);
        }
        Op::CvtPs2Pd => {
            let v = f32s(cpu.sse_src(m, 8, false));
            cpu.set_xmm(d, from_f64s([v[0] as f64, v[1] as f64]));
        }
        Op::CvtPd2Ps => {
            let v = f64s(cpu.sse_src(m, 16, true));
            cpu.set_xmm(d, from_f32s([v[0] as f32, v[1] as f32, 0.0, 0.0]));
        }
        Op::CvtDq2Ps => {
            let v = lanes(cpu.sse_src(m, 16, true), 32);
            let f = |x: u64| (x as u32 as i32) as f32;
            cpu.set_xmm(d, from_f32s([f(v[0]), f(v[1]), f(v[2]), f(v[3])]));
        }
        Op::CvtPs2Dq { trunc } => {
            let v = f32s(cpu.sse_src(m, 16, true));
            let l: Vec<u64> = v.iter().map(|&x| to_int(x as f64, rc, trunc, 32)).collect();
            cpu.set_xmm(d, join(&l, 32));
        }
        Op::CvtDq2Pd => {
            let v = lanes(cpu.sse_src(m, 8, false), 32);
            cpu.set_xmm(d, from_f64s([(v[0] as u32 as i32) as f64, (v[1] as u32 as i32) as f64]));
        }
        Op::CvtPd2Dq { trunc } => {
            let v = f64s(cpu.sse_src(m, 16, true));
            let l = [to_int(v[0], rc, trunc, 32), to_int(v[1], rc, trunc, 32), 0, 0];
            cpu.set_xmm(d, join(&l, 32));
        }

        // ---- Packed integer ----
        Op::Padd { bits, sat } | Op::Psub { bits, sat } => {
            let b = lanes(cpu.sse_src(m, 16, true), bits);
            let a = lanes(cpu.xmm[d as usize], bits);
            let sub = matches!(s.op, Op::Psub { .. });
            let r: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| match sat {
                Sat::Wrap => if sub { x.wrapping_sub(y) } else { x.wrapping_add(y) },
                Sat::Signed => {
                    let (x, y) = (sx(x, bits), sx(y, bits));
                    sat_signed(if sub { x - y } else { x + y }, bits)
                }
                Sat::Unsigned => sat_unsigned(if sub { x as i64 - y as i64 } else { x as i64 + y as i64 }, bits),
            }).collect();
            cpu.set_xmm(d, join(&r, bits));
        }
        Op::Pmullw | Op::Pmulhw | Op::Pmulhuw => {
            let b = lanes(cpu.sse_src(m, 16, true), 16);
            let a = lanes(cpu.xmm[d as usize], 16);
            let r: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| match s.op {
                Op::Pmullw => (sx(x, 16) * sx(y, 16)) as u64,
                Op::Pmulhw => ((sx(x, 16) * sx(y, 16)) >> 16) as u64,
                _ => (x * y) >> 16,
            }).collect();
            cpu.set_xmm(d, join(&r, 16));
        }
        Op::Pmuludq => {
            let b = lanes(cpu.sse_src(m, 16, true), 32);
            let a = lanes(cpu.xmm[d as usize], 32);
            cpu.set_xmm(d, join(&[a[0] * b[0], a[2] * b[2]], 64));
        }
        Op::Pmaddwd => {
            let b = lanes(cpu.sse_src(m, 16, true), 16);
            let a = lanes(cpu.xmm[d as usize], 16);
            let r: Vec<u64> = (0..4).map(|i| {
                (sx(a[2 * i], 16) * sx(b[2 * i], 16) + sx(a[2 * i + 1], 16) * sx(b[2 * i + 1], 16)) as u64
            }).collect();
            cpu.set_xmm(d, join(&r, 32));
        }
        Op::Psadbw => {
            let b = lanes(cpu.sse_src(m, 16, true), 8);
            let a = lanes(cpu.xmm[d as usize], 8);
            let half = |lo: usize| -> u64 { (lo..lo + 8).map(|i| (a[i] as i64 - b[i] as i64).unsigned_abs()).sum() };
            cpu.set_xmm(d, join(&[half(0), half(8)], 64));
        }
        Op::Pavg { bits } => {
            let b = lanes(cpu.sse_src(m, 16, true), bits);
            let a = lanes(cpu.xmm[d as usize], bits);
            let r: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| (x + y + 1) >> 1).collect();
            cpu.set_xmm(d, join(&r, bits));
        }
        Op::Pmax { bits, signed } | Op::Pmin { bits, signed } => {
            let b = lanes(cpu.sse_src(m, 16, true), bits);
            let a = lanes(cpu.xmm[d as usize], bits);
            let max = matches!(s.op, Op::Pmax { .. });
            let r: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| {
                let take_x = if signed { (sx(x, bits) > sx(y, bits)) == max } else { (x > y) == max };
                if x == y || take_x { x } else { y }
            }).collect();
            cpu.set_xmm(d, join(&r, bits));
        }
        Op::Pcmpeq { bits } | Op::Pcmpgt { bits } => {
            let b = lanes(cpu.sse_src(m, 16, true), bits);
            let a = lanes(cpu.xmm[d as usize], bits);
            let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
            let r: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| {
                let t = if matches!(s.op, Op::Pcmpeq { .. }) { x == y } else { sx(x, bits) > sx(y, bits) };
                if t { mask } else { 0 }
            }).collect();
            cpu.set_xmm(d, join(&r, bits));
        }
        Op::Packss { .. } | Op::Packus => {
            // Narrow every lane of dst then src to half its width, saturating.
            let (bits, unsigned) = match s.op { Op::Packss { bits } => (bits, false), _ => (16, true) };
            let b = lanes(cpu.sse_src(m, 16, true), bits);
            let a = lanes(cpu.xmm[d as usize], bits);
            let half = bits / 2;
            let r: Vec<u64> = a.iter().chain(&b).map(|&x| {
                let v = sx(x, bits);
                if unsigned { sat_unsigned(v, half) } else { sat_signed(v, half) }
            }).collect();
            cpu.set_xmm(d, join(&r, half));
        }
        Op::Punpckl { bits } | Op::Punpckh { bits } => {
            let b = lanes(cpu.sse_src(m, 16, true), bits);
            let a = lanes(cpu.xmm[d as usize], bits);
            let n = a.len() / 2;
            let off = if matches!(s.op, Op::Punpckh { .. }) { n } else { 0 };
            let mut r = Vec::with_capacity(2 * n);
            for i in 0..n { r.push(a[off + i]); r.push(b[off + i]); }
            cpu.set_xmm(d, join(&r, bits));
        }
        Op::Pshufd => {
            let b = lanes(cpu.sse_src(m, 16, true), 32);
            let imm = s.imm as usize;
            let r = [b[imm & 3], b[(imm >> 2) & 3], b[(imm >> 4) & 3], b[(imm >> 6) & 3]];
            cpu.set_xmm(d, join(&r, 32));
        }
        Op::Pshufhw | Op::Pshuflw => {
            let b = lanes(cpu.sse_src(m, 16, true), 16);
            let imm = s.imm as usize;
            let base = if s.op == Op::Pshufhw { 4 } else { 0 };
            let mut r = b.clone();
            for i in 0..4 { r[base + i] = b[base + ((imm >> (2 * i)) & 3)]; }
            cpu.set_xmm(d, join(&r, 16));
        }
        Op::Pinsrw => {
            let v = if m.is_reg() { cpu.reg_w(m.rm, 32) as u16 } else { cpu.read_rm16(m) };
            if cpu.pending_exception.is_some() { return; }
            let mut l = lanes(cpu.xmm[d as usize], 16);
            l[(s.imm & 7) as usize] = v as u64;
            cpu.set_xmm(d, join(&l, 16));
        }
        Op::Pextrw => {
            let l = lanes(cpu.xmm[m.rm as usize], 16);
            cpu.set_reg_w(d, 32, l[(s.imm & 7) as usize]);
        }
        Op::Pshift { kind, bits, imm } => {
            // The count is imm8, or the low 64 bits of the XMM/m128 operand;
            // a count past the lane width shifts everything out (or fills
            // with the sign for SRA).
            let count = if imm { s.imm as u64 } else { cpu.sse_src(m, 16, true) as u64 };
            if cpu.pending_exception.is_some() { return; }
            let a = lanes(cpu.xmm[d as usize], bits);
            let r: Vec<u64> = a.iter().map(|&x| match kind {
                ShiftKind::Sll => if count >= bits as u64 { 0 } else { x << count },
                ShiftKind::Srl => if count >= bits as u64 { 0 } else { x >> count },
                ShiftKind::Sra => {
                    let c = count.min(bits as u64 - 1);
                    (sx(x, bits) >> c) as u64
                }
            }).collect();
            cpu.set_xmm(d, join(&r, bits));
        }
        Op::Pshiftdq { left } => {
            let n = (s.imm as u32).min(16) * 8;
            let a = cpu.xmm[d as usize];
            let r = if n >= 128 { 0 } else if left { a << n } else { a >> n };
            cpu.set_xmm(d, r);
        }

        // ---- State ----
        Op::Ldmxcsr => {
            if m.is_reg() { cpu.raise_ud(); return; }
            let v = cpu.read_rm32(m);
            if cpu.pending_exception.is_some() { return; }
            if v & !MXCSR_MASK != 0 { cpu.raise_gp(0); return; }
            cpu.mxcsr = v;
        }
        Op::Stmxcsr => {
            if m.is_reg() { cpu.raise_ud(); return; }
            let v = cpu.mxcsr;
            cpu.write_rm_w(m, 32, v as u64);
        }
        Op::Fxsave => {
            if m.is_reg() { cpu.raise_ud(); return; }
            let (seg, off) = cpu.sse_operand(m);
            if cpu.linear_addr(seg, off) & 15 != 0 { cpu.raise_gp(0); return; }
            let buf = fxsave_image(cpu);
            cpu.write_linear_bytes(seg, off, &buf);
        }
        Op::Fxrstor => {
            if m.is_reg() { cpu.raise_ud(); return; }
            let (seg, off) = cpu.sse_operand(m);
            if cpu.linear_addr(seg, off) & 15 != 0 { cpu.raise_gp(0); return; }
            let mut buf = [0u8; 512];
            cpu.read_linear_bytes(seg, off, &mut buf);
            if cpu.pending_exception.is_some() { return; }
            fxrstor_image(cpu, &buf);
        }
    }
}

/// Build the 512-byte FXSAVE image: x87 control/status/abridged tag/opcode,
/// FIP/FDP (not tracked, stored as zero), MXCSR and its mask, ST0-7 as
/// 80-bit values in 16-byte slots, then XMM0-15 (only XMM0-7 outside
/// 64-bit mode; the rest of the area is left as it was, as the hardware
/// leaves it).
pub fn fxsave_image(cpu: &Cpu) -> Vec<u8> {
    let mut b = vec![0u8; 512];
    b[0..2].copy_from_slice(&cpu.fpu.control.to_le_bytes());
    // The status word carries TOP in bits 11-13.
    let sw = (cpu.fpu.status & !0x3800) | ((cpu.fpu.top as u16 & 7) << 11);
    b[2..4].copy_from_slice(&sw.to_le_bytes());
    // Abridged tag: one bit per physical register, 1 = not empty.
    let mut ftw = 0u8;
    for i in 0..8 { if (cpu.fpu.tag >> i) & 1 == 0 { ftw |= 1 << i; } }
    b[4] = ftw;
    b[24..28].copy_from_slice(&cpu.mxcsr.to_le_bytes());
    b[28..32].copy_from_slice(&MXCSR_MASK.to_le_bytes());
    for i in 0..8 {
        // Slot i holds ST(i), i.e. relative to TOP.
        let (sig, se) = f64_to_f80(cpu.fpu.st_i(i));
        let o = 32 + i * 16;
        b[o..o + 8].copy_from_slice(&sig.to_le_bytes());
        b[o + 8..o + 10].copy_from_slice(&se.to_le_bytes());
    }
    let n = if cpu.long_mode() { 16 } else { 8 };
    for i in 0..n {
        let o = 160 + i * 16;
        b[o..o + 16].copy_from_slice(&cpu.xmm[i].to_le_bytes());
    }
    b
}

/// Load FPU/SSE state from a 512-byte FXSAVE image.
pub fn fxrstor_image(cpu: &mut Cpu, b: &[u8]) {
    let u16at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    cpu.fpu.control = u16at(0);
    let sw = u16at(2);
    cpu.fpu.status = sw;
    cpu.fpu.top = ((sw >> 11) & 7) as usize;
    // The abridged tag back to the Fpu's word: one "empty" bit per register
    // in the low byte (the high byte is unused there and stays set).
    let ftw = b[4];
    let mut tag = 0xFF00u16;
    for i in 0..8 { if (ftw >> i) & 1 == 0 { tag |= 1 << i; } }
    cpu.fpu.tag = tag;
    let mx = u32at(24);
    if mx & !MXCSR_MASK != 0 { cpu.raise_gp(0); return; }
    cpu.mxcsr = mx;
    for i in 0..8 {
        let o = 32 + i * 16;
        let sig = u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let se = u16at(o + 8);
        let idx = (cpu.fpu.top + i) % 8;
        cpu.fpu.st[idx] = f80_to_f64(sig, se);
    }
    let n = if cpu.long_mode() { 16 } else { 8 };
    for i in 0..n {
        let o = 160 + i * 16;
        cpu.xmm[i] = u128::from_le_bytes(b[o..o + 16].try_into().unwrap());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{Cpu, flags, CR4_OSFXSR};

    /// A CPU in long mode with SSE enabled, `code` at the entry point, and
    /// a scratch page at DATA for memory operands.
    const DATA: u64 = 0x20_0000;
    fn cpu64(code: &[u8]) -> Cpu {
        let mut cpu = crate::instructions::tests::long_cpu(code);
        cpu.cr4 |= CR4_OSFXSR;
        cpu
    }
    fn run(cpu: &mut Cpu) { crate::instructions::tests::run64(cpu); }
    fn xmm(lo: u64, hi: u64) -> u128 { lo as u128 | ((hi as u128) << 64) }
    fn ps(a: f32, b: f32, c: f32, d: f32) -> u128 { from_f32s([a, b, c, d]) }
    fn pd(a: f64, b: f64) -> u128 { from_f64s([a, b]) }
    fn wr128(cpu: &mut Cpu, addr: u64, v: u128) { cpu.mem.write_u128(addr as usize, v); }
    fn rd128(cpu: &Cpu, addr: u64) -> u128 { cpu.mem.read_u128(addr as usize) }
    /// `mov rax, imm64`.
    fn mov_rax(v: u64) -> Vec<u8> { let mut c = vec![0x48, 0xB8]; c.extend_from_slice(&v.to_le_bytes()); c }

    #[test]
    fn movups_store_and_load_are_different_directions() {
        // movups (%rax),%xmm0 ; movups %xmm0,0x10(%rax) ; hlt
        let mut code = mov_rax(DATA);
        code.extend_from_slice(&[0x0F, 0x10, 0x00, 0x0F, 0x11, 0x40, 0x10, 0xF4]);
        let mut cpu = cpu64(&code);
        wr128(&mut cpu, DATA, xmm(0x1122_3344_5566_7788, 0x99AA_BBCC_DDEE_FF00));
        run(&mut cpu);
        assert_eq!(cpu.xmm[0], xmm(0x1122_3344_5566_7788, 0x99AA_BBCC_DDEE_FF00));
        assert_eq!(rd128(&cpu, DATA + 0x10), xmm(0x1122_3344_5566_7788, 0x99AA_BBCC_DDEE_FF00));
    }

    #[test]
    fn musl_bin_sentinel_sequence() {
        // The malloc idiom that found the store bug:
        //   movq %rax,%xmm0 ; punpcklqdq %xmm0,%xmm0 ; movups %xmm0,(%rax)
        let mut code = mov_rax(DATA);
        code.extend_from_slice(&[0x66, 0x48, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0x6C, 0xC0, 0x0F, 0x11, 0x00, 0xF4]);
        let mut cpu = cpu64(&code);
        run(&mut cpu);
        assert_eq!(rd128(&cpu, DATA), xmm(DATA, DATA));
    }

    #[test]
    fn movaps_requires_alignment() {
        // movaps 8(%rax),%xmm0 -> #GP (rax = DATA, 16-aligned)
        let mut code = mov_rax(DATA);
        code.extend_from_slice(&[0x0F, 0x28, 0x40, 0x08, 0xF4]);
        let mut cpu = cpu64(&code);
        cpu.step(); // mov
        cpu.step(); // movaps
        assert_eq!(cpu.pending_exception, Some((0x0D, Some(0))));
    }

    #[test]
    fn sse_faults_ud_without_osfxsr() {
        let code = [0x0F, 0x57, 0xC0, 0xF4]; // xorps %xmm0,%xmm0
        let mut cpu = cpu64(&code);
        cpu.cr4 &= !CR4_OSFXSR;
        cpu.step();
        assert_eq!(cpu.pending_exception, Some((0x06, None)));
    }

    #[test]
    fn movss_and_movsd_semantics() {
        // movss (%rax),%xmm1 : zero-extends. movss %xmm1,%xmm2 : merges low.
        // movsd %xmm1,%xmm3 : merges low 64.
        let mut code = mov_rax(DATA);
        code.extend_from_slice(&[
            0xF3, 0x0F, 0x10, 0x08,       // movss (%rax),%xmm1
            0xF3, 0x0F, 0x10, 0xD1,       // movss %xmm1,%xmm2
            0xF2, 0x0F, 0x10, 0xD9,       // movsd %xmm1,%xmm3
            0xF2, 0x0F, 0x11, 0x58, 0x20, // movsd %xmm3,0x20(%rax)
            0xF4]);
        let mut cpu = cpu64(&code);
        wr128(&mut cpu, DATA, xmm(0xAAAA_AAAA_1234_5678, 0xBBBB_BBBB_BBBB_BBBB));
        cpu.xmm[2] = xmm(0x1111_1111_1111_1111, 0x2222_2222_2222_2222);
        cpu.xmm[3] = xmm(0x3333_3333_3333_3333, 0x4444_4444_4444_4444);
        run(&mut cpu);
        assert_eq!(cpu.xmm[1], 0x1234_5678);
        assert_eq!(cpu.xmm[2], xmm(0x1111_1111_1234_5678, 0x2222_2222_2222_2222));
        assert_eq!(cpu.xmm[3], xmm(0x0000_0000_1234_5678, 0x4444_4444_4444_4444));
        assert_eq!(cpu.mem.read_u64((DATA + 0x20) as usize), 0x1234_5678);
        // The store wrote exactly 8 bytes.
        assert_eq!(cpu.mem.read_u64((DATA + 0x28) as usize), 0);
    }

    #[test]
    fn scalar_arith_touches_only_the_low_element() {
        // addss %xmm1,%xmm0 ; addsd %xmm3,%xmm2 ; hlt
        let code = [0xF3, 0x0F, 0x58, 0xC1, 0xF2, 0x0F, 0x58, 0xD3, 0xF4];
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = ps(1.5, 2.0, 3.0, 4.0);
        cpu.xmm[1] = ps(0.25, 100.0, 100.0, 100.0);
        cpu.xmm[2] = pd(1.5, 9.0);
        cpu.xmm[3] = pd(0.25, 100.0);
        run(&mut cpu);
        assert_eq!(cpu.xmm[0], ps(1.75, 2.0, 3.0, 4.0));
        assert_eq!(cpu.xmm[2], pd(1.75, 9.0));
    }

    #[test]
    fn packed_arith_and_minmax() {
        // mulps %xmm1,%xmm0 ; minpd %xmm3,%xmm2 ; maxps %xmm5,%xmm4 ; hlt
        let code = [0x0F, 0x59, 0xC1, 0x66, 0x0F, 0x5D, 0xD3, 0x0F, 0x5F, 0xE5, 0xF4];
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = ps(1.0, 2.0, 3.0, 4.0);
        cpu.xmm[1] = ps(2.0, 2.0, 2.0, 0.5);
        cpu.xmm[2] = pd(1.0, 5.0);
        cpu.xmm[3] = pd(2.0, -1.0);
        cpu.xmm[4] = ps(1.0, f32::NAN, 0.0, -0.0);
        cpu.xmm[5] = ps(2.0, 1.0, -0.0, 0.0);
        run(&mut cpu);
        assert_eq!(cpu.xmm[0], ps(2.0, 4.0, 6.0, 2.0));
        assert_eq!(cpu.xmm[2], pd(1.0, -1.0));
        // NaN or equal: MAX returns the source operand.
        assert_eq!(f32s(cpu.xmm[4]), [2.0, 1.0, -0.0, 0.0]);
    }

    #[test]
    fn cmpps_predicates_handle_nan() {
        // cmpunordps %xmm1,%xmm0 (imm 3); cmpneqss %xmm3,%xmm2 (imm 4)
        let code = [0x0F, 0xC2, 0xC1, 0x03, 0xF3, 0x0F, 0xC2, 0xD3, 0x04, 0xF4];
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = ps(1.0, f32::NAN, 3.0, 4.0);
        cpu.xmm[1] = ps(1.0, 2.0, f32::NAN, 4.0);
        cpu.xmm[2] = ps(f32::NAN, 7.0, 7.0, 7.0);
        cpu.xmm[3] = ps(1.0, 0.0, 0.0, 0.0);
        run(&mut cpu);
        assert_eq!(cpu.xmm[0], xmm(0xFFFF_FFFF_0000_0000, 0x0000_0000_FFFF_FFFF));
        assert_eq!(cpu.xmm[2], (ps(0.0, 7.0, 7.0, 7.0) & !LO32) | LO32);
    }

    #[test]
    fn ucomiss_sets_the_flags_the_manual_says() {
        // ucomiss %xmm1,%xmm0 three times with different values isn't one
        // program; drive execute directly.
        for (a, b, zf, pf, cf) in [(1.0f32, 2.0f32, false, false, true), (2.0, 1.0, false, false, false),
                                   (1.0, 1.0, true, false, false), (f32::NAN, 1.0, true, true, true)] {
            let code = [0x0F, 0x2E, 0xC1, 0xF4];
            let mut cpu = cpu64(&code);
            cpu.xmm[0] = a.to_bits() as u128;
            cpu.xmm[1] = b.to_bits() as u128;
            cpu.flags |= flags::OF | flags::SF;
            run(&mut cpu);
            assert_eq!((cpu.get_flag(flags::ZF), cpu.get_flag(flags::PF), cpu.get_flag(flags::CF)), (zf, pf, cf));
            assert!(!cpu.get_flag(flags::OF) && !cpu.get_flag(flags::SF));
        }
    }

    #[test]
    fn conversions_round_per_mxcsr() {
        // cvtsi2sd %rax,%xmm0 ; cvttsd2si %xmm0,%rbx ; cvtsd2si %xmm0,%rcx ;
        // cvtsi2ss %eax,%xmm1 ; cvtss2sd %xmm1,%xmm2 ; hlt
        let mut code = mov_rax(-7i64 as u64);
        code.extend_from_slice(&[
            0xF2, 0x48, 0x0F, 0x2A, 0xC0,
            0xF2, 0x48, 0x0F, 0x2C, 0xD8,
            0xF2, 0x48, 0x0F, 0x2D, 0xC8,
            0xF3, 0x0F, 0x2A, 0xC8,
            0xF3, 0x0F, 0x5A, 0xD1,
            0xF4]);
        let mut cpu = cpu64(&code);
        run(&mut cpu);
        assert_eq!(f64::from_bits(cpu.xmm[0] as u64), -7.0);
        assert_eq!(cpu.regs[3] as i64, -7);
        assert_eq!(cpu.regs[1] as i64, -7);
        assert_eq!(f32::from_bits(cpu.xmm[1] as u32), -7.0);
        assert_eq!(f64::from_bits(cpu.xmm[2] as u64), -7.0);

        // 2.5 -> nearest-even 2, and 3 under round-up (RC=2), 2 truncated.
        for (rc, trunc, want) in [(0u32, false, 2i64), (2, false, 3), (1, false, 2), (0, true, 2), (3, false, 2)] {
            let mut cpu = cpu64(&[0xF2, 0x48, 0x0F, if trunc { 0x2C } else { 0x2D }, 0xC0, 0xF4]);
            cpu.mxcsr = MXCSR_DEFAULT & !(3 << 13) | (rc << 13);
            cpu.xmm[0] = 2.5f64.to_bits() as u128;
            run(&mut cpu);
            assert_eq!(cpu.regs[0] as i64, want, "rc={rc} trunc={trunc}");
        }
        // NaN and overflow give the integer indefinite.
        let mut cpu = cpu64(&[0xF2, 0x0F, 0x2C, 0xC0, 0xF4]);
        cpu.xmm[0] = 1e30f64.to_bits() as u128;
        run(&mut cpu);
        assert_eq!(cpu.regs[0], 0x8000_0000);
    }

    #[test]
    fn packed_conversions() {
        // cvtdq2ps %xmm1,%xmm0 ; cvttps2dq %xmm0,%xmm2 ; cvtps2pd %xmm1,%xmm3 ; cvtpd2ps %xmm3,%xmm4
        // cvtdq2pd %xmm1,%xmm5 ; cvtpd2dq %xmm5,%xmm6
        let code = [0x0F, 0x5B, 0xC1, 0xF3, 0x0F, 0x5B, 0xD0, 0x0F, 0x5A, 0xD9, 0x66, 0x0F, 0x5A, 0xE3,
                    0xF3, 0x0F, 0xE6, 0xE9, 0x66, 0x0F, 0xE6, 0xF5, 0xF4];
        let mut cpu = cpu64(&code);
        cpu.xmm[1] = join(&[1, (-2i32) as u32 as u64, 3, 0x7FFF_FFFF], 32);
        run(&mut cpu);
        assert_eq!(f32s(cpu.xmm[0]), [1.0, -2.0, 3.0, 2147483648.0]);
        assert_eq!(lanes(cpu.xmm[2], 32), vec![1, 0xFFFF_FFFE, 3, 0x8000_0000]);
        // cvtps2pd converts the two low singles of the *bit pattern* in xmm1.
        assert_eq!(f64s(cpu.xmm[3])[0], f32::from_bits(1) as f64);
        assert!(f64s(cpu.xmm[3])[1].is_nan());
        assert_eq!(f64s(cpu.xmm[5]), [1.0, -2.0]);
        assert_eq!(lanes(cpu.xmm[6], 32), vec![1, 0xFFFF_FFFE, 0, 0]);
    }

    #[test]
    fn packed_integer_add_sub_saturate() {
        // paddb, paddusb, paddsw, psubusw, psubq
        let code = [0x66, 0x0F, 0xFC, 0xC1,   // paddb  %xmm1,%xmm0
                    0x66, 0x0F, 0xDC, 0xD3,   // paddusb %xmm3,%xmm2
                    0x66, 0x0F, 0xED, 0xE5,   // paddsw %xmm5,%xmm4
                    0x66, 0x0F, 0xD9, 0xF7,   // psubusw %xmm7,%xmm6
                    0x66, 0x45, 0x0F, 0xFB, 0xC1, // psubq %xmm9,%xmm8
                    0xF4];
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = join(&[0xFF, 0x7F, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8);
        cpu.xmm[1] = join(&[0x01, 0x01, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8);
        cpu.xmm[2] = join(&[0xFF, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8);
        cpu.xmm[3] = join(&[0x01, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8);
        cpu.xmm[4] = join(&[0x7FFF, 0x8000, 5, 0, 0, 0, 0, 0], 16);
        cpu.xmm[5] = join(&[0x0001, 0xFFFF, 5, 0, 0, 0, 0, 0], 16);
        cpu.xmm[6] = join(&[3, 0xFFFF, 0, 0, 0, 0, 0, 0], 16);
        cpu.xmm[7] = join(&[5, 1, 0, 0, 0, 0, 0, 0], 16);
        cpu.xmm[8] = xmm(0, 10);
        cpu.xmm[9] = xmm(1, 3);
        run(&mut cpu);
        assert_eq!(lanes(cpu.xmm[0], 8)[..3], [0x00, 0x80, 0x00]);
        assert_eq!(lanes(cpu.xmm[2], 8)[..2], [0xFF, 0xFF]);
        assert_eq!(lanes(cpu.xmm[4], 16)[..3], [0x7FFF, 0x8000, 10]);
        assert_eq!(lanes(cpu.xmm[6], 16)[..2], [0, 0xFFFE]);
        assert_eq!(cpu.xmm[8], xmm(u64::MAX, 7));
    }

    #[test]
    fn packed_multiply_and_horizontal_ops() {
        // pmullw, pmulhw, pmulhuw, pmuludq, pmaddwd, psadbw, pavgb
        let code = [0x66, 0x0F, 0xD5, 0xC1, 0x66, 0x0F, 0xE5, 0xD1, 0x66, 0x0F, 0xE4, 0xD9,
                    0x66, 0x0F, 0xF4, 0xE1, 0x66, 0x0F, 0xF5, 0xE9, 0x66, 0x0F, 0xF6, 0xF1,
                    0x66, 0x0F, 0xE0, 0xF9, 0xF4];
        let mut cpu = cpu64(&code);
        let w = join(&[0xFFFF, 2, 0x8000, 3, 0, 0, 0, 0], 16); // -1, 2, -32768, 3
        cpu.xmm[1] = w;
        cpu.xmm[0] = join(&[3, 3, 2, 3, 0, 0, 0, 0], 16);
        cpu.xmm[2] = join(&[3, 0x4000, 2, 3, 0, 0, 0, 0], 16);
        cpu.xmm[3] = join(&[3, 0x4000, 2, 3, 0, 0, 0, 0], 16);
        cpu.xmm[4] = join(&[0xFFFF_FFFF, 7, 5, 9], 32);
        cpu.xmm[5] = join(&[1, 2, 3, 4, 0, 0, 0, 0], 16);
        cpu.xmm[6] = join(&[10, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 8);
        cpu.xmm[7] = join(&[1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8);
        run(&mut cpu);
        assert_eq!(lanes(cpu.xmm[0], 16)[..4], [0xFFFD, 6, 0, 9]);           // low words of -3, 6, -65536, 9
        assert_eq!(lanes(cpu.xmm[2], 16)[..4], [0xFFFF, 0, 0xFFFF, 0]);      // signed high words
        assert_eq!(lanes(cpu.xmm[3], 16)[..4], [2, 0, 1, 0]);                // unsigned high words
        // pmuludq multiplies dwords 0 and 2: xmm1's are 0x0002FFFF and 0.
        assert_eq!(lanes(cpu.xmm[4], 64), vec![0xFFFF_FFFF * 0x0002_FFFF, 0]);
        assert_eq!(lanes(cpu.xmm[5], 32)[..2], [(-1 * 1 + 2 * 2) as u32 as u64, (-32768 * 3 + 3 * 4) as i32 as u32 as u64]);
        // psadbw against xmm1's bytes [FF FF 02 00 00 80 03 00 | 0 ...].
        assert_eq!(lanes(cpu.xmm[6], 64), vec![245 + 255 + 2 + 128 + 3, 1]);
        assert_eq!(lanes(cpu.xmm[7], 8)[..2], [128, 129]);
    }

    #[test]
    fn packed_compare_pack_unpack_shuffle() {
        let code = [0x66, 0x0F, 0x74, 0xC1,        // pcmpeqb %xmm1,%xmm0
                    0x66, 0x0F, 0x66, 0xD3,        // pcmpgtd %xmm3,%xmm2
                    0x66, 0x0F, 0x63, 0xE5,        // packsswb %xmm5,%xmm4
                    0x66, 0x0F, 0x67, 0xF7,        // packuswb %xmm7,%xmm6
                    0x66, 0x45, 0x0F, 0x6B, 0xC1,  // packssdw %xmm9,%xmm8
                    0x66, 0x45, 0x0F, 0x68, 0xD3,  // punpckhbw %xmm11,%xmm10
                    0x66, 0x45, 0x0F, 0x70, 0xE5, 0x1B, // pshufd $0x1B,%xmm13,%xmm12
                    0x66, 0x45, 0x0F, 0x62, 0xF7,  // punpckldq %xmm15,%xmm14
                    0xF4];
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = join(&[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8);
        cpu.xmm[1] = join(&[1, 9, 3, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8);
        cpu.xmm[2] = join(&[5, 0xFFFF_FFFF, 0, 0], 32);
        cpu.xmm[3] = join(&[4, 0, 0, 0], 32);
        cpu.xmm[4] = join(&[0x0100, 0xFF00, 0x7F, 0x80, 0, 0, 0, 0], 16);
        cpu.xmm[5] = join(&[1, 2, 3, 4, 5, 6, 7, 8], 16);
        cpu.xmm[6] = join(&[0x0100, 0xFF00, 0x7F, 0xFF, 0, 0, 0, 0], 16);
        cpu.xmm[7] = 0;
        cpu.xmm[8] = join(&[0x0001_0000, 0xFFFF_0000, 7, 0xFFFF_FFF0], 32);
        cpu.xmm[9] = 0;
        cpu.xmm[10] = join(&[0, 0, 0, 0, 0, 0, 0, 0, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7], 8);
        cpu.xmm[11] = join(&[0, 0, 0, 0, 0, 0, 0, 0, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7], 8);
        cpu.xmm[13] = join(&[0x10, 0x11, 0x12, 0x13], 32);
        cpu.xmm[14] = join(&[1, 2, 3, 4], 32);
        cpu.xmm[15] = join(&[5, 6, 7, 8], 32);
        run(&mut cpu);
        assert_eq!(lanes(cpu.xmm[0], 8)[..4], [0xFF, 0, 0xFF, 0]);
        assert_eq!(lanes(cpu.xmm[2], 32), vec![0xFFFF_FFFF, 0, 0, 0]);
        assert_eq!(lanes(cpu.xmm[4], 8)[..8], [0x7F, 0x80, 0x7F, 0x7F, 0, 0, 0, 0]);
        assert_eq!(lanes(cpu.xmm[4], 8)[8..], [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(lanes(cpu.xmm[6], 8)[..4], [0xFF, 0, 0x7F, 0xFF]);
        assert_eq!(lanes(cpu.xmm[8], 16)[..4], [0x7FFF, 0x8000, 7, 0xFFF0]);
        assert_eq!(lanes(cpu.xmm[10], 8), vec![0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2, 0xA3, 0xB3, 0xA4, 0xB4, 0xA5, 0xB5, 0xA6, 0xB6, 0xA7, 0xB7]);
        assert_eq!(lanes(cpu.xmm[12], 32), vec![0x13, 0x12, 0x11, 0x10]);
        assert_eq!(lanes(cpu.xmm[14], 32), vec![1, 5, 2, 6]);
    }

    #[test]
    fn packed_shifts_immediate_and_register() {
        let code = [0x66, 0x0F, 0x71, 0xD0, 0x04,   // psrlw $4,%xmm0
                    0x66, 0x0F, 0x72, 0xE1, 0x1F,   // psrad $31,%xmm1
                    0x66, 0x0F, 0x73, 0xF2, 0x21,   // psllq $33,%xmm2
                    0x66, 0x0F, 0x73, 0xFB, 0x03,   // pslldq $3,%xmm3
                    0x66, 0x0F, 0x73, 0xDC, 0x0F,   // psrldq $15,%xmm4
                    0x66, 0x0F, 0xF2, 0xEE,         // pslld %xmm6,%xmm5
                    0x66, 0x41, 0x0F, 0x71, 0xD0, 0x10, // psrlw $16,%xmm8 -> zero
                    0xF4];
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = join(&[0xF0F0, 0x1234, 0, 0, 0, 0, 0, 0], 16);
        cpu.xmm[1] = join(&[0x8000_0000, 1, 0, 0], 32);
        cpu.xmm[2] = xmm(1, 1);
        cpu.xmm[3] = xmm(0x0102_0304_0506_0708, 0x090A_0B0C_0D0E_0F10);
        cpu.xmm[4] = xmm(0x0102_0304_0506_0708, 0x090A_0B0C_0D0E_0F10);
        cpu.xmm[5] = join(&[1, 2, 3, 4], 32);
        cpu.xmm[6] = xmm(4, 0);
        cpu.xmm[8] = xmm(u64::MAX, u64::MAX);
        run(&mut cpu);
        assert_eq!(lanes(cpu.xmm[0], 16)[..2], [0x0F0F, 0x0123]);
        assert_eq!(lanes(cpu.xmm[1], 32)[..2], [0xFFFF_FFFF, 0]);
        assert_eq!(cpu.xmm[2], xmm(1 << 33, 1 << 33));
        assert_eq!(cpu.xmm[3], xmm(0x0405_0607_0800_0000, 0x0C0D_0E0F_1001_0203));
        assert_eq!(cpu.xmm[4], 0x09);
        assert_eq!(lanes(cpu.xmm[5], 32), vec![16, 32, 48, 64]);
        assert_eq!(cpu.xmm[8], 0);
    }

    #[test]
    fn movd_movq_pextrw_pinsrw_movmsk() {
        let mut code = mov_rax(0x8000_0000_1234_5678);
        code.extend_from_slice(&[
            0x66, 0x0F, 0x6E, 0xC0,             // movd %eax,%xmm0
            0x66, 0x48, 0x0F, 0x6E, 0xC8,       // movq %rax,%xmm1
            0x66, 0x0F, 0x7E, 0xCB,             // movd %xmm1,%ebx
            0x66, 0x48, 0x0F, 0x7E, 0xC9,       // movq %xmm1,%rcx
            0xF3, 0x0F, 0x7E, 0xD1,             // movq %xmm1,%xmm2 (zero upper)
            0x66, 0x0F, 0xD6, 0xCB,             // movq %xmm1,%xmm3
            0x66, 0x0F, 0xC5, 0xD1, 0x01,       // pextrw $1,%xmm1,%edx
            0x66, 0x0F, 0xC4, 0xE0, 0x07,       // pinsrw $7,%eax,%xmm4
            0x66, 0x0F, 0xD7, 0xF1,             // pmovmskb %xmm1,%esi
            0x0F, 0x50, 0xF9,                   // movmskps %xmm1,%edi
            0xF4]);
        let mut cpu = cpu64(&code);
        cpu.xmm[2] = xmm(0, u64::MAX);
        cpu.xmm[3] = xmm(0, u64::MAX);
        cpu.regs[3] = u64::MAX;
        run(&mut cpu);
        assert_eq!(cpu.xmm[0], 0x1234_5678);
        assert_eq!(cpu.xmm[1], 0x8000_0000_1234_5678);
        assert_eq!(cpu.regs[3], 0x1234_5678);        // movd zero-extends into rbx
        assert_eq!(cpu.regs[1], 0x8000_0000_1234_5678);
        assert_eq!(cpu.xmm[2], 0x8000_0000_1234_5678);
        assert_eq!(cpu.xmm[3], 0x8000_0000_1234_5678);
        assert_eq!(cpu.regs[2], 0x1234);
        assert_eq!(cpu.xmm[4] >> 112, 0x5678);
        assert_eq!(cpu.regs[6], 0x0080);             // byte 7 has its sign bit set
        assert_eq!(cpu.regs[7], 0b0010);
    }

    #[test]
    fn shufps_unpck_movhlps_movlhps_movhps() {
        let mut code = mov_rax(DATA);
        code.extend_from_slice(&[
            0x0F, 0xC6, 0xC1, 0x4E,       // shufps $0x4E,%xmm1,%xmm0  (2,3 from dst; 0,1 from src)
            0x0F, 0x14, 0xD3,             // unpcklps %xmm3,%xmm2
            0x66, 0x0F, 0x15, 0xE5,       // unpckhpd %xmm5,%xmm4
            0x0F, 0x12, 0xF7,             // movhlps %xmm7,%xmm6
            0x0F, 0x16, 0xFE,             // movlhps %xmm6,%xmm7
            0x0F, 0x16, 0x00,             // movhps (%rax),%xmm0
            0x0F, 0x13, 0x48, 0x10,       // movlps %xmm1,0x10(%rax)
            0xF4]);
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = ps(0.0, 1.0, 2.0, 3.0);
        cpu.xmm[1] = ps(10.0, 11.0, 12.0, 13.0);
        cpu.xmm[2] = ps(0.0, 1.0, 2.0, 3.0);
        cpu.xmm[3] = ps(10.0, 11.0, 12.0, 13.0);
        cpu.xmm[4] = pd(0.0, 1.0);
        cpu.xmm[5] = pd(10.0, 11.0);
        cpu.xmm[6] = xmm(0x6666, 0x6666_0000);
        cpu.xmm[7] = xmm(0x7777, 0x7777_0000);
        wr128(&mut cpu, DATA, xmm(0xDEAD_BEEF, 0));
        run(&mut cpu);
        assert_eq!(f32s(cpu.xmm[0])[..2], [2.0, 3.0]);
        assert_eq!(cpu.xmm[0] >> 64, 0xDEAD_BEEF);   // movhps replaced the high half
        assert_eq!(f32s(cpu.xmm[2]), [0.0, 10.0, 1.0, 11.0]);
        assert_eq!(f64s(cpu.xmm[4]), [1.0, 11.0]);
        assert_eq!(cpu.xmm[6], xmm(0x7777_0000, 0x6666_0000));
        assert_eq!(cpu.xmm[7], xmm(0x7777, 0x7777_0000));
        assert_eq!(cpu.mem.read_u64((DATA + 0x10) as usize), 10.0f32.to_bits() as u64 | ((11.0f32.to_bits() as u64) << 32));
    }

    #[test]
    fn logic_sqrt_and_sse3_forms() {
        let code = [0x0F, 0x54, 0xC1,          // andps
                    0x66, 0x0F, 0x55, 0xD3,    // andnpd
                    0x66, 0x0F, 0xEB, 0xE5,    // por
                    0x0F, 0x57, 0xF6,          // xorps %xmm6,%xmm6
                    0x66, 0x0F, 0x51, 0xFF,    // sqrtpd %xmm7,%xmm7
                    0xF2, 0x0F, 0x12, 0xC7,    // movddup %xmm7,%xmm0
                    0xF3, 0x0F, 0x16, 0xCF,    // movshdup %xmm7,%xmm1
                    0x66, 0x0F, 0x7C, 0xD7,    // haddpd %xmm7,%xmm2
                    0xF2, 0x0F, 0xD0, 0xDF,    // addsubps %xmm7,%xmm3
                    0xF4];
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = xmm(0xFF00, 0xF0F0);
        cpu.xmm[1] = xmm(0x0FF0, 0xFFFF);
        cpu.xmm[2] = xmm(0xFF00, 0);
        cpu.xmm[3] = xmm(0xFFFF, 0xF);
        cpu.xmm[4] = xmm(1, 2);
        cpu.xmm[5] = xmm(4, 8);
        cpu.xmm[6] = xmm(u64::MAX, 3);
        cpu.xmm[7] = pd(16.0, 9.0);
        run(&mut cpu);
        assert_eq!(cpu.xmm[0] >> 64 & 0xFFFF, 0);  // overwritten by movddup below; check via xmm7 first
        assert_eq!(cpu.xmm[7], pd(4.0, 3.0));
        assert_eq!(cpu.xmm[0], pd(4.0, 4.0));
        assert_eq!(f32s(cpu.xmm[1]), { let s = f32s(pd(4.0, 3.0)); [s[1], s[1], s[3], s[3]] });
        assert_eq!(cpu.xmm[2] >> 64, 7.0f64.to_bits() as u128); // haddpd: high = src0+src1
        assert_eq!(cpu.xmm[4], xmm(5, 10));
        assert_eq!(cpu.xmm[6], 0);
    }

    #[test]
    fn andnpd_and_addsub_values() {
        let code = [0x66, 0x0F, 0x55, 0xC1, 0xF2, 0x0F, 0xD0, 0xD3, 0xF4];
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = xmm(0xFF00, 0);
        cpu.xmm[1] = xmm(0x0FF0, 0xFFFF);
        cpu.xmm[2] = ps(1.0, 1.0, 1.0, 1.0);
        cpu.xmm[3] = ps(0.5, 0.5, 0.5, 0.5);
        run(&mut cpu);
        assert_eq!(cpu.xmm[0], xmm(0x00F0, 0xFFFF));
        assert_eq!(f32s(cpu.xmm[2]), [0.5, 1.5, 0.5, 1.5]);
    }

    #[test]
    fn ldmxcsr_stmxcsr_and_reserved_bits() {
        let mut code = mov_rax(DATA);
        code.extend_from_slice(&[0x0F, 0xAE, 0x10, 0x0F, 0xAE, 0x58, 0x08, 0xF4]); // ldmxcsr (%rax); stmxcsr 8(%rax)
        let mut cpu = cpu64(&code);
        cpu.mem.write_u32(DATA as usize, 0x7F80 | (1 << 6)); // DAZ, round-nearest, all masked
        run(&mut cpu);
        assert_eq!(cpu.mxcsr, 0x7FC0);
        assert_eq!(cpu.mem.read_u32((DATA + 8) as usize), 0x7FC0);
        // A reserved bit is #GP.
        let mut cpu = cpu64(&code);
        cpu.mem.write_u32(DATA as usize, 0x1_0000);
        cpu.step(); cpu.step();
        assert_eq!(cpu.pending_exception, Some((0x0D, Some(0))));
    }

    #[test]
    fn fxsave_fxrstor_round_trip() {
        // fxsave (%rax) ; then clobber ; fxrstor (%rax) ; hlt
        let mut code = mov_rax(DATA);
        code.extend_from_slice(&[0x0F, 0xAE, 0x00, 0xF4]);
        let mut cpu = cpu64(&code);
        cpu.fpu.control = 0x027F;
        cpu.fpu.push(3.5);
        cpu.fpu.push(-1.0e-310); // a double denormal survives the 80-bit trip
        cpu.mxcsr = 0x1FA0;
        for i in 0..16 { cpu.xmm[i] = xmm(0x1000 + i as u64, 0x2000 + i as u64); }
        run(&mut cpu);
        let img = (0..512).map(|i| cpu.mem.read_u8(DATA as usize + i)).collect::<Vec<_>>();
        assert_eq!(u16::from_le_bytes([img[0], img[1]]), 0x027F);
        assert_eq!(img[4], 0b1100_0000);                       // ST7 and ST6 physical slots valid (top=6)
        assert_eq!(u32::from_le_bytes([img[24], img[25], img[26], img[27]]), 0x1FA0);
        assert_eq!(u32::from_le_bytes([img[28], img[29], img[30], img[31]]), MXCSR_MASK);
        assert_eq!(u128::from_le_bytes(img[160 + 15 * 16..160 + 16 * 16].try_into().unwrap()), xmm(0x100F, 0x200F));
        // ST(0) slot: 80-bit -1e-310.
        let sig = u64::from_le_bytes(img[32..40].try_into().unwrap());
        let se = u16::from_le_bytes([img[40], img[41]]);
        assert_eq!(f80_to_f64(sig, se), -1.0e-310);

        // Now restore into a fresh CPU.
        let mut code = mov_rax(DATA);
        code.extend_from_slice(&[0x0F, 0xAE, 0x08, 0xF4]);
        let mut cpu2 = cpu64(&code);
        for i in 0..512 { cpu2.mem.write_u8(DATA as usize + i, img[i]); }
        run(&mut cpu2);
        assert_eq!(cpu2.fpu.control, 0x027F);
        assert_eq!(cpu2.fpu.top, 6);
        assert_eq!(cpu2.fpu.st_i(0), -1.0e-310);
        assert_eq!(cpu2.fpu.st_i(1), 3.5);
        assert_eq!(cpu2.mxcsr, 0x1FA0);
        assert_eq!(cpu2.xmm[9], xmm(0x1009, 0x2009));
    }

    #[test]
    fn f80_conversion_edge_cases() {
        for v in [0.0, -0.0, 1.0, -1.5, 1e300, -1e-300, f64::MAX, f64::MIN_POSITIVE, 5e-324, f64::INFINITY, f64::NEG_INFINITY] {
            let (s, e) = f64_to_f80(v);
            assert_eq!(f80_to_f64(s, e).to_bits(), v.to_bits(), "{v}");
        }
        let (s, e) = f64_to_f80(f64::NAN);
        assert!(f80_to_f64(s, e).is_nan());
        assert_eq!(f64_to_f80(1.0), (0x8000_0000_0000_0000, 0x3FFF));
    }

    #[test]
    fn mmx_forms_are_undefined_opcodes() {
        // paddb %mm1,%mm0 (no prefix) -> #UD; the CPU has no MMX unit.
        let mut cpu = cpu64(&[0x0F, 0xFC, 0xC1, 0xF4]);
        cpu.step();
        assert_eq!(cpu.pending_exception, Some((0x06, None)));
    }

    #[test]
    fn faulting_load_commits_nothing() {
        // movups from an unmapped address must leave xmm0 alone.
        let mut code = mov_rax(0xFFFF_8000_0000_0000);
        code.extend_from_slice(&[0x0F, 0x10, 0x00, 0xF4]);
        let mut cpu = cpu64(&code);
        cpu.xmm[0] = 0x1234;
        cpu.step(); cpu.step();
        assert!(cpu.pending_exception.is_some());
        assert_eq!(cpu.xmm[0], 0x1234);
    }
}
