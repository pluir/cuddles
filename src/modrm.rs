//! ModR/M byte decoding and register-index helpers.
//!
//! The ModR/M byte encodes a register field (`reg`) and a register/memory
//! operand (`rm`), qualified by a 2-bit `mod` field. When `mod == 3`, `rm`
//! selects a register directly; otherwise it encodes a memory addressing mode
//! (with an optional displacement fetched from the instruction stream).
//!
//! In 32-bit address mode a SIB (scale-index-base) byte follows the ModR/M
//! byte when `rm == 4`, and displacements may be 32 bits wide.

use crate::cpu::{Reg8, Reg16, Reg32};

/// Decoded ModR/M descriptor (without the displacement bytes, which the CPU
/// fetches separately and stores back here).
///
/// `reg` and `rm` are *register indices*, already widened to four bits by the
/// REX prefix. `rm_raw` keeps the three bits as encoded, because every
/// addressing decision is made on those and only those: `rm == 100b` means a
/// SIB byte follows and `mod == 00, rm == 101b` means RIP-relative, whatever
/// REX.B says. Extending before those tests turns R12 into a SIB escape and
/// R13 into a RIP-relative operand -- which is exactly the bug the split
/// exists to prevent.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModRm {
    pub mod_field: u8,
    /// The `reg` field, extended by REX.R: a register index 0-15.
    pub reg: u8,
    /// The `rm` field as a register index, extended by REX.B when
    /// `mod == 3` (when it names a register rather than an addressing mode).
    pub rm: u8,
    /// The `rm` field exactly as encoded, three bits.
    pub rm_raw: u8,
    /// REX.B and REX.X as they applied to this byte, for the base and index
    /// registers of a memory operand.
    pub rex_b: bool,
    pub rex_x: bool,
    /// True when the operand is RIP-relative (64-bit addressing, mod == 00,
    /// rm == 101b).
    pub rip_rel: bool,
    /// SIB byte, present in 32/64-bit address mode when rm == 4 (mod != 3).
    pub sib: Option<u8>,
    pub disp8: Option<u16>,
    pub disp16: Option<u16>,
    pub disp32: Option<u32>,
}

impl ModRm {
    pub fn from_byte(byte: u8) -> Self {
        ModRm {
            mod_field: (byte >> 6) & 0b11,
            reg: (byte >> 3) & 0b111,
            rm: byte & 0b111,
            rm_raw: byte & 0b111,
            rex_b: false,
            rex_x: false,
            rip_rel: false,
            sib: None,
            disp8: None,
            disp16: None,
            disp32: None,
        }
    }

    /// True when `mod == 3` (register-direct operand).
    pub fn is_reg(&self) -> bool {
        self.mod_field == 3
    }
}

/// Helpers to map a 3-bit `reg`/`rm` index to a concrete register enum.
pub struct Reg;

impl Reg {
    pub fn reg8(i: u8) -> Reg8 {
        match i & 7 {
            0 => Reg8::Al, 1 => Reg8::Cl, 2 => Reg8::Dl, 3 => Reg8::Bl,
            4 => Reg8::Ah, 5 => Reg8::Ch, 6 => Reg8::Dh, _ => Reg8::Bh,
        }
    }

    pub fn reg16(i: u8) -> Reg16 {
        match i & 7 {
            0 => Reg16::Ax, 1 => Reg16::Cx, 2 => Reg16::Dx, 3 => Reg16::Bx,
            4 => Reg16::Sp, 5 => Reg16::Bp, 6 => Reg16::Si, _ => Reg16::Di,
        }
    }

    pub fn reg32(i: u8) -> Reg32 {
        match i & 7 {
            0 => Reg32::Eax, 1 => Reg32::Ecx, 2 => Reg32::Edx, 3 => Reg32::Ebx,
            4 => Reg32::Esp, 5 => Reg32::Ebp, 6 => Reg32::Esi, _ => Reg32::Edi,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_fields() {
        let m = ModRm::from_byte(0b11_010_001);
        assert_eq!(m.mod_field, 3);
        assert_eq!(m.reg, 0b010);
        assert_eq!(m.rm, 0b001);
        assert!(m.is_reg());
    }
}
