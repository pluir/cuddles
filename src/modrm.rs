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
#[derive(Clone, Copy, Debug, Default)]
pub struct ModRm {
    pub mod_field: u8,
    pub reg: u8,
    pub rm: u8,
    /// SIB byte, present in 32-bit address mode when rm == 4 (mod != 3).
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
