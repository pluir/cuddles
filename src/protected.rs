//! Protected-mode support: segment descriptors, the GDT/IDT, and address
//! translation.
//!
//! A segment descriptor is an 8-byte structure. We parse it into a
//! `Descriptor` with a base, a limit, and an attribute byte, and cache the
//! resolved base/limit/attributes for each loaded segment register.

use crate::cpu::SegReg;

/// A parsed segment descriptor.
#[derive(Clone, Copy, Debug, Default)]
pub struct Descriptor {
    pub base: u32,
    pub limit: u32,
    /// Attribute byte: bits 7-4 = type, bit 3 = S, bits 2-1 = DPL, bit 0 = P.
    pub attr: u8,
    /// Granularity bit (G) and default operand size bit (D/B).
    pub g: bool,
    pub d_b: bool,
}

impl Descriptor {
    /// Parse an 8-byte descriptor (little-endian u64).
    pub fn parse(raw: u64) -> Descriptor {
        // 386 descriptor layout:
        //   bits 0-15:  limit15:0
        //   bits 16-31: base23:16
        //   bits 32-39: base15:0
        //   bits 40-47: type/attr
        //   bits 48-51: limit19:16
        //   bits 52-55: AVL, L, D/B, G
        //   bits 56-63: base31:24
        let base = ((raw >> 32) & 0xFF) as u32
            | (((raw >> 16) & 0xFFFF) as u32) << 16
            | (((raw >> 56) & 0xFF) as u32) << 24;
        let limit20 = ((raw & 0xFFFF) as u32) | ((((raw >> 48) & 0xF) as u32) << 16);
        let g = (raw >> 55) & 1 == 1;
        let d_b = (raw >> 54) & 1 == 1;
        let limit = if g { (limit20 << 12) | 0xFFF } else { limit20 };
        let attr = ((raw >> 40) & 0xFF) as u8;
        Descriptor { base, limit, attr, g, d_b }
    }

    /// True if the descriptor is present.
    pub fn present(&self) -> bool { self.attr & 1 != 0 }

    /// True if this is a code segment (S=1, type bit 3 = 1).
    pub fn is_code(&self) -> bool {
        (self.attr & 0x18) == 0x18
    }

    /// True if this is a data segment (S=1, type bit 3 = 0).
    pub fn is_data(&self) -> bool {
        (self.attr & 0x18) == 0x10
    }

    /// True if this is a system segment (S=0).
    pub fn is_system(&self) -> bool {
        self.attr & 0x10 == 0
    }

    /// The DPL (descriptor privilege level), bits 6-5 of the attribute.
    pub fn dpl(&self) -> u8 { (self.attr >> 5) & 0x3 }
}

/// Read a descriptor from the GDT/IDT at `base + index*8`.
pub fn read_descriptor(mem: &crate::memory::Memory, base: u32, index: u16) -> Descriptor {
    let addr = Memory::phys32(base.wrapping_add((index as u32) * 8));
    Descriptor::parse(mem.read_u64(addr))
}

use crate::memory::Memory;

/// Translate a logical address in protected mode to a physical address.
/// `seg` is the cached descriptor for the segment register.
pub fn translate(seg: &Descriptor, offset: u32) -> u32 {
    seg.base.wrapping_add(offset)
}

/// The index of a segment register in the cached-descriptor arrays.
pub fn seg_index(s: SegReg) -> usize {
    s as usize
}
