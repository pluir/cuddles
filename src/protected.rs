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
    /// Access byte: bit 7 = P (present), bits 6-5 = DPL, bit 4 = S
    /// (0 = system segment), bits 3-0 = type.
    pub attr: u8,
    /// Granularity bit (G) and default operand size bit (D/B).
    pub g: bool,
    pub d_b: bool,
    /// The L bit (bit 53): this is a 64-bit code segment. It is what
    /// distinguishes 64-bit code from compatibility-mode code inside long
    /// mode, and it is mutually exclusive with D/B -- a descriptor with both
    /// set is illegal, because "64-bit operands" and "32-bit operands" cannot
    /// both be the default.
    pub l: bool,
}

impl Descriptor {
    /// Parse an 8-byte descriptor (little-endian u64).
    pub fn parse(raw: u64) -> Descriptor {
        // 386 descriptor layout, as a little-endian u64:
        //   bits 0-15:  limit 15:0
        //   bits 16-31: base 15:0
        //   bits 32-39: base 23:16
        //   bits 40-47: type/attr
        //   bits 48-51: limit 19:16
        //   bits 52-55: AVL, L, D/B, G  (L marks a 64-bit code segment)
        //   bits 56-63: base 31:24
        let base = (((raw >> 16) & 0xFFFF) as u32)
            | ((((raw >> 32) & 0xFF) as u32) << 16)
            | ((((raw >> 56) & 0xFF) as u32) << 24);
        let limit20 = ((raw & 0xFFFF) as u32) | ((((raw >> 48) & 0xF) as u32) << 16);
        let g = (raw >> 55) & 1 == 1;
        let d_b = (raw >> 54) & 1 == 1;
        let l = (raw >> 53) & 1 == 1;
        let limit = if g { (limit20 << 12) | 0xFFF } else { limit20 };
        let attr = ((raw >> 40) & 0xFF) as u8;
        Descriptor { base, limit, attr, g, d_b, l }
    }

    /// True if the descriptor is present.
    pub fn present(&self) -> bool { self.attr & 0x80 != 0 }

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

/// Read a descriptor from a **physical** address.
///
/// The caller resolves the table entry's linear address first (see
/// `Cpu::linear_to_phys_ro`): a descriptor table lives wherever the kernel
/// says it does, which on a kernel running in the higher half is not where
/// its bytes physically are.
pub fn read_descriptor(mem: &crate::memory::Memory, entry_phys: usize) -> Descriptor {
    Descriptor::parse(mem.read_u64(entry_phys))
}

/// Translate a logical address in protected mode to a physical address.
/// `seg` is the cached descriptor for the segment register.
pub fn translate(seg: &Descriptor, offset: u32) -> u32 {
    seg.base.wrapping_add(offset)
}

/// The index of a segment register in the cached-descriptor arrays.
pub fn seg_index(s: SegReg) -> usize {
    s as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_base_is_assembled_from_three_fields() {
        // base = 0xC0FF_1234, limit = 0x000F_FFFF, present code segment,
        // granularity and D/B set. Fields are scattered across the 8 bytes,
        // and putting any of them in the wrong place still yields a plausible
        // number -- which is why this is pinned.
        let raw: u64 = 0x0000_FFFF                    // limit 15:0
            | (0x1234u64 << 16)                       // base 15:0
            | (0xFFu64 << 32)                         // base 23:16
            | (0x9Au64 << 40)                         // access: present, code
            | (0xCFu64 << 48)                         // limit 19:16 + G + D/B
            | (0xC0u64 << 56);                        // base 31:24
        let d = Descriptor::parse(raw);
        assert_eq!(d.base, 0xC0FF_1234);
        assert_eq!(d.limit, 0xFFFF_FFFF);
        assert!(d.present());
        assert!(d.is_code());
        assert_eq!(d.dpl(), 0);
        assert!(d.g);
        assert!(d.d_b);
        assert!(!d.l);
    }

    #[test]
    fn the_l_bit_marks_a_64_bit_code_segment() {
        // A long-mode kernel code segment: present, code, L set, D/B clear.
        // The pair is mutually exclusive, and reading L out of the D/B bit is
        // how a 64-bit segment gets mistaken for a 32-bit one.
        let raw: u64 = (0x9Au64 << 40) | (0x20u64 << 48);
        let d = Descriptor::parse(raw);
        assert!(d.l);
        assert!(!d.d_b);
        assert!(d.is_code());
    }

    #[test]
    fn dpl_is_read_from_the_access_byte() {
        // A ring-3 data segment: access byte 0xF2 (present, DPL 3, data, RW).
        let raw: u64 = 0xF2u64 << 40;
        assert_eq!(Descriptor::parse(raw).dpl(), 3);
    }
}
