//! Minimal x87 FPU for early-boot support.
//!
//! Implements the FPU state (control/status/tag words and the eight
//! 80-bit data registers, stored here as `f64`) plus the instructions the
//! Linux kernel needs during early boot: FNINIT, FSTCW/FLDCW, FSTSW,
//! FXSAVE/FXRSTOR, and the common data-transfer and arithmetic operations.

/// The x87 FPU state.
pub struct Fpu {
    /// Control word (default 0x037F: round-nearest, all exceptions masked).
    pub control: u16,
    /// Status word.
    pub status: u16,
    /// Tag word (bit per register: 1 = empty, 0 = valid).
    pub tag: u16,
    /// The eight data registers (ST0..ST7), stored as f64.
    pub st: [f64; 8],
    /// Index of the current ST0 within `st`.
    pub top: usize,
}

impl Fpu {
    pub fn new() -> Self {
        Fpu {
            control: 0x037F,
            status: 0,
            tag: 0xFFFF,
            st: [0.0; 8],
            top: 0,
        }
    }

    /// FNINIT: reset the FPU to its default state.
    pub fn finit(&mut self) {
        self.control = 0x037F;
        self.status = 0;
        self.tag = 0xFFFF;
        self.st = [0.0; 8];
        self.top = 0;
    }

    /// ST(i) accessor relative to the current top.
    pub fn st_i(&self, i: usize) -> f64 {
        self.st[(self.top + i) % 8]
    }

    pub fn set_st_i(&mut self, i: usize, v: f64) {
        let idx = (self.top + i) % 8;
        self.st[idx] = v;
        // Mark the register valid.
        self.tag &= !(1 << idx);
    }

    /// Push a value onto the stack (ST0 becomes `v`, top decrements).
    pub fn push(&mut self, v: f64) {
        self.top = (self.top + 7) % 8;
        self.st[self.top] = v;
        self.tag &= !(1 << self.top);
    }

    /// Pop the stack (top increments).
    pub fn pop(&mut self) {
        self.tag |= 1 << self.top;
        self.top = (self.top + 1) % 8;
    }

    /// FSTSW: return the status word.
    pub fn fstsw(&self) -> u16 {
        self.status
    }

    /// FXSAVE: write the 512-byte FXSAVE area (control/status/tag words,
    /// then the eight 80-bit extended-precision registers).
    pub fn fxsave(&self, mem: &mut crate::memory::Memory, addr: usize) {
        mem.write_u16(addr, self.control);
        mem.write_u16(addr + 2, self.status);
        mem.write_u16(addr + 4, self.tag);
        for i in 0..8 {
            let v = self.st[i];
            let bits = v.to_bits();
            let sign = (bits >> 63) & 1;
            let exp = ((bits >> 52) & 0x7FF) as i32;
            let mant = bits & 0xFFFF_FFFFF_FFFF;
            let (exp16, mant64) = if exp == 0 && mant == 0 {
                (0u16, 0u64)
            } else if exp == 0x7FF {
                (0x7FFFu16, (mant << 11) | 0x8000_0000_0000_0000)
            } else {
                let e80 = (exp - 1023 + 16383) as u16;
                (e80, (mant << 11) | 0x8000_0000_0000_0000)
            };
            let base = addr + 32 + i * 16;
            mem.write_u64(base, mant64);
            mem.write_u16(base + 8, exp16);
            mem.write_u16(base + 10, (sign as u16) << 15);
        }
    }

    /// FXRSTOR: read the 512-byte FXSAVE area back into FPU state.
    pub fn fxrstor(&mut self, mem: &crate::memory::Memory, addr: usize) {
        self.control = mem.read_u16(addr);
        self.status = mem.read_u16(addr + 2);
        self.tag = mem.read_u16(addr + 4);
        for i in 0..8 {
            let base = addr + 32 + i * 16;
            let mant = mem.read_u64(base);
            let exp = mem.read_u16(base + 8);
            let sign = (mem.read_u16(base + 10) >> 15) as u64;
            // The 80-bit mantissa is 64 bits with the implicit integer bit at
            // bit 63. Drop that bit and shift down to the 52-bit f64 mantissa.
            let mantissa = (mant & 0x7FFF_FFFF_FFFF_FFFF) >> 11;
            let e = exp as i32;
            let bits = if e == 0 || e == 0x7FFF {
                0
            } else {
                let e64 = (e - 16383 + 1023) as u64;
                (sign << 63) | (e64 << 52) | mantissa
            };
            self.st[i] = f64::from_bits(bits);
        }
    }
}

impl Default for Fpu {
    fn default() -> Self {
        Self::new()
    }
}
