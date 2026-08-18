//! 8259 Programmable Interrupt Controller (PIC).
//!
//! Two cascaded PICs (master at 0x20/0x21, slave at 0xA0/0xA1) provide 15
//! hardware interrupt lines (IRQ0-IRQ15). Each PIC maps its 8 IRQs onto a
//! configurable base vector in the CPU's IDT/IVT.
//!
//! I/O ports:
//!   master: 0x20 (command), 0x21 (data)
//!   slave:  0xA0 (command), 0xA1 (data)
//!
//! The master's IRQ2 is the cascade input from the slave.

/// The 8259 PIC pair.
pub struct Pic {
    // Master
    pub master_base: u8,      // base vector for IRQ0-7
    pub master_imr: u8,       // interrupt mask register
    pub master_irr: u8,       // interrupt request register (latched)
    pub master_isr: u8,       // in-service register
    pub master_auto_eoi: bool,
    // Slave
    pub slave_base: u8,       // base vector for IRQ8-15
    pub slave_imr: u8,
    pub slave_irr: u8,
    pub slave_isr: u8,
    pub slave_auto_eoi: bool,
    // Cascade: master IRQ2 is the slave's output.
    pub cascade_irq: u8,      // which master IRQ the slave feeds (default 2)
    // ICW2 has been written (initialization complete).
    pub initialized: bool,
    // ICW1 pending state.
    pub icw2_pending: bool,
    pub icw2_for_slave: bool,
    // OCW3 selected the ISR (rather than the IRR) for the next command-port
    // read, per PIC.
    pub master_read_isr: bool,
    pub slave_read_isr: bool,
}

impl Pic {
    pub fn new() -> Self {
        Pic {
            master_base: 0x08,
            master_imr: 0,
            master_irr: 0,
            master_isr: 0,
            master_auto_eoi: false,
            slave_base: 0x70,
            slave_imr: 0,
            slave_irr: 0,
            slave_isr: 0,
            slave_auto_eoi: false,
            cascade_irq: 2,
            initialized: false,
            icw2_pending: false,
            icw2_for_slave: false,
            master_read_isr: false,
            slave_read_isr: false,
        }
    }

    /// Write a command byte to a PIC (port 0x20 or 0xA0).
    pub fn write_command(&mut self, port: u8, val: u8) {
        let slave = port == 0xA0;
        if val & 0x10 != 0 {
            // ICW1: initialization command word 1.
            // Bits 7-5: ICW4 needed (bit 0 of ICW1 = ICW4), edge/level, etc.
            // We only need ICW2 next.
            self.icw2_pending = true;
            self.icw2_for_slave = slave;
            if slave {
                self.slave_irr = 0;
                self.slave_isr = 0;
            } else {
                self.master_irr = 0;
                self.master_isr = 0;
            }
            return;
        }
        // Bits 4:3 tell OCW2 (00) from OCW3 (01).
        match val & 0x18 {
            0x00 => {
                // OCW2: bits 7:5 are the command. 001 is a non-specific EOI
                // (clear the highest-priority in-service bit), 011 a
                // specific EOI (clear the bit named in 2:0); the rotate
                // variants (101, 111) EOI the same way -- priority rotation
                // is not modelled. Linux uses the specific form, `0x60 |
                // irq`, so decoding on the low bits alone missed every EOI
                // but IRQ0's.
                let isr = if slave { &mut self.slave_isr } else { &mut self.master_isr };
                match val >> 5 {
                    0b001 | 0b101 => {
                        if *isr != 0 { *isr &= !(1 << lowest_bit(*isr)); }
                    }
                    0b011 | 0b111 => { *isr &= !(1 << (val & 7)); }
                    _ => {}
                }
                // The master's cascade line stays in service while the slave
                // has anything in service; a slave EOI that empties it lets
                // the master's IRQ2 go too.
                if slave && self.slave_isr == 0 {
                    self.master_isr &= !(1 << self.cascade_irq);
                }
            }
            0x08 => {
                // OCW3: bits 1:0 = 10 read IRR, 11 read ISR on the next read
                // of the command port.
                if val & 3 == 3 || val & 3 == 2 {
                    let read_isr = val & 3 == 3;
                    if slave { self.slave_read_isr = read_isr; } else { self.master_read_isr = read_isr; }
                }
            }
            _ => {}
        }
    }

    /// Write a data byte (ICW2 base vector or OCW1 mask) to a PIC
    /// (port 0x21 or 0xA1).
    pub fn write_data(&mut self, port: u8, val: u8) {
        let slave = port == 0xA1;
        if self.icw2_pending {
            // ICW2: base vector.
            if slave {
                self.slave_base = val & 0xF8;
            } else {
                self.master_base = val & 0xF8;
            }
            self.icw2_pending = false;
            self.initialized = true;
            return;
        }
        // OCW1: interrupt mask register.
        if slave {
            self.slave_imr = val;
        } else {
            self.master_imr = val;
        }
    }

    /// Read a command byte (IRR, or ISR after an OCW3 asked for it) from a
    /// PIC (port 0x20 or 0xA0).
    pub fn read_command(&self, port: u8) -> u8 {
        if port == 0xA0 {
            if self.slave_read_isr { self.slave_isr } else { self.slave_irr }
        } else if self.master_read_isr {
            self.master_isr
        } else {
            self.master_irr
        }
    }

    /// Read the interrupt mask register (port 0x21 or 0xA1).
    pub fn read_data(&self, port: u8) -> u8 {
        if port == 0xA1 { self.slave_imr } else { self.master_imr }
    }

    /// Raise an IRQ line (0-15). Latches the request.
    pub fn raise_irq(&mut self, irq: u8) {
        if irq < 8 {
            self.master_irr |= 1 << irq;
        } else {
            self.slave_irr |= 1 << (irq - 8);
        }
    }

    /// Clear a pending IRQ line (used after servicing).
    pub fn clear_irq(&mut self, irq: u8) {
        if irq < 8 {
            self.master_irr &= !(1 << irq);
        } else {
            self.slave_irr &= !(1 << (irq - 8));
        }
    }

    /// The IRQ (0-15) the PIC pair would deliver next: the highest-priority
    /// pending unmasked request that is not blocked by an in-service bit of
    /// equal or higher priority (fixed priority, IRQ0 highest, with the
    /// slave's eight lines sitting at the master's cascade input). This is
    /// what keeps a handler from being re-entered by its own interrupt
    /// before it EOIs, and it is the whole of the "in a handler" state --
    /// there is no separate flag, so a handler that leaves through SYSRET
    /// rather than IRET blocks nothing.
    fn next_irq(&self) -> Option<u8> {
        let pending = self.master_irr & !self.master_imr;
        let cascade = 1u8 << self.cascade_irq;
        // Priority walk on the master: stop at the first in-service bit.
        for i in 0..8u8 {
            let bit = 1u8 << i;
            if self.master_isr & bit != 0 && bit != cascade {
                return None;
            }
            if bit == cascade {
                // The slave's lines live here. Its own priority walk applies,
                // and its requests are visible only when the cascade line is
                // not masked.
                if self.master_imr & cascade == 0 {
                    let spending = self.slave_irr & !self.slave_imr;
                    for j in 0..8u8 {
                        let sbit = 1u8 << j;
                        if self.slave_isr & sbit != 0 { return None; }
                        if spending & sbit != 0 { return Some(8 + j); }
                    }
                }
                if self.master_isr & bit != 0 { return None; }
                continue;
            }
            if pending & bit != 0 {
                return Some(i);
            }
        }
        None
    }

    /// Acknowledge an interrupt: return the vector of the IRQ `next_irq`
    /// names, moving it from requested to in-service. `None` when nothing
    /// can be delivered.
    pub fn acknowledge(&mut self) -> Option<u8> {
        let irq = self.next_irq()?;
        if irq >= 8 {
            let bit = irq - 8;
            self.slave_irr &= !(1 << bit);
            self.slave_isr |= 1 << bit;
            self.master_isr |= 1 << self.cascade_irq;
            Some(self.slave_base + bit)
        } else {
            self.master_irr &= !(1 << irq);
            self.master_isr |= 1 << irq;
            Some(self.master_base + irq)
        }
    }

    /// True when an interrupt is waiting and could be delivered now.
    pub fn has_pending(&self) -> bool {
        self.next_irq().is_some()
    }

    /// The vector currently being serviced, if any (for EOI bookkeeping).
    pub fn in_service(&self) -> bool {
        self.master_isr != 0 || self.slave_isr != 0
    }
}

/// Index of the lowest set bit in a byte.
fn lowest_bit(v: u8) -> u8 {
    for i in 0..8 {
        if v & (1 << i) != 0 { return i; }
    }
    0
}

impl Default for Pic {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_and_acknowledge_master() {
        let mut pic = Pic::new();
        // ICW1 (port 0x20), ICW2 base = 0x08 (port 0x21).
        pic.write_command(0x20, 0x11);
        pic.write_data(0x21, 0x08);
        assert_eq!(pic.master_base, 0x08);
        // Raise IRQ1 (keyboard) and acknowledge.
        pic.raise_irq(1);
        assert_eq!(pic.acknowledge(), Some(0x09));
        assert!(pic.in_service());
    }

    #[test]
    fn masked_irq_not_acknowledged() {
        let mut pic = Pic::new();
        pic.write_command(0x20, 0x11);
        pic.write_data(0x21, 0x08);
        pic.master_imr = 0x02; // mask IRQ1
        pic.raise_irq(1);
        assert_eq!(pic.acknowledge(), None);
    }

    #[test]
    fn in_service_blocks_the_same_and_lower_priority_until_eoi() {
        let mut pic = Pic::new();
        pic.write_command(0x20, 0x11);
        pic.write_data(0x21, 0x20);
        pic.raise_irq(1);
        assert_eq!(pic.acknowledge(), Some(0x21));
        // IRQ1 in service: another IRQ1 waits, and so does IRQ3 (lower
        // priority) -- but IRQ0 (higher) gets through.
        pic.raise_irq(1);
        pic.raise_irq(3);
        assert_eq!(pic.acknowledge(), None);
        pic.raise_irq(0);
        assert_eq!(pic.acknowledge(), Some(0x20));
        // Specific EOI for IRQ0 (0x60), then for IRQ1 (0x61): the way Linux
        // ends every interrupt.
        pic.write_command(0x20, 0x60);
        assert_eq!(pic.acknowledge(), None, "IRQ1 still in service");
        pic.write_command(0x20, 0x61);
        assert_eq!(pic.acknowledge(), Some(0x21));
        pic.write_command(0x20, 0x20); // non-specific EOI
        assert_eq!(pic.acknowledge(), Some(0x23));
    }

    #[test]
    fn ocw3_selects_isr_reads() {
        let mut pic = Pic::new();
        pic.write_command(0x20, 0x11);
        pic.write_data(0x21, 0x20);
        pic.raise_irq(4);
        pic.acknowledge();
        assert_eq!(pic.read_command(0x20), 0, "IRR: acknowledged");
        pic.write_command(0x20, 0x0B);
        assert_eq!(pic.read_command(0x20), 0x10, "ISR");
        pic.write_command(0x20, 0x0A);
        assert_eq!(pic.read_command(0x20), 0);
    }

    #[test]
    fn slave_irq_maps_to_base() {
        let mut pic = Pic::new();
        pic.write_command(0x20, 0x11);
        pic.write_data(0x21, 0x08);
        pic.write_command(0xA0, 0x11);
        pic.write_data(0xA1, 0x70);
        pic.raise_irq(8); // slave IRQ0
        assert_eq!(pic.acknowledge(), Some(0x70));
    }
}
