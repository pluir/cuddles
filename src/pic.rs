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
        match val & 0x7 {
            0 => { // EOI (end of interrupt)
                if slave {
                    self.slave_isr = 0;
                } else {
                    self.master_isr = 0;
                    // If the master was servicing the cascade, clear it too.
                    if self.master_isr & (1 << self.cascade_irq) != 0 {
                        self.master_isr &= !(1 << self.cascade_irq);
                    }
                }
            }
            3 => { // read ISR
                // Reading the ISR is done via read_command; nothing to do here.
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

    /// Read a command byte (IRR or ISR) from a PIC (port 0x20 or 0xA0).
    pub fn read_command(&self, port: u8) -> u8 {
        // We always return the IRR for simplicity.
        if port == 0xA0 { self.slave_irr } else { self.master_irr }
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

    /// Acknowledge an interrupt: return the vector of the highest-priority
    /// pending, unmasked IRQ, and mark it in-service. Returns `None` if no
    /// interrupt is pending.
    pub fn acknowledge(&mut self) -> Option<u8> {
        // Slave has priority (its IRQs 8-15 map above master IRQ2 cascade).
        // Check slave first.
        if self.slave_irr & !self.slave_imr != 0 {
            let pending = self.slave_irr & !self.slave_imr;
            let bit = lowest_bit(pending);
            self.slave_irr &= !(1 << bit);
            self.slave_isr |= 1 << bit;
            // Cascade: master IRQ2 is in service.
            self.master_isr |= 1 << self.cascade_irq;
            return Some(self.slave_base + bit);
        }
        // Master (excluding the cascade line, handled above).
        let pending = self.master_irr & !self.master_imr;
        if pending != 0 {
            let bit = lowest_bit(pending);
            self.master_irr &= !(1 << bit);
            self.master_isr |= 1 << bit;
            return Some(self.master_base + bit);
        }
        None
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
