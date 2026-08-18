//! 8254 Programmable Interval Timer (PIT).
//!
//! Three 16-bit countdown channels. Channel 0 is wired to the PIC's IRQ0
//! (the system timer). The timer is ticked once per emulated instruction.
//!
//! I/O ports:
//!   0x40 - channel 0 data
//!   0x41 - channel 1 data
//!   0x42 - channel 2 data
//!   0x43 - control word
//!
//! Control word (port 0x43):
//!   bits 7-6: channel select (0=ch0, 1=ch1, 2=ch2, 3=read-back)
//!   bits 5-4: access mode (0=latch, 1=LSB, 2=MSB, 3=word)
//!   bits 3-1: operating mode
//!   bit 0:    BCD (1 = BCD counting)

/// The 8254 PIT.
pub struct Pit {
    // Channel 0 (system timer -> IRQ0)
    pub ch0_count: u16,
    pub ch0_reload: u16,
    pub ch0_mode: u8,
    pub ch0_bcd: bool,
    pub ch0_access: u8,
    pub ch0_lsb_pending: bool,
    pub ch0_lsb: u8,
    // Channel 1 (kept minimal; not wired to an IRQ)
    pub ch1_count: u16,
    pub ch1_reload: u16,
    pub ch1_mode: u8,
    pub ch1_access: u8,
    pub ch1_lsb_pending: bool,
    pub ch1_lsb: u8,
    // Channel 2 (speaker; kept minimal)
    pub ch2_count: u16,
    pub ch2_reload: u16,
    pub ch2_mode: u8,
    pub ch2_access: u8,
    pub ch2_lsb_pending: bool,
    pub ch2_lsb: u8,
    /// Channel 2's GATE input, driven by bit 0 of port 0x61.
    pub ch2_gate: bool,
    /// Channel 2's OUT line, visible as bit 5 of port 0x61. In mode 0 it is
    /// low from the moment a count is loaded and goes high when the count
    /// reaches terminal, which is the edge the kernel's PIT-based TSC
    /// calibration spins on.
    pub ch2_out: bool,
    /// Bit 1 of port 0x61: the speaker data line. Stored so a read-back
    /// returns what was written.
    pub speaker_on: bool,
    /// Bit 4 of port 0x61 toggles every DRAM refresh cycle. Firmware and
    /// some timing loops watch it for motion.
    pub refresh_toggle: bool,
    // IRQ0 output, asserted when channel 0 wraps. The PIC latches the edge.
    pub irq0: bool,
    // State set by the most recent control word.
    pub active_channel: u8,
    pub active_access: u8,
    pub active_mode: u8,
    pub active_bcd: bool,
}

impl Pit {
    pub fn new() -> Self {
        Pit {
            ch0_count: 0, ch0_reload: 0, ch0_mode: 3, ch0_bcd: false,
            ch0_access: 3, ch0_lsb_pending: false, ch0_lsb: 0,
            ch1_count: 0, ch1_reload: 0, ch1_mode: 3, ch1_access: 3,
            ch1_lsb_pending: false, ch1_lsb: 0,
            ch2_count: 0, ch2_reload: 0, ch2_mode: 3, ch2_access: 3,
            ch2_lsb_pending: false, ch2_lsb: 0,
            ch2_gate: false, ch2_out: false, speaker_on: false,
            refresh_toggle: false,
            irq0: false,
            active_channel: 0, active_access: 3, active_mode: 3, active_bcd: false,
        }
    }

    /// Write a control word (port 0x43).
    pub fn write_control(&mut self, val: u8) {
        let channel = (val >> 6) & 3;
        let access = (val >> 4) & 3;
        let mode = (val >> 1) & 7;
        let bcd = val & 1 == 1;
        if channel == 3 {
            // Read-back command: not fully implemented; treat as no-op.
            return;
        }
        self.active_channel = channel;
        self.active_access = access;
        self.active_mode = mode;
        self.active_bcd = bcd;
        // Store the mode/access/bcd into the selected channel.
        match channel {
            0 => { self.ch0_mode = mode; self.ch0_access = access; self.ch0_bcd = bcd; }
            1 => { self.ch1_mode = mode; self.ch1_access = access; }
            _ => { self.ch2_mode = mode; self.ch2_access = access; }
        }
    }

    /// Write a data byte to the active channel (ports 0x40-0x42).
    pub fn write_data(&mut self, val: u8) {
        match self.active_channel {
            0 => self.write_ch0(val),
            1 => self.write_ch1(val),
            _ => self.write_ch2(val),
        }
    }

    fn write_ch0(&mut self, val: u8) {
        match self.ch0_access {
            1 => { // LSB only
                self.ch0_reload = (self.ch0_reload & 0xFF00) | val as u16;
                self.ch0_count = self.ch0_reload;
            }
            2 => { // MSB only
                self.ch0_reload = (self.ch0_reload & 0x00FF) | ((val as u16) << 8);
                self.ch0_count = self.ch0_reload;
            }
            _ => { // word: LSB then MSB
                if self.ch0_lsb_pending {
                    self.ch0_reload = (self.ch0_lsb as u16) | ((val as u16) << 8);
                    self.ch0_count = self.ch0_reload;
                    self.ch0_lsb_pending = false;
                } else {
                    self.ch0_lsb = val;
                    self.ch0_lsb_pending = true;
                }
            }
        }
    }

    fn write_ch1(&mut self, val: u8) {
        match self.ch1_access {
            1 => { self.ch1_reload = (self.ch1_reload & 0xFF00) | val as u16; self.ch1_count = self.ch1_reload; }
            2 => { self.ch1_reload = (self.ch1_reload & 0x00FF) | ((val as u16) << 8); self.ch1_count = self.ch1_reload; }
            _ => {
                if self.ch1_lsb_pending {
                    self.ch1_reload = (self.ch1_lsb as u16) | ((val as u16) << 8);
                    self.ch1_count = self.ch1_reload;
                    self.ch1_lsb_pending = false;
                } else {
                    self.ch1_lsb = val;
                    self.ch1_lsb_pending = true;
                }
            }
        }
    }

    fn write_ch2(&mut self, val: u8) {
        match self.ch2_access {
            1 => { self.ch2_reload = (self.ch2_reload & 0xFF00) | val as u16; self.ch2_count = self.ch2_reload; self.ch2_out = false; }
            2 => { self.ch2_reload = (self.ch2_reload & 0x00FF) | ((val as u16) << 8); self.ch2_count = self.ch2_reload; self.ch2_out = false; }
            _ => {
                if self.ch2_lsb_pending {
                    self.ch2_reload = (self.ch2_lsb as u16) | ((val as u16) << 8);
                    self.ch2_count = self.ch2_reload;
                    self.ch2_lsb_pending = false;
                    // Loading a new count restarts the countdown with OUT low.
                    self.ch2_out = false;
                } else {
                    self.ch2_lsb = val;
                    self.ch2_lsb_pending = true;
                }
            }
        }
    }

    /// Port 0x61 (system control port B) read: the speaker/gate bits back,
    /// the refresh toggle, and channel 2's OUT line.
    pub fn read_port61(&mut self) -> u8 {
        self.refresh_toggle = !self.refresh_toggle;
        (self.ch2_gate as u8)
            | ((self.speaker_on as u8) << 1)
            | ((self.refresh_toggle as u8) << 4)
            | ((self.ch2_out as u8) << 5)
    }

    /// Port 0x61 write: bit 0 gates channel 2, bit 1 enables the speaker.
    pub fn write_port61(&mut self, val: u8) {
        self.ch2_gate = val & 1 != 0;
        self.speaker_on = val & 2 != 0;
    }

    /// Read a data byte from a channel (ports 0x40-0x42). Returns the
    /// current count's LSB or MSB depending on the access mode.
    pub fn read_data(&mut self, port: u8) -> u8 {
        let access = match port {
            0 => self.ch0_access,
            1 => self.ch1_access,
            _ => self.ch2_access,
        };
        match access {
            1 => {
                let c = match port { 0 => self.ch0_count, 1 => self.ch1_count, _ => self.ch2_count };
                (c & 0xFF) as u8
            }
            2 => {
                let c = match port { 0 => self.ch0_count, 1 => self.ch1_count, _ => self.ch2_count };
                (c >> 8) as u8
            }
            _ => {
                // Word access: return LSB then MSB on successive reads.
                let pending = match port {
                    0 => self.ch0_lsb_pending,
                    1 => self.ch1_lsb_pending,
                    _ => self.ch2_lsb_pending,
                };
                if !pending {
                    let c = match port { 0 => self.ch0_count, 1 => self.ch1_count, _ => self.ch2_count };
                    match port {
                        0 => { self.ch0_lsb = (c & 0xFF) as u8; self.ch0_lsb_pending = true; }
                        1 => { self.ch1_lsb = (c & 0xFF) as u8; self.ch1_lsb_pending = true; }
                        _ => { self.ch2_lsb = (c & 0xFF) as u8; self.ch2_lsb_pending = true; }
                    }
                    (c & 0xFF) as u8
                } else {
                    let c = match port { 0 => self.ch0_count, 1 => self.ch1_count, _ => self.ch2_count };
                    match port {
                        0 => { self.ch0_lsb_pending = false; }
                        1 => { self.ch1_lsb_pending = false; }
                        _ => { self.ch2_lsb_pending = false; }
                    }
                    (c >> 8) as u8
                }
            }
        }
    }

    /// Advance the timer by `cycles` (one per emulated instruction). Channel
    /// 0 drives IRQ0.
    pub fn tick(&mut self, cycles: u64) {
        if self.ch0_count > 0 {
            let c = self.ch0_count as u64;
            if c <= cycles {
                // Wrap: reload and assert IRQ0.
                self.ch0_count = self.ch0_reload;
                self.irq0 = true;
            } else {
                self.ch0_count = (c - cycles) as u16;
            }
        }
        // Channels 1 and 2 also count down (no IRQ wiring).
        if self.ch1_count > 0 {
            self.ch1_count = self.ch1_count.saturating_sub(cycles as u16);
            if self.ch1_count == 0 { self.ch1_count = self.ch1_reload; }
        }
        // Channel 2 counts only while its gate is high. Reaching terminal
        // count raises OUT and, in mode 0, leaves it there until a new count
        // is written -- the level the calibration loop is waiting for.
        if self.ch2_gate && !self.ch2_out {
            let c = self.ch2_count as u64;
            if c <= cycles {
                self.ch2_count = 0;
                self.ch2_out = true;
            } else {
                self.ch2_count = (c - cycles) as u16;
            }
        }
    }
}

impl Default for Pit {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel0_counts_down_and_asserts_irq0() {
        let mut pit = Pit::new();
        // Program channel 0, word access, mode 3, count = 5.
        pit.write_control(0x36); // ch0, word, mode 3, binary
        pit.write_data(5);
        pit.write_data(0);
        assert_eq!(pit.ch0_count, 5);
        assert!(!pit.irq0);
        pit.tick(3);
        assert_eq!(pit.ch0_count, 2);
        assert!(!pit.irq0);
        pit.tick(2);
        assert!(pit.irq0);
        assert_eq!(pit.ch0_count, 5); // reloaded
    }

    #[test]
    fn control_word_selects_channel() {
        let mut pit = Pit::new();
        // Program channel 1, word access, mode 3.
        pit.write_control(0x76); // ch1, word, mode 3
        pit.write_data(0x34);
        pit.write_data(0x12);
        assert_eq!(pit.ch1_count, 0x1234);
        assert_eq!(pit.ch1_reload, 0x1234);
    }
}
