//! 8042 keyboard controller (PS/2).
//!
//! The 8042 sits between the keyboard and the CPU. The CPU reads scancodes
//! from port 0x60 and status from port 0x64, and writes commands to 0x64 and
//! data to 0x60. It raises IRQ1 when a scancode is available.
//!
//! I/O ports:
//!   0x60 - data (read: scancode; write: command data)
//!   0x64 - status (read) / command (write)
//!
//! Status bits (port 0x64):
//!   bit 0: output buffer full (a scancode is ready to read)
//!   bit 1: input buffer full
//!   bit 5: output buffer contains a command response (vs. scancode)

/// The 8042 keyboard controller.
pub struct Kbd {
    /// Scancodes waiting to be read from port 0x60.
    pub scancodes: std::collections::VecDeque<u8>,
    /// Status register (read from port 0x64).
    pub status: u8,
    /// IRQ1 asserted when a scancode is available.
    pub irq1: bool,
    /// Pending command byte (written to 0x64).
    pub command: u8,
    /// Whether the controller is in command-response mode.
    pub command_response: bool,
}

impl Kbd {
    pub fn new() -> Self {
        Kbd {
            scancodes: std::collections::VecDeque::new(),
            status: 0,
            irq1: false,
            command: 0,
            command_response: false,
        }
    }

    /// Queue a scancode (e.g. from a host-side key event). Raises IRQ1.
    pub fn push_scancode(&mut self, scancode: u8) {
        self.scancodes.push_back(scancode);
        self.status |= 0x01; // output buffer full
        self.irq1 = true;
    }

    /// Read from port 0x60 (data).
    pub fn read_data(&mut self) -> u8 {
        if let Some(sc) = self.scancodes.pop_front() {
            if self.scancodes.is_empty() {
                self.status &= !0x01;
                self.irq1 = false;
            }
            sc
        } else if self.command_response {
            self.command_response = false;
            self.status &= !0x20;
            // Acknowledge byte.
            0xFA
        } else {
            0
        }
    }

    /// Read from port 0x64 (status).
    pub fn read_status(&self) -> u8 {
        self.status
    }

    /// Write to port 0x64 (command).
    pub fn write_command(&mut self, cmd: u8) {
        self.command = cmd;
        match cmd {
            0x20 => { // read command byte: next data read returns it
                self.command_response = true;
                self.status |= 0x20;
            }
            0xAA => { // self-test: respond with 0x55
                self.command_response = true;
                self.status |= 0x20;
            }
            _ => {}
        }
    }

    /// Write to port 0x60 (data).
    pub fn write_data(&mut self, _val: u8) {
        // Data written to the 8042 (e.g. command byte). Kept minimal.
    }
}

impl Default for Kbd {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scancode_queued_and_read() {
        let mut kbd = Kbd::new();
        kbd.push_scancode(0x1E); // 'A' make
        assert!(kbd.irq1);
        assert_eq!(kbd.read_status() & 0x01, 0x01);
        assert_eq!(kbd.read_data(), 0x1E);
        assert!(!kbd.irq1);
        assert_eq!(kbd.read_status() & 0x01, 0x00);
    }

    #[test]
    fn command_byte_readback() {
        let mut kbd = Kbd::new();
        kbd.write_command(0x20);
        assert_eq!(kbd.read_data(), 0xFA); // acknowledge
    }

    #[test]
    fn empty_read_returns_zero() {
        let mut kbd = Kbd::new();
        assert_eq!(kbd.read_data(), 0);
    }
}
