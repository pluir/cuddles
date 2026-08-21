//! 8042 keyboard controller (PS/2).
//!
//! The 8042 sits between the keyboard and the CPU. The CPU reads bytes from
//! port 0x60 and status from port 0x64, writes *controller* commands to 0x64
//! and *data* -- either a command's argument or a command for the keyboard
//! device itself -- to 0x60. It raises IRQ1 when a byte is waiting.
//!
//! Two things make this more than a scancode queue, and Linux needs both:
//!
//! 1. **The controller command byte** (CTR, sometimes CCB). `i8042_probe`
//!    reads it with command 0x20 before it will touch anything else, and
//!    gives up with `Can't read CTR` -- and the whole console with it -- if
//!    the read does not come back. It then writes it back with 0x60 to turn
//!    interrupts and translation on and off, so the byte has to *round trip*;
//!    answering every command with a fixed acknowledge byte is what left the
//!    emulated machine without a keyboard.
//! 2. **One output buffer, shared.** Command responses and scancodes come
//!    back through the same port, and the status register says whether the
//!    byte at the head came from the keyboard or the auxiliary (mouse) port.
//!    A response is not "a different mode" -- it is a byte in the same queue.
//!
//! There is no mouse on this machine. That is not modelled by leaving the
//! auxiliary port silent by accident: the port test answers 0xFF ("no
//! device") so the probe concludes it and moves on, and a byte written
//! through to the absent device is dropped so the wait times out, which is
//! exactly what the absence looks like on real hardware.

use std::collections::VecDeque;

// Status register (port 0x64).
const ST_OBF: u8 = 0x01; // output buffer full
const ST_SYS: u8 = 0x04; // system flag, mirrors CTR bit 2
const ST_CMD: u8 = 0x08; // last write went to 0x64 rather than 0x60
const ST_UNLOCKED: u8 = 0x10; // keyboard not inhibited
const ST_AUX: u8 = 0x20; // the byte at the head came from the aux port

// Controller command byte.
const CTR_KBD_INT: u8 = 0x01; // IRQ1 on a keyboard byte
const CTR_SYS: u8 = 0x04;
const CTR_KBD_DISABLE: u8 = 0x10;
const CTR_AUX_DISABLE: u8 = 0x20;

/// What a BIOS leaves behind: keyboard interrupt on, system flag set,
/// translation on, and the auxiliary port disabled because nothing is on it.
const CTR_DEFAULT: u8 = CTR_KBD_INT | CTR_SYS | CTR_AUX_DISABLE | 0x40;

/// Acknowledge, the keyboard's answer to almost every command.
const ACK: u8 = 0xFA;

/// The 8042 keyboard controller.
pub struct Kbd {
    /// Bytes waiting at port 0x60, each flagged with whether it came from the
    /// auxiliary port. Scancodes and command responses share this queue, as
    /// they share the one buffer on the real device.
    out: VecDeque<(u8, bool)>,
    /// The controller command byte: read with 0x20, written with 0x60.
    pub ctr: u8,
    /// A controller command that is still waiting for its argument at 0x60.
    pending: Option<u8>,
    /// A keyboard command that is still waiting for its argument at 0x60.
    dev_pending: Option<u8>,
    /// Set when a byte is waiting and its interrupt is enabled. The CPU
    /// consumes this as an edge and hands it to the PIC.
    pub irq1: bool,
    /// True when the last write went to 0x64 (status bit 3).
    cmd_written: bool,
    /// Keyboard scanning, toggled by the device commands 0xF4 and 0xF5. A
    /// keyboard told to stop scanning sends nothing, so host key events are
    /// dropped rather than queued behind the guest's back.
    scanning: bool,
    /// Set the first time the *guest* enables scanning (0xF4), which is what
    /// `atkbd` does when it attaches. Scripted keystrokes wait for this:
    /// typing into a machine whose keyboard driver has not bound yet only
    /// throws the bytes away.
    pub driver_attached: bool,
}

impl Kbd {
    pub fn new() -> Self {
        Kbd {
            out: VecDeque::new(),
            ctr: CTR_DEFAULT,
            pending: None,
            dev_pending: None,
            irq1: false,
            cmd_written: false,
            scanning: true,
            driver_attached: false,
        }
    }

    /// Queue a byte from the keyboard.
    fn reply(&mut self, byte: u8) {
        self.out.push_back((byte, false));
        self.refresh_irq();
    }

    /// Queue a byte from the auxiliary port.
    fn reply_aux(&mut self, byte: u8) {
        self.out.push_back((byte, true));
        self.refresh_irq();
    }

    /// IRQ1 follows the byte at the head of the queue: a keyboard byte with
    /// its interrupt enabled asserts it, anything else (an aux byte, an empty
    /// queue, interrupts masked in the CTR) does not.
    fn refresh_irq(&mut self) {
        self.irq1 = matches!(self.out.front(), Some(&(_, false)))
            && (self.ctr & CTR_KBD_INT) != 0;
    }

    /// Queue a scancode from a host key event. Raises IRQ1.
    pub fn push_scancode(&mut self, scancode: u8) {
        if !self.scanning {
            return;
        }
        self.reply(scancode);
    }

    /// Read from port 0x60 (data).
    pub fn read_data(&mut self) -> u8 {
        let byte = self.out.pop_front().map(|(b, _)| b).unwrap_or(0);
        self.refresh_irq();
        byte
    }

    /// True when nothing is waiting to be read. Scripted input feeds one
    /// scancode at a time and waits for this, so the pace follows the guest
    /// draining the buffer rather than a count of instructions.
    pub fn idle(&self) -> bool {
        self.out.is_empty()
    }

    /// Read from port 0x64 (status).
    pub fn read_status(&self) -> u8 {
        let mut status = ST_UNLOCKED;
        if !self.out.is_empty() {
            status |= ST_OBF;
        }
        if self.ctr & CTR_SYS != 0 {
            status |= ST_SYS;
        }
        if self.cmd_written {
            status |= ST_CMD;
        }
        if matches!(self.out.front(), Some(&(_, true))) {
            status |= ST_AUX;
        }
        status
    }

    /// Write to port 0x64: a command for the controller itself.
    pub fn write_command(&mut self, cmd: u8) {
        self.cmd_written = true;
        match cmd {
            // Read controller RAM. Byte 0 is the command byte; the other 31
            // are scratch nobody here keeps, and read back as zero.
            0x20 => {
                let ctr = self.ctr;
                self.reply(ctr);
            }
            0x21..=0x3F => self.reply(0),
            // Write controller RAM: the byte follows at port 0x60.
            0x60..=0x7F => self.pending = Some(cmd),
            // Disable / enable the auxiliary port.
            0xA7 => self.ctr |= CTR_AUX_DISABLE,
            0xA8 => self.ctr &= !CTR_AUX_DISABLE,
            // Test the auxiliary port: 0xFF is "no device down there".
            0xA9 => self.reply(0xFF),
            // Controller self-test: 0x55 is pass, and it sets the system flag.
            0xAA => {
                self.ctr |= CTR_SYS;
                self.reply(0x55);
            }
            // Test the first PS/2 port: 0x00 is pass.
            0xAB => self.reply(0x00),
            // Disable / enable the keyboard port.
            0xAD => self.ctr |= CTR_KBD_DISABLE,
            0xAE => self.ctr &= !CTR_KBD_DISABLE,
            // Read the input port. Bit 7 clear means not inhibited.
            0xC0 => self.reply(0x00),
            // Read the output port. A20 (bit 1) is always on here.
            0xD0 => self.reply(0x02),
            // Each of these takes an argument at port 0x60.
            0xD1..=0xD4 => self.pending = Some(cmd),
            // Pulse the reset line. A real machine reboots; there is nothing
            // useful to do here, and faulting would be worse than ignoring it.
            0xFE => {}
            _ => {}
        }
    }

    /// Write to port 0x60: either the argument to a controller command, or a
    /// command for the keyboard device.
    pub fn write_data(&mut self, val: u8) {
        self.cmd_written = false;
        if let Some(cmd) = self.pending.take() {
            match cmd {
                0x60 => {
                    self.ctr = val;
                    // Interrupts may have just been enabled or masked, and a
                    // byte may already be waiting behind that decision.
                    self.refresh_irq();
                }
                0x61..=0x7F => {} // scratch RAM, not modelled
                0xD1 => {}        // output port; A20 is always on
                0xD2 => self.reply(val),     // echo into the keyboard buffer
                0xD3 => self.reply_aux(val), // echo into the aux buffer
                // Send to the auxiliary device. There is none, so nothing
                // answers and the guest's wait times out -- which is what
                // "no mouse" looks like from the other side of the port.
                0xD4 => {}
                _ => {}
            }
            return;
        }
        if let Some(cmd) = self.dev_pending.take() {
            match cmd {
                // Scancode set: an argument of 0 is a query, not a set.
                0xF0 => {
                    self.reply(ACK);
                    if val == 0 {
                        self.reply(0x02);
                    }
                }
                // Set LEDs, set typematic rate.
                _ => self.reply(ACK),
            }
            return;
        }
        match val {
            // Reset: acknowledge, then report the self-test passed.
            0xFF => {
                self.reply(ACK);
                self.reply(0xAA);
            }
            0xEE => self.reply(0xEE), // echo, and not acknowledged
            // Identify: an MF2 keyboard.
            0xF2 => {
                self.reply(ACK);
                self.reply(0xAB);
                self.reply(0x83);
            }
            0xF4 => {
                self.scanning = true;
                self.driver_attached = true;
                self.reply(ACK);
            }
            0xF5 => {
                self.scanning = false;
                self.reply(ACK);
            }
            // These take an argument, which arrives as the next write.
            0xED | 0xF0 | 0xF3 => {
                self.dev_pending = Some(val);
                self.reply(ACK);
            }
            _ => self.reply(ACK),
        }
    }
}


/// Translate an ASCII byte to its set-1 make code and whether shift is held.
///
/// Set 1 because the controller's translation bit is on, which is how a PC
/// has looked to software since the AT: the keyboard sends set 2 and the
/// 8042 hands set 1 to the CPU. Only the keys a scripted line needs are
/// here -- letters, digits, punctuation, space, tab, newline and backspace.
pub fn ascii_to_scancode(c: u8) -> Option<(u8, bool)> {
    /// The unshifted keycap of every key this table can type.
    const KEYS: &[(u8, u8)] = &[
        (b'1', 0x02), (b'2', 0x03), (b'3', 0x04), (b'4', 0x05), (b'5', 0x06),
        (b'6', 0x07), (b'7', 0x08), (b'8', 0x09), (b'9', 0x0A), (b'0', 0x0B),
        (b'-', 0x0C), (b'=', 0x0D), (0x08, 0x0E), (b'\t', 0x0F),
        (b'q', 0x10), (b'w', 0x11), (b'e', 0x12), (b'r', 0x13), (b't', 0x14),
        (b'y', 0x15), (b'u', 0x16), (b'i', 0x17), (b'o', 0x18), (b'p', 0x19),
        (b'[', 0x1A), (b']', 0x1B), (b'\n', 0x1C),
        (b'a', 0x1E), (b's', 0x1F), (b'd', 0x20), (b'f', 0x21), (b'g', 0x22),
        (b'h', 0x23), (b'j', 0x24), (b'k', 0x25), (b'l', 0x26), (b';', 0x27),
        (b'\'', 0x28), (b'`', 0x29), (b'\\', 0x2B),
        (b'z', 0x2C), (b'x', 0x2D), (b'c', 0x2E), (b'v', 0x2F), (b'b', 0x30),
        (b'n', 0x31), (b'm', 0x32), (b',', 0x33), (b'.', 0x34), (b'/', 0x35),
        (b' ', 0x39),
    ];
    /// The shifted keycap of a key already in `KEYS`, and the key it sits on.
    const SHIFTED: &[(u8, u8)] = &[
        (b'!', b'1'), (b'@', b'2'), (b'#', b'3'), (b'$', b'4'), (b'%', b'5'),
        (b'^', b'6'), (b'&', b'7'), (b'*', b'8'), (b'(', b'9'), (b')', b'0'),
        (b'_', b'-'), (b'+', b'='), (b'{', b'['), (b'}', b']'), (b':', b';'),
        (b'"', b'\''), (b'~', b'`'), (b'|', b'\\'), (b'<', b','), (b'>', b'.'),
        (b'?', b'/'),
    ];
    let key = |k: u8| KEYS.iter().find(|(ch, _)| *ch == k).map(|(_, sc)| *sc);
    if c.is_ascii_uppercase() {
        return key(c.to_ascii_lowercase()).map(|sc| (sc, true));
    }
    if let Some((_, base)) = SHIFTED.iter().find(|(ch, _)| *ch == c) {
        return key(*base).map(|sc| (sc, true));
    }
    key(c).map(|sc| (sc, false))
}

/// Expand text into the make and break codes a keyboard would send,
/// bracketing shifted characters with left shift down and up. A character
/// with no key on a US layout is skipped rather than mistyped.
pub fn scancodes_for(text: &str) -> Vec<u8> {
    const LSHIFT_MAKE: u8 = 0x2A;
    const LSHIFT_BREAK: u8 = 0xAA;
    let mut out = Vec::new();
    for c in text.bytes() {
        let (sc, shift) = match ascii_to_scancode(c) {
            Some(v) => v,
            None => continue,
        };
        if shift {
            out.push(LSHIFT_MAKE);
        }
        out.push(sc);
        out.push(sc | 0x80); // every key that goes down comes back up
        if shift {
            out.push(LSHIFT_BREAK);
        }
    }
    out
}

impl Default for Kbd {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the controller the way `i8042_probe` does and read one byte.
    fn command(kbd: &mut Kbd, cmd: u8) -> u8 {
        kbd.write_command(cmd);
        assert_eq!(kbd.read_status() & ST_OBF, ST_OBF, "command {cmd:02X} answered nothing");
        kbd.read_data()
    }

    #[test]
    fn scancode_queued_and_read() {
        let mut kbd = Kbd::new();
        kbd.push_scancode(0x1E); // 'A' make
        assert!(kbd.irq1);
        assert_eq!(kbd.read_status() & ST_OBF, ST_OBF);
        assert_eq!(kbd.read_data(), 0x1E);
        assert!(!kbd.irq1);
        assert_eq!(kbd.read_status() & ST_OBF, 0x00);
    }

    #[test]
    fn empty_read_returns_zero() {
        let mut kbd = Kbd::new();
        assert_eq!(kbd.read_data(), 0);
    }

    #[test]
    fn the_control_byte_round_trips() {
        // The whole probe hangs off this: read it with 0x20, write it back
        // with 0x60, read it again and see the change. Answering 0x20 with a
        // fixed acknowledge byte is what failed the probe with "Can't read
        // CTR" and left the machine with no console.
        let mut kbd = Kbd::new();
        let initial = command(&mut kbd, 0x20);
        assert_eq!(initial, CTR_DEFAULT);

        kbd.write_command(0x60);
        kbd.write_data(initial & !CTR_KBD_INT); // mask the keyboard interrupt
        assert_eq!(command(&mut kbd, 0x20), CTR_DEFAULT & !CTR_KBD_INT);
    }

    #[test]
    fn self_test_and_port_test_report_pass() {
        let mut kbd = Kbd::new();
        assert_eq!(command(&mut kbd, 0xAA), 0x55, "controller self-test");
        assert_eq!(command(&mut kbd, 0xAB), 0x00, "first port test");
    }

    #[test]
    fn the_auxiliary_port_reports_no_device() {
        // 0xFF is what stops Linux registering a mouse that is not there.
        let mut kbd = Kbd::new();
        assert_eq!(command(&mut kbd, 0xA9), 0xFF);

        // And a byte written through to the absent device is dropped, so the
        // guest's wait for a reply times out rather than being answered.
        kbd.write_command(0xD4);
        kbd.write_data(0xF2);
        assert_eq!(kbd.read_status() & ST_OBF, 0, "nothing answers for a mouse");
    }

    #[test]
    fn the_keyboard_answers_reset_and_identify() {
        let mut kbd = Kbd::new();
        kbd.write_data(0xFF);
        assert_eq!(kbd.read_data(), ACK);
        assert_eq!(kbd.read_data(), 0xAA, "self-test passed");

        kbd.write_data(0xF2);
        assert_eq!(kbd.read_data(), ACK);
        assert_eq!(kbd.read_data(), 0xAB, "MF2 keyboard, low byte");
        assert_eq!(kbd.read_data(), 0x83, "MF2 keyboard, high byte");
    }

    #[test]
    fn a_command_with_an_argument_consumes_the_next_write() {
        // Set-LEDs takes a byte, and that byte must not be read back as a
        // command in its own right -- 0xED 0x02 is one exchange, not two.
        let mut kbd = Kbd::new();
        kbd.write_data(0xED);
        assert_eq!(kbd.read_data(), ACK);
        kbd.write_data(0x02);
        assert_eq!(kbd.read_data(), ACK);
        assert_eq!(kbd.read_status() & ST_OBF, 0, "exactly two acknowledges");
    }

    #[test]
    fn interrupts_follow_the_control_byte() {
        // With the keyboard interrupt masked a scancode still queues, but
        // raises nothing. Linux masks it for the length of the probe.
        let mut kbd = Kbd::new();
        kbd.write_command(0x60);
        kbd.write_data(CTR_DEFAULT & !CTR_KBD_INT);
        kbd.push_scancode(0x1E);
        assert!(!kbd.irq1, "masked in the CTR");
        assert_eq!(kbd.read_status() & ST_OBF, ST_OBF, "but still waiting");

        // Unmasking with a byte already queued asserts it, which is what
        // makes the first keypress after the probe arrive.
        kbd.write_command(0x60);
        kbd.write_data(CTR_DEFAULT);
        assert!(kbd.irq1);
    }

    #[test]
    fn a_keyboard_told_to_stop_scanning_sends_nothing() {
        let mut kbd = Kbd::new();
        kbd.write_data(0xF5); // disable scanning
        assert_eq!(kbd.read_data(), ACK);
        kbd.push_scancode(0x1E);
        assert_eq!(kbd.read_status() & ST_OBF, 0, "dropped, not queued");

        kbd.write_data(0xF4); // enable scanning
        assert_eq!(kbd.read_data(), ACK);
        kbd.push_scancode(0x1E);
        assert_eq!(kbd.read_data(), 0x1E);
    }

    #[test]
    fn the_status_register_says_where_a_byte_came_from() {
        // Command 0xD3 pushes a byte into the auxiliary buffer. It must read
        // back with the AUX bit set and must not raise IRQ1, which belongs to
        // the keyboard.
        let mut kbd = Kbd::new();
        kbd.write_command(0xD3);
        kbd.write_data(0x5A);
        assert_eq!(kbd.read_status() & ST_AUX, ST_AUX);
        assert!(!kbd.irq1, "an aux byte is IRQ12's business, not IRQ1's");
        assert_eq!(kbd.read_data(), 0x5A);
        assert_eq!(kbd.read_status() & ST_AUX, 0);
    }

    #[test]
    fn text_becomes_make_and_break_codes() {
        // Every key that goes down has to come back up: a make code with no
        // break leaves the guest's driver holding the key down forever.
        assert_eq!(scancodes_for("a"), vec![0x1E, 0x9E]);
        assert_eq!(scancodes_for("A"), vec![0x2A, 0x1E, 0x9E, 0xAA]);
        assert_eq!(scancodes_for("\n"), vec![0x1C, 0x9C]);
        // A character with no key on a US layout is skipped, not mistyped.
        assert_eq!(scancodes_for("\u{20AC}"), Vec::<u8>::new());
    }

    #[test]
    fn shifted_punctuation_sits_on_the_unshifted_key() {
        assert_eq!(ascii_to_scancode(b'?'), Some((0x35, true)), "shift and '/'");
        assert_eq!(ascii_to_scancode(b'/'), Some((0x35, false)));
    }

    #[test]
    fn a_driver_enabling_scanning_is_recorded() {
        // Scripted input waits for this: it is atkbd attaching.
        let mut kbd = Kbd::new();
        assert!(!kbd.driver_attached);
        kbd.write_data(0xF4);
        assert!(kbd.driver_attached);
    }

    #[test]
    fn the_status_register_reports_the_last_write_port() {
        let mut kbd = Kbd::new();
        kbd.write_command(0xAA);
        assert_eq!(kbd.read_status() & ST_CMD, ST_CMD, "last write was 0x64");
        kbd.write_data(0xF4);
        assert_eq!(kbd.read_status() & ST_CMD, 0, "last write was 0x60");
    }
}
