//! IDE/ATA disk controller.
//!
//! A minimal ATA (PIO mode) controller on the primary channel. The CPU talks
//! to it through 16-bit registers at I/O ports 0x1F0-0x1F7 (data + command
//! block) and 0x3F6 (device control). It supports the classic "read sectors"
//! (0x20) and "write sectors" (0x30) commands in LBA28 addressing.
//!
//! I/O ports:
//!   0x1F0 - data register (16-bit)
//!   0x1F1 - error register
//!   0x1F2 - sector count
//!   0x1F3 - LBA low
//!   0x1F4 - LBA mid
//!   0x1F5 - LBA high
//!   0x1F6 - drive/head register (LBA bits 24-27 + drive select)
//!   0x1F7 - status (read) / command (write)
//!   0x3F6 - device control / alternate status

/// Sector size in bytes (standard ATA).
pub const SECTOR_SIZE: usize = 512;

/// The IDE/ATA disk controller.
pub struct Ide {
    /// The disk image (a sequence of 512-byte sectors).
    pub disk: Vec<u8>,
    /// True when a disk image is loaded.
    pub present: bool,
    // Command block registers.
    pub error: u8,
    pub sector_count: u8,
    pub lba_low: u8,
    pub lba_mid: u8,
    pub lba_high: u8,
    pub drive_head: u8,
    pub status: u8,
    // Data buffer for the current PIO transfer.
    pub data_buffer: Vec<u8>,
    pub data_index: usize,
    /// Pending command (0x20 read, 0x30 write).
    pub pending_command: u8,
    /// True while a transfer is in progress.
    pub busy: bool,
}

impl Ide {
    pub fn new() -> Self {
        Ide {
            disk: Vec::new(),
            present: false,
            error: 0,
            sector_count: 0,
            lba_low: 0,
            lba_mid: 0,
            lba_high: 0,
            drive_head: 0,
            status: 0x40, // DRDY (drive ready)
            data_buffer: Vec::new(),
            data_index: 0,
            pending_command: 0,
            busy: false,
        }
    }

    /// Load a disk image.
    pub fn load_disk(&mut self, image: Vec<u8>) {
        self.disk = image;
        self.present = true;
        self.status = 0x40;
    }

    /// Compute the 28-bit LBA from the command block registers.
    pub fn lba(&self) -> u32 {
        (self.lba_low as u32)
            | ((self.lba_mid as u32) << 8)
            | ((self.lba_high as u32) << 16)
            | (((self.drive_head & 0x0F) as u32) << 24)
    }

    /// Write to a command block register (ports 0x1F1-0x1F6).
    pub fn write_reg(&mut self, port: u16, val: u8) {
        match port {
            0x1F1 => self.error = val,
            0x1F2 => self.sector_count = val,
            0x1F3 => self.lba_low = val,
            0x1F4 => self.lba_mid = val,
            0x1F5 => self.lba_high = val,
            0x1F6 => self.drive_head = val,
            _ => {}
        }
    }

    /// Write to the command register (port 0x1F7). Starts a transfer.
    pub fn write_command(&mut self, cmd: u8) {
        self.pending_command = cmd;
        self.busy = true;
        self.status = 0x80; // BSY
        let lba = self.lba();
        let count = self.sector_count.max(1) as usize;
        let start = (lba as usize) * SECTOR_SIZE;
        match cmd {
            0x20 | 0x21 => {
                // Read sectors.
                self.data_buffer.clear();
                if self.present && start + count * SECTOR_SIZE <= self.disk.len() {
                    self.data_buffer.extend_from_slice(&self.disk[start..start + count * SECTOR_SIZE]);
                    self.error = 0;
                } else {
                    self.error = 0x04; // sector not found
                }
            }
            0x30 | 0x31 => {
                // Write sectors: expect data to be written to the data port
                // first. Buffer is sized to receive it.
                self.data_buffer = vec![0; count * SECTOR_SIZE];
                self.data_index = 0;
                self.error = 0;
            }
            _ => {
                self.error = 0x04;
            }
        }
        self.data_index = 0;
        self.busy = false;
        self.status = 0x40; // DRDY, transfer ready
    }

    /// Read a 16-bit word from the data register (port 0x1F0).
    pub fn read_data(&mut self) -> u16 {
        if self.data_index + 1 < self.data_buffer.len() {
            let lo = self.data_buffer[self.data_index] as u16;
            let hi = self.data_buffer[self.data_index + 1] as u16;
            self.data_index += 2;
            lo | (hi << 8)
        } else if self.data_index < self.data_buffer.len() {
            let lo = self.data_buffer[self.data_index] as u16;
            self.data_index += 1;
            lo
        } else {
            0
        }
    }

    /// Write a 16-bit word to the data register (port 0x1F0).
    pub fn write_data(&mut self, val: u16) {
        if self.data_index + 1 < self.data_buffer.len() {
            self.data_buffer[self.data_index] = (val & 0xFF) as u8;
            self.data_buffer[self.data_index + 1] = (val >> 8) as u8;
            self.data_index += 2;
        } else if self.data_index < self.data_buffer.len() {
            self.data_buffer[self.data_index] = (val & 0xFF) as u8;
            self.data_index += 1;
        }
    }

    /// Read the status register (port 0x1F7).
    pub fn read_status(&self) -> u8 {
        self.status
    }

    /// Finalize a pending write command: flush the buffered data to the disk.
    pub fn flush_write(&mut self) {
        if self.pending_command == 0x30 || self.pending_command == 0x31 {
            let lba = self.lba();
            let start = (lba as usize) * SECTOR_SIZE;
            if self.present && start + self.data_buffer.len() <= self.disk.len() {
                self.disk[start..start + self.data_buffer.len()]
                    .copy_from_slice(&self.data_buffer);
            }
            self.pending_command = 0;
        }
    }
}

impl Default for Ide {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lba_computed_from_registers() {
        let mut ide = Ide::new();
        ide.lba_low = 0x34;
        ide.lba_mid = 0x12;
        ide.lba_high = 0x00;
        ide.drive_head = 0xE0 | 0x00; // LBA mode, drive 0, LBA bits 24-27 = 0
        assert_eq!(ide.lba(), 0x1234);
    }

    #[test]
    fn read_sectors() {
        let mut ide = Ide::new();
        let mut disk = vec![0u8; SECTOR_SIZE * 2];
        disk[0..4].copy_from_slice(b"DATA");
        ide.load_disk(disk);
        // LBA 0, count 1.
        ide.sector_count = 1;
        ide.lba_low = 0;
        ide.lba_mid = 0;
        ide.lba_high = 0;
        ide.drive_head = 0xE0;
        ide.write_command(0x20);
        assert_eq!(ide.read_data(), b'D' as u16 | ((b'A' as u16) << 8));
        assert_eq!(ide.read_data(), b'T' as u16 | ((b'A' as u16) << 8));
    }

    #[test]
    fn write_sectors() {
        let mut ide = Ide::new();
        let disk = vec![0u8; SECTOR_SIZE * 2];
        ide.load_disk(disk);
        ide.sector_count = 1;
        ide.lba_low = 0;
        ide.lba_mid = 0;
        ide.lba_high = 0;
        ide.drive_head = 0xE0;
        ide.write_command(0x30);
        ide.write_data(0x4241); // "AB"
        ide.write_data(0x4443); // "CD"
        ide.flush_write();
        assert_eq!(&ide.disk[0..4], b"ABCD");
    }

    #[test]
    fn read_missing_sector_sets_error() {
        let mut ide = Ide::new();
        // No disk loaded.
        ide.sector_count = 1;
        ide.lba_low = 0;
        ide.write_command(0x20);
        assert_eq!(ide.error, 0x04);
    }
}
