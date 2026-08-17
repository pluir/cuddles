//! 8237 DMA controller.
//!
//! Four DMA channels (0-3). Each channel has a base address, a word count,
//! and a page register (which supplies the high address bits, giving a 20-bit
//! physical address). The controller can transfer data between memory and an
//! I/O device, or between two memory regions.
//!
//! I/O ports:
//!   0x00-0x07 - channel 0-3 address/count (byte and word)
//!   0x08-0x0F - command/status registers
//!   0x81-0x8F - page registers (high address bits per channel)
//!
//! This implementation keeps the address/count registers and page registers
//! so a real OS can program a transfer; the actual bus transfer is simulated
//! by the CPU's `dma_transfer` helper.

/// A single DMA channel.
#[derive(Clone, Copy, Debug)]
pub struct DmaChannel {
    /// Base address (low 16 bits).
    pub base: u16,
    /// Word count (number of 16-bit transfers minus 1).
    pub count: u16,
    /// Page register (high address bits).
    pub page: u8,
    /// Whether the channel is enabled.
    pub enabled: bool,
    /// Auto-initialize (reload base/count after transfer).
    pub auto_init: bool,
    /// Transfer direction: true = write (memory -> I/O), false = read.
    pub write: bool,
}

impl Default for DmaChannel {
    fn default() -> Self {
        DmaChannel { base: 0, count: 0, page: 0, enabled: false, auto_init: false, write: false }
    }
}

/// The 8237 DMA controller.
pub struct Dma {
    pub channels: [DmaChannel; 4],
    /// Command register (port 0x08).
    pub command: u8,
    /// Status register (port 0x08 read).
    pub status: u8,
    /// Mask register (port 0x0A).
    pub mask: u8,
    /// Currently selected channel for byte/word programming.
    pub selected: u8,
    /// Whether the next access to a channel port is the high byte.
    pub high_byte: bool,
}

impl Dma {
    pub fn new() -> Self {
        Dma {
            channels: [DmaChannel::default(); 4],
            command: 0,
            status: 0,
            mask: 0,
            selected: 0,
            high_byte: false,
        }
    }

    /// Compute the 20-bit physical address for a channel.
    pub fn address(&self, ch: usize) -> usize {
        let ch = ch & 3;
        let c = &self.channels[ch];
        ((c.page as usize) << 16) | c.base as usize
    }

    /// Write to a channel address/count port (0x00-0x07).
    pub fn write_channel(&mut self, port: u8, val: u8) {
        let ch = (port >> 1) & 3;
        let is_count = port & 1 == 1;
        let c = &mut self.channels[ch as usize];
        if self.high_byte {
            if is_count {
                c.count = (c.count & 0x00FF) | ((val as u16) << 8);
            } else {
                c.base = (c.base & 0x00FF) | ((val as u16) << 8);
            }
            self.high_byte = false;
        } else {
            if is_count {
                c.count = (c.count & 0xFF00) | val as u16;
            } else {
                c.base = (c.base & 0xFF00) | val as u16;
            }
            self.high_byte = true;
        }
    }

    /// Write to a page register (0x81-0x8F).
    pub fn write_page(&mut self, port: u8, val: u8) {
        let ch = match port {
            0x87 => 0,
            0x83 => 1,
            0x81 => 2,
            0x82 => 3,
            _ => return,
        };
        self.channels[ch].page = val;
    }

    /// Write to a command/mask register.
    pub fn write_command(&mut self, port: u8, val: u8) {
        match port {
            0x08 => self.command = val,
            0x0A => {
                // Single mask register: bits 1-0 = channel, bit 2 = mask.
                let ch = (val & 3) as usize;
                self.channels[ch].enabled = val & 4 == 0;
                self.mask = val;
            }
            _ => {}
        }
    }

    /// Read the status register.
    pub fn read_status(&self) -> u8 {
        self.status
    }

    /// Perform a memory-to-memory transfer on a channel. Returns the number
    /// of bytes moved. `mem` is the emulated memory.
    pub fn transfer(&mut self, mem: &mut crate::memory::Memory, ch: usize) -> usize {
        let ch = ch & 3;
        if !self.channels[ch].enabled {
            return 0;
        }
        let base = self.address(ch);
        // DMA transfers happen in 16-bit words; count is words - 1.
        let words = (self.channels[ch].count as usize) + 1;
        // Simulate a memory-to-memory copy within the same region (a common
        // use: channel 0 as the source, channel 1 as the destination). For
        // simplicity we copy `words * 2` bytes starting at `base`.
        let n = words * 2;
        let src = base;
        let dst = base + n; // simple contiguous copy
        for i in 0..n {
            let b = mem.read_u8(src + i);
            mem.write_u8(dst + i, b);
        }
        self.status |= 1 << ch;
        if self.channels[ch].auto_init {
            // reload handled by keeping base/count
        } else {
            self.channels[ch].enabled = false;
        }
        n
    }
}

impl Default for Dma {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Memory;

    #[test]
    fn channel_address_combines_page_and_base() {
        let mut dma = Dma::new();
        dma.channels[2].base = 0x1234;
        dma.channels[2].page = 0x10;
        assert_eq!(dma.address(2), 0x101234);
    }

    #[test]
    fn program_channel_via_ports() {
        let mut dma = Dma::new();
        // Channel 0 base port 0x00: write LSB then MSB.
        dma.write_channel(0x00, 0x34);
        dma.write_channel(0x00, 0x12);
        assert_eq!(dma.channels[0].base, 0x1234);
        // Count port 0x01.
        dma.write_channel(0x01, 0xFF);
        dma.write_channel(0x01, 0x00);
        assert_eq!(dma.channels[0].count, 0x00FF);
    }

    #[test]
    fn page_register_writes() {
        let mut dma = Dma::new();
        dma.write_page(0x81, 0xAB);
        assert_eq!(dma.channels[2].page, 0xAB);
    }

    #[test]
    fn transfer_copies_memory() {
        let mut dma = Dma::new();
        let mut mem = Memory::new();
        mem.write_u16(0x1000, 0xCAFE);
        mem.write_u16(0x1002, 0xBEEF);
        dma.channels[0].base = 0x1000;
        dma.channels[0].count = 1; // 2 words
        dma.channels[0].enabled = true;
        let n = dma.transfer(&mut mem, 0);
        assert_eq!(n, 4);
        assert_eq!(mem.read_u16(0x1004), 0xCAFE);
        assert_eq!(mem.read_u16(0x1006), 0xBEEF);
    }
}
