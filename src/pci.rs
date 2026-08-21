//! PCI bus: configuration space through the 0xCF8/0xCFC port pair.
//!
//! This is configuration mechanism #1, the one every PC has used since the
//! 486: write a *address* to 0xCF8 naming bus/device/function/register, then
//! read or write the value at 0xCFC. The address register's top bit enables
//! the cycle; without it the ports are inert, which is how firmware probes
//! for the mechanism in the first place.
//!
//! Three details are what make an enumeration terminate rather than wander:
//!
//! 1. **An absent device reads back all ones.** `pci_scan_slot` decides a
//!    slot is empty by reading 0xFFFFFFFF from the vendor/device register, so
//!    a bus that returns zeros for empty slots invents a device 0x0000:0x0000
//!    in every one of the 32 slots.
//! 2. **BARs report their size by writing all ones.** The kernel writes
//!    0xFFFFFFFF to a base address register and reads back a mask whose low
//!    clear bits give the region's size, then writes the real address. A BAR
//!    that stores what it is given describes a region of no size, and Linux
//!    assigns it nothing.
//! 3. **The low bits of a BAR are type, not address.** Bit 0 says I/O rather
//!    than memory, and it is read-only; folding it into the stored address
//!    moves the region by a byte the first time the kernel writes one.
//!
//! Only what a guest needs to find the devices this machine has is here: one
//! bus, no bridges, no capabilities list, no MSI.

/// A single function's 256-byte configuration space, plus the writable mask
/// that keeps read-only fields read-only.
#[derive(Clone)]
pub struct Function {
    /// The configuration space as the guest sees it.
    pub config: [u8; 256],
    /// Size of each base address region, indexed by BAR number. Zero means
    /// the BAR is unimplemented and reads back as zero.
    pub bar_size: [u32; 6],
}

impl Function {
    fn new() -> Self {
        Function { config: [0; 256], bar_size: [0; 6] }
    }

    fn put16(&mut self, off: usize, val: u16) {
        self.config[off..off + 2].copy_from_slice(&val.to_le_bytes());
    }

    fn put32(&mut self, off: usize, val: u32) {
        self.config[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }

    pub fn read32(&self, off: usize) -> u32 {
        u32::from_le_bytes(self.config[off..off + 4].try_into().unwrap())
    }

    /// The current base address of an I/O BAR, with the type bits masked off.
    pub fn io_base(&self, bar: usize) -> u16 {
        (self.read32(0x10 + bar * 4) & !0x3) as u16
    }
}

/// The PCI bus: the address latch and the devices on it.
pub struct Pci {
    /// The value last written to 0xCF8.
    pub address: u32,
    /// Slot -> function 0. One bus, function 0 only: enough for a machine
    /// whose devices are all single-function, and an absent slot is `None`
    /// so it can read back as all ones.
    pub slots: Vec<Option<Function>>,
}

/// Where the virtio block device's I/O region is put before the guest moves
/// it. Any free port range does; the kernel reassigns BARs during
/// enumeration and the device follows whatever it writes.
pub const VIRTIO_BLK_IO_BASE: u16 = 0xC000;

/// The slot the virtio block device sits in.
pub const VIRTIO_BLK_SLOT: usize = 1;

/// The interrupt line the virtio device drives.
pub const VIRTIO_IRQ: u8 = 11;

impl Pci {
    pub fn new() -> Self {
        let mut slots = vec![None; 32];

        // 00:00.0 -- the host bridge. A bus with no device 0 is not a bus
        // Linux will scan; it reads the bridge before anything else.
        let mut bridge = Function::new();
        bridge.put16(0x00, 0x8086); // Intel
        bridge.put16(0x02, 0x1237); // 440FX
        bridge.put16(0x04, 0x0006); // command: memory + bus master
        bridge.put16(0x06, 0x0200); // status: has capabilities? no -- devsel
        bridge.config[0x08] = 0x02; // revision
        bridge.config[0x0A] = 0x00; // subclass: host bridge
        bridge.config[0x0B] = 0x06; // class: bridge
        bridge.config[0x0E] = 0x00; // header type 0, single function
        slots[0] = Some(bridge);

        // 00:01.0 -- a transitional virtio block device. Device ID 0x1000 +
        // 1 is the legacy block device; the subsystem ID repeats it, which
        // is what a transitional driver actually keys off.
        let mut blk = Function::new();
        blk.put16(0x00, 0x1AF4); // Red Hat / virtio
        blk.put16(0x02, 0x1001); // legacy block device
        blk.put16(0x04, 0x0001); // command: I/O space enabled
        blk.put16(0x06, 0x0000);
        blk.config[0x08] = 0x00; // revision 0 marks it legacy, not 1.0
        blk.config[0x09] = 0x00; // programming interface
        blk.config[0x0A] = 0x00; // subclass
        blk.config[0x0B] = 0x01; // class: mass storage
        blk.config[0x0E] = 0x00; // header type 0
        blk.put16(0x2C, 0x1AF4); // subsystem vendor
        blk.put16(0x2E, 0x0002); // subsystem device: block
        blk.config[0x3C] = VIRTIO_IRQ; // interrupt line
        blk.config[0x3D] = 0x01; // interrupt pin A
        // BAR0: 64 bytes of I/O space. Bit 0 set marks it I/O.
        blk.put32(0x10, VIRTIO_BLK_IO_BASE as u32 | 1);
        blk.bar_size[0] = 64;
        slots[VIRTIO_BLK_SLOT] = Some(blk);

        Pci { address: 0, slots }
    }

    /// Decode the address latch into (slot, register offset), or `None` when
    /// the enable bit is clear or the cycle names a bus or function this
    /// machine does not have.
    fn target(&self) -> Option<(usize, usize)> {
        if self.address & 0x8000_0000 == 0 {
            return None;
        }
        let bus = (self.address >> 16) & 0xFF;
        let slot = ((self.address >> 11) & 0x1F) as usize;
        let func = (self.address >> 8) & 0x07;
        if bus != 0 || func != 0 {
            return None;
        }
        Some((slot, (self.address & 0xFC) as usize))
    }

    /// Read from the data port. `size` is 1, 2 or 4 bytes and `lane` is the
    /// byte offset within the aligned register, taken from the low bits of
    /// the port address rather than from the latch.
    pub fn read_data(&self, lane: usize, size: usize) -> u32 {
        let Some((slot, reg)) = self.target() else {
            return 0xFFFF_FFFF;
        };
        let Some(func) = self.slots.get(slot).and_then(|s| s.as_ref()) else {
            // An empty slot reads as all ones. This is what ends a scan.
            return 0xFFFF_FFFF;
        };
        let off = reg + lane;
        let mut value = 0u32;
        for i in 0..size {
            let byte = func.config.get(off + i).copied().unwrap_or(0xFF);
            value |= (byte as u32) << (i * 8);
        }
        value
    }

    /// Write to the data port.
    pub fn write_data(&mut self, lane: usize, size: usize, value: u32) {
        let Some((slot, reg)) = self.target() else { return };
        let Some(func) = self.slots.get_mut(slot).and_then(|s| s.as_mut()) else { return };

        // Base address registers are not plain storage: a write of all ones
        // is a *size query*, and the type bits in the low end are read-only.
        if (0x10..0x28).contains(&reg) && size == 4 && lane == 0 {
            let bar = (reg - 0x10) / 4;
            let size_bytes = func.bar_size[bar];
            if size_bytes == 0 {
                return; // unimplemented BAR: stays zero
            }
            let is_io = func.config[0x10 + bar * 4] & 1 != 0;
            let type_bits = if is_io { 1 } else { 0 };
            let mask = !(size_bytes - 1);
            let stored = if value == 0xFFFF_FFFF {
                // Report the size: the low bits below the region size read
                // back clear, with the type bits preserved.
                mask | type_bits
            } else {
                (value & mask) | type_bits
            };
            func.put32(reg, stored);
            return;
        }

        // Everything else is byte-writable except the identity registers,
        // which a guest must not be able to rename.
        for i in 0..size {
            let off = reg + lane + i;
            if off >= 256 || Self::read_only(off) {
                continue;
            }
            func.config[off] = ((value >> (i * 8)) & 0xFF) as u8;
        }
    }

    /// Registers a guest may not change: the identity of the device and the
    /// fields describing its class and header layout.
    fn read_only(off: usize) -> bool {
        matches!(off, 0x00..=0x03 | 0x08..=0x0B | 0x0E | 0x2C..=0x2F | 0x3D)
    }

    /// The virtio block device's current I/O base, or `None` when the guest
    /// has not enabled I/O decoding for it.
    pub fn virtio_io_base(&self) -> Option<u16> {
        let func = self.slots.get(VIRTIO_BLK_SLOT)?.as_ref()?;
        if func.config[0x04] & 0x01 == 0 {
            return None; // I/O space decoding disabled
        }
        let base = func.io_base(0);
        if base == 0 { None } else { Some(base) }
    }
}

impl Default for Pci {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(slot: u32, reg: u32) -> u32 {
        0x8000_0000 | (slot << 11) | (reg & 0xFC)
    }

    #[test]
    fn the_host_bridge_answers_at_slot_zero() {
        let mut pci = Pci::new();
        pci.address = addr(0, 0x00);
        assert_eq!(pci.read_data(0, 4), 0x1237_8086, "vendor and device");
    }

    #[test]
    fn an_empty_slot_reads_as_all_ones() {
        // This is what ends a bus scan. Zeros here invent a device in every
        // one of the 32 slots.
        let mut pci = Pci::new();
        pci.address = addr(5, 0x00);
        assert_eq!(pci.read_data(0, 4), 0xFFFF_FFFF);
    }

    #[test]
    fn the_cycle_needs_the_enable_bit() {
        let mut pci = Pci::new();
        pci.address = addr(0, 0x00) & !0x8000_0000;
        assert_eq!(pci.read_data(0, 4), 0xFFFF_FFFF, "inert without bit 31");
    }

    #[test]
    fn the_block_device_identifies_as_transitional_virtio() {
        let mut pci = Pci::new();
        pci.address = addr(VIRTIO_BLK_SLOT as u32, 0x00);
        assert_eq!(pci.read_data(0, 4), 0x1001_1AF4);
        pci.address = addr(VIRTIO_BLK_SLOT as u32, 0x2C);
        assert_eq!(pci.read_data(0, 4), 0x0002_1AF4, "subsystem says block");
        pci.address = addr(VIRTIO_BLK_SLOT as u32, 0x08);
        assert_eq!(pci.read_data(0, 4) >> 24, 0x01, "class: mass storage");
        assert_eq!(pci.read_data(0, 4) & 0xFF, 0x00, "revision 0 is legacy");
    }

    #[test]
    fn a_bar_reports_its_size_then_takes_an_address() {
        let mut pci = Pci::new();
        pci.address = addr(VIRTIO_BLK_SLOT as u32, 0x10);
        // Writing all ones asks how big the region is: the low six bits read
        // back clear for a 64-byte region, and bit 0 stays set for I/O.
        pci.write_data(0, 4, 0xFFFF_FFFF);
        let probed = pci.read_data(0, 4);
        assert_eq!(probed & 1, 1, "still marked I/O");
        assert_eq!(probed & !0x3, 0xFFFF_FFC0, "64 bytes");

        // Then the kernel writes where it wants it.
        pci.write_data(0, 4, 0xD000);
        assert_eq!(pci.read_data(0, 4), 0xD001, "address with the I/O bit");
        assert_eq!(pci.virtio_io_base(), Some(0xD000));
    }

    #[test]
    fn the_identity_registers_are_read_only() {
        let mut pci = Pci::new();
        pci.address = addr(VIRTIO_BLK_SLOT as u32, 0x00);
        pci.write_data(0, 4, 0xDEAD_BEEF);
        assert_eq!(pci.read_data(0, 4), 0x1001_1AF4, "device cannot be renamed");
    }

    #[test]
    fn io_decoding_can_be_turned_off() {
        // The kernel clears the command register while it sizes BARs. A
        // device that keeps decoding through that answers ports the kernel
        // believes belong to nobody.
        let mut pci = Pci::new();
        assert!(pci.virtio_io_base().is_some());
        pci.address = addr(VIRTIO_BLK_SLOT as u32, 0x04);
        pci.write_data(0, 2, 0x0000);
        assert_eq!(pci.virtio_io_base(), None);
    }

    #[test]
    fn byte_and_word_reads_take_their_lane_from_the_port() {
        let mut pci = Pci::new();
        pci.address = addr(VIRTIO_BLK_SLOT as u32, 0x00);
        assert_eq!(pci.read_data(2, 2), 0x1001, "device id is the high half");
        assert_eq!(pci.read_data(0, 2), 0x1AF4, "vendor id is the low half");
        assert_eq!(pci.read_data(0, 1), 0xF4);
    }
}
