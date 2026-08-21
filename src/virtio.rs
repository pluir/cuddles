//! A transitional virtio block device, driven through legacy port I/O.
//!
//! "Legacy" here is the pre-1.0 interface: a small window of registers in an
//! I/O BAR rather than the capability structures virtio 1.0 puts in memory
//! space. It is much less work and Linux still speaks it, because the driver
//! is *transitional* -- a PCI revision of 0 with device ID 0x1001 is what
//! selects this path rather than the modern one.
//!
//! ## The ring
//!
//! One split virtqueue lives in guest memory at `queue_pfn * 4096`, in three
//! parts laid out back to back:
//!
//! ```text
//!   descriptors   16 bytes each   addr, len, flags, next
//!   available      4 + 2*size + 2  what the driver has handed us
//!   (padding to the next 4096 boundary)
//!   used           4 + 8*size + 2  what we hand back
//! ```
//!
//! The padding is not optional and not cosmetic: the legacy layout aligns
//! the used ring to a page, so a device that packs it straight after the
//! available ring writes completions into whatever the driver put there
//! instead.
//!
//! ## A request
//!
//! Each request is a chain of descriptors: a 16-byte header the device
//! reads, some data buffers, and a single status byte the device writes.
//! The direction of the data is the header's business, not the descriptor
//! flags' -- though the flags have to agree, and a driver that marks a read
//! buffer device-readable has told us to write into memory it owns.
//!
//! The device consumes the chain, fills in the used ring, and raises its
//! interrupt. `last_avail` is what makes that resumable: it is the point the
//! device has reached in the driver's ring, and it only ever moves forward.

use crate::memory::Memory;

/// Descriptor flags.
const VRING_DESC_F_NEXT: u16 = 1;
const VRING_DESC_F_WRITE: u16 = 2;

/// Request types in the header.
const VIRTIO_BLK_T_IN: u32 = 0;
const VIRTIO_BLK_T_OUT: u32 = 1;
const VIRTIO_BLK_T_FLUSH: u32 = 4;
const VIRTIO_BLK_T_GET_ID: u32 = 8;

/// Status byte the device writes at the end of a chain.
const VIRTIO_BLK_S_OK: u8 = 0;
const VIRTIO_BLK_S_IOERR: u8 = 1;
const VIRTIO_BLK_S_UNSUPP: u8 = 2;

/// How many descriptors the one queue has. A power of two, as the ring
/// arithmetic assumes.
pub const QUEUE_SIZE: u16 = 128;

/// The alignment the legacy layout puts between the available and used rings.
const QUEUE_ALIGN: u64 = 4096;

/// Sector size, fixed by the virtio block specification regardless of what
/// the backing image thinks a block is.
pub const SECTOR: u64 = 512;

/// Device status bits the driver walks through while bringing us up.
pub const STATUS_DRIVER_OK: u8 = 4;

pub struct VirtioBlk {
    /// The backing image, a whole number of sectors.
    pub disk: Vec<u8>,
    /// Features the driver accepted. Nothing is offered, so this stays zero
    /// on any driver that follows the protocol.
    pub guest_features: u32,
    /// Device status, written by the driver as it initialises.
    pub status: u8,
    /// Which queue the register window addresses. There is only one.
    pub queue_sel: u16,
    /// Page frame of the ring, or zero when the driver has not set one.
    pub queue_pfn: u32,
    /// Interrupt status. Reading it clears it, which is how the driver
    /// acknowledges.
    pub isr: u8,
    /// Set when the device wants its interrupt line asserted.
    pub irq: bool,
    /// How far into the driver's available ring the device has got.
    last_avail: u16,
}

impl VirtioBlk {
    pub fn new() -> Self {
        VirtioBlk {
            disk: Vec::new(),
            guest_features: 0,
            status: 0,
            queue_sel: 0,
            queue_pfn: 0,
            isr: 0,
            irq: false,
            last_avail: 0,
        }
    }

    /// Attach a backing image. The capacity the driver reads is derived from
    /// it, so this must happen before the guest looks.
    pub fn attach(&mut self, image: Vec<u8>) {
        self.disk = image;
    }

    pub fn present(&self) -> bool {
        !self.disk.is_empty()
    }

    /// Capacity in 512-byte sectors, which is the unit the config space uses.
    fn capacity(&self) -> u64 {
        self.disk.len() as u64 / SECTOR
    }

    /// Read from the register window at `offset` within the I/O BAR.
    pub fn read(&mut self, offset: u16, size: usize) -> u32 {
        match offset {
            0x00 => 0, // device features: none offered
            0x04 => self.guest_features,
            0x08 => self.queue_pfn,
            0x0C => QUEUE_SIZE as u32,
            0x0E => self.queue_sel as u32,
            0x12 => self.status as u32,
            0x13 => {
                // Read-to-clear, and it drops the interrupt line with it.
                let isr = self.isr;
                self.isr = 0;
                self.irq = false;
                isr as u32
            }
            // Device configuration: capacity first, in sectors.
            0x14..=0x1B => {
                let capacity = self.capacity();
                let shift = (offset - 0x14) * 8;
                let value = capacity >> shift;
                match size {
                    1 => (value & 0xFF) as u32,
                    2 => (value & 0xFFFF) as u32,
                    _ => value as u32,
                }
            }
            _ => 0,
        }
    }

    /// Write to the register window. A write to the notify register is what
    /// makes the device do work, so it needs the guest's memory.
    pub fn write(&mut self, offset: u16, value: u32, mem: &mut Memory) {
        match offset {
            0x04 => self.guest_features = value,
            0x08 => {
                self.queue_pfn = value;
                // A driver resetting the ring restarts the device's position
                // in it; keeping the old index would skip the driver's first
                // requests and hang the queue at the first notify.
                self.last_avail = 0;
            }
            0x0E => self.queue_sel = value as u16,
            0x10 => self.notify(mem),
            0x12 => {
                self.status = value as u8;
                if value == 0 {
                    // Reset.
                    self.queue_pfn = 0;
                    self.last_avail = 0;
                    self.isr = 0;
                    self.irq = false;
                }
            }
            _ => {}
        }
    }

    /// The three ring addresses derived from the page frame.
    fn ring(&self) -> (u64, u64, u64) {
        let desc = self.queue_pfn as u64 * 4096;
        let avail = desc + 16 * QUEUE_SIZE as u64;
        let used_unaligned = avail + 4 + 2 * QUEUE_SIZE as u64 + 2;
        let used = used_unaligned.div_ceil(QUEUE_ALIGN) * QUEUE_ALIGN;
        (desc, avail, used)
    }

    /// Consume everything the driver has made available.
    fn notify(&mut self, mem: &mut Memory) {
        if self.queue_pfn == 0 {
            return;
        }
        let (desc, avail, used) = self.ring();
        let avail_idx = mem.read_u16(avail as usize + 2);
        let mut worked = false;
        // `wrapping_sub` because both indices are free-running u16 counters:
        // they wrap at 65536 rather than at the queue size, and comparing
        // them with `<` stalls the queue for good on the wrap.
        while avail_idx.wrapping_sub(self.last_avail) != 0 {
            let slot = (self.last_avail % QUEUE_SIZE) as usize;
            let head = mem.read_u16(avail as usize + 4 + slot * 2);
            let written = self.request(mem, desc, head);
            let used_slot = (mem.read_u16(used as usize + 2) % QUEUE_SIZE) as usize;
            let entry = used as usize + 4 + used_slot * 8;
            mem.write_u32(entry, head as u32);
            mem.write_u32(entry + 4, written);
            let used_idx = mem.read_u16(used as usize + 2);
            mem.write_u16(used as usize + 2, used_idx.wrapping_add(1));
            self.last_avail = self.last_avail.wrapping_add(1);
            worked = true;
        }
        if worked {
            self.isr |= 1;
            self.irq = true;
        }
    }

    /// Walk one descriptor chain and carry out the request it describes.
    /// Returns the number of bytes written into the guest's buffers.
    fn request(&mut self, mem: &mut Memory, desc: u64, head: u16) -> u32 {
        // Collect the chain first: the header says what to do, and it is the
        // first descriptor, so nothing can be decided until it is read.
        let mut chain = Vec::new();
        let mut index = head;
        for _ in 0..QUEUE_SIZE {
            let entry = desc as usize + index as usize * 16;
            let addr = mem.read_u64(entry);
            let len = mem.read_u32(entry + 8);
            let flags = mem.read_u16(entry + 12);
            let next = mem.read_u16(entry + 14);
            chain.push((addr, len, flags));
            if flags & VRING_DESC_F_NEXT == 0 {
                break;
            }
            index = next;
        }
        if chain.len() < 2 {
            return 0; // a header and a status byte are the minimum
        }

        let (hdr_addr, _, _) = chain[0];
        let req_type = mem.read_u32(hdr_addr as usize);
        let sector = mem.read_u64(hdr_addr as usize + 8);

        // The last descriptor is the status byte; everything between it and
        // the header is data.
        let status_at = chain[chain.len() - 1].0;
        let data = &chain[1..chain.len() - 1];

        let mut written = 0u32;
        let status = match req_type {
            VIRTIO_BLK_T_IN => {
                let mut offset = sector * SECTOR;
                let mut ok = true;
                for &(addr, len, flags) in data {
                    if flags & VRING_DESC_F_WRITE == 0 {
                        ok = false;
                        break;
                    }
                    for i in 0..len as u64 {
                        let byte = self.disk.get((offset + i) as usize).copied();
                        match byte {
                            Some(b) => mem.write_u8((addr + i) as usize, b),
                            None => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    offset += len as u64;
                    written += len;
                }
                if ok { VIRTIO_BLK_S_OK } else { VIRTIO_BLK_S_IOERR }
            }
            VIRTIO_BLK_T_OUT => {
                let mut offset = sector * SECTOR;
                let mut ok = true;
                for &(addr, len, _) in data {
                    for i in 0..len as u64 {
                        let at = (offset + i) as usize;
                        if at >= self.disk.len() {
                            ok = false;
                            break;
                        }
                        self.disk[at] = mem.read_u8((addr + i) as usize);
                    }
                    offset += len as u64;
                }
                if ok { VIRTIO_BLK_S_OK } else { VIRTIO_BLK_S_IOERR }
            }
            VIRTIO_BLK_T_FLUSH => VIRTIO_BLK_S_OK, // the image is memory
            VIRTIO_BLK_T_GET_ID => {
                // A 20-byte serial. Blank is legal and says "no id".
                for &(addr, len, _) in data {
                    for i in 0..len as u64 {
                        mem.write_u8((addr + i) as usize, 0);
                    }
                    written += len;
                }
                VIRTIO_BLK_S_OK
            }
            _ => VIRTIO_BLK_S_UNSUPP,
        };

        mem.write_u8(status_at as usize, status);
        written + 1 // the status byte counts as written
    }
}

impl Default for VirtioBlk {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PFN: u32 = 0x10; // ring at physical 0x10000

    /// Build a device with a disk whose every sector is filled with its own
    /// number, so a read can be checked against where it came from.
    fn device() -> VirtioBlk {
        let mut dev = VirtioBlk::new();
        let mut disk = vec![0u8; 8 * SECTOR as usize];
        for s in 0..8 {
            for b in 0..SECTOR as usize {
                disk[s * SECTOR as usize + b] = s as u8;
            }
        }
        dev.attach(disk);
        dev.queue_pfn = PFN;
        dev
    }

    fn rings() -> (u64, u64, u64) {
        let desc = PFN as u64 * 4096;
        let avail = desc + 16 * QUEUE_SIZE as u64;
        let used = (avail + 4 + 2 * QUEUE_SIZE as u64 + 2).div_ceil(QUEUE_ALIGN) * QUEUE_ALIGN;
        (desc, avail, used)
    }

    /// Write one descriptor.
    fn put_desc(mem: &mut Memory, desc: u64, i: u16, addr: u64, len: u32, flags: u16, next: u16) {
        let at = desc as usize + i as usize * 16;
        mem.write_u64(at, addr);
        mem.write_u32(at + 8, len);
        mem.write_u16(at + 12, flags);
        mem.write_u16(at + 14, next);
    }

    /// Lay down a three-descriptor request and make it available.
    fn submit(mem: &mut Memory, req_type: u32, sector: u64, buf: u64, len: u32, write: bool) -> u64 {
        let (desc, avail, _) = rings();
        let hdr = 0x20000u64;
        let status = 0x20100u64;
        mem.write_u32(hdr as usize, req_type);
        mem.write_u32(hdr as usize + 4, 0);
        mem.write_u64(hdr as usize + 8, sector);
        put_desc(mem, desc, 0, hdr, 16, VRING_DESC_F_NEXT, 1);
        let data_flags = VRING_DESC_F_NEXT | if write { VRING_DESC_F_WRITE } else { 0 };
        put_desc(mem, desc, 1, buf, len, data_flags, 2);
        put_desc(mem, desc, 2, status, 1, VRING_DESC_F_WRITE, 0);
        let idx = mem.read_u16(avail as usize + 2);
        mem.write_u16(avail as usize + 4 + (idx % QUEUE_SIZE) as usize * 2, 0);
        mem.write_u16(avail as usize + 2, idx.wrapping_add(1));
        status
    }

    #[test]
    fn capacity_is_reported_in_sectors() {
        let mut dev = device();
        assert_eq!(dev.read(0x14, 4), 8, "eight sectors");
    }

    #[test]
    fn the_used_ring_is_page_aligned_after_the_available_ring() {
        // The legacy layout pads to a page here. Packing the used ring
        // straight after the available one writes completions over whatever
        // the driver put in the gap.
        let dev = device();
        let (_, avail, used) = dev.ring();
        assert!(used > avail);
        assert_eq!(used % QUEUE_ALIGN, 0);
    }

    #[test]
    fn a_read_request_fills_the_guest_buffer() {
        let mut dev = device();
        let mut mem = Memory::with_size(1 << 20);
        let buf = 0x30000u64;
        let status = submit(&mut mem, VIRTIO_BLK_T_IN, 3, buf, SECTOR as u32, true);
        dev.write(0x10, 0, &mut mem); // notify

        assert_eq!(mem.read_u8(status as usize), VIRTIO_BLK_S_OK);
        assert_eq!(mem.read_u8(buf as usize), 3, "sector 3 is full of 3s");
        assert_eq!(mem.read_u8(buf as usize + 511), 3);
        assert!(dev.irq, "a completed request raises the interrupt");
        assert_eq!(dev.isr & 1, 1);
    }

    #[test]
    fn a_write_request_reaches_the_image() {
        let mut dev = device();
        let mut mem = Memory::with_size(1 << 20);
        let buf = 0x30000u64;
        for i in 0..SECTOR as usize {
            mem.write_u8(buf as usize + i, 0xAB);
        }
        let status = submit(&mut mem, VIRTIO_BLK_T_OUT, 5, buf, SECTOR as u32, false);
        dev.write(0x10, 0, &mut mem);

        assert_eq!(mem.read_u8(status as usize), VIRTIO_BLK_S_OK);
        assert_eq!(dev.disk[5 * SECTOR as usize], 0xAB, "sector 5 was written");
        assert_eq!(dev.disk[4 * SECTOR as usize], 4, "sector 4 untouched");
    }

    #[test]
    fn reading_past_the_end_of_the_image_is_an_error_not_a_panic() {
        let mut dev = device();
        let mut mem = Memory::with_size(1 << 20);
        let status = submit(&mut mem, VIRTIO_BLK_T_IN, 99, 0x30000, SECTOR as u32, true);
        dev.write(0x10, 0, &mut mem);
        assert_eq!(mem.read_u8(status as usize), VIRTIO_BLK_S_IOERR);
    }

    #[test]
    fn an_unknown_request_type_is_refused() {
        let mut dev = device();
        let mut mem = Memory::with_size(1 << 20);
        let status = submit(&mut mem, 0x2A, 0, 0x30000, SECTOR as u32, true);
        dev.write(0x10, 0, &mut mem);
        assert_eq!(mem.read_u8(status as usize), VIRTIO_BLK_S_UNSUPP);
    }

    #[test]
    fn reading_the_isr_clears_it_and_drops_the_line() {
        let mut dev = device();
        let mut mem = Memory::with_size(1 << 20);
        submit(&mut mem, VIRTIO_BLK_T_IN, 0, 0x30000, SECTOR as u32, true);
        dev.write(0x10, 0, &mut mem);
        assert!(dev.irq);
        assert_eq!(dev.read(0x13, 1), 1, "interrupt was ours");
        assert_eq!(dev.read(0x13, 1), 0, "and reading cleared it");
        assert!(!dev.irq);
    }

    #[test]
    fn two_requests_in_one_notify_are_both_consumed() {
        // The driver may make several available before notifying once; a
        // device that handles one per notify leaves the rest to time out.
        let mut dev = device();
        let mut mem = Memory::with_size(1 << 20);
        let (desc, avail, used) = rings();
        for (i, sector) in [(0u16, 1u64), (3u16, 6u64)] {
            let hdr = 0x20000 + i as u64 * 0x100;
            let status = 0x21000 + i as u64 * 0x100;
            let buf = 0x30000 + i as u64 * 0x1000;
            mem.write_u32(hdr as usize, VIRTIO_BLK_T_IN);
            mem.write_u64(hdr as usize + 8, sector);
            put_desc(&mut mem, desc, i, hdr, 16, VRING_DESC_F_NEXT, i + 1);
            put_desc(&mut mem, desc, i + 1, buf, SECTOR as u32,
                     VRING_DESC_F_NEXT | VRING_DESC_F_WRITE, i + 2);
            put_desc(&mut mem, desc, i + 2, status, 1, VRING_DESC_F_WRITE, 0);
            let idx = mem.read_u16(avail as usize + 2);
            mem.write_u16(avail as usize + 4 + (idx % QUEUE_SIZE) as usize * 2, i);
            mem.write_u16(avail as usize + 2, idx.wrapping_add(1));
        }
        dev.write(0x10, 0, &mut mem);

        assert_eq!(mem.read_u16(used as usize + 2), 2, "both completed");
        // The buffers are keyed off the descriptor index, so the second
        // chain (index 3) reads into 0x30000 + 3 * 0x1000.
        assert_eq!(mem.read_u8(0x30000), 1, "first read sector 1");
        assert_eq!(mem.read_u8(0x33000), 6, "second read sector 6");
    }

    #[test]
    fn the_available_index_is_compared_by_wrapping() {
        // Both indices are free-running u16 counters. Comparing them with
        // `<` stalls the queue for good the first time one wraps past 65535.
        let mut dev = device();
        let mut mem = Memory::with_size(1 << 20);
        let (_, avail, used) = rings();
        dev.last_avail = 0xFFFF;
        mem.write_u16(avail as usize + 2, 0xFFFF);
        submit(&mut mem, VIRTIO_BLK_T_IN, 2, 0x30000, SECTOR as u32, true);
        assert_eq!(mem.read_u16(avail as usize + 2), 0, "the driver wrapped");
        dev.write(0x10, 0, &mut mem);
        assert_eq!(mem.read_u16(used as usize + 2), 1, "served across the wrap");
        assert_eq!(mem.read_u8(0x30000), 2);
    }

    #[test]
    fn a_reset_puts_the_device_back_to_the_start_of_the_ring() {
        let mut dev = device();
        let mut mem = Memory::with_size(1 << 20);
        submit(&mut mem, VIRTIO_BLK_T_IN, 0, 0x30000, SECTOR as u32, true);
        dev.write(0x10, 0, &mut mem);
        assert_eq!(dev.last_avail, 1);
        dev.write(0x12, 0, &mut mem); // status = 0 is a reset
        assert_eq!(dev.last_avail, 0);
        assert_eq!(dev.queue_pfn, 0);
    }
}
