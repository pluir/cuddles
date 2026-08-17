//! 32-bit paging: page-directory / page-table walk.
//!
//! When CR0.PG is set, the linear address produced by segment translation is
//! translated through a two-level page table structure rooted at CR3 (the
//! page-directory base register).
//!
//! Layout of a 32-bit linear address:
//!   bits 31-22  page-directory index (1024 entries)
//!   bits 21-12  page-table index (1024 entries)
//!   bits 11-0   offset within the 4 KiB page
//!
//! A page-directory entry (PDE) points to a page table; a page-table entry
//! (PTE) points to a 4 KiB page. If the PDE has the PS (page size) bit set,
//! it maps a 4 MiB page directly. The present bit (bit 0) must be set for a
//! valid mapping.

use crate::memory::Memory;

/// Translate a linear address to a physical address using the page tables
/// rooted at `cr3`. Returns `None` if the page is not present.
///
/// Note: page-fault exceptions are not yet raised; a not-present page simply
/// yields `None` for the caller to handle (the CPU currently maps it to 0).
pub fn translate(mem: &Memory, cr3: u32, linear: u32) -> Option<usize> {
    let pd_index = (linear >> 22) & 0x3FF;
    let pt_index = (linear >> 12) & 0x3FF;
    let offset = linear & 0xFFF;

    let pd_addr = Memory::phys32(cr3.wrapping_add(pd_index * 4));
    let pde = mem.read_u32(pd_addr);
    if pde & 1 == 0 {
        return None;
    }
    if pde & 0x80 != 0 {
        // 4 MiB page: base is PDE bits 31-22, offset is linear bits 21-0.
        let base = pde & 0xFFC0_0000;
        return Some(Memory::phys32(base | (linear & 0x3F_FFFF)));
    }
    let pt_addr = Memory::phys32((pde & 0xFFFFF000).wrapping_add(pt_index * 4));
    let pte = mem.read_u32(pt_addr);
    if pte & 1 == 0 {
        return None;
    }
    let base = pte & 0xFFFFF000;
    Some(Memory::phys32(base | offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_4k_page() {
        let mut m = Memory::new();
        // Page directory at 0x1000, page table at 0x2000.
        // Map linear 0x0040_0000 (PD index 1, PT index 0) to physical 0x1000.
        m.write_u32(0x1000 + 1 * 4, 0x2003); // PDE: PT base 0x2000, present|rw
        m.write_u32(0x2000 + 0 * 4, 0x1003); // PTE: page base 0x1000, present|rw
        assert_eq!(translate(&m, 0x1000, 0x0040_0000), Some(0x1000));
        // Offset within the page.
        assert_eq!(translate(&m, 0x1000, 0x0040_0123), Some(0x1123));
    }

    #[test]
    fn not_present_returns_none() {
        let mut m = Memory::new();
        m.write_u32(0x1000 + 1 * 4, 0x2000); // PDE present bit clear
        assert_eq!(translate(&m, 0x1000, 0x0040_0000), None);
    }

    #[test]
    fn maps_4mb_page() {
        let mut m = Memory::new();
        // PDE with PS bit (0x80): maps a 4 MiB page at base 0x0000_0000.
        m.write_u32(0x1000 + 0 * 4, 0x0000_0083);
        assert_eq!(translate(&m, 0x1000, 0x0000_1234), Some(0x1234));
        assert_eq!(translate(&m, 0x1000, 0x0020_0000), Some(0x0020_0000));
    }
}
