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
//!
//! The walk also reports the *permissions* of the mapping, which is not
//! optional detail: a page's effective R/W and U/S bits are the AND of the
//! PDE's and the PTE's, and with CR0.WP set even a supervisor write to a
//! read-only page must fault. Linux boots straight into a check for exactly
//! that (`test_wp_bit`) and panics if the CPU gets it wrong.

use crate::memory::Memory;

/// Page-table entry bits used here.
pub mod pte {
    /// Present.
    pub const P: u32 = 1 << 0;
    /// Read/write (0 = read-only).
    pub const RW: u32 = 1 << 1;
    /// User/supervisor (0 = supervisor-only).
    pub const US: u32 = 1 << 2;
    /// Page size (PDE only; 1 = 4 MiB page).
    pub const PS: u32 = 1 << 7;
    /// Accessed.
    pub const A: u32 = 1 << 5;
    /// Dirty.
    pub const D: u32 = 1 << 6;
}

/// The result of a successful page-table walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mapping {
    /// Physical address the linear address maps to.
    pub phys: usize,
    /// True when the mapping permits writes (PDE.RW & PTE.RW).
    pub writable: bool,
    /// True when the mapping is reachable from user mode (PDE.US & PTE.US).
    pub user: bool,
    /// Physical address of the PDE, so the CPU can set its accessed bit.
    pub pde_addr: usize,
    /// Physical address of the PTE, or `None` for a 4 MiB page (where the
    /// PDE itself carries the accessed/dirty bits).
    pub pte_addr: Option<usize>,
}

/// Translate a linear address through the page tables rooted at `cr3`,
/// reporting the mapping's permissions. Returns `None` if the page is not
/// present — the caller decides what kind of fault that is.
pub fn translate(mem: &Memory, cr3: u32, linear: u32) -> Option<Mapping> {
    let pd_index = (linear >> 22) & 0x3FF;
    let pt_index = (linear >> 12) & 0x3FF;
    let offset = linear & 0xFFF;

    let pde_addr = Memory::phys32(cr3.wrapping_add(pd_index * 4));
    let pde = mem.read_u32(pde_addr);
    if pde & pte::P == 0 {
        return None;
    }
    if pde & pte::PS != 0 {
        // 4 MiB page: base is PDE bits 31-22, offset is linear bits 21-0.
        let base = pde & 0xFFC0_0000;
        return Some(Mapping {
            phys: Memory::phys32(base | (linear & 0x3F_FFFF)),
            writable: pde & pte::RW != 0,
            user: pde & pte::US != 0,
            pde_addr,
            pte_addr: None,
        });
    }
    let pte_addr = Memory::phys32((pde & 0xFFFFF000).wrapping_add(pt_index * 4));
    let entry = mem.read_u32(pte_addr);
    if entry & pte::P == 0 {
        return None;
    }
    let base = entry & 0xFFFFF000;
    Some(Mapping {
        // The effective permission is the AND of both levels: a read-only PDE
        // makes every page under it read-only however the PTE is marked.
        phys: Memory::phys32(base | offset),
        writable: (pde & pte::RW != 0) && (entry & pte::RW != 0),
        user: (pde & pte::US != 0) && (entry & pte::US != 0),
        pde_addr,
        pte_addr: Some(pte_addr),
    })
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
        m.write_u32(0x2000, 0x1003); // PTE: page base 0x1000, present|rw
        let map = translate(&m, 0x1000, 0x0040_0000).unwrap();
        assert_eq!(map.phys, 0x1000);
        assert!(map.writable);
        // Offset within the page.
        assert_eq!(translate(&m, 0x1000, 0x0040_0123).unwrap().phys, 0x1123);
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
        m.write_u32(0x1000, 0x0000_0083);
        assert_eq!(translate(&m, 0x1000, 0x0000_1234).unwrap().phys, 0x1234);
        assert_eq!(translate(&m, 0x1000, 0x0020_0000).unwrap().phys, 0x0020_0000);
    }

    #[test]
    fn read_only_page_reports_not_writable() {
        let mut m = Memory::new();
        m.write_u32(0x1000 + 1 * 4, 0x2003); // PDE present|rw
        m.write_u32(0x2000, 0x1001); // PTE present, read-only
        let map = translate(&m, 0x1000, 0x0040_0000).unwrap();
        assert_eq!(map.phys, 0x1000);
        assert!(!map.writable);
    }

    #[test]
    fn read_only_directory_makes_every_page_under_it_read_only() {
        let mut m = Memory::new();
        m.write_u32(0x1000 + 1 * 4, 0x2001); // PDE present, read-only
        m.write_u32(0x2000, 0x1003); // PTE present|rw
        assert!(!translate(&m, 0x1000, 0x0040_0000).unwrap().writable);
    }

    #[test]
    fn user_bit_is_the_and_of_both_levels() {
        let mut m = Memory::new();
        m.write_u32(0x1000 + 1 * 4, 0x2003); // PDE supervisor-only
        m.write_u32(0x2000, 0x1007); // PTE user
        assert!(!translate(&m, 0x1000, 0x0040_0000).unwrap().user);
        m.write_u32(0x1000 + 1 * 4, 0x2007); // PDE user too
        assert!(translate(&m, 0x1000, 0x0040_0000).unwrap().user);
    }
}
