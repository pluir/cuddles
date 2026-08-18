//! Page-table walks, for all three of x86's paging structures.
//!
//! When CR0.PG is set, the linear address produced by segment translation is
//! translated through a tree of tables rooted at CR3. Which tree depends on
//! two bits that live a long way apart:
//!
//! | CR4.PAE | EFER.LMA | structure | levels | entry | max page |
//! |---------|----------|-----------|--------|-------|----------|
//! | 0       | 0        | legacy    | 2      | 4 B   | 4 MiB    |
//! | 1       | 0        | PAE       | 3      | 8 B   | 2 MiB    |
//! | 1       | 1        | long      | 4      | 8 B   | 1 GiB    |
//!
//! Legacy paging is the 386's: a 1024-entry page directory of 4-byte entries,
//! each naming a 1024-entry page table, and it can only describe 4 GiB of
//! physical memory. PAE widens the entries to 8 bytes to reach 52 physical
//! address bits — at the cost of a third level, because 8-byte entries halve
//! how much of the address space one table covers. Long mode adds a fourth
//! level on top of the same 8-byte entries, which is what takes the *linear*
//! address from 32 bits to 48.
//!
//! The walk also reports the *permissions* of the mapping, which is not
//! optional detail: a page's effective R/W, U/S and NX are the combination of
//! every level's, and with CR0.WP set even a supervisor write to a read-only
//! page must fault. Linux boots straight into a check for exactly that
//! (`test_wp_bit`) and panics if the CPU gets it wrong.

use crate::memory::Memory;

/// Page-table entry bits. These are the same bits at every level and in every
/// mode, which is the one thing the three structures agree on.
pub mod pte {
    /// Present.
    pub const P: u64 = 1 << 0;
    /// Read/write (0 = read-only).
    pub const RW: u64 = 1 << 1;
    /// User/supervisor (0 = supervisor-only).
    pub const US: u64 = 1 << 2;
    /// Accessed.
    pub const A: u64 = 1 << 5;
    /// Dirty (leaf entries only).
    pub const D: u64 = 1 << 6;
    /// Page size: this entry maps a large page instead of naming a table.
    pub const PS: u64 = 1 << 7;
    /// No-execute. Only meaningful with 8-byte entries and EFER.NXE set.
    pub const NX: u64 = 1 << 63;
}

/// Physical-address field of an 8-byte entry (bits 51:12).
const ADDR_MASK_64: u64 = 0x000F_FFFF_FFFF_F000;
/// Physical-address field of a 4-byte entry (bits 31:12).
const ADDR_MASK_32: u64 = 0xFFFF_F000;

/// Which paging structure CR3 roots.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PagingMode {
    /// CR0.PG is clear: the linear address *is* the physical address.
    Off,
    /// 32-bit, two levels of 4-byte entries.
    Legacy,
    /// PAE: three levels of 8-byte entries.
    Pae,
    /// Long mode: four levels of 8-byte entries.
    Long,
}

/// The result of a successful page-table walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mapping {
    /// Physical address the linear address maps to.
    pub phys: u64,
    /// True when the mapping permits writes (the AND of every level's RW).
    pub writable: bool,
    /// True when the mapping is reachable from user mode (AND of every US).
    pub user: bool,
    /// True when instructions may be fetched from it (no level set NX, or
    /// EFER.NXE is clear so the bit means nothing).
    pub exec: bool,
    /// Physical addresses of the entries the walk read, outermost first. The
    /// last is the leaf. Kept so the CPU can set accessed and dirty bits
    /// without walking a second time.
    pub walk: [u64; 4],
    /// How many of `walk` are valid.
    pub walk_len: u8,
    /// Width of each entry in bytes: 4 for legacy paging, 8 otherwise.
    pub entry_bytes: u8,
}

impl Mapping {
    /// The leaf entry's physical address — the one that carries the dirty bit.
    #[inline]
    pub fn leaf(&self) -> u64 { self.walk[(self.walk_len - 1) as usize] }
}

/// Translate `linear` through the structure `mode` names, rooted at `cr3`.
/// Returns `None` if any level is not present — the caller decides what kind
/// of fault that is.
#[inline]
pub fn translate_mode(
    mem: &Memory, cr3: u64, linear: u64, mode: PagingMode, nxe: bool,
) -> Option<Mapping> {
    match mode {
        PagingMode::Off => Some(Mapping {
            phys: linear, writable: true, user: true, exec: true,
            walk: [0; 4], walk_len: 0, entry_bytes: 0,
        }),
        PagingMode::Legacy => translate_legacy(mem, cr3, linear as u32),
        PagingMode::Pae => translate_pae(mem, cr3, linear as u32, nxe),
        PagingMode::Long => translate_long(mem, cr3, linear, nxe),
    }
}

/// 32-bit paging: page directory -> page table, 4-byte entries, 4 KiB or
/// 4 MiB pages.
pub fn translate_legacy(mem: &Memory, cr3: u64, linear: u32) -> Option<Mapping> {
    let pd_index = (linear >> 22) & 0x3FF;
    let pt_index = (linear >> 12) & 0x3FF;
    let offset = (linear & 0xFFF) as u64;

    let pde_addr = (cr3 & ADDR_MASK_32).wrapping_add((pd_index as u64) * 4);
    let pde = mem.read_u32(pde_addr as usize) as u64;
    if pde & pte::P == 0 {
        return None;
    }
    if pde & pte::PS != 0 {
        // 4 MiB page: base is PDE bits 31-22, offset is linear bits 21-0.
        let base = pde & 0xFFC0_0000;
        return Some(Mapping {
            phys: base | (linear as u64 & 0x3F_FFFF),
            writable: pde & pte::RW != 0,
            user: pde & pte::US != 0,
            exec: true,
            walk: [pde_addr, 0, 0, 0],
            walk_len: 1,
            entry_bytes: 4,
        });
    }
    let pte_addr = (pde & ADDR_MASK_32).wrapping_add((pt_index as u64) * 4);
    let entry = mem.read_u32(pte_addr as usize) as u64;
    if entry & pte::P == 0 {
        return None;
    }
    Some(Mapping {
        // The effective permission is the AND of both levels: a read-only PDE
        // makes every page under it read-only however the PTE is marked.
        phys: (entry & ADDR_MASK_32) | offset,
        writable: (pde & pte::RW != 0) && (entry & pte::RW != 0),
        user: (pde & pte::US != 0) && (entry & pte::US != 0),
        exec: true,
        walk: [pde_addr, pte_addr, 0, 0],
        walk_len: 2,
        entry_bytes: 4,
    })
}

/// PAE paging: a four-entry page-directory-pointer table -> page directory ->
/// page table, 8-byte entries, 4 KiB or 2 MiB pages.
///
/// The PDPTE is the odd one out: only its present bit is architecturally
/// meaningful, so it contributes nothing to the permission AND.
pub fn translate_pae(mem: &Memory, cr3: u64, linear: u32, nxe: bool) -> Option<Mapping> {
    let pdpt_index = (linear >> 30) & 0x3;
    let pd_index = (linear >> 21) & 0x1FF;
    let pt_index = (linear >> 12) & 0x1FF;

    // The PDPT is 32-byte aligned, not page aligned.
    let pdpte_addr = (cr3 & !0x1F).wrapping_add((pdpt_index as u64) * 8);
    let pdpte = mem.read_u64(pdpte_addr as usize);
    if pdpte & pte::P == 0 {
        return None;
    }
    let pde_addr = (pdpte & ADDR_MASK_64).wrapping_add((pd_index as u64) * 8);
    let pde = mem.read_u64(pde_addr as usize);
    if pde & pte::P == 0 {
        return None;
    }
    if pde & pte::PS != 0 {
        // 2 MiB page.
        return Some(Mapping {
            phys: (pde & ADDR_MASK_64 & !0x1F_FFFF) | (linear as u64 & 0x1F_FFFF),
            writable: pde & pte::RW != 0,
            user: pde & pte::US != 0,
            exec: !nxe || (pde & pte::NX == 0),
            walk: [pdpte_addr, pde_addr, 0, 0],
            walk_len: 2,
            entry_bytes: 8,
        });
    }
    let pte_addr = (pde & ADDR_MASK_64).wrapping_add((pt_index as u64) * 8);
    let entry = mem.read_u64(pte_addr as usize);
    if entry & pte::P == 0 {
        return None;
    }
    Some(Mapping {
        phys: (entry & ADDR_MASK_64) | (linear as u64 & 0xFFF),
        writable: (pde & pte::RW != 0) && (entry & pte::RW != 0),
        user: (pde & pte::US != 0) && (entry & pte::US != 0),
        exec: !nxe || ((pde | entry) & pte::NX == 0),
        walk: [pdpte_addr, pde_addr, pte_addr, 0],
        walk_len: 3,
        entry_bytes: 8,
    })
}

/// Long-mode paging: PML4 -> PDPT -> PD -> PT, 8-byte entries, 4 KiB, 2 MiB
/// or 1 GiB pages. Every level carries RW, US and NX, so all four take part
/// in the permission combination.
pub fn translate_long(mem: &Memory, cr3: u64, linear: u64, nxe: bool) -> Option<Mapping> {
    let pml4_index = (linear >> 39) & 0x1FF;
    let pdpt_index = (linear >> 30) & 0x1FF;
    let pd_index = (linear >> 21) & 0x1FF;
    let pt_index = (linear >> 12) & 0x1FF;

    let pml4e_addr = (cr3 & ADDR_MASK_64).wrapping_add(pml4_index * 8);
    let pml4e = mem.read_u64(pml4e_addr as usize);
    if pml4e & pte::P == 0 {
        return None;
    }
    let pdpte_addr = (pml4e & ADDR_MASK_64).wrapping_add(pdpt_index * 8);
    let pdpte = mem.read_u64(pdpte_addr as usize);
    if pdpte & pte::P == 0 {
        return None;
    }
    if pdpte & pte::PS != 0 {
        // 1 GiB page.
        let acc = pml4e | pdpte;
        return Some(Mapping {
            phys: (pdpte & ADDR_MASK_64 & !0x3FFF_FFFF) | (linear & 0x3FFF_FFFF),
            writable: (pml4e & pte::RW != 0) && (pdpte & pte::RW != 0),
            user: (pml4e & pte::US != 0) && (pdpte & pte::US != 0),
            exec: !nxe || (acc & pte::NX == 0),
            walk: [pml4e_addr, pdpte_addr, 0, 0],
            walk_len: 2,
            entry_bytes: 8,
        });
    }
    let pde_addr = (pdpte & ADDR_MASK_64).wrapping_add(pd_index * 8);
    let pde = mem.read_u64(pde_addr as usize);
    if pde & pte::P == 0 {
        return None;
    }
    if pde & pte::PS != 0 {
        // 2 MiB page.
        let acc = pml4e | pdpte | pde;
        return Some(Mapping {
            phys: (pde & ADDR_MASK_64 & !0x1F_FFFF) | (linear & 0x1F_FFFF),
            writable: (pml4e & pdpte & pde & pte::RW) != 0,
            user: (pml4e & pdpte & pde & pte::US) != 0,
            exec: !nxe || (acc & pte::NX == 0),
            walk: [pml4e_addr, pdpte_addr, pde_addr, 0],
            walk_len: 3,
            entry_bytes: 8,
        });
    }
    let pte_addr = (pde & ADDR_MASK_64).wrapping_add(pt_index * 8);
    let entry = mem.read_u64(pte_addr as usize);
    if entry & pte::P == 0 {
        return None;
    }
    let acc = pml4e | pdpte | pde | entry;
    Some(Mapping {
        phys: (entry & ADDR_MASK_64) | (linear & 0xFFF),
        writable: (pml4e & pdpte & pde & entry & pte::RW) != 0,
        user: (pml4e & pdpte & pde & entry & pte::US) != 0,
        exec: !nxe || (acc & pte::NX == 0),
        walk: [pml4e_addr, pdpte_addr, pde_addr, pte_addr],
        walk_len: 4,
        entry_bytes: 8,
    })
}

/// True when a 64-bit linear address is *canonical*: bits 63:48 must all
/// repeat bit 47. A non-canonical address is a #GP, not a page fault — the
/// address never reaches the page tables at all, which is what stops the
/// unused middle of the 64-bit address space from being quietly aliased.
#[inline]
pub fn canonical(linear: u64) -> bool {
    let top = linear >> 47;
    top == 0 || top == 0x1FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy(m: &Memory, linear: u32) -> Option<Mapping> {
        translate_legacy(m, 0x1000, linear)
    }

    #[test]
    fn maps_4k_page() {
        let mut m = Memory::new();
        // Page directory at 0x1000, page table at 0x2000.
        // Map linear 0x0040_0000 (PD index 1, PT index 0) to physical 0x1000.
        m.write_u32(0x1000 + 4, 0x2003); // PDE: PT base 0x2000, present|rw
        m.write_u32(0x2000, 0x1003); // PTE: page base 0x1000, present|rw
        let map = legacy(&m, 0x0040_0000).unwrap();
        assert_eq!(map.phys, 0x1000);
        assert!(map.writable);
        // Offset within the page.
        assert_eq!(legacy(&m, 0x0040_0123).unwrap().phys, 0x1123);
    }

    #[test]
    fn not_present_returns_none() {
        let mut m = Memory::new();
        m.write_u32(0x1000 + 4, 0x2000); // PDE present bit clear
        assert_eq!(legacy(&m, 0x0040_0000), None);
    }

    #[test]
    fn maps_4mb_page() {
        let mut m = Memory::new();
        // PDE with PS bit (0x80): maps a 4 MiB page at base 0x0000_0000.
        m.write_u32(0x1000, 0x0000_0083);
        assert_eq!(legacy(&m, 0x0000_1234).unwrap().phys, 0x1234);
        assert_eq!(legacy(&m, 0x0020_0000).unwrap().phys, 0x0020_0000);
    }

    #[test]
    fn read_only_page_reports_not_writable() {
        let mut m = Memory::new();
        m.write_u32(0x1000 + 4, 0x2003); // PDE present|rw
        m.write_u32(0x2000, 0x1001); // PTE present, read-only
        let map = legacy(&m, 0x0040_0000).unwrap();
        assert_eq!(map.phys, 0x1000);
        assert!(!map.writable);
    }

    #[test]
    fn read_only_directory_makes_every_page_under_it_read_only() {
        let mut m = Memory::new();
        m.write_u32(0x1000 + 4, 0x2001); // PDE present, read-only
        m.write_u32(0x2000, 0x1003); // PTE present|rw
        assert!(!legacy(&m, 0x0040_0000).unwrap().writable);
    }

    #[test]
    fn user_bit_is_the_and_of_both_levels() {
        let mut m = Memory::new();
        m.write_u32(0x1000 + 4, 0x2003); // PDE supervisor-only
        m.write_u32(0x2000, 0x1007); // PTE user
        assert!(!legacy(&m, 0x0040_0000).unwrap().user);
        m.write_u32(0x1000 + 4, 0x2007); // PDE user too
        assert!(legacy(&m, 0x0040_0000).unwrap().user);
    }

    // ---- PAE ----

    #[test]
    fn pae_walks_three_levels() {
        let mut m = Memory::new();
        // PDPT at 0x1000, PD at 0x2000, PT at 0x3000.
        m.write_u64(0x1000, 0x2001); // PDPTE 0 -> PD, present
        m.write_u64(0x2000, 0x3003); // PDE 0 -> PT, present|rw
        m.write_u64(0x3000 + 8, 0x5_5000 | 3); // PTE 1 -> page 0x55000
        let map = translate_pae(&m, 0x1000, 0x1234, false).unwrap();
        assert_eq!(map.phys, 0x5_5234);
        assert!(map.writable);
        assert_eq!(map.walk_len, 3);
        assert_eq!(map.entry_bytes, 8);
    }

    #[test]
    fn pae_maps_a_2mb_page() {
        let mut m = Memory::new();
        m.write_u64(0x1000, 0x2001);
        m.write_u64(0x2000, 0x40_0000 | 0x83); // PDE with PS: 2 MiB at 4 MiB
        let map = translate_pae(&m, 0x1000, 0x1_2345, false).unwrap();
        assert_eq!(map.phys, 0x40_0000 + 0x1_2345);
    }

    // ---- Long mode ----

    #[test]
    fn long_mode_walks_four_levels() {
        let mut m = Memory::new();
        // PML4 at 0x1000 -> PDPT 0x2000 -> PD 0x3000 -> PT 0x4000.
        m.write_u64(0x1000, 0x2003);
        m.write_u64(0x2000, 0x3003);
        m.write_u64(0x3000, 0x4003);
        m.write_u64(0x4000 + 8, 0x9_9000 | 3);
        let map = translate_long(&m, 0x1000, 0x1ABC, false).unwrap();
        assert_eq!(map.phys, 0x9_9ABC);
        assert_eq!(map.walk_len, 4);
    }

    #[test]
    fn long_mode_indexes_the_high_half() {
        let mut m = Memory::new();
        // 0xFFFF_8000_0000_0000 is PML4 index 256, everything else zero.
        m.write_u64(0x1000 + 256 * 8, 0x2003);
        m.write_u64(0x2000, 0x3003);
        m.write_u64(0x3000, 0x4003);
        m.write_u64(0x4000, 0x7_7000 | 3);
        let map = translate_long(&m, 0x1000, 0xFFFF_8000_0000_0000, false).unwrap();
        assert_eq!(map.phys, 0x7_7000);
    }

    #[test]
    fn long_mode_maps_a_1gb_page() {
        let mut m = Memory::new();
        m.write_u64(0x1000, 0x2003);
        m.write_u64(0x2000, 0x8000_0000 | 0x83); // PDPTE with PS: 1 GiB at 2 GiB
        let map = translate_long(&m, 0x1000, 0x1234_5678, false).unwrap();
        assert_eq!(map.phys, 0x8000_0000 + 0x1234_5678);
        assert_eq!(map.walk_len, 2);
    }

    #[test]
    fn nx_only_bites_when_the_feature_is_enabled() {
        let mut m = Memory::new();
        m.write_u64(0x1000, 0x2003);
        m.write_u64(0x2000, 0x3003);
        m.write_u64(0x3000, 0x4003);
        m.write_u64(0x4000, 0x5000 | 3 | pte::NX);
        // With EFER.NXE clear, bit 63 is reserved and means nothing.
        assert!(translate_long(&m, 0x1000, 0, false).unwrap().exec);
        assert!(!translate_long(&m, 0x1000, 0, true).unwrap().exec);
    }

    #[test]
    fn a_higher_level_nx_covers_everything_under_it() {
        let mut m = Memory::new();
        m.write_u64(0x1000, 0x2003 | pte::NX); // PML4E says no-execute
        m.write_u64(0x2000, 0x3003);
        m.write_u64(0x3000, 0x4003);
        m.write_u64(0x4000, 0x5003);
        assert!(!translate_long(&m, 0x1000, 0, true).unwrap().exec);
    }

    #[test]
    fn canonical_addresses_are_the_two_ends_of_the_range() {
        assert!(canonical(0));
        assert!(canonical(0x0000_7FFF_FFFF_FFFF));
        assert!(canonical(0xFFFF_8000_0000_0000));
        assert!(canonical(0xFFFF_FFFF_FFFF_FFFF));
        // The middle of the address space is not addressable.
        assert!(!canonical(0x0000_8000_0000_0000));
        assert!(!canonical(0x1234_5678_9ABC_DEF0));
    }
}
