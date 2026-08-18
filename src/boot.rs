//! Linux boot-protocol loader.
//!
//! Loads a bzImage (the Linux boot protocol, Documentation/x86/boot.rst)
//! into the emulator and enters the 32-bit protected-mode kernel entry point
//! directly, bypassing the real-mode setup code. It parses the setup header,
//! loads the protected-mode kernel at `code32_start`, builds a `boot_params`
//! structure at 0x90000 (with the E820 memory map and command line), sets up
//! a flat GDT, enables protected mode, and jumps to the kernel with ESI
//! pointing at `boot_params` — exactly what the kernel's `startup_32` expects.

use crate::cpu::{Cpu, SegReg};
use crate::memory::Memory;

/// Physical address where we build the `boot_params` structure.
pub const BOOT_PARAMS_ADDR: u32 = 0x90000;
/// Physical address of the flat GDT.
pub const GDT_ADDR: u32 = 0x1000;
/// Physical address of the command line.
pub const CMDLINE_ADDR: u32 = 0x20000;
/// Flat code selector (GDT index 1).
pub const KERNEL_CS: u16 = 0x08;
/// Flat data selector (GDT index 2).
pub const KERNEL_DS: u16 = 0x10;

/// Offsets of the setup-header fields, relative to the start of the boot
/// sector (file offset 0 of the bzImage). These match the Linux boot
/// protocol (Documentation/x86/boot.rst).
const SETUP_SECTS: usize = 0x1F1;
const SYSSIZE: usize = 0x1F4;
const BOOT_FLAG: usize = 0x1FE;
const HDR_MAGIC: usize = 0x202;
const TYPE_OF_LOADER: usize = 0x210;
const LOADFLAGS: usize = 0x211;
const CODE32_START: usize = 0x214;
const RAMDISK_IMAGE: usize = 0x218;
const RAMDISK_SIZE: usize = 0x21C;
const CMD_LINE_PTR: usize = 0x228;
const KERNEL_ALIGNMENT: usize = 0x22E;
const RELOCATABLE: usize = 0x232;
const PREF_ADDRESS: usize = 0x250;
const INIT_SIZE: usize = 0x258;

/// Offsets within `struct boot_params` (relative to `BOOT_PARAMS_ADDR`).
///
/// `struct screen_info` occupies the first 0x40 bytes. It is what tells Linux
/// there is a text console to write on: `vgacon_startup()` treats zero rows
/// or columns as "screen_info was never filled in" and falls back to the
/// dummy console, which silently discards every message.
const BP_ORIG_X: usize = 0x00;
const BP_ORIG_Y: usize = 0x01;
const BP_ORIG_VIDEO_PAGE: usize = 0x04;
const BP_ORIG_VIDEO_MODE: usize = 0x06;
const BP_ORIG_VIDEO_COLS: usize = 0x07;
const BP_ORIG_VIDEO_EGA_BX: usize = 0x0A;
const BP_ORIG_VIDEO_LINES: usize = 0x0E;
const BP_ORIG_VIDEO_ISVGA: usize = 0x0F;
const BP_ORIG_VIDEO_POINTS: usize = 0x10;

/// `orig_video_isVGA` value for a colour VGA text console (VIDEO_TYPE_VGAC).
const VIDEO_TYPE_VGAC: u8 = 0x22;
/// BIOS video mode 3: 80x25 colour text.
const VIDEO_MODE_COLOR_TEXT: u8 = 0x03;

const BP_HDR: usize = 0x1F1; // struct setup_header hdr
const BP_ALT_MEM_K: usize = 0x1E0;
const BP_E820_ENTRIES: usize = 0x1E8;
const BP_E820_TABLE: usize = 0x2D0;

/// Info parsed from the bzImage setup header.
#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub setup_sects: usize,
    pub syssize: u32,
    pub code32_start: u32,
    pub loadflags: u8,
    pub relocatable: bool,
    pub kernel_alignment: u32,
    pub pref_address: u64,
    pub init_size: u32,
    pub kernel_offset: usize,
    pub kernel_len: usize,
}

fn le16(b: &[u8], off: usize) -> u16 {
    b[off] as u16 | ((b[off + 1] as u16) << 8)
}
fn le32(b: &[u8], off: usize) -> u32 {
    b[off] as u32
        | ((b[off + 1] as u32) << 8)
        | ((b[off + 2] as u32) << 16)
        | ((b[off + 3] as u32) << 24)
}
fn le64(b: &[u8], off: usize) -> u64 {
    let mut v = 0u64;
    for i in 0..8 {
        v |= (b[off + i] as u64) << (8 * i);
    }
    v
}

/// Parse the setup header of a bzImage.
///
/// The setup header lives at file offset `0x1F1` (after the 512-byte boot
/// sector). We validate the boot flag (0xAA55) and the "HdrS" magic, then
/// read the fields we need to load and enter the kernel.
pub fn parse_bzimage(image: &[u8]) -> Result<KernelInfo, String> {
    if image.len() < 0x260 {
        return Err("image too small to contain a setup header".into());
    }
    let boot_flag = le16(image, BOOT_FLAG);
    if boot_flag != 0xAA55 {
        return Err(format!("bad boot flag 0x{:04X} (not a bzImage?)", boot_flag));
    }
    let magic = le32(image, HDR_MAGIC);
    if magic != 0x53726448 {
        // "HdrS"
        return Err("bad setup header magic (not 'HdrS')".into());
    }
    let setup_sects = image[SETUP_SECTS] as usize;
    let syssize = le32(image, SYSSIZE);
    let code32_start = le32(image, CODE32_START);
    let loadflags = image[LOADFLAGS];
    let relocatable = image[RELOCATABLE] != 0;
    let kernel_alignment = le32(image, KERNEL_ALIGNMENT);
    let pref_address = le64(image, PREF_ADDRESS);
    let init_size = le32(image, INIT_SIZE);
    let kernel_offset = (setup_sects + 1) * 512;
    let kernel_len = syssize as usize * 16;
    if image.len() < kernel_offset + kernel_len {
        return Err(format!(
            "image too small for kernel (need {} bytes at offset {}, have {})",
            kernel_len,
            kernel_offset,
            image.len()
        ));
    }
    Ok(KernelInfo {
        setup_sects,
        syssize,
        code32_start,
        loadflags,
        relocatable,
        kernel_alignment,
        pref_address,
        init_size,
        kernel_offset,
        kernel_len,
    })
}

/// Write a flat GDT (null, flat code, flat data) at `GDT_ADDR`.
fn write_gdt(mem: &mut Memory) {
    // Null descriptor.
    mem.write_u64(GDT_ADDR as usize, 0);
    // Flat code: base 0, limit 4 GB, G=1, D=1, present, DPL0, type 0x9A.
    mem.write_u64(GDT_ADDR as usize + 8, 0x00CF9A000000FFFF);
    // Flat data: base 0, limit 4 GB, G=1, D=1, present, DPL0, type 0x92.
    mem.write_u64(GDT_ADDR as usize + 16, 0x00CF92000000FFFF);
}

/// Fill in `boot_params.screen_info` to describe the emulated VGA text
/// console: mode 3, 80x25, colour VGA. Without this Linux picks the dummy
/// console and nothing ever reaches the screen.
fn write_screen_info(mem: &mut Memory) {
    let bp = BOOT_PARAMS_ADDR as usize;
    mem.write_u8(bp + BP_ORIG_X, 0);
    mem.write_u8(bp + BP_ORIG_Y, 0);
    mem.write_u16(bp + BP_ORIG_VIDEO_PAGE, 0);
    mem.write_u8(bp + BP_ORIG_VIDEO_MODE, VIDEO_MODE_COLOR_TEXT);
    mem.write_u8(bp + BP_ORIG_VIDEO_COLS, crate::bios::SCREEN_COLS as u8);
    // EGA BX: 3 means colour, 64 KB of video memory.
    mem.write_u16(bp + BP_ORIG_VIDEO_EGA_BX, 3);
    mem.write_u8(bp + BP_ORIG_VIDEO_LINES, crate::bios::SCREEN_ROWS as u8);
    mem.write_u8(bp + BP_ORIG_VIDEO_ISVGA, VIDEO_TYPE_VGAC);
    // Character cell height, in scan lines.
    mem.write_u16(bp + BP_ORIG_VIDEO_POINTS, 16);
}

/// Write the E820 memory map into `boot_params`.
///
/// The map is the machine's own (`Memory::e820`), not a second description of
/// it: a bootloader that told the kernel something the BIOS would not have is
/// how a guest ends up using RAM that is not there. `boot_params` has room
/// for 128 entries; a real machine's map is a handful.
fn write_e820(mem: &mut Memory) {
    let map = BOOT_PARAMS_ADDR as usize + BP_E820_TABLE;
    let entries = mem.e820();
    for (i, (base, len, typ)) in entries.iter().enumerate() {
        let e = map + i * 20;
        mem.write_u64(e, *base);
        mem.write_u64(e + 8, *len);
        mem.write_u32(e + 16, *typ);
    }
    mem.write_u8(BOOT_PARAMS_ADDR as usize + BP_E820_ENTRIES, entries.len() as u8);
    // alt_mem_k is the pre-E820 fallback: extended memory in KB above 1 MiB,
    // as a 32-bit count, so it tops out at 4 TiB and is only consulted when
    // the E820 map is missing.
    let low = mem.ram_size().min(crate::memory::MMIO_HOLE_START);
    let ext_kb = low.saturating_sub(0x10_0000) / 1024;
    mem.write_u32(BOOT_PARAMS_ADDR as usize + BP_ALT_MEM_K, ext_kb as u32);
}

// ---------------------------------------------------------------------------
// Long mode
// ---------------------------------------------------------------------------

/// Physical address of the long-mode GDT, kept clear of the 32-bit one so a
/// 64-bit boot does not have to disturb it.
pub const GDT64_ADDR: u32 = 0x1800;
/// Physical address of the PML4. The four levels sit in consecutive pages.
pub const PML4_ADDR: u32 = 0x2000;
const PDPT_ADDR: u32 = 0x3000;
/// Page directories for the identity map: one per GiB covered.
const PD_ADDR: u32 = 0x4000;
/// 64-bit code selector (GDT index 1) and data selector (index 2).
pub const KERNEL_CS64: u16 = 0x08;
pub const KERNEL_DS64: u16 = 0x10;
/// How much physical memory the firmware identity-maps before handing over.
/// Four gigabytes, in 2 MiB pages -- enough for any payload to reach its own
/// image, the boot structures and the framebuffer without building its own
/// tables first.
const IDENTITY_MAP_BYTES: u64 = 4 << 30;

/// Write a GDT with a null descriptor, a 64-bit code segment and a data
/// segment.
///
/// The 64-bit code segment is recognisable by what it *lacks*: no base, no
/// limit (both are ignored in 64-bit mode), D/B clear and **L set**. A
/// descriptor with both L and D/B set is illegal, and one with D/B set
/// instead of L is a perfectly valid 32-bit segment -- which is why a
/// mistake here does not fault, it just quietly runs 64-bit code as 32-bit.
fn write_gdt64(mem: &mut Memory) {
    mem.write_u64(GDT64_ADDR as usize, 0);
    // Code: present, DPL 0, code, readable; G=1, L=1, D=0.
    mem.write_u64(GDT64_ADDR as usize + 8, 0x00AF_9A00_0000_FFFF);
    // Data: present, DPL 0, data, writable. Long mode ignores the rest.
    mem.write_u64(GDT64_ADDR as usize + 16, 0x00CF_9200_0000_FFFF);
}

/// Build a 4-level page table that identity-maps the low `IDENTITY_MAP_BYTES`
/// of physical memory with 2 MiB pages, and returns the PML4 address.
///
/// 2 MiB pages are not an optimisation here, they are what makes this
/// tractable: mapping four gigabytes with 4 KiB pages needs two thousand page
/// tables and eight megabytes of them, where the large-page form needs four.
pub fn build_identity_map(mem: &mut Memory) -> u64 {
    use crate::paging::pte;
    let present_rw = pte::P | pte::RW;
    // PML4[0] -> PDPT.
    for i in 0..512 {
        mem.write_u64(PML4_ADDR as usize + i * 8, 0);
    }
    mem.write_u64(PML4_ADDR as usize, PDPT_ADDR as u64 | present_rw);
    let gibs = (IDENTITY_MAP_BYTES / (1 << 30)) as usize;
    for i in 0..512 {
        let e = if i < gibs {
            (PD_ADDR as u64 + (i as u64) * 0x1000) | present_rw
        } else {
            0
        };
        mem.write_u64(PDPT_ADDR as usize + i * 8, e);
    }
    // Each page directory maps one GiB as 512 2 MiB pages.
    for g in 0..gibs {
        let pd = PD_ADDR as usize + g * 0x1000;
        for i in 0..512 {
            let phys = (g as u64) * (1 << 30) + (i as u64) * (2 << 20);
            mem.write_u64(pd + i * 8, phys | present_rw | pte::PS);
        }
    }
    PML4_ADDR as u64
}

/// Put the CPU into 64-bit long mode with an identity-mapped low 4 GiB, a
/// flat 64-bit GDT and a stack, exactly as firmware hands over to a 64-bit
/// payload.
///
/// The order of the four steps below is the whole of the long-mode entry
/// sequence, and none of them can move:
///
/// 1. **CR4.PAE** first. Long mode is built on PAE's 8-byte entries, and the
///    CPU refuses to enable it without.
/// 2. **CR3** pointing at a PML4 that already maps the code about to run.
///    Paging comes on in the next step, and the instruction after it is
///    fetched through these tables -- so an identity map is not a nicety, it
///    is what stops the CPU from fetching its next instruction from nowhere.
/// 3. **EFER.LME**, which *asks* for long mode without entering it.
/// 4. **CR0.PG**, which enters it: the hardware sets EFER.LMA in response.
///
/// Real firmware then far-jumps into a 64-bit code segment, because it is
/// running 32-bit code when it does this. Here the segment is simply loaded
/// with L set, which is the same state that far jump produces.
pub fn enter_long_mode(cpu: &mut Cpu, entry: u64, stack: u64) {
    write_gdt64(&mut cpu.mem);
    let pml4 = build_identity_map(&mut cpu.mem);

    cpu.gdt_base = GDT64_ADDR as u64;
    cpu.gdt_limit = 23;
    cpu.cr4 |= crate::cpu::CR4_PAE;
    cpu.cr3 = pml4;
    cpu.efer |= crate::cpu::efer::LME | crate::cpu::efer::NXE | crate::cpu::efer::SCE;
    cpu.pe = true;
    cpu.cr0 |= crate::cpu::CR0_PE | crate::cpu::CR0_PG | crate::cpu::CR0_WP;
    cpu.update_long_mode();
    cpu.flush_tlb();

    cpu.load_seg(SegReg::Cs, KERNEL_CS64);
    cpu.load_seg(SegReg::Ds, KERNEL_DS64);
    cpu.load_seg(SegReg::Es, KERNEL_DS64);
    cpu.load_seg(SegReg::Ss, KERNEL_DS64);
    cpu.load_seg(SegReg::Fs, KERNEL_DS64);
    cpu.load_seg(SegReg::Gs, KERNEL_DS64);
    cpu.set_rsp(stack);
    cpu.rip = entry;
    cpu.set_flag(crate::cpu::flags::IF, false);
    cpu.halted = false;
    cpu.invalidate_phys_ip();
}

/// Load a 64-bit ELF kernel and enter it in long mode.
///
/// The ELF64 header is not the 32-bit one with wider fields in the same
/// places: `e_entry` moves from offset 24 to 24 but grows to eight bytes,
/// `e_phoff` moves from 28 to 32, and every program-header field is at a
/// different offset. Reading an ELF64 with the ELF32 offsets yields
/// plausible-looking nonsense rather than an error, so the class byte is
/// checked first.
pub fn load_elf64_kernel(
    cpu: &mut Cpu, elf: &[u8], cmdline: &str, initrd: Option<&[u8]>,
) -> Result<u64, String> {
    if elf.len() < 64 || &elf[0..4] != b"\x7fELF" {
        return Err("not an ELF file".into());
    }
    if elf[4] != 2 {
        return Err("not a 64-bit ELF (class is ELF32)".into());
    }
    let e_entry = le64(elf, 24);
    let e_phoff = le64(elf, 32) as usize;
    let e_phentsize = le16(elf, 54) as usize;
    let e_phnum = le16(elf, 56) as usize;
    if e_phnum == 0 || e_phentsize < 56 {
        return Err("ELF has no program headers".into());
    }
    let mut top = 0u64;
    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        if off + 56 > elf.len() {
            return Err("program header table runs past the end of the file".into());
        }
        let p_type = le32(elf, off);
        let p_offset = le64(elf, off + 8) as usize;
        let p_paddr = le64(elf, off + 24) as usize;
        let p_filesz = le64(elf, off + 32) as usize;
        let p_memsz = le64(elf, off + 40) as usize;
        if p_type != 1 {
            continue;
        }
        if p_offset + p_filesz > elf.len() {
            return Err("PT_LOAD segment runs past the end of the file".into());
        }
        for j in 0..p_filesz {
            cpu.mem.write_u8(p_paddr + j, elf[p_offset + j]);
        }
        for j in p_filesz..p_memsz {
            cpu.mem.write_u8(p_paddr + j, 0);
        }
        top = top.max((p_paddr + p_memsz) as u64);
    }

    // The same boot structures a 32-bit kernel gets: a described console, the
    // machine's own memory map, and the command line. A 64-bit kernel reads
    // them from the same `boot_params`, at the same offsets.
    write_screen_info(&mut cpu.mem);
    write_e820(&mut cpu.mem);
    let cl = cmdline.as_bytes();
    let cl_len = cl.len().min(1023);
    for i in 0..cl_len {
        cpu.mem.write_u8(CMDLINE_ADDR as usize + i, cl[i]);
    }
    cpu.mem.write_u8(CMDLINE_ADDR as usize + cl_len, 0);
    let hdr = BOOT_PARAMS_ADDR as usize + BP_HDR;
    cpu.mem.write_u32(hdr + (CMD_LINE_PTR - SETUP_SECTS), CMDLINE_ADDR);
    cpu.mem.write_u8(hdr + (TYPE_OF_LOADER - SETUP_SECTS), 0xFF);
    if let Some(rd) = initrd {
        let limit = cpu.mem.ram_size().min(crate::memory::MMIO_HOLE_START);
        let addr = ((limit as usize - rd.len()) & !0xFFF) as u32;
        for (i, b) in rd.iter().enumerate() {
            cpu.mem.write_u8(addr as usize + i, *b);
        }
        cpu.mem.write_u32(hdr + (RAMDISK_IMAGE - SETUP_SECTS), addr);
        cpu.mem.write_u32(hdr + (RAMDISK_SIZE - SETUP_SECTS), rd.len() as u32);
    }

    // A stack above the loaded image, page-aligned and clear of the boot
    // structures below 1 MiB.
    let stack = ((top.max(0x10_0000) + 0x10_0000) & !0xFFF).min(IDENTITY_MAP_BYTES - 0x1000);
    enter_long_mode(cpu, e_entry, stack);
    // RSI holds boot_params, as the 64-bit boot protocol specifies.
    cpu.set_reg64_raw(6, BOOT_PARAMS_ADDR as u64);
    Ok(e_entry)
}

/// Load a flat 64-bit binary at `addr` and run it in long mode.
///
/// The 64-bit counterpart of the `--boot` path: no ELF, no boot protocol,
/// just bytes at an address with the machine already in long mode. It is the
/// smallest thing that can demonstrate a 64-bit CPU.
pub fn load_flat64(cpu: &mut Cpu, code: &[u8], addr: u64) -> Result<u64, String> {
    for (i, b) in code.iter().enumerate() {
        cpu.mem.write_u8(addr as usize + i, *b);
    }
    let stack = ((addr + code.len() as u64 + 0x1_0000) & !0xFFF)
        .min(IDENTITY_MAP_BYTES - 0x1000);
    enter_long_mode(cpu, addr, stack);
    Ok(addr)
}

/// Load a decompressed kernel ELF into the emulator and enter its entry point.
///
/// This is the path a bootloader uses for an *uncompressed* kernel: parse the
/// ELF program headers, load each PT_LOAD segment at its physical address, and
/// jump to the entry point. It lets us boot a kernel without running the
/// in-kernel decompressor (which our emulator does not yet execute correctly).
pub fn load_elf_kernel(cpu: &mut Cpu, elf: &[u8], cmdline: &str) -> Result<u32, String> {
    load_elf_kernel_with_initrd(cpu, elf, cmdline, None)
}

/// As `load_elf_kernel`, but also loads an initial ramdisk. The kernel's
/// `rd_load_image` copies it into /dev/ram0, which `root=/dev/ram0` then
/// mounts -- the path an ext2 root image on this ISO expects.
pub fn load_elf_kernel_with_initrd(
    cpu: &mut Cpu, elf: &[u8], cmdline: &str, initrd: Option<&[u8]>,
) -> Result<u32, String> {
    if elf.len() < 52 || &elf[0..4] != b"\x7fELF" {
        return Err("not an ELF file".into());
    }
    let e_entry = le32(elf, 24);
    let e_phoff = le32(elf, 28) as usize;
    let e_phentsize = le16(elf, 42) as usize;
    let e_phnum = le16(elf, 44) as usize;
    if e_phnum == 0 || e_phentsize < 32 {
        return Err("ELF has no program headers".into());
    }
    let mem = &mut cpu.mem;
    for i in 0..e_phnum {
        let off = e_phoff + i * e_phentsize;
        let p_type = le32(elf, off);
        let p_offset = le32(elf, off + 4) as usize;
        let p_paddr = le32(elf, off + 12) as usize;
        let p_filesz = le32(elf, off + 16) as usize;
        let p_memsz = le32(elf, off + 20) as usize;
        if p_type == 1 {
            // PT_LOAD: copy file bytes, zero the rest of the segment.
            for j in 0..p_filesz {
                mem.write_u8(p_paddr + j, elf[p_offset + j]);
            }
            for j in p_filesz..p_memsz {
                mem.write_u8(p_paddr + j, 0);
            }
        }
    }
    // Build boot_params, GDT, and enter protected mode (same as load_kernel).
    setup_protected_mode_with_initrd(cpu, cmdline, initrd);
    cpu.set_eip(e_entry);
    cpu.halted = false;
    Ok(e_entry)
}

/// Shared protected-mode setup used by both loaders: build boot_params at
/// 0x90000, write the flat GDT, enable protected mode, load flat segments.
/// As `setup_protected_mode`, but also places an initial ramdisk in memory and
/// records it in `boot_params.hdr` so the kernel can find it.
fn setup_protected_mode_with_initrd(cpu: &mut Cpu, cmdline: &str, initrd: Option<&[u8]>) {
    if let Some(rd) = initrd {
        // Bootloaders load the ramdisk as high in low memory as it will go,
        // clear of the kernel image and of the bootmem allocations that
        // follow it. The kernel reserves the region from these two fields
        // before it allocates anything else.
        // Kept below the 32-bit MMIO hole and below 4 GiB: the 32-bit
        // kernel's `ramdisk_image` field is a u32, so a ramdisk placed above
        // that is a ramdisk the kernel cannot describe to itself.
        let top = cpu.mem.ram_size().min(crate::memory::MMIO_HOLE_START);
        let addr = ((top as usize - rd.len()) & !0xFFF) as u32;
        for (i, b) in rd.iter().enumerate() {
            cpu.mem.write_u8(addr as usize + i, *b);
        }
        let hdr = BOOT_PARAMS_ADDR as usize + BP_HDR;
        cpu.mem.write_u32(hdr + (RAMDISK_IMAGE - SETUP_SECTS), addr);
        cpu.mem.write_u32(hdr + (RAMDISK_SIZE - SETUP_SECTS), rd.len() as u32);
    }
    setup_protected_mode_inner(cpu, cmdline)
}

fn setup_protected_mode_inner(cpu: &mut Cpu, cmdline: &str) {
    let mem = &mut cpu.mem;
    // Describe the text console in boot_params.screen_info.
    write_screen_info(mem);
    // Write the E820 memory map.
    write_e820(mem);
    // Write the command line (NUL-terminated).
    let cl = cmdline.as_bytes();
    let cl_len = cl.len().min(1023);
    for i in 0..cl_len {
        mem.write_u8(CMDLINE_ADDR as usize + i, cl[i]);
    }
    mem.write_u8(CMDLINE_ADDR as usize + cl_len, 0);
    // cmd_line_ptr in boot_params.hdr.
    let hdr_dst = BOOT_PARAMS_ADDR as usize + BP_HDR;
    mem.write_u32(hdr_dst + (CMD_LINE_PTR - SETUP_SECTS), CMDLINE_ADDR);
    // type_of_loader: 0xFF is "undefined bootloader". It has to be non-zero
    // or setup_arch() ignores ramdisk_image entirely and the initrd is never
    // reserved, never unpacked, and never mentioned in the log.
    mem.write_u8(hdr_dst + (TYPE_OF_LOADER - SETUP_SECTS), 0xFF);
    // Set up the flat GDT.
    write_gdt(mem);
    // Enter protected mode.
    cpu.gdt_base = GDT_ADDR as u64;
    cpu.gdt_limit = 24;
    cpu.pe = true;
    cpu.cr0 |= 1;
    cpu.load_seg(SegReg::Cs, KERNEL_CS);
    cpu.load_seg(SegReg::Ds, KERNEL_DS);
    cpu.load_seg(SegReg::Es, KERNEL_DS);
    cpu.load_seg(SegReg::Ss, KERNEL_DS);
    cpu.load_seg(SegReg::Fs, KERNEL_DS);
    cpu.load_seg(SegReg::Gs, KERNEL_DS);
    cpu.set_esp(0x8FFF0);
    cpu.set_esi(BOOT_PARAMS_ADDR);
    cpu.set_flag(crate::cpu::flags::IF, false);
}

/// Load a bzImage and set up the CPU to enter the 32-bit kernel.
///
/// Returns the parsed `KernelInfo` on success. On return the CPU is in
/// protected mode with flat segments, ESI = `BOOT_PARAMS_ADDR`, EIP =
/// `code32_start`, and interrupts disabled — ready for `cpu.run(...)`.
pub fn load_kernel(cpu: &mut Cpu, image: &[u8], cmdline: &str) -> Result<KernelInfo, String> {
    let info = parse_bzimage(image)?;
    let mem = &mut cpu.mem;

    // Load the protected-mode kernel at code32_start.
    let load_addr = info.code32_start as usize;
    mem.load(
        load_addr,
        &image[info.kernel_offset..info.kernel_offset + info.kernel_len],
    );

    // Build boot_params at 0x90000. Copy the setup header into boot_params.hdr.
    let hdr_dst = BOOT_PARAMS_ADDR as usize + BP_HDR;
    for i in 0..0xEC {
        mem.write_u8(hdr_dst + i, image[SETUP_SECTS + i]);
    }
    // type_of_loader = 0xFF (unknown) at hdr + 0x1F.
    mem.write_u8(hdr_dst + (TYPE_OF_LOADER - SETUP_SECTS), 0xFF);
    // loadflags |= 0x01 (LOADED_HIGH) at hdr + 0x20.
    let lf = mem.read_u8(hdr_dst + (LOADFLAGS - SETUP_SECTS)) | 0x01;
    mem.write_u8(hdr_dst + (LOADFLAGS - SETUP_SECTS), lf);
    // code32_start at hdr + 0x23.
    mem.write_u32(hdr_dst + (CODE32_START - SETUP_SECTS), info.code32_start);
    // cmd_line_ptr at hdr + 0x37.
    mem.write_u32(hdr_dst + (CMD_LINE_PTR - SETUP_SECTS), CMDLINE_ADDR);
    // Write the command line (NUL-terminated).
    let cl = cmdline.as_bytes();
    let cl_len = cl.len().min(1023);
    for i in 0..cl_len {
        mem.write_u8(CMDLINE_ADDR as usize + i, cl[i]);
    }
    mem.write_u8(CMDLINE_ADDR as usize + cl_len, 0);

    // Write the E820 memory map.
    write_e820(mem);

    // Set up the flat GDT.
    write_gdt(mem);

    // Enter protected mode.
    cpu.gdt_base = GDT_ADDR as u64;
    cpu.gdt_limit = 24; // 3 descriptors * 8 bytes - 1
    cpu.pe = true;
    cpu.cr0 |= 1; // CR0.PE
    // Load flat segments (resolves the GDT descriptors).
    cpu.load_seg(SegReg::Cs, KERNEL_CS);
    cpu.load_seg(SegReg::Ds, KERNEL_DS);
    cpu.load_seg(SegReg::Es, KERNEL_DS);
    cpu.load_seg(SegReg::Ss, KERNEL_DS);
    cpu.load_seg(SegReg::Fs, KERNEL_DS);
    cpu.load_seg(SegReg::Gs, KERNEL_DS);
    // Set up a stack below boot_params.
    cpu.set_esp(0x8FFF0);
    // ESI = boot_params address (the kernel reads boot_params from ESI).
    cpu.set_esi(BOOT_PARAMS_ADDR);
    // Disable interrupts (the kernel sets up its own IDT before enabling them).
    cpu.set_flag(crate::cpu::flags::IF, false);
    // Jump to the kernel entry point.
    cpu.set_eip(info.code32_start);
    cpu.halted = false;

    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::Cpu;

    /// Build a minimal synthetic bzImage with a valid setup header using the
    /// real boot-protocol offsets.
    fn make_bzimage(setup_sects: usize, syssize: u32, code32_start: u32) -> Vec<u8> {
        let kernel_len = syssize as usize * 16;
        let total = (setup_sects + 1) * 512 + kernel_len;
        let mut img = vec![0u8; total];
        img[SETUP_SECTS] = setup_sects as u8;
        img[SYSSIZE..SYSSIZE + 4].copy_from_slice(&syssize.to_le_bytes());
        img[BOOT_FLAG] = 0x55;
        img[BOOT_FLAG + 1] = 0xAA;
        img[HDR_MAGIC..HDR_MAGIC + 4].copy_from_slice(&0x53726448u32.to_le_bytes());
        img[CODE32_START..CODE32_START + 4].copy_from_slice(&code32_start.to_le_bytes());
        img[LOADFLAGS] = 0x01;
        // Put a marker in the kernel payload so we can verify it was loaded.
        img[(setup_sects + 1) * 512] = 0xAA;
        img[(setup_sects + 1) * 512 + 1] = 0xBB;
        img
    }

    #[test]
    fn parse_bzimage_parses_setup_header() {
        let img = make_bzimage(4, 256, 0x100000); // 4 setup sects, 4 KiB kernel
        let info = parse_bzimage(&img).unwrap();
        assert_eq!(info.setup_sects, 4);
        assert_eq!(info.syssize, 256);
        assert_eq!(info.code32_start, 0x100000);
        assert_eq!(info.kernel_offset, (4 + 1) * 512);
        assert_eq!(info.kernel_len, 256 * 16);
        assert!(!info.relocatable);
    }

    #[test]
    fn parse_bzimage_rejects_bad_magic() {
        let mut img = make_bzimage(4, 256, 0x100000);
        img[HDR_MAGIC] = 0x00; // corrupt "HdrS"
        assert!(parse_bzimage(&img).is_err());
    }

    #[test]
    fn load_kernel_sets_up_boot_params() {
        let mut cpu = Cpu::new();
        let img = make_bzimage(4, 256, 0x100000);
        let info = load_kernel(&mut cpu, &img, "console=tty0").unwrap();
        assert_eq!(info.code32_start, 0x100000);
        // Kernel loaded at code32_start.
        assert_eq!(cpu.mem.read_u8(0x100000), 0xAA);
        assert_eq!(cpu.mem.read_u8(0x100001), 0xBB);
        // boot_params.hdr at 0x90000 + 0x1F1: type_of_loader = 0xFF.
        assert_eq!(cpu.mem.read_u8(0x90000 + BP_HDR + (TYPE_OF_LOADER - SETUP_SECTS)), 0xFF);
        // loadflags has LOADED_HIGH (0x01).
        assert_eq!(
            cpu.mem.read_u8(0x90000 + BP_HDR + (LOADFLAGS - SETUP_SECTS)) & 0x01,
            0x01
        );
        // code32_start in boot_params.
        assert_eq!(
            cpu.mem.read_u32(0x90000 + BP_HDR + (CODE32_START - SETUP_SECTS)),
            0x100000
        );
        // cmd_line_ptr set.
        assert_eq!(
            cpu.mem.read_u32(0x90000 + BP_HDR + (CMD_LINE_PTR - SETUP_SECTS)),
            CMDLINE_ADDR
        );
        // Command line written.
        let mut s = String::new();
        let mut i = 0;
        loop {
            let c = cpu.mem.read_u8(CMDLINE_ADDR as usize + i);
            if c == 0 {
                break;
            }
            s.push(c as char);
            i += 1;
        }
        assert_eq!(s, "console=tty0");
        // E820 map: the machine's own, entry for entry.
        let expected = cpu.mem.e820();
        assert_eq!(cpu.mem.read_u8(0x90000 + BP_E820_ENTRIES) as usize, expected.len());
        for (i, (base, len, typ)) in expected.iter().enumerate() {
            let e = 0x90000 + BP_E820_TABLE + i * 20;
            assert_eq!(cpu.mem.read_u64(e), *base);
            assert_eq!(cpu.mem.read_u64(e + 8), *len);
            assert_eq!(cpu.mem.read_u32(e + 16), *typ);
        }
    }

    /// Build a minimal but well-formed ELF64 with one PT_LOAD segment.
    fn make_elf64(entry: u64, paddr: u64, body: &[u8]) -> Vec<u8> {
        let ph_off = 64usize;
        let body_off = ph_off + 56;
        let mut e = vec![0u8; body_off + body.len()];
        e[0..4].copy_from_slice(b"\x7fELF");
        e[4] = 2; // ELFCLASS64
        e[5] = 1; // little-endian
        e[6] = 1; // version
        e[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
        e[18..20].copy_from_slice(&0x3Eu16.to_le_bytes()); // EM_X86_64
        e[20..24].copy_from_slice(&1u32.to_le_bytes());
        e[24..32].copy_from_slice(&entry.to_le_bytes());
        e[32..40].copy_from_slice(&(ph_off as u64).to_le_bytes());
        e[52..54].copy_from_slice(&64u16.to_le_bytes()); // e_ehsize
        e[54..56].copy_from_slice(&56u16.to_le_bytes()); // e_phentsize
        e[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum
        let ph = ph_off;
        e[ph..ph + 4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
        e[ph + 4..ph + 8].copy_from_slice(&5u32.to_le_bytes()); // R+X
        e[ph + 8..ph + 16].copy_from_slice(&(body_off as u64).to_le_bytes());
        e[ph + 16..ph + 24].copy_from_slice(&paddr.to_le_bytes()); // p_vaddr
        e[ph + 24..ph + 32].copy_from_slice(&paddr.to_le_bytes()); // p_paddr
        e[ph + 32..ph + 40].copy_from_slice(&(body.len() as u64).to_le_bytes());
        // A memsz larger than filesz, so the zero-fill is exercised too.
        e[ph + 40..ph + 48].copy_from_slice(&((body.len() + 16) as u64).to_le_bytes());
        e[body_off..].copy_from_slice(body);
        e
    }

    #[test]
    fn load_elf64_kernel_loads_segments_and_enters_long_mode() {
        let mut cpu = Cpu::new();
        // A body that ends in HLT, loaded at 1 MiB.
        let body = [0x48, 0x31, 0xC0, 0xF4]; // xor %rax,%rax ; hlt
        let elf = make_elf64(0x10_0000, 0x10_0000, &body);
        let entry = load_elf64_kernel(&mut cpu, &elf, "console=tty0", None).unwrap();
        assert_eq!(entry, 0x10_0000);

        // The segment landed at its physical address, and the memsz tail was
        // zeroed rather than left as whatever was there.
        for (i, b) in body.iter().enumerate() {
            assert_eq!(cpu.mem.read_u8(0x10_0000 + i), *b);
        }
        assert_eq!(cpu.mem.read_u8(0x10_0000 + body.len()), 0);

        // The machine is in 64-bit long mode, with RSI on boot_params as the
        // 64-bit boot protocol specifies.
        assert_eq!(cpu.mode(), crate::cpu::Mode::Long);
        assert_eq!(cpu.rip, 0x10_0000);
        assert_eq!(cpu.reg64(6), BOOT_PARAMS_ADDR as u64);
        assert!(!cpu.get_flag(crate::cpu::flags::IF));

        // ...and it actually runs.
        cpu.run(16);
        assert!(cpu.halted && !cpu.triple_fault);
    }

    #[test]
    fn load_elf64_kernel_refuses_a_32_bit_elf() {
        // The two headers differ field by field, and reading an ELF32 with
        // ELF64 offsets yields plausible nonsense rather than an error -- so
        // the class byte is checked before anything else is believed.
        let mut cpu = Cpu::new();
        let mut elf = make_elf64(0x10_0000, 0x10_0000, &[0xF4]);
        elf[4] = 1; // ELFCLASS32
        let err = load_elf64_kernel(&mut cpu, &elf, "", None).unwrap_err();
        assert!(err.contains("32"), "unhelpful error: {}", err);
    }

    #[test]
    fn load_elf64_kernel_refuses_a_segment_that_runs_off_the_end() {
        let mut cpu = Cpu::new();
        let mut elf = make_elf64(0x10_0000, 0x10_0000, &[0xF4]);
        // Claim a filesz far larger than the file.
        let ph = 64;
        elf[ph + 32..ph + 40].copy_from_slice(&0x10_0000u64.to_le_bytes());
        assert!(load_elf64_kernel(&mut cpu, &elf, "", None).is_err());
    }

    #[test]
    fn the_identity_map_covers_the_low_four_gib_with_large_pages() {
        let mut cpu = Cpu::new();
        enter_long_mode(&mut cpu, 0x10_0000, 0x20_0000);
        // Every one of these translates to itself, through 2 MiB pages.
        for linear in [0u64, 0xB8000, 0x10_0000, 0x8000_0000, 0xFFFF_F000] {
            let phys = cpu.apply_paging(linear);
            assert!(cpu.pending_exception.is_none(), "{:X} did not translate", linear);
            assert_eq!(phys as u64, linear, "identity map at {:X}", linear);
        }
        // Past the mapped region there is nothing, and that is a page fault
        // rather than a silent alias.
        cpu.pending_exception = None;
        let _ = cpu.apply_paging(0x1_0000_0000);
        assert_eq!(cpu.pending_exception.unwrap().0, 0x0E);
    }

    #[test]
    fn the_boot_map_a_big_machine_reports_matches_the_machine() {
        // The BIOS map and boot_params both come from `Memory::e820`, so a
        // machine cannot describe itself two different ways -- which is the
        // failure mode that has a guest using RAM that is not there.
        let mut cpu = Cpu::with_ram(crate::memory::MMIO_HOLE_START as usize + (128 << 20));
        let elf = make_elf64(0x10_0000, 0x10_0000, &[0xF4]);
        load_elf64_kernel(&mut cpu, &elf, "", None).unwrap();
        let expected = cpu.mem.e820();
        assert_eq!(expected.len(), 6, "hole split: low, EBDA, VGA, low RAM, hole, high RAM");
        let n = cpu.mem.read_u8(BOOT_PARAMS_ADDR as usize + BP_E820_ENTRIES) as usize;
        assert_eq!(n, expected.len());
        for (i, (base, len, typ)) in expected.iter().enumerate() {
            let e = BOOT_PARAMS_ADDR as usize + BP_E820_TABLE + i * 20;
            assert_eq!(cpu.mem.read_u64(e), *base);
            assert_eq!(cpu.mem.read_u64(e + 8), *len);
            assert_eq!(cpu.mem.read_u32(e + 16), *typ);
        }
        // alt_mem_k describes only what is below the hole: it is a 32-bit
        // count of kilobytes and cannot say anything about the rest.
        let alt = cpu.mem.read_u32(BOOT_PARAMS_ADDR as usize + BP_ALT_MEM_K) as u64;
        assert_eq!(alt, (crate::memory::MMIO_HOLE_START - 0x10_0000) / 1024);
    }

    #[test]
    fn load_kernel_enters_protected_mode() {
        let mut cpu = Cpu::new();
        let img = make_bzimage(4, 256, 0x100000);
        load_kernel(&mut cpu, &img, "").unwrap();
        assert!(cpu.pe);
        assert_eq!(cpu.cr0 & 1, 1); // PE
        assert_eq!(cpu.eip(), 0x100000);
        assert_eq!(cpu.esi(), BOOT_PARAMS_ADDR);
        assert_eq!(cpu.cs, KERNEL_CS);
        assert_eq!(cpu.ds, KERNEL_DS);
        // Flat code segment: base 0.
        assert_eq!(cpu.seg_desc[SegReg::Cs as usize].base, 0);
        // Interrupts disabled.
        assert!(!cpu.get_flag(crate::cpu::flags::IF));
    }
}
