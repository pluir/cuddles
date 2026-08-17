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
const CMD_LINE_PTR: usize = 0x228;
const KERNEL_ALIGNMENT: usize = 0x22E;
const RELOCATABLE: usize = 0x232;
const PREF_ADDRESS: usize = 0x250;
const INIT_SIZE: usize = 0x258;

/// Offsets within `struct boot_params` (relative to `BOOT_PARAMS_ADDR`).
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

/// Write the E820 memory map into `boot_params`.
fn write_e820(mem: &mut Memory) {
    let map = BOOT_PARAMS_ADDR as usize + BP_E820_TABLE;
    // Conventional memory 0-640K (type 1, usable).
    mem.write_u64(map, 0);
    mem.write_u64(map + 8, 0xA0000);
    mem.write_u32(map + 16, 1);
    // EBDA / VGA area 0xA0000-0x100000 (type 2, reserved).
    mem.write_u64(map + 20, 0xA0000);
    mem.write_u64(map + 28, 0x60000);
    mem.write_u32(map + 36, 2);
    // Extended memory 0x100000 to Memory::SIZE (type 1, usable).
    mem.write_u64(map + 40, 0x100000);
    mem.write_u64(map + 48, (Memory::SIZE as u64) - 0x100000);
    mem.write_u32(map + 56, 1);
    mem.write_u8(BOOT_PARAMS_ADDR as usize + BP_E820_ENTRIES, 3);
    // alt_mem_k (extended memory in KB above 1 MiB), derived from Memory::SIZE.
    let ext_kb = ((Memory::SIZE as u64) - 0x100000) / 1024;
    mem.write_u32(BOOT_PARAMS_ADDR as usize + BP_ALT_MEM_K, ext_kb as u32);
}

/// Load a decompressed kernel ELF into the emulator and enter its entry point.
///
/// This is the path a bootloader uses for an *uncompressed* kernel: parse the
/// ELF program headers, load each PT_LOAD segment at its physical address, and
/// jump to the entry point. It lets us boot a kernel without running the
/// in-kernel decompressor (which our emulator does not yet execute correctly).
pub fn load_elf_kernel(cpu: &mut Cpu, elf: &[u8], cmdline: &str) -> Result<u32, String> {
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
    setup_protected_mode(cpu, cmdline);
    cpu.eip = e_entry;
    cpu.halted = false;
    Ok(e_entry)
}

/// Shared protected-mode setup used by both loaders: build boot_params at
/// 0x90000, write the flat GDT, enable protected mode, load flat segments.
fn setup_protected_mode(cpu: &mut Cpu, cmdline: &str) {
    let mem = &mut cpu.mem;
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
    // Set up the flat GDT.
    write_gdt(mem);
    // Enter protected mode.
    cpu.gdt_base = GDT_ADDR;
    cpu.gdt_limit = 24;
    cpu.pe = true;
    cpu.cr0 |= 1;
    cpu.load_seg(SegReg::Cs, KERNEL_CS);
    cpu.load_seg(SegReg::Ds, KERNEL_DS);
    cpu.load_seg(SegReg::Es, KERNEL_DS);
    cpu.load_seg(SegReg::Ss, KERNEL_DS);
    cpu.load_seg(SegReg::Fs, KERNEL_DS);
    cpu.load_seg(SegReg::Gs, KERNEL_DS);
    cpu.esp = 0x8FFF0;
    cpu.esi = BOOT_PARAMS_ADDR;
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
    cpu.gdt_base = GDT_ADDR;
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
    cpu.esp = 0x8FFF0;
    // ESI = boot_params address (the kernel reads boot_params from ESI).
    cpu.esi = BOOT_PARAMS_ADDR;
    // Disable interrupts (the kernel sets up its own IDT before enabling them).
    cpu.set_flag(crate::cpu::flags::IF, false);
    // Jump to the kernel entry point.
    cpu.eip = info.code32_start;
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
        // E820 map: 3 entries.
        assert_eq!(cpu.mem.read_u8(0x90000 + BP_E820_ENTRIES), 3);
        // First entry: base 0, size 0xA0000, type 1.
        assert_eq!(cpu.mem.read_u64(0x90000 + BP_E820_TABLE), 0);
        assert_eq!(cpu.mem.read_u64(0x90000 + BP_E820_TABLE + 8), 0xA0000);
        assert_eq!(cpu.mem.read_u32(0x90000 + BP_E820_TABLE + 16), 1);
    }

    #[test]
    fn load_kernel_enters_protected_mode() {
        let mut cpu = Cpu::new();
        let img = make_bzimage(4, 256, 0x100000);
        load_kernel(&mut cpu, &img, "").unwrap();
        assert!(cpu.pe);
        assert_eq!(cpu.cr0 & 1, 1); // PE
        assert_eq!(cpu.eip, 0x100000);
        assert_eq!(cpu.esi, BOOT_PARAMS_ADDR);
        assert_eq!(cpu.cs, KERNEL_CS);
        assert_eq!(cpu.ds, KERNEL_DS);
        // Flat code segment: base 0.
        assert_eq!(cpu.seg_desc[SegReg::Cs as usize].base, 0);
        // Interrupts disabled.
        assert!(!cpu.get_flag(crate::cpu::flags::IF));
    }
}
