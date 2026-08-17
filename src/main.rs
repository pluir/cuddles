//! x86emu binary: load a flat binary at a segment:offset and run it.
//!
//! Usage:
//!   x86emu <file> [segment:offset] [max_instructions]
//!   x86emu --boot <bootsector.bin> [max_instructions]
//!
//! Defaults: load at 0000:0x0100 (like a DOS .COM), run up to 100000
//! instructions or until HLT. Prints the final register state.
//! `--boot` loads a 512-byte boot sector at 0000:0x7C00 (like a real PC)
//! and prints the emulated text screen afterwards.

use std::env;
use std::process::ExitCode;

use x86emu::Cpu;
use x86emu::memory::Memory;

fn parse_addr(s: &str) -> Option<(u16, u16)> {
    let (seg, off) = s.split_once(':')?;
    Some((u16::from_str_radix(seg, 16).ok()?, u16::from_str_radix(off, 16).ok()?))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: x86emu <file> [segment:offset] [max_instructions]");
        eprintln!("       x86emu --boot <bootsector.bin> [max_instructions]");
        return ExitCode::from(2);
    }

    if args[1] == "--boot" {
        return boot(&args[2..]);
    }
    if args[1] == "--kernel" {
        return kernel(&args[2..]);
    }
    if args[1] == "--kernel-elf" {
        return kernel_elf(&args[2..]);
    }
    if args[1] == "--dump" {
        return dump(&args[2..]);
    }

    let path = &args[1];
    let (segment, offset) = args.get(2)
        .and_then(|s| parse_addr(s))
        .unwrap_or((0x0000, 0x0100));
    let max = args.get(3)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(100_000);

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {}: {}", path, e);
            return ExitCode::from(1);
        }
    };

    let mut cpu = Cpu::new();
    cpu.cs = segment;
    cpu.ip = offset;
    cpu.ds = segment;
    cpu.es = segment;
    cpu.ss = 0x0000;
    cpu.sp = 0xFFFE; // stack grows down from the top of the segment

    cpu.mem.load(Memory::phys(segment, offset), &bytes);

    let ran = cpu.run(max);

    println!("executed {} instructions", ran);
    if cpu.halted {
        println!("halted");
    } else {
        println!("stopped (instruction limit reached)");
    }
    print_state(&cpu);
    ExitCode::from(0)
}

/// Boot a 512-byte boot sector at 0000:0x7C00 and print the text screen.
fn boot(args: &[String]) -> ExitCode {    if args.is_empty() {
        eprintln!("usage: x86emu --boot <bootsector.bin> [max_instructions]");
        return ExitCode::from(2);
    }
    let path = &args[0];
    let max = args.get(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(100_000);

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {}: {}", path, e);
            return ExitCode::from(1);
        }
    };
    if bytes.len() < 512 {
        eprintln!("error: boot sector must be at least 512 bytes (got {})", bytes.len());
        return ExitCode::from(1);
    }

    let mut cpu = Cpu::new();
    // Real PCs load the boot sector at 0000:0x7C00.
    cpu.cs = 0x0000;
    cpu.ip = 0x7C00;
    cpu.ds = 0x0000;
    cpu.es = 0x0000;
    cpu.ss = 0x0000;
    cpu.sp = 0xFFFE;
    cpu.mem.load(0x7C00, &bytes[..512]);

    let ran = cpu.run(max);

    println!("executed {} instructions", ran);
    if cpu.halted {
        println!("halted");
    } else {
        println!("stopped (instruction limit reached)");
    }
    print_state(&cpu);
    println!();
    println!("--- text screen ---");
    for row in 0..x86emu::bios::SCREEN_ROWS {
        let mut line = String::new();
        for col in 0..x86emu::bios::SCREEN_COLS {
            let cell = cpu.mem.vga_text[row * x86emu::bios::SCREEN_COLS + col];
            let ch = (cell & 0xFF) as u8;
            line.push(if ch == 0 { ' ' } else { ch as char });
        }
        println!("{}", line.trim_end());
    }
    ExitCode::from(0)
}

/// Load a Linux bzImage via the boot protocol and run the 32-bit kernel.
fn kernel(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: x86emu --kernel <bzImage> [max_instructions]");
        return ExitCode::from(2);
    }
    let path = &args[0];
    let max = args.get(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5_000_000);

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {}: {}", path, e);
            return ExitCode::from(1);
        }
    };

    let mut cpu = Cpu::new();
    match x86emu::boot::load_kernel(&mut cpu, &bytes, "console=tty0") {
        Ok(info) => {
            println!("loaded bzImage: setup_sects={} syssize={} code32_start={:08X}",
                info.setup_sects, info.syssize, info.code32_start);
        }
        Err(e) => {
            eprintln!("error loading kernel: {}", e);
            return ExitCode::from(1);
        }
    }

    let ran = cpu.run(max);

    println!("executed {} instructions", ran);
    if cpu.halted {
        println!("halted");
    } else {
        println!("stopped (instruction limit reached)");
    }
    print_state(&cpu);
    println!();
    println!("--- text screen ---");
    for row in 0..x86emu::bios::SCREEN_ROWS {
        let mut line = String::new();
        for col in 0..x86emu::bios::SCREEN_COLS {
            let cell = cpu.mem.vga_text[row * x86emu::bios::SCREEN_COLS + col];
            let ch = (cell & 0xFF) as u8;
            line.push(if ch == 0 { ' ' } else { ch as char });
        }
        println!("{}", line.trim_end());
    }
    ExitCode::from(0)
}

/// Dump a region of emulated memory to a file. Used for debugging the
/// kernel decompressor: `x86emu --dump <bzImage> <outfile> <addr> <len> <max_instructions>`
/// loads the kernel, runs it, and writes `len` bytes from physical `addr` to `outfile`.
fn dump(args: &[String]) -> ExitCode {
    if args.len() < 5 {
        eprintln!("usage: x86emu --dump <bzImage> <outfile> <addr> <len> <max_instructions>");
        return ExitCode::from(2);
    }
    let path = &args[0];
    let outfile = &args[1];
    let addr = usize::from_str_radix(&args[2], 16).unwrap_or(0);
    let len = usize::from_str_radix(&args[3], 16).unwrap_or(0);
    let max = args[4].parse::<u64>().unwrap_or(5_000_000);

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {}: {}", path, e);
            return ExitCode::from(1);
        }
    };

    let mut cpu = Cpu::new();
    match x86emu::boot::load_kernel(&mut cpu, &bytes, "console=tty0") {
        Ok(info) => {
            println!("loaded bzImage: setup_sects={} syssize={} code32_start={:08X}",
                info.setup_sects, info.syssize, info.code32_start);
        }
        Err(e) => {
            eprintln!("error loading kernel: {}", e);
            return ExitCode::from(1);
        }
    }

    let ran = cpu.run(max);
    println!("executed {} instructions", ran);
    if cpu.halted {
        println!("halted");
    } else {
        println!("stopped (instruction limit reached)");
    }

    // Dump the requested region.
    let mut buf = Vec::with_capacity(len);
    for i in 0..len {
        buf.push(cpu.mem.read_u8(addr + i));
    }
    match std::fs::write(outfile, &buf) {
        Ok(_) => println!("dumped {} bytes from {:08X} to {}", len, addr, outfile),
        Err(e) => {
            eprintln!("error writing {}: {}", outfile, e);
            return ExitCode::from(1);
        }
    }
    ExitCode::from(0)
}

/// Load a decompressed kernel ELF and run it (bypasses the in-kernel
/// decompressor, which the emulator does not yet execute correctly).
fn kernel_elf(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: x86emu --kernel-elf <kernel.elf> [max_instructions]");
        return ExitCode::from(2);
    }
    let path = &args[0];
    let max = args.get(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5_000_000);

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {}: {}", path, e);
            return ExitCode::from(1);
        }
    };

    let mut cpu = Cpu::new();
    match x86emu::boot::load_elf_kernel(&mut cpu, &bytes, "console=tty0") {
        Ok(entry) => {
            println!("loaded kernel ELF, entry={:08X}", entry);
        }
        Err(e) => {
            eprintln!("error loading kernel ELF: {}", e);
            return ExitCode::from(1);
        }
    }

    let ran = cpu.run(max);

    println!("executed {} instructions", ran);
    if cpu.halted {
        println!("halted");
    } else {
        println!("stopped (instruction limit reached)");
    }
    print_state(&cpu);
    println!();
    println!("--- text screen ---");
    for row in 0..x86emu::bios::SCREEN_ROWS {
        let mut line = String::new();
        for col in 0..x86emu::bios::SCREEN_COLS {
            let cell = cpu.mem.vga_text[row * x86emu::bios::SCREEN_COLS + col];
            let ch = (cell & 0xFF) as u8;
            line.push(if ch == 0 { ' ' } else { ch as char });
        }
        println!("{}", line.trim_end());
    }
    ExitCode::from(0)
}

fn print_state(cpu: &Cpu) {
    println!("EAX={:08X} EBX={:08X} ECX={:08X} EDX={:08X}",
        cpu.eax, cpu.ebx, cpu.ecx, cpu.edx);
    println!("AX={:04X} BX={:04X} CX={:04X} DX={:04X}",
        cpu.ax, cpu.bx, cpu.cx, cpu.dx);
    println!("ESP={:08X} EBP={:08X} ESI={:08X} EDI={:08X}",
        cpu.esp, cpu.ebp, cpu.esi, cpu.edi);
    println!("SP={:04X} BP={:04X} SI={:04X} DI={:04X}",
        cpu.sp, cpu.bp, cpu.si, cpu.di);
    println!("CS={:04X} DS={:04X} ES={:04X} SS={:04X} FS={:04X} GS={:04X}",
        cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs);
    println!("EIP={:08X} IP={:04X} FLAGS={:04X}", cpu.eip, cpu.ip, cpu.flags);
    if cpu.pe {
        println!("protected mode: GDT base={:08X} limit={:04X}  IDT base={:08X} limit={:04X}",
            cpu.gdt_base, cpu.gdt_limit, cpu.idt_base, cpu.idt_limit);
    }
    let f = |on: bool, ch: char| if on { ch } else { '-' };
    print!("flags: ");
    print!("{} ", f(cpu.get_flag(x86emu::cpu::flags::CF), 'C'));
    print!("{} ", f(cpu.get_flag(x86emu::cpu::flags::PF), 'P'));
    print!("{} ", f(cpu.get_flag(x86emu::cpu::flags::AF), 'A'));
    print!("{} ", f(cpu.get_flag(x86emu::cpu::flags::ZF), 'Z'));
    print!("{} ", f(cpu.get_flag(x86emu::cpu::flags::SF), 'S'));
    print!("{} ", f(cpu.get_flag(x86emu::cpu::flags::TF), 'T'));
    print!("{} ", f(cpu.get_flag(x86emu::cpu::flags::IF), 'I'));
    print!("{} ", f(cpu.get_flag(x86emu::cpu::flags::DF), 'D'));
    println!("{} ", f(cpu.get_flag(x86emu::cpu::flags::OF), 'O'));
}