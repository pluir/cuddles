//! x86emu binary: load a flat binary at a segment:offset and run it.
//!
//! Usage:
//!   x86emu [--mem SIZE] <file> [segment:offset] [max_instructions]
//!   x86emu [--mem SIZE] --boot <bootsector.bin> [max_instructions]
//!   x86emu [--mem SIZE] --kernel-elf <kernel.elf> [--initrd f] [--cmdline s] [n]
//!   x86emu [--mem SIZE] --kernel-elf64 <kernel.elf> [--initrd f] [--cmdline s] [n]
//!   x86emu [--mem SIZE] --long <flat64.bin> [load_addr_hex] [max_instructions]
//!
//! Defaults: load at 0000:0x0100 (like a DOS .COM), run up to 100000
//! instructions or until HLT. Prints the final register state.
//! `--boot` loads a 512-byte boot sector at 0000:0x7C00 (like a real PC)
//! and prints the emulated text screen afterwards.
//!
//! `--mem` takes a size with an optional K/M/G suffix (a bare number is
//! MiB), so `--mem 8G` builds a machine with 8 GiB of RAM. It applies to
//! every mode, and is the only place the machine's size is chosen -- the
//! BIOS memory map and `boot_params` are both derived from it.

use std::env;
use std::process::ExitCode;

use x86emu::Cpu;
use x86emu::memory::Memory;

fn parse_addr(s: &str) -> Option<(u16, u16)> {
    let (seg, off) = s.split_once(':')?;
    Some((u16::from_str_radix(seg, 16).ok()?, u16::from_str_radix(off, 16).ok()?))
}

/// Parse a `--mem` size: a number with an optional K/M/G suffix. A bare
/// number is MiB, which is what everyone means by "give it 512".
fn parse_mem(s: &str) -> Option<usize> {
    let s = s.trim();
    let (digits, mult) = match s.chars().last()? {
        'k' | 'K' => (&s[..s.len() - 1], 1usize << 10),
        'm' | 'M' => (&s[..s.len() - 1], 1usize << 20),
        'g' | 'G' => (&s[..s.len() - 1], 1usize << 30),
        _ => (s, 1usize << 20),
    };
    let n: usize = digits.trim().parse().ok()?;
    n.checked_mul(mult)
}

/// Pull a `--mem SIZE` option out of the argument list, wherever it appears.
/// Returning the remaining arguments keeps every mode's own parsing unaware
/// of it.
fn take_mem(args: &[String]) -> (usize, Vec<String>) {
    let mut ram = Memory::DEFAULT_SIZE;
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--mem" {
            match args.get(i + 1).and_then(|v| parse_mem(v)) {
                Some(n) => ram = n,
                None => eprintln!("warning: --mem wants a size like 512M or 8G; ignoring"),
            }
            i += 2;
            continue;
        }
        rest.push(args[i].clone());
        i += 1;
    }
    (ram, rest)
}

fn main() -> ExitCode {
    let argv: Vec<String> = env::args().collect();
    let (ram, args) = take_mem(&argv);
    if args.len() < 2 {
        eprintln!("usage: x86emu [--mem SIZE] <file> [segment:offset] [max_instructions]");
        eprintln!("       x86emu [--mem SIZE] --boot <bootsector.bin> [max_instructions]");
        eprintln!("       x86emu [--mem SIZE] --kernel-elf <kernel.elf> [--initrd f] [--cmdline s] [n]");
        eprintln!("       x86emu [--mem SIZE] --kernel-elf64 <kernel.elf> [--initrd f] [--cmdline s] [n]");
        eprintln!("       x86emu [--mem SIZE] --long <flat64.bin> [load_addr_hex] [max_instructions]");
        return ExitCode::from(2);
    }

    if args[1] == "--boot" {
        return boot(ram, &args[2..]);
    }
    if args[1] == "--kernel" {
        return kernel(ram, &args[2..]);
    }
    if args[1] == "--kernel-elf" {
        return kernel_elf(ram, &args[2..], false);
    }
    if args[1] == "--kernel-elf64" {
        return kernel_elf(ram, &args[2..], true);
    }
    if args[1] == "--long" {
        return long_flat(ram, &args[2..]);
    }
    if args[1] == "--dump" {
        return dump(ram, &args[2..]);
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

    let mut cpu = Cpu::with_ram(ram);
    cpu.cs = segment;
    cpu.ip = offset;
    cpu.ds = segment;
    cpu.es = segment;
    cpu.ss = 0x0000;
    cpu.set_sp(0xFFFE); // stack grows down from the top of the segment

    cpu.mem.load(Memory::phys(segment, offset), &bytes);

    let ran = cpu.run(max);

    println!("executed {} instructions", ran);
    if cpu.halted {
        println!("halted");
    } else {
        println!("stopped (instruction limit reached)");
    }
    print_state(&cpu);
    print_debug(&mut cpu);
    ExitCode::from(0)
}

/// Boot a 512-byte boot sector at 0000:0x7C00 and print the text screen.
fn boot(ram: usize, args: &[String]) -> ExitCode {
    if args.is_empty() {
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

    let mut cpu = Cpu::with_ram(ram);
    // Real PCs load the boot sector at 0000:0x7C00.
    cpu.cs = 0x0000;
    cpu.ip = 0x7C00;
    cpu.ds = 0x0000;
    cpu.es = 0x0000;
    cpu.ss = 0x0000;
    cpu.set_sp(0xFFFE);
    cpu.mem.load(0x7C00, &bytes[..512]);

    let ran = cpu.run(max);

    println!("executed {} instructions", ran);
    if cpu.halted {
        println!("halted");
    } else {
        println!("stopped (instruction limit reached)");
    }
    print_state(&cpu);
    print_debug(&mut cpu);
    print_screen(&cpu);
    ExitCode::from(0)
}

/// Load a Linux bzImage via the boot protocol and run the 32-bit kernel.
fn kernel(ram: usize, args: &[String]) -> ExitCode {
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

    let mut cpu = Cpu::with_ram(ram);
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
    print_debug(&mut cpu);
    print_screen(&cpu);
    ExitCode::from(0)
}

/// Dump a region of emulated memory to a file. Used for debugging the
/// kernel decompressor: `x86emu --dump <bzImage> <outfile> <addr> <len> <max_instructions>`
/// loads the kernel, runs it, and writes `len` bytes from physical `addr` to `outfile`.
fn dump(ram: usize, args: &[String]) -> ExitCode {
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

    let mut cpu = Cpu::with_ram(ram);
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
///
/// Usage: --kernel-elf <kernel.elf> [--initrd <file>] [--cmdline <str>]
///        [max_instructions]
fn kernel_elf(ram: usize, args: &[String], long_mode: bool) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: x86emu --kernel-elf[64] <kernel.elf> [--initrd <file>] \
                   [--cmdline <str>] [max_instructions]");
        return ExitCode::from(2);
    }
    let path = &args[0];
    let mut initrd_path: Option<&String> = None;
    let mut cmdline = String::from("console=tty0");
    let mut max: u64 = 5_000_000;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--initrd" => { initrd_path = args.get(i + 1); i += 2; }
            "--cmdline" => {
                if let Some(c) = args.get(i + 1) { cmdline = c.clone(); }
                i += 2;
            }
            other => {
                if let Ok(n) = other.parse::<u64>() { max = n; }
                i += 1;
            }
        }
    }

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {}: {}", path, e);
            return ExitCode::from(1);
        }
    };
    let initrd = match initrd_path {
        Some(p) => match std::fs::read(p) {
            Ok(b) => { println!("loaded initrd {} ({} bytes)", p, b.len()); Some(b) }
            Err(e) => {
                eprintln!("error reading {}: {}", p, e);
                return ExitCode::from(1);
            }
        },
        None => None,
    };

    let mut cpu = Cpu::with_ram(ram);
    println!("machine: {} MiB of RAM", cpu.mem.ram_size() >> 20);
    let loaded = if long_mode {
        x86emu::boot::load_elf64_kernel(&mut cpu, &bytes, &cmdline, initrd.as_deref())
    } else {
        x86emu::boot::load_elf_kernel_with_initrd(&mut cpu, &bytes, &cmdline, initrd.as_deref())
            .map(|e| e as u64)
    };
    match loaded {
        Ok(entry) => {
            println!("loaded {} kernel ELF, entry={:016X}  cmdline={:?}",
                if long_mode { "64-bit" } else { "32-bit" }, entry, cmdline);
        }
        Err(e) => {
            eprintln!("error loading kernel ELF: {}", e);
            return ExitCode::from(1);
        }
    }

    let ran = cpu.run(max);

    println!("executed {} instructions", ran);
    if cpu.halted {
        if cpu.triple_fault {
            println!("halted (TRIPLE FAULT: exception fired with no IDT installed)");
        } else {
            println!("halted");
        }
    } else {
        println!("stopped (instruction limit reached)");
    }
    print_state(&cpu);
    print_debug(&mut cpu);
    print_screen(&cpu);
    ExitCode::from(0)
}

/// Run a flat 64-bit binary in long mode.
///
/// The 64-bit counterpart of `--boot`: no ELF and no boot protocol, just
/// bytes loaded at an address with the machine already in long mode, an
/// identity-mapped low 4 GiB and a stack. The smallest thing that
/// demonstrates a 64-bit CPU end to end.
fn long_flat(ram: usize, args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: x86emu --long <flat64.bin> [load_addr_hex] [max_instructions]");
        return ExitCode::from(2);
    }
    let path = &args[0];
    let addr = args.get(1)
        .and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x10_0000);
    let max = args.get(2)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1_000_000);

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {}: {}", path, e);
            return ExitCode::from(1);
        }
    };

    let mut cpu = Cpu::with_ram(ram);
    println!("machine: {} MiB of RAM", cpu.mem.ram_size() >> 20);
    match x86emu::boot::load_flat64(&mut cpu, &bytes, addr) {
        Ok(entry) => println!("long mode: {} bytes at {:016X}, RSP={:016X}",
            bytes.len(), entry, cpu.rsp()),
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    }

    let ran = cpu.run(max);
    println!("executed {} instructions", ran);
    if cpu.halted {
        if cpu.triple_fault {
            println!("halted (TRIPLE FAULT)");
        } else {
            println!("halted");
        }
    } else {
        println!("stopped (instruction limit reached)");
    }
    print_state(&cpu);
    print_debug(&mut cpu);
    print_screen(&cpu);
    ExitCode::from(0)
}

/// Dump the debug instrumentation (exception log + the tail of the EIP ring
/// buffer) when X86EMU_DEBUG is set. Addresses are raw; run them through
/// `tools/sym.py` to get kernel symbol names.
/// Dump a region of physical memory to a file when X86EMU_DUMP_PHYS is set,
/// as `<hex addr>:<hex len>:<path>`. The kernel's printk log buffer lives in
/// .bss and is not in kallsyms, so pulling it out of RAM is the way to read a
/// panic message that never reached a console.
fn dump_phys(cpu: &Cpu) {
    let spec = match std::env::var("X86EMU_DUMP_PHYS") {
        Ok(v) => v,
        Err(_) => return,
    };
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 3 {
        eprintln!("X86EMU_DUMP_PHYS wants <hexaddr>:<hexlen>:<path>");
        return;
    }
    let addr = usize::from_str_radix(parts[0].trim_start_matches("0x"), 16).unwrap_or(0);
    let len = usize::from_str_radix(parts[1].trim_start_matches("0x"), 16).unwrap_or(0);
    let mut buf = Vec::with_capacity(len);
    for i in 0..len {
        buf.push(cpu.mem.read_u8(addr + i));
    }
    match std::fs::write(parts[2], &buf) {
        Ok(_) => println!("dumped {:X} bytes from phys {:08X} to {}", len, addr, parts[2]),
        Err(e) => eprintln!("error writing {}: {}", parts[2], e),
    }
}

/// Dump a *linear* range through the current page tables when
/// X86EMU_DUMP_LINEAR is set, as `<hex addr>:<hex len>:<path>`. Physical
/// dumps cannot reach a user process's address space; this can.
fn dump_linear(cpu: &mut Cpu) {
    let spec = match std::env::var("X86EMU_DUMP_LINEAR") {
        Ok(v) => v,
        Err(_) => return,
    };
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 3 {
        eprintln!("X86EMU_DUMP_LINEAR wants <hexaddr>:<hexlen>:<path>");
        return;
    }
    let addr = u64::from_str_radix(parts[0].trim_start_matches("0x"), 16).unwrap_or(0);
    let len = u64::from_str_radix(parts[1].trim_start_matches("0x"), 16).unwrap_or(0);
    let mut buf = Vec::with_capacity(len as usize);
    for i in 0..len {
        let phys = cpu.apply_paging(addr.wrapping_add(i));
        // A hole in the mapping reads as zero rather than aborting the dump.
        cpu.pending_exception = None;
        buf.push(cpu.mem.read_u8(phys));
    }
    match std::fs::write(parts[2], &buf) {
        Ok(_) => println!("dumped {:X} bytes from linear {:016X} to {}", len, addr, parts[2]),
        Err(e) => eprintln!("error writing {}: {}", parts[2], e),
    }
}

fn print_debug(cpu: &mut Cpu) {
    dump_phys(cpu);
    dump_linear(cpu);
    if !cpu.unknown_ops.is_empty() {
        println!();
        println!("--- unimplemented opcodes ---");
        for (op, (count, eip)) in cpu.unknown_ops.iter() {
            if *op > 0xFF {
                println!("  0F {:02X}  hits={:<10} first at rip={:016X}",
                    op & 0xFF, count, eip);
            } else {
                println!("  {:02X}     hits={:<10} first at rip={:016X}",
                    op, count, eip);
            }
        }
    }
    if !cpu.debug_enabled {
        return;
    }
    println!();
    println!("hardware interrupts delivered: {}", cpu.irq_count);
    println!("PIT  ch0 count={:04X} reload={:04X} mode={} access={}",
        cpu.pit.ch0_count, cpu.pit.ch0_reload, cpu.pit.ch0_mode, cpu.pit.ch0_access);
    println!("PIC  master imr={:02X} irr={:02X} isr={:02X} base={:02X}",
        cpu.pic.master_imr, cpu.pic.master_irr, cpu.pic.master_isr, cpu.pic.master_base);
    println!("PIC  slave  imr={:02X} irr={:02X} isr={:02X} base={:02X}",
        cpu.pic.slave_imr, cpu.pic.slave_irr, cpu.pic.slave_isr, cpu.pic.slave_base);
    for (v, n) in cpu.irq_vectors.iter().enumerate() {
        if *n > 0 {
            println!("  IRQ vector {:02X}: {} (IDT -> {:016X})", v, n, cpu.idt_target(v as u8));
        }
    }
    println!("user-mode instructions: {}  ring switches into the kernel: {}",
        cpu.user_instructions, cpu.ring_switches);
    println!("TR selector={:04X} base={:016X} limit={:08X}  LDT base={:016X}",
        cpu.tr_selector, cpu.tr_base, cpu.tr_limit, cpu.ldt_base);
    if !cpu.watch_log.is_empty() {
        println!("--- writes covering the watched address ---");
        for (n, eip, addr) in cpu.watch_log.iter().take(40) {
            println!("  [{}] rip={:016X} store at {:012X}", n, eip, addr);
        }
    }
    if !cpu.syscall_log.is_empty() {
        println!("--- user system calls (int 0x80) ---");
        for (n, ax, bx, cx, dx) in cpu.syscall_log.iter() {
            println!("  [{}] rax={} rbx={:016X} rcx={:016X} rdx={:016X}", n, ax, bx, cx, dx);
        }
    }
    if !cpu.mem.store_log.is_empty() {
        println!("--- stores to the watched physical address ---");
        for (addr, val, width, eip) in cpu.mem.store_log.iter() {
            println!("  {:012X} <- {:016X} ({} bytes) from rip={:016X}", addr, val, width, eip);
        }
    }
    println!("--- exception counts ---");
    for (v, n) in cpu.exc_counts.iter().enumerate() {
        if *n > 0 {
            println!("  vector {:2} (0x{:02X}): {}", v, v, n);
        }
    }
    println!("--- first {} exceptions ---", cpu.exc_log.len());
    for (n, vector, code, eip, cr2) in cpu.exc_log.iter().take(40) {
        println!("  [{}] vec={:02X} err={:?} rip={:016X} cr2={:016X}",
            n, vector, code, eip, cr2);
    }
    let ring = &cpu.eip_ring;
    if !ring.is_empty() {
        let tail: usize = std::env::var("X86EMU_DEBUG")
            .ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let tail = tail.min(ring.len());
        println!("--- last {} instruction pointers ---", tail);
        for i in 0..tail {
            let idx = (cpu.eip_ring_pos + ring.len() - tail + i) % ring.len();
            println!("  {:016X}", ring[idx]);
        }
    }
}


/// Print the emulated text screen.
///
/// The visible 80x25 window starts at the CRTC's start address, not at the
/// beginning of video memory: that register is how a VGA text console
/// scrolls. Reading from cell zero shows the first twenty-five lines of the
/// boot for ever, however far the machine actually got.
fn print_screen(cpu: &Cpu) {
    let cells = &cpu.mem.vga_text;
    let start = cpu.vga.start_cell();
    println!();
    println!("--- text screen ---");
    for row in 0..x86emu::bios::SCREEN_ROWS {
        let mut line = String::new();
        for col in 0..x86emu::bios::SCREEN_COLS {
            let idx = (start + row * x86emu::bios::SCREEN_COLS + col) % cells.len();
            let ch = (cells[idx] & 0xFF) as u8;
            line.push(if ch == 0 || ch == 0xFF { ' ' } else { ch as char });
        }
        println!("{}", line.trim_end());
    }
}

fn print_state(cpu: &Cpu) {
    use x86emu::cpu::Mode;
    let mode = cpu.mode();
    if mode == Mode::Long {
        // In 64-bit mode the registers are named and printed at their real
        // width; showing the low halves would hide exactly the bits that
        // matter.
        const NAMES: [&str; 16] = ["RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI",
                                   "R8 ", "R9 ", "R10", "R11", "R12", "R13", "R14", "R15"];
        for row in 0..4 {
            let mut line = String::new();
            for col in 0..4 {
                let i = row * 4 + col;
                line.push_str(&format!("{}={:016X} ", NAMES[i], cpu.regs[i]));
            }
            println!("{}", line.trim_end());
        }
        println!("RIP={:016X} RFLAGS={:08X}", cpu.rip, cpu.flags);
    } else {
        println!("EAX={:08X} EBX={:08X} ECX={:08X} EDX={:08X}",
            cpu.eax(), cpu.ebx(), cpu.ecx(), cpu.edx());
        println!("AX={:04X} BX={:04X} CX={:04X} DX={:04X}",
            cpu.ax(), cpu.bx(), cpu.cx(), cpu.dx());
        println!("ESP={:08X} EBP={:08X} ESI={:08X} EDI={:08X}",
            cpu.esp(), cpu.ebp(), cpu.esi(), cpu.edi());
        println!("SP={:04X} BP={:04X} SI={:04X} DI={:04X}",
            cpu.sp(), cpu.bp(), cpu.si(), cpu.di());
        println!("EIP={:08X} IP={:04X} FLAGS={:04X}", cpu.eip(), cpu.ip, cpu.flags);
    }
    println!("CS={:04X} DS={:04X} ES={:04X} SS={:04X} FS={:04X} GS={:04X}",
        cpu.cs, cpu.ds, cpu.es, cpu.ss, cpu.fs, cpu.gs);
    println!("mode: {}", match mode {
        Mode::Real => "real",
        Mode::Protected => "protected (32-bit)",
        Mode::Compat => "long mode, compatibility (32-bit code)",
        Mode::Long => "long mode, 64-bit",
    });
    if cpu.pe {
        println!("GDT base={:016X} limit={:04X}  IDT base={:016X} limit={:04X}",
            cpu.gdt_base, cpu.gdt_limit, cpu.idt_base, cpu.idt_limit);
        println!("CR0={:08X} CR3={:016X} CR4={:08X} EFER={:016X}",
            cpu.cr0, cpu.cr3, cpu.cr4, cpu.efer);
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