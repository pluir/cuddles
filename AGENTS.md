# AGENTS.md — working notes for x86emu

Guidance for AI agents and humans working on this codebase. Read this before
making changes.

## What this project is

x86emu is a from-scratch x86 PC emulator written in Rust, built up in stages.
The long-term goal is a full PC emulator (CPU + memory + devices) that can
eventually boot real firmware. It is deliberately incremental: each stage adds
one clean, well-tested layer on top of the previous one.

Current state: a solid **16-bit real-mode 8086-style core** (registers, flags,
memory, ModR/M addressing, a broad instruction set) plus a **minimal BIOS
layer** (native Rust handlers for `INT 0x10/0x16/0x13/0x15`), **32-bit protected
mode** (GDT/IDT, segment descriptors, 32-bit registers and addressing,
protected-mode interrupts), **32-bit paging** (page tables, CR0–CR4,
virtual → physical translation), the full **PC device set** (8254 PIT,
8259 PIC, VGA, 8042 keyboard, 8237 DMA, IDE/ATA disk) with hardware
interrupts, **exceptions** (`#DE`, `#BP`, `#OF`, `#UD`, `#PF`), and a
**Linux boot-protocol loader** that is actively booting a real 32-bit kernel
(in progress — the decompressor runs end to end). The project is under git
version control.

## Layout

| Path | Purpose |
|------|---------|
| `src/lib.rs` | Crate root; declares modules. |
| `src/cpu.rs` | `Cpu` struct: registers, flags, fetch-decode-execute loop, stack, ModR/M operand helpers. |
| `src/instructions.rs` | The instruction decoder (`decode`) and executor (`execute`). The largest file. |
| `src/modrm.rs` | ModR/M byte decoding and register-index helpers. |
| `src/memory.rs` | Flat 16 MiB `Memory` with segment:offset → physical translation. |
| `src/protected.rs` | Segment descriptors, GDT/IDT parsing, protected-mode translation. |
| `src/paging.rs` | 32-bit page-directory/page-table walk (4 KiB and 4 MiB pages). |
| `src/pit.rs` | 8254 Programmable Interval Timer (channel 0 -> IRQ0). |
| `src/pic.rs` | 8259 Programmable Interrupt Controller (master + slave). |
| `src/vga.rs` | VGA display: text mode (80x25) + graphics modes 12h/13h. |
| `src/kbd.rs` | 8042 keyboard controller (scancodes, IRQ1). |
| `src/dma.rs` | 8237 DMA controller (4 channels, page registers). |
| `src/ide.rs` | IDE/ATA disk controller (PIO, LBA28, IRQ14). |
| `src/boot.rs` | Linux boot-protocol loader: parse bzImage, load kernel, build `boot_params`, enter protected mode. |
| `src/bios.rs` | `Bios` struct: native Rust handlers for `INT 0x10/0x16/0x13/0x15`. |
| `src/main.rs` | CLI: load a flat binary, boot a boot sector, or boot a Linux bzImage. |
| `examples/` | `gen_add.rs` (generates `add.bin`), plus prebuilt `add.bin` and `boot.bin`. |
| `gen_boot.py` | Python script that hand-assembles `examples/boot.bin`. |
| `images/` | Downloaded OS images for the boot effort: `bzImage` (a real 32-bit buildroot kernel, extracted from the copy/images `linux.iso`). Large `.iso`/`.bin` downloads are git-ignored. |

The project is a git repository (initialized at the project root, branch
`master`). Keep commits focused and the working tree clean before finishing a
change.

## Build & test

```sh
cargo build
cargo test          # all tests must pass before finishing a change
cargo run -- examples/add.bin
cargo run -- --boot examples/boot.bin
```

## How the emulator works

- **Memory** is a flat `Vec<u8>` of 16 MiB. Real-mode logical addresses
  `segment:offset` map to physical `segment * 16 + offset`, masked to 20 bits
  (wraps at 1 MiB). See `Memory::phys`. Protected mode translates through
  cached segment descriptors (`Cpu::translate`). When CR0.PG is set, the
  resulting linear address is further translated through the page tables
  rooted at CR3 (`Cpu::apply_paging` → `paging::translate`). The VGA text
  window at physical `0xB8000` (80x25 cells) is memory-mapped: reads/writes
  in that range are routed to `Memory::vga_text` so the CPU can write the
  text screen directly (as Linux does), not only through the BIOS. The BIOS
  teletype/scroll services and the CLI's screen dump read this same window.
- **The CPU** keeps registers as individual `u16`/`u32` fields plus a packed
  `flags` word. The 32-bit registers (EAX/...) are kept in sync with their
  16-bit halves (AX/...) — writing AX updates the low half of EAX. Byte
  registers (AL/AH, ...) are handled through `reg8`/`set_reg8` helpers.
- **Decode/execute** live in `instructions.rs`. `decode` reads opcode bytes
  (and ModR/M + immediates) from the instruction stream via `Cpu::fetch_*`;
  `execute` mutates the `Cpu`. `Cpu::step` calls both and bumps
  `instructions_executed`.
- **Prefixes** (`0x66` operand-size, `0x67` address-size, segment overrides,
  REP) are consumed at the top of `decode` and stored on the `Cpu`
  (`opsize`/`addrsize`/`seg_override`). They are reset at the *start* of the
  next `decode` so `execute` can still see them. The *default* operand and
  address size is 16-bit in real mode, but in protected mode it is derived
  from the code segment's D bit (D=1 → 32-bit); a `0x66`/`0x67` prefix
  *toggles* the size rather than forcing 32-bit. This matters for booting a
  32-bit kernel, whose code runs in a D=1 segment with no size prefixes.
- **The BIOS** is *not* machine code. It is a set of host-side Rust service
  routines. The `INT` executor checks the vector: if it is a BIOS vector
  (0x10/0x15/0x16/0x13), it calls the matching `Bios` method directly (no IVT frame
  is pushed); otherwise it dispatches through the real Interrupt Vector Table
  at physical `0:vector*4` (real mode) or the IDT (protected mode). `INT 0x15`
  implements the E820 memory map (`AH=0xE820`, entry-by-entry with the
  `'SMAP'` signature and EBX continuation), E801 (`AH=0xE8`/`AL=0x01`) and
  `AH=0x88` (extended memory in KB) — the RAM-layout queries Linux makes very
  early in boot.
- **Devices & hardware interrupts.** The `Cpu` owns a `Pit`, a `Pic`, a
  `Vga`, a `Kbd`, a `Dma`, and an `Ide`. `Cpu::step` calls
  `deliver_hardware_interrupt` before each instruction: it ticks the PIT
  (channel 0 asserts IRQ0), latches the keyboard (IRQ1) and IDE (IRQ14) into
  the PIC, and if the PIC acknowledges a pending unmasked IRQ, dispatches it
  through the IVT (real mode) or IDT (protected mode). `IRET` clears
  `servicing_irq` so the next interrupt can fire. Port I/O goes through
  `Cpu::port_in`/`port_out` (IN/OUT instructions `0xE4`-`0xEF`), which route
  to the PIT (`0x40`-`0x43`), PIC (`0x20`/`0x21`, `0xA0`/`0xA1`), 8042
  (`0x60`/`0x64`), DMA (`0x00`-`0x0F`, `0x81`-`0x8F`), and IDE (`0x1F0`-
  `0x1F7`, `0x3F6`). Note the IDE ports exceed `u8` range, so they are
  reached through the 16-bit `port_in16`/`port_out16` paths. The BIOS video
  output is routed through the `Vga` device (text + graphics framebuffers),
  and `INT 0x13` disk I/O is backed by the `Ide` device's disk image.
- **Exceptions.** The `Cpu` has a `pending_exception: Option<(u8,
  Option<u32>)>` field. Instructions that fault (DIV/IDIV by zero → `#DE`,
  `INT3` → `#BP`, `INTO` with OF set → `#OF`, invalid opcode → `#UD`) set it
  directly. Page faults are raised inside `apply_paging`/`translate` when
  CR0.PG is set and a page is not present (recording the faulting linear
  address in CR2). Because address translation now needs to record faults,
  the translation and ModR/M operand methods (`translate`, `apply_paging`,
  `phys_ip`, `modrm_addr`, `read_rm*`/`write_rm*`, `push*`/`pop*`) take
  `&mut self`. `Cpu::step` dispatches any pending exception at the top of the
  next instruction via `dispatch_exception`, which pushes the error code (if
  any) and vectors through the IDT (protected mode) or IVT (real mode). If an
  exception fires but the IDT/IVT is not set up for its vector (e.g. a fault
  before the kernel installs its IDT), the CPU **triple-faults**: it sets
  `triple_fault = true` and halts instead of dispatching to a garbage entry
  and looping forever (a real CPU resets on a triple fault).
- **The Linux boot loader** (`src/boot.rs`) implements the Linux boot
  protocol (Documentation/x86/boot.rst). `parse_bzimage` reads the setup
  header at file offset `0x1F1` (the boot sector occupies file offset 0,
  and the setup-header fields begin at `0x1F1` within it), validating the
  `0xAA55` boot flag and the `"HdrS"` magic. `load_kernel`
  loads the protected-mode kernel at `code32_start`, builds a `boot_params`
  structure at `0x90000` (setup header, E820 memory map, command line at
  `0x20000`), writes a flat GDT at `0x1000`, enables protected mode (CR0.PE),
  loads flat segments, sets `ESI = boot_params`, and jumps to the kernel with
  `EIP = code32_start` — exactly what the kernel's `startup_32` expects. The
  `--kernel` CLI mode drives it. There is also `load_elf_kernel` (driven by
  `--kernel-elf`), which loads an already-decompressed kernel ELF directly —
  the path a bootloader uses for an uncompressed kernel, and a way to boot a
  kernel without running the in-kernel decompressor (which the emulator does
  not yet execute correctly). The decompressed ELF is extracted from the
  bzImage with `images/parse_bz4.py` (saved as `images/golden_kernel.bin`).

## Conventions

- **One layer at a time.** Prefer completing the current layer over starting a
  new one. The roadmap below is the intended order.
- **Every new instruction or feature gets a test.** Tests live in
  `#[cfg(test)] mod tests` at the bottom of the relevant file. Keep the suite
  green before finishing.
- **Flag semantics matter.** The 8086 has subtle rules (e.g. `INC`/`DEC`
  preserve CF; `NEG` sets CF unless the result is zero; rotate-through-carry
  for `RCL`/`RCR`). Get these right and test them.
- **Default segments.** BP-based ModR/M addressing uses SS; everything else
  uses DS. See `Cpu::modrm_addr`.
- **Wrapping arithmetic.** Use `wrapping_add`/`wrapping_sub` for 16-bit
  register and address math.
- **Keep the README in sync.** When you add instructions, features, or CLI
  options, update the README's implemented list and roadmap.

## Roadmap

- [x] 16-bit real-mode core: registers, flags, memory, ModR/M, core ALU/MOV/stack/jumps.
- [x] Finish the real-mode instruction set: shifts/rotates, MUL/DIV, LEA, CBW/CWD, LOOP/JCXZ, far JMP/CALL/RET, string ops with REP.
- [x] Interrupt handling: real `INT`/`IRET` with the Interrupt Vector Table, plus PUSHF/POPF.
- [x] BIOS/firmware layer: `INT 0x10` (video), `INT 0x16` (keyboard), `INT 0x13` (disk).
- [x] 32-bit protected mode: GDT/IDT, segment descriptors, 32-bit registers and addressing, protected-mode interrupts.
- [x] Paging: page tables, CR0–CR4, virtual → physical translation.
- [x] Devices (part 1): 8254 PIT and 8259 PIC, hardware interrupts (IRQ0 from the timer), IN/OUT port I/O.
- [x] Devices (part 2): VGA text/graphics framebuffer, 8042 keyboard controller, DMA, IDE/ATA disk, boot a real OS image.
- [x] Exceptions: `#DE`, `#BP`, `#OF`, `#UD`, `#PF` (with CR2), dispatched through the IVT/IDT with optional error codes. Exceptions that fire with no IDT installed (or while handling another) **triple-fault** — the CPU halts cleanly instead of looping forever.
- [ ] Boot a real OS (Linux, 32-bit): VGA text memory-mapped at `0xB8000` (done), E820/E801/`0x88` memory map via `INT 0x15` (done), `CPUID` (0x0F 0xA2) and `RDTSC` (0x0F 0x31) (done), boot-protocol loader (done: parse bzImage, load kernel at `code32_start`, build `boot_params`, flat GDT, enter protected mode, jump with `ESI = boot_params`). A real 32-bit buildroot kernel is at `images/bzImage`; the loader now uses the correct boot-protocol field offsets and the kernel's decompressor runs end to end and jumps to the decompressed kernel. Missing instructions it needed (flag-control, shift-with-imm8 `0xC0/0xC1`, group-5 `FF`) are added, and the decoder derives default operand/address size from the code segment's D bit. The in-kernel decompressor does not yet produce correct output, so `--kernel-elf` loads the decompressed ELF directly (`images/golden_kernel.bin`); booting that, the kernel now runs real 32-bit code (CPUID/RDTSC/RDMSR, paging setup) before hitting a page fault it can't yet handle (no IDT installed). Missing instructions added: `TEST acc,imm` (0xA8/0xA9), `RDMSR`/`WRMSR` (0x0F 0x32/0x30), bit tests `BT/BTS/BTR/BTC` (0x0F 0xA3/0xAB/0xB3/0xBB, group 8 0x0F 0xBA). The "IDT problem" (an exception firing before the IDT is installed loops forever through the empty IDT) is now handled with **triple-fault detection**: the CPU halts cleanly with a diagnostic instead of looping. Next: keep chasing what breaks until the kernel reaches console output.

## Common tasks

- **Add an instruction:** add a variant to the `Inst` enum in
  `instructions.rs`, add its opcode case(s) in `decode`, add the execution
  logic in `execute`, then add a test.
- **Add a BIOS service:** add a method to `Bios` in `bios.rs`, dispatch it from
  the `INT` executor, and add a test.
- **Add a device:** create a module (e.g. `src/pit.rs`), add a field to the
  `Cpu` struct, route its ports through `Cpu::port_in`/`port_out`, wire any
  IRQ into `deliver_hardware_interrupt`, and add tests.
- **Regenerate the demo boot sector:** edit `gen_boot.py`, then run
  `python gen_boot.py` to rewrite `examples/boot.bin`.
- **Boot a Linux kernel:** `cargo run -- --kernel <bzImage> [max_instructions]`.
  The loader parses the setup header, loads the kernel, and enters protected
  mode at `code32_start`. If the kernel crashes early, check which instruction
  or BIOS service it hit first (the register dump + text screen will show it)
  and implement the missing piece.
