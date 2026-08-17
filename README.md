# x86emu — an incremental x86 PC emulator in Rust

A from-scratch x86 emulator, built up in stages. The goal is a full PC
emulator (CPU + memory + devices) that can eventually boot real firmware.
We're starting with a solid 16-bit real-mode CPU core and layering on
protected mode, paging, and devices.

## Current stage: 16-bit real-mode core + BIOS + 32-bit protected mode + paging + devices + exceptions + booting a real Linux kernel (in progress)

The project is under git version control (see the repo root). A real 32-bit
Linux kernel image (buildroot bzImage, extracted from the copy/images
`linux.iso`) is kept at `images/bzImage` for the boot effort.

### What works

- **128 MiB flat memory** with real-mode `segment:offset` → 20-bit physical
  address translation (`segment * 16 + offset`, with 1 MiB wraparound) and
  32-bit protected-mode translation through segment descriptors.
- **Registers**: 8 × 16-bit general registers (AX/CX/DX/BX/SP/BP/SI/DI) with
  byte-register access (AL/AH/.../BH), 8 × 32-bit registers (EAX/.../EDI)
  kept in sync with their 16-bit halves, 6 segment registers (ES/CS/SS/DS/
  FS/GS), the instruction pointer, and the FLAGS register.
- **Flags**: CF, PF, AF, ZF, SF, TF, IF, DF, OF — computed correctly for the
  implemented ALU operations (including the INC/DEC rule that CF is preserved).
- **ModR/M decoding**: the full 16-bit memory addressing modes
  (`[BX+SI]`, `[BP+DI]`, disp8/disp16, register-direct), with the correct
  default segment (DS, or SS for BP-based addressing) — plus 32-bit
  addressing with the SIB byte and 32-bit displacements.
- **Size-override prefixes**: `0x66` (operand size) and `0x67` (address
  size), plus segment overrides (CS/SS/DS/ES/FS/GS).
- **Instruction set**:
  - `MOV` in all common forms (reg↔r/m, imm→reg, imm→r/m, sreg↔r/m,
    accumulator↔moffs), 8/16/32-bit.
  - The ALU group `ADD/OR/ADC/SBB/AND/SUB/XOR/CMP` in their r/m8, r/m16,
    r/m32, r/m↔reg, AL/AX/EAX-imm forms, and the group-1 `0x80/0x81/0x83`
    immediate forms.
  - `INC`/`DEC` (16/32-bit), `PUSH`/`POP` (16/32-bit), `PUSH imm`.
  - `JMP rel8/16/32`, all `Jcc rel8` conditional jumps.
  - `CALL rel16/32` / `RET` (near), far `JMP/CALL/RET`.
  - `XCHG AX/EAX, reg`, `NOP`, `HLT`.
  - `INT imm8` (real-mode through the IVT, protected-mode through the IDT),
    `IRET` (16/32-bit), `PUSHF`, `POPF`.
  - **Exceptions**: `#DE` (divide error, on DIV/IDIV by zero), `#BP`
    (breakpoint, `INT3`), `#OF` (overflow, `INTO`), `#UD` (invalid opcode),
    and `#PF` (page fault, when paging is enabled and a page is not present).
    Exceptions are dispatched through the IVT (real mode) or IDT (protected
    mode) with an optional error code; page faults record the faulting linear
    address in CR2. If an exception fires but the IDT/IVT is not set up for
    it (e.g. a fault before the kernel installs its IDT), the CPU
    **triple-faults**: it halts with `triple_fault = true` instead of
    dispatching to a garbage entry and looping forever (as a real CPU would
    reset).
  - Shifts/rotates `SHL/SHR/SAR/ROL/ROR/RCL/RCR` (group 2, imm8 or CL count).
  - Group 3 `TEST/NOT/NEG/MUL/IMUL/DIV/IDIV` (8/16/32-bit, with the
    DX:AX and EDX:EAX 64-bit forms for MUL/DIV).
  - Group 5 `INC/DEC/CALL/JMP/PUSH r/m` (0xFF).
  - Flag-control instructions `CLC/STC/CLI/STI/CLD/STD/CMC`.
  - `LEA`, `CBW`, `CWD`, `CWDE`, `CDQ`.
  - `LOOP/LOOPZ/LOOPNZ`, `JCXZ`.
  - String ops `MOVS/STOS/LODS/CMPS/SCAS` (byte/word) with the DF flag and
    `REP/REPE/REPNE` prefixes.
  - `LGDT`/`LIDT` (0x0F 0x01 /2 and /3).
  - `MOV r32, cr` / `MOV cr, r32` (0x0F 0x20 / 0x0F 0x22) for CR0/CR2/CR3/CR4.
  - `CPUID` (0x0F 0xA2) — returns the vendor string "GenuineIntel" (leaf 0)
    and family/model/feature flags (leaf 1, including the TSC bit).
  - `RDTSC` (0x0F 0x31) — reads the time-stamp counter (incremented each
    step) into EDX:EAX.
  - `RDMSR`/`WRMSR` (0x0F 0x32 / 0x0F 0x30) — read/write model-specific
    registers (all MSRs report 0; writes are no-ops).
  - Bit tests `BT/BTS/BTR/BTC` (0x0F 0xA3/0xAB/0xB3/0xBB and group 8
    `0x0F 0xBA` /4-/7), 16/32-bit.
  - `TEST AL/AX/EAX, imm` (0xA8/0xA9).
- **Linux boot-protocol loader** (`src/boot.rs`): parses a bzImage setup
  header, loads the protected-mode kernel at `code32_start`, builds a
  `boot_params` structure at `0x90000` (setup header, E820 memory map,
  command line), sets up a flat GDT, enables protected mode, and jumps to
  the kernel entry point with `ESI = boot_params` — exactly what the
  kernel's `startup_32` expects. Driven by the `--kernel` CLI mode. There is
  also `load_elf_kernel` (driven by `--kernel-elf`), which loads an
  already-decompressed kernel ELF directly — the path a bootloader uses for
  an uncompressed kernel, and a way to boot a kernel without running the
  in-kernel decompressor.
- **Protected mode**: GDT/IDT parsing, segment-descriptor caching, 32-bit
  address translation, and protected-mode interrupt dispatch through the IDT.
- **Paging**: 32-bit two-level page tables (page directory + page table),
  4 KiB and 4 MiB pages, CR0.PG/CR3/CR4, and virtual → physical translation
  applied to all memory accesses when paging is enabled. Not-present pages
  raise a `#PF` exception (with the faulting address in CR2).
- **Devices**:
  - **8254 PIT** (`src/pit.rs`): three 16-bit countdown channels. Channel 0
    is wired to the PIC's IRQ0 (the system timer); the timer ticks once per
    emulated instruction. I/O ports `0x40`-`0x43`.
  - **8259 PIC** (`src/pic.rs`): master + slave, 15 hardware IRQs mapped onto
    configurable base vectors. I/O ports `0x20`/`0x21` (master) and
    `0xA0`/`0xA1` (slave).
  - **VGA** (`src/vga.rs`): text mode (80x25) plus graphics modes 12h
    (640x480, 16 colours) and 13h (320x200, 256 colours), with a
    memory-mapped-style framebuffer. The text window at physical `0xB8000`
    is memory-mapped into `Memory` (reads/writes in that range are routed to
    `Memory::vga_text`), so the CPU can write the text screen directly — as
    Linux does — not only through the BIOS.
  - **8042 keyboard controller** (`src/kbd.rs`): scancode queue at port
    `0x60`, status at `0x64`, raises IRQ1.
  - **8237 DMA controller** (`src/dma.rs`): four channels with base/count/
    page registers (ports `0x00`-`0x0F`, `0x81`-`0x8F`) and a simulated
    memory-to-memory transfer.
  - **IDE/ATA disk** (`src/ide.rs`): PIO-mode controller on the primary
    channel (ports `0x1F0`-`0x1F7`, `0x3F6`), LBA28 read/write sectors,
    raises IRQ14.
  - **Hardware interrupts**: the CPU checks for a pending IRQ before each
    instruction; a pending interrupt is dispatched through the IVT (real
    mode) or IDT (protected mode), and `IRET` clears the in-service state.
    IRQ0 (timer), IRQ1 (keyboard) and IRQ14 (IDE) are wired.
  - **IN/OUT instructions** (`0xE4`-`0xE7`, `0xEC`-`0xEF`) for port I/O.
- **Minimal BIOS** (native Rust handlers, dispatched from `INT`):
  - `INT 0x10` video: `AH=0x0E` teletype (prints a char at the cursor and
    advances it), `AH=0x02` set cursor, `AH=0x03` get cursor, `AH=0x00` set
    video mode (text/graphics), `AH=0x13` write string. Output is routed
    through the VGA device (80x25 text screen with 16-colour attribute cells).
  - `INT 0x16` keyboard: `AH=0x00` read a queued key, `AH=0x01` check buffer.
  - `INT 0x13` disk: `AH=0x02`/`0x03` read/write sectors, backed by the IDE
    device's disk image.
  - `INT 0x15` memory map: `AH=0xE820` (E820, entry-by-entry with the
    `'SMAP'` signature and EBX continuation), `AH=0xE8`/`AL=0x01` (E801),
    and `AH=0x88` (extended memory in KB). Reports the physical RAM layout
    Linux queries very early in boot.
- **Stack**: `PUSH`/`POP`/`CALL`/`RET` use SS:SP (16-bit) or SS:ESP (32-bit).
- **A binary** (`x86emu`) that loads a flat `.bin` at a `segment:offset` and
  runs it, printing the final register state — plus a `--boot` mode that loads
  a 512-byte boot sector at `0000:0x7C00` and prints the emulated text screen.

### Working on this codebase

See **`AGENTS.md`** for the project's working notes: layout, how the emulator
works, code conventions, the roadmap, and common tasks (adding an instruction,
adding a BIOS service, regenerating the demo boot sector). Read it before
making changes.

### Project layout

```
src/
  lib.rs          — crate root, re-exports
  cpu.rs          — registers, flags, fetch-decode-execute loop, stack, ModR/M helpers
  memory.rs       — 128 MiB flat memory + segment translation
  modrm.rs        — ModR/M byte decoding + register-index helpers
  instructions.rs — instruction decoder + executor + ALU flag computation
  protected.rs    — segment descriptors, GDT/IDT parsing, protected-mode translation
  paging.rs       — 32-bit page-directory/page-table walk (4 KiB and 4 MiB pages)
  pit.rs          — 8254 Programmable Interval Timer (channel 0 -> IRQ0)
  pic.rs          — 8259 Programmable Interrupt Controller (master + slave)
  vga.rs          — VGA display: text mode + graphics modes 12h/13h
  kbd.rs          — 8042 keyboard controller (scancodes, IRQ1)
  dma.rs          — 8237 DMA controller (4 channels, page registers)
  ide.rs          — IDE/ATA disk controller (PIO, LBA28, IRQ14)
  boot.rs         — Linux boot-protocol loader (parse bzImage, load kernel, boot_params)
  bios.rs         — minimal BIOS: INT 0x10/0x15/0x16/0x13 handlers + text screen
  main.rs         — CLI binary: load a .bin and run it, --boot a boot sector, or --kernel a bzImage
examples/
  gen_add.rs      — writes a tiny test program (examples/add.bin)
  add.bin         — mov ax,0x1234 ; mov bx,2 ; add ax,bx ; hlt
  boot.bin        — a 512-byte boot sector that prints "Hello from x86emu!"
```

### Running it

```sh
cargo test                 # run the unit tests
cargo run --release --example gen_add   # generate examples/add.bin
cargo run --release -- examples/add.bin 0000:0100 100
cargo run --release -- --boot examples/boot.bin   # boot a boot sector
cargo run --release -- --kernel bzImage            # boot a Linux bzImage
```

Sample output (flat binary):
```
executed 4 instructions
halted
AX=1236 BX=0002 CX=0000 DX=0000
...
flags: - P - - - - - - -
```
(`0x1234 + 2 = 0x1236`, parity flag set.)

Sample output (boot sector):
```
executed 113 instructions
halted
AX=0E00 BX=0000 CX=0000 DX=0000
...
--- text screen ---
Hello from x86emu!
```

### Roadmap

1. **More real-mode instructions** — *done*: shifts/rotates, `MUL/DIV`,
   `LEA`, `CBW/CWD`, string ops with `REP`, `LOOP` variants, far
   `JMP/CALL/RET`, and `INT`/`IRET` with a real Interrupt Vector Table.
2. **BIOS/firmware layer** — *done*: a minimal BIOS with `INT 0x10` (video),
   `INT 0x16` (keyboard) and `INT 0x13` (disk) handlers, plus a `--boot` mode
   that loads a boot sector at `0000:0x7C00` and prints the text screen.
3. **32-bit protected mode** — *done*: GDT/IDT/segment descriptors, 32-bit
   registers and addressing (SIB, 32-bit displacements), the `0x66`/`0x67`
   size-override prefixes, `LGDT`/`LIDT`, and protected-mode interrupt
   dispatch through the IDT.
4. **Paging** — *done*: 32-bit page tables (page directory + page table),
   4 KiB and 4 MiB pages, CR0.PG/CR3/CR4, and virtual → physical translation.
5. **PC devices** — *done*: PIT (8254), PIC (8259), VGA (text + graphics
   modes 12h/13h), 8042 keyboard controller, 8237 DMA, and IDE/ATA disk, with
   hardware interrupts (IRQ0 timer, IRQ1 keyboard, IRQ14 IDE) wired through
   the PIC. The emulator now has the device set needed to boot a real OS
   image.
6. **Exceptions** — *done*: `#DE`, `#BP`, `#OF`, `#UD` and `#PF`, dispatched
   through the IVT/IDT with optional error codes; page faults record CR2.
   Exceptions that fire with no IDT installed (or while handling another)
   **triple-fault** — the CPU halts cleanly instead of looping forever.
   This is the keystone that lets a real OS handle faults instead of the
   emulator silently misbehaving.
7. **Boot a real OS (Linux, 32-bit)** — *in progress*. Milestone 1 (console
   output) is done: the VGA text window at `0xB8000` is memory-mapped so the
   kernel can write the screen directly, the E820/E801/`0x88` memory map
   (`INT 15h`) is implemented, and `CPUID`/`RDTSC` are implemented. The
   boot-protocol loader is done: it parses the bzImage setup header, loads
   the kernel at `code32_start`, fills `boot_params` (E820 map + command
   line), sets up a flat GDT, enters protected mode, and jumps to the kernel
   with `ESI = boot_params`. A real 32-bit buildroot kernel image has been
   downloaded (`images/bzImage`) and the loader was fixed to use the correct
   boot-protocol field offsets. Progress so far: the kernel's decompressor
   runs end to end and jumps to the decompressed kernel; the missing
   instructions it needs (flag-control `CLC/STC/CLI/STI/CLD/STD/CMC`,
   shift-with-imm8 `0xC0/0xC1`, group-5 `FF`) have been added, and the
   decoder now derives the default operand/address size from the code
   segment's D bit (with `0x66`/`0x67` toggling it) so 32-bit kernel code
   decodes correctly. The in-kernel decompressor does not yet produce
   correct output in the emulator, so the `--kernel-elf` mode loads the
   decompressed kernel ELF directly (extracted from the bzImage with
   `images/parse_bz4.py`). Booting that ELF, the kernel now runs real 32-bit
   code: it probes CPUID/RDTSC/RDMSR, sets up paging, and gets well into
   early boot before hitting a page fault (which it can't yet handle because
   it hasn't installed its IDT). The missing instructions it needs
   (`TEST acc,imm`, `RDMSR`/`WRMSR`, bit tests `BT/BTS/BTR/BTC`) have been
   added. The current blocker is the "IDT problem": the kernel enables
   paging and then faults before its IDT is installed, so the exception
   dispatches through the empty IDT and loops forever. This is now handled
   with **triple-fault detection** — the CPU halts cleanly with a diagnostic
   instead of looping — which is the correct behavior (a real CPU resets on
   a triple fault). It has not yet reached console output.