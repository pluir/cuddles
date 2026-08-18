# x86emu — an incremental x86 PC emulator in Rust

A from-scratch x86 emulator, built up in stages. The goal is a full PC
emulator (CPU + memory + devices) that can eventually boot real firmware.
We're starting with a solid 16-bit real-mode CPU core and layering on
protected mode, paging, and devices.

## Current stage: boots Linux to a shell

A real 32-bit Linux kernel (buildroot, 2.6.34.14) boots on this emulator from
`startup_32` to a busybox shell prompt: it brings up its devices, mounts an
ext2 root filesystem from a ramdisk, starts `/sbin/init` in ring 3, and
prints the whole thing to the emulated VGA text console.

```
[    8.896200] RAMDISK: ext2 filesystem found at block 0
[    8.908775] RAMDISK: Loading 3883KiB [1 disk] into ram disk... done.
[   21.496390] VFS: Mounted root (ext2 filesystem) on device 1:0.
[   21.507731] Freeing unused kernel memory: 184k freed

/root%
```

The project is under git version control (see the repo root). The kernel and
root filesystem live in `images/` (git-ignored: they are downloaded, not
source).

### Booting Linux

Everything comes out of one ISO:

```sh
python tools/extract_iso.py images/linux.iso                    # list it
python tools/extract_iso.py images/linux.iso BZIMAGE  images/bzImage
python tools/extract_iso.py images/linux.iso ROOT.BIN images/root.bin
python tools/unpack_bzimage.py images/bzImage images/golden_kernel.bin
cargo build --release
./target/release/x86emu --kernel-elf images/golden_kernel.bin \
    --initrd images/root.bin \
    --cmdline "root=/dev/ram0 rw console=tty0" 3000000000
```

The emulated text screen is printed when the run ends. A full boot to the
shell prompt takes a little over 300 million instructions.

Debugging switches, all off by default and all environment variables:

| Variable | Effect |
|---|---|
| `X86EMU_DEBUG=<n>` | Exception log, interrupt counts, device state, and the last `n` instruction pointers. |
| `X86EMU_TRACE=1` | One line per instruction to `trace.txt`. |
| `X86EMU_TRACE_FROM=<n>` | Start tracing at instruction `n` (a whole boot is billions of lines). |
| `X86EMU_TRAP_EIP=<hex>` | Stop the moment execution reaches an address. |
| `X86EMU_TRAP_USER=<n>` | Stop before the `n`th user-mode instruction. |
| `X86EMU_WATCH=<hex>` / `X86EMU_WATCH_PHYS=<hex>` | Log stores covering a linear / physical address. |
| `X86EMU_WATCH_STORE=<hex>` | Log stores at the memory layer, with the value and the EIP that wrote it. |
| `X86EMU_DUMP_PHYS`, `X86EMU_DUMP_LINEAR` | Dump `<hexaddr>:<hexlen>:<path>` at the end of the run. |
| `X86EMU_EPOCH=<secs>` | Pin the RTC so a boot is reproducible instruction for instruction. |
| `X86EMU_NO_TLB=1` | Walk the page tables on every access, to rule the TLB in or out. |

`tools/kallsyms.py` extracts a symbol table from the (stripped) kernel image
and `tools/sym.py` turns an address into a name — between them, "stuck at
`C02EF525`" becomes "stuck in `early_page_fault`".

### What works

- **128 MiB flat memory** with real-mode `segment:offset` → 20-bit physical
  address translation (`segment * 16 + offset`, with 1 MiB wraparound) and
  32-bit protected-mode translation through segment descriptors.
- **Registers**: eight general registers stored once, 32 bits wide
  (EAX/.../EDI), with AX/.../DI and AL/AH/... as *views* of them rather than
  separate copies; six segment registers (ES/CS/SS/DS/FS/GS); the instruction
  pointer; and a full 32-bit EFLAGS.
- **Flags**: CF, PF, AF, ZF, SF, TF, IF, DF, OF, plus the high-half bits a
  32-bit OS cares about — `AC` (18) and `ID` (21), which Linux toggles through
  `PUSHFD`/`POPFD` to identify the CPU. Computed correctly for the implemented
  ALU operations, including the INC/DEC rule that CF is preserved.
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
  - `LGDT`/`LIDT` (0x0F 0x01 /2 and /3) — the memory operand address
    follows the current addressing mode (16-bit `modrm_addr` or 32-bit
    `modrm_addr32`), so `lidt [disp32]` works correctly in 32-bit mode.
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
  - `PUSHA`/`POPA` (0x60/0x61), `POP r/m` (0x8F), `LEAVE` (0xC9),
    `RET imm16` (0xC2), `INC`/`DEC r/m8` (group 4, 0xFE).
  - `SETcc` (0x0F 0x90-0x9F), `CMOVcc` (0x0F 0x40-0x4F).
  - Two- and three-operand `IMUL` (0x0F 0xAF, 0x69, 0x6B) and the
    double-precision shifts `SHLD`/`SHRD` (0x0F 0xA4/0xA5/0xAC/0xAD).
  - Bit scans `BSF`/`BSR` (0x0F 0xBC/0xBD) and `BSWAP` (0x0F 0xC8+r).
  - Atomics: `XCHG r/m,r` (0x86/0x87), `CMPXCHG` (0x0F 0xB0/0xB1),
    `CMPXCHG8B` (0x0F 0xC7 /1), `XADD` (0x0F 0xC0/0xC1), and the `LOCK`
    prefix (consumed — this is a uniprocessor emulator).
  - Segment `PUSH`/`POP` including FS/GS (0x0F 0xA0/0xA1/0xA8/0xA9),
    `MOV` to and from the debug registers (0x0F 0x21/0x23), and
    `LLDT`/`LTR`/`SLDT`/`STR` (0x0F 0x00).
  - The multi-byte `NOP` (0x0F 0x1F) and the fences (0x0F 0xAE /5-/7), which
    GCC and the kernel's alternatives machinery both emit.
  - **x87 FPU** (`src/fpu.rs`): the D8-DF escape opcodes — `FNINIT`,
    `FSTCW`/`FLDCW`, `FSTSW` (AX and m16), `FLD`/`FST`/`FSTP` (m32/m64 and
    ST(i)), `FILD`/`FISTP` (m16/m32), simplified `FADD/FSUB/FMUL/FDIV`
    (ST0 op m), and `FXSAVE`/`FXRSTOR` (0x0F 0xAE /0 and /1). The FPU has
    the control/status/tag words and eight 80-bit data registers (stored as
    `f64`). Enough for the Linux kernel's early FPU probing.
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
  applied to all memory accesses when paging is enabled, cached in a
  256-entry TLB. **Page-level protection** is enforced: the effective R/W and
  U/S bits are the AND of the directory's and the table's, a supervisor write
  to a read-only page faults when `CR0.WP` is set, a user access to a
  supervisor page faults, and the accessed and dirty bits are maintained. The
  `#PF` error code reports present/write/user, and the faulting linear address
  goes in CR2.
- **Restartable faults**: a fault reports the address of the instruction that
  faulted (traps report the next one), and a faulting instruction commits
  nothing — no register write, no memory write, no stack-pointer movement, and
  a `REP` string stops at the element that faulted with its index and count
  registers pointing at it. This is what makes demand paging, copy-on-write
  and the kernel's exception-table fixups work.
- **Ring 3**: an interrupt or exception taken at CPL 3 switches to the ring-0
  stack recorded in the **TSS** and pushes the outer SS:ESP below the usual
  frame; `IRET` to a less privileged code segment restores them. Gate types
  are honoured (an interrupt gate clears IF, a trap gate does not), and the
  LDT is consulted for selectors with the table-indicator bit set.
- **Devices**:
  - **8254 PIT** (`src/pit.rs`): three 16-bit countdown channels. Channel 0
    is wired to the PIC's IRQ0 (the system timer); the timer ticks once per
    emulated instruction. Channel 2's gate and output line are visible on
    port `0x61`, which is what the kernel's PIT-based TSC calibration spins
    on. I/O ports `0x40`-`0x43` and `0x61`.
  - **MC146818 CMOS RTC** (`src/cmos.rs`): the real-time clock and
    configuration RAM on ports `0x70`/`0x71`, with the date taken from the
    host at reset and advanced with emulated time. Linux reads it on every
    boot and spins until the update-in-progress bit clears — a machine
    without one hangs there, before it ever reaches the timer.
  - **8259 PIC** (`src/pic.rs`): master + slave, 15 hardware IRQs mapped onto
    configurable base vectors. I/O ports `0x20`/`0x21` (master) and
    `0xA0`/`0xA1` (slave).
  - **VGA** (`src/vga.rs`): text mode (80x25) plus graphics modes 12h
    (640x480, 16 colours) and 13h (320x200, 256 colours), with a
    memory-mapped-style framebuffer, plus the CRTC registers on `0x3D4`/
    `0x3D5`. The whole 32 KiB text aperture at physical `0xB8000` is
    memory-mapped into `Memory`, so the CPU can write the screen directly —
    as Linux does — not only through the BIOS, and the CRTC's start-address
    register selects which part of it is on screen. That register is how a
    text console scrolls: model one screenful and everything after the
    twenty-fifth line disappears.
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
- **A binary** (`x86emu`) with four modes: a flat `.bin` at a
  `segment:offset`; `--boot` for a 512-byte boot sector at `0000:0x7C00`;
  `--kernel` for a bzImage through the Linux boot protocol; and
  `--kernel-elf` for an already-decompressed kernel ELF, which also takes
  `--initrd <file>` and `--cmdline <string>`. Every mode prints the final
  register state and the emulated text screen.

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
  memory.rs       — flat `Memory` (128 MiB, `Memory::SIZE` is the single source
                  of truth for RAM size) + segment translation
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
  fpu.rs          — x87 FPU: control/status/tag words, 8 data registers, D8-DF instructions
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
7. **Boot a real OS (Linux, 32-bit)** — *done*. A real 32-bit buildroot
   kernel (2.6.34.14) boots on the emulator all the way to a busybox shell
   prompt: it prints its whole log to the emulated VGA text console, mounts an
   ext2 root filesystem out of an initial ramdisk, runs `/sbin/init` in ring 3,
   and reaches an interactive prompt. See **Booting Linux** above for how to
   run it.

   What it took, beyond the earlier stages, was mostly *correctness* rather
   than new features — a kernel exercises the parts of an emulator that a
   hand-written test program never reaches, and it fails a long way from the
   cause. The bugs worth naming, because each one is a trap the next
   emulator will fall into too:

   - **`CMP` wrote its result back.** The r/m↔reg forms shared their commit
     path with `SUB`, so every comparison corrupted its destination.
   - **32-bit shifts ran as 8-bit shifts.** `Shift` carried a `w: bool` and
     the decoder passed `!w32`, so in a 32-bit segment `shr %cl,%edx` shifted
     only DL. The width is now explicit (8/16/32) and one width-generic
     routine implements every shift and rotate.
   - **EFLAGS was 16 bits.** Linux toggles the `AC` and `ID` bits through
     `PUSHFD`/`POPFD` to identify the CPU; a 16-bit flags word cannot carry
     them, and `PUSHF`/`POPF` moved the wrong number of bytes besides.
   - **The exception error code was pushed first, not last.** The frame is
     EFLAGS, CS, EIP, error code — error code on top. Pushed first, every
     field the kernel's entry stub reads is off by one slot, and it reports
     faults at the CS selector's "address".
   - **Faults reported the *next* instruction.** A fault must save the
     address of the instruction that faulted so it can be restarted. Without
     that, the kernel's exception-table fixups never match and demand paging
     cannot work at all.
   - **A faulting instruction still committed its result.** `add (%edi),%edx`
     whose load page-faults has to leave EDX alone; committing first means the
     restart adds to an already-modified value.
   - **`REP` string instructions ran to completion through a page fault**,
     leaving the index and count registers past the end, so the restart
     copied nothing. They now stop at the faulting element.
   - **The instruction-fetch cache advanced past a page boundary without
     re-translating**, so an immediate or displacement whose last byte sat at
     offset 0xFFF was completed from an unrelated physical page.
   - **`BT`/`BTS`/`BTR`/`BTC` with a register operand used the register
     *number* as the bit index** instead of its value, so every `test_bit()`
     in the kernel answered from the wrong bit — which had it believing every
     interrupt vector was already claimed.
   - **Byte port I/O truncated the port to 8 bits**, so `out %al,%dx` with
     DX=0x3D4 wrote port 0xD4. Port addresses are 16 bits.
   - **Segment overrides were ignored on the `moffs` forms**, so
     `mov %gs:0xC,%eax` — the i386 way to read thread-local storage — read
     through DS.
   - **AX and EAX were separate fields.** Every 32-bit write had to remember
     to refresh the 16-bit half, and the ones that forgot left the two
     disagreeing until the next byte-register write rebuilt the 32-bit
     register from the stale copy. The 16-bit registers are now *views* of
     the 32-bit ones, so the class of bug is gone rather than fixed.
   - **`CR0.WP` was not implemented.** Linux checks that a supervisor write
     to a read-only page faults, and panics on a CPU that gets it wrong.
   - **`HLT` stopped the machine** instead of waiting for an interrupt, so
     the idle loop ended the run.
   - **Hardware interrupts were delivered regardless of `IF`.**
   - **`FST`/`FSTP` always wrote 8 bytes**, so a single-precision store
     clobbered the four bytes after its destination.

   New instructions the kernel and userspace needed: `PUSHA`/`POPA`,
   `POP r/m`, `SETcc`, two- and three-operand `IMUL`, `SHLD`/`SHRD`,
   `BSF`/`BSR`, `XCHG r/m,r`, `CMPXCHG`, `CMPXCHG8B`, `XADD`, `BSWAP`,
   `CMOVcc`, `LEAVE`, `RET imm16`, `INC`/`DEC r/m8`, segment `PUSH`/`POP`
   (including FS/GS), `MOV` to and from the debug registers, `LLDT`/`LTR`/
   `SLDT`/`STR`, the `LOCK` prefix, the multi-byte `NOP`, and the fences.

   New machine parts: an **MC146818 CMOS RTC** (`src/cmos.rs`) — Linux spins
   on its update-in-progress bit at every boot, so a machine without one
   hangs before it reaches the timer — **PIT channel 2** with its gate and
   output visible on port 0x61, which is what the kernel's TSC calibration
   waits on, and the **VGA CRTC registers**, without which the console
   scrolls off into video memory the emulator was not modelling.

   **Ring 3** works: an interrupt taken at CPL 3 switches to the ring-0 stack
   from the TSS and pushes the outer SS:ESP, and `IRET` back to user mode
   restores them. Page-level protection distinguishes user from supervisor
   and read from write, with the accessed and dirty bits maintained, so
   demand paging and copy-on-write behave as the kernel expects.

   Not done: the in-kernel bzImage decompressor still does not run correctly,
   so `--kernel-elf` loads the decompressed ELF directly. The 8042 keyboard
   controller does not implement the command interface the kernel probes
   (`i8042: probe failed`), so the shell has no keyboard input yet.
