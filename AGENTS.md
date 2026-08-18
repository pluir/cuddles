# AGENTS.md — working notes for x86emu

Guidance for AI agents and humans working on this codebase. Read this before
making changes.

## What this project is

x86emu is a from-scratch x86 PC emulator written in Rust, built up in stages.
The long-term goal is a full PC emulator (CPU + memory + devices) that can
eventually boot real firmware. It is deliberately incremental: each stage adds
one clean, well-tested layer on top of the previous one.

Current state: **all three x86 modes, and it boots Linux to a shell.** A real
32-bit buildroot kernel (2.6.34.14) runs from `startup_32` to a busybox prompt
— device init, an ext2 root filesystem off a ramdisk, `/sbin/init` in ring 3,
the whole log on the emulated VGA text console.

Underneath that: a 16-bit real-mode 8086-style core, a minimal BIOS (native
Rust handlers for `INT 0x10/0x16/0x13/0x15`), 32-bit protected mode (GDT/IDT,
descriptors, 32-bit registers and addressing), **64-bit long mode** (REX,
sixteen 64-bit registers, RIP-relative addressing, `SYSCALL`/`SWAPGS`, 16-byte
interrupt gates with an IST, compatibility mode), paging in all three of its
structures (32-bit two-level, PAE three-level, long-mode four-level) with
page-level protection (`CR0.WP`, user/supervisor, NX, accessed and dirty
bits), **ring 3** (TSS stack switch on a gate, `IRET` back to user mode), the
PC device set (8254 PIT, 8259 PIC, VGA with CRTC, 8042 keyboard, 8237 DMA,
IDE/ATA disk, MC146818 RTC) with hardware interrupts, exceptions with
restartable faults, **run-time-sized RAM that can exceed 4 GiB**, and Linux
boot-protocol loaders (32- and 64-bit) with initrd support. The project is
under git version control.

**Reproducing the boot** (see README for the extraction steps):

```sh
./target/release/x86emu --kernel-elf images/golden_kernel.bin \
    --initrd images/root.bin \
    --cmdline "root=/dev/ram0 rw console=tty0" 3000000000
```

**Running 64-bit code** (no download needed):

```sh
cargo run --release --example gen_long64
./target/release/x86emu --long examples/long64.bin
```

## Layout

| Path | Purpose |
|------|---------|
| `src/lib.rs` | Crate root; declares modules. |
| `src/cpu.rs` | `Cpu` struct: registers, flags, fetch-decode-execute loop, stack, ModR/M operand helpers. |
| `src/instructions.rs` | The instruction decoder (`decode`) and executor (`execute`). The largest file. |
| `src/modrm.rs` | ModR/M byte decoding and register-index helpers. |
| `src/memory.rs` | `Memory`, **sized at run time** (`Memory::with_size`, `--mem`). `e820_map()` is the single description of the layout, and RAM past `MMIO_HOLE_START` is wired above 4 GiB. `slot()` is the only thing that knows where RAM lives; an address with none behind it reads `0xFF`. |
| `src/protected.rs` | Segment descriptors, GDT/IDT parsing, protected-mode translation. `Descriptor::l` is the 64-bit-code bit. |
| `src/paging.rs` | Page-table walks for all three structures: legacy (2 levels, 4-byte entries), PAE (3, 8-byte) and long mode (4, 8-byte), with 4 MiB / 2 MiB / 1 GiB large pages, NX, and the canonical-address check. |
| `src/pit.rs` | 8254 Programmable Interval Timer. Channel 0 -> IRQ0; channel 2's gate and output are visible on port 0x61, which is what the kernel's TSC calibration spins on. |
| `src/cmos.rs` | MC146818 CMOS RTC (ports 0x70/0x71). Not optional: Linux spins on its update-in-progress bit at every boot, and a machine without one hangs there. |
| `src/pic.rs` | 8259 Programmable Interrupt Controller (master + slave). |
| `src/vga.rs` | VGA display: text mode + graphics modes 12h/13h, and the CRTC registers (0x3D4/0x3D5). The start-address register is how a text console scrolls — the emulator models the whole 32 KiB text aperture, not one screenful. |
| `src/kbd.rs` | 8042 keyboard controller (scancodes, IRQ1). |
| `src/dma.rs` | 8237 DMA controller (4 channels, page registers). |
| `src/ide.rs` | IDE/ATA disk controller (PIO, LBA28, IRQ14). |
| `src/boot.rs` | Linux boot-protocol loader: parse bzImage, load kernel, build `boot_params`, enter protected mode. Also `enter_long_mode()` (the four-step handover), `build_identity_map()`, `load_elf64_kernel()` and `load_flat64()`. |
| `src/fpu.rs` | x87 FPU: control/status/tag words, 8 data registers (as `f64`), the D8-DF instructions. |
| `src/bios.rs` | `Bios` struct: native Rust handlers for `INT 0x10/0x16/0x13/0x15`. |
| `src/main.rs` | CLI: a flat binary, `--boot`, `--kernel`, `--kernel-elf`, `--kernel-elf64`, `--long`, and `--mem SIZE` (applies to every mode). |
| `examples/` | `gen_add.rs` (generates `add.bin`) and `gen_long64.rs` (generates `long64.bin`, the 64-bit demo), plus the prebuilt `add.bin`, `boot.bin` and `long64.bin`. |
| `gen_boot.py` | Python script that hand-assembles `examples/boot.bin`. |
| `tools/` | Host-side helpers for the boot effort: `extract_iso.py` (pull the kernel and root filesystem out of the ISO), `unpack_bzimage.py` (decompress a bzImage to the ELF `--kernel-elf` wants), `kallsyms.py` (symbol table out of a stripped kernel), `sym.py` (address -> name), `elfsyms.py`. |
| `images/` | Downloaded OS images for the boot effort: `linux.iso` and, extracted from it, `bzImage`, `golden_kernel.bin` (the decompressed kernel ELF) and `root.bin` (the ext2 root filesystem, loaded as an initrd). All git-ignored — they are downloads, not source. |

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

- **Memory** is a flat `Vec<u8>` sized at construction (`Memory::with_size`,
  `Cpu::with_ram`, `--mem`). `Memory::e820()` is the single description of the
  layout: the BIOS E820/E801/0x88 handlers and the boot loader's `boot_params`
  all read it, so a machine cannot describe itself two different ways.
  **A machine larger than `MMIO_HOLE_START` (3 GiB) has its extra RAM above
  4 GiB**, and `slot()` is the one function that knows it — a single compare
  for any address on a machine that fits below the hole. An address with no
  RAM behind it reads `0xFF` and swallows writes; it does **not** alias back
  into low memory, and see "Things not to undo" for why that mattered.
  Real-mode logical addresses `segment:offset` map to physical
  `segment * 16 + offset`, masked to 20 bits (wraps at 1 MiB). See
  `Memory::phys`. Protected mode translates through cached segment
  descriptors (`Cpu::translate`); 64-bit mode has no segment bases at all
  except FS and GS, whose bases come from MSRs. When CR0.PG is set, the
  resulting linear address goes through whichever paging structure
  `Cpu::paging_mode()` names (`Cpu::apply_paging` → `paging::translate_mode`). The VGA text
  window at physical `0xB8000` (80x25 cells) is memory-mapped: reads/writes
  in that range are routed to `Memory::vga_text` so the CPU can write the
  text screen directly (as Linux does), not only through the BIOS. The BIOS
  teletype/scroll services and the CLI's screen dump read this same window.
- **The CPU** keeps the sixteen general registers **once**, 64 bits wide, in
  `regs: [u64; 16]`. `eax()`, `ax()`/`set_ax()` and `reg8_idx` are all *views*
  of the same store; there is no separate narrow storage anywhere. This used
  to be two sets of fields kept in sync by hand, and the writes that forgot to
  sync produced corruption a long way from the cause — `rep movsb` cleared
  `ecx` directly, `cx` kept the old count, and the next `mov $4,%cl` rebuilt
  `ecx` from the stale half. Do not reintroduce a second copy at any width.
  The **width rules are asymmetric and that is not a simplification**: a
  32-bit write zero-extends into the whole 64-bit register, while 16- and
  8-bit writes preserve what is above them (`set_reg_w`). EFLAGS is a full
  **32 bits** (`flags: u32`): Linux identifies the CPU by toggling the `AC`
  (18) and `ID` (21) bits through `PUSHFD`/`POPFD`.
- **Mode.** `Cpu::mode()` returns Real / Protected / Compat / Long.
  `long_mode()` is EFER.LMA — the *machine* is in long mode — and `long64()`
  additionally requires the current code segment's **L** bit. Almost
  everything that cares (operand size, address size, stack width, register
  width) keys off `long64()`, not `long_mode()`, because a 64-bit kernel runs
  32-bit processes in compatibility mode with LMA still set. `osize()` and
  `asize()` give the current widths in bits.
- **Decode/execute** live in `instructions.rs`. `decode` reads opcode bytes
  (and ModR/M + immediates) from the instruction stream via `Cpu::fetch_*`;
  `execute` mutates the `Cpu`. `Cpu::step` calls both and bumps
  `instructions_executed`.
- **Prefixes** (`0x66` operand-size, `0x67` address-size, segment overrides,
  REP, and **REX**) are consumed at the top of `decode` and stored on the
  `Cpu` (`opsize`/`addrsize`/`addr64`/`seg_override`/`rex_*`). They are reset
  at the *start* of the next `decode` so `execute` can still see them — which
  is exactly how the width-generic execute arms read `cpu.osize()`. The
  *default* operand and address size is 16-bit in real mode; in protected mode
  it comes from the code segment's D bit (D=1 → 32-bit) and `0x66`/`0x67`
  *toggle* it. In 64-bit mode neither comes from the segment (D must be 0 when
  L is set): operands default to 32 and addresses to 64, `0x66` selects 16-bit
  operands, `0x67` selects 32-bit addressing, and **REX.W** selects 64-bit
  operands and wins over `0x66`. REX must be the **last** prefix before the
  opcode; a legacy prefix after one cancels it, which the prefix loop
  implements by clearing the REX flags.
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
- **Faults are restartable, and that is a contract, not a detail.** Three
  rules together make demand paging and the kernel's exception tables work,
  and each was a separate bug:
  1. A *fault* reports the address of the instruction that faulted, not the
     one after it (`eip_start`, restored in `dispatch_exception`). Traps
     (`#BP`, `#OF`) report the next instruction. Without this the kernel's
     `__ex_table` lookups never match and a demand-paged instruction is never
     retried.
  2. A faulting instruction **commits nothing**. `set_reg*` and `write_rm*`
     drop their write once `pending_exception` is set, and `push`/`pop`
     translate before moving the stack pointer. `add (%edi),%edx` whose load
     faults must leave EDX alone, or the retry adds to an already-added value.
     The string instructions' index/count writeback is the deliberate
     exception — it uses `set_reg32_raw`, because recording *where the fault
     stopped* is the whole point.
  3. A fault during instruction *fetch* aborts the instruction: `step()` skips
     `execute` when decode left an exception pending, rather than running
     whatever bytes the failed translation returned.
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
- **Page-level protection.** The walk reports a mapping's permissions as well
  as its address, and the effective R/W, U/S and NX are the **combination of
  every level's** — two for legacy paging, four in long mode.
  `Cpu::apply_paging_access` takes the access type: a supervisor write to a
  read-only page faults when `CR0.WP` is set (Linux's `test_wp_bit` panics on
  a CPU that gets this wrong), a user access to a supervisor page faults, and
  the error code records present/write/user. `apply_paging_fetch` is a
  *separate* entry point because of NX: a no-execute page must still be
  readable, so only the fetch path may fault on it, and only when `EFER.NXE`
  is set (bit 63 is reserved otherwise). The TLB caches the permissions
  alongside the translation — a check that only ran on a miss would let the
  second write through. Accessed and dirty bits are maintained on every level
  walked, at the entry width the structure uses. Writing **CR4 flushes the
  TLB**: `__flush_tlb_global()` is literally a CR4 write with PGE toggled.
- **Ring 3.** `Cpu::cpl()` is the CS selector's RPL. A gate to a more
  privileged segment switches to the ring-0 stack from the **TSS** (`LTR`
  caches its base) and pushes the outer SS:ESP below the usual frame;
  `IRET` to a less privileged CS pops them back. The CPL changes **before**
  the frame is written — the pushes happen at the new privilege level, and
  doing it the other way round makes the frame a user write to supervisor
  memory, which paging then rejects.
- **Long mode.** Entering it is a four-step handover and none of the steps can
  move (`boot::enter_long_mode` is the reference): **CR4.PAE** first, because
  long mode is built on PAE's 8-byte entries; **CR3** pointing at a PML4 that
  already maps the code about to run; **EFER.LME**, which only *asks*; and
  **CR0.PG**, which enters — the CPU sets `EFER.LMA` itself in response
  (`Cpu::update_long_mode`). The instruction after the one that sets PG is
  fetched through the new page tables, which is why an identity map is not a
  nicety. Segmentation is gone: CS/DS/ES/SS have base 0 and no limit, and only
  FS/GS keep a base, from an MSR (`fs_base`/`gs_base`) rather than a
  descriptor. A **non-canonical** linear address is a `#GP` *before* the page
  tables, so the unused middle of the address space is a hole, not an alias.
- **Long-mode interrupts** are a different shape and every difference bites.
  A gate is **sixteen** bytes, with the 64-bit offset in three pieces and an
  IST index sharing a byte with the reserved field. The frame is five 8-byte
  words and **SS:RSP are pushed always**, privilege change or not — which is
  what makes `IRETQ` one shape rather than two; a handler that pops three
  words returns to nonsense. A stack switch loads a **null** SS (a 64-bit TSS
  has no SS0 to load). An **IST** index in the gate switches stacks
  *unconditionally*, which is the point: a double fault or an NMI must reach a
  stack that is good even when the one in RSP is what broke. See `long_int`.
- **`SYSCALL` does not switch stacks.** It reads its entry point from LSTAR,
  puts the return address in RCX and the flags in R11, clears the RFLAGS bits
  SFMASK names, and takes its segments from STAR — consulting no table and
  touching no memory, which is what makes it fast. It therefore lands in the
  kernel still on the *user* stack, which is why `SWAPGS` exists and why every
  64-bit entry stub starts with it.
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
  `--kernel` CLI mode drives it.

  Two `boot_params` fields are load-bearing and easy to leave blank.
  **`screen_info`** (the first 0x40 bytes) must describe the text console:
  `vgacon_startup()` treats zero rows or columns as "never filled in" and
  falls back to the dummy console, which discards every message — a boot that
  looks dead but is not. **`type_of_loader`** must be non-zero or `setup_arch`
  ignores `ramdisk_image` entirely, so the initrd is never reserved, never
  unpacked and never mentioned. An initrd is passed with `--initrd`; it is
  placed as high in low memory as it fits, as a bootloader does. There is also `load_elf_kernel` (driven by
  `--kernel-elf`), which loads an already-decompressed kernel ELF directly —
  the path a bootloader uses for an uncompressed kernel, and a way to boot a
  kernel without running the in-kernel decompressor (which the emulator does
  not yet execute correctly). The decompressed ELF is extracted from the
  bzImage with `images/parse_bz4.py` (saved as `images/golden_kernel.bin`).

## Performance

The number to watch is not the rate but whether a boot still completes:
measure by timing a **fixed instruction count**, since a change that breaks
the boot can otherwise look like a speed-up.

```sh
time ./target/release/x86emu --kernel-elf images/golden_kernel.bin \
    --initrd images/root.bin --cmdline "root=/dev/ram0 rw console=tty0" 300000000
```

Adding 64-bit and run-time-sized RAM cost roughly a seventh of the rate
(measured that way, against the same command on the previous commit, same
machine, best of two). That is the price of bounds-checked memory instead of a
compile-time mask, a register file that is 64 bits wide underneath every
narrow access, and width-parameterised ALU and shift paths. It is worth
knowing, and worth re-measuring the same way before believing any claim about
a change's cost. Key optimizations:

- **TLB.** `Cpu` has a 256-entry direct-mapped TLB (`tlb: [TlbEntry; 256]`)
  that caches linear-page → physical-page translations, 64 bits wide on both
  sides (a long-mode linear address is 48 bits, a physical one up to 52).
  `apply_paging()` checks the TLB first (fast path: one array index +
  comparison) and only walks the page tables on a miss. `flush_tlb()` clears all entries (called
  on MOV CR3 and when CR0.PG toggles). `invlpg()` invalidates a single entry
  (called by the INVLPG instruction, 0F 01 /7). Without the TLB, every byte
  fetch did a 2-read page-table walk; with it, ~99% of translations are a
  single array lookup.
- **Trace gating.** Instruction tracing (writing one line per instruction to
  `trace.txt`) is disabled by default and only enabled when the
  `X86EMU_TRACE` environment variable is set. The file handle is cached in
  the `Cpu` struct (`trace_file: Option<File>`) rather than opened and
  closed per instruction. When tracing is off, zero I/O happens.
- **Fast multi-byte memory access.** `read_u16`/`read_u32`/`read_u64` and
  their write counterparts have fast paths that read/write directly from the
  `data` slice using `get_unchecked` (after `slot()` and the VGA check),
  avoiding the per-byte branching of the byte-by-byte path. The 64-bit forms
  go through one unaligned 8-byte access rather than eight single-byte ones —
  they are the common case now, because a page-table entry is 8 bytes in two
  of the three paging modes.
- **`rip_mask`.** RIP wraps at 32 bits in a legacy mode and at 64 in long
  mode. Rather than test the mode on every byte fetched, `step()` computes the
  mask once per instruction and `advance_ip` ANDs with it.
- **Batched interrupt checks.** `deliver_hardware_interrupt()` ticks the
  PIT every instruction (for timing accuracy) but only checks the
  keyboard/IDE IRQs and calls `pic.acknowledge()` every 64 instructions
  (`IRQ_CHECK_INTERVAL`), reducing per-instruction overhead.

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

### Things not to undo

Each of these looks like an oddity and is a fix. They cost a lot to find, and
every one of them is the sort of thing a tidy-up would quietly revert.

- **Widths are explicit, never a bool.** `Shift`/`ShiftImm` carry `width: u32`
  (8/16/32/64) and the bit-test instructions carry a `BitOffset`. A `w: bool`
  with `!w32` at the call site is exactly how 32-bit shifts came to run as
  8-bit ones — and the same shape came back with REX.W, where a hard-coded
  `if w32 { 32 } else { 16 }` in the decoder made `shl $32,%rax` a no-op. Any
  instruction that carries a width takes it from `cpu.osize()`.
- **`w32` in the decoder selects a *shape*, not a width.** The "32-bit"
  `Inst` variants cover both 32 and 64 bits; `cpu.osize()` at execute time is
  what says which. That is why the prefix state survives into `execute`.
- **The narrow registers are views**, EFLAGS is `u32`, and port numbers are
  `u16`. Do not narrow any of them back. `regs` is `[u64; 16]` even in a
  16-bit boot: one store, many widths.
- **A 32-bit register write zero-extends; a 16- or 8-bit one does not.** That
  asymmetry is x86-64's, it is load-bearing for compiled 64-bit code
  (`mov $0,%eax` is how RAX gets cleared), and it costs nothing in legacy
  modes where nothing can observe the upper half. See `set_reg_w`.
- **`rm_raw` is not a duplicate of `rm`.** Every *addressing* decision is made
  on the raw three bits — `rm == 100b` means a SIB byte follows, `mod == 00,
  rm == 101b` means RIP-relative — while `rm` is the REX-extended register
  index. Test the extended value and R12 becomes a SIB escape, R13 becomes a
  RIP-relative operand.
- **Descriptor tables are named by *linear* address**, and are translated as
  such (`Cpu::linear_to_phys_ro`). This was invisible for a long time because
  masking every physical address into a 128 MiB store folded `0xC02C1000` onto
  `0x002C1000` — which is exactly what the kernel's direct map maps it to. The
  coincidence dies the moment a machine has more RAM than the mask, so an
  unbacked address now reads as an open bus and the translation is done
  properly.
- **`long64()`, not `long_mode()`,** decides operand width, address width,
  stack width and register width. A 64-bit kernel runs 32-bit processes in
  compatibility mode with LMA still set; keying off the machine rather than
  the code segment runs them as 64-bit code.
- **`EFER.LMA` is the CPU's bit, not software's.** `write_msr` preserves it
  and `update_long_mode()` derives it from `CR0.PG && EFER.LME`. Letting a
  WRMSR set it directly makes long mode look active with no page tables under
  it.
- **The long-mode interrupt frame always carries SS:RSP**, and `IRETQ` always
  pops five words. There is no privilege-dependent variant to "optimize" into.
- **CMP and TEST commit nothing**; the ALU r/m↔reg forms check `AluOp::Cmp`
  before writing back, exactly as the immediate forms always did.
- **Stores translate as stores.** `translate_write` / `modrm_addr*_write`, not
  the read-side helpers — that is what applies `CR0.WP` and the user/supervisor
  check. The moffs and x87 store forms were on the read path and silently
  bypassed both.
- **Segment overrides apply to every memory operand**, including the `moffs`
  forms and string sources. `mov %gs:0xC,%eax` is how i386 userspace reads
  TLS.
- **`REP` loops check `pending_exception` each iteration** and leave the index
  and count registers at the element that faulted.
- **The instruction-fetch cache is dropped when a fetch lands on a new page**,
  including the case where a 16-bit fetch *ends* at offset 0xFFF.
- **`HLT` waits for an interrupt** when IF is set and only ends the run when
  it is clear, and maskable interrupts are delivered only when IF is set.
- **`X86EMU_*` diagnostics stay.** They are the only reason a bug 47 million
  instructions into a boot is findable at all, and they cost nothing when
  unset. Do not "clean them up".

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
- [x] Scale the RAM: sized at run time (`--mem`), with more than 4 GiB wired above the 32-bit MMIO hole and described that way in the BIOS map and `boot_params`.
- [x] 64-bit long mode: REX, sixteen 64-bit registers, RIP-relative addressing, PAE and 4-level paging with NX and 1 GiB pages, 16-byte interrupt gates with an IST, `SYSCALL`/`SYSRET`/`SWAPGS`, the FS/GS base MSRs, compatibility mode, and `--long` / `--kernel-elf64`.
- [x] Boot a real OS (Linux, 32-bit): **done — it reaches a busybox shell.**
      The kernel prints its whole log to the emulated VGA console, mounts an
      ext2 root filesystem from an initrd, and runs `/sbin/init` in ring 3.
      What it took is written up in the README's roadmap entry; the short
      version is that almost none of it was missing features and almost all of
      it was subtly wrong semantics, each found by a crash thousands of
      instructions downstream. See "Debugging a boot" below for the tools that
      make that tractable.
- [ ] **SSE/SSE2**, the missing piece for a 64-bit *userspace*. The x86-64 ABI
      requires SSE2, so a 64-bit Linux boot needs it before init can run —
      `movaps`/`movdqa`/`movq` and the XMM register file are the next layer,
      and they are a SIMD problem rather than a 64-bit one. The 64-bit CPU
      core is complete and tested without them; a 64-bit kernel is not
      claimed, and the tests say what is actually verified.
- [ ] A 64-bit Linux boot, once SSE is in: `--kernel-elf64` already loads an
      ELF64 kernel, builds `boot_params` and hands over in long mode.
- [ ] Keyboard input: the kernel's i8042 probe fails (`Can't read CTR`), so
      the shell that comes up cannot be typed at. `src/kbd.rs` implements the
      scancode/status side but not the controller command interface
      (0x64 commands 0x20/0x60/0xAA/0xAB and the responses they expect).
- [ ] Run the in-kernel bzImage decompressor, so `--kernel` works end to end
      and `--kernel-elf` becomes a convenience rather than the only route.
- [ ] Data accesses that straddle a page boundary translate once, from the
      first byte's page. Nothing hit it during the boot (the compiler keeps
      such accesses aligned), but it is wrong and worth fixing before it
      surprises someone: the fetch path already handles the equivalent case.

## Debugging a boot

A kernel does not fail where it breaks. Every bug in this project's Linux
effort announced itself thousands or millions of instructions after the
instruction that caused it, usually as a segfault in unrelated code. These are
the tools that turn that into a bounded search; they are all environment
variables, all off by default, and none of them cost anything when unset.

1. **Name the address.** `python tools/kallsyms.py images/golden_kernel.bin >
   images/kernel.syms` extracts a symbol table from the stripped kernel (it
   scans `.rodata` for the kallsyms table), and `python tools/sym.py C02EF525`
   turns an address into `early_page_fault+0x5`. Do this first: "stuck at
   C01A2BD0" is not a lead, "stuck in `delay_loop`" is.
2. **See the state.** `X86EMU_DEBUG=<n>` prints the exception log (vector,
   error code, faulting EIP, CR2 — the first ones, which is what you want),
   per-vector interrupt counts with the IDT target each resolves to, PIT/PIC
   state, user-mode instruction and ring-switch counts, and the last `n`
   instruction pointers. Any unimplemented opcode is reported **always**, with
   a hit count and where it was first seen.
3. **Read the kernel's own log.** Console output only appears on the emulated
   screen once `screen_info` is right and the console is registered — but
   `printk` has been filling its ring buffer since the first line. Dump RAM
   with `X86EMU_DUMP_PHYS=0:8000000:mem.bin` and search it for
   `Linux version`; the panic and its call trace are in there, with symbol
   names already resolved by the kernel.
4. **Stop at the right instant.** `X86EMU_TRAP_EIP=<hex>` halts the moment
   execution reaches an address (a full 64-bit one in long mode), `X86EMU_TRAP_USER=<n>` before the `n`th
   user-mode instruction. Combine with `X86EMU_DEBUG` to get the ring buffer
   at that point, or with `X86EMU_TRACE=1` + `X86EMU_TRACE_FROM=<n>` for a
   full instruction trace of just the window that matters.
5. **Find who wrote the bad value.** `X86EMU_WATCH=<linear>` and
   `X86EMU_WATCH_PHYS=<phys>` log stores through the translation path;
   `X86EMU_WATCH_STORE=<phys>` logs them at the memory layer with the value
   and the EIP that wrote it, which is the one that cannot be fooled. A store
   that the translation watch misses but the memory watch sees is itself the
   finding: something is writing through a path that skips the write-side
   translation.
6. **Make the run reproducible.** `X86EMU_EPOCH=<secs>` pins the RTC, so two
   runs execute the same instructions in the same order and an instruction
   count from one run is meaningful in the next. Without it the host clock
   leaks in and the boot drifts.
7. **Rule out the caches.** `X86EMU_NO_TLB=1` walks the page tables on every
   access. If the symptom survives, it is not a stale translation.

`X86EMU_DUMP_LINEAR=<hexaddr>:<hexlen>:<file>` dumps guest *virtual* memory
through the current page tables, which is the only way to read a user
process's address space — useful for pulling out a shared library's code to
decode by hand.

## Common tasks

- **Add an instruction:** add a variant to the `Inst` enum in
  `instructions.rs`, add its opcode case(s) in `decode`, add the execution
  logic in `execute`, then add a test. If it has an operand size, take the
  width from `cpu.osize()` in `execute` and use `read_rm_w`/`write_rm_w` and
  `reg_w`/`set_reg_w` — that is what makes it work at 16, 32 **and** 64 bits
  from one implementation. Only reach for a fixed-width helper when the
  instruction genuinely has one width (`SETcc` is always a byte).
- **Add a 64-bit test:** `long_cpu(&code)` in `instructions.rs`'s tests builds
  a machine already in long mode through the real boot path
  (`boot::load_flat64`), so a test exercises the same handover a payload gets.
  `run64` runs to `HLT` with a bound and fails rather than hanging.
- **Add a BIOS service:** add a method to `Bios` in `bios.rs`, dispatch it from
  the `INT` executor, and add a test.
- **Add a device:** create a module (e.g. `src/pit.rs`), add a field to the
  `Cpu` struct, route its ports through `Cpu::port_in`/`port_out`, wire any
  IRQ into `deliver_hardware_interrupt`, and add tests.
- **Regenerate the demo boot sector:** edit `gen_boot.py`, then run
  `python gen_boot.py` to rewrite `examples/boot.bin`.
- **Boot Linux:** build in release (a boot is ~300 million instructions; the
  debug build is far too slow) and run

  ```sh
  ./target/release/x86emu --kernel-elf images/golden_kernel.bin \
      --initrd images/root.bin \
      --cmdline "root=/dev/ram0 rw console=tty0" 3000000000
  ```

  The emulated text screen is printed at the end. `--kernel <bzImage>` is the
  boot-protocol path; it still stops in the in-kernel decompressor, which is
  why `--kernel-elf` takes the already-decompressed ELF.

  If it breaks, work through "Debugging a boot" above rather than guessing:
  resolve the address to a symbol, read the kernel's own printk buffer out of
  RAM, and only then start tracing. A missing instruction is reported
  automatically; anything else is almost certainly a semantic bug in an
  instruction that already exists.
