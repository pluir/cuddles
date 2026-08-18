// examples/gen_long64.rs — writes examples/long64.bin, a flat 64-bit program
// for x86emu's `--long` mode.
//
// Run with:  cargo run --release --example gen_long64
// Then:      cargo run --release -- --long examples/long64.bin
//
// The program runs in 64-bit long mode and does three things that only a
// 64-bit CPU can do, so that a machine which merely *claims* long mode fails
// visibly rather than quietly:
//
//   1. Loads a full 64-bit immediate (`movabs`) — the only instruction that
//      carries one — and doubles it, which needs a 64-bit add.
//   2. Uses R8-R15, the registers REX added, and RIP-relative addressing,
//      which is how position-independent 64-bit code finds its own data.
//   3. Writes the message to the VGA text buffer through a 64-bit pointer.
//
// The emulator prints the text screen and the full 64-bit register file when
// the program halts, so the result is readable without a debugger.

fn main() {
    let msg = b"Hello from 64-bit long mode!";
    let mut p: Vec<u8> = Vec::new();

    // movabs $0xB8000,%rdi        -- the VGA text buffer
    p.extend_from_slice(&[0x48, 0xBF]);
    p.extend_from_slice(&0xB8000u64.to_le_bytes());

    // lea msg(%rip),%rsi          -- displacement patched once the length of
    //                                the rest of the code is known
    let lea_at = p.len();
    p.extend_from_slice(&[0x48, 0x8D, 0x35, 0, 0, 0, 0]);

    // mov $0x0F,%ah               -- bright white on black
    p.extend_from_slice(&[0xB4, 0x0F]);

    // print:
    let print = p.len();
    p.push(0xAC); //                  lods %ds:(%rsi),%al
    p.extend_from_slice(&[0x84, 0xC0]); //   test %al,%al
    let je_at = p.len();
    p.extend_from_slice(&[0x74, 0]); //      je done   (patched below)
    p.extend_from_slice(&[0x66, 0x89, 0x07]); // mov %ax,(%rdi)
    p.extend_from_slice(&[0x48, 0x83, 0xC7, 0x02]); // add $2,%rdi
    let jmp_at = p.len();
    let back = (print as i64) - (jmp_at as i64 + 2);
    p.extend_from_slice(&[0xEB, back as i8 as u8]); // jmp print

    // done:
    let done = p.len();
    p[je_at + 1] = ((done as i64) - (je_at as i64 + 2)) as i8 as u8;

    // movabs $0x0123456789ABCDEF,%rax ; mov %rax,%r8 ; add %r8,%rax
    //
    // The result, 0x02468ACF13579BDE, does not fit in 32 bits: a CPU that
    // runs this as a 32-bit add leaves the top half of RAX at zero, which the
    // register dump shows immediately.
    p.extend_from_slice(&[0x48, 0xB8]);
    p.extend_from_slice(&0x0123_4567_89AB_CDEFu64.to_le_bytes());
    p.extend_from_slice(&[0x49, 0x89, 0xC0]); // mov %rax,%r8
    p.extend_from_slice(&[0x4C, 0x01, 0xC0]); // add %r8,%rax

    // hlt
    p.push(0xF4);

    // The message follows the code; RIP-relative addressing reaches it
    // without knowing where the program was loaded.
    let msg_at = p.len();
    p.extend_from_slice(msg);
    p.push(0);

    // Patch the LEA displacement: measured from the end of the instruction.
    let disp = (msg_at as i64) - (lea_at as i64 + 7);
    p[lea_at + 3..lea_at + 7].copy_from_slice(&(disp as i32).to_le_bytes());

    std::fs::write("examples/long64.bin", &p).expect("write long64.bin");
    println!("wrote examples/long64.bin ({} bytes)", p.len());
}
