// examples/gen_add.rs — a tiny program that writes a test .bin for x86emu.
// Run with:  cargo run --release --example gen_add
// It writes examples/add.bin:  mov ax,0x1234 ; mov bx,2 ; add ax,bx ; hlt

fn main() {
    let prog: [u8; 9] = [
        0xB8, 0x34, 0x12, // mov ax, 0x1234
        0xBB, 0x02, 0x00, // mov bx, 2
        0x01, 0xD8,       // add ax, bx
        0xF4,             // hlt
    ];
    std::fs::write("examples/add.bin", &prog).expect("write add.bin");
    println!("wrote examples/add.bin ({} bytes)", prog.len());
}