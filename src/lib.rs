//! x86emu — an incremental x86 PC emulator.
//!
//! This crate currently implements a 16-bit real-mode 8086-style CPU core:
//! registers, flags, 1 MiB of memory, an instruction decoder with ModR/M
//! addressing, and a first batch of common instructions. Protected mode,
//! paging, and PC devices will be layered on top of this in later stages.

pub mod cpu;
pub mod memory;
pub mod modrm;
pub mod instructions;
pub mod bios;
pub mod protected;
pub mod paging;
pub mod cmos;
pub mod pit;
pub mod pic;
pub mod vga;
pub mod kbd;
pub mod dma;
pub mod ide;
pub mod boot;
pub mod fpu;
pub mod sse;
pub mod vmx;

pub use cpu::Cpu;
pub use memory::Memory;