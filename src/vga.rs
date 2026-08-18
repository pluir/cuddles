//! VGA display device.
//!
//! Supports the standard text mode (80x25) and the two most common graphics
//! modes: 12h (640x480, 16 colours) and 13h (320x200, 256 colours). In a real
//! PC the framebuffer is memory-mapped at 0xA0000 (graphics) and 0xB8000
//! (text); here the device owns its framebuffer and the BIOS writes to it.
//!
//! The framebuffer is exposed for display (e.g. the CLI can dump it), and the
//! BIOS `INT 0x10` services read/write through this device.

/// Text mode width (standard VGA 80 columns).
pub const TEXT_COLS: usize = 80;
/// Text mode height (standard VGA 25 rows).
pub const TEXT_ROWS: usize = 25;
/// Graphics mode 13h width (320 pixels).
pub const MODE13_W: usize = 320;
/// Graphics mode 13h height (200 pixels).
pub const MODE13_H: usize = 200;
/// Graphics mode 12h width (640 pixels).
pub const MODE12_W: usize = 640;
/// Graphics mode 12h height (480 pixels).
pub const MODE12_H: usize = 480;

/// The VGA device.
/// CRTC register indices this emulator tracks.
mod crtc {
    /// Start address, high byte.
    pub const START_HI: u8 = 0x0C;
    /// Start address, low byte.
    pub const START_LO: u8 = 0x0D;
    /// Cursor location, high byte.
    pub const CURSOR_HI: u8 = 0x0E;
    /// Cursor location, low byte.
    pub const CURSOR_LO: u8 = 0x0F;
}

pub struct Vga {
    /// Currently selected CRTC register (port 0x3D4).
    pub crtc_index: u8,
    /// CRTC register file (port 0x3D5).
    pub crtc: [u8; 32],
    /// Current video mode (BIOS mode number).
    pub mode: u8,
    /// Text cells: `char | (attr << 8)`, 80x25.
    pub text: Vec<u16>,
    /// Graphics framebuffer (mode 13h: 320x200 bytes; mode 12h: 640x480).
    pub framebuffer: Vec<u8>,
    /// Framebuffer width in pixels.
    pub fb_w: usize,
    /// Framebuffer height in pixels.
    pub fb_h: usize,
    /// Cursor position (row, col) in text mode.
    pub cursor_row: u8,
    pub cursor_col: u8,
}

impl Vga {
    pub fn new() -> Self {
        Vga {
            crtc_index: 0,
            crtc: [0; 32],
            mode: 0x03,
            text: vec![0x0720; TEXT_COLS * TEXT_ROWS],
            framebuffer: Vec::new(),
            fb_w: 0,
            fb_h: 0,
            cursor_row: 0,
            cursor_col: 0,
        }
    }

    /// Set the video mode. Returns the mode actually set.
    pub fn set_mode(&mut self, mode: u8) -> u8 {
        self.mode = mode;
        match mode {
            0x12 => {
                self.fb_w = MODE12_W;
                self.fb_h = MODE12_H;
                self.framebuffer = vec![0; MODE12_W * MODE12_H];
            }
            0x13 => {
                self.fb_w = MODE13_W;
                self.fb_h = MODE13_H;
                self.framebuffer = vec![0; MODE13_W * MODE13_H];
            }
            _ => {
                // Text mode (0x00-0x03 etc).
                self.text = vec![0x0720; TEXT_COLS * TEXT_ROWS];
                self.fb_w = 0;
                self.fb_h = 0;
                self.framebuffer = Vec::new();
                self.cursor_row = 0;
                self.cursor_col = 0;
            }
        }
        self.mode
    }

    /// True if the current mode is a graphics mode.
    pub fn is_graphics(&self) -> bool {
        self.mode == 0x12 || self.mode == 0x13
    }

    /// Put a character at a text cell (with attribute).
    pub fn put_char_at(&mut self, row: usize, col: usize, ch: u8, attr: u16) {
        if row < TEXT_ROWS && col < TEXT_COLS {
            self.text[row * TEXT_COLS + col] = (attr << 8) | ch as u16;
        }
    }

    /// Put a single pixel in graphics mode. `color` is the palette index.
    pub fn put_pixel(&mut self, x: usize, y: usize, color: u8) {
        if self.is_graphics() && x < self.fb_w && y < self.fb_h {
            self.framebuffer[y * self.fb_w + x] = color;
        }
    }

    /// Read a pixel's colour index in graphics mode.
    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        if self.is_graphics() && x < self.fb_w && y < self.fb_h {
            self.framebuffer[y * self.fb_w + x]
        } else {
            0
        }
    }

    /// Scroll the text screen up one line.
    pub fn scroll(&mut self) {
        self.text.copy_within(TEXT_COLS.., 0);
        let last = TEXT_COLS * (TEXT_ROWS - 1);
        for i in last..TEXT_COLS * TEXT_ROWS {
            self.text[i] = 0x0720;
        }
    }
}

impl Vga {
    /// Port 0x3D4: select a CRTC register.
    pub fn write_crtc_index(&mut self, val: u8) {
        self.crtc_index = val & 0x1F;
    }

    /// Port 0x3D5: write the selected CRTC register.
    pub fn write_crtc_data(&mut self, val: u8) {
        self.crtc[self.crtc_index as usize] = val;
    }

    /// Port 0x3D5: read the selected CRTC register.
    pub fn read_crtc_data(&self) -> u8 {
        self.crtc[self.crtc_index as usize]
    }

    /// First character cell displayed, in cells from the start of the text
    /// window. This is how a text console scrolls: it leaves the characters
    /// where they are and moves the window over them.
    pub fn start_cell(&self) -> usize {
        ((self.crtc[crtc::START_HI as usize] as usize) << 8)
            | self.crtc[crtc::START_LO as usize] as usize
    }

    /// Cursor position, in cells from the start of the text window.
    pub fn cursor_cell(&self) -> usize {
        ((self.crtc[crtc::CURSOR_HI as usize] as usize) << 8)
            | self.crtc[crtc::CURSOR_LO as usize] as usize
    }
}

impl Default for Vga {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_mode_put_char() {
        let mut vga = Vga::new();
        vga.put_char_at(0, 0, b'A', 0x07);
        assert_eq!(vga.text[0] & 0xFF, b'A' as u16);
        assert_eq!(vga.text[0] >> 8, 0x07);
    }

    #[test]
    fn mode13_framebuffer_size() {
        let mut vga = Vga::new();
        vga.set_mode(0x13);
        assert!(vga.is_graphics());
        assert_eq!(vga.framebuffer.len(), MODE13_W * MODE13_H);
        vga.put_pixel(10, 10, 0x3C);
        assert_eq!(vga.get_pixel(10, 10), 0x3C);
    }

    #[test]
    fn mode12_framebuffer_size() {
        let mut vga = Vga::new();
        vga.set_mode(0x12);
        assert_eq!(vga.framebuffer.len(), MODE12_W * MODE12_H);
        vga.put_pixel(100, 100, 0x0F);
        assert_eq!(vga.get_pixel(100, 100), 0x0F);
    }

    #[test]
    fn out_of_bounds_pixel_ignored() {
        let mut vga = Vga::new();
        vga.set_mode(0x13);
        vga.put_pixel(500, 500, 0xFF); // out of range
        assert_eq!(vga.get_pixel(500, 500), 0);
    }

    #[test]
    fn scroll_moves_lines_up() {
        let mut vga = Vga::new();
        vga.put_char_at(1, 0, b'X', 0x07);
        vga.scroll();
        assert_eq!(vga.text[0] & 0xFF, b'X' as u16);
        assert_eq!(vga.text[TEXT_COLS] & 0xFF, b' ' as u16);
    }
}
