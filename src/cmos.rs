//! MC146818 CMOS RTC — the PC's real-time clock and configuration RAM.
//!
//! Two I/O ports: 0x70 selects a register (its top bit is the NMI-disable
//! line, not part of the index), 0x71 reads or writes the selected one. The
//! first 14 registers are the clock; the rest is battery-backed RAM the BIOS
//! uses to record the machine's configuration.
//!
//! Linux reads this on every boot (`mach_get_cmos_time`), and it does so by
//! spinning until the update-in-progress bit of status register A clears. A
//! machine with no RTC answers 0xFF on every port read, so UIP reads as set
//! and the kernel hangs there forever — which is why "no device" is not a
//! workable stand-in for this one.
//!
//! The clock is read from the host at reset and then advances with emulated
//! time, so the guest sees a plausible date without this file knowing
//! anything about the machine it runs on.

/// Register indices worth naming.
mod reg {
    pub const SECONDS: u8 = 0x00;
    pub const MINUTES: u8 = 0x02;
    pub const HOURS: u8 = 0x04;
    pub const WEEKDAY: u8 = 0x06;
    pub const DAY: u8 = 0x07;
    pub const MONTH: u8 = 0x08;
    pub const YEAR: u8 = 0x09;
    pub const STATUS_A: u8 = 0x0A;
    pub const STATUS_B: u8 = 0x0B;
    pub const STATUS_C: u8 = 0x0C;
    pub const STATUS_D: u8 = 0x0D;
    pub const CENTURY: u8 = 0x32;
}

/// Status B bit 1: hours are 24-hour rather than 12-hour with a PM flag.
const SB_24HOUR: u8 = 0x02;
/// Status B bit 2: values are binary rather than BCD.
const SB_BINARY: u8 = 0x04;

pub struct Cmos {
    /// Currently selected register (port 0x70, NMI bit masked off).
    pub index: u8,
    /// Battery-backed configuration RAM. The clock registers are computed on
    /// read rather than stored here.
    pub ram: [u8; 128],
    /// Wall-clock seconds since the Unix epoch at reset.
    pub epoch_secs: u64,
    /// Emulated seconds elapsed since reset.
    pub elapsed_secs: u64,
    /// True once the guest has written the NMI-disable bit (recorded so the
    /// state is observable; nothing here raises NMIs).
    pub nmi_disabled: bool,
}

impl Cmos {
    pub fn new() -> Self {
        // X86EMU_EPOCH pins the clock so a boot is reproducible instruction
        // for instruction; without it the guest sees the host's real time.
        if let Some(fixed) = std::env::var("X86EMU_EPOCH").ok().and_then(|v| v.parse().ok()) {
            let mut c = Cmos::blank();
            c.epoch_secs = fixed;
            return c;
        }
        let epoch_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            // A host clock before 1970 is not worth a panic; any fixed date
            // boots just as well.
            .unwrap_or(0);
        let mut ram = [0u8; 128];
        // Status B: 24-hour, BCD (the format the BIOS conventionally leaves).
        ram[reg::STATUS_B as usize] = SB_24HOUR;
        // Status D bit 7: the RTC's battery is good and the time is valid.
        // Linux does not gate on it, but firmware does, and a zero here reads
        // as "this clock is meaningless".
        ram[reg::STATUS_D as usize] = 0x80;
        // Equipment byte: one floppy, 80x25 colour display.
        ram[0x14] = 0x21;
        Cmos {
            index: 0,
            ram,
            epoch_secs,
            elapsed_secs: 0,
            nmi_disabled: false,
        }
    }

    /// A CMOS with the standard defaults and the clock at the epoch.
    fn blank() -> Self {
        let mut ram = [0u8; 128];
        ram[reg::STATUS_B as usize] = SB_24HOUR;
        ram[reg::STATUS_D as usize] = 0x80;
        ram[0x14] = 0x21;
        Cmos { index: 0, ram, epoch_secs: 0, elapsed_secs: 0, nmi_disabled: false }
    }

    /// Advance the clock by one second of emulated time.
    pub fn tick_second(&mut self) {
        self.elapsed_secs = self.elapsed_secs.wrapping_add(1);
    }

    /// Port 0x70: select a register. Bit 7 is the NMI-disable line.
    pub fn write_index(&mut self, val: u8) {
        self.nmi_disabled = val & 0x80 != 0;
        self.index = val & 0x7F;
    }

    /// Port 0x71 read.
    pub fn read_data(&mut self) -> u8 {
        let (y, mo, d, h, mi, s, wd) = self.civil_time();
        let bcd = self.ram[reg::STATUS_B as usize] & SB_BINARY == 0;
        let enc = |v: u32| if bcd { to_bcd(v) } else { v as u8 };
        match self.index {
            reg::SECONDS => enc(s),
            reg::MINUTES => enc(mi),
            reg::HOURS => enc(h),
            // Day of week is 1-7 with Sunday = 1.
            reg::WEEKDAY => enc(wd + 1),
            reg::DAY => enc(d),
            reg::MONTH => enc(mo),
            reg::YEAR => enc(y % 100),
            reg::CENTURY => enc(y / 100),
            // Status A: UIP (bit 7) is always clear. There is no window here
            // in which a read could catch the clock mid-update, so reporting
            // an update in progress would only ever stall the guest.
            reg::STATUS_A => self.ram[reg::STATUS_A as usize] & 0x7F,
            // Status C latches interrupt causes and clears on read.
            reg::STATUS_C => {
                let v = self.ram[reg::STATUS_C as usize];
                self.ram[reg::STATUS_C as usize] = 0;
                v
            }
            i => self.ram[(i & 0x7F) as usize],
        }
    }

    /// Port 0x71 write. The clock registers are derived, not stored, so a
    /// write to one is accepted and dropped rather than corrupting the date.
    pub fn write_data(&mut self, val: u8) {
        match self.index {
            reg::SECONDS | reg::MINUTES | reg::HOURS | reg::WEEKDAY
            | reg::DAY | reg::MONTH | reg::YEAR | reg::CENTURY => {}
            i => self.ram[(i & 0x7F) as usize] = val,
        }
    }

    /// Current date and time as (year, month, day, hour, minute, second,
    /// weekday), weekday 0 = Sunday.
    fn civil_time(&self) -> (u32, u32, u32, u32, u32, u32, u32) {
        let t = self.epoch_secs.wrapping_add(self.elapsed_secs);
        let days = (t / 86_400) as i64;
        let rem = (t % 86_400) as u32;
        let (y, mo, d) = civil_from_days(days);
        // 1970-01-01 was a Thursday (weekday 4 counting Sunday as 0).
        let wd = ((days + 4) % 7) as u32;
        (y as u32, mo, d, rem / 3600, (rem % 3600) / 60, rem % 60, wd)
    }
}

/// Days since 1970-01-01 to a civil (year, month, day).
///
/// Howard Hinnant's `civil_from_days`: exact for the whole proleptic
/// Gregorian calendar, and no table of month lengths to get wrong.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March = 0
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Binary to packed BCD, as the RTC reports its registers by default.
fn to_bcd(v: u32) -> u8 {
    (((v / 10) << 4) | (v % 10)) as u8
}

impl Default for Cmos {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(epoch: u64) -> Cmos {
        let mut c = Cmos::new();
        c.epoch_secs = epoch;
        c.elapsed_secs = 0;
        c
    }

    #[test]
    fn civil_conversion_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(59), (1970, 3, 1));
        // 2000-02-29 exists: 2000 is a leap year despite being a century.
        assert_eq!(civil_from_days(11016), (2000, 2, 29));
        assert_eq!(civil_from_days(19723), (2024, 1, 1));
    }

    #[test]
    fn reads_the_clock_in_bcd() {
        // 2024-01-01 00:00:00 UTC = 1704067200.
        let mut c = at(1_704_067_200);
        c.write_index(0x00);
        assert_eq!(c.read_data(), 0x00); // seconds
        c.write_index(0x04);
        assert_eq!(c.read_data(), 0x00); // hours
        c.write_index(0x07);
        assert_eq!(c.read_data(), 0x01); // day
        c.write_index(0x08);
        assert_eq!(c.read_data(), 0x01); // month
        c.write_index(0x09);
        assert_eq!(c.read_data(), 0x24); // year 24, BCD
        c.write_index(0x32);
        assert_eq!(c.read_data(), 0x20); // century
    }

    #[test]
    fn binary_mode_is_honoured() {
        let mut c = at(1_704_067_200 + 3600 * 13 + 59);
        c.write_index(reg::STATUS_B);
        c.write_data(SB_24HOUR | SB_BINARY);
        c.write_index(reg::HOURS);
        assert_eq!(c.read_data(), 13);
        c.write_index(reg::SECONDS);
        assert_eq!(c.read_data(), 59);
    }

    #[test]
    fn update_in_progress_never_reads_as_set() {
        // Linux spins on this bit; if it can ever read as 1 the boot hangs.
        let mut c = at(0);
        c.ram[reg::STATUS_A as usize] = 0xFF;
        c.write_index(reg::STATUS_A);
        assert_eq!(c.read_data() & 0x80, 0);
    }

    #[test]
    fn index_write_masks_the_nmi_bit() {
        let mut c = at(0);
        c.write_index(0x80 | 0x15);
        assert_eq!(c.index, 0x15);
        assert!(c.nmi_disabled);
    }

    #[test]
    fn configuration_ram_round_trips() {
        let mut c = at(0);
        c.write_index(0x40);
        c.write_data(0xA5);
        c.write_index(0x40);
        assert_eq!(c.read_data(), 0xA5);
    }

    #[test]
    fn status_c_clears_on_read() {
        let mut c = at(0);
        c.ram[reg::STATUS_C as usize] = 0x40;
        c.write_index(reg::STATUS_C);
        assert_eq!(c.read_data(), 0x40);
        c.write_index(reg::STATUS_C);
        assert_eq!(c.read_data(), 0x00);
    }
}
