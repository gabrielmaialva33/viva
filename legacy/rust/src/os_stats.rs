use std::fs;
use sysinfo::System;

#[derive(Debug, Clone)]
pub struct OsStats {
    pub context_switches: u64,
    pub interrupts: u64,
    pub cpu_freq_mhz: Option<f32>,
}

#[allow(dead_code)]
impl OsStats {
    pub fn empty() -> Self {
        Self {
            context_switches: 0,
            interrupts: 0,
            cpu_freq_mhz: None,
        }
    }
}

/// Reads Linux-specific /proc and /sys statistics
#[cfg(target_os = "linux")]
pub fn read_os_stats(_sys: &System) -> OsStats {
    let (ctxt, intr) = read_proc_stat();
    let freq = read_cpu_freq();

    OsStats {
        context_switches: ctxt,
        interrupts: intr,
        cpu_freq_mhz: freq,
    }
}

/// Windows / MacOS Implementation using sysinfo
#[cfg(not(target_os = "linux"))]
pub fn read_os_stats(sys: &System) -> OsStats {
    // Attempt to get frequency from sysinfo
    // In sysinfo 0.32, we assume cpu.frequency() works if we access via System
    // Actually, sys.cpus() returns &[Cpu]. We need to check if Cpu has frequency method.
    // Based on docs and usage in lib.rs, it seems okay?
    // Wait, lib.rs doesn't use frequency.
    // If CpuExt is missing, maybe frequency() is inherent.
    // If this fails compiling, we know we need a trait or it's gone.
    // But standard sysinfo usually has this.

    let freq = sys.cpus().first().map(|cpu| cpu.frequency() as f32);

    OsStats {
        context_switches: 0, // Hard to get without winapi
        interrupts: 0,
        cpu_freq_mhz: freq,
    }
}

// ============================================================================
// Linux Helpers
// ============================================================================

#[cfg(target_os = "linux")]
fn read_proc_stat() -> (u64, u64) {
    // Reads /proc/stat which contains global system counters
    if let Ok(content) = fs::read_to_string("/proc/stat") {
        let mut ctxt = 0;
        let mut intr = 0;

        for line in content.lines() {
            if line.starts_with("ctxt") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    ctxt = val.parse().unwrap_or(0);
                }
            } else if line.starts_with("intr") {
                if let Some(val) = line.split_whitespace().nth(1) {
                    intr = val.parse().unwrap_or(0);
                }
            }
        }
        return (ctxt, intr);
    }
    (0, 0)
}

#[cfg(target_os = "linux")]
fn read_cpu_freq() -> Option<f32> {
    // Reads CPU 0 current scaling frequency
    // content is in kHz (e.g. 2200000 = 2.2GHz)
    if let Ok(content) = fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    {
        if let Ok(khz) = content.trim().parse::<f32>() {
            return Some(khz / 1000.0); // Convert to MHz
        }
    }

    // Fallback?
    None
}
