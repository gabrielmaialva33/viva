use super::trait_def::*;
use std::process::Command;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;
use sysinfo::System;

pub struct WslSensor {
    sysinfo: System,
    // Store metrics in atomics for lock-free read
    cpu_temp_bits: Arc<AtomicU32>,
    gpu_temp_bits: Arc<AtomicU32>,
    gpu_usage_bits: Arc<AtomicU32>,
    gpu_mem_bits: Arc<AtomicU32>,
}

impl WslSensor {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        // Init with default values (CPU 40C, GPU 0C/0%)
        let cpu_temp_bits = Arc::new(AtomicU32::new(f32::to_bits(40.0)));
        let gpu_temp_bits = Arc::new(AtomicU32::new(f32::to_bits(0.0)));
        let gpu_usage_bits = Arc::new(AtomicU32::new(f32::to_bits(0.0)));
        let gpu_mem_bits = Arc::new(AtomicU32::new(f32::to_bits(0.0)));

        let c_t = cpu_temp_bits.clone();
        let g_t = gpu_temp_bits.clone();
        let g_u = gpu_usage_bits.clone();
        let g_m = gpu_mem_bits.clone();

        // The "Nerve" Thread: Bridges the gap to Windows host
        thread::spawn(move || {
            loop {
                // 1. CPU Temp (PowerShell)
                let temp = Self::fetch_cpu_temp_via_powershell();
                c_t.store(f32::to_bits(temp), Ordering::Relaxed);

                // 2. GPU Stats (nvidia-smi CLI)
                // This is safer on WSL than linking libnvidia-ml.so dynamically if env is messy
                if let Some((t, u, m)) = Self::fetch_gpu_via_smi() {
                    g_t.store(f32::to_bits(t), Ordering::Relaxed);
                    g_u.store(f32::to_bits(u), Ordering::Relaxed);
                    g_m.store(f32::to_bits(m), Ordering::Relaxed);
                }

                thread::sleep(Duration::from_secs(2));
            }
        });

        Self {
            sysinfo: sys,
            cpu_temp_bits,
            gpu_temp_bits,
            gpu_usage_bits,
            gpu_mem_bits,
        }
    }

    fn fetch_cpu_temp_via_powershell() -> f32 {
        // Attempt to read via WMI
        // Note: Needs Admin usually. If fails, returns default.
        let output = Command::new("powershell.exe")
            .args(&[
                "-Command",
                "Get-WmiObject MSAcpi_ThermalZoneTemperature -Namespace root/wmi | Select -ExpandProperty CurrentTemperature",
            ])
            .output();

        if let Ok(o) = output {
            if let Ok(s) = String::from_utf8(o.stdout) {
                if let Ok(val) = s.trim().parse::<f32>() {
                    // WMI returns deci-Kelvin
                    return (val - 2732.0) / 10.0;
                }
            }
        }

        // Fallback: If we can't read CPU temp, assume 45C (warm)
        45.0
    }

    fn fetch_gpu_via_smi() -> Option<(f32, f32, f32)> {
        // nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader,nounits
        // Using absolute path for reliability on WSL
        let output = Command::new("/usr/lib/wsl/lib/nvidia-smi")
            .args(&[
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;

        let s = String::from_utf8(output.stdout).ok()?;
        let parts: Vec<&str> = s.split(',').map(|s| s.trim()).collect();

        if parts.len() >= 3 {
            let temp = parts[0].parse::<f32>().unwrap_or(0.0);
            let usage = parts[1].parse::<f32>().unwrap_or(0.0);
            let mem = parts[2].parse::<f32>().unwrap_or(0.0);
            return Some((temp, usage, mem));
        }
        None
    }
}

impl SensorReader for WslSensor {
    fn read_cpu(&mut self) -> CpuReading {
        self.sysinfo.refresh_cpu_all();
        let usage = self.sysinfo.global_cpu_usage();

        let freqs: Vec<f32> = self
            .sysinfo
            .cpus()
            .iter()
            .map(|c| c.frequency() as f32)
            .collect();

        // Get cached temp from background thread
        let temp_bits = self.cpu_temp_bits.load(Ordering::Relaxed);
        let temp = f32::from_bits(temp_bits);

        CpuReading {
            usage,
            temp: Some(temp),
            energy_uj: None, // RAPL not avail on WSL usually
            freq_mhz: freqs,
        }
    }

    fn read_gpu(&mut self) -> Option<GpuReading> {
        let temp = f32::from_bits(self.gpu_temp_bits.load(Ordering::Relaxed));
        let usage = f32::from_bits(self.gpu_usage_bits.load(Ordering::Relaxed));
        let mem = f32::from_bits(self.gpu_mem_bits.load(Ordering::Relaxed));

        // If we haven't fetched yet (0.0), maybe return None or just return 0s?
        // Better to return Some if we believe we have a GPU.
        // Assuming 4090 exists.

        Some(GpuReading {
            usage: Some(usage),
            vram_used: Some(mem as u64),
            vram_total: Some(24564), // Hardcoded 4090 24GB for now
            temp: Some(temp),
            power_watts: None,
        })
    }

    fn read_memory(&mut self) -> MemoryReading {
        self.sysinfo.refresh_memory();
        MemoryReading {
            used_percent: (self.sysinfo.used_memory() as f64 / self.sysinfo.total_memory() as f64)
                as f32
                * 100.0,
            available_gb: self.sysinfo.available_memory() as f32 / 1024.0 / 1024.0 / 1024.0,
            total_gb: self.sysinfo.total_memory() as f32 / 1024.0 / 1024.0 / 1024.0,
            swap_percent: (self.sysinfo.used_swap() as f64
                / self.sysinfo.total_swap().max(1) as f64) as f32
                * 100.0,
        }
    }

    fn read_thermal(&mut self) -> ThermalReading {
        let cpu_t = f32::from_bits(self.cpu_temp_bits.load(Ordering::Relaxed));
        let gpu_t = f32::from_bits(self.gpu_temp_bits.load(Ordering::Relaxed));

        let highest = cpu_t.max(gpu_t);

        ThermalReading {
            highest_temp: highest,
            // Simple pressure model: >60C starts pressure, >90C is full pressure
            pressure: (highest - 60.0).max(0.0) / 30.0,
        }
    }

    fn platform_name(&self) -> &'static str {
        "wsl_nerve_bridge"
    }
}
