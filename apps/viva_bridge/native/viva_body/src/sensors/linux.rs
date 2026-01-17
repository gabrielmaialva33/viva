use super::trait_def::*;
use sysinfo::System;
use std::fs;
use std::path::Path;
use nvml_wrapper::Nvml;

pub struct LinuxSensor {
    sysinfo: System,
    rapl_path: Option<String>,
    hwmon_path: Option<String>,
    nvml: Option<Nvml>,
}

impl LinuxSensor {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let rapl = Self::find_rapl_path();
        let hwmon = Self::find_hwmon_temp();
        let nvml = Nvml::init().ok(); // Silently fail if no NVIDIA

        if nvml.is_some() {
            eprintln!("[viva_body] NVML initialized successfully.");
        } else {
            eprintln!("[viva_body] NVML not available.");
        }

        Self {
            sysinfo: sys,
            rapl_path: rapl,
            hwmon_path: hwmon,
            nvml: nvml,
        }
    }

    // ... (find_rapl_path, etc. same as before)
    fn find_rapl_path() -> Option<String> {
        let base = "/sys/class/powercap";
        if Path::new(base).exists() {
            let path = format!("{}/intel-rapl/intel-rapl:0/energy_uj", base);
            if Path::new(&path).exists() {
                return Some(path);
            }
        }
        None
    }

    fn find_hwmon_temp() -> Option<String> {
        if let Ok(entries) = fs::read_dir("/sys/class/hwmon") {
            for entry in entries.flatten() {
                let path = entry.path();
                let name_path = path.join("name");
                if let Ok(name) = fs::read_to_string(&name_path) {
                    let name = name.trim();
                    if name == "coretemp" || name == "k10temp" || name == "amdgpu" {
                        let temp_path = path.join("temp1_input");
                        if temp_path.exists() {
                             return Some(temp_path.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
        None
    }

    fn read_rapl(&self) -> Option<u64> {
        if let Some(path) = &self.rapl_path {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(val) = content.trim().parse::<u64>() {
                    return Some(val);
                }
            }
        }
        None
    }

    fn read_sys_thermal(&self) -> f32 {
        if let Some(path) = &self.hwmon_path {
             if let Ok(content) = fs::read_to_string(path) {
                 if let Ok(val) = content.trim().parse::<f32>() {
                     return val / 1000.0;
                 }
            }
        }
        40.0
    }
}

impl SensorReader for LinuxSensor {
    fn read_cpu(&mut self) -> CpuReading {
        self.sysinfo.refresh_cpu_all();
        let usage = self.sysinfo.global_cpu_usage();

        let freqs: Vec<f32> = self.sysinfo.cpus().iter()
            .map(|c| c.frequency() as f32)
            .collect();
        let temp = self.read_sys_thermal();

        CpuReading {
            usage,
            temp: Some(temp),
            energy_uj: self.read_rapl(),
            freq_mhz: freqs,
        }
    }

    fn read_gpu(&mut self) -> Option<GpuReading> {
        let nvml = self.nvml.as_ref()?;
        let device = nvml.device_by_index(0).ok()?;

        Some(GpuReading {
            usage: device.utilization_rates().ok().map(|u| u.gpu as f32),
            vram_used: device.memory_info().ok().map(|m| m.used / 1024 / 1024), // Bytes -> MB
            vram_total: device.memory_info().ok().map(|m| m.total / 1024 / 1024),
            temp: device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu).ok().map(|t| t as f32),
            power_watts: device.power_usage().ok().map(|p| p as f32 / 1000.0), // mW -> W
        })
    }

    fn read_memory(&mut self) -> MemoryReading {
        self.sysinfo.refresh_memory();
        MemoryReading {
            used_percent: (self.sysinfo.used_memory() as f64 / self.sysinfo.total_memory() as f64) as f32 * 100.0,
            available_gb: self.sysinfo.available_memory() as f32 / 1024.0 / 1024.0 / 1024.0,
            total_gb: self.sysinfo.total_memory() as f32 / 1024.0 / 1024.0 / 1024.0,
            swap_percent: (self.sysinfo.used_swap() as f64 / self.sysinfo.total_swap().max(1) as f64) as f32 * 100.0,
        }
    }

    fn read_thermal(&mut self) -> ThermalReading {
        let t = self.read_sys_thermal();
        ThermalReading {
            highest_temp: t,
            pressure: (t - 40.0).max(0.0) / 60.0,
        }
    }

    fn platform_name(&self) -> &'static str {
        "linux_nvml"
    }
}
