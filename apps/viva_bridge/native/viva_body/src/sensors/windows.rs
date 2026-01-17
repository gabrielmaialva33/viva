use super::trait_def::*;
use nvml_wrapper::Nvml;
use sysinfo::{Components, System};

pub struct WindowsSensor {
    sysinfo: System,
    components: Components,
    nvml: Option<Nvml>,
}

impl WindowsSensor {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        // Components for temperature (if supported/allowed)
        let components = Components::new_with_refreshed_list();

        // Quietly attempt NVML init
        let nvml = Nvml::init().ok();
        if nvml.is_some() {
            // Log? Maybe not to spam stdout if capturing NIF
        }

        Self {
            sysinfo: sys,
            components,
            nvml,
        }
    }
}

impl SensorReader for WindowsSensor {
    fn read_cpu(&mut self) -> CpuReading {
        self.sysinfo.refresh_cpu_all();
        let usage = self.sysinfo.global_cpu_usage();

        let freqs: Vec<f32> = self
            .sysinfo
            .cpus()
            .iter()
            .map(|c| c.frequency() as f32)
            .collect();

        // Try to find CPU temp in components
        let mut temp: Option<f32> = None;
        self.components.refresh(true);
        for comp in &self.components {
            let label = comp.label().to_lowercase();
            if label.contains("cpu") || label.contains("core") || label.contains("package") {
                temp = Some(comp.temperature());
                break; // Just take the first one for now
            }
        }

        CpuReading {
            usage,
            temp,
            energy_uj: None, // Hard to get RAPL on Windows without driver
            freq_mhz: freqs,
        }
    }

    fn read_gpu(&mut self) -> Option<GpuReading> {
        let nvml = self.nvml.as_ref()?;
        let device = nvml.device_by_index(0).ok()?;

        Some(GpuReading {
            usage: device.utilization_rates().ok().map(|u| u.gpu as f32),
            vram_used: device.memory_info().ok().map(|m| m.used / 1024 / 1024),
            vram_total: device.memory_info().ok().map(|m| m.total / 1024 / 1024),
            temp: device
                .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                .ok()
                .map(|t| t as f32),
            power_watts: device.power_usage().ok().map(|p| p as f32 / 1000.0),
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
        // Aggregate mostly from components
        let mut max_t = 0.0;
        for comp in &self.components {
            let t = comp.temperature();
            if t > max_t {
                max_t = t;
            }
        }

        // If no sensors found (common on Windows without Admin), fallback to 40.0
        if max_t < 1.0 {
            max_t = 40.0;
        }

        ThermalReading {
            highest_temp: max_t,
            pressure: (max_t - 40.0).max(0.0) / 60.0,
        }
    }

    fn platform_name(&self) -> &'static str {
        "windows_nvml"
    }
}
