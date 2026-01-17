use super::trait_def::*;
use sysinfo::System;

pub struct FallbackSensor {
    sysinfo: System,
}

impl FallbackSensor {
    pub fn new() -> Self {
        Self {
            sysinfo: System::new_all(),
        }
    }
}

impl SensorReader for FallbackSensor {
    fn read_cpu(&mut self) -> CpuReading {
        self.sysinfo.refresh_cpu_all();
        let usage = self.sysinfo.global_cpu_usage();
        CpuReading {
            usage,
            temp: None,
            energy_uj: None,
            freq_mhz: vec![],
        }
    }

    fn read_gpu(&mut self) -> Option<GpuReading> { None }

    fn read_memory(&mut self) -> MemoryReading {
        MemoryReading::default()
    }

    fn read_thermal(&mut self) -> ThermalReading {
        ThermalReading::default()
    }

    fn platform_name(&self) -> &'static str { "generic_fallback" }
}
