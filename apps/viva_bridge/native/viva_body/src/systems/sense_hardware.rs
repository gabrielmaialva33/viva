use crate::prelude::*;
use crate::resources::host_sensor::HostSensor;
use crate::components::cpu_sense::CpuSense;
use crate::components::gpu_sense::GpuSense;
use crate::components::memory_sense::MemorySense;
use crate::components::thermal_sense::ThermalSense;

pub fn sense_hardware_system(
    mut sensor: ResMut<HostSensor>,
    mut cpu_query: Query<&mut CpuSense>,
    mut gpu_query: Query<&mut GpuSense>,
    mut mem_query: Query<&mut MemorySense>,
    mut thermal_query: Query<&mut ThermalSense>,
) {
    // 1. Read CPU
    let cpu_data = sensor.0.read_cpu();
    for mut cpu in cpu_query.iter_mut() {
        cpu.usage_percent = cpu_data.usage;
        cpu.freq_mhz = cpu_data.freq_mhz.clone();
        cpu.energy_uj = cpu_data.energy_uj;
        cpu.temp_celsius = cpu_data.temp; // Fixed: assigning to Option<f32>
    }

    // 2. Read GPU
    if let Some(gpu_data) = sensor.0.read_gpu() {
        for mut gpu in gpu_query.iter_mut() {
            gpu.usage_percent = gpu_data.usage;
            gpu.vram_used_mb = gpu_data.vram_used;
            gpu.vram_total_mb = gpu_data.vram_total;
            gpu.temp_celsius = gpu_data.temp;
            gpu.power_watts = gpu_data.power_watts;
        }
    }

    // 3. Read Memory
    let mem_data = sensor.0.read_memory();
    for mut mem in mem_query.iter_mut() {
        mem.used_percent = mem_data.used_percent;
        mem.available_gb = mem_data.available_gb;
        mem.swap_used_percent = mem_data.swap_percent;
        mem.total_gb = mem_data.total_gb;
    }

    // 4. Read Thermal (Aggregate)
    let thermal_data = sensor.0.read_thermal();
    for mut thermal in thermal_query.iter_mut() {
        thermal.highest_temp = thermal_data.highest_temp;
        thermal.thermal_pressure = thermal_data.pressure;
    }
}
