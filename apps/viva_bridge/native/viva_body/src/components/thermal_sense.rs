use crate::prelude::*;

#[derive(Component, Default, Clone, Debug)]
pub struct ThermalSense {
    // Aggregated thermal zones
    pub highest_temp: f32,
    pub critical_zones: usize, // Count of zones near T_max
    pub thermal_pressure: f32, // 0.0 - 1.0 (how close to throttling)
}
