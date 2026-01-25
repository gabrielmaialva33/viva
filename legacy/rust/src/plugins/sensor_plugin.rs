use crate::prelude::*;
use crate::resources::body_config::BodyConfig;
use crate::resources::host_sensor::HostSensor;
use crate::systems::calculate_stress::calculate_stress_system;
use crate::systems::sense_hardware::sense_hardware_system;
use std::time::Duration;

#[cfg(target_os = "linux")]
use crate::sensors::linux::LinuxSensor;
#[cfg(target_os = "linux")]
use crate::sensors::wsl::WslSensor;

#[cfg(target_os = "windows")]
use crate::sensors::windows::WindowsSensor;

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
use crate::sensors::fallback::FallbackSensor;

use crate::sensors::fallback::FallbackSensor;

#[cfg(target_os = "linux")]
fn is_wsl() -> bool {
    if let Ok(content) = std::fs::read_to_string("/proc/version") {
        content.to_lowercase().contains("microsoft") || content.to_lowercase().contains("wsl")
    } else {
        false
    }
}

pub struct SensorPlugin;

impl Plugin for SensorPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(target_os = "linux")]
        #[cfg(target_os = "linux")]
        {
            if is_wsl() {
                eprintln!("[viva_body] Detected WSL2 environment. Activating Nerve Bridge...");
                app.insert_resource(HostSensor(Box::new(WslSensor::new())));
            } else {
                app.insert_resource(HostSensor(Box::new(LinuxSensor::new())));
            }
        }

        #[cfg(target_os = "windows")]
        app.insert_resource(HostSensor(Box::new(WindowsSensor::new())));

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        app.insert_resource(HostSensor(Box::new(FallbackSensor::new())));

        app.insert_resource(BodyConfig::default());

        app.insert_resource(Time::<Fixed>::from_seconds(0.5));

        app.add_systems(
            FixedUpdate,
            (sense_hardware_system, calculate_stress_system).chain(),
        );
    }
}
