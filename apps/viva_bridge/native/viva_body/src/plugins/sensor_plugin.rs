use crate::prelude::*;
use crate::resources::host_sensor::HostSensor;
use crate::resources::body_config::BodyConfig;
use crate::systems::sense_hardware::sense_hardware_system;
use crate::systems::calculate_stress::calculate_stress_system;
use std::time::Duration;

#[cfg(target_os = "linux")]
use crate::sensors::linux::LinuxSensor;

#[cfg(target_os = "windows")]
use crate::sensors::windows::WindowsSensor;

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
use crate::sensors::fallback::FallbackSensor;

pub struct SensorPlugin;

impl Plugin for SensorPlugin {
    fn build(&self, app: &mut App) {
        #[cfg(target_os = "linux")]
        app.insert_resource(HostSensor(Box::new(LinuxSensor::new())));

        #[cfg(target_os = "windows")]
        app.insert_resource(HostSensor(Box::new(WindowsSensor::new())));

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        app.insert_resource(HostSensor(Box::new(FallbackSensor::new())));

        app.insert_resource(BodyConfig::default());

        app.insert_resource(Time::<Fixed>::from_seconds(0.5));

        app.add_systems(FixedUpdate, (
            sense_hardware_system,
            calculate_stress_system
        ).chain());
    }
}
