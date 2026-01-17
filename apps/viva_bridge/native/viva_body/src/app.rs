use crate::prelude::*;
// Plugins
use crate::plugins::sensor_plugin::SensorPlugin;
use crate::plugins::dynamics_plugin::DynamicsPlugin;
use crate::plugins::bridge_plugin::BridgePlugin;
// Components
use crate::components::cpu_sense::CpuSense;
use crate::components::gpu_sense::GpuSense;
use crate::components::memory_sense::MemorySense;
use crate::components::thermal_sense::ThermalSense;
use crate::components::bio_rhythm::BioRhythm;
use crate::components::emotional_state::EmotionalState;

// Explicit imports for headless plugins
use bevy_app::ScheduleRunnerPlugin;
use bevy_time::TimePlugin;
use bevy_log::LogPlugin;
use bevy_core::Name;
use bevy_core::{TaskPoolPlugin, TypeRegistrationPlugin, FrameCountPlugin};

pub fn create_body_app() -> App {
    let mut app = App::new();

    // 1. Core Plugins (Headless)
    app.add_plugins((
        TaskPoolPlugin::default(),
        TypeRegistrationPlugin::default(),
        FrameCountPlugin::default(),
        TimePlugin::default(),
        ScheduleRunnerPlugin::run_loop(std::time::Duration::from_secs_f64(1.0 / 60.0)),
        LogPlugin::default(),
    ));

    // 2. Custom Plugins
    app.add_plugins(SensorPlugin);
    app.add_plugins(DynamicsPlugin);
    app.add_plugins(BridgePlugin);

    // 3. Spawn Primary Body Entity
    app.world_mut().spawn((
        Name::new("VIVA Physical Body"),
        CpuSense::default(),
        GpuSense::default(),
        MemorySense::default(),
        ThermalSense::default(),
        BioRhythm::default(),
        EmotionalState::default(),
    ));

    app
}
