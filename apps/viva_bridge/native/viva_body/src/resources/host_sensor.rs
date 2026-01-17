use crate::prelude::*;
use crate::sensors::trait_def::SensorReader;

// If SensorReader is Send + Sync, then Box<dyn SensorReader> is also Send + Sync automatically.
// We wrap it in a struct.
#[derive(Resource)]
pub struct HostSensor(pub Box<dyn SensorReader>);

// No unsafe impl needed because trait bound in trait_def.rs enforces Send + Sync
