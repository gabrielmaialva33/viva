use crate::prelude::*;
use crate::systems::evolve_dynamics::evolve_dynamics_system;

pub struct DynamicsPlugin;

impl Plugin for DynamicsPlugin {
    fn build(&self, app: &mut App) {
        // Run dynamics after sensing and stress calculation
        app.add_systems(FixedUpdate, evolve_dynamics_system.chain());
    }
}
