//! PhysicsPlugin - Bevy integration for Jolt Physics
//!
//! Integrates physics simulation into the VIVA Body ECS.
//! Features:
//! - 60Hz physics simulation (Jolt)
//! - Deterministic tick-based timing (PCSX2 pattern)
//! - Qualia mapping (collisions → PAD emotions)
//! - Active Inference predictions (snapshot/rollback)

use super::{PhysicsWorld, PredictionEngine, QualiaProcessor, SimulationClock};
use crate::components::emotional_state::EmotionalState;
use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoSystemConfigs;

/// System set for physics systems
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PhysicsSet {
    /// Physics simulation step
    Step,
    /// Qualia mapping (physics → emotions)
    Qualia,
    /// Prediction (Active Inference)
    Predict,
}

/// Bevy plugin for Jolt Physics integration
pub struct PhysicsPlugin {
    /// Enable physics simulation
    pub enabled: bool,
    /// Enable Active Inference prediction
    pub prediction_enabled: bool,
    /// Delta time for physics step
    pub dt: f32,
}

impl Default for PhysicsPlugin {
    fn default() -> Self {
        Self {
            enabled: true,
            prediction_enabled: true,
            dt: 1.0 / 60.0,
        }
    }
}

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        // Insert resources
        app.insert_resource(PhysicsWorld::new());
        app.insert_resource(PredictionEngine::new());
        app.insert_resource(QualiaProcessor::new());
        app.insert_resource(SimulationClock::new()); // Deterministic tick-based timing
        app.insert_resource(PhysicsEnabled(self.enabled));
        app.insert_resource(PredictionEnabled(self.prediction_enabled));
        app.insert_resource(PhysicsDt(self.dt)); // Keep for backwards compat

        // Configure system sets: Step → Qualia → Predict
        app.configure_sets(
            Update,
            (PhysicsSet::Step, PhysicsSet::Qualia, PhysicsSet::Predict).chain(),
        );

        // Add systems
        if self.enabled {
            app.add_systems(
                Update,
                (
                    physics_step_system
                        .in_set(PhysicsSet::Step)
                        .run_if(physics_enabled),
                    qualia_system
                        .in_set(PhysicsSet::Qualia)
                        .run_if(physics_enabled),
                ),
            );
        }

        if self.prediction_enabled {
            app.add_systems(
                Update,
                prediction_system
                    .in_set(PhysicsSet::Predict)
                    .run_if(prediction_enabled)
                    .after(PhysicsSet::Qualia),
            );
        }
    }
}

/// Resource to toggle physics at runtime
#[derive(Resource)]
pub struct PhysicsEnabled(pub bool);

/// Resource to toggle prediction at runtime
#[derive(Resource)]
pub struct PredictionEnabled(pub bool);

/// Resource for physics delta time
#[derive(Resource)]
pub struct PhysicsDt(pub f32);

/// Run condition: physics enabled
fn physics_enabled(enabled: Res<PhysicsEnabled>) -> bool {
    enabled.0
}

/// Run condition: prediction enabled
fn prediction_enabled(enabled: Res<PredictionEnabled>) -> bool {
    enabled.0
}

/// System: Step physics simulation with deterministic clock
fn physics_step_system(
    mut physics: ResMut<PhysicsWorld>,
    mut clock: ResMut<SimulationClock>,
    _dt: Res<PhysicsDt>, // Kept for backwards compat, but use clock.delta_time()
) {
    // Advance logical tick first (deterministic)
    clock.advance();

    // Step physics with fixed delta time from clock
    physics.step(clock.delta_time());

    // Check if we should process events (checkpoint pattern from PCSX2)
    // Events are batched and processed at intervals, not every tick
    if clock.should_check_events() {
        clock.acknowledge_event_check();
        // Event processing happens in qualia_system
    }
}

/// System: Map physics events to emotional qualia
fn qualia_system(
    mut physics: ResMut<PhysicsWorld>,
    mut qualia: ResMut<QualiaProcessor>,
    mut emotional_query: Query<&mut EmotionalState>,
) {
    // Get collision events
    let events = physics.collision_events();
    if events.is_empty() {
        return;
    }

    // Process collisions → PAD delta
    let pad_delta = qualia.process_collisions(events);

    // Apply to emotional state
    if let Ok(mut emotional) = emotional_query.get_single_mut() {
        emotional.pleasure += pad_delta.pleasure;
        emotional.arousal += pad_delta.arousal;
        emotional.dominance += pad_delta.dominance;

        // Clamp to valid range
        emotional.pleasure = emotional.pleasure.clamp(-1.0, 1.0);
        emotional.arousal = emotional.arousal.clamp(-1.0, 1.0);
        emotional.dominance = emotional.dominance.clamp(-1.0, 1.0);
    }

    // Clear processed events
    physics.clear_collision_events();
}

/// System: Run Active Inference prediction (adaptive)
fn prediction_system(
    mut physics: ResMut<PhysicsWorld>,
    mut prediction: ResMut<PredictionEngine>,
    emotional_query: Query<&EmotionalState>,
) {
    // Get current emotional state
    let emotional = match emotional_query.get_single() {
        Ok(e) => e,
        Err(_) => return, // No entity to predict for
    };

    // Calculate stress from arousal (simplified)
    let stress = emotional.arousal.abs() as f32;

    // Adaptive frequency: only predict when stressed
    if !prediction.should_predict(stress) {
        return;
    }

    // Define possible actions (simplified)
    let possible_actions = vec![
        super::prediction::PredictedAction::Idle,
        super::prediction::PredictedAction::Move(glam::Vec3::X),
        super::prediction::PredictedAction::Move(glam::Vec3::NEG_X),
        super::prediction::PredictedAction::Move(glam::Vec3::Z),
        super::prediction::PredictedAction::Move(glam::Vec3::NEG_Z),
    ];

    // Run prediction
    prediction.predict_futures(&mut physics, emotional.arousal, &possible_actions);
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::App;

    #[test]
    fn test_plugin_builds() {
        let mut app = App::new();
        app.add_plugins(PhysicsPlugin::default());
        // Should not panic
    }

    #[test]
    fn test_plugin_disabled() {
        let mut app = App::new();
        app.add_plugins(PhysicsPlugin {
            enabled: false,
            prediction_enabled: false,
            dt: 1.0 / 60.0,
        });
        // Should not panic
    }

    #[test]
    fn test_qualia_integration() {
        // Test the full pipeline: Physics → Qualia → EmotionalState
        use super::super::{CollisionEvent, PadDelta, QualiaProcessor};
        use glam::Vec3;

        // 1. Create qualia processor
        let mut processor = QualiaProcessor::new();

        // 2. Simulate collision events (as if from physics)
        let events = vec![
            CollisionEvent {
                body1: 1,
                body2: 2,
                impulse: 50.0, // Medium impact
                contact_point: Vec3::ZERO,
            },
            CollisionEvent {
                body1: 1,
                body2: 3,
                impulse: 25.0, // Lighter impact
                contact_point: Vec3::new(1.0, 0.0, 0.0),
            },
        ];

        // 3. Process through qualia
        let pad_delta = processor.process_collisions(&events);

        // 4. Verify emotional response
        // Collisions should cause: -pleasure (pain), +arousal (startle), -dominance
        assert!(pad_delta.pleasure < 0.0, "Collisions should cause displeasure");
        assert!(pad_delta.arousal > 0.0, "Collisions should increase arousal");
        assert!(pad_delta.dominance < 0.0, "Being hit should reduce dominance");

        // 5. Simulate applying to EmotionalState
        let mut emotional = EmotionalState {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            in_bifurcation: false,
        };

        emotional.pleasure += pad_delta.pleasure;
        emotional.arousal += pad_delta.arousal;
        emotional.dominance += pad_delta.dominance;

        // 6. Verify state changed appropriately
        assert!(emotional.pleasure < 0.0);
        assert!(emotional.arousal > 0.0);
        assert!(emotional.dominance < 0.0);

        println!(
            "Qualia integration OK: P={:.3} A={:.3} D={:.3}",
            emotional.pleasure, emotional.arousal, emotional.dominance
        );
    }

    #[test]
    fn test_snapshot_prediction_flow() {
        // Test Active Inference flow: snapshot → simulate futures → rollback
        use super::super::PhysicsWorld;
        use glam::Vec3;

        let mut world = PhysicsWorld::new();

        // Setup scene
        let _floor = world.create_floor();
        let sphere = world.create_sphere(Vec3::new(0.0, 5.0, 0.0), 0.5);

        // Save state before prediction
        let snapshot = world.save_state();
        let initial_pos = world.get_position(sphere);

        // Simulate possible future A: let it fall
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
        let future_a_pos = world.get_position(sphere);

        // Rollback
        world.restore_state(&snapshot);
        let restored_pos = world.get_position(sphere);

        // Verify rollback worked
        assert!(
            (restored_pos.y - initial_pos.y).abs() < 0.1,
            "Rollback should restore position"
        );

        // Simulate possible future B: different action (same for now)
        world.set_velocity(sphere, Vec3::new(5.0, 0.0, 0.0)); // Push sideways
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
        let future_b_pos = world.get_position(sphere);

        // Futures should differ
        assert!(
            (future_a_pos.x - future_b_pos.x).abs() > 0.1,
            "Different actions should lead to different futures"
        );

        println!(
            "Prediction flow OK: initial={:.2}, future_a={:.2}, future_b_x={:.2}",
            initial_pos.y, future_a_pos.y, future_b_pos.x
        );
    }
}
