//! Physics synchronization systems
//!
//! Updates ball positions based on physics state from WebSocket.

use bevy::prelude::*;

use crate::assets::{BALL_RADIUS_VISUAL, TABLE_HEIGHT};
use crate::components::{CueBall, PhysicsState, PoolBall};

/// System to sync ball positions from physics state
pub fn sync_ball_positions(
    physics_state: Res<PhysicsState>,
    mut cue_query: Query<(&mut Transform, &mut Visibility), With<CueBall>>,
    mut ball_query: Query<(&mut Transform, &mut Visibility, &PoolBall), Without<CueBall>>,
) {
    // Skip if no physics data yet
    if !physics_state.initialized || physics_state.balls.is_empty() {
        return;
    }

    // Update cue ball (id = 0)
    if let Ok((mut transform, mut visibility)) = cue_query.get_single_mut() {
        if let Some(state) = physics_state.balls.iter().find(|b| b.id == 0) {
            if state.active {
                // Convert physics coordinates to render coordinates
                // Physics uses Y-up, we need to ensure proper mapping
                transform.translation = convert_physics_to_render(state.position);
                *visibility = Visibility::Visible;
            } else {
                *visibility = Visibility::Hidden;
            }
        }
    }

    // Update numbered balls (id = 1-15)
    for (mut transform, mut visibility, ball) in ball_query.iter_mut() {
        if let Some(state) = physics_state.balls.iter().find(|b| b.id == ball.number) {
            if state.active {
                transform.translation = convert_physics_to_render(state.position);
                *visibility = Visibility::Visible;

                // Apply rotation based on angular velocity (visual effect)
                // This is approximate - real ball rotation would need quaternion tracking
                if state.angular_velocity.length_squared() > 0.001 {
                    let axis = state.angular_velocity.normalize_or_zero();
                    let angle = state.angular_velocity.length() * 0.016; // ~60fps frame time
                    transform.rotate(Quat::from_axis_angle(axis, angle));
                }
            } else {
                *visibility = Visibility::Hidden;
            }
        }
    }
}

/// Convert physics coordinates to render coordinates
/// Physics engine might use different coordinate system
fn convert_physics_to_render(physics_pos: Vec3) -> Vec3 {
    // Ensure ball is above table surface
    Vec3::new(
        physics_pos.x,
        (TABLE_HEIGHT / 2.0 + BALL_RADIUS_VISUAL).max(physics_pos.y),
        physics_pos.z,
    )
}

/// System to interpolate ball positions for smooth rendering
pub fn interpolate_ball_positions(
    time: Res<Time>,
    physics_state: Res<PhysicsState>,
    mut cue_query: Query<&mut Transform, With<CueBall>>,
    mut ball_query: Query<(&mut Transform, &PoolBall), Without<CueBall>>,
) {
    // Skip if no physics data
    if !physics_state.initialized {
        return;
    }

    let dt = time.delta_secs();
    let lerp_factor = (dt * 30.0).min(1.0); // Smooth at 30Hz physics rate

    // Interpolate cue ball
    if let Ok(mut transform) = cue_query.get_single_mut() {
        if let Some(state) = physics_state.balls.iter().find(|b| b.id == 0 && b.active) {
            let target = convert_physics_to_render(state.position);
            transform.translation = transform.translation.lerp(target, lerp_factor);
        }
    }

    // Interpolate numbered balls
    for (mut transform, ball) in ball_query.iter_mut() {
        if let Some(state) = physics_state
            .balls
            .iter()
            .find(|b| b.id == ball.number && b.active)
        {
            let target = convert_physics_to_render(state.position);
            transform.translation = transform.translation.lerp(target, lerp_factor);
        }
    }
}

/// System to reset ball positions when game resets
pub fn handle_game_reset(
    mut cue_query: Query<(&mut Transform, &mut Visibility), With<CueBall>>,
    mut ball_query: Query<(&mut Transform, &mut Visibility, &mut PoolBall), Without<CueBall>>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    // R key resets to initial positions (for testing without server)
    if keyboard.just_pressed(KeyCode::KeyR) {
        // Reset cue ball
        if let Ok((mut transform, mut visibility)) = cue_query.get_single_mut() {
            transform.translation = crate::assets::cue_ball_position();
            *visibility = Visibility::Visible;
        }

        // Reset numbered balls
        let positions = crate::assets::rack_positions();
        let ball_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

        for (mut transform, mut visibility, mut ball) in ball_query.iter_mut() {
            if let Some(idx) = ball_order.iter().position(|&n| n == ball.number) {
                if idx < positions.len() {
                    transform.translation = positions[idx];
                    *visibility = Visibility::Visible;
                    ball.is_pocketed = false;
                }
            }
        }

        info!("Ball positions reset");
    }
}

/// Plugin for physics synchronization
pub struct PhysicsSyncPlugin;

impl Plugin for PhysicsSyncPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                sync_ball_positions,
                interpolate_ball_positions,
                handle_game_reset,
            )
                .chain(),
        );
    }
}
