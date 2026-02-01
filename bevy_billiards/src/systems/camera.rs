//! Camera systems for orbital camera control
//!
//! Provides smooth orbital camera with mouse/keyboard controls.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use std::f32::consts::{FRAC_PI_2, PI};

use crate::components::OrbitCamera;

/// Camera movement speed constants
const ORBIT_SENSITIVITY: f32 = 0.005;
const ZOOM_SENSITIVITY: f32 = 0.5;
const PAN_SENSITIVITY: f32 = 0.01;
const MIN_RADIUS: f32 = 1.0;
const MAX_RADIUS: f32 = 10.0;
const MIN_ELEVATION: f32 = 0.1;
const MAX_ELEVATION: f32 = FRAC_PI_2 - 0.05;
const KEYBOARD_ORBIT_SPEED: f32 = 2.0;
const KEYBOARD_ZOOM_SPEED: f32 = 3.0;

/// System to handle camera input and update orbital parameters
pub fn camera_input_system(
    mut camera_query: Query<&mut OrbitCamera>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    mouse_scroll: Res<AccumulatedMouseScroll>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    let Ok(mut orbit) = camera_query.get_single_mut() else {
        return;
    };

    let dt = time.delta_secs();

    // Mouse orbit (right mouse button drag)
    if mouse_buttons.pressed(MouseButton::Right) {
        let delta = mouse_motion.delta;
        orbit.azimuth -= delta.x * ORBIT_SENSITIVITY;
        orbit.elevation += delta.y * ORBIT_SENSITIVITY;
    }

    // Mouse pan (middle mouse button drag)
    if mouse_buttons.pressed(MouseButton::Middle) {
        let delta = mouse_motion.delta;
        // Calculate pan direction based on camera orientation
        let right = Vec3::new(orbit.azimuth.cos(), 0.0, -orbit.azimuth.sin());
        let forward = Vec3::new(-orbit.azimuth.sin(), 0.0, -orbit.azimuth.cos());
        orbit.focus += right * delta.x * PAN_SENSITIVITY;
        orbit.focus += forward * delta.y * PAN_SENSITIVITY;
    }

    // Mouse scroll zoom
    let scroll = mouse_scroll.delta.y;
    if scroll.abs() > 0.0 {
        orbit.radius -= scroll * ZOOM_SENSITIVITY;
    }

    // Keyboard controls
    // Orbit with arrow keys or WASD when holding Shift
    if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight) {
        if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
            orbit.azimuth += KEYBOARD_ORBIT_SPEED * dt;
        }
        if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
            orbit.azimuth -= KEYBOARD_ORBIT_SPEED * dt;
        }
        if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
            orbit.elevation += KEYBOARD_ORBIT_SPEED * dt;
        }
        if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
            orbit.elevation -= KEYBOARD_ORBIT_SPEED * dt;
        }
    }

    // Zoom with +/- or Q/E
    if keyboard.pressed(KeyCode::KeyQ) || keyboard.pressed(KeyCode::Minus) {
        orbit.radius += KEYBOARD_ZOOM_SPEED * dt;
    }
    if keyboard.pressed(KeyCode::KeyE) || keyboard.pressed(KeyCode::Equal) {
        orbit.radius -= KEYBOARD_ZOOM_SPEED * dt;
    }

    // Reset camera with Home key
    if keyboard.just_pressed(KeyCode::Home) {
        *orbit = OrbitCamera::default();
    }

    // Clamp values
    orbit.azimuth = orbit.azimuth.rem_euclid(2.0 * PI);
    orbit.elevation = orbit.elevation.clamp(MIN_ELEVATION, MAX_ELEVATION);
    orbit.radius = orbit.radius.clamp(MIN_RADIUS, MAX_RADIUS);
}

/// System to update camera transform from orbital parameters
pub fn camera_transform_system(
    mut camera_query: Query<(&OrbitCamera, &mut Transform), With<Camera3d>>,
) {
    for (orbit, mut transform) in camera_query.iter_mut() {
        let target_position = orbit.position();

        // Smooth interpolation (lerp)
        transform.translation = transform
            .translation
            .lerp(target_position, orbit.smoothing);

        // Always look at focus point
        transform.look_at(orbit.focus, Vec3::Y);
    }
}

/// System to handle preset camera views
pub fn camera_preset_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut camera_query: Query<&mut OrbitCamera>,
) {
    let Ok(mut orbit) = camera_query.get_single_mut() else {
        return;
    };

    // Number keys for preset views
    if keyboard.just_pressed(KeyCode::Digit1) {
        // Top-down view
        orbit.azimuth = 0.0;
        orbit.elevation = FRAC_PI_2 - 0.1;
        orbit.radius = 3.5;
        orbit.focus = Vec3::ZERO;
    }

    if keyboard.just_pressed(KeyCode::Digit2) {
        // Side view (player perspective)
        orbit.azimuth = 0.0;
        orbit.elevation = 0.4;
        orbit.radius = 2.5;
        orbit.focus = Vec3::ZERO;
    }

    if keyboard.just_pressed(KeyCode::Digit3) {
        // Corner view
        orbit.azimuth = PI / 4.0;
        orbit.elevation = 0.6;
        orbit.radius = 3.0;
        orbit.focus = Vec3::ZERO;
    }

    if keyboard.just_pressed(KeyCode::Digit4) {
        // End view (from behind cue ball area)
        orbit.azimuth = PI;
        orbit.elevation = 0.3;
        orbit.radius = 2.0;
        orbit.focus = Vec3::new(-0.5, 0.0, 0.0);
    }
}

/// Plugin for camera systems
pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                camera_input_system,
                camera_preset_system,
                camera_transform_system,
            )
                .chain(),
        );
    }
}
