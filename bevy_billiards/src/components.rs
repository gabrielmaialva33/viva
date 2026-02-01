//! ECS Components for the billiards visualization
//!
//! Defines all component types used by the billiards game entities.

use bevy::prelude::*;

/// Marker component for the cue ball (white ball)
#[derive(Component, Debug, Clone, Copy)]
pub struct CueBall;

/// Component for numbered pool balls (1-15)
#[derive(Component, Debug, Clone, Copy)]
pub struct PoolBall {
    /// Ball number (1-15)
    pub number: u8,
    /// Whether this is a solid (1-7) or stripe (9-15)
    pub is_stripe: bool,
    /// Whether this ball has been pocketed
    pub is_pocketed: bool,
}

impl PoolBall {
    pub fn new(number: u8) -> Self {
        Self {
            number,
            is_stripe: number > 8,
            is_pocketed: false,
        }
    }

    /// Get the base color for this ball
    pub fn color(&self) -> Color {
        match self.number {
            1 => Color::srgb(1.0, 0.84, 0.0),   // Yellow
            2 => Color::srgb(0.0, 0.0, 0.8),    // Blue
            3 => Color::srgb(0.8, 0.0, 0.0),    // Red
            4 => Color::srgb(0.5, 0.0, 0.5),    // Purple
            5 => Color::srgb(1.0, 0.5, 0.0),    // Orange
            6 => Color::srgb(0.0, 0.5, 0.0),    // Green
            7 => Color::srgb(0.5, 0.0, 0.0),    // Maroon
            8 => Color::srgb(0.05, 0.05, 0.05), // Black (8-ball)
            9 => Color::srgb(1.0, 0.84, 0.0),   // Yellow stripe
            10 => Color::srgb(0.0, 0.0, 0.8),   // Blue stripe
            11 => Color::srgb(0.8, 0.0, 0.0),   // Red stripe
            12 => Color::srgb(0.5, 0.0, 0.5),   // Purple stripe
            13 => Color::srgb(1.0, 0.5, 0.0),   // Orange stripe
            14 => Color::srgb(0.0, 0.5, 0.0),   // Green stripe
            15 => Color::srgb(0.5, 0.0, 0.0),   // Maroon stripe
            _ => Color::WHITE,
        }
    }
}

/// Marker component for the billiards table
#[derive(Component, Debug, Clone, Copy)]
pub struct BilliardsTable;

/// Component for table pockets
#[derive(Component, Debug, Clone, Copy)]
pub struct Pocket {
    /// Pocket position index (0-5, clockwise from top-left)
    pub index: u8,
}

/// Component for table rails/cushions
#[derive(Component, Debug, Clone, Copy)]
pub struct Rail {
    /// Rail index (0-5)
    pub index: u8,
}

/// Marker component for the orbital camera
#[derive(Component, Debug, Clone)]
pub struct OrbitCamera {
    /// Distance from the focus point
    pub radius: f32,
    /// Horizontal angle in radians
    pub azimuth: f32,
    /// Vertical angle in radians (0 = horizontal, PI/2 = top-down)
    pub elevation: f32,
    /// Point the camera orbits around
    pub focus: Vec3,
    /// Smoothing factor for camera movement (0-1)
    pub smoothing: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            radius: 3.0,
            azimuth: 0.0,
            elevation: 0.8,
            focus: Vec3::ZERO,
            smoothing: 0.1,
        }
    }
}

impl OrbitCamera {
    /// Calculate the camera position from orbital parameters
    pub fn position(&self) -> Vec3 {
        let x = self.radius * self.elevation.cos() * self.azimuth.sin();
        let y = self.radius * self.elevation.sin();
        let z = self.radius * self.elevation.cos() * self.azimuth.cos();
        self.focus + Vec3::new(x, y, z)
    }
}

/// Resource for tracking training state from the NEAT controller
#[derive(Resource, Debug, Clone, Default)]
pub struct TrainingState {
    /// Current generation number
    pub generation: u32,
    /// Current genome's fitness score
    pub fitness: f32,
    /// Current game score (balls pocketed)
    pub score: u32,
    /// Number of balls remaining on the table
    pub balls_remaining: u8,
    /// Best fitness achieved this session
    pub best_fitness: f32,
    /// Total games played
    pub games_played: u32,
    /// Whether the simulation is paused
    pub paused: bool,
    /// Simulation speed multiplier
    pub speed_multiplier: f32,
}

/// Physics state received from WebSocket
#[derive(Debug, Clone, Default)]
pub struct BallState {
    /// Ball ID (0 = cue, 1-15 = numbered balls)
    pub id: u8,
    /// Position in 3D space
    pub position: Vec3,
    /// Velocity vector
    pub velocity: Vec3,
    /// Angular velocity
    pub angular_velocity: Vec3,
    /// Whether the ball is active (not pocketed)
    pub active: bool,
}

/// Resource holding the current physics state from the server
#[derive(Resource, Debug, Clone, Default)]
pub struct PhysicsState {
    /// State of all balls
    pub balls: Vec<BallState>,
    /// Timestamp of this state
    pub timestamp: f64,
    /// Whether we have received initial state
    pub initialized: bool,
}

/// Resource for WebSocket connection state
#[derive(Resource, Debug, Clone)]
pub struct NetworkState {
    /// WebSocket server URL
    pub server_url: String,
    /// Whether we're currently connected
    pub connected: bool,
    /// Number of messages received
    pub messages_received: u64,
    /// Last error message, if any
    pub last_error: Option<String>,
}

impl Default for NetworkState {
    fn default() -> Self {
        Self {
            server_url: "ws://127.0.0.1:9000".to_string(),
            connected: false,
            messages_received: 0,
            last_error: None,
        }
    }
}

/// Event for ball pocketed
#[derive(Event, Debug, Clone)]
pub struct BallPocketed {
    pub ball_number: u8,
    pub pocket_index: u8,
}

/// Event for game reset
#[derive(Event, Debug, Clone)]
pub struct GameReset;

/// Event for receiving network messages
#[derive(Event, Debug, Clone)]
pub struct NetworkMessage {
    pub payload: String,
}
