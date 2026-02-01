//! Asset loading and material definitions for billiards visualization
//!
//! Handles creation of meshes, materials, and textures for all game objects.

use bevy::prelude::*;
use std::f32::consts::PI;

/// Resource holding all shared materials
#[derive(Resource)]
pub struct BilliardsMaterials {
    /// Cue ball material (white, glossy)
    pub cue_ball: Handle<StandardMaterial>,
    /// Materials for balls 1-15 (indexed by ball number - 1)
    pub ball_materials: Vec<Handle<StandardMaterial>>,
    /// Green felt table surface
    pub table_felt: Handle<StandardMaterial>,
    /// Wood material for table frame and rails
    pub table_wood: Handle<StandardMaterial>,
    /// Dark material for pockets
    pub pocket: Handle<StandardMaterial>,
    /// Rail cushion material (slightly different green)
    pub rail_cushion: Handle<StandardMaterial>,
}

/// Resource holding shared meshes
#[derive(Resource)]
pub struct BilliardsMeshes {
    /// Sphere mesh for balls
    pub ball: Handle<Mesh>,
    /// Table surface mesh
    pub table_surface: Handle<Mesh>,
    /// Table frame mesh
    pub table_frame: Handle<Mesh>,
    /// Pocket mesh (cylinder)
    pub pocket: Handle<Mesh>,
    /// Rail mesh
    pub rail: Handle<Mesh>,
}

/// Ball dimensions (regulation pool ball)
pub const BALL_RADIUS: f32 = 0.028575; // 57.15mm diameter = 2.25 inches
pub const BALL_RADIUS_VISUAL: f32 = 0.05; // Slightly larger for visibility

/// Table dimensions (regulation 9-foot table, scaled down)
pub const TABLE_LENGTH: f32 = 2.54;  // 254cm = 100 inches (playing surface)
pub const TABLE_WIDTH: f32 = 1.27;   // 127cm = 50 inches
pub const TABLE_HEIGHT: f32 = 0.05;  // Table surface thickness
pub const RAIL_HEIGHT: f32 = 0.04;   // Rail cushion height
pub const RAIL_WIDTH: f32 = 0.08;    // Rail width
pub const POCKET_RADIUS: f32 = 0.06; // Pocket opening radius

impl BilliardsMaterials {
    /// Create all materials for the billiards scene
    pub fn new(materials: &mut Assets<StandardMaterial>) -> Self {
        // Cue ball - pure white, very glossy
        let cue_ball = materials.add(StandardMaterial {
            base_color: Color::srgb(0.98, 0.98, 0.98),
            perceptual_roughness: 0.1,
            metallic: 0.0,
            reflectance: 0.8,
            ..default()
        });

        // Create materials for balls 1-15
        let ball_colors = [
            Color::srgb(1.0, 0.84, 0.0),   // 1 - Yellow
            Color::srgb(0.0, 0.0, 0.8),    // 2 - Blue
            Color::srgb(0.9, 0.1, 0.1),    // 3 - Red
            Color::srgb(0.5, 0.0, 0.5),    // 4 - Purple
            Color::srgb(1.0, 0.5, 0.0),    // 5 - Orange
            Color::srgb(0.0, 0.6, 0.2),    // 6 - Green
            Color::srgb(0.6, 0.1, 0.1),    // 7 - Maroon
            Color::srgb(0.02, 0.02, 0.02), // 8 - Black
            Color::srgb(1.0, 0.84, 0.0),   // 9 - Yellow stripe
            Color::srgb(0.0, 0.0, 0.8),    // 10 - Blue stripe
            Color::srgb(0.9, 0.1, 0.1),    // 11 - Red stripe
            Color::srgb(0.5, 0.0, 0.5),    // 12 - Purple stripe
            Color::srgb(1.0, 0.5, 0.0),    // 13 - Orange stripe
            Color::srgb(0.0, 0.6, 0.2),    // 14 - Green stripe
            Color::srgb(0.6, 0.1, 0.1),    // 15 - Maroon stripe
        ];

        let ball_materials: Vec<Handle<StandardMaterial>> = ball_colors
            .iter()
            .map(|&color| {
                materials.add(StandardMaterial {
                    base_color: color,
                    perceptual_roughness: 0.15,
                    metallic: 0.0,
                    reflectance: 0.7,
                    ..default()
                })
            })
            .collect();

        // Table felt - deep green with slight texture feel
        let table_felt = materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.35, 0.15),
            perceptual_roughness: 0.9,
            metallic: 0.0,
            reflectance: 0.1,
            ..default()
        });

        // Table wood - rich brown wood
        let table_wood = materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.2, 0.1),
            perceptual_roughness: 0.6,
            metallic: 0.0,
            reflectance: 0.3,
            ..default()
        });

        // Pocket - dark black
        let pocket = materials.add(StandardMaterial {
            base_color: Color::srgb(0.02, 0.02, 0.02),
            perceptual_roughness: 0.95,
            metallic: 0.0,
            reflectance: 0.05,
            ..default()
        });

        // Rail cushion - slightly different green
        let rail_cushion = materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.4, 0.18),
            perceptual_roughness: 0.7,
            metallic: 0.0,
            reflectance: 0.2,
            ..default()
        });

        Self {
            cue_ball,
            ball_materials,
            table_felt,
            table_wood,
            pocket,
            rail_cushion,
        }
    }
}

impl BilliardsMeshes {
    /// Create all meshes for the billiards scene
    pub fn new(meshes: &mut Assets<Mesh>) -> Self {
        // High-poly sphere for balls (smooth appearance)
        let ball = meshes.add(Sphere::new(BALL_RADIUS_VISUAL).mesh().uv(32, 18));

        // Table surface - simple box
        let table_surface = meshes.add(Cuboid::new(TABLE_LENGTH, TABLE_HEIGHT, TABLE_WIDTH));

        // Table frame - larger box under the surface
        let table_frame = meshes.add(Cuboid::new(
            TABLE_LENGTH + RAIL_WIDTH * 2.0,
            TABLE_HEIGHT * 2.0,
            TABLE_WIDTH + RAIL_WIDTH * 2.0,
        ));

        // Pocket - cylinder
        let pocket = meshes.add(Cylinder::new(POCKET_RADIUS, 0.1));

        // Rail - box for cushions
        let rail = meshes.add(Cuboid::new(1.0, RAIL_HEIGHT, RAIL_WIDTH));

        Self {
            ball,
            table_surface,
            table_frame,
            pocket,
            rail,
        }
    }
}

/// Calculate pocket positions on the table
pub fn pocket_positions() -> [Vec3; 6] {
    let half_length = TABLE_LENGTH / 2.0;
    let half_width = TABLE_WIDTH / 2.0;
    let y = TABLE_HEIGHT / 2.0 + 0.01; // Slightly above table surface

    [
        // Top row (left to right)
        Vec3::new(-half_length, y, -half_width), // Top-left corner
        Vec3::new(0.0, y, -half_width),          // Top-middle (side pocket)
        Vec3::new(half_length, y, -half_width),  // Top-right corner
        // Bottom row (left to right)
        Vec3::new(-half_length, y, half_width),  // Bottom-left corner
        Vec3::new(0.0, y, half_width),           // Bottom-middle (side pocket)
        Vec3::new(half_length, y, half_width),   // Bottom-right corner
    ]
}

/// Calculate rail positions and sizes
pub fn rail_configs() -> [(Vec3, Vec3, f32); 6] {
    let half_length = TABLE_LENGTH / 2.0;
    let half_width = TABLE_WIDTH / 2.0;
    let y = TABLE_HEIGHT / 2.0 + RAIL_HEIGHT / 2.0;
    let pocket_gap = POCKET_RADIUS * 1.5;

    // (position, scale, rotation_y)
    [
        // Top rail left
        (
            Vec3::new(-half_length / 2.0 - pocket_gap / 2.0, y, -half_width - RAIL_WIDTH / 2.0),
            Vec3::new(half_length - pocket_gap * 2.0, 1.0, 1.0),
            0.0,
        ),
        // Top rail right
        (
            Vec3::new(half_length / 2.0 + pocket_gap / 2.0, y, -half_width - RAIL_WIDTH / 2.0),
            Vec3::new(half_length - pocket_gap * 2.0, 1.0, 1.0),
            0.0,
        ),
        // Bottom rail left
        (
            Vec3::new(-half_length / 2.0 - pocket_gap / 2.0, y, half_width + RAIL_WIDTH / 2.0),
            Vec3::new(half_length - pocket_gap * 2.0, 1.0, 1.0),
            0.0,
        ),
        // Bottom rail right
        (
            Vec3::new(half_length / 2.0 + pocket_gap / 2.0, y, half_width + RAIL_WIDTH / 2.0),
            Vec3::new(half_length - pocket_gap * 2.0, 1.0, 1.0),
            0.0,
        ),
        // Left rail
        (
            Vec3::new(-half_length - RAIL_WIDTH / 2.0, y, 0.0),
            Vec3::new(TABLE_WIDTH - pocket_gap * 2.0, 1.0, 1.0),
            PI / 2.0,
        ),
        // Right rail
        (
            Vec3::new(half_length + RAIL_WIDTH / 2.0, y, 0.0),
            Vec3::new(TABLE_WIDTH - pocket_gap * 2.0, 1.0, 1.0),
            PI / 2.0,
        ),
    ]
}

/// Standard ball rack positions (triangle formation)
pub fn rack_positions() -> [Vec3; 15] {
    let ball_diameter = BALL_RADIUS_VISUAL * 2.0;
    let row_offset = ball_diameter * 0.866; // sqrt(3)/2 for equilateral triangle
    let start_x = TABLE_LENGTH / 4.0; // Foot spot area
    let y = TABLE_HEIGHT / 2.0 + BALL_RADIUS_VISUAL;

    let mut positions = [Vec3::ZERO; 15];
    let mut idx = 0;

    // 5 rows: 1, 2, 3, 4, 5 balls
    for row in 0..5 {
        let row_z_start = -(row as f32) * ball_diameter / 2.0;
        for col in 0..=row {
            positions[idx] = Vec3::new(
                start_x + (row as f32) * row_offset,
                y,
                row_z_start + (col as f32) * ball_diameter,
            );
            idx += 1;
        }
    }

    positions
}

/// Cue ball starting position
pub fn cue_ball_position() -> Vec3 {
    Vec3::new(
        -TABLE_LENGTH / 4.0,
        TABLE_HEIGHT / 2.0 + BALL_RADIUS_VISUAL,
        0.0,
    )
}
