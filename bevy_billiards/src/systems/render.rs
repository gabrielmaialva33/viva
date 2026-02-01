//! Render systems for spawning and updating billiards entities
//!
//! Handles initial scene setup and visual updates.

use bevy::prelude::*;

use crate::assets::{
    pocket_positions, rail_configs, rack_positions, cue_ball_position,
    BilliardsMaterials, BilliardsMeshes, TABLE_HEIGHT,
};
use crate::components::{
    BilliardsTable, CueBall, OrbitCamera, Pocket, PoolBall, Rail,
};

/// System to spawn the billiards table and all static geometry
pub fn spawn_table(
    mut commands: Commands,
    materials: Res<BilliardsMaterials>,
    meshes: Res<BilliardsMeshes>,
) {
    // Table surface (felt)
    commands.spawn((
        Mesh3d(meshes.table_surface.clone()),
        MeshMaterial3d(materials.table_felt.clone()),
        Transform::from_xyz(0.0, 0.0, 0.0),
        BilliardsTable,
        Name::new("Table Surface"),
    ));

    // Table frame (wood under the felt)
    commands.spawn((
        Mesh3d(meshes.table_frame.clone()),
        MeshMaterial3d(materials.table_wood.clone()),
        Transform::from_xyz(0.0, -TABLE_HEIGHT * 1.5, 0.0),
        Name::new("Table Frame"),
    ));

    // Spawn pockets
    for (i, pos) in pocket_positions().iter().enumerate() {
        commands.spawn((
            Mesh3d(meshes.pocket.clone()),
            MeshMaterial3d(materials.pocket.clone()),
            Transform::from_translation(*pos)
                .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
            Pocket { index: i as u8 },
            Name::new(format!("Pocket {}", i)),
        ));
    }

    // Spawn rails
    for (i, (pos, scale, rotation)) in rail_configs().iter().enumerate() {
        commands.spawn((
            Mesh3d(meshes.rail.clone()),
            MeshMaterial3d(materials.rail_cushion.clone()),
            Transform::from_translation(*pos)
                .with_rotation(Quat::from_rotation_y(*rotation))
                .with_scale(*scale),
            Rail { index: i as u8 },
            Name::new(format!("Rail {}", i)),
        ));
    }

    info!("Table spawned with {} pockets and {} rails", 6, 6);
}

/// System to spawn all balls in initial positions
pub fn spawn_balls(
    mut commands: Commands,
    materials: Res<BilliardsMaterials>,
    meshes: Res<BilliardsMeshes>,
) {
    // Spawn cue ball
    commands.spawn((
        Mesh3d(meshes.ball.clone()),
        MeshMaterial3d(materials.cue_ball.clone()),
        Transform::from_translation(cue_ball_position()),
        CueBall,
        Name::new("Cue Ball"),
    ));

    // Spawn numbered balls in rack formation
    let positions = rack_positions();

    // Standard 8-ball rack order (8 in center)
    let ball_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

    for (i, &ball_num) in ball_order.iter().enumerate() {
        if i >= positions.len() {
            break;
        }

        let material = materials.ball_materials[(ball_num - 1) as usize].clone();

        commands.spawn((
            Mesh3d(meshes.ball.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(positions[i]),
            PoolBall::new(ball_num),
            Name::new(format!("Ball {}", ball_num)),
        ));
    }

    info!("Spawned cue ball and 15 numbered balls");
}

/// System to spawn lighting
pub fn spawn_lighting(mut commands: Commands) {
    // Main overhead light (like a pool table lamp)
    commands.spawn((
        PointLight {
            color: Color::srgb(1.0, 0.98, 0.9), // Warm white
            intensity: 800_000.0,
            range: 10.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 2.0, 0.0),
        Name::new("Main Light"),
    ));

    // Secondary fill lights for even illumination
    let fill_positions = [
        Vec3::new(-1.0, 1.5, -0.5),
        Vec3::new(1.0, 1.5, -0.5),
        Vec3::new(-1.0, 1.5, 0.5),
        Vec3::new(1.0, 1.5, 0.5),
    ];

    for (i, pos) in fill_positions.iter().enumerate() {
        commands.spawn((
            PointLight {
                color: Color::srgb(0.95, 0.95, 1.0), // Slightly cool
                intensity: 100_000.0,
                range: 5.0,
                shadows_enabled: false,
                ..default()
            },
            Transform::from_translation(*pos),
            Name::new(format!("Fill Light {}", i)),
        ));
    }

    // Ambient light for overall scene brightness
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.4, 0.4, 0.45),
        brightness: 200.0,
    });

    info!("Lighting setup complete");
}

/// System to spawn the camera
pub fn spawn_camera(mut commands: Commands) {
    let orbit = OrbitCamera {
        radius: 3.0,
        azimuth: 0.3,
        elevation: 0.7,
        focus: Vec3::ZERO,
        smoothing: 0.15,
    };

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(orbit.position()).looking_at(orbit.focus, Vec3::Y),
        orbit,
        Name::new("Main Camera"),
    ));

    info!("Camera spawned");
}

/// Plugin for render systems
pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Startup,
            (
                spawn_lighting,
                spawn_camera,
                spawn_table,
                spawn_balls,
            )
                .chain(),
        );
    }
}
