//! Main billiards visualization plugin
//!
//! Combines all sub-plugins into a single cohesive plugin.

use bevy::prelude::*;

use crate::assets::{BilliardsMaterials, BilliardsMeshes};
use crate::components::TrainingState;
use crate::network::NetworkPlugin;
use crate::systems::{CameraPlugin, PhysicsSyncPlugin, RenderPlugin, UiPlugin};

/// Main plugin that bundles all billiards functionality
pub struct BilliardsPlugin;

impl Plugin for BilliardsPlugin {
    fn build(&self, app: &mut App) {
        app
            // Initialize resources first
            .init_resource::<TrainingState>()
            // Add setup systems that create shared resources
            .add_systems(PreStartup, setup_assets)
            // Add all sub-plugins
            .add_plugins((
                NetworkPlugin,
                RenderPlugin,
                CameraPlugin,
                PhysicsSyncPlugin,
                UiPlugin,
            ));

        info!("BilliardsPlugin initialized");
    }
}

/// System to create shared materials and meshes
fn setup_assets(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let billiards_materials = BilliardsMaterials::new(&mut materials);
    let billiards_meshes = BilliardsMeshes::new(&mut meshes);

    commands.insert_resource(billiards_materials);
    commands.insert_resource(billiards_meshes);

    info!("Billiards assets created");
}

/// Configuration for the billiards plugin
#[derive(Resource, Clone)]
pub struct BilliardsConfig {
    /// WebSocket server URL for physics updates
    pub server_url: String,
    /// Enable debug rendering (collision shapes, etc.)
    pub debug_render: bool,
    /// Maximum simulation speed multiplier
    pub max_speed: f32,
}

impl Default for BilliardsConfig {
    fn default() -> Self {
        Self {
            server_url: "ws://127.0.0.1:9000".to_string(),
            debug_render: false,
            max_speed: 10.0,
        }
    }
}

/// Extended plugin with configuration
pub struct BilliardsPluginConfigured {
    pub config: BilliardsConfig,
}

impl BilliardsPluginConfigured {
    pub fn new(config: BilliardsConfig) -> Self {
        Self { config }
    }
}

impl Plugin for BilliardsPluginConfigured {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.config.clone())
            .add_plugins(BilliardsPlugin);
    }
}
