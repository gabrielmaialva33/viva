//! VIVA Billiards Visualization
//!
//! 3D pool/billiards visualization for NEAT training using Bevy 0.15+.
//! Receives physics state via WebSocket and renders with PBR materials.
//!
//! # Controls
//! - Right Mouse: Orbit camera
//! - Middle Mouse: Pan camera
//! - Scroll: Zoom in/out
//! - 1-4: Camera presets
//! - R: Reset ball positions
//! - Home: Reset camera
//! - Escape: Exit
//!
//! # Usage
//! ```bash
//! cargo run --release
//! ```

use bevy::prelude::*;
use bevy::render::{
    settings::{RenderCreation, WgpuSettings, WgpuFeatures},
    RenderPlugin,
};
use bevy::window::{PresentMode, WindowMode};

mod assets;
mod components;
mod network;
mod plugins;
mod systems;

use plugins::BilliardsPlugin;

fn main() {
    App::new()
        // Configure default plugins with custom window settings
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "VIVA Billiards - NEAT Training Visualization".to_string(),
                        resolution: (1600.0, 900.0).into(),
                        present_mode: PresentMode::AutoVsync,
                        mode: WindowMode::Windowed,
                        resizable: true,
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        // Disable features not supported by Mesa GL in WSL2
                        features: WgpuFeatures::empty(),
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
        )
        // Add our billiards plugin
        .add_plugins(BilliardsPlugin)
        // Input handling
        .add_systems(Update, handle_exit)
        .run();
}

/// Handle escape key to exit
fn handle_exit(keyboard: Res<ButtonInput<KeyCode>>, mut exit: EventWriter<AppExit>) {
    if keyboard.just_pressed(KeyCode::Escape) {
        exit.send(AppExit::Success);
    }
}
