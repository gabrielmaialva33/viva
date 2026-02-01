//! Game systems organized by responsibility
//!
//! Each module handles a specific aspect of the visualization.

pub mod camera;
pub mod physics_sync;
pub mod render;
pub mod ui;

pub use camera::CameraPlugin;
pub use physics_sync::PhysicsSyncPlugin;
pub use render::RenderPlugin;
pub use ui::UiPlugin;
