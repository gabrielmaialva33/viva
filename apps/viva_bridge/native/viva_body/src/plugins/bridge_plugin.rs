use crate::prelude::*;
use crate::systems::sync_soul::sync_soul_system;
// use crate::resources::soul_channel::SoulChannel; // We expect resource to be inserted by main app

pub struct BridgePlugin;

impl Plugin for BridgePlugin {
    fn build(&self, app: &mut App) {
        // Run sync after dynamics (PostUpdate or end of Update)
        // We use FixedUpdate to match the tick rate, but at the end of the chain
        app.add_systems(FixedUpdate, sync_soul_system.in_set(BridgeSystemSet));
    }
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BridgeSystemSet;
