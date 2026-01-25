//! Wrapper to enforce Send/Sync on Bevy App
//! SAFETY: We only use this in a context (NIF) where we know we won't access
//! non-thread-safe resources across threads incorrectly. Bevy Apps are generally movable.

use bevy_app::App;

pub struct VivaBodyApp(pub App);

unsafe impl Send for VivaBodyApp {}
unsafe impl Sync for VivaBodyApp {}
