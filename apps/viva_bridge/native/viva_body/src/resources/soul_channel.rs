use crossbeam_channel::{Sender, Receiver, unbounded};
use crate::prelude::*;

#[derive(Clone, Debug)]
pub enum BodyUpdate {
    StateChanged {
        stress: f32,
        fatigue: f32,
        needs_rest: bool,
        cpu_usage: f32,
        gpu_usage: Option<f32>,
        gpu_temp: Option<f32>,
    },
    CriticalStress(f32),
    NeedsRest,
}

#[derive(Clone, Debug)]
pub enum SoulCommand {
    ApplyStimulus { p: f64, a: f64, d: f64 },
    SetTickRate(f32),
    Shutdown,
}

#[derive(Resource)]
pub struct SoulChannel {
    pub tx: Sender<BodyUpdate>,
    pub rx: Receiver<SoulCommand>,
}

// Struct to be held by NIF/Bridge side to communicate with Body
pub struct SoulBridge {
    pub rx: Receiver<BodyUpdate>,
    pub tx: Sender<SoulCommand>,
}

pub fn create_channel() -> (SoulChannel, SoulBridge) {
    let (body_tx, soul_rx) = unbounded();
    let (soul_tx, body_rx) = unbounded();

    (
        SoulChannel { tx: body_tx, rx: body_rx },
        SoulBridge { rx: soul_rx, tx: soul_tx },
    )
}
