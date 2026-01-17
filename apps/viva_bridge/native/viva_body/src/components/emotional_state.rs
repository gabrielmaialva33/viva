use crate::prelude::*;
use serde::Serialize;

#[derive(Component, Default, Clone, Debug, Serialize)]
pub struct EmotionalState {
    pub pleasure: f64,
    pub arousal: f64,
    pub dominance: f64,
    pub in_bifurcation: bool,
}

impl EmotionalState {
    pub fn new() -> Self {
        Self {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            in_bifurcation: false,
        }
    }
}
