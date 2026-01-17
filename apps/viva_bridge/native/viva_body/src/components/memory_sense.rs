use crate::prelude::*;

#[derive(Component, Default, Clone, Debug)]
pub struct MemorySense {
    pub used_percent: f32,
    pub available_gb: f32,
    pub total_gb: f32,
    pub swap_used_percent: f32,
}
