use crate::dynamics::OUParams;
use crate::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Resource, Clone, Serialize, Deserialize)]
pub struct BodyConfig {
    pub sensor_tick_rate_hz: f64,
    pub metabolism_enabled: bool,
    // Dynamics
    pub dt: f64,
    pub cusp_enabled: bool,
    pub cusp_sensitivity: f64,
    pub ou_params: [OUParams; 3],
    pub seed: u64,
}

impl Default for BodyConfig {
    fn default() -> Self {
        Self {
            sensor_tick_rate_hz: 2.0,
            metabolism_enabled: true,
            dt: 0.5,
            cusp_enabled: true,
            cusp_sensitivity: 0.5,
            ou_params: [
                OUParams {
                    theta: 0.3,
                    mu: 0.0,
                    sigma: 0.15,
                },
                OUParams {
                    theta: 0.5,
                    mu: 0.0,
                    sigma: 0.25,
                },
                OUParams {
                    theta: 0.2,
                    mu: 0.0,
                    sigma: 0.10,
                },
            ],
            seed: 0,
        }
    }
}
