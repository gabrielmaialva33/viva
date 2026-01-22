//! PredictionEngine - Active Inference for VIVA
//!
//! Simulates multiple possible futures and selects the best action.
//! Number of futures is dynamic based on arousal level.

use super::world::{PhysicsSnapshot, PhysicsWorld};
use bevy_ecs::system::Resource;
use glam::Vec3;

/// Configuration for prediction engine
#[derive(Clone, Debug)]
pub struct PredictionConfig {
    /// Minimum futures to simulate (when calm)
    pub min_futures: usize,
    /// Maximum futures to simulate (when agitated)
    pub max_futures: usize,
    /// How many physics ticks to look ahead
    pub lookahead_ticks: u32,
    /// Delta time per tick
    pub tick_delta: f32,
    /// Stress threshold for always-on prediction
    pub stress_threshold: f32,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            min_futures: 3,
            max_futures: 10,
            lookahead_ticks: 10,
            tick_delta: 1.0 / 60.0,
            stress_threshold: 0.5,
        }
    }
}

/// A possible future state
#[derive(Clone, Debug)]
pub struct FuturePrediction {
    /// Action that led to this future
    pub action: PredictedAction,
    /// Final position after simulation
    pub final_position: Vec3,
    /// Accumulated energy cost
    pub energy_cost: f32,
    /// Constraint violations encountered
    pub constraint_violations: u32,
    /// Predicted pleasure delta
    pub pleasure_delta: f64,
    /// Predicted arousal delta
    pub arousal_delta: f64,
    /// Predicted dominance delta
    pub dominance_delta: f64,
}

/// Action that can be simulated
#[derive(Clone, Debug, PartialEq)]
pub enum PredictedAction {
    /// Do nothing
    Idle,
    /// Move in direction
    Move(Vec3),
    /// Apply impulse
    Impulse(Vec3),
    /// Custom action
    Custom(String),
}

/// Prediction engine for Active Inference
#[derive(Resource)]
pub struct PredictionEngine {
    config: PredictionConfig,
    /// Current predictions (cleared each cycle)
    predictions: Vec<FuturePrediction>,
    /// Selected best action
    selected_action: Option<PredictedAction>,
    /// Last prediction error (for learning)
    last_prediction_error: f64,
}

impl PredictionEngine {
    /// Create new prediction engine
    pub fn new() -> Self {
        Self::with_config(PredictionConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: PredictionConfig) -> Self {
        Self {
            config,
            predictions: Vec::new(),
            selected_action: None,
            last_prediction_error: 0.0,
        }
    }

    /// Calculate how many futures to simulate based on arousal
    ///
    /// - arousal < 0.3 → min_futures (calm)
    /// - arousal >= 0.7 → max_futures (agitated)
    /// - otherwise → interpolate
    pub fn futures_count(&self, arousal: f64) -> usize {
        if arousal < 0.3 {
            self.config.min_futures
        } else if arousal >= 0.7 {
            self.config.max_futures
        } else {
            // Linear interpolation between 0.3 and 0.7
            let t = (arousal - 0.3) / 0.4;
            let range = self.config.max_futures - self.config.min_futures;
            self.config.min_futures + (t * range as f64) as usize
        }
    }

    /// Should run prediction this tick? (adaptive frequency)
    ///
    /// - stress > threshold → always run (2Hz effective)
    /// - stress <= threshold → on-demand only
    pub fn should_predict(&self, stress: f32) -> bool {
        stress > self.config.stress_threshold
    }

    /// Clear predictions for new cycle
    pub fn clear(&mut self) {
        self.predictions.clear();
        self.selected_action = None;
    }

    /// Record a simulated future
    pub fn record_future(&mut self, prediction: FuturePrediction) {
        self.predictions.push(prediction);
    }

    /// Select best action based on recorded futures
    ///
    /// Criteria:
    /// 1. Minimize constraint violations
    /// 2. Maximize pleasure delta
    /// 3. Minimize energy cost
    pub fn select_best_action(&mut self) -> Option<&PredictedAction> {
        if self.predictions.is_empty() {
            return None;
        }

        // Score each prediction (higher = better)
        let best_idx = self
            .predictions
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score_a = Self::score_prediction(a);
                let score_b = Self::score_prediction(b);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)?;

        self.selected_action = Some(self.predictions[best_idx].action.clone());
        self.selected_action.as_ref()
    }

    /// Score a prediction (higher = better)
    fn score_prediction(pred: &FuturePrediction) -> f64 {
        // Weights for different criteria
        const W_PLEASURE: f64 = 1.0;
        const W_VIOLATIONS: f64 = -0.5;
        const W_ENERGY: f64 = -0.1;
        const W_DOMINANCE: f64 = 0.3;

        W_PLEASURE * pred.pleasure_delta
            + W_VIOLATIONS * pred.constraint_violations as f64
            + W_ENERGY * pred.energy_cost as f64
            + W_DOMINANCE * pred.dominance_delta
    }

    /// Get selected action
    pub fn selected_action(&self) -> Option<&PredictedAction> {
        self.selected_action.as_ref()
    }

    /// Get all predictions
    pub fn predictions(&self) -> &[FuturePrediction] {
        &self.predictions
    }

    /// Update prediction error (for learning)
    pub fn update_error(&mut self, actual: f64, predicted: f64) {
        self.last_prediction_error = (actual - predicted).abs();
    }

    /// Get last prediction error
    pub fn last_error(&self) -> f64 {
        self.last_prediction_error
    }

    // =========================================================================
    // High-level prediction API
    // =========================================================================

    /// Run full prediction cycle
    ///
    /// 1. Snapshot current state
    /// 2. For each possible action:
    ///    a. Apply action
    ///    b. Simulate N ticks
    ///    c. Record result
    ///    d. Rollback
    /// 3. Select best action
    pub fn predict_futures(
        &mut self,
        physics: &mut PhysicsWorld,
        arousal: f64,
        possible_actions: &[PredictedAction],
    ) {
        self.clear();

        let num_futures = self.futures_count(arousal).min(possible_actions.len());
        let snapshot = physics.save_state();

        for action in possible_actions.iter().take(num_futures) {
            // Apply action (simplified - real impl would modify bodies)
            let _impulse = match action {
                PredictedAction::Idle => Vec3::ZERO,
                PredictedAction::Move(dir) => *dir * 0.1,
                PredictedAction::Impulse(imp) => *imp,
                PredictedAction::Custom(_) => Vec3::ZERO,
            };

            // Simulate forward
            physics.step_n(self.config.tick_delta, self.config.lookahead_ticks);

            // Record result (simplified - real impl would read body states)
            let prediction = FuturePrediction {
                action: action.clone(),
                final_position: Vec3::ZERO, // TODO: read from physics
                energy_cost: 0.0,           // TODO: calculate
                constraint_violations: 0,   // TODO: count
                pleasure_delta: 0.0,        // TODO: calculate from qualia
                arousal_delta: 0.0,
                dominance_delta: 0.0,
            };
            self.record_future(prediction);

            // Rollback
            physics.restore_state(&snapshot);
        }

        self.select_best_action();
    }
}

impl Default for PredictionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_futures_count_calm() {
        let engine = PredictionEngine::new();
        assert_eq!(engine.futures_count(0.0), 3);
        assert_eq!(engine.futures_count(0.2), 3);
    }

    #[test]
    fn test_futures_count_agitated() {
        let engine = PredictionEngine::new();
        assert_eq!(engine.futures_count(0.7), 10);
        assert_eq!(engine.futures_count(1.0), 10);
    }

    #[test]
    fn test_futures_count_moderate() {
        let engine = PredictionEngine::new();
        let count = engine.futures_count(0.5);
        assert!(count > 3 && count < 10);
    }

    #[test]
    fn test_should_predict() {
        let engine = PredictionEngine::new();
        assert!(!engine.should_predict(0.3));
        assert!(engine.should_predict(0.6));
    }

    #[test]
    fn test_select_best_action() {
        let mut engine = PredictionEngine::new();

        // Add predictions
        engine.record_future(FuturePrediction {
            action: PredictedAction::Idle,
            final_position: Vec3::ZERO,
            energy_cost: 0.0,
            constraint_violations: 0,
            pleasure_delta: 0.1,
            arousal_delta: 0.0,
            dominance_delta: 0.0,
        });

        engine.record_future(FuturePrediction {
            action: PredictedAction::Move(Vec3::X),
            final_position: Vec3::X,
            energy_cost: 0.5,
            constraint_violations: 0,
            pleasure_delta: 0.5, // Better pleasure
            arousal_delta: 0.1,
            dominance_delta: 0.2,
        });

        let best = engine.select_best_action();
        assert!(best.is_some());
        assert_eq!(*best.unwrap(), PredictedAction::Move(Vec3::X));
    }
}
