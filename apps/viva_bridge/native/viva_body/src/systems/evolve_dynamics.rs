use crate::prelude::*;
use crate::components::emotional_state::EmotionalState;
use crate::components::bio_rhythm::BioRhythm;
use crate::resources::body_config::BodyConfig;
use crate::dynamics::{DynAffect, OUParams};
use rand_distr::{Distribution, Normal};

pub fn evolve_dynamics_system(
    config: Res<BodyConfig>,
    time: Res<Time>, // Use regular Time (delta seconds since last update)
    mut query: Query<(&mut EmotionalState, &BioRhythm)>,
) {
    let dt = time.delta_secs_f64();
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    for (mut emotion, bio) in query.iter_mut() {
        // 1. Construct DynAffect model from config
        let model = DynAffect {
            ou_params: config.ou_params,
            cusp_enabled: config.cusp_enabled,
            cusp_sensitivity: config.cusp_sensitivity,
        };

        // 2. Generate noise
        let noise = [
            normal.sample(&mut rng),
            normal.sample(&mut rng),
            normal.sample(&mut rng),
        ];

        // 3. Compute external bias from BioRhythm stress
        // High stress pushes pleasure down (negative bias)
        // High system entropy (chaos) pushes pleasure down
        let stress_bias = -1.0 * (bio.stress_level as f64);
        let entropy_bias = -0.5 * (bio.system_entropy as f64);
        let external_bias = stress_bias + entropy_bias;

        // 4. Current PAD array
        let mut pad = [emotion.pleasure, emotion.arousal, emotion.dominance];

        // 5. Evolve
        model.step(&mut pad, dt, &noise, external_bias);

        // 6. Update component
        emotion.pleasure = pad[0];
        emotion.arousal = pad[1];
        emotion.dominance = pad[2];

        // 7. Check Bifurcation status
        // Arousal acts as splitting factor 'c'
        let c = emotion.arousal.abs() * config.cusp_sensitivity;
        let y = external_bias;
        emotion.in_bifurcation = crate::dynamics::cusp_is_bifurcation(c, y);
    }
}
