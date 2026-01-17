use crate::prelude::*;
use crate::components::cpu_sense::CpuSense;
use crate::components::bio_rhythm::BioRhythm;

pub fn calculate_stress_system(
    cpu_query: Query<&CpuSense>,
    mut bio_query: Query<&mut BioRhythm>,
) {
    if let Ok(cpu) = cpu_query.get_single() {
        // Simple heuristic for now
        let stress = cpu.usage_percent / 100.0;

        for mut bio in bio_query.iter_mut() {
            bio.stress_level = stress;
            bio.fatigue += stress * 0.001; // Accumulate fatigue
            if bio.fatigue > 1.0 { bio.fatigue = 1.0; }

            // Heartrate derived from stress
            // 60bpm rest + 120 * stress
            bio.heartrate_bpm = 60.0 + (120.0 * stress);
        }
    }
}
