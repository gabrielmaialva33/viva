use crate::components::bio_rhythm::BioRhythm;
use crate::components::cpu_sense::CpuSense;
use crate::components::gpu_sense::GpuSense;
use crate::prelude::*;
use crate::resources::soul_channel::{BodyUpdate, SoulChannel};

pub fn sync_soul_system(
    bio_query: Query<&BioRhythm, Changed<BioRhythm>>,
    cpu_query: Query<&CpuSense>,
    gpu_query: Query<&GpuSense>,
    channel: Option<Res<SoulChannel>>,
) {
    let Some(channel) = channel else { return };

    // Create defaults that live for the scope if needed
    let default_cpu = CpuSense::default();
    let default_gpu = GpuSense::default();

    if let Ok(bio) = bio_query.get_single() {
        // Use the defaults if query fails (e.g. entities haven't spawned yet or matched)
        let cpu = cpu_query.get_single().unwrap_or(&default_cpu);
        let gpu = gpu_query.get_single().unwrap_or(&default_gpu);

        let _ = channel.tx.try_send(BodyUpdate::StateChanged {
            stress: bio.stress_level,
            fatigue: bio.fatigue,
            needs_rest: bio.needs_rest,
            cpu_usage: cpu.usage_percent,
            gpu_usage: gpu.usage_percent,
            gpu_temp: gpu.temp_celsius,
        });

        if bio.stress_level > 0.8 {
            let _ = channel
                .tx
                .try_send(BodyUpdate::CriticalStress(bio.stress_level));
        }
    }
}
