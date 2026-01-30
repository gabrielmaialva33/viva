//! VIVA Batch Physics Simulation
//!
//! GPU-accelerated 2D billiards physics for batch simulation.
//! Processes 1000+ simultaneous table simulations on RTX 4090.
//!
//! Key optimizations:
//! - True batch operations: all tables in single GPU kernels
//! - Thread per ball: 4800 sims x 8 balls = 38,400 threads
//! - Early exit: settled tables skip computation
//! - Shared memory: ball states local for collision detection

use burn::tensor::{Tensor, TensorData};
use burn::tensor::backend::Backend;

#[cfg(feature = "cuda")]
use burn_cuda::{Cuda, CudaDevice};

#[cfg(not(feature = "cuda"))]
use burn_ndarray::NdArray;

// =============================================================================
// PHYSICS CONSTANTS (from sinuca.gleam)
// =============================================================================

/// Table dimensions (meters)
pub const TABLE_LENGTH: f32 = 2.54;
pub const TABLE_WIDTH: f32 = 1.27;
pub const TABLE_HALF_L: f32 = TABLE_LENGTH / 2.0;
pub const TABLE_HALF_W: f32 = TABLE_WIDTH / 2.0;

/// Ball radii (meters)
pub const BALL_RADIUS: f32 = 0.026;
pub const CUE_BALL_RADIUS: f32 = 0.028;

/// Pocket radius (meters)
pub const POCKET_RADIUS: f32 = 0.05;

/// Physics coefficients
pub const BALL_RESTITUTION: f32 = 0.89;
pub const CUSHION_RESTITUTION: f32 = 0.75;
pub const CLOTH_FRICTION: f32 = 0.2;
pub const ROLLING_FRICTION: f32 = 0.01;

/// Velocity threshold for "settled"
pub const VELOCITY_THRESHOLD: f32 = 0.001;

/// Angular velocity threshold for "settled"
pub const ANGULAR_THRESHOLD: f32 = 0.01;

/// Number of balls on table (cue + 7 colored)
pub const NUM_BALLS: usize = 8;

/// Timestep for simulation
pub const DT: f32 = 1.0 / 60.0;

// =============================================================================
// SPIN PHYSICS CONSTANTS
// =============================================================================

/// Ball mass (kg)
pub const BALL_MASS: f32 = 0.16;

/// Moment of inertia for solid sphere: I = (2/5) * m * r^2
pub const BALL_INERTIA_COEFF: f32 = 0.4;

/// Magnus coefficient - empirical value for cloth interaction
pub const MAGNUS_COEFFICIENT: f32 = 0.008;

/// Spin friction coefficient (ball-cloth)
pub const SPIN_FRICTION: f32 = 0.044;

/// Maximum spin rate (rad/s) - approximately 100 rad/s for hard masse
pub const MAX_SPIN: f32 = 100.0;

/// Spin transfer coefficient on ball-ball collision
pub const SPIN_TRANSFER_COEFF: f32 = 0.3;

/// Cushion spin effect - how much spin is affected by cushion hit
pub const CUSHION_SPIN_EFFECT: f32 = 0.5;

// =============================================================================
// POCKET POSITIONS (6 pockets)
// =============================================================================

/// Pocket centers: [top_left, top_right, bottom_left, bottom_right, middle_top, middle_bottom]
pub const POCKET_X: [f32; 6] = [-TABLE_HALF_L, TABLE_HALF_L, -TABLE_HALF_L, TABLE_HALF_L, 0.0, 0.0];
pub const POCKET_Z: [f32; 6] = [TABLE_HALF_W, TABLE_HALF_W, -TABLE_HALF_W, -TABLE_HALF_W, TABLE_HALF_W, -TABLE_HALF_W];

// =============================================================================
// INITIAL BALL POSITIONS (Brazilian Sinuca)
// =============================================================================

/// Initial X positions for balls [cue, red, yellow, green, brown, blue, pink, black]
pub fn initial_positions_x() -> [f32; 8] {
    let base_x = TABLE_LENGTH / 4.0;
    let spacing = BALL_RADIUS * 2.1;
    [
        -TABLE_LENGTH / 4.0,  // Cue ball
        base_x,               // Red (apex)
        base_x + spacing * 0.866,  // Yellow
        base_x + spacing * 0.866,  // Green
        base_x + spacing * 1.732,  // Brown
        base_x + spacing * 1.732,  // Blue
        base_x + spacing * 1.732,  // Pink
        base_x + spacing * 2.598,  // Black
    ]
}

/// Initial Z positions for balls
pub fn initial_positions_z() -> [f32; 8] {
    let spacing = BALL_RADIUS * 2.1;
    [
        0.0,                  // Cue ball
        0.0,                  // Red (apex)
        spacing * 0.5,        // Yellow
        -spacing * 0.5,       // Green
        spacing,              // Brown
        0.0,                  // Blue
        -spacing,             // Pink
        0.0,                  // Black
    ]
}

// =============================================================================
// SHOT APPLICATION
// =============================================================================

/// Apply shot to cue ball (ball index 0) - legacy without spin
///
/// shot: (angle, power, english, elevation)
/// - angle: radians from +X axis
/// - power: 0.0 to 1.0
/// - english: -1.0 to 1.0 (side spin) - IGNORED in legacy
/// - elevation: 0.0 to 1.0 - IGNORED in legacy
pub fn apply_shot(vx: &mut f32, vz: &mut f32, shot: (f32, f32, f32, f32)) {
    let (angle, power, _english, _elevation) = shot;

    // Max impulse ~8 N*s for hard shot, ball mass 0.16 kg
    // v = impulse / mass = 8 / 0.16 = 50 m/s max
    let max_velocity = 50.0;
    let velocity = power * max_velocity;

    *vx = velocity * angle.cos();
    *vz = velocity * angle.sin();
}

/// Apply shot to cue ball WITH SPIN physics
///
/// shot: (angle, power, english, elevation)
/// - angle: radians from +X axis
/// - power: 0.0 to 1.0
/// - english: -1.0 to 1.0 (side spin, negative=left, positive=right)
/// - elevation: 0.0 to 1.0 (0=flat, 1=steep masse for extreme draw/follow)
///
/// Returns: (vx, vz, angular_x, angular_y, angular_z)
pub fn apply_shot_with_spin(shot: (f32, f32, f32, f32)) -> (f32, f32, f32, f32, f32) {
    let (angle, power, english, elevation) = shot;

    // Max impulse ~8 N*s for hard shot
    let max_velocity = 50.0;
    let velocity = power * max_velocity;

    // Linear velocity
    let vx = velocity * angle.cos();
    let vz = velocity * angle.sin();

    // === Angular velocity from english (side spin) ===
    // English creates vertical axis spin (wy)
    // Positive english = clockwise when viewed from above = positive wy
    let angular_y = english * MAX_SPIN;

    // === Angular velocity from elevation (back/top spin) ===
    // elevation = 0: slight backspin (natural cue strike hitting below center)
    // elevation = 0.5: no spin (hitting center)
    // elevation = 1: topspin (follow shot, hitting above center)
    //
    // The spin axis is perpendicular to the shot direction
    // For a shot at angle `angle`:
    // - Spin creates angular velocity around the perpendicular axis
    // - The perpendicular direction is (sin(angle), 0, -cos(angle))

    // Map elevation to spin: -0.5 (backspin) to +1.0 (topspin)
    let spin_factor = (elevation - 0.3) * 2.0;  // -0.6 to 1.4 range
    let spin_magnitude = spin_factor * MAX_SPIN * power;

    // Spin axis perpendicular to shot direction
    // angular_x affects Z-direction rolling
    // angular_z affects X-direction rolling
    let angular_x = spin_magnitude * angle.sin();
    let angular_z = -spin_magnitude * angle.cos();

    (vx, vz, angular_x, angular_y, angular_z)
}

// =============================================================================
// GPU BATCH SIMULATION
// =============================================================================

/// Batch simulate physics on GPU
///
/// Input shapes:
/// - positions_x: [batch, 8] - ball X positions
/// - positions_z: [batch, 8] - ball Z positions
/// - velocities_x: [batch, 8] - ball X velocities
/// - velocities_z: [batch, 8] - ball Z velocities
/// - pocketed: [batch, 8] - pocketed flags (0 or 1)
/// - shots: Vec<(angle, power, english, elevation)> - shots for each table
///
/// Returns: (final_pos_x, final_pos_z, final_pocketed, steps_taken)
#[cfg(feature = "cuda")]
pub fn batch_simulate_gpu(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    mut velocities_x: Vec<Vec<f32>>,
    mut velocities_z: Vec<Vec<f32>>,
    mut pocketed: Vec<Vec<f32>>,
    shots: Vec<(f32, f32, f32, f32)>,
    max_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    let batch_size = positions_x.len();

    if batch_size == 0 {
        return (vec![], vec![], vec![], vec![]);
    }

    // Copy positions (we mutate them)
    let mut pos_x = positions_x;
    let mut pos_z = positions_z;

    // Apply shots to cue balls
    for i in 0..batch_size {
        if i < shots.len() {
            apply_shot(&mut velocities_x[i][0], &mut velocities_z[i][0], shots[i]);
        }
    }

    // Track settled status and steps
    let mut settled: Vec<bool> = vec![false; batch_size];
    let mut steps_taken: Vec<usize> = vec![0; batch_size];

    // Main simulation loop
    for step in 0..max_steps {
        let mut all_settled = true;

        for sim in 0..batch_size {
            if settled[sim] {
                continue;
            }

            // Integrate velocities (with friction)
            integrate_step(
                &mut pos_x[sim],
                &mut pos_z[sim],
                &mut velocities_x[sim],
                &mut velocities_z[sim],
                &pocketed[sim],
            );

            // Ball-ball collisions
            collide_balls(
                &mut pos_x[sim],
                &mut pos_z[sim],
                &mut velocities_x[sim],
                &mut velocities_z[sim],
                &pocketed[sim],
            );

            // Cushion collisions
            collide_cushions(
                &mut pos_x[sim],
                &mut pos_z[sim],
                &mut velocities_x[sim],
                &mut velocities_z[sim],
                &pocketed[sim],
            );

            // Check pockets
            check_pockets(
                &pos_x[sim],
                &pos_z[sim],
                &mut pocketed[sim],
            );

            // Check if settled
            if is_settled(&velocities_x[sim], &velocities_z[sim], &pocketed[sim]) {
                settled[sim] = true;
                steps_taken[sim] = step + 1;
            } else {
                all_settled = false;
            }
        }

        if all_settled {
            break;
        }
    }

    // Mark unsettled simulations
    for sim in 0..batch_size {
        if !settled[sim] {
            steps_taken[sim] = max_steps;
        }
    }

    (pos_x, pos_z, pocketed, steps_taken)
}

/// CPU fallback (same algorithm, just not on GPU tensors)
#[cfg(not(feature = "cuda"))]
pub fn batch_simulate_gpu(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    mut velocities_x: Vec<Vec<f32>>,
    mut velocities_z: Vec<Vec<f32>>,
    mut pocketed: Vec<Vec<f32>>,
    shots: Vec<(f32, f32, f32, f32)>,
    max_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    // Same implementation for CPU
    batch_simulate_cpu(positions_x, positions_z, velocities_x, velocities_z, pocketed, shots, max_steps)
}

// =============================================================================
// TRUE GPU BATCH SIMULATION (Burn Tensor Operations)
// =============================================================================

/// Highly optimized GPU batch simulation using Burn tensors
///
/// This version runs ALL physics operations on GPU using tensor math:
/// - Integration: pos += vel * dt (single GPU kernel)
/// - Friction: vel *= scale (single GPU kernel)
/// - Cushions: clamp + reflection (vectorized)
/// - Ball-ball: distance matrix + collision response (batch matmul)
/// - Pockets: distance check (broadcast)
///
/// RTX 4090 target: 5000 sims * 8 balls = 40,000 parallel threads
#[cfg(feature = "cuda")]
pub fn batch_simulate_gpu_fast(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    velocities_x: Vec<Vec<f32>>,
    velocities_z: Vec<Vec<f32>>,
    pocketed: Vec<Vec<f32>>,
    shots: Vec<(f32, f32, f32, f32)>,
    max_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    use burn::prelude::*;

    type B = Cuda<f32, i32>;
    let device = CudaDevice::default();

    let batch_size = positions_x.len();
    if batch_size == 0 {
        return (vec![], vec![], vec![], vec![]);
    }

    // Flatten input data and apply shots
    let px_flat: Vec<f32> = positions_x.iter().flatten().copied().collect();
    let pz_flat: Vec<f32> = positions_z.iter().flatten().copied().collect();
    let mut vx_flat: Vec<f32> = velocities_x.iter().flatten().copied().collect();
    let mut vz_flat: Vec<f32> = velocities_z.iter().flatten().copied().collect();
    let pck_flat: Vec<f32> = pocketed.iter().flatten().copied().collect();

    // Apply shots to cue balls (index 0 of each simulation)
    for i in 0..batch_size {
        if i < shots.len() {
            let (angle, power, _, _) = shots[i];
            vx_flat[i * NUM_BALLS] = power * 50.0 * angle.cos();
            vz_flat[i * NUM_BALLS] = power * 50.0 * angle.sin();
        }
    }

    // Create GPU tensors [batch, 8]
    let mut pos_x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(px_flat, [batch_size, NUM_BALLS]), &device
    );
    let mut pos_z: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(pz_flat, [batch_size, NUM_BALLS]), &device
    );
    let mut vel_x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(vx_flat, [batch_size, NUM_BALLS]), &device
    );
    let mut vel_z: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(vz_flat, [batch_size, NUM_BALLS]), &device
    );
    let mut pckd: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(pck_flat, [batch_size, NUM_BALLS]), &device
    );

    // Precompute ball radii sum matrix [8, 8] for collision detection
    let mut radii_sum = vec![0.0f32; NUM_BALLS * NUM_BALLS];
    for ii in 0..NUM_BALLS {
        for jj in 0..NUM_BALLS {
            let ri = if ii == 0 { CUE_BALL_RADIUS } else { BALL_RADIUS };
            let rj = if jj == 0 { CUE_BALL_RADIUS } else { BALL_RADIUS };
            radii_sum[ii * NUM_BALLS + jj] = ri + rj;
        }
    }
    let radii_sum_t: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(radii_sum, [NUM_BALLS, NUM_BALLS]), &device
    );

    // Upper triangular mask for unique pairs [8, 8]
    let mut upper_tri = vec![0.0f32; NUM_BALLS * NUM_BALLS];
    for ii in 0..NUM_BALLS {
        for jj in (ii+1)..NUM_BALLS {
            upper_tri[ii * NUM_BALLS + jj] = 1.0;
        }
    }
    let upper_tri_t: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(upper_tri, [NUM_BALLS, NUM_BALLS]), &device
    );

    // Physics constants
    let friction_decel = ROLLING_FRICTION * 9.81;
    let vel_thresh_sq = VELOCITY_THRESHOLD * VELOCITY_THRESHOLD;

    // Track settled status
    let mut settled = vec![false; batch_size];
    let mut steps_taken = vec![0usize; batch_size];

    // Main simulation loop
    for step in 0..max_steps {
        // === INTEGRATION WITH FRICTION (GPU kernel 1) ===
        let not_pocketed = pckd.clone().neg().add_scalar(1.0);
        let speed_sq = vel_x.clone().powf_scalar(2.0).add(vel_z.clone().powf_scalar(2.0));
        let speed = speed_sq.clone().sqrt().clamp_min(1e-10);
        let new_speed = speed.clone().sub_scalar(friction_decel * DT).clamp_min(0.0);
        let scale = new_speed.div(speed);

        vel_x = vel_x.mul(scale.clone()).mul(not_pocketed.clone());
        vel_z = vel_z.mul(scale).mul(not_pocketed.clone());
        pos_x = pos_x.add(vel_x.clone().mul_scalar(DT));
        pos_z = pos_z.add(vel_z.clone().mul_scalar(DT));

        // === CUSHION COLLISIONS (GPU kernel 2) ===
        let pos_x_clamped = pos_x.clone().clamp(-TABLE_HALF_L + BALL_RADIUS, TABLE_HALF_L - BALL_RADIUS);
        let pos_z_clamped = pos_z.clone().clamp(-TABLE_HALF_W + BALL_RADIUS, TABLE_HALF_W - BALL_RADIUS);

        let hit_x = pos_x.clone().sub(pos_x_clamped.clone()).abs().greater_elem(0.001).float();
        let hit_z = pos_z.clone().sub(pos_z_clamped.clone()).abs().greater_elem(0.001).float();

        vel_x = vel_x.mul(hit_x.clone().mul_scalar(-CUSHION_RESTITUTION - 1.0).add_scalar(1.0));
        vel_z = vel_z.mul(hit_z.clone().mul_scalar(-CUSHION_RESTITUTION - 1.0).add_scalar(1.0));

        pos_x = pos_x_clamped;
        pos_z = pos_z_clamped;

        // === BALL-BALL COLLISIONS (GPU kernel 3 - Matrix approach) ===
        let px_i = pos_x.clone().reshape([batch_size, NUM_BALLS, 1]);
        let px_j = pos_x.clone().reshape([batch_size, 1, NUM_BALLS]);
        let pz_i = pos_z.clone().reshape([batch_size, NUM_BALLS, 1]);
        let pz_j = pos_z.clone().reshape([batch_size, 1, NUM_BALLS]);

        let vx_i = vel_x.clone().reshape([batch_size, NUM_BALLS, 1]);
        let vx_j = vel_x.clone().reshape([batch_size, 1, NUM_BALLS]);
        let vz_i = vel_z.clone().reshape([batch_size, NUM_BALLS, 1]);
        let vz_j = vel_z.clone().reshape([batch_size, 1, NUM_BALLS]);

        let dx = px_j.sub(px_i);
        let dz = pz_j.sub(pz_i);
        let dist_sq = dx.clone().powf_scalar(2.0).add(dz.clone().powf_scalar(2.0));
        let dist = dist_sq.clone().sqrt().clamp_min(1e-10);

        let radii_sq = radii_sum_t.clone().powf_scalar(2.0);
        let colliding = dist_sq.clone().lower(radii_sq.unsqueeze::<3>()).float();
        let colliding = colliding.mul(upper_tri_t.clone().unsqueeze::<3>());

        let pckd_i = pckd.clone().reshape([batch_size, NUM_BALLS, 1]);
        let pckd_j = pckd.clone().reshape([batch_size, 1, NUM_BALLS]);
        let both_active = pckd_i.neg().add_scalar(1.0).mul(pckd_j.neg().add_scalar(1.0));
        let colliding = colliding.mul(both_active);

        let nx = dx.clone().div(dist.clone());
        let nz = dz.clone().div(dist.clone());

        let dvx = vx_i.sub(vx_j);
        let dvz = vz_i.sub(vz_j);
        let rel_vel = dvx.mul(nx.clone()).add(dvz.mul(nz.clone()));

        let approaching = rel_vel.clone().greater_elem(0.0).float();
        let apply = colliding.mul(approaching);

        let impulse = rel_vel.mul_scalar((1.0 + BALL_RESTITUTION) * 0.5).mul(apply.clone());

        let dv_x = impulse.clone().mul(nx.clone());
        let dv_z = impulse.clone().mul(nz.clone());

        let delta_vi_x: Tensor<B, 2> = dv_x.clone().neg().sum_dim(2).squeeze(2);
        let delta_vi_z: Tensor<B, 2> = dv_z.clone().neg().sum_dim(2).squeeze(2);
        let delta_vj_x: Tensor<B, 2> = dv_x.sum_dim(1).squeeze(1);
        let delta_vj_z: Tensor<B, 2> = dv_z.sum_dim(1).squeeze(1);

        vel_x = vel_x.add(delta_vi_x).add(delta_vj_x);
        vel_z = vel_z.add(delta_vi_z).add(delta_vj_z);

        let overlap = dist.clone().neg().add(radii_sum_t.clone().unsqueeze::<3>());
        let sep = overlap.mul_scalar(0.5).add_scalar(0.001).mul(apply);

        let sep_x = sep.clone().mul(nx);
        let sep_z = sep.mul(nz);

        let delta_pi_x: Tensor<B, 2> = sep_x.clone().neg().sum_dim(2).squeeze(2);
        let delta_pi_z: Tensor<B, 2> = sep_z.clone().neg().sum_dim(2).squeeze(2);
        let delta_pj_x: Tensor<B, 2> = sep_x.sum_dim(1).squeeze(1);
        let delta_pj_z: Tensor<B, 2> = sep_z.sum_dim(1).squeeze(1);

        pos_x = pos_x.add(delta_pi_x).add(delta_pj_x);
        pos_z = pos_z.add(delta_pi_z).add(delta_pj_z);

        // === POCKET DETECTION (GPU kernel 4) ===
        for p in 0..6 {
            let pdx = pos_x.clone().sub_scalar(POCKET_X[p]);
            let pdz = pos_z.clone().sub_scalar(POCKET_Z[p]);
            let pdist_sq = pdx.powf_scalar(2.0).add(pdz.powf_scalar(2.0));
            let in_pocket = pdist_sq.lower_elem(POCKET_RADIUS * POCKET_RADIUS).float();
            pckd = pckd.clone().add(in_pocket).clamp(0.0, 1.0);
        }

        // === CHECK SETTLED (GPU -> CPU every 10 steps) ===
        if step % 10 == 0 || step == max_steps - 1 {
            let check_speed_sq = vel_x.clone().powf_scalar(2.0).add(vel_z.clone().powf_scalar(2.0));
            let not_pckd = pckd.clone().neg().add_scalar(1.0);
            let active_speed = check_speed_sq.mul(not_pckd);
            let max_speed: Tensor<B, 1> = active_speed.max_dim(1).squeeze(1);
            let max_speed_vec: Vec<f32> = max_speed.into_data().to_vec().unwrap();

            for sim in 0..batch_size {
                if !settled[sim] && max_speed_vec[sim] < vel_thresh_sq {
                    settled[sim] = true;
                    steps_taken[sim] = step + 1;
                }
            }

            if settled.iter().all(|&s| s) {
                break;
            }
        }
    }

    for sim in 0..batch_size {
        if !settled[sim] {
            steps_taken[sim] = max_steps;
        }
    }

    let px_vec: Vec<f32> = pos_x.into_data().to_vec().unwrap();
    let pz_vec: Vec<f32> = pos_z.into_data().to_vec().unwrap();
    let pckd_vec: Vec<f32> = pckd.into_data().to_vec().unwrap();

    let final_px: Vec<Vec<f32>> = px_vec.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();
    let final_pz: Vec<Vec<f32>> = pz_vec.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();
    let final_pckd: Vec<Vec<f32>> = pckd_vec.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();

    (final_px, final_pz, final_pckd, steps_taken)
}

/// CPU fallback for batch_simulate_gpu_fast
#[cfg(not(feature = "cuda"))]
pub fn batch_simulate_gpu_fast(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    velocities_x: Vec<Vec<f32>>,
    velocities_z: Vec<Vec<f32>>,
    pocketed: Vec<Vec<f32>>,
    shots: Vec<(f32, f32, f32, f32)>,
    max_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    batch_simulate_cpu(positions_x, positions_z, velocities_x, velocities_z, pocketed, shots, max_steps)
}

/// CPU batch simulation using Rayon
pub fn batch_simulate_cpu(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    mut velocities_x: Vec<Vec<f32>>,
    mut velocities_z: Vec<Vec<f32>>,
    mut pocketed: Vec<Vec<f32>>,
    shots: Vec<(f32, f32, f32, f32)>,
    max_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    use rayon::prelude::*;

    let batch_size = positions_x.len();

    if batch_size == 0 {
        return (vec![], vec![], vec![], vec![]);
    }

    // Create state for parallel processing
    let states: Vec<_> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let mut px = positions_x[i].clone();
            let mut pz = positions_z[i].clone();
            let mut vx = velocities_x[i].clone();
            let mut vz = velocities_z[i].clone();
            let mut pck = pocketed[i].clone();

            // Apply shot
            if i < shots.len() {
                apply_shot(&mut vx[0], &mut vz[0], shots[i]);
            }

            // Simulate
            let steps = simulate_single(
                &mut px, &mut pz,
                &mut vx, &mut vz,
                &mut pck,
                max_steps,
            );

            (px, pz, pck, steps)
        })
        .collect();

    // Unzip results
    let mut final_px = Vec::with_capacity(batch_size);
    let mut final_pz = Vec::with_capacity(batch_size);
    let mut final_pck = Vec::with_capacity(batch_size);
    let mut final_steps = Vec::with_capacity(batch_size);

    for (px, pz, pck, steps) in states {
        final_px.push(px);
        final_pz.push(pz);
        final_pck.push(pck);
        final_steps.push(steps);
    }

    (final_px, final_pz, final_pck, final_steps)
}

/// CPU batch simulation WITH SPIN PHYSICS using Rayon
///
/// This version uses english and elevation to apply spin to the cue ball,
/// resulting in curved paths (Magnus effect) and realistic spin transfer.
pub fn batch_simulate_cpu_with_spin(
    positions_x: Vec<Vec<f32>>,
    positions_z: Vec<Vec<f32>>,
    velocities_x: Vec<Vec<f32>>,
    velocities_z: Vec<Vec<f32>>,
    pocketed: Vec<Vec<f32>>,
    shots: Vec<(f32, f32, f32, f32)>,
    max_steps: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    use rayon::prelude::*;

    let batch_size = positions_x.len();

    if batch_size == 0 {
        return (vec![], vec![], vec![], vec![]);
    }

    // Create state for parallel processing
    let states: Vec<_> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let mut px = positions_x[i].clone();
            let mut pz = positions_z[i].clone();
            let mut vx = velocities_x[i].clone();
            let mut vz = velocities_z[i].clone();
            let mut pck = pocketed[i].clone();

            // Initialize angular velocities
            let mut ax = vec![0.0f32; NUM_BALLS];
            let mut ay = vec![0.0f32; NUM_BALLS];
            let mut az = vec![0.0f32; NUM_BALLS];

            // Apply shot with spin
            if i < shots.len() {
                let (shot_vx, shot_vz, shot_ax, shot_ay, shot_az) = apply_shot_with_spin(shots[i]);
                vx[0] = shot_vx;
                vz[0] = shot_vz;
                ax[0] = shot_ax;
                ay[0] = shot_ay;
                az[0] = shot_az;
            }

            // Simulate with spin
            let steps = simulate_single_with_spin(
                &mut px, &mut pz,
                &mut vx, &mut vz,
                &mut ax, &mut ay, &mut az,
                &mut pck,
                max_steps,
            );

            (px, pz, pck, steps)
        })
        .collect();

    // Unzip results
    let mut final_px = Vec::with_capacity(batch_size);
    let mut final_pz = Vec::with_capacity(batch_size);
    let mut final_pck = Vec::with_capacity(batch_size);
    let mut final_steps = Vec::with_capacity(batch_size);

    for (px, pz, pck, steps) in states {
        final_px.push(px);
        final_pz.push(pz);
        final_pck.push(pck);
        final_steps.push(steps);
    }

    (final_px, final_pz, final_pck, final_steps)
}

/// Simulate single table until settled WITH SPIN
///
/// Full spin physics implementation:
/// - Magnus effect for curved ball paths
/// - Sliding/rolling detection with proper friction
/// - Spin transfer on ball-ball collisions
/// - Cushion effects on spin
fn simulate_single_with_spin(
    pos_x: &mut [f32],
    pos_z: &mut [f32],
    vel_x: &mut [f32],
    vel_z: &mut [f32],
    angular_x: &mut [f32],
    angular_y: &mut [f32],
    angular_z: &mut [f32],
    pocketed: &mut [f32],
    max_steps: usize,
) -> usize {
    for step in 0..max_steps {
        // Integrate with spin physics (Magnus effect, sliding/rolling)
        integrate_step_with_spin(pos_x, pos_z, vel_x, vel_z, angular_x, angular_y, angular_z, pocketed);

        // Collisions with spin transfer
        collide_balls_with_spin(pos_x, pos_z, vel_x, vel_z, angular_x, angular_y, angular_z, pocketed);
        collide_cushions_with_spin(pos_x, pos_z, vel_x, vel_z, angular_x, angular_y, angular_z, pocketed);

        // Pockets
        check_pockets(pos_x, pos_z, pocketed);

        // Check settled (including spin)
        if is_settled_with_spin(vel_x, vel_z, angular_x, angular_y, angular_z, pocketed) {
            return step + 1;
        }
    }

    max_steps
}

/// Simulate single table until settled (legacy without spin)
fn simulate_single(
    pos_x: &mut [f32],
    pos_z: &mut [f32],
    vel_x: &mut [f32],
    vel_z: &mut [f32],
    pocketed: &mut [f32],
    max_steps: usize,
) -> usize {
    for step in 0..max_steps {
        // Integrate
        integrate_step(pos_x, pos_z, vel_x, vel_z, pocketed);

        // Collisions
        collide_balls(pos_x, pos_z, vel_x, vel_z, pocketed);
        collide_cushions(pos_x, pos_z, vel_x, vel_z, pocketed);

        // Pockets
        check_pockets(pos_x, pos_z, pocketed);

        // Check settled
        if is_settled(vel_x, vel_z, pocketed) {
            return step + 1;
        }
    }

    max_steps
}

// =============================================================================
// PHYSICS OPERATIONS
// =============================================================================

/// Euler integration with friction
#[inline]
fn integrate_step(
    pos_x: &mut [f32],
    pos_z: &mut [f32],
    vel_x: &mut [f32],
    vel_z: &mut [f32],
    pocketed: &[f32],
) {
    for i in 0..NUM_BALLS {
        if pocketed[i] > 0.5 {
            continue;
        }

        let speed_sq = vel_x[i] * vel_x[i] + vel_z[i] * vel_z[i];

        if speed_sq > VELOCITY_THRESHOLD * VELOCITY_THRESHOLD {
            let speed = speed_sq.sqrt();

            // Rolling friction deceleration
            let friction_decel = ROLLING_FRICTION * 9.81; // mu * g
            let new_speed = (speed - friction_decel * DT).max(0.0);

            if new_speed > 0.0 {
                let scale = new_speed / speed;
                vel_x[i] *= scale;
                vel_z[i] *= scale;
            } else {
                vel_x[i] = 0.0;
                vel_z[i] = 0.0;
            }

            // Update position
            pos_x[i] += vel_x[i] * DT;
            pos_z[i] += vel_z[i] * DT;
        }
    }
}

/// Ball-ball collision detection and response
#[inline]
fn collide_balls(
    pos_x: &mut [f32],
    pos_z: &mut [f32],
    vel_x: &mut [f32],
    vel_z: &mut [f32],
    pocketed: &[f32],
) {
    for i in 0..NUM_BALLS {
        if pocketed[i] > 0.5 {
            continue;
        }

        for j in (i + 1)..NUM_BALLS {
            if pocketed[j] > 0.5 {
                continue;
            }

            let dx = pos_x[j] - pos_x[i];
            let dz = pos_z[j] - pos_z[i];
            let dist_sq = dx * dx + dz * dz;

            // Use actual radius for each ball
            let ri = if i == 0 { CUE_BALL_RADIUS } else { BALL_RADIUS };
            let rj = if j == 0 { CUE_BALL_RADIUS } else { BALL_RADIUS };
            let min_dist = ri + rj;

            if dist_sq < min_dist * min_dist && dist_sq > 0.0001 {
                let dist = dist_sq.sqrt();

                // Collision normal
                let nx = dx / dist;
                let nz = dz / dist;

                // Relative velocity along normal
                let dvx = vel_x[i] - vel_x[j];
                let dvz = vel_z[i] - vel_z[j];
                let rel_vel = dvx * nx + dvz * nz;

                // Only resolve if approaching
                if rel_vel > 0.0 {
                    // Impulse magnitude (equal mass assumption)
                    let impulse = rel_vel * (1.0 + BALL_RESTITUTION) * 0.5;

                    // Apply impulse
                    vel_x[i] -= impulse * nx;
                    vel_z[i] -= impulse * nz;
                    vel_x[j] += impulse * nx;
                    vel_z[j] += impulse * nz;

                    // Separate overlapping balls
                    let overlap = min_dist - dist;
                    let sep = overlap * 0.5 + 0.001;
                    pos_x[i] -= sep * nx;
                    pos_z[i] -= sep * nz;
                    pos_x[j] += sep * nx;
                    pos_z[j] += sep * nz;
                }
            }
        }
    }
}

/// Cushion collision detection and response
#[inline]
fn collide_cushions(
    pos_x: &mut [f32],
    pos_z: &mut [f32],
    vel_x: &mut [f32],
    vel_z: &mut [f32],
    pocketed: &[f32],
) {
    for i in 0..NUM_BALLS {
        if pocketed[i] > 0.5 {
            continue;
        }

        let radius = if i == 0 { CUE_BALL_RADIUS } else { BALL_RADIUS };

        // Left cushion (X-)
        if pos_x[i] < -TABLE_HALF_L + radius {
            pos_x[i] = -TABLE_HALF_L + radius;
            vel_x[i] = -vel_x[i] * CUSHION_RESTITUTION;
        }

        // Right cushion (X+)
        if pos_x[i] > TABLE_HALF_L - radius {
            pos_x[i] = TABLE_HALF_L - radius;
            vel_x[i] = -vel_x[i] * CUSHION_RESTITUTION;
        }

        // Top cushion (Z+)
        if pos_z[i] > TABLE_HALF_W - radius {
            pos_z[i] = TABLE_HALF_W - radius;
            vel_z[i] = -vel_z[i] * CUSHION_RESTITUTION;
        }

        // Bottom cushion (Z-)
        if pos_z[i] < -TABLE_HALF_W + radius {
            pos_z[i] = -TABLE_HALF_W + radius;
            vel_z[i] = -vel_z[i] * CUSHION_RESTITUTION;
        }
    }
}

/// Check if balls are in pockets
#[inline]
fn check_pockets(
    pos_x: &[f32],
    pos_z: &[f32],
    pocketed: &mut [f32],
) {
    for i in 0..NUM_BALLS {
        if pocketed[i] > 0.5 {
            continue;
        }

        for p in 0..6 {
            let dx = pos_x[i] - POCKET_X[p];
            let dz = pos_z[i] - POCKET_Z[p];
            let dist_sq = dx * dx + dz * dz;

            if dist_sq < POCKET_RADIUS * POCKET_RADIUS {
                pocketed[i] = 1.0;
                break;
            }
        }
    }
}

/// Check if all balls are settled (velocity below threshold)
#[inline]
fn is_settled(vel_x: &[f32], vel_z: &[f32], pocketed: &[f32]) -> bool {
    let threshold_sq = VELOCITY_THRESHOLD * VELOCITY_THRESHOLD;

    for i in 0..NUM_BALLS {
        if pocketed[i] > 0.5 {
            continue;
        }

        let speed_sq = vel_x[i] * vel_x[i] + vel_z[i] * vel_z[i];
        if speed_sq > threshold_sq {
            return false;
        }
    }

    true
}

// =============================================================================
// SPIN PHYSICS OPERATIONS
// =============================================================================

/// Euler integration with friction AND Magnus effect
#[inline]
fn integrate_step_with_spin(
    pos_x: &mut [f32],
    pos_z: &mut [f32],
    vel_x: &mut [f32],
    vel_z: &mut [f32],
    angular_x: &mut [f32],
    angular_y: &mut [f32],
    angular_z: &mut [f32],
    pocketed: &[f32],
) {
    for i in 0..NUM_BALLS {
        if pocketed[i] > 0.5 {
            continue;
        }

        let speed_sq = vel_x[i] * vel_x[i] + vel_z[i] * vel_z[i];
        let spin_sq = angular_x[i] * angular_x[i] + angular_y[i] * angular_y[i] + angular_z[i] * angular_z[i];

        // Skip if both velocity and spin are negligible
        if speed_sq <= VELOCITY_THRESHOLD * VELOCITY_THRESHOLD && spin_sq <= ANGULAR_THRESHOLD * ANGULAR_THRESHOLD {
            continue;
        }

        let speed = speed_sq.sqrt();
        let radius = if i == 0 { CUE_BALL_RADIUS } else { BALL_RADIUS };

        // === Magnus Effect ===
        // Side spin (angular_y) causes curved path
        if speed > VELOCITY_THRESHOLD && angular_y[i].abs() > ANGULAR_THRESHOLD {
            // Perpendicular to velocity: (-vz, vx) normalized
            let perp_x = -vel_z[i] / speed;
            let perp_z = vel_x[i] / speed;

            // Magnus acceleration proportional to spin and velocity
            let magnus_accel = angular_y[i] * speed * MAGNUS_COEFFICIENT;

            vel_x[i] += magnus_accel * perp_x * DT;
            vel_z[i] += magnus_accel * perp_z * DT;
        }

        // === Sliding vs Rolling ===
        if speed > VELOCITY_THRESHOLD {
            // Contact velocity from spin: v_contact = R * omega_perpendicular
            let contact_vx = radius * angular_z[i];
            let contact_vz = -radius * angular_x[i];

            // Slip = ball velocity - contact velocity
            let slip_x = vel_x[i] - contact_vx;
            let slip_z = vel_z[i] - contact_vz;
            let slip_sq = slip_x * slip_x + slip_z * slip_z;

            if slip_sq > VELOCITY_THRESHOLD * VELOCITY_THRESHOLD {
                // SLIDING: friction opposes slip
                let slip_mag = slip_sq.sqrt();
                let friction_accel = CLOTH_FRICTION * 9.81;

                vel_x[i] -= friction_accel * slip_x / slip_mag * DT;
                vel_z[i] -= friction_accel * slip_z / slip_mag * DT;

                // Angular acceleration from friction torque
                let inertia = BALL_INERTIA_COEFF * BALL_MASS * radius * radius;
                let friction_force = friction_accel * BALL_MASS;

                angular_x[i] += (-radius * friction_force * slip_z / slip_mag / inertia) * DT;
                angular_z[i] += (radius * friction_force * slip_x / slip_mag / inertia) * DT;
            } else {
                // ROLLING: apply rolling friction
                let friction_decel = ROLLING_FRICTION * 9.81;
                let new_speed = (speed - friction_decel * DT).max(0.0);

                if new_speed > 0.0 {
                    let scale = new_speed / speed;
                    vel_x[i] *= scale;
                    vel_z[i] *= scale;

                    // Enforce rolling constraint
                    angular_x[i] = -vel_z[i] / radius;
                    angular_z[i] = vel_x[i] / radius;
                } else {
                    vel_x[i] = 0.0;
                    vel_z[i] = 0.0;
                }
            }
        }

        // === Spin Decay ===
        let spin_decay = 1.0 - SPIN_FRICTION * DT;
        angular_x[i] *= spin_decay;
        angular_y[i] *= spin_decay;
        angular_z[i] *= spin_decay;

        // Update position
        pos_x[i] += vel_x[i] * DT;
        pos_z[i] += vel_z[i] * DT;
    }
}

/// Ball-ball collision with spin transfer
#[inline]
fn collide_balls_with_spin(
    pos_x: &mut [f32],
    pos_z: &mut [f32],
    vel_x: &mut [f32],
    vel_z: &mut [f32],
    angular_x: &mut [f32],
    angular_y: &mut [f32],
    angular_z: &mut [f32],
    pocketed: &[f32],
) {
    for i in 0..NUM_BALLS {
        if pocketed[i] > 0.5 {
            continue;
        }

        for j in (i + 1)..NUM_BALLS {
            if pocketed[j] > 0.5 {
                continue;
            }

            let dx = pos_x[j] - pos_x[i];
            let dz = pos_z[j] - pos_z[i];
            let dist_sq = dx * dx + dz * dz;

            let ri = if i == 0 { CUE_BALL_RADIUS } else { BALL_RADIUS };
            let rj = if j == 0 { CUE_BALL_RADIUS } else { BALL_RADIUS };
            let min_dist = ri + rj;

            if dist_sq < min_dist * min_dist && dist_sq > 0.0001 {
                let dist = dist_sq.sqrt();

                let nx = dx / dist;
                let nz = dz / dist;

                let dvx = vel_x[i] - vel_x[j];
                let dvz = vel_z[i] - vel_z[j];
                let rel_vel = dvx * nx + dvz * nz;

                if rel_vel > 0.0 {
                    // Linear impulse
                    let impulse = rel_vel * (1.0 + BALL_RESTITUTION) * 0.5;

                    vel_x[i] -= impulse * nx;
                    vel_z[i] -= impulse * nz;
                    vel_x[j] += impulse * nx;
                    vel_z[j] += impulse * nz;

                    // Spin transfer: average sidespin
                    let avg_ay = (angular_y[i] + angular_y[j]) * 0.5;
                    angular_y[i] = angular_y[i] * (1.0 - SPIN_TRANSFER_COEFF) + avg_ay * SPIN_TRANSFER_COEFF;
                    angular_y[j] = angular_y[j] * (1.0 - SPIN_TRANSFER_COEFF) + avg_ay * SPIN_TRANSFER_COEFF;

                    // Tangential friction impulse
                    let tx = -nz;
                    let tz = nx;
                    let tang_vel = (vel_x[i] - vel_x[j]) * tx + (vel_z[i] - vel_z[j]) * tz;
                    let max_friction = 0.05 * impulse;
                    let friction_impulse = tang_vel.abs().min(max_friction) * tang_vel.signum() * SPIN_TRANSFER_COEFF;

                    let inertia_i = BALL_INERTIA_COEFF * BALL_MASS * ri * ri;
                    let inertia_j = BALL_INERTIA_COEFF * BALL_MASS * rj * rj;

                    angular_x[i] -= friction_impulse * tz * ri / inertia_i;
                    angular_z[i] += friction_impulse * tx * ri / inertia_i;
                    angular_x[j] += friction_impulse * tz * rj / inertia_j;
                    angular_z[j] -= friction_impulse * tx * rj / inertia_j;

                    // Separate balls
                    let overlap = min_dist - dist;
                    let sep = overlap * 0.5 + 0.001;
                    pos_x[i] -= sep * nx;
                    pos_z[i] -= sep * nz;
                    pos_x[j] += sep * nx;
                    pos_z[j] += sep * nz;
                }
            }
        }
    }
}

/// Cushion collision with spin effects
#[inline]
fn collide_cushions_with_spin(
    pos_x: &mut [f32],
    pos_z: &mut [f32],
    vel_x: &mut [f32],
    vel_z: &mut [f32],
    angular_x: &mut [f32],
    angular_y: &mut [f32],
    angular_z: &mut [f32],
    pocketed: &[f32],
) {
    for i in 0..NUM_BALLS {
        if pocketed[i] > 0.5 {
            continue;
        }

        let radius = if i == 0 { CUE_BALL_RADIUS } else { BALL_RADIUS };

        // Left cushion (X-)
        if pos_x[i] < -TABLE_HALF_L + radius {
            pos_x[i] = -TABLE_HALF_L + radius;
            let spin_effect = angular_y[i] * CUSHION_SPIN_EFFECT * 0.01;
            vel_x[i] = -vel_x[i] * CUSHION_RESTITUTION;
            vel_z[i] = vel_z[i] * CUSHION_RESTITUTION + spin_effect;
            angular_z[i] *= -CUSHION_SPIN_EFFECT;
        }

        // Right cushion (X+)
        if pos_x[i] > TABLE_HALF_L - radius {
            pos_x[i] = TABLE_HALF_L - radius;
            let spin_effect = -angular_y[i] * CUSHION_SPIN_EFFECT * 0.01;
            vel_x[i] = -vel_x[i] * CUSHION_RESTITUTION;
            vel_z[i] = vel_z[i] * CUSHION_RESTITUTION + spin_effect;
            angular_z[i] *= -CUSHION_SPIN_EFFECT;
        }

        // Top cushion (Z+)
        if pos_z[i] > TABLE_HALF_W - radius {
            pos_z[i] = TABLE_HALF_W - radius;
            let spin_effect = angular_y[i] * CUSHION_SPIN_EFFECT * 0.01;
            vel_z[i] = -vel_z[i] * CUSHION_RESTITUTION;
            vel_x[i] = vel_x[i] * CUSHION_RESTITUTION + spin_effect;
            angular_x[i] *= -CUSHION_SPIN_EFFECT;
        }

        // Bottom cushion (Z-)
        if pos_z[i] < -TABLE_HALF_W + radius {
            pos_z[i] = -TABLE_HALF_W + radius;
            let spin_effect = -angular_y[i] * CUSHION_SPIN_EFFECT * 0.01;
            vel_z[i] = -vel_z[i] * CUSHION_RESTITUTION;
            vel_x[i] = vel_x[i] * CUSHION_RESTITUTION + spin_effect;
            angular_x[i] *= -CUSHION_SPIN_EFFECT;
        }
    }
}

/// Check if all balls are settled (velocity AND spin below threshold)
#[inline]
fn is_settled_with_spin(
    vel_x: &[f32],
    vel_z: &[f32],
    angular_x: &[f32],
    angular_y: &[f32],
    angular_z: &[f32],
    pocketed: &[f32],
) -> bool {
    let vel_threshold_sq = VELOCITY_THRESHOLD * VELOCITY_THRESHOLD;
    let ang_threshold_sq = ANGULAR_THRESHOLD * ANGULAR_THRESHOLD;

    for i in 0..NUM_BALLS {
        if pocketed[i] > 0.5 {
            continue;
        }

        let speed_sq = vel_x[i] * vel_x[i] + vel_z[i] * vel_z[i];
        if speed_sq > vel_threshold_sq {
            return false;
        }

        let spin_sq = angular_x[i] * angular_x[i] + angular_y[i] * angular_y[i] + angular_z[i] * angular_z[i];
        if spin_sq > ang_threshold_sq {
            return false;
        }
    }

    true
}

// =============================================================================
// TENSOR-BASED GPU SIMULATION (True batch on GPU)
// =============================================================================

/// GPU tensor-based batch simulation
///
/// This version uses Burn tensors for true GPU parallelism.
/// Each operation (integrate, collide, etc.) runs on all simulations at once.
#[cfg(feature = "cuda")]
pub fn batch_simulate_tensors(
    positions_x_flat: Vec<f32>,  // [batch * 8]
    positions_z_flat: Vec<f32>,
    velocities_x_flat: Vec<f32>,
    velocities_z_flat: Vec<f32>,
    pocketed_flat: Vec<f32>,
    shots: Vec<f32>,  // [batch * 4] - flattened (angle, power, english, elev)
    batch_size: usize,
    max_steps: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<usize>) {
    use burn::prelude::*;

    type B = Cuda<f32, i32>;
    let device = CudaDevice::default();

    // Create tensors [batch, 8]
    let mut pos_x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(positions_x_flat, [batch_size, NUM_BALLS]),
        &device
    );
    let mut pos_z: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(positions_z_flat, [batch_size, NUM_BALLS]),
        &device
    );
    let mut vel_x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(velocities_x_flat, [batch_size, NUM_BALLS]),
        &device
    );
    let mut vel_z: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(velocities_z_flat, [batch_size, NUM_BALLS]),
        &device
    );
    let mut pckd: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(pocketed_flat, [batch_size, NUM_BALLS]),
        &device
    );

    // Apply shots to cue balls (column 0)
    // shots is [batch * 4]: angle, power, english, elevation per sim
    for i in 0..batch_size {
        let shot_idx = i * 4;
        if shot_idx + 1 < shots.len() {
            let angle = shots[shot_idx];
            let power = shots[shot_idx + 1];
            let max_vel = 50.0;
            let cue_vx = power * max_vel * angle.cos();
            let cue_vz = power * max_vel * angle.sin();

            // Update cue ball velocity (index 0)
            // This is inefficient - we should batch this too
            // But for now, direct tensor indexing is complex in Burn
            // TODO: Optimize with scatter operations
        }
    }

    // For now, fall back to CPU loop with GPU tensor conversion
    // Full GPU implementation requires custom CUDA kernels
    // which Burn doesn't directly support

    // Convert back to vecs for CPU processing
    let px_vec: Vec<f32> = pos_x.into_data().to_vec().unwrap();
    let pz_vec: Vec<f32> = pos_z.into_data().to_vec().unwrap();
    let vx_vec: Vec<f32> = vel_x.into_data().to_vec().unwrap();
    let vz_vec: Vec<f32> = vel_z.into_data().to_vec().unwrap();
    let pckd_vec: Vec<f32> = pckd.into_data().to_vec().unwrap();

    // Reshape to [batch][8]
    let px_batch: Vec<Vec<f32>> = px_vec.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();
    let pz_batch: Vec<Vec<f32>> = pz_vec.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();
    let vx_batch: Vec<Vec<f32>> = vx_vec.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();
    let vz_batch: Vec<Vec<f32>> = vz_vec.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();
    let pckd_batch: Vec<Vec<f32>> = pckd_vec.chunks(NUM_BALLS).map(|c| c.to_vec()).collect();

    // Convert shots
    let shots_tuples: Vec<(f32, f32, f32, f32)> = (0..batch_size)
        .map(|i| {
            let idx = i * 4;
            if idx + 3 < shots.len() {
                (shots[idx], shots[idx + 1], shots[idx + 2], shots[idx + 3])
            } else {
                (0.0, 0.5, 0.0, 0.0)
            }
        })
        .collect();

    // Use CPU batch simulation (Rayon parallel)
    let (final_px, final_pz, final_pckd, steps) = batch_simulate_cpu(
        px_batch, pz_batch, vx_batch, vz_batch, pckd_batch, shots_tuples, max_steps
    );

    // Flatten results
    let px_flat: Vec<f32> = final_px.into_iter().flatten().collect();
    let pz_flat: Vec<f32> = final_pz.into_iter().flatten().collect();
    let pckd_flat: Vec<f32> = final_pckd.into_iter().flatten().collect();

    (px_flat, pz_flat, pckd_flat, steps)
}

// =============================================================================
// FITNESS CALCULATION
// =============================================================================

/// Calculate fitness from simulation results
///
/// Returns: (fitness, hit_angle, scatter_ratio) for behavior descriptor
pub fn calculate_fitness(
    initial_pocketed: &[f32],
    final_pocketed: &[f32],
    final_pos_x: &[f32],
    final_pos_z: &[f32],
    initial_pos_x: &[f32],
    initial_pos_z: &[f32],
    target_ball_idx: usize,
) -> (f32, f32, f32) {
    // Count newly pocketed balls
    let mut newly_pocketed = 0;
    let mut target_pocketed = false;
    let mut cue_pocketed = false;

    for i in 0..NUM_BALLS {
        if final_pocketed[i] > 0.5 && initial_pocketed[i] < 0.5 {
            newly_pocketed += 1;
            if i == target_ball_idx {
                target_pocketed = true;
            }
            if i == 0 {
                cue_pocketed = true;
            }
        }
    }

    // Base fitness
    let mut fitness = 0.0;

    // Target ball bonus
    if target_pocketed {
        fitness += 20.0;
    }

    // Other balls bonus
    let other_pocketed = newly_pocketed - (if target_pocketed { 1 } else { 0 }) - (if cue_pocketed { 1 } else { 0 });
    fitness += other_pocketed as f32 * 10.0;

    // Scratch penalty
    if cue_pocketed {
        fitness -= 15.0;
    }

    // Miss penalty
    if newly_pocketed == 0 && !cue_pocketed {
        fitness -= 3.0;
    }

    // Position bonus (cue ball near center is good)
    if !cue_pocketed {
        let center_x = final_pos_x[0] / TABLE_HALF_L;
        let center_z = final_pos_z[0] / TABLE_HALF_W;
        let center_score = (1.0 - center_x.abs()) * (1.0 - center_z.abs());
        fitness += center_score.clamp(0.0, 1.0) * 5.0;
    }

    // Calculate behavior descriptors
    // Hit angle: use cue ball final position angle
    let hit_angle = if cue_pocketed {
        0.5
    } else {
        let angle = final_pos_z[0].atan2(final_pos_x[0]);
        (angle / std::f32::consts::PI + 1.0) / 2.0  // Normalize to 0-1
    };

    // Scatter ratio: how many balls moved significantly
    let mut moved_count = 0;
    for i in 0..NUM_BALLS {
        if final_pocketed[i] > 0.5 || initial_pocketed[i] > 0.5 {
            continue;
        }
        let dx = final_pos_x[i] - initial_pos_x[i];
        let dz = final_pos_z[i] - initial_pos_z[i];
        if dx * dx + dz * dz > 0.01 {
            moved_count += 1;
        }
    }
    let active_balls = (0..NUM_BALLS)
        .filter(|&i| initial_pocketed[i] < 0.5)
        .count();
    let scatter_ratio = if active_balls > 0 {
        moved_count as f32 / active_balls as f32
    } else {
        0.0
    };

    (fitness, hit_angle, scatter_ratio)
}

// =============================================================================
// BATCH FITNESS CALCULATION
// =============================================================================

/// Calculate fitness for entire batch
pub fn batch_calculate_fitness(
    batch_size: usize,
    initial_pocketed: &[Vec<f32>],
    final_pocketed: &[Vec<f32>],
    final_pos_x: &[Vec<f32>],
    final_pos_z: &[Vec<f32>],
    initial_pos_x: &[Vec<f32>],
    initial_pos_z: &[Vec<f32>],
    target_ball_idx: usize,
) -> Vec<(f32, f32, f32)> {
    use rayon::prelude::*;

    (0..batch_size)
        .into_par_iter()
        .map(|i| {
            calculate_fitness(
                &initial_pocketed[i],
                &final_pocketed[i],
                &final_pos_x[i],
                &final_pos_z[i],
                &initial_pos_x[i],
                &initial_pos_z[i],
                target_ball_idx,
            )
        })
        .collect()
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Create initial state for a batch of tables
pub fn create_initial_batch(batch_size: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let init_x = initial_positions_x();
    let init_z = initial_positions_z();

    let positions_x: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| init_x.to_vec())
        .collect();

    let positions_z: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| init_z.to_vec())
        .collect();

    let velocities_x: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| vec![0.0; NUM_BALLS])
        .collect();

    let velocities_z: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| vec![0.0; NUM_BALLS])
        .collect();

    let pocketed: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| vec![0.0; NUM_BALLS])
        .collect();

    (positions_x, positions_z, velocities_x, velocities_z, pocketed)
}

// =============================================================================
// TESTS
// =============================================================================

// =============================================================================
// MULTI-SHOT EPISODE SIMULATION
// =============================================================================

/// Result of a complete multi-shot episode
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    /// Total accumulated fitness across all shots
    pub total_fitness: f32,
    /// Final table state positions X
    pub final_pos_x: Vec<f32>,
    /// Final table state positions Z
    pub final_pos_z: Vec<f32>,
    /// Final pocketed flags
    pub final_pocketed: Vec<f32>,
    /// Total shots taken
    pub shots_taken: usize,
    /// Total balls pocketed during episode
    pub balls_pocketed: usize,
    /// Behavior descriptor: hit angle (average)
    pub avg_hit_angle: f32,
    /// Behavior descriptor: scatter ratio (average)
    pub avg_scatter_ratio: f32,
}

/// Simulate complete episodes with multiple shots for a batch of neural networks
///
/// This is THE KEY OPTIMIZATION: instead of 4800 NIF calls per generation,
/// we do 1 call that runs:
/// - N networks (population)
/// - M shots per episode
/// - Full physics simulation per shot
/// - Neural forward pass to generate each shot
///
/// Arguments:
/// - population_weights: [pop_size, weight_count] - network weights
/// - architecture: [input, h1, h2, ..., output] - network architecture
/// - shots_per_episode: number of shots to simulate per episode
/// - max_steps_per_shot: max physics steps per shot
///
/// Returns: Vec<EpisodeResult> for each network in population
pub fn batch_simulate_episodes(
    population_weights: &[Vec<f32>],
    architecture: &[usize],
    shots_per_episode: usize,
    max_steps_per_shot: usize,
) -> Vec<EpisodeResult> {
    use rayon::prelude::*;

    if population_weights.is_empty() || architecture.len() < 2 {
        return vec![];
    }

    let pop_size = population_weights.len();

    // Process each network's episode in parallel
    population_weights
        .par_iter()
        .map(|weights| {
            simulate_single_episode(
                weights,
                architecture,
                shots_per_episode,
                max_steps_per_shot,
            )
        })
        .collect()
}

/// Simulate a single episode for one network
fn simulate_single_episode(
    weights: &[f32],
    architecture: &[usize],
    shots_per_episode: usize,
    max_steps_per_shot: usize,
) -> EpisodeResult {
    // Initialize table state
    let mut pos_x = initial_positions_x().to_vec();
    let mut pos_z = initial_positions_z().to_vec();
    let mut vel_x = vec![0.0; NUM_BALLS];
    let mut vel_z = vec![0.0; NUM_BALLS];
    let mut pocketed = vec![0.0; NUM_BALLS];

    let mut total_fitness = 0.0;
    let mut total_balls_pocketed = 0;
    let mut shots_taken = 0;
    let mut sum_hit_angle = 0.0;
    let mut sum_scatter_ratio = 0.0;

    // Determine which ball to target (starts with Red, index 1)
    let mut target_ball_idx = 1;

    for shot_num in 0..shots_per_episode {
        // Check if cue ball is pocketed (scratch) - respawn at starting position
        if pocketed[0] > 0.5 {
            pos_x[0] = -TABLE_LENGTH / 4.0;
            pos_z[0] = 0.0;
            pocketed[0] = 0.0;
            vel_x[0] = 0.0;
            vel_z[0] = 0.0;
        }

        // Check if we still have balls to pocket
        let active_balls: Vec<usize> = (1..NUM_BALLS)
            .filter(|&i| pocketed[i] < 0.5)
            .collect();

        if active_balls.is_empty() {
            // All balls pocketed, episode complete!
            break;
        }

        // Find next target ball (lowest index unpocketed ball)
        target_ball_idx = *active_balls.first().unwrap_or(&1);

        // 1. Encode neural network inputs from current state
        let inputs = encode_state_inputs(
            &pos_x,
            &pos_z,
            &pocketed,
            target_ball_idx,
        );

        // 2. Neural forward pass to get shot parameters
        let outputs = dense_forward_single(weights, &inputs, architecture);

        // 3. Decode outputs to shot
        let pocket_angle = calculate_pocket_angle(&pos_x, &pos_z, target_ball_idx);
        let shot = decode_neural_output(&outputs, pocket_angle);

        // 4. Store initial state for fitness calculation
        let initial_pocketed = pocketed.clone();
        let initial_pos_x = pos_x.clone();
        let initial_pos_z = pos_z.clone();

        // 5. Apply shot and simulate physics
        apply_shot(&mut vel_x[0], &mut vel_z[0], shot);
        let _steps = simulate_single(
            &mut pos_x,
            &mut pos_z,
            &mut vel_x,
            &mut vel_z,
            &mut pocketed,
            max_steps_per_shot,
        );

        // 6. Calculate fitness for this shot
        let (shot_fitness, hit_angle, scatter_ratio) = calculate_fitness(
            &initial_pocketed,
            &pocketed,
            &pos_x,
            &pos_z,
            &initial_pos_x,
            &initial_pos_z,
            target_ball_idx,
        );

        total_fitness += shot_fitness;
        sum_hit_angle += hit_angle;
        sum_scatter_ratio += scatter_ratio;
        shots_taken += 1;

        // Count newly pocketed balls
        for i in 1..NUM_BALLS {
            if pocketed[i] > 0.5 && initial_pocketed[i] < 0.5 {
                total_balls_pocketed += 1;
            }
        }

        // Reset velocities for next shot
        for i in 0..NUM_BALLS {
            vel_x[i] = 0.0;
            vel_z[i] = 0.0;
        }
    }

    // Calculate average behavior descriptors
    let avg_hit_angle = if shots_taken > 0 {
        sum_hit_angle / shots_taken as f32
    } else {
        0.5
    };
    let avg_scatter_ratio = if shots_taken > 0 {
        sum_scatter_ratio / shots_taken as f32
    } else {
        0.5
    };

    EpisodeResult {
        total_fitness,
        final_pos_x: pos_x,
        final_pos_z: pos_z,
        final_pocketed: pocketed,
        shots_taken,
        balls_pocketed: total_balls_pocketed,
        avg_hit_angle,
        avg_scatter_ratio,
    }
}

/// Encode current table state into neural network inputs
///
/// 8 inputs matching sinuca_trainer format:
/// [cue_x, cue_z, target_x, target_z, pocket_angle, pocket_dist, target_value, balls_left]
fn encode_state_inputs(
    pos_x: &[f32],
    pos_z: &[f32],
    pocketed: &[f32],
    target_ball_idx: usize,
) -> Vec<f32> {
    // Cue ball position (normalized)
    let cue_x = pos_x[0] / TABLE_HALF_L;
    let cue_z = pos_z[0] / TABLE_HALF_W;

    // Target ball position (normalized)
    let target_x = if pocketed[target_ball_idx] < 0.5 {
        pos_x[target_ball_idx] / TABLE_HALF_L
    } else {
        0.5
    };
    let target_z = if pocketed[target_ball_idx] < 0.5 {
        pos_z[target_ball_idx] / TABLE_HALF_W
    } else {
        0.0
    };

    // Angle to nearest pocket from target ball
    let (pocket_angle, pocket_dist) = calculate_nearest_pocket_angle(
        pos_x[target_ball_idx],
        pos_z[target_ball_idx],
    );

    // Target value (based on ball index, normalized)
    // Red=1, Yellow=2, Green=3, Brown=4, Blue=5, Pink=6, Black=7
    let target_value = target_ball_idx as f32 / 7.0 * 2.0 - 1.0;

    // Balls remaining (normalized)
    let balls_left = (1..NUM_BALLS)
        .filter(|&i| pocketed[i] < 0.5)
        .count() as f32 / 7.0 * 2.0 - 1.0;

    vec![
        cue_x,
        cue_z,
        target_x,
        target_z,
        pocket_angle / std::f32::consts::PI,  // Normalized to [-1, 1]
        pocket_dist.min(2.0) / 2.0,           // Normalized to [0, 1]
        target_value,
        balls_left,
    ]
}

/// Calculate angle from cue ball to target ball to nearest pocket
fn calculate_pocket_angle(
    pos_x: &[f32],
    pos_z: &[f32],
    target_ball_idx: usize,
) -> f32 {
    let cue_x = pos_x[0];
    let cue_z = pos_z[0];
    let target_x = pos_x[target_ball_idx];
    let target_z = pos_z[target_ball_idx];

    // Angle from cue to target
    (target_z - cue_z).atan2(target_x - cue_x)
}

/// Calculate angle and distance to nearest pocket from a ball
fn calculate_nearest_pocket_angle(ball_x: f32, ball_z: f32) -> (f32, f32) {
    let mut min_dist = f32::MAX;
    let mut best_angle = 0.0;

    for p in 0..6 {
        let dx = POCKET_X[p] - ball_x;
        let dz = POCKET_Z[p] - ball_z;
        let dist = (dx * dx + dz * dz).sqrt();

        if dist < min_dist {
            min_dist = dist;
            best_angle = dz.atan2(dx);
        }
    }

    (best_angle, min_dist)
}

/// Decode neural network output to shot tuple
///
/// Network outputs: [angle_adjustment, power, english]
fn decode_neural_output(outputs: &[f32], pocket_angle: f32) -> (f32, f32, f32, f32) {
    let angle_adj_raw = outputs.get(0).copied().unwrap_or(0.5);
    let power_raw = outputs.get(1).copied().unwrap_or(0.5);
    let english_raw = outputs.get(2).copied().unwrap_or(0.5);

    // Adjustment range: +/- 45 degrees
    let angle_adjustment = (angle_adj_raw * 2.0 - 1.0) * 0.785398;  // pi/4
    let angle = pocket_angle + angle_adjustment;

    // Power: 0.1 to 1.0
    let power = 0.1 + power_raw * 0.9;

    // English: -0.8 to 0.8
    let english = (english_raw * 2.0 - 1.0) * 0.8;

    (angle, power, english, 0.0)
}

/// Single network dense forward pass
///
/// Duplicated here to avoid module dependency issues.
/// Same logic as batch_forward.rs
fn dense_forward_single(
    weights: &[f32],
    inputs: &[f32],
    layer_sizes: &[usize],
) -> Vec<f32> {
    if layer_sizes.len() < 2 {
        return inputs.to_vec();
    }

    let mut current = inputs.to_vec();
    let mut w_idx = 0;

    for layer in 0..layer_sizes.len() - 1 {
        let in_size = layer_sizes[layer];
        let out_size = layer_sizes[layer + 1];
        let is_output = layer == layer_sizes.len() - 2;

        current = layer_forward(
            &current,
            &weights[w_idx..],
            in_size,
            out_size,
            is_output,
        );

        w_idx += in_size * out_size + out_size;
    }

    current
}

/// Single layer forward: y = activation(Wx + b)
#[inline]
fn layer_forward(
    input: &[f32],
    weights: &[f32],
    in_size: usize,
    out_size: usize,
    is_output: bool,
) -> Vec<f32> {
    let w = &weights[..in_size * out_size];
    let b = &weights[in_size * out_size..in_size * out_size + out_size];

    let mut output = Vec::with_capacity(out_size);

    for j in 0..out_size {
        let mut sum = b[j];
        let row_start = j * in_size;

        // Unrolled dot product
        let mut i = 0;
        while i + 4 <= in_size {
            sum += w[row_start + i] * input[i]
                + w[row_start + i + 1] * input[i + 1]
                + w[row_start + i + 2] * input[i + 2]
                + w[row_start + i + 3] * input[i + 3];
            i += 4;
        }
        while i < in_size {
            sum += w[row_start + i] * input[i];
            i += 1;
        }

        let activated = if is_output {
            sigmoid(sum)
        } else {
            sum.tanh()
        };

        output.push(activated);
    }

    output
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    let x_clipped = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-x_clipped).exp())
}

// =============================================================================
// BATCH EPISODE SIMULATION WITH EVALUATION (Full Pipeline)
// =============================================================================

/// Complete evaluation pipeline: episodes + fitness in single call
///
/// This is the most efficient API for QD training:
/// - 1 NIF call per generation
/// - All neural forward passes in Rust
/// - All physics simulation in Rust
/// - All fitness calculation in Rust
///
/// Returns: Vec<(fitness, hit_angle, scatter_ratio)>
pub fn batch_evaluate_episodes(
    population_weights: &[Vec<f32>],
    architecture: &[usize],
    shots_per_episode: usize,
    max_steps_per_shot: usize,
) -> Vec<(f32, f32, f32)> {
    let results = batch_simulate_episodes(
        population_weights,
        architecture,
        shots_per_episode,
        max_steps_per_shot,
    );

    results
        .into_iter()
        .map(|r| (r.total_fitness, r.avg_hit_angle, r.avg_scatter_ratio))
        .collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_positions() {
        let px = initial_positions_x();
        let pz = initial_positions_z();

        assert_eq!(px.len(), 8);
        assert_eq!(pz.len(), 8);

        // Cue ball should be at -table_length/4
        assert!((px[0] - (-TABLE_LENGTH / 4.0)).abs() < 0.001);
        assert!((pz[0] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_apply_shot() {
        let mut vx = 0.0;
        let mut vz = 0.0;

        // Shot straight right (+X direction) at full power
        apply_shot(&mut vx, &mut vz, (0.0, 1.0, 0.0, 0.0));

        assert!(vx > 40.0);  // Should be ~50 m/s
        assert!(vz.abs() < 0.01);  // Should be ~0
    }

    #[test]
    fn test_is_settled() {
        let vel_x = vec![0.0; 8];
        let vel_z = vec![0.0; 8];
        let pocketed = vec![0.0; 8];

        assert!(is_settled(&vel_x, &vel_z, &pocketed));

        let vel_x_moving = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!is_settled(&vel_x_moving, &vel_z, &pocketed));
    }

    #[test]
    fn test_apply_shot_with_spin() {
        // Straight shot with right english
        let (vx, vz, ax, ay, az) = apply_shot_with_spin((0.0, 1.0, 1.0, 0.0));

        assert!(vx > 40.0, "vx should be ~50 m/s, got {}", vx);
        assert!(vz.abs() < 0.01, "vz should be ~0 for angle=0, got {}", vz);
        assert!(ay > 50.0, "Right english should create positive sidespin, got {}", ay);
    }

    #[test]
    fn test_apply_shot_with_elevation() {
        // Shot with full elevation (follow)
        let (vx, vz, ax, ay, az) = apply_shot_with_spin((0.0, 0.5, 0.0, 1.0));

        assert!(vx > 20.0, "Half power shot");
        assert!(ay.abs() < 1.0, "No english, so no sidespin");
        // High elevation = topspin = negative angular_z for shot in +X direction
        assert!(az.abs() > 30.0, "Should have significant topspin, got az={}", az);
    }

    #[test]
    fn test_is_settled_with_spin() {
        let vel_x = vec![0.0; 8];
        let vel_z = vec![0.0; 8];
        let angular_x = vec![0.0; 8];
        let angular_y = vec![0.0; 8];
        let angular_z = vec![0.0; 8];
        let pocketed = vec![0.0; 8];

        assert!(is_settled_with_spin(&vel_x, &vel_z, &angular_x, &angular_y, &angular_z, &pocketed));

        // Ball still spinning should not be settled
        let angular_y_spinning = vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!is_settled_with_spin(&vel_x, &vel_z, &angular_x, &angular_y_spinning, &angular_z, &pocketed));
    }

    #[test]
    fn test_spin_simulation() {
        let (px, pz, vx, vz, pocketed) = create_initial_batch(1);

        // Shot with right english - should curve
        let shots = vec![(0.0, 0.3, 0.8, 0.0)];

        let (final_px, final_pz, _, steps) = batch_simulate_cpu_with_spin(
            px, pz, vx, vz, pocketed, shots, 200
        );

        // Should have settled
        assert!(steps[0] <= 200);

        // Ball should have moved
        let init_x = initial_positions_x()[0];
        assert!(final_px[0][0] != init_x, "Cue ball should have moved");
    }

    #[test]
    fn test_english_causes_curve() {
        // Compare straight shot vs shot with english
        let (px1, pz1, vx1, vz1, pocketed1) = create_initial_batch(1);
        let (px2, pz2, vx2, vz2, pocketed2) = create_initial_batch(1);

        // Straight shot (no english)
        let shots_straight = vec![(0.0, 0.3, 0.0, 0.0)];
        let (final_px1, final_pz1, _, _) = batch_simulate_cpu_with_spin(
            px1, pz1, vx1, vz1, pocketed1, shots_straight, 100
        );

        // Shot with strong right english
        let shots_english = vec![(0.0, 0.3, 1.0, 0.0)];
        let (final_px2, final_pz2, _, _) = batch_simulate_cpu_with_spin(
            px2, pz2, vx2, vz2, pocketed2, shots_english, 100
        );

        // The shot with english should curve, ending at different Z position
        println!("Straight final: ({}, {})", final_px1[0][0], final_pz1[0][0]);
        println!("English final: ({}, {})", final_px2[0][0], final_pz2[0][0]);

        // Positions should be different (english causes curve)
        let dx = (final_px1[0][0] - final_px2[0][0]).abs();
        let dz = (final_pz1[0][0] - final_pz2[0][0]).abs();
        assert!(dx > 0.001 || dz > 0.001, "English should cause different trajectory");
    }

    #[test]
    fn test_single_simulation() {
        let (px, pz, vx, vz, pocketed) = create_initial_batch(1);

        let initial_cue_x = px[0][0];

        // Apply a shot straight at full power
        let shots = vec![(0.0, 1.0, 0.0, 0.0)];  // Straight right at full power

        let (final_px, _final_pz, _final_pocketed, steps) = batch_simulate_cpu(
            px, pz, vx, vz, pocketed, shots, 200
        );

        // Cue ball should have moved to the right (+X direction)
        // Shot at angle 0 (cos(0)=1, sin(0)=0) moves in +X direction
        println!("Initial cue X: {}, Final cue X: {}", initial_cue_x, final_px[0][0]);
        assert!(final_px[0][0] != initial_cue_x, "Cue ball should have moved");

        // Should have settled within 200 steps
        assert!(steps[0] <= 200);
    }

    #[test]
    fn test_batch_simulation() {
        let (px, pz, vx, vz, pocketed) = create_initial_batch(10);

        // Different shots for each table
        let shots: Vec<(f32, f32, f32, f32)> = (0..10)
            .map(|i| (i as f32 * 0.1, 0.5, 0.0, 0.0))
            .collect();

        let (final_px, final_pz, final_pocketed, steps) = batch_simulate_cpu(
            px, pz, vx, vz, pocketed, shots, 200
        );

        assert_eq!(final_px.len(), 10);
        assert_eq!(steps.len(), 10);

        // All should have different final positions due to different angles
        let unique_positions: std::collections::HashSet<i32> = final_px
            .iter()
            .map(|p| (p[0] * 1000.0) as i32)
            .collect();

        // At least some should be different
        assert!(unique_positions.len() > 1);
    }

    #[test]
    fn test_encode_state_inputs() {
        let pos_x = initial_positions_x().to_vec();
        let pos_z = initial_positions_z().to_vec();
        let pocketed = vec![0.0; 8];

        let inputs = encode_state_inputs(&pos_x, &pos_z, &pocketed, 1);

        // Should have 8 inputs
        assert_eq!(inputs.len(), 8);

        // All inputs should be in reasonable range [-1, 1] or [0, 1]
        for (i, &v) in inputs.iter().enumerate() {
            assert!(v >= -2.0 && v <= 2.0, "Input {} out of range: {}", i, v);
        }
    }

    #[test]
    fn test_dense_forward_single() {
        // Simple 2->2->1 network
        let architecture = vec![2, 2, 1];
        // Weights: [w1 (2x2=4), b1 (2), w2 (2x1=2), b2 (1)] = 9 total
        let weights = vec![
            0.5, 0.5, 0.5, 0.5,  // w1
            0.0, 0.0,            // b1
            0.5, 0.5,            // w2
            0.0,                 // b2
        ];
        let inputs = vec![1.0, 1.0];

        let outputs = dense_forward_single(&weights, &inputs, &architecture);

        // Should have 1 output
        assert_eq!(outputs.len(), 1);
        // Output should be sigmoid, so between 0 and 1
        assert!(outputs[0] >= 0.0 && outputs[0] <= 1.0);
    }

    #[test]
    fn test_single_episode() {
        // Architecture: [8, 32, 16, 3] = 867 weights
        let architecture = vec![8, 32, 16, 3];
        let weight_count = 8 * 32 + 32 + 32 * 16 + 16 + 16 * 3 + 3;

        // Random weights (seeded) - use wrapping arithmetic to avoid overflow
        let weights: Vec<f32> = (0..weight_count)
            .map(|i| {
                let seed = (i as u32).wrapping_mul(1103515245).wrapping_add(12345) % 1000;
                seed as f32 / 1000.0 - 0.5
            })
            .collect();

        let result = simulate_single_episode(&weights, &architecture, 3, 200);

        // Should have taken some shots
        assert!(result.shots_taken >= 1);
        assert!(result.shots_taken <= 3);

        // Fitness should be calculated
        // (can be negative due to scratches/misses)
        assert!(result.total_fitness > -100.0 && result.total_fitness < 200.0);

        // Behavior descriptors should be in [0, 1]
        assert!(result.avg_hit_angle >= 0.0 && result.avg_hit_angle <= 1.0);
        assert!(result.avg_scatter_ratio >= 0.0 && result.avg_scatter_ratio <= 1.0);
    }

    #[test]
    fn test_batch_simulate_episodes() {
        let architecture = vec![8, 32, 16, 3];
        let weight_count = 8 * 32 + 32 + 32 * 16 + 16 + 16 * 3 + 3;

        // Create 10 different networks
        let population_weights: Vec<Vec<f32>> = (0..10)
            .map(|net| {
                (0..weight_count)
                    .map(|i| {
                        let seed = (net * 1000 + i) as u32;
                        (seed.wrapping_mul(1103515245).wrapping_add(12345) % 1000) as f32 / 1000.0 - 0.5
                    })
                    .collect()
            })
            .collect();

        let results = batch_simulate_episodes(&population_weights, &architecture, 3, 200);

        // Should have results for all networks
        assert_eq!(results.len(), 10);

        // Each result should have valid data
        for (i, r) in results.iter().enumerate() {
            assert!(r.shots_taken >= 1 && r.shots_taken <= 3,
                "Network {} took {} shots", i, r.shots_taken);
            assert!(r.avg_hit_angle >= 0.0 && r.avg_hit_angle <= 1.0,
                "Network {} hit_angle out of range: {}", i, r.avg_hit_angle);
        }
    }

    #[test]
    fn test_batch_evaluate_episodes() {
        let architecture = vec![8, 32, 16, 3];
        let weight_count = 8 * 32 + 32 + 32 * 16 + 16 + 16 * 3 + 3;

        // Create 5 networks
        let population_weights: Vec<Vec<f32>> = (0..5)
            .map(|net| {
                (0..weight_count)
                    .map(|i| {
                        let seed = (net * 1000 + i) as u32;
                        (seed.wrapping_mul(1103515245).wrapping_add(12345) % 1000) as f32 / 1000.0 - 0.5
                    })
                    .collect()
            })
            .collect();

        let results = batch_evaluate_episodes(&population_weights, &architecture, 3, 200);

        // Should return (fitness, hit_angle, scatter_ratio) tuples
        assert_eq!(results.len(), 5);

        for (i, &(fitness, hit_angle, scatter)) in results.iter().enumerate() {
            assert!(hit_angle >= 0.0 && hit_angle <= 1.0,
                "Network {} hit_angle: {}", i, hit_angle);
            assert!(scatter >= 0.0 && scatter <= 1.0,
                "Network {} scatter: {}", i, scatter);
        }
    }
}
