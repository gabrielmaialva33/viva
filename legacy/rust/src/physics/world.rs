//! PhysicsWorld - Jolt Physics wrapper for VIVA
//!
//! Provides deterministic rigid body simulation with snapshot/rollback
//! capabilities for Active Inference predictions.
//!
//! Uses joltc-sys directly for full API access (step, bodies, etc).

use bevy_ecs::system::Resource;
use glam::Vec3;
use joltc_sys::*;
use std::ffi::c_void;
use std::ptr;
use std::sync::Once;

// Jolt initialization (once per process)
static JOLT_INIT: Once = Once::new();

/// Initialize Jolt Physics (call once per process)
fn init_jolt() {
    JOLT_INIT.call_once(|| unsafe {
        JPC_RegisterDefaultAllocator();
        JPC_FactoryInit();
        JPC_RegisterTypes();
    });
}

// ============================================================================
// Layer constants
// ============================================================================

pub const OL_NON_MOVING: JPC_ObjectLayer = 0;
pub const OL_MOVING: JPC_ObjectLayer = 1;

pub const BPL_NON_MOVING: JPC_BroadPhaseLayer = 0;
pub const BPL_MOVING: JPC_BroadPhaseLayer = 1;
pub const BPL_COUNT: u32 = 2;

// ============================================================================
// Layer interface callbacks (C ABI)
// ============================================================================

unsafe extern "C" fn bpl_get_num_broad_phase_layers(_this: *const c_void) -> u32 {
    BPL_COUNT
}

unsafe extern "C" fn bpl_get_broad_phase_layer(
    _this: *const c_void,
    layer: JPC_ObjectLayer,
) -> JPC_BroadPhaseLayer {
    match layer {
        OL_NON_MOVING => BPL_NON_MOVING,
        OL_MOVING => BPL_MOVING,
        _ => BPL_MOVING,
    }
}

const BPL_FNS: JPC_BroadPhaseLayerInterfaceFns = JPC_BroadPhaseLayerInterfaceFns {
    GetNumBroadPhaseLayers: Some(bpl_get_num_broad_phase_layers),
    GetBroadPhaseLayer: Some(bpl_get_broad_phase_layer),
};

unsafe extern "C" fn ovb_should_collide(
    _this: *const c_void,
    layer1: JPC_ObjectLayer,
    layer2: JPC_BroadPhaseLayer,
) -> bool {
    match layer1 {
        OL_NON_MOVING => layer2 == BPL_MOVING,
        OL_MOVING => true,
        _ => true,
    }
}

const OVB_FNS: JPC_ObjectVsBroadPhaseLayerFilterFns = JPC_ObjectVsBroadPhaseLayerFilterFns {
    ShouldCollide: Some(ovb_should_collide),
};

unsafe extern "C" fn ovo_should_collide(
    _this: *const c_void,
    layer1: JPC_ObjectLayer,
    layer2: JPC_ObjectLayer,
) -> bool {
    match layer1 {
        OL_NON_MOVING => layer2 == OL_MOVING,
        OL_MOVING => true,
        _ => true,
    }
}

const OVO_FNS: JPC_ObjectLayerPairFilterFns = JPC_ObjectLayerPairFilterFns {
    ShouldCollide: Some(ovo_should_collide),
};

// ============================================================================
// Snapshot types
// ============================================================================

/// Snapshot of physics state for rollback
#[derive(Clone, Default)]
pub struct PhysicsSnapshot {
    /// Serialized body states
    pub body_states: Vec<BodyState>,
}

/// State of a single body (for snapshot/rollback)
#[derive(Clone, Debug, Default)]
pub struct BodyState {
    pub id: JPC_BodyID,
    pub position: Vec3,
    pub rotation: [f32; 4], // Quaternion [x, y, z, w]
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub is_active: bool,
}

/// Configuration for physics world
#[derive(Clone, Debug)]
pub struct PhysicsConfig {
    /// Gravity vector (default: -9.81 Y)
    pub gravity: Vec3,
    /// Maximum bodies in simulation
    pub max_bodies: u32,
    /// Maximum body pairs for collision
    pub max_body_pairs: u32,
    /// Maximum contact constraints
    pub max_contact_constraints: u32,
    /// Temp allocator size in bytes (default: 10MB)
    pub temp_allocator_size: usize,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            max_bodies: 1024,
            max_body_pairs: 1024,
            max_contact_constraints: 1024,
            temp_allocator_size: 10 * 1024 * 1024, // 10 MB
        }
    }
}

// ============================================================================
// PhysicsWorld
// ============================================================================

/// Collision event for qualia mapping
#[derive(Clone, Debug)]
pub struct CollisionEvent {
    pub body1: JPC_BodyID,
    pub body2: JPC_BodyID,
    pub impulse: f32,
    pub contact_point: Vec3,
}

/// Jolt Physics world with full API access via joltc-sys
#[derive(Resource)]
pub struct PhysicsWorld {
    // Raw pointers to Jolt objects
    system: *mut JPC_PhysicsSystem,
    temp_allocator: *mut JPC_TempAllocatorImpl,
    job_system: *mut JPC_JobSystemThreadPool,
    broad_phase_layer_interface: *mut JPC_BroadPhaseLayerInterface,
    object_vs_broad_phase_filter: *mut JPC_ObjectVsBroadPhaseLayerFilter,
    object_layer_pair_filter: *mut JPC_ObjectLayerPairFilter,

    config: PhysicsConfig,
    initialized: bool,
    step_count: u64,

    // Tracked bodies for snapshot/rollback
    tracked_bodies: Vec<JPC_BodyID>,

    // Collision events for qualia (cleared each step)
    collision_events: Vec<CollisionEvent>,
}

// Safety: PhysicsWorld manages Jolt resources and is only accessed
// through Bevy's single-threaded resource system.
unsafe impl Send for PhysicsWorld {}
unsafe impl Sync for PhysicsWorld {}

impl PhysicsWorld {
    /// Create a new physics world with default configuration
    pub fn new() -> Self {
        Self::with_config(PhysicsConfig::default())
    }

    /// Create a new physics world with custom configuration
    pub fn with_config(config: PhysicsConfig) -> Self {
        // Initialize Jolt (once per process)
        init_jolt();

        unsafe {
            // Create temp allocator
            let temp_allocator = JPC_TempAllocatorImpl_new(config.temp_allocator_size as u32);

            // Create job system (uses all available threads)
            let job_system = JPC_JobSystemThreadPool_new2(
                JPC_MAX_PHYSICS_JOBS as u32,
                JPC_MAX_PHYSICS_BARRIERS as u32,
            );

            // Create layer interfaces
            let broad_phase_layer_interface =
                JPC_BroadPhaseLayerInterface_new(ptr::null(), BPL_FNS);
            let object_vs_broad_phase_filter =
                JPC_ObjectVsBroadPhaseLayerFilter_new(ptr::null_mut(), OVB_FNS);
            let object_layer_pair_filter =
                JPC_ObjectLayerPairFilter_new(ptr::null_mut(), OVO_FNS);

            // Create physics system
            let system = JPC_PhysicsSystem_new();

            // Initialize physics system
            JPC_PhysicsSystem_Init(
                system,
                config.max_bodies,
                0, // num_body_mutexes (0 = auto)
                config.max_body_pairs,
                config.max_contact_constraints,
                broad_phase_layer_interface,
                object_vs_broad_phase_filter,
                object_layer_pair_filter,
            );

            Self {
                system,
                temp_allocator,
                job_system,
                broad_phase_layer_interface,
                object_vs_broad_phase_filter,
                object_layer_pair_filter,
                config,
                initialized: true,
                step_count: 0,
                tracked_bodies: Vec::new(),
                collision_events: Vec::new(),
            }
        }
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get gravity
    pub fn gravity(&self) -> Vec3 {
        self.config.gravity
    }

    /// Get step count
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Get body interface for adding/removing bodies
    pub fn body_interface(&self) -> *mut JPC_BodyInterface {
        unsafe { JPC_PhysicsSystem_GetBodyInterface(self.system) }
    }

    // =========================================================================
    // Simulation
    // =========================================================================

    /// Step physics simulation (REAL implementation)
    pub fn step(&mut self, delta_time: f32) {
        if !self.initialized {
            return;
        }

        unsafe {
            JPC_PhysicsSystem_Update(
                self.system,
                delta_time,
                1, // collision_steps
                self.temp_allocator,
                self.job_system,
            );
        }

        self.step_count += 1;
    }

    /// Step physics N times
    pub fn step_n(&mut self, delta_time: f32, steps: u32) {
        for _ in 0..steps {
            self.step(delta_time);
        }
    }

    // =========================================================================
    // Body Management
    // =========================================================================

    /// Create a static floor body
    pub fn create_floor(&mut self) -> JPC_BodyID {
        unsafe {
            let body_interface = self.body_interface();

            // Create box shape for floor
            let settings = JPC_BoxShapeSettings {
                HalfExtent: JPC_Vec3 {
                    x: 100.0,
                    y: 1.0,
                    z: 100.0,
                    _w: 100.0,
                },
                ..Default::default()
            };

            let mut shape: *mut JPC_Shape = ptr::null_mut();
            let mut err: *mut JPC_String = ptr::null_mut();

            if !JPC_BoxShapeSettings_Create(&settings, &mut shape, &mut err) {
                return JPC_BodyID::default();
            }

            let body_settings = JPC_BodyCreationSettings {
                Position: JPC_RVec3 {
                    x: 0.0,
                    y: -1.0,
                    z: 0.0,
                    _w: 0.0,
                },
                MotionType: JPC_MOTION_TYPE_STATIC,
                ObjectLayer: OL_NON_MOVING,
                Shape: shape,
                ..Default::default()
            };

            let body = JPC_BodyInterface_CreateBody(body_interface, &body_settings);
            let body_id = JPC_Body_GetID(body);
            JPC_BodyInterface_AddBody(body_interface, body_id, JPC_ACTIVATION_DONT_ACTIVATE);

            // Track this body for snapshot/rollback
            self.tracked_bodies.push(body_id);

            body_id
        }
    }

    /// Create a dynamic sphere body
    pub fn create_sphere(&mut self, position: Vec3, radius: f32) -> JPC_BodyID {
        unsafe {
            let body_interface = self.body_interface();

            let settings = JPC_SphereShapeSettings {
                Radius: radius,
                ..Default::default()
            };

            let mut shape: *mut JPC_Shape = ptr::null_mut();
            let mut err: *mut JPC_String = ptr::null_mut();

            if !JPC_SphereShapeSettings_Create(&settings, &mut shape, &mut err) {
                return JPC_BodyID::default();
            }

            let body_settings = JPC_BodyCreationSettings {
                Position: JPC_RVec3 {
                    x: position.x,
                    y: position.y,
                    z: position.z,
                    _w: 0.0,
                },
                MotionType: JPC_MOTION_TYPE_DYNAMIC,
                ObjectLayer: OL_MOVING,
                Shape: shape,
                ..Default::default()
            };

            let body = JPC_BodyInterface_CreateBody(body_interface, &body_settings);
            let body_id = JPC_Body_GetID(body);
            JPC_BodyInterface_AddBody(body_interface, body_id, JPC_ACTIVATION_ACTIVATE);

            // Track this body for snapshot/rollback
            self.tracked_bodies.push(body_id);

            body_id
        }
    }

    /// Get body position
    pub fn get_position(&self, body_id: JPC_BodyID) -> Vec3 {
        unsafe {
            let body_interface = JPC_PhysicsSystem_GetBodyInterface(self.system);
            let pos = JPC_BodyInterface_GetPosition(body_interface, body_id);
            Vec3::new(pos.x, pos.y, pos.z)
        }
    }

    /// Get body velocity
    pub fn get_velocity(&self, body_id: JPC_BodyID) -> Vec3 {
        unsafe {
            let body_interface = JPC_PhysicsSystem_GetBodyInterface(self.system);
            let vel = JPC_BodyInterface_GetLinearVelocity(body_interface, body_id);
            Vec3::new(vel.x, vel.y, vel.z)
        }
    }

    /// Set body velocity
    pub fn set_velocity(&mut self, body_id: JPC_BodyID, velocity: Vec3) {
        unsafe {
            let body_interface = self.body_interface();
            let vel = JPC_Vec3 {
                x: velocity.x,
                y: velocity.y,
                z: velocity.z,
                _w: 0.0,
            };
            JPC_BodyInterface_SetLinearVelocity(body_interface, body_id, vel);
        }
    }

    /// Check if body is active
    pub fn is_active(&self, body_id: JPC_BodyID) -> bool {
        unsafe {
            let body_interface = JPC_PhysicsSystem_GetBodyInterface(self.system);
            JPC_BodyInterface_IsActive(body_interface, body_id)
        }
    }

    /// Remove a body from the world
    pub fn remove_body(&mut self, body_id: JPC_BodyID) {
        unsafe {
            let body_interface = self.body_interface();
            JPC_BodyInterface_RemoveBody(body_interface, body_id);
            JPC_BodyInterface_DestroyBody(body_interface, body_id);
        }

        // Remove from tracking
        self.tracked_bodies.retain(|&id| id != body_id);
    }

    /// Get rotation as quaternion [x, y, z, w]
    pub fn get_rotation(&self, body_id: JPC_BodyID) -> [f32; 4] {
        unsafe {
            let body_interface = JPC_PhysicsSystem_GetBodyInterface(self.system);
            let rot = JPC_BodyInterface_GetRotation(body_interface, body_id);
            [rot.x, rot.y, rot.z, rot.w]
        }
    }

    /// Set rotation from quaternion [x, y, z, w]
    pub fn set_rotation(&mut self, body_id: JPC_BodyID, rotation: [f32; 4]) {
        unsafe {
            let body_interface = self.body_interface();
            let rot = JPC_Quat {
                x: rotation[0],
                y: rotation[1],
                z: rotation[2],
                w: rotation[3],
            };
            JPC_BodyInterface_SetRotation(body_interface, body_id, rot, JPC_ACTIVATION_ACTIVATE);
        }
    }

    /// Get angular velocity
    pub fn get_angular_velocity(&self, body_id: JPC_BodyID) -> Vec3 {
        unsafe {
            let body_interface = JPC_PhysicsSystem_GetBodyInterface(self.system);
            let vel = JPC_BodyInterface_GetAngularVelocity(body_interface, body_id);
            Vec3::new(vel.x, vel.y, vel.z)
        }
    }

    /// Set angular velocity
    pub fn set_angular_velocity(&mut self, body_id: JPC_BodyID, velocity: Vec3) {
        unsafe {
            let body_interface = self.body_interface();
            let vel = JPC_Vec3 {
                x: velocity.x,
                y: velocity.y,
                z: velocity.z,
                _w: 0.0,
            };
            JPC_BodyInterface_SetAngularVelocity(body_interface, body_id, vel);
        }
    }

    /// Set position (teleport)
    pub fn set_position(&mut self, body_id: JPC_BodyID, position: Vec3) {
        unsafe {
            let body_interface = self.body_interface();
            let pos = JPC_RVec3 {
                x: position.x,
                y: position.y,
                z: position.z,
                _w: 0.0,
            };
            JPC_BodyInterface_SetPosition(body_interface, body_id, pos, JPC_ACTIVATION_ACTIVATE);
        }
    }

    /// Get collision events from last step
    pub fn collision_events(&self) -> &[CollisionEvent] {
        &self.collision_events
    }

    /// Clear collision events
    pub fn clear_collision_events(&mut self) {
        self.collision_events.clear();
    }

    /// Get number of tracked bodies
    pub fn tracked_body_count(&self) -> usize {
        self.tracked_bodies.len()
    }

    // =========================================================================
    // Snapshot/Rollback API (for Active Inference)
    // =========================================================================

    /// Save current physics state for later rollback
    ///
    /// Manually serializes all tracked body states since joltc-sys doesn't
    /// expose Jolt's native StateRecorder.
    pub fn save_state(&self) -> PhysicsSnapshot {
        let mut body_states = Vec::with_capacity(self.tracked_bodies.len());

        for &body_id in &self.tracked_bodies {
            let position = self.get_position(body_id);
            let rotation = self.get_rotation(body_id);
            let velocity = self.get_velocity(body_id);
            let angular_velocity = self.get_angular_velocity(body_id);
            let is_active = self.is_active(body_id);

            body_states.push(BodyState {
                id: body_id,
                position,
                rotation,
                velocity,
                angular_velocity,
                is_active,
            });
        }

        PhysicsSnapshot { body_states }
    }

    /// Restore physics state from snapshot
    ///
    /// Restores all body positions, rotations, and velocities.
    pub fn restore_state(&mut self, snapshot: &PhysicsSnapshot) {
        for state in &snapshot.body_states {
            // Check if body still exists in our tracking
            if !self.tracked_bodies.contains(&state.id) {
                continue;
            }

            self.set_position(state.id, state.position);
            self.set_rotation(state.id, state.rotation);
            self.set_velocity(state.id, state.velocity);
            self.set_angular_velocity(state.id, state.angular_velocity);

            // Activate/deactivate based on snapshot
            if state.is_active {
                unsafe {
                    let body_interface = self.body_interface();
                    JPC_BodyInterface_ActivateBody(body_interface, state.id);
                }
            } else {
                unsafe {
                    let body_interface = self.body_interface();
                    JPC_BodyInterface_DeactivateBody(body_interface, state.id);
                }
            }
        }
    }

    /// Check if native SaveState is available
    pub fn has_native_snapshot(&self) -> bool {
        false // Using manual implementation instead
    }

    /// Check if manual snapshot is available
    pub fn has_manual_snapshot(&self) -> bool {
        true
    }
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PhysicsWorld {
    fn drop(&mut self) {
        if !self.initialized {
            return;
        }

        unsafe {
            // Clean up in reverse order of creation
            JPC_PhysicsSystem_delete(self.system);
            JPC_BroadPhaseLayerInterface_delete(self.broad_phase_layer_interface);
            JPC_ObjectVsBroadPhaseLayerFilter_delete(self.object_vs_broad_phase_filter);
            JPC_ObjectLayerPairFilter_delete(self.object_layer_pair_filter);
            JPC_JobSystemThreadPool_delete(self.job_system);
            JPC_TempAllocatorImpl_delete(self.temp_allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_world_init() {
        let world = PhysicsWorld::new();
        assert!(world.is_initialized());
        assert_eq!(world.gravity(), Vec3::new(0.0, -9.81, 0.0));
    }

    #[test]
    fn test_physics_step_real() {
        let mut world = PhysicsWorld::new();
        assert_eq!(world.step_count(), 0);
        world.step(1.0 / 60.0);
        assert_eq!(world.step_count(), 1);
    }

    #[test]
    fn test_create_floor() {
        let mut world = PhysicsWorld::new();
        let floor_id = world.create_floor();
        // Floor should be static, not active
        assert!(!world.is_active(floor_id));
    }

    #[test]
    fn test_create_sphere_and_simulate() {
        let mut world = PhysicsWorld::new();

        // Create floor
        let _floor = world.create_floor();

        // Create sphere above floor
        let sphere = world.create_sphere(Vec3::new(0.0, 5.0, 0.0), 0.5);

        // Sphere should be active
        assert!(world.is_active(sphere));

        // Get initial position
        let pos1 = world.get_position(sphere);
        assert!((pos1.y - 5.0).abs() < 0.1);

        // Simulate some steps
        for _ in 0..60 {
            world.step(1.0 / 60.0);
        }

        // Sphere should have fallen
        let pos2 = world.get_position(sphere);
        assert!(pos2.y < pos1.y, "Sphere should fall due to gravity");
    }

    #[test]
    fn test_snapshot_rollback() {
        let mut world = PhysicsWorld::new();

        // Create floor and sphere
        let _floor = world.create_floor();
        let sphere = world.create_sphere(Vec3::new(0.0, 5.0, 0.0), 0.5);

        // Save initial state
        let snapshot = world.save_state();
        let initial_pos = world.get_position(sphere);

        // Simulate some steps (sphere falls)
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }

        // Sphere should have moved
        let mid_pos = world.get_position(sphere);
        assert!(mid_pos.y < initial_pos.y, "Sphere should have fallen");

        // Rollback to initial state
        world.restore_state(&snapshot);

        // Position should be restored
        let restored_pos = world.get_position(sphere);
        assert!(
            (restored_pos.y - initial_pos.y).abs() < 0.1,
            "Position should be restored after rollback"
        );
    }

    #[test]
    fn test_tracked_body_count() {
        let mut world = PhysicsWorld::new();
        assert_eq!(world.tracked_body_count(), 0);

        let _floor = world.create_floor();
        assert_eq!(world.tracked_body_count(), 1);

        let sphere = world.create_sphere(Vec3::new(0.0, 5.0, 0.0), 0.5);
        assert_eq!(world.tracked_body_count(), 2);

        world.remove_body(sphere);
        assert_eq!(world.tracked_body_count(), 1);
    }
}
