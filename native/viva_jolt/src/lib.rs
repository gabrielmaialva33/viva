//! VIVA JoltPhysics NIF - AAA Physics for Gleam
//!
//! Complete physics engine with raycast, kinematic character control, and **ContactListener**.
//!
//! v0.2.0 - ContactListener support via jolt-rust's joltc-sys

use rustler::{Atom, Encoder, Env, OwnedEnv, LocalPid, ResourceArc, Term};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Mutex, Once};
use joltc_sys::*;

// =============================================================================
// GLOBAL INITIALIZATION
// =============================================================================

static JOLT_INIT: Once = Once::new();

fn ensure_jolt_initialized() {
    JOLT_INIT.call_once(|| {
        unsafe {
            JPC_RegisterDefaultAllocator();
            JPC_FactoryInit();
            JPC_RegisterTypes();
        }
    });
}

// =============================================================================
// ATOMS
// =============================================================================

mod atoms {
    rustler::atoms! {
        ok,
        error,
        nil,
        hit,
        miss,
        static_body,
        kinematic,
        dynamic,
        // Contact events
        contact_added,
        contact_persisted,
        contact_removed,
    }
}

// =============================================================================
// LAYER CALLBACKS
// =============================================================================

const NUM_BROAD_PHASE_LAYERS: u32 = 2;
const BROAD_PHASE_LAYER_NON_MOVING: u8 = 0;
const BROAD_PHASE_LAYER_MOVING: u8 = 1;
const OBJECT_LAYER_NON_MOVING: u16 = 0;
const OBJECT_LAYER_MOVING: u16 = 1;

unsafe extern "C" fn get_num_broad_phase_layers(_self: *const c_void) -> u32 {
    NUM_BROAD_PHASE_LAYERS
}

unsafe extern "C" fn get_broad_phase_layer(_self: *const c_void, layer: JPC_ObjectLayer) -> JPC_BroadPhaseLayer {
    if layer == OBJECT_LAYER_NON_MOVING {
        BROAD_PHASE_LAYER_NON_MOVING
    } else {
        BROAD_PHASE_LAYER_MOVING
    }
}

unsafe extern "C" fn object_vs_broad_should_collide(
    _self: *const c_void,
    _layer1: JPC_ObjectLayer,
    _layer2: JPC_BroadPhaseLayer,
) -> bool {
    true
}

unsafe extern "C" fn object_pair_should_collide(
    _self: *const c_void,
    layer1: JPC_ObjectLayer,
    layer2: JPC_ObjectLayer,
) -> bool {
    !(layer1 == OBJECT_LAYER_NON_MOVING && layer2 == OBJECT_LAYER_NON_MOVING)
}

// =============================================================================
// CONTACT EVENT (stored in queue for Gleam to poll)
// =============================================================================

#[derive(Debug, Clone)]
pub struct ContactEvent {
    pub event_type: ContactEventType,
    pub body1_index: u32,
    pub body2_index: u32,
    pub normal: [f32; 3],
    pub penetration_depth: f32,
    pub contact_point: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
pub enum ContactEventType {
    Added,
    Persisted,
    Removed,
}

// =============================================================================
// CONTACT LISTENER STATE (shared between callback and world)
// =============================================================================

struct ContactListenerState {
    events: VecDeque<ContactEvent>,
    body_id_to_index: Vec<JPC_BodyID>,
}

impl ContactListenerState {
    fn new() -> Self {
        Self {
            events: VecDeque::with_capacity(256),
            body_id_to_index: Vec::new(),
        }
    }

    fn find_body_index(&self, body_id: JPC_BodyID) -> Option<u32> {
        self.body_id_to_index.iter().position(|&id| id == body_id).map(|i| i as u32)
    }
}

// =============================================================================
// CONTACT LISTENER CALLBACKS
// =============================================================================

unsafe extern "C" fn on_contact_validate(
    _self: *mut c_void,
    _body1: *const JPC_Body,
    _body2: *const JPC_Body,
    _base_offset: JPC_RVec3,
    _collision_result: *const JPC_CollideShapeResult,
) -> JPC_ValidateResult {
    // Accept all contacts
    JPC_VALIDATE_RESULT_ACCEPT_ALL_CONTACTS
}

unsafe extern "C" fn on_contact_added(
    self_: *mut c_void,
    body1: *const JPC_Body,
    body2: *const JPC_Body,
    manifold: *const JPC_ContactManifold,
    _settings: *mut JPC_ContactSettings,
) {
    let state = &mut *(self_ as *mut Mutex<ContactListenerState>);
    if let Ok(mut state) = state.lock() {
        let body1_id = JPC_Body_GetID(body1);
        let body2_id = JPC_Body_GetID(body2);

        if let (Some(idx1), Some(idx2)) = (state.find_body_index(body1_id), state.find_body_index(body2_id)) {
            let m = &*manifold;
            state.events.push_back(ContactEvent {
                event_type: ContactEventType::Added,
                body1_index: idx1,
                body2_index: idx2,
                normal: [m.WorldSpaceNormal.x, m.WorldSpaceNormal.y, m.WorldSpaceNormal.z],
                penetration_depth: m.PenetrationDepth,
                contact_point: [m.BaseOffset.x as f32, m.BaseOffset.y as f32, m.BaseOffset.z as f32],
            });
        }
    }
}

unsafe extern "C" fn on_contact_persisted(
    self_: *mut c_void,
    body1: *const JPC_Body,
    body2: *const JPC_Body,
    manifold: *const JPC_ContactManifold,
    _settings: *mut JPC_ContactSettings,
) {
    let state = &mut *(self_ as *mut Mutex<ContactListenerState>);
    if let Ok(mut state) = state.lock() {
        let body1_id = JPC_Body_GetID(body1);
        let body2_id = JPC_Body_GetID(body2);

        if let (Some(idx1), Some(idx2)) = (state.find_body_index(body1_id), state.find_body_index(body2_id)) {
            let m = &*manifold;
            state.events.push_back(ContactEvent {
                event_type: ContactEventType::Persisted,
                body1_index: idx1,
                body2_index: idx2,
                normal: [m.WorldSpaceNormal.x, m.WorldSpaceNormal.y, m.WorldSpaceNormal.z],
                penetration_depth: m.PenetrationDepth,
                contact_point: [m.BaseOffset.x as f32, m.BaseOffset.y as f32, m.BaseOffset.z as f32],
            });
        }
    }
}

unsafe extern "C" fn on_contact_removed(
    self_: *mut c_void,
    sub_shape_pair: *const JPC_SubShapeIDPair,
) {
    let state = &mut *(self_ as *mut Mutex<ContactListenerState>);
    if let Ok(mut state) = state.lock() {
        let pair = &*sub_shape_pair;
        if let (Some(idx1), Some(idx2)) = (state.find_body_index(pair.Body1ID), state.find_body_index(pair.Body2ID)) {
            state.events.push_back(ContactEvent {
                event_type: ContactEventType::Removed,
                body1_index: idx1,
                body2_index: idx2,
                normal: [0.0, 0.0, 0.0],
                penetration_depth: 0.0,
                contact_point: [0.0, 0.0, 0.0],
            });
        }
    }
}

// =============================================================================
// PHYSICS WORLD
// =============================================================================

pub struct PhysicsWorld {
    physics_system: *mut JPC_PhysicsSystem,
    body_interface: *mut JPC_BodyInterface,
    temp_allocator: *mut JPC_TempAllocatorImpl,
    job_system: *mut JPC_JobSystemThreadPool,
    broad_phase_layer_interface: *mut JPC_BroadPhaseLayerInterface,
    object_vs_broad_filter: *mut JPC_ObjectVsBroadPhaseLayerFilter,
    object_pair_filter: *mut JPC_ObjectLayerPairFilter,
    contact_listener: *mut JPC_ContactListener,
    contact_state: Box<Mutex<ContactListenerState>>,
    bodies: Vec<JPC_BodyID>,
    tick: u64,
}

unsafe impl Send for PhysicsWorld {}
unsafe impl Sync for PhysicsWorld {}

impl PhysicsWorld {
    pub fn new() -> Self {
        ensure_jolt_initialized();

        unsafe {
            let temp_allocator = JPC_TempAllocatorImpl_new(10 * 1024 * 1024);
            let job_system = JPC_JobSystemThreadPool_new2(
                JPC_MAX_PHYSICS_JOBS as u32,
                JPC_MAX_PHYSICS_BARRIERS as u32,
            );

            let broad_phase_fns = JPC_BroadPhaseLayerInterfaceFns {
                GetNumBroadPhaseLayers: Some(get_num_broad_phase_layers),
                GetBroadPhaseLayer: Some(get_broad_phase_layer),
            };
            let broad_phase_layer_interface = JPC_BroadPhaseLayerInterface_new(
                std::ptr::null(),
                broad_phase_fns,
            );

            let object_vs_broad_fns = JPC_ObjectVsBroadPhaseLayerFilterFns {
                ShouldCollide: Some(object_vs_broad_should_collide),
            };
            let object_vs_broad_filter = JPC_ObjectVsBroadPhaseLayerFilter_new(
                std::ptr::null(),
                object_vs_broad_fns,
            );

            let object_pair_fns = JPC_ObjectLayerPairFilterFns {
                ShouldCollide: Some(object_pair_should_collide),
            };
            let object_pair_filter = JPC_ObjectLayerPairFilter_new(
                std::ptr::null(),
                object_pair_fns,
            );

            let physics_system = JPC_PhysicsSystem_new();
            JPC_PhysicsSystem_Init(
                physics_system,
                65536,
                0,
                65536,
                10240,
                broad_phase_layer_interface,
                object_vs_broad_filter,
                object_pair_filter,
            );

            let body_interface = JPC_PhysicsSystem_GetBodyInterface(physics_system);

            // Create contact listener state
            let contact_state = Box::new(Mutex::new(ContactListenerState::new()));
            let state_ptr = &*contact_state as *const Mutex<ContactListenerState> as *mut c_void;

            let contact_fns = JPC_ContactListenerFns {
                OnContactValidate: Some(on_contact_validate),
                OnContactAdded: Some(on_contact_added),
                OnContactPersisted: Some(on_contact_persisted),
                OnContactRemoved: Some(on_contact_removed),
            };
            let contact_listener = JPC_ContactListener_new(state_ptr, contact_fns);

            // Register contact listener
            JPC_PhysicsSystem_SetContactListener(physics_system, contact_listener);

            PhysicsWorld {
                physics_system,
                body_interface,
                temp_allocator,
                job_system,
                broad_phase_layer_interface,
                object_vs_broad_filter,
                object_pair_filter,
                contact_listener,
                contact_state,
                bodies: Vec::new(),
                tick: 0,
            }
        }
    }

    // =========================================================================
    // SIMULATION
    // =========================================================================

    pub fn step(&mut self, dt: f32) {
        // Update body index lookup for contact listener
        if let Ok(mut state) = self.contact_state.lock() {
            state.body_id_to_index = self.bodies.clone();
        }

        unsafe {
            JPC_PhysicsSystem_Update(
                self.physics_system,
                dt,
                1,
                self.temp_allocator,
                self.job_system as *mut JPC_JobSystem,
            );
            self.tick += 1;
        }
    }

    pub fn optimize(&mut self) {
        unsafe {
            JPC_PhysicsSystem_OptimizeBroadPhase(self.physics_system);
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    // =========================================================================
    // CONTACT EVENTS
    // =========================================================================

    pub fn get_contact_events(&self) -> Vec<ContactEvent> {
        if let Ok(mut state) = self.contact_state.lock() {
            state.events.drain(..).collect()
        } else {
            Vec::new()
        }
    }

    pub fn has_contact_events(&self) -> bool {
        if let Ok(state) = self.contact_state.lock() {
            !state.events.is_empty()
        } else {
            false
        }
    }

    // =========================================================================
    // BODY CREATION
    // =========================================================================

    pub fn create_sphere(&mut self, pos: [f32; 3], radius: f32, motion_type: JPC_MotionType) -> u32 {
        unsafe {
            let mut settings: JPC_SphereShapeSettings = std::mem::zeroed();
            JPC_SphereShapeSettings_default(&mut settings);
            settings.Radius = radius;
            settings.Density = 1000.0;

            let mut shape: *mut JPC_Shape = std::ptr::null_mut();
            let mut error: *mut JPC_String = std::ptr::null_mut();
            JPC_SphereShapeSettings_Create(&settings, &mut shape, &mut error);

            if shape.is_null() { return u32::MAX; }
            self.add_body(pos, shape, motion_type)
        }
    }

    pub fn create_box(&mut self, pos: [f32; 3], half_extents: [f32; 3], motion_type: JPC_MotionType) -> u32 {
        unsafe {
            let mut settings: JPC_BoxShapeSettings = std::mem::zeroed();
            JPC_BoxShapeSettings_default(&mut settings);
            settings.HalfExtent = JPC_Vec3 {
                x: half_extents[0],
                y: half_extents[1],
                z: half_extents[2],
                _w: 0.0,
            };
            settings.Density = 1000.0;

            let mut shape: *mut JPC_Shape = std::ptr::null_mut();
            let mut error: *mut JPC_String = std::ptr::null_mut();
            JPC_BoxShapeSettings_Create(&settings, &mut shape, &mut error);

            if shape.is_null() { return u32::MAX; }
            self.add_body(pos, shape, motion_type)
        }
    }

    pub fn create_capsule(&mut self, pos: [f32; 3], half_height: f32, radius: f32, motion_type: JPC_MotionType) -> u32 {
        unsafe {
            let mut settings: JPC_CapsuleShapeSettings = std::mem::zeroed();
            JPC_CapsuleShapeSettings_default(&mut settings);
            settings.HalfHeightOfCylinder = half_height;
            settings.Radius = radius;
            settings.Density = 1000.0;

            let mut shape: *mut JPC_Shape = std::ptr::null_mut();
            let mut error: *mut JPC_String = std::ptr::null_mut();
            JPC_CapsuleShapeSettings_Create(&settings, &mut shape, &mut error);

            if shape.is_null() { return u32::MAX; }
            self.add_body(pos, shape, motion_type)
        }
    }

    pub fn create_cylinder(&mut self, pos: [f32; 3], half_height: f32, radius: f32, motion_type: JPC_MotionType) -> u32 {
        unsafe {
            let mut settings: JPC_CylinderShapeSettings = std::mem::zeroed();
            JPC_CylinderShapeSettings_default(&mut settings);
            settings.HalfHeight = half_height;
            settings.Radius = radius;
            settings.Density = 1000.0;

            let mut shape: *mut JPC_Shape = std::ptr::null_mut();
            let mut error: *mut JPC_String = std::ptr::null_mut();
            JPC_CylinderShapeSettings_Create(&settings, &mut shape, &mut error);

            if shape.is_null() { return u32::MAX; }
            self.add_body(pos, shape, motion_type)
        }
    }

    unsafe fn add_body(&mut self, pos: [f32; 3], shape: *mut JPC_Shape, motion_type: JPC_MotionType) -> u32 {
        let layer = if motion_type == JPC_MOTION_TYPE_STATIC {
            OBJECT_LAYER_NON_MOVING
        } else {
            OBJECT_LAYER_MOVING
        };

        let mut settings: JPC_BodyCreationSettings = std::mem::zeroed();
        JPC_BodyCreationSettings_default(&mut settings);
        settings.Position = JPC_RVec3 { x: pos[0], y: pos[1], z: pos[2], _w: 0.0 };
        settings.Rotation = JPC_Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };
        settings.Shape = shape;
        settings.MotionType = motion_type;
        settings.ObjectLayer = layer;
        settings.Friction = 0.5;
        settings.Restitution = 0.5;

        let activation = if motion_type == JPC_MOTION_TYPE_STATIC {
            JPC_ACTIVATION_DONT_ACTIVATE
        } else {
            JPC_ACTIVATION_ACTIVATE
        };

        let body_id = JPC_BodyInterface_CreateAndAddBody(self.body_interface, &settings, activation);

        let index = self.bodies.len() as u32;
        self.bodies.push(body_id);
        index
    }

    // =========================================================================
    // POSITION & ROTATION
    // =========================================================================

    pub fn get_position(&self, index: u32) -> Option<[f64; 3]> {
        self.bodies.get(index as usize).map(|&body_id| {
            unsafe {
                let pos = JPC_BodyInterface_GetPosition(self.body_interface, body_id);
                [pos.x as f64, pos.y as f64, pos.z as f64]
            }
        })
    }

    pub fn set_position(&mut self, index: u32, pos: [f32; 3]) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                let p = JPC_RVec3 { x: pos[0], y: pos[1], z: pos[2], _w: 0.0 };
                JPC_BodyInterface_SetPosition(self.body_interface, body_id, p, JPC_ACTIVATION_ACTIVATE);
            }
            true
        } else {
            false
        }
    }

    pub fn get_rotation(&self, index: u32) -> Option<[f64; 4]> {
        self.bodies.get(index as usize).map(|&body_id| {
            unsafe {
                let rot = JPC_BodyInterface_GetRotation(self.body_interface, body_id);
                [rot.x as f64, rot.y as f64, rot.z as f64, rot.w as f64]
            }
        })
    }

    pub fn set_rotation(&mut self, index: u32, rot: [f32; 4]) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                let r = JPC_Quat { x: rot[0], y: rot[1], z: rot[2], w: rot[3] };
                JPC_BodyInterface_SetRotation(self.body_interface, body_id, r, JPC_ACTIVATION_ACTIVATE);
            }
            true
        } else {
            false
        }
    }

    // =========================================================================
    // LINEAR VELOCITY
    // =========================================================================

    pub fn get_velocity(&self, index: u32) -> Option<[f64; 3]> {
        self.bodies.get(index as usize).map(|&body_id| {
            unsafe {
                let vel = JPC_BodyInterface_GetLinearVelocity(self.body_interface, body_id);
                [vel.x as f64, vel.y as f64, vel.z as f64]
            }
        })
    }

    pub fn set_velocity(&mut self, index: u32, vel: [f32; 3]) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                let v = JPC_Vec3 { x: vel[0], y: vel[1], z: vel[2], _w: 0.0 };
                JPC_BodyInterface_SetLinearVelocity(self.body_interface, body_id, v);
            }
            true
        } else {
            false
        }
    }

    // =========================================================================
    // ANGULAR VELOCITY
    // =========================================================================

    pub fn get_angular_velocity(&self, index: u32) -> Option<[f64; 3]> {
        self.bodies.get(index as usize).map(|&body_id| {
            unsafe {
                let vel = JPC_BodyInterface_GetAngularVelocity(self.body_interface, body_id);
                [vel.x as f64, vel.y as f64, vel.z as f64]
            }
        })
    }

    pub fn set_angular_velocity(&mut self, index: u32, vel: [f32; 3]) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                let v = JPC_Vec3 { x: vel[0], y: vel[1], z: vel[2], _w: 0.0 };
                JPC_BodyInterface_SetAngularVelocity(self.body_interface, body_id, v);
            }
            true
        } else {
            false
        }
    }

    // =========================================================================
    // FORCES & IMPULSES
    // =========================================================================

    pub fn add_force(&mut self, index: u32, force: [f32; 3]) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                let f = JPC_Vec3 { x: force[0], y: force[1], z: force[2], _w: 0.0 };
                JPC_BodyInterface_AddForce(self.body_interface, body_id, f);
            }
            true
        } else {
            false
        }
    }

    pub fn add_torque(&mut self, index: u32, torque: [f32; 3]) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                let t = JPC_Vec3 { x: torque[0], y: torque[1], z: torque[2], _w: 0.0 };
                JPC_BodyInterface_AddTorque(self.body_interface, body_id, t);
            }
            true
        } else {
            false
        }
    }

    pub fn add_impulse(&mut self, index: u32, impulse: [f32; 3]) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                let i = JPC_Vec3 { x: impulse[0], y: impulse[1], z: impulse[2], _w: 0.0 };
                JPC_BodyInterface_AddImpulse(self.body_interface, body_id, i);
            }
            true
        } else {
            false
        }
    }

    pub fn add_angular_impulse(&mut self, index: u32, impulse: [f32; 3]) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                let i = JPC_Vec3 { x: impulse[0], y: impulse[1], z: impulse[2], _w: 0.0 };
                JPC_BodyInterface_AddAngularImpulse(self.body_interface, body_id, i);
            }
            true
        } else {
            false
        }
    }

    // =========================================================================
    // KINEMATIC MOVEMENT (Character Controller)
    // =========================================================================

    pub fn move_kinematic(&mut self, index: u32, target_pos: [f32; 3], target_rot: [f32; 4], dt: f32) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                let pos = JPC_RVec3 { x: target_pos[0], y: target_pos[1], z: target_pos[2], _w: 0.0 };
                let rot = JPC_Quat { x: target_rot[0], y: target_rot[1], z: target_rot[2], w: target_rot[3] };
                JPC_BodyInterface_MoveKinematic(self.body_interface, body_id, pos, rot, dt);
            }
            true
        } else {
            false
        }
    }

    // =========================================================================
    // BODY PROPERTIES
    // =========================================================================

    pub fn is_active(&self, index: u32) -> bool {
        self.bodies.get(index as usize).map_or(false, |&body_id| {
            unsafe { JPC_BodyInterface_IsActive(self.body_interface, body_id) }
        })
    }

    pub fn activate(&mut self, index: u32) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                JPC_BodyInterface_ActivateBody(self.body_interface, body_id);
            }
            true
        } else {
            false
        }
    }

    pub fn deactivate(&mut self, index: u32) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                JPC_BodyInterface_DeactivateBody(self.body_interface, body_id);
            }
            true
        } else {
            false
        }
    }

    pub fn set_friction(&mut self, index: u32, friction: f32) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                JPC_BodyInterface_SetFriction(self.body_interface, body_id, friction);
            }
            true
        } else {
            false
        }
    }

    pub fn get_friction(&self, index: u32) -> Option<f32> {
        self.bodies.get(index as usize).map(|&body_id| {
            unsafe { JPC_BodyInterface_GetFriction(self.body_interface, body_id) }
        })
    }

    pub fn set_restitution(&mut self, index: u32, restitution: f32) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                JPC_BodyInterface_SetRestitution(self.body_interface, body_id, restitution);
            }
            true
        } else {
            false
        }
    }

    pub fn get_restitution(&self, index: u32) -> Option<f32> {
        self.bodies.get(index as usize).map(|&body_id| {
            unsafe { JPC_BodyInterface_GetRestitution(self.body_interface, body_id) }
        })
    }

    pub fn set_gravity_factor(&mut self, index: u32, factor: f32) -> bool {
        if let Some(&body_id) = self.bodies.get(index as usize) {
            unsafe {
                JPC_BodyInterface_SetGravityFactor(self.body_interface, body_id, factor);
            }
            true
        } else {
            false
        }
    }

    pub fn get_gravity_factor(&self, index: u32) -> Option<f32> {
        self.bodies.get(index as usize).map(|&body_id| {
            unsafe { JPC_BodyInterface_GetGravityFactor(self.body_interface, body_id) }
        })
    }

    // =========================================================================
    // RAYCAST
    // =========================================================================

    pub fn cast_ray(&self, origin: [f32; 3], direction: [f32; 3]) -> Option<(u32, f64, [f64; 3])> {
        unsafe {
            let narrow_phase = JPC_PhysicsSystem_GetNarrowPhaseQuery(self.physics_system);

            let ray = JPC_RRayCast {
                Origin: JPC_RVec3 { x: origin[0], y: origin[1], z: origin[2], _w: 0.0 },
                Direction: JPC_Vec3 { x: direction[0], y: direction[1], z: direction[2], _w: 0.0 },
            };

            let mut args: JPC_NarrowPhaseQuery_CastRayArgs = std::mem::zeroed();
            args.Ray = ray;

            let hit = JPC_NarrowPhaseQuery_CastRay(narrow_phase, &mut args);

            if hit {
                let hit_body_id = args.Result.BodyID;
                let body_index = self.bodies.iter().position(|&b| b == hit_body_id);

                if let Some(idx) = body_index {
                    let fraction = args.Result.Fraction as f64;
                    let hit_point = [
                        (origin[0] + direction[0] * args.Result.Fraction) as f64,
                        (origin[1] + direction[1] * args.Result.Fraction) as f64,
                        (origin[2] + direction[2] * args.Result.Fraction) as f64,
                    ];
                    return Some((idx as u32, fraction, hit_point));
                }
            }
            None
        }
    }
}

impl Drop for PhysicsWorld {
    fn drop(&mut self) {
        unsafe {
            for &body_id in &self.bodies {
                JPC_BodyInterface_RemoveBody(self.body_interface, body_id);
                JPC_BodyInterface_DestroyBody(self.body_interface, body_id);
            }
            JPC_ContactListener_delete(self.contact_listener);
            JPC_PhysicsSystem_delete(self.physics_system);
            JPC_TempAllocatorImpl_delete(self.temp_allocator);
            JPC_JobSystemThreadPool_delete(self.job_system);
            JPC_BroadPhaseLayerInterface_delete(self.broad_phase_layer_interface);
            JPC_ObjectVsBroadPhaseLayerFilter_delete(self.object_vs_broad_filter);
            JPC_ObjectLayerPairFilter_delete(self.object_pair_filter);
        }
    }
}

// =============================================================================
// RESOURCE
// =============================================================================

struct WorldResource {
    world: Mutex<PhysicsWorld>,
}

// =============================================================================
// NIFs - WORLD
// =============================================================================

#[rustler::nif]
fn world_new() -> ResourceArc<WorldResource> {
    ResourceArc::new(WorldResource {
        world: Mutex::new(PhysicsWorld::new()),
    })
}

#[rustler::nif(schedule = "DirtyCpu")]
fn world_step(resource: ResourceArc<WorldResource>, dt: f64) -> u64 {
    let mut world = resource.world.lock().unwrap();
    world.step(dt as f32);
    world.tick()
}

#[rustler::nif(schedule = "DirtyCpu")]
fn world_step_n(resource: ResourceArc<WorldResource>, n: u32, dt: f64) -> u64 {
    let mut world = resource.world.lock().unwrap();
    for _ in 0..n {
        world.step(dt as f32);
    }
    world.tick()
}

#[rustler::nif]
fn world_optimize(resource: ResourceArc<WorldResource>) -> Atom {
    let mut world = resource.world.lock().unwrap();
    world.optimize();
    atoms::ok()
}

#[rustler::nif]
fn body_count(resource: ResourceArc<WorldResource>) -> u32 {
    let world = resource.world.lock().unwrap();
    world.body_count() as u32
}

#[rustler::nif]
fn tick(resource: ResourceArc<WorldResource>) -> u64 {
    let world = resource.world.lock().unwrap();
    world.tick()
}

// =============================================================================
// NIFs - CONTACT EVENTS
// =============================================================================

#[rustler::nif]
fn get_contacts<'a>(env: Env<'a>, resource: ResourceArc<WorldResource>) -> Term<'a> {
    let world = resource.world.lock().unwrap();
    let events = world.get_contact_events();

    let encoded: Vec<Term<'a>> = events.iter().map(|event| {
        let event_type = match event.event_type {
            ContactEventType::Added => atoms::contact_added(),
            ContactEventType::Persisted => atoms::contact_persisted(),
            ContactEventType::Removed => atoms::contact_removed(),
        };

        (
            event_type,
            event.body1_index,
            event.body2_index,
            (event.normal[0] as f64, event.normal[1] as f64, event.normal[2] as f64),
            event.penetration_depth as f64,
            (event.contact_point[0] as f64, event.contact_point[1] as f64, event.contact_point[2] as f64),
        ).encode(env)
    }).collect();

    encoded.encode(env)
}

#[rustler::nif]
fn has_contacts(resource: ResourceArc<WorldResource>) -> bool {
    let world = resource.world.lock().unwrap();
    world.has_contact_events()
}

// =============================================================================
// NIFs - BODY CREATION
// =============================================================================

#[rustler::nif]
fn create_box(
    resource: ResourceArc<WorldResource>,
    pos: (f64, f64, f64),
    half_extents: (f64, f64, f64),
    motion_type: Atom,
) -> u32 {
    let mut world = resource.world.lock().unwrap();
    let mt = motion_type_from_atom(motion_type);
    world.create_box(
        [pos.0 as f32, pos.1 as f32, pos.2 as f32],
        [half_extents.0 as f32, half_extents.1 as f32, half_extents.2 as f32],
        mt,
    )
}

#[rustler::nif]
fn create_sphere(
    resource: ResourceArc<WorldResource>,
    pos: (f64, f64, f64),
    radius: f64,
    motion_type: Atom,
) -> u32 {
    let mut world = resource.world.lock().unwrap();
    let mt = motion_type_from_atom(motion_type);
    world.create_sphere([pos.0 as f32, pos.1 as f32, pos.2 as f32], radius as f32, mt)
}

#[rustler::nif]
fn create_capsule(
    resource: ResourceArc<WorldResource>,
    pos: (f64, f64, f64),
    half_height: f64,
    radius: f64,
    motion_type: Atom,
) -> u32 {
    let mut world = resource.world.lock().unwrap();
    let mt = motion_type_from_atom(motion_type);
    world.create_capsule([pos.0 as f32, pos.1 as f32, pos.2 as f32], half_height as f32, radius as f32, mt)
}

#[rustler::nif]
fn create_cylinder(
    resource: ResourceArc<WorldResource>,
    pos: (f64, f64, f64),
    half_height: f64,
    radius: f64,
    motion_type: Atom,
) -> u32 {
    let mut world = resource.world.lock().unwrap();
    let mt = motion_type_from_atom(motion_type);
    world.create_cylinder([pos.0 as f32, pos.1 as f32, pos.2 as f32], half_height as f32, radius as f32, mt)
}

// =============================================================================
// NIFs - POSITION & ROTATION
// =============================================================================

#[rustler::nif]
fn get_position<'a>(env: Env<'a>, resource: ResourceArc<WorldResource>, index: u32) -> Term<'a> {
    let world = resource.world.lock().unwrap();
    match world.get_position(index) {
        Some([x, y, z]) => (atoms::ok(), (x, y, z)).encode(env),
        None => (atoms::error(), atoms::nil()).encode(env),
    }
}

#[rustler::nif]
fn set_position(resource: ResourceArc<WorldResource>, index: u32, pos: (f64, f64, f64)) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.set_position(index, [pos.0 as f32, pos.1 as f32, pos.2 as f32])
}

#[rustler::nif]
fn get_rotation<'a>(env: Env<'a>, resource: ResourceArc<WorldResource>, index: u32) -> Term<'a> {
    let world = resource.world.lock().unwrap();
    match world.get_rotation(index) {
        Some([x, y, z, w]) => (atoms::ok(), (x, y, z, w)).encode(env),
        None => (atoms::error(), atoms::nil()).encode(env),
    }
}

#[rustler::nif]
fn set_rotation(resource: ResourceArc<WorldResource>, index: u32, rot: (f64, f64, f64, f64)) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.set_rotation(index, [rot.0 as f32, rot.1 as f32, rot.2 as f32, rot.3 as f32])
}

// =============================================================================
// NIFs - VELOCITY
// =============================================================================

#[rustler::nif]
fn get_velocity<'a>(env: Env<'a>, resource: ResourceArc<WorldResource>, index: u32) -> Term<'a> {
    let world = resource.world.lock().unwrap();
    match world.get_velocity(index) {
        Some([x, y, z]) => (atoms::ok(), (x, y, z)).encode(env),
        None => (atoms::error(), atoms::nil()).encode(env),
    }
}

#[rustler::nif]
fn set_velocity(resource: ResourceArc<WorldResource>, index: u32, vel: (f64, f64, f64)) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.set_velocity(index, [vel.0 as f32, vel.1 as f32, vel.2 as f32])
}

#[rustler::nif]
fn get_angular_velocity<'a>(env: Env<'a>, resource: ResourceArc<WorldResource>, index: u32) -> Term<'a> {
    let world = resource.world.lock().unwrap();
    match world.get_angular_velocity(index) {
        Some([x, y, z]) => (atoms::ok(), (x, y, z)).encode(env),
        None => (atoms::error(), atoms::nil()).encode(env),
    }
}

#[rustler::nif]
fn set_angular_velocity(resource: ResourceArc<WorldResource>, index: u32, vel: (f64, f64, f64)) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.set_angular_velocity(index, [vel.0 as f32, vel.1 as f32, vel.2 as f32])
}

// =============================================================================
// NIFs - FORCES & IMPULSES
// =============================================================================

#[rustler::nif]
fn add_force(resource: ResourceArc<WorldResource>, index: u32, force: (f64, f64, f64)) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.add_force(index, [force.0 as f32, force.1 as f32, force.2 as f32])
}

#[rustler::nif]
fn add_torque(resource: ResourceArc<WorldResource>, index: u32, torque: (f64, f64, f64)) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.add_torque(index, [torque.0 as f32, torque.1 as f32, torque.2 as f32])
}

#[rustler::nif]
fn add_impulse(resource: ResourceArc<WorldResource>, index: u32, impulse: (f64, f64, f64)) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.add_impulse(index, [impulse.0 as f32, impulse.1 as f32, impulse.2 as f32])
}

#[rustler::nif]
fn add_angular_impulse(resource: ResourceArc<WorldResource>, index: u32, impulse: (f64, f64, f64)) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.add_angular_impulse(index, [impulse.0 as f32, impulse.1 as f32, impulse.2 as f32])
}

// =============================================================================
// NIFs - KINEMATIC MOVEMENT
// =============================================================================

#[rustler::nif]
fn move_kinematic(
    resource: ResourceArc<WorldResource>,
    index: u32,
    target_pos: (f64, f64, f64),
    target_rot: (f64, f64, f64, f64),
    dt: f64,
) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.move_kinematic(
        index,
        [target_pos.0 as f32, target_pos.1 as f32, target_pos.2 as f32],
        [target_rot.0 as f32, target_rot.1 as f32, target_rot.2 as f32, target_rot.3 as f32],
        dt as f32,
    )
}

// =============================================================================
// NIFs - BODY PROPERTIES
// =============================================================================

#[rustler::nif]
fn is_active(resource: ResourceArc<WorldResource>, index: u32) -> bool {
    let world = resource.world.lock().unwrap();
    world.is_active(index)
}

#[rustler::nif]
fn activate_body(resource: ResourceArc<WorldResource>, index: u32) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.activate(index)
}

#[rustler::nif]
fn deactivate_body(resource: ResourceArc<WorldResource>, index: u32) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.deactivate(index)
}

#[rustler::nif]
fn set_friction(resource: ResourceArc<WorldResource>, index: u32, friction: f64) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.set_friction(index, friction as f32)
}

#[rustler::nif]
fn get_friction<'a>(env: Env<'a>, resource: ResourceArc<WorldResource>, index: u32) -> Term<'a> {
    let world = resource.world.lock().unwrap();
    match world.get_friction(index) {
        Some(f) => (atoms::ok(), f as f64).encode(env),
        None => (atoms::error(), atoms::nil()).encode(env),
    }
}

#[rustler::nif]
fn set_restitution(resource: ResourceArc<WorldResource>, index: u32, restitution: f64) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.set_restitution(index, restitution as f32)
}

#[rustler::nif]
fn get_restitution<'a>(env: Env<'a>, resource: ResourceArc<WorldResource>, index: u32) -> Term<'a> {
    let world = resource.world.lock().unwrap();
    match world.get_restitution(index) {
        Some(r) => (atoms::ok(), r as f64).encode(env),
        None => (atoms::error(), atoms::nil()).encode(env),
    }
}

#[rustler::nif]
fn set_gravity_factor(resource: ResourceArc<WorldResource>, index: u32, factor: f64) -> bool {
    let mut world = resource.world.lock().unwrap();
    world.set_gravity_factor(index, factor as f32)
}

#[rustler::nif]
fn get_gravity_factor<'a>(env: Env<'a>, resource: ResourceArc<WorldResource>, index: u32) -> Term<'a> {
    let world = resource.world.lock().unwrap();
    match world.get_gravity_factor(index) {
        Some(f) => (atoms::ok(), f as f64).encode(env),
        None => (atoms::error(), atoms::nil()).encode(env),
    }
}

// =============================================================================
// NIFs - RAYCAST
// =============================================================================

#[rustler::nif]
fn cast_ray<'a>(
    env: Env<'a>,
    resource: ResourceArc<WorldResource>,
    origin: (f64, f64, f64),
    direction: (f64, f64, f64),
) -> Term<'a> {
    let world = resource.world.lock().unwrap();
    match world.cast_ray(
        [origin.0 as f32, origin.1 as f32, origin.2 as f32],
        [direction.0 as f32, direction.1 as f32, direction.2 as f32],
    ) {
        Some((body_index, fraction, hit_point)) => {
            (atoms::hit(), body_index, fraction, (hit_point[0], hit_point[1], hit_point[2])).encode(env)
        }
        None => atoms::miss().encode(env),
    }
}

// =============================================================================
// NIFs - UTILITY
// =============================================================================

#[rustler::nif]
fn native_check() -> &'static str {
    "VIVA_JOLT_v0.2.0_CONTACT_LISTENER"
}

fn motion_type_from_atom(atom: Atom) -> JPC_MotionType {
    if atom == atoms::static_body() {
        JPC_MOTION_TYPE_STATIC
    } else if atom == atoms::kinematic() {
        JPC_MOTION_TYPE_KINEMATIC
    } else {
        JPC_MOTION_TYPE_DYNAMIC
    }
}

// =============================================================================
// NIF INIT
// =============================================================================

rustler::init!("Elixir.Viva.Jolt.Native", load = load);

fn load(env: Env, _info: Term) -> bool {
    rustler::resource!(WorldResource, env);
    true
}
