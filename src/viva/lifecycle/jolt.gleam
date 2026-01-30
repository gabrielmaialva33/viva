//// VIVA JoltPhysics - AAA Physics for Gleam
////
//// High-performance physics simulation using JoltPhysics (Horizon Forbidden West engine).
//// Complete API: raycast, kinematic character control, forces, torques, and more.

// =============================================================================
// TYPES
// =============================================================================

/// 3D Vector
pub type Vec3 {
  Vec3(x: Float, y: Float, z: Float)
}

/// Quaternion (x, y, z, w)
pub type Quat {
  Quat(x: Float, y: Float, z: Float, w: Float)
}

/// Motion type determines how a body behaves
pub type MotionType {
  /// Never moves, infinite mass (floors, walls)
  Static
  /// Moves via code, ignores forces (elevators, platforms, characters)
  Kinematic
  /// Full physics simulation (objects, projectiles)
  Dynamic
}

/// Body handle
pub type BodyId {
  BodyId(index: Int)
}

/// Body state snapshot
pub type BodyState {
  BodyState(
    id: BodyId,
    position: Vec3,
    velocity: Vec3,
    angular_velocity: Vec3,
    rotation: Quat,
    active: Bool,
  )
}

/// Raycast hit result
pub type RayHit {
  RayHit(
    body: BodyId,
    fraction: Float,
    point: Vec3,
  )
}

/// Raycast result
pub type RayResult {
  Hit(RayHit)
  Miss
}

/// Contact event type
pub type ContactEventType {
  ContactAdded
  ContactPersisted
  ContactRemoved
}

/// Contact event from physics simulation
pub type ContactEvent {
  ContactEvent(
    event_type: ContactEventType,
    body1: BodyId,
    body2: BodyId,
    normal: Vec3,
    penetration_depth: Float,
    contact_point: Vec3,
  )
}

/// Physics world resource (opaque)
pub type World

/// Opaque Erlang atom type
pub type Atom

// =============================================================================
// NATIVE BINDINGS
// =============================================================================

// World
@external(erlang, "Elixir.Viva.Jolt.Native", "world_new")
pub fn world_new() -> World

@external(erlang, "Elixir.Viva.Jolt.Native", "world_step")
fn native_step(world: World, dt: Float) -> Int

@external(erlang, "Elixir.Viva.Jolt.Native", "world_step_n")
fn native_step_n(world: World, n: Int, dt: Float) -> Int

@external(erlang, "Elixir.Viva.Jolt.Native", "world_optimize")
fn native_optimize(world: World) -> Atom

@external(erlang, "Elixir.Viva.Jolt.Native", "body_count")
fn native_body_count(world: World) -> Int

@external(erlang, "Elixir.Viva.Jolt.Native", "tick")
fn native_tick(world: World) -> Int

// Body creation
@external(erlang, "Elixir.Viva.Jolt.Native", "create_box")
fn native_create_box(
  world: World,
  pos: #(Float, Float, Float),
  half_extents: #(Float, Float, Float),
  motion_type: Atom,
) -> Int

@external(erlang, "Elixir.Viva.Jolt.Native", "create_sphere")
fn native_create_sphere(
  world: World,
  pos: #(Float, Float, Float),
  radius: Float,
  motion_type: Atom,
) -> Int

@external(erlang, "Elixir.Viva.Jolt.Native", "create_capsule")
fn native_create_capsule(
  world: World,
  pos: #(Float, Float, Float),
  half_height: Float,
  radius: Float,
  motion_type: Atom,
) -> Int

@external(erlang, "Elixir.Viva.Jolt.Native", "create_cylinder")
fn native_create_cylinder(
  world: World,
  pos: #(Float, Float, Float),
  half_height: Float,
  radius: Float,
  motion_type: Atom,
) -> Int

// Position & Rotation
@external(erlang, "Elixir.Viva.Jolt.Native", "get_position")
fn native_get_position(world: World, index: Int) -> Result(#(Float, Float, Float), Nil)

@external(erlang, "Elixir.Viva.Jolt.Native", "set_position")
fn native_set_position(world: World, index: Int, pos: #(Float, Float, Float)) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "get_rotation")
fn native_get_rotation(world: World, index: Int) -> Result(#(Float, Float, Float, Float), Nil)

@external(erlang, "Elixir.Viva.Jolt.Native", "set_rotation")
fn native_set_rotation(world: World, index: Int, rot: #(Float, Float, Float, Float)) -> Bool

// Velocity
@external(erlang, "Elixir.Viva.Jolt.Native", "get_velocity")
fn native_get_velocity(world: World, index: Int) -> Result(#(Float, Float, Float), Nil)

@external(erlang, "Elixir.Viva.Jolt.Native", "set_velocity")
fn native_set_velocity(world: World, index: Int, vel: #(Float, Float, Float)) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "get_angular_velocity")
fn native_get_angular_velocity(world: World, index: Int) -> Result(#(Float, Float, Float), Nil)

@external(erlang, "Elixir.Viva.Jolt.Native", "set_angular_velocity")
fn native_set_angular_velocity(world: World, index: Int, vel: #(Float, Float, Float)) -> Bool

// Forces & Impulses
@external(erlang, "Elixir.Viva.Jolt.Native", "add_force")
fn native_add_force(world: World, index: Int, force: #(Float, Float, Float)) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "add_torque")
fn native_add_torque(world: World, index: Int, torque: #(Float, Float, Float)) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "add_impulse")
fn native_add_impulse(world: World, index: Int, impulse: #(Float, Float, Float)) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "add_angular_impulse")
fn native_add_angular_impulse(world: World, index: Int, impulse: #(Float, Float, Float)) -> Bool

// Kinematic movement
@external(erlang, "Elixir.Viva.Jolt.Native", "move_kinematic")
fn native_move_kinematic(
  world: World,
  index: Int,
  target_pos: #(Float, Float, Float),
  target_rot: #(Float, Float, Float, Float),
  dt: Float,
) -> Bool

// Body properties
@external(erlang, "Elixir.Viva.Jolt.Native", "is_active")
fn native_is_active(world: World, index: Int) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "activate_body")
fn native_activate_body(world: World, index: Int) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "deactivate_body")
fn native_deactivate_body(world: World, index: Int) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "set_friction")
fn native_set_friction(world: World, index: Int, friction: Float) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "get_friction")
fn native_get_friction(world: World, index: Int) -> Result(Float, Nil)

@external(erlang, "Elixir.Viva.Jolt.Native", "set_restitution")
fn native_set_restitution(world: World, index: Int, restitution: Float) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "get_restitution")
fn native_get_restitution(world: World, index: Int) -> Result(Float, Nil)

@external(erlang, "Elixir.Viva.Jolt.Native", "set_gravity_factor")
fn native_set_gravity_factor(world: World, index: Int, factor: Float) -> Bool

@external(erlang, "Elixir.Viva.Jolt.Native", "get_gravity_factor")
fn native_get_gravity_factor(world: World, index: Int) -> Result(Float, Nil)

// Raycast - returns either {hit, body_index, fraction, {x, y, z}} or miss atom
@external(erlang, "Elixir.Viva.Jolt.Native", "cast_ray")
fn native_cast_ray(
  world: World,
  origin: #(Float, Float, Float),
  direction: #(Float, Float, Float),
) -> RaycastResult

// Contact events
@external(erlang, "Elixir.Viva.Jolt.Native", "get_contacts")
fn native_get_contacts(world: World) -> List(ContactTuple)

@external(erlang, "Elixir.Viva.Jolt.Native", "has_contacts")
pub fn has_contacts(world: World) -> Bool

// Internal type for raycast result from NIF
type RaycastResult

// Internal type for contact tuple from NIF
// {event_type, body1_idx, body2_idx, {nx, ny, nz}, depth, {px, py, pz}}
type ContactTuple

@external(erlang, "erlang", "binary_to_atom")
fn to_atom(s: String) -> Atom

@external(erlang, "Elixir.Viva.Jolt.Native", "native_check")
pub fn check() -> String

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create Vec3 from components
pub fn vec3(x: Float, y: Float, z: Float) -> Vec3 {
  Vec3(x, y, z)
}

/// Zero vector
pub fn vec3_zero() -> Vec3 {
  Vec3(0.0, 0.0, 0.0)
}

/// Unit vectors
pub fn vec3_up() -> Vec3 { Vec3(0.0, 1.0, 0.0) }
pub fn vec3_down() -> Vec3 { Vec3(0.0, -1.0, 0.0) }
pub fn vec3_forward() -> Vec3 { Vec3(0.0, 0.0, 1.0) }
pub fn vec3_back() -> Vec3 { Vec3(0.0, 0.0, -1.0) }
pub fn vec3_right() -> Vec3 { Vec3(1.0, 0.0, 0.0) }
pub fn vec3_left() -> Vec3 { Vec3(-1.0, 0.0, 0.0) }

/// Create Quat from components
pub fn quat(x: Float, y: Float, z: Float, w: Float) -> Quat {
  Quat(x, y, z, w)
}

/// Identity quaternion (no rotation)
pub fn quat_identity() -> Quat {
  Quat(0.0, 0.0, 0.0, 1.0)
}

// =============================================================================
// WORLD API
// =============================================================================

/// Step physics simulation
pub fn step(world: World, dt: Float) -> Int {
  native_step(world, dt)
}

/// Step physics simulation N times
pub fn step_n(world: World, n: Int, dt: Float) -> Int {
  native_step_n(world, n, dt)
}

/// Optimize broad phase (call after adding static bodies)
pub fn optimize(world: World) -> Nil {
  let _ = native_optimize(world)
  Nil
}

/// Get body count
pub fn body_count(world: World) -> Int {
  native_body_count(world)
}

/// Get tick count
pub fn get_tick(world: World) -> Int {
  native_tick(world)
}

// =============================================================================
// BODY CREATION
// =============================================================================

/// Create a box body
pub fn create_box(
  world: World,
  position: Vec3,
  half_extents: Vec3,
  motion_type: MotionType,
) -> BodyId {
  let Vec3(px, py, pz) = position
  let Vec3(hx, hy, hz) = half_extents
  let mt = motion_type_to_atom(motion_type)
  let index = native_create_box(world, #(px, py, pz), #(hx, hy, hz), mt)
  BodyId(index)
}

/// Create a sphere body
pub fn create_sphere(
  world: World,
  position: Vec3,
  radius: Float,
  motion_type: MotionType,
) -> BodyId {
  let Vec3(px, py, pz) = position
  let mt = motion_type_to_atom(motion_type)
  let index = native_create_sphere(world, #(px, py, pz), radius, mt)
  BodyId(index)
}

/// Create a capsule body (good for characters)
pub fn create_capsule(
  world: World,
  position: Vec3,
  half_height: Float,
  radius: Float,
  motion_type: MotionType,
) -> BodyId {
  let Vec3(px, py, pz) = position
  let mt = motion_type_to_atom(motion_type)
  let index = native_create_capsule(world, #(px, py, pz), half_height, radius, mt)
  BodyId(index)
}

/// Create a cylinder body
pub fn create_cylinder(
  world: World,
  position: Vec3,
  half_height: Float,
  radius: Float,
  motion_type: MotionType,
) -> BodyId {
  let Vec3(px, py, pz) = position
  let mt = motion_type_to_atom(motion_type)
  let index = native_create_cylinder(world, #(px, py, pz), half_height, radius, mt)
  BodyId(index)
}

// =============================================================================
// BODY STATE
// =============================================================================

/// Get complete body state
pub fn get_state(world: World, body: BodyId) -> Result(BodyState, Nil) {
  let BodyId(index) = body
  case native_get_position(world, index) {
    Ok(#(px, py, pz)) -> {
      let vel = case native_get_velocity(world, index) {
        Ok(#(vx, vy, vz)) -> Vec3(vx, vy, vz)
        Error(_) -> vec3_zero()
      }
      let ang_vel = case native_get_angular_velocity(world, index) {
        Ok(#(ax, ay, az)) -> Vec3(ax, ay, az)
        Error(_) -> vec3_zero()
      }
      let rot = case native_get_rotation(world, index) {
        Ok(#(rx, ry, rz, rw)) -> Quat(rx, ry, rz, rw)
        Error(_) -> quat_identity()
      }
      let active = native_is_active(world, index)
      Ok(BodyState(
        id: body,
        position: Vec3(px, py, pz),
        velocity: vel,
        angular_velocity: ang_vel,
        rotation: rot,
        active: active,
      ))
    }
    Error(_) -> Error(Nil)
  }
}

/// Get body position
pub fn get_position(world: World, body: BodyId) -> Result(Vec3, Nil) {
  let BodyId(index) = body
  case native_get_position(world, index) {
    Ok(#(x, y, z)) -> Ok(Vec3(x, y, z))
    Error(_) -> Error(Nil)
  }
}

/// Set body position (teleport)
pub fn set_position(world: World, body: BodyId, position: Vec3) -> Bool {
  let BodyId(index) = body
  let Vec3(x, y, z) = position
  native_set_position(world, index, #(x, y, z))
}

/// Get body rotation
pub fn get_rotation(world: World, body: BodyId) -> Result(Quat, Nil) {
  let BodyId(index) = body
  case native_get_rotation(world, index) {
    Ok(#(x, y, z, w)) -> Ok(Quat(x, y, z, w))
    Error(_) -> Error(Nil)
  }
}

/// Set body rotation
pub fn set_rotation(world: World, body: BodyId, rotation: Quat) -> Bool {
  let BodyId(index) = body
  let Quat(x, y, z, w) = rotation
  native_set_rotation(world, index, #(x, y, z, w))
}

// =============================================================================
// VELOCITY
// =============================================================================

/// Get body linear velocity
pub fn get_velocity(world: World, body: BodyId) -> Result(Vec3, Nil) {
  let BodyId(index) = body
  case native_get_velocity(world, index) {
    Ok(#(x, y, z)) -> Ok(Vec3(x, y, z))
    Error(_) -> Error(Nil)
  }
}

/// Set body linear velocity
pub fn set_velocity(world: World, body: BodyId, velocity: Vec3) -> Bool {
  let BodyId(index) = body
  let Vec3(x, y, z) = velocity
  native_set_velocity(world, index, #(x, y, z))
}

/// Get body angular velocity
pub fn get_angular_velocity(world: World, body: BodyId) -> Result(Vec3, Nil) {
  let BodyId(index) = body
  case native_get_angular_velocity(world, index) {
    Ok(#(x, y, z)) -> Ok(Vec3(x, y, z))
    Error(_) -> Error(Nil)
  }
}

/// Set body angular velocity
pub fn set_angular_velocity(world: World, body: BodyId, velocity: Vec3) -> Bool {
  let BodyId(index) = body
  let Vec3(x, y, z) = velocity
  native_set_angular_velocity(world, index, #(x, y, z))
}

// =============================================================================
// FORCES & IMPULSES
// =============================================================================

/// Add continuous force to body (N)
pub fn add_force(world: World, body: BodyId, force: Vec3) -> Bool {
  let BodyId(index) = body
  let Vec3(x, y, z) = force
  native_add_force(world, index, #(x, y, z))
}

/// Add continuous torque to body (Nm)
pub fn add_torque(world: World, body: BodyId, torque: Vec3) -> Bool {
  let BodyId(index) = body
  let Vec3(x, y, z) = torque
  native_add_torque(world, index, #(x, y, z))
}

/// Add instant impulse to body (Ns)
pub fn add_impulse(world: World, body: BodyId, impulse: Vec3) -> Bool {
  let BodyId(index) = body
  let Vec3(x, y, z) = impulse
  native_add_impulse(world, index, #(x, y, z))
}

/// Add instant angular impulse to body (Nms)
pub fn add_angular_impulse(world: World, body: BodyId, impulse: Vec3) -> Bool {
  let BodyId(index) = body
  let Vec3(x, y, z) = impulse
  native_add_angular_impulse(world, index, #(x, y, z))
}

// =============================================================================
// KINEMATIC MOVEMENT (Character Controller)
// =============================================================================

/// Move kinematic body towards target position/rotation over dt seconds
/// The physics engine calculates the required velocity to reach the target
/// This is perfect for character controllers that need collision response
pub fn move_kinematic(
  world: World,
  body: BodyId,
  target_position: Vec3,
  target_rotation: Quat,
  dt: Float,
) -> Bool {
  let BodyId(index) = body
  let Vec3(px, py, pz) = target_position
  let Quat(rx, ry, rz, rw) = target_rotation
  native_move_kinematic(world, index, #(px, py, pz), #(rx, ry, rz, rw), dt)
}

/// Simplified kinematic move (keeps current rotation)
pub fn move_kinematic_to(world: World, body: BodyId, target: Vec3, dt: Float) -> Bool {
  move_kinematic(world, body, target, quat_identity(), dt)
}

// =============================================================================
// BODY PROPERTIES
// =============================================================================

/// Check if body is active (simulating)
pub fn is_active(world: World, body: BodyId) -> Bool {
  let BodyId(index) = body
  native_is_active(world, index)
}

/// Wake up sleeping body
pub fn activate(world: World, body: BodyId) -> Bool {
  let BodyId(index) = body
  native_activate_body(world, index)
}

/// Put body to sleep
pub fn deactivate(world: World, body: BodyId) -> Bool {
  let BodyId(index) = body
  native_deactivate_body(world, index)
}

/// Set friction coefficient (0-1)
pub fn set_friction(world: World, body: BodyId, friction: Float) -> Bool {
  let BodyId(index) = body
  native_set_friction(world, index, friction)
}

/// Get friction coefficient
pub fn get_friction(world: World, body: BodyId) -> Result(Float, Nil) {
  let BodyId(index) = body
  native_get_friction(world, index)
}

/// Set restitution/bounciness (0-1)
pub fn set_restitution(world: World, body: BodyId, restitution: Float) -> Bool {
  let BodyId(index) = body
  native_set_restitution(world, index, restitution)
}

/// Get restitution/bounciness
pub fn get_restitution(world: World, body: BodyId) -> Result(Float, Nil) {
  let BodyId(index) = body
  native_get_restitution(world, index)
}

/// Set gravity factor (0=no gravity, 1=normal, 2=double, etc)
pub fn set_gravity_factor(world: World, body: BodyId, factor: Float) -> Bool {
  let BodyId(index) = body
  native_set_gravity_factor(world, index, factor)
}

/// Get gravity factor
pub fn get_gravity_factor(world: World, body: BodyId) -> Result(Float, Nil) {
  let BodyId(index) = body
  native_get_gravity_factor(world, index)
}

// =============================================================================
// RAYCAST
// =============================================================================

/// Cast a ray and find the first body hit
/// origin: starting point of ray
/// direction: direction and length of ray (not normalized!)
/// Returns Hit with body, fraction (0-1), and hit point, or Miss
pub fn cast_ray(world: World, origin: Vec3, direction: Vec3) -> RayResult {
  let Vec3(ox, oy, oz) = origin
  let Vec3(dx, dy, dz) = direction
  decode_raycast_result(native_cast_ray(world, #(ox, oy, oz), #(dx, dy, dz)))
}

/// Cast ray downward from position (ground check)
pub fn cast_ray_down(world: World, from: Vec3, distance: Float) -> RayResult {
  cast_ray(world, from, Vec3(0.0, -1.0 *. distance, 0.0))
}

// =============================================================================
// CONTACT EVENTS
// =============================================================================

/// Get all contact events that occurred since last call
/// Events are cleared after reading, so call this once per frame
/// Returns list of ContactEvent with type (Added/Persisted/Removed),
/// both body IDs, collision normal, penetration depth, and contact point
pub fn get_contacts(world: World) -> List(ContactEvent) {
  native_get_contacts(world)
  |> list_map(decode_contact_tuple)
}

fn list_map(list: List(a), f: fn(a) -> b) -> List(b) {
  case list {
    [] -> []
    [head, ..tail] -> [f(head), ..list_map(tail, f)]
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn motion_type_to_atom(mt: MotionType) -> Atom {
  case mt {
    Static -> to_atom("static_body")
    Kinematic -> to_atom("kinematic")
    Dynamic -> to_atom("dynamic")
  }
}

// Decode NIF raycast result
// NIF returns either: {hit, body_index, fraction, {px, py, pz}} or atom miss
@external(erlang, "erlang", "is_atom")
fn is_atom(term: RaycastResult) -> Bool

fn decode_raycast_result(result: RaycastResult) -> RayResult {
  case is_atom(result) {
    True -> Miss
    False -> decode_hit_tuple(result)
  }
}

@external(erlang, "erlang", "element")
fn element(n: Int, tuple: a) -> b

fn decode_hit_tuple(result: RaycastResult) -> RayResult {
  let body_index: Int = element(2, result)
  let fraction: Float = element(3, result)
  let point_tuple: #(Float, Float, Float) = element(4, result)
  let #(px, py, pz) = point_tuple
  Hit(RayHit(
    body: BodyId(body_index),
    fraction: fraction,
    point: Vec3(px, py, pz),
  ))
}

// Decode contact tuple from NIF
// Format: {event_type_atom, body1_idx, body2_idx, {nx, ny, nz}, depth, {px, py, pz}}
fn decode_contact_tuple(tuple: ContactTuple) -> ContactEvent {
  let event_atom: Atom = element(1, tuple)
  let body1_idx: Int = element(2, tuple)
  let body2_idx: Int = element(3, tuple)
  let normal_tuple: #(Float, Float, Float) = element(4, tuple)
  let depth: Float = element(5, tuple)
  let point_tuple: #(Float, Float, Float) = element(6, tuple)
  let #(nx, ny, nz) = normal_tuple
  let #(px, py, pz) = point_tuple
  let event_type = decode_contact_event_type(event_atom)
  ContactEvent(
    event_type: event_type,
    body1: BodyId(body1_idx),
    body2: BodyId(body2_idx),
    normal: Vec3(nx, ny, nz),
    penetration_depth: depth,
    contact_point: Vec3(px, py, pz),
  )
}

@external(erlang, "erlang", "atom_to_binary")
fn atom_to_string(atom: Atom) -> String

fn decode_contact_event_type(atom: Atom) -> ContactEventType {
  case atom_to_string(atom) {
    "contact_added" -> ContactAdded
    "contact_persisted" -> ContactPersisted
    "contact_removed" -> ContactRemoved
    _ -> ContactAdded
  }
}
