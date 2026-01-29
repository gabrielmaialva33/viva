//// JoltPhysics Complete Test
//// Demonstrates: physics, raycast, kinematic character control, CONTACT EVENTS

import gleam/io
import viva/jolt

pub fn main() {
  io.println("=== JoltPhysics Complete Test v0.2.0 ===")
  io.println("Features: Physics, Raycast, Kinematic, ContactListener")
  io.println("")

  // Check NIF loaded
  io.println("NIF: " <> jolt.check())

  // Create physics world
  io.println("")
  io.println("--- World Setup ---")
  let world = jolt.world_new()

  // Create floor (static)
  let floor = jolt.create_box(
    world,
    jolt.vec3(0.0, -1.0, 0.0),
    jolt.vec3(50.0, 1.0, 50.0),
    jolt.Static,
  )
  io.println("Floor created: body " <> int_to_string(floor.index))

  // Create falling sphere (dynamic)
  let sphere = jolt.create_sphere(
    world,
    jolt.vec3(0.0, 10.0, 0.0),
    0.5,
    jolt.Dynamic,
  )
  io.println("Sphere created: body " <> int_to_string(sphere.index))

  // Create kinematic character (for character controller demo)
  let character = jolt.create_capsule(
    world,
    jolt.vec3(5.0, 2.0, 0.0),
    0.9,  // half height
    0.4,  // radius
    jolt.Kinematic,
  )
  io.println("Character (kinematic capsule): body " <> int_to_string(character.index))

  // Create a wall for raycast test
  let wall = jolt.create_box(
    world,
    jolt.vec3(10.0, 5.0, 0.0),
    jolt.vec3(1.0, 5.0, 5.0),
    jolt.Static,
  )
  io.println("Wall created: body " <> int_to_string(wall.index))

  // Optimize broad phase
  jolt.optimize(world)
  io.println("Broad phase optimized")

  // Initial state
  io.println("")
  io.println("--- Initial State ---")
  print_body_state(world, "Sphere", sphere)

  // Test raycast - ray from origin towards wall
  io.println("")
  io.println("--- Raycast Test ---")
  let ray_origin = jolt.vec3(0.0, 5.0, 0.0)
  let ray_direction = jolt.vec3(20.0, 0.0, 0.0)  // 20 units to the right
  io.println("Casting ray from (0,5,0) direction (20,0,0)...")
  case jolt.cast_ray(world, ray_origin, ray_direction) {
    jolt.Hit(hit) -> {
      io.println("  HIT!")
      io.println("  Body: " <> int_to_string(hit.body.index))
      io.println("  Fraction: " <> float_to_string(hit.fraction))
      io.println("  Point: (" <> float_to_string(hit.point.x) <> ", " <>
                              float_to_string(hit.point.y) <> ", " <>
                              float_to_string(hit.point.z) <> ")")
    }
    jolt.Miss -> io.println("  MISS")
  }

  // Test raycast downward (ground check)
  io.println("")
  io.println("Casting ray down from sphere position...")
  case jolt.get_position(world, sphere) {
    Ok(pos) -> {
      case jolt.cast_ray_down(world, pos, 20.0) {
        jolt.Hit(hit) -> {
          io.println("  Ground found at fraction: " <> float_to_string(hit.fraction))
          io.println("  Ground point Y: " <> float_to_string(hit.point.y))
        }
        jolt.Miss -> io.println("  No ground found")
      }
    }
    Error(_) -> io.println("  Could not get sphere position")
  }

  // Simulate physics with contact detection
  io.println("")
  io.println("--- Physics Simulation (120 steps) with Contact Detection ---")
  io.println("Sphere falling from y=10 to floor at y=0...")
  simulate_with_contacts(world, 120, 0)
  io.println("Simulation complete")

  // After simulation
  io.println("")
  io.println("--- After Simulation ---")
  print_body_state(world, "Sphere", sphere)

  // Test kinematic character movement
  io.println("")
  io.println("--- Kinematic Character Movement ---")
  io.println("Moving character from (5,2,0) to (5,2,3)...")
  let target = jolt.vec3(5.0, 2.0, 3.0)
  let _ = jolt.move_kinematic_to(world, character, target, 1.0 /. 60.0)
  let _ = jolt.step(world, 1.0 /. 60.0)
  print_body_state(world, "Character", character)

  // Test body properties
  io.println("")
  io.println("--- Body Properties ---")
  case jolt.get_friction(world, sphere) {
    Ok(f) -> io.println("Sphere friction: " <> float_to_string(f))
    Error(_) -> Nil
  }
  case jolt.get_restitution(world, sphere) {
    Ok(r) -> io.println("Sphere restitution: " <> float_to_string(r))
    Error(_) -> Nil
  }
  case jolt.get_gravity_factor(world, sphere) {
    Ok(g) -> io.println("Sphere gravity factor: " <> float_to_string(g))
    Error(_) -> Nil
  }

  // Test gravity modification
  io.println("")
  io.println("--- Gravity Test ---")
  io.println("Setting sphere gravity factor to 0 (weightless)...")
  let _ = jolt.set_gravity_factor(world, sphere, 0.0)
  let _ = jolt.set_position(world, sphere, jolt.vec3(0.0, 5.0, 0.0))
  let _ = jolt.set_velocity(world, sphere, jolt.vec3_zero())
  let _ = jolt.activate(world, sphere)
  simulate_loop(world, 30)
  case jolt.get_position(world, sphere) {
    Ok(pos) -> io.println("Sphere Y after 30 steps (no gravity): " <> float_to_string(pos.y))
    Error(_) -> Nil
  }

  // Summary
  io.println("")
  io.println("--- Summary ---")
  io.println("Tick: " <> int_to_string(jolt.get_tick(world)))
  io.println("Bodies: " <> int_to_string(jolt.body_count(world)))

  io.println("")
  io.println("=== Test Complete ===")
}

fn simulate_loop(world, remaining) {
  case remaining {
    0 -> Nil
    _ -> {
      let _ = jolt.step(world, 1.0 /. 60.0)
      simulate_loop(world, remaining - 1)
    }
  }
}

fn simulate_with_contacts(world, remaining, total_contacts) {
  case remaining {
    0 -> io.println("Total contact events: " <> int_to_string(total_contacts))
    _ -> {
      let _ = jolt.step(world, 1.0 /. 60.0)
      let contacts = jolt.get_contacts(world)
      let contact_count = list_length(contacts)
      case contact_count > 0 {
        True -> {
          io.println("Frame " <> int_to_string(60 - remaining + 1) <> ": " <> int_to_string(contact_count) <> " contact(s)")
          print_contacts(contacts)
        }
        False -> Nil
      }
      simulate_with_contacts(world, remaining - 1, total_contacts + contact_count)
    }
  }
}

fn print_contacts(contacts: List(jolt.ContactEvent)) {
  case contacts {
    [] -> Nil
    [contact, ..rest] -> {
      let type_str = case contact.event_type {
        jolt.ContactAdded -> "ADDED"
        jolt.ContactPersisted -> "PERSISTED"
        jolt.ContactRemoved -> "REMOVED"
      }
      io.println("  " <> type_str <> " body " <> int_to_string(contact.body1.index) <> " <-> body " <> int_to_string(contact.body2.index) <>
        " depth=" <> float_to_string(contact.penetration_depth))
      print_contacts(rest)
    }
  }
}

fn list_length(list) {
  case list {
    [] -> 0
    [_, ..rest] -> 1 + list_length(rest)
  }
}

fn print_body_state(world, name, body) {
  case jolt.get_state(world, body) {
    Ok(state) -> {
      io.println(name <> " state:")
      io.println("  Position: (" <> float_to_string(state.position.x) <> ", " <>
                                  float_to_string(state.position.y) <> ", " <>
                                  float_to_string(state.position.z) <> ")")
      io.println("  Velocity: (" <> float_to_string(state.velocity.x) <> ", " <>
                                  float_to_string(state.velocity.y) <> ", " <>
                                  float_to_string(state.velocity.z) <> ")")
      io.println("  Active: " <> bool_to_string(state.active))
    }
    Error(_) -> io.println(name <> ": Error getting state")
  }
}

@external(erlang, "erlang", "integer_to_binary")
fn int_to_string(i: Int) -> String

@external(erlang, "erlang", "float_to_binary")
fn float_to_string(f: Float) -> String

fn bool_to_string(b: Bool) -> String {
  case b {
    True -> "true"
    False -> "false"
  }
}
