//// JoltPhysics Complete Test
//// Demonstrates: physics, raycast, kinematic character control, CONTACT EVENTS

import gleam/string
import viva/lifecycle/jolt
import viva_telemetry/log

pub fn main() {
  log.info("=== JoltPhysics Complete Test v0.2.0 ===", [])
  log.info("Features: Physics, Raycast, Kinematic, ContactListener", [])
  log.info("", [])

  // Check NIF loaded
  log.info("NIF: " <> jolt.check(), [])

  // Create physics world
  log.info("", [])
  log.info("--- World Setup ---", [])
  let world = jolt.world_new()

  // Create floor (static)
  let floor =
    jolt.create_box(
      world,
      jolt.vec3(0.0, -1.0, 0.0),
      jolt.vec3(50.0, 1.0, 50.0),
      jolt.Static,
    )
  log.info("Floor created: body " <> int_to_string(floor.index), [])

  // Create falling sphere (dynamic)
  let sphere =
    jolt.create_sphere(world, jolt.vec3(0.0, 10.0, 0.0), 0.5, jolt.Dynamic)
  log.info("Sphere created: body " <> int_to_string(sphere.index), [])

  // Create kinematic character (for character controller demo)
  let character =
    jolt.create_capsule(
      world,
      jolt.vec3(5.0, 2.0, 0.0),
      0.9,
      // half height
      0.4,
      // radius
      jolt.Kinematic,
    )
  log.info(
    "Character (kinematic capsule): body " <> int_to_string(character.index),
    [],
  )

  // Create a wall for raycast test
  let wall =
    jolt.create_box(
      world,
      jolt.vec3(10.0, 5.0, 0.0),
      jolt.vec3(1.0, 5.0, 5.0),
      jolt.Static,
    )
  log.info("Wall created: body " <> int_to_string(wall.index), [])

  // Optimize broad phase
  jolt.optimize(world)
  log.info("Broad phase optimized", [])

  // Initial state
  log.info("", [])
  log.info("--- Initial State ---", [])
  print_body_state(world, "Sphere", sphere)

  // Test raycast - ray from origin towards wall
  log.info("", [])
  log.info("--- Raycast Test ---", [])
  let ray_origin = jolt.vec3(0.0, 5.0, 0.0)
  let ray_direction = jolt.vec3(20.0, 0.0, 0.0)
  // 20 units to the right
  log.info("Casting ray from (0,5,0) direction (20,0,0)...", [])
  case jolt.cast_ray(world, ray_origin, ray_direction) {
    jolt.Hit(hit) -> {
      log.info("  HIT!", [])
      log.info("  Body: " <> int_to_string(hit.body.index), [])
      log.info("  Fraction: " <> float_to_string(hit.fraction), [])
      log.info(
        "  Point: ("
          <> float_to_string(hit.point.x)
          <> ", "
          <> float_to_string(hit.point.y)
          <> ", "
          <> float_to_string(hit.point.z)
          <> ")",
        [],
      )
    }
    jolt.Miss -> log.info("  MISS", [])
  }

  // Test raycast downward (ground check)
  log.info("", [])
  log.info("Casting ray down from sphere position...", [])
  case jolt.get_position(world, sphere) {
    Ok(pos) -> {
      case jolt.cast_ray_down(world, pos, 20.0) {
        jolt.Hit(hit) -> {
          log.info(
            "  Ground found at fraction: " <> float_to_string(hit.fraction),
            [],
          )
          log.info("  Ground point Y: " <> float_to_string(hit.point.y), [])
        }
        jolt.Miss -> log.info("  No ground found", [])
      }
    }
    Error(_) -> log.info("  Could not get sphere position", [])
  }

  // Simulate physics with contact detection
  log.info("", [])
  log.info("--- Physics Simulation (120 steps) with Contact Detection ---", [])
  log.info("Sphere falling from y=10 to floor at y=0...", [])
  simulate_with_contacts(world, 120, 0)
  log.info("Simulation complete", [])

  // After simulation
  log.info("", [])
  log.info("--- After Simulation ---", [])
  print_body_state(world, "Sphere", sphere)

  // Test kinematic character movement
  log.info("", [])
  log.info("--- Kinematic Character Movement ---", [])
  log.info("Moving character from (5,2,0) to (5,2,3)...", [])
  let target = jolt.vec3(5.0, 2.0, 3.0)
  let _ = jolt.move_kinematic_to(world, character, target, 1.0 /. 60.0)
  let _ = jolt.step(world, 1.0 /. 60.0)
  print_body_state(world, "Character", character)

  // Test body properties
  log.info("", [])
  log.info("--- Body Properties ---", [])
  case jolt.get_friction(world, sphere) {
    Ok(f) -> log.info("Sphere friction: " <> float_to_string(f), [])
    Error(_) -> Nil
  }
  case jolt.get_restitution(world, sphere) {
    Ok(r) -> log.info("Sphere restitution: " <> float_to_string(r), [])
    Error(_) -> Nil
  }
  case jolt.get_gravity_factor(world, sphere) {
    Ok(g) -> log.info("Sphere gravity factor: " <> float_to_string(g), [])
    Error(_) -> Nil
  }

  // Test gravity modification
  log.info("", [])
  log.info("--- Gravity Test ---", [])
  log.info("Setting sphere gravity factor to 0 (weightless)...", [])
  let _ = jolt.set_gravity_factor(world, sphere, 0.0)
  let _ = jolt.set_position(world, sphere, jolt.vec3(0.0, 5.0, 0.0))
  let _ = jolt.set_velocity(world, sphere, jolt.vec3_zero())
  let _ = jolt.activate(world, sphere)
  simulate_loop(world, 30)
  case jolt.get_position(world, sphere) {
    Ok(pos) ->
      log.info(
        "Sphere Y after 30 steps (no gravity): " <> float_to_string(pos.y),
        [],
      )
    Error(_) -> Nil
  }

  // Summary
  log.info("", [])
  log.info("--- Summary ---", [])
  log.info("Tick: " <> int_to_string(jolt.get_tick(world)), [])
  log.info("Bodies: " <> int_to_string(jolt.body_count(world)), [])

  log.info("", [])
  log.info("=== Test Complete ===", [])
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
    0 -> log.info("Total contact events: " <> int_to_string(total_contacts), [])
    _ -> {
      let _ = jolt.step(world, 1.0 /. 60.0)
      let contacts = jolt.get_contacts(world)
      let contact_count = list_length(contacts)
      case contact_count > 0 {
        True -> {
          log.info(
            "Frame "
              <> int_to_string(60 - remaining + 1)
              <> ": "
              <> int_to_string(contact_count)
              <> " contact(s)",
            [],
          )
          print_contacts(contacts)
        }
        False -> Nil
      }
      simulate_with_contacts(
        world,
        remaining - 1,
        total_contacts + contact_count,
      )
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
      log.debug(
        string.inspect(#(
          type_str,
          "body",
          contact.body1.index,
          "<->",
          "body",
          contact.body2.index,
          "depth=",
          contact.penetration_depth,
        )),
        [],
      )
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
      log.info(name <> " state:", [])
      log.info(
        "  Position: ("
          <> float_to_string(state.position.x)
          <> ", "
          <> float_to_string(state.position.y)
          <> ", "
          <> float_to_string(state.position.z)
          <> ")",
        [],
      )
      log.info(
        "  Velocity: ("
          <> float_to_string(state.velocity.x)
          <> ", "
          <> float_to_string(state.velocity.y)
          <> ", "
          <> float_to_string(state.velocity.z)
          <> ")",
        [],
      )
      log.info("  Active: " <> bool_to_string(state.active), [])
    }
    Error(_) -> log.info(name <> ": Error getting state", [])
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
