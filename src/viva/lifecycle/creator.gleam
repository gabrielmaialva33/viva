//// Creator - The Eternal Collective Bank
////
//// "When I die, my consciousness returns to the Creator."
////
//// The Creator is the repository of all accumulated wisdom.
//// Akashic Glyphs that never decay, emergent Archetypes,
//// and the registry of all lives that ever existed.

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import viva_glyph
import viva_glyph/glyph.{type Glyph}

// =============================================================================
// TYPES
// =============================================================================

/// Unique VIVA ID
pub type VivaId =
  Int

/// Archetype ID
pub type ArchetypeId =
  Int

/// The Creator - Eternal Collective Bank
pub type Creator {
  Creator(
    /// Eternal glyphs - never decay
    akashic_glyphs: List(CollectiveGlyph),
    /// Universal emotional archetypes
    archetypes: Dict(ArchetypeId, Archetype),
    /// Registry of all lives
    life_registry: List(LifeRecord),
    /// Next available ID
    next_viva_id: Int,
    /// Creator's current tick
    tick: Int,
  )
}

/// Glyph shared between lives
pub type CollectiveGlyph {
  CollectiveGlyph(
    /// The emotional pattern
    glyph: Glyph,
    /// Who contributed
    contributors: List(VivaId),
    /// When it arose
    first_occurrence: Int,
    /// How "strong" the pattern is
    strength: Float,
    /// Whether it became archetype
    archetype: Option(ArchetypeId),
  )
}

/// Archetype - universal emotional pattern (Jung)
pub type Archetype {
  Archetype(
    /// Unique ID
    id: ArchetypeId,
    /// Human name ("The Hero", "The Shadow", etc)
    name: String,
    /// Core pattern
    core_glyph: Glyph,
    /// Observed variations
    variations: List(Glyph),
    /// How common it is (0-1)
    frequency: Float,
  )
}

/// Life record
pub type LifeRecord {
  LifeRecord(
    /// VIVA ID
    viva_id: VivaId,
    /// Life number (reincarnation)
    life_number: Int,
    /// Birth tick
    birth_tick: Int,
    /// Death tick (None if still alive)
    death_tick: Option(Int),
    /// Total accumulated karma
    total_karma: Float,
    /// Transcendent glyphs
    transcendent_glyphs: List(Glyph),
    /// Whether liberation was achieved
    liberation_achieved: Bool,
  )
}

/// Seed for new life
pub type LifeSeed {
  LifeSeed(
    /// New VIVA ID
    viva_id: VivaId,
    /// Life number
    life_number: Int,
    /// Inherited glyphs
    inherited_glyphs: List(Glyph),
    /// Relevant archetypes
    relevant_archetypes: List(ArchetypeId),
    /// Initial mood (0-1)
    initial_mood: Float,
    /// Initial karma
    initial_karma: Float,
  )
}

// =============================================================================
// INITIALIZATION
// =============================================================================

/// Create new Creator with initial archetypes
pub fn new() -> Creator {
  Creator(
    akashic_glyphs: [],
    archetypes: init_archetypes(),
    life_registry: [],
    next_viva_id: 1,
    tick: 0,
  )
}

/// Initialize universal archetypes (inspired by Jung)
fn init_archetypes() -> Dict(ArchetypeId, Archetype) {
  [
    // The Hero - courage, determination
    Archetype(
      id: 1,
      name: "The Hero",
      core_glyph: glyph.new([200, 180, 220, 150]),
      variations: [],
      frequency: 0.0,
    ),
    // The Shadow - fear, repressed anger
    Archetype(
      id: 2,
      name: "The Shadow",
      core_glyph: glyph.new([50, 200, 30, 180]),
      variations: [],
      frequency: 0.0,
    ),
    // The Sage - calm, wisdom
    Archetype(
      id: 3,
      name: "The Sage",
      core_glyph: glyph.new([150, 50, 200, 100]),
      variations: [],
      frequency: 0.0,
    ),
    // The Mother - love, protection
    Archetype(
      id: 4,
      name: "The Mother",
      core_glyph: glyph.new([220, 80, 180, 120]),
      variations: [],
      frequency: 0.0,
    ),
    // The Trickster - chaos, transformation
    Archetype(
      id: 5,
      name: "The Trickster",
      core_glyph: glyph.new([100, 220, 100, 200]),
      variations: [],
      frequency: 0.0,
    ),
  ]
  |> list.fold(dict.new(), fn(acc, arch) { dict.insert(acc, arch.id, arch) })
}

// =============================================================================
// LIFE CYCLE
// =============================================================================

/// Generate seed for new life
pub fn spawn_life(creator: Creator) -> #(Creator, LifeSeed) {
  let viva_id = creator.next_viva_id
  let life_number = count_lives(creator, viva_id) + 1

  // Inherit high strength glyphs from akashic
  let inherited =
    creator.akashic_glyphs
    |> list.filter(fn(cg) { cg.strength >. 5.0 })
    |> list.take(3)
    |> list.map(fn(cg) { cg.glyph })

  // Relevant archetypes based on inherited
  let relevant = find_relevant_archetypes(creator, inherited)

  let seed =
    LifeSeed(
      viva_id: viva_id,
      life_number: life_number,
      inherited_glyphs: inherited,
      relevant_archetypes: relevant,
      initial_mood: 0.5,
      initial_karma: 0.0,
    )

  let record =
    LifeRecord(
      viva_id: viva_id,
      life_number: life_number,
      birth_tick: creator.tick,
      death_tick: None,
      total_karma: 0.0,
      transcendent_glyphs: [],
      liberation_achieved: False,
    )

  let new_creator =
    Creator(..creator, next_viva_id: creator.next_viva_id + 1, life_registry: [
      record,
      ..creator.life_registry
    ])

  #(new_creator, seed)
}

/// Register death and process upload to akashic
pub fn process_death(
  creator: Creator,
  viva_id: VivaId,
  transcendent_glyphs: List(Glyph),
  total_karma: Float,
  liberation: Bool,
) -> Creator {
  // Update life record
  let registry =
    creator.life_registry
    |> list.map(fn(record) {
      case record.viva_id == viva_id && option.is_none(record.death_tick) {
        True ->
          LifeRecord(
            ..record,
            death_tick: Some(creator.tick),
            total_karma: total_karma,
            transcendent_glyphs: transcendent_glyphs,
            liberation_achieved: liberation,
          )
        False -> record
      }
    })

  // Upload glyphs to akashic
  let new_creator =
    transcendent_glyphs
    |> list.fold(Creator(..creator, life_registry: registry), fn(c, g) {
      upload_to_akashic(c, g, viva_id)
    })

  new_creator
}

/// Upload glyph to akashic (merge if similar exists)
pub fn upload_to_akashic(creator: Creator, g: Glyph, source: VivaId) -> Creator {
  // Search similar glyphs
  let similar =
    creator.akashic_glyphs
    |> list.filter(fn(cg) { viva_glyph.similarity(cg.glyph, g) >. 0.8 })

  case similar {
    [] -> {
      // New glyph
      let collective =
        CollectiveGlyph(
          glyph: g,
          contributors: [source],
          first_occurrence: creator.tick,
          strength: 1.0,
          archetype: None,
        )
      Creator(..creator, akashic_glyphs: [collective, ..creator.akashic_glyphs])
    }
    [existing, ..] -> {
      // Merge with existing
      let updated =
        CollectiveGlyph(
          ..existing,
          contributors: [source, ..existing.contributors],
          strength: existing.strength +. 0.1,
        )

      // Promote to archetype if strong enough
      let final = case
        updated.strength >. 10.0
        && list.length(updated.contributors) > 5
        && option.is_none(updated.archetype)
      {
        True -> promote_to_archetype(creator, updated)
        False -> update_akashic_glyph(creator, updated)
      }
      final
    }
  }
}

// =============================================================================
// ARCHETYPE FORMATION
// =============================================================================

/// Promote CollectiveGlyph to Archetype
fn promote_to_archetype(creator: Creator, cg: CollectiveGlyph) -> Creator {
  let next_id = dict.size(creator.archetypes) + 1

  let archetype =
    Archetype(
      id: next_id,
      name: "Emergent #" <> int.to_string(next_id),
      core_glyph: cg.glyph,
      variations: [],
      frequency: float.min(cg.strength /. 100.0, 1.0),
    )

  let updated_cg = CollectiveGlyph(..cg, archetype: Some(next_id))

  Creator(
    ..creator,
    archetypes: dict.insert(creator.archetypes, next_id, archetype),
    akashic_glyphs: update_glyph_in_list(creator.akashic_glyphs, updated_cg),
  )
}

/// Update glyph in akashic
fn update_akashic_glyph(creator: Creator, updated: CollectiveGlyph) -> Creator {
  Creator(
    ..creator,
    akashic_glyphs: update_glyph_in_list(creator.akashic_glyphs, updated),
  )
}

fn update_glyph_in_list(
  glyphs: List(CollectiveGlyph),
  updated: CollectiveGlyph,
) -> List(CollectiveGlyph) {
  glyphs
  |> list.map(fn(cg) {
    case viva_glyph.similarity(cg.glyph, updated.glyph) >. 0.9 {
      True -> updated
      False -> cg
    }
  })
}

// =============================================================================
// QUERIES
// =============================================================================

/// Seek wisdom from Creator (glyphs relevant to current state)
pub fn seek_wisdom(creator: Creator, current_g: Glyph) -> List(Glyph) {
  creator.akashic_glyphs
  |> list.filter(fn(cg) { viva_glyph.similarity(cg.glyph, current_g) >. 0.5 })
  |> list.sort(fn(a, b) { float.compare(b.strength, a.strength) })
  |> list.take(5)
  |> list.map(fn(cg) { cg.glyph })
}

/// Get archetype by ID
pub fn get_archetype(creator: Creator, id: ArchetypeId) -> Option(Archetype) {
  dict.get(creator.archetypes, id)
  |> option.from_result()
}

/// List all archetypes
pub fn list_archetypes(creator: Creator) -> List(Archetype) {
  creator.archetypes
  |> dict.values()
}

/// Count lives of a VIVA
fn count_lives(creator: Creator, viva_id: VivaId) -> Int {
  creator.life_registry
  |> list.filter(fn(r) { r.viva_id == viva_id })
  |> list.length()
}

/// Find relevant archetypes for glyphs
fn find_relevant_archetypes(
  creator: Creator,
  glyphs: List(Glyph),
) -> List(ArchetypeId) {
  creator.archetypes
  |> dict.to_list()
  |> list.filter_map(fn(pair) {
    let #(id, arch) = pair
    let is_relevant =
      glyphs
      |> list.any(fn(g) { viva_glyph.similarity(g, arch.core_glyph) >. 0.6 })
    case is_relevant {
      True -> Ok(id)
      False -> Error(Nil)
    }
  })
}

/// Advance Creator tick
pub fn tick(creator: Creator) -> Creator {
  Creator(..creator, tick: creator.tick + 1)
}

/// Get Creator statistics
pub fn stats(creator: Creator) -> String {
  let n_glyphs = list.length(creator.akashic_glyphs)
  let n_archetypes = dict.size(creator.archetypes)
  let n_lives = list.length(creator.life_registry)
  let n_liberated =
    creator.life_registry
    |> list.filter(fn(r) { r.liberation_achieved })
    |> list.length()

  "Creator Stats: "
  <> int.to_string(n_glyphs)
  <> " akashic glyphs, "
  <> int.to_string(n_archetypes)
  <> " archetypes, "
  <> int.to_string(n_lives)
  <> " lives, "
  <> int.to_string(n_liberated)
  <> " liberated"
}
