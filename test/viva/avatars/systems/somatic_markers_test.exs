defmodule Viva.Avatars.Systems.SomaticMarkersTest do
  use ExUnit.Case, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.SomaticMarkersState
  alias Viva.Avatars.Systems.SomaticMarkers

  describe "recall/2" do
    test "returns neutral bias when no markers exist" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "conversation_partner"}

      {updated_somatic, bias} = SomaticMarkers.recall(somatic, stimulus)

      assert bias.bias == 0.0
      assert bias.signal == nil
      assert bias.markers_activated == 0
      assert updated_somatic == somatic
    end

    test "activates matching social marker" do
      marker = %{
        valence: 0.8,
        strength: 0.7,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"conversation_partner" => marker}
      }

      stimulus = %{type: :social, source: "conversation_partner"}

      {updated_somatic, bias} = SomaticMarkers.recall(somatic, stimulus)

      assert bias.bias > 0
      assert bias.markers_activated == 1
      assert updated_somatic.current_bias > 0
      assert updated_somatic.body_signal != nil
    end

    test "activates matching activity marker" do
      marker = %{
        valence: -0.6,
        strength: 0.8,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | activity_markers: %{social: marker}
      }

      stimulus = %{type: :social, source: "unknown"}

      {updated_somatic, bias} = SomaticMarkers.recall(somatic, stimulus)

      assert bias.bias < 0
      assert bias.markers_activated == 1
      assert updated_somatic.current_bias < 0
    end

    test "activates matching context marker" do
      marker = %{
        valence: 0.5,
        strength: 0.6,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | context_markers: %{"conversation" => marker}
      }

      stimulus = %{type: :social, source: "someone", social_context: :conversation}

      {updated_somatic, bias} = SomaticMarkers.recall(somatic, stimulus)

      assert bias.bias > 0
      assert bias.markers_activated == 1
      assert updated_somatic.current_bias > 0
    end

    test "combines multiple matching markers" do
      social_marker = %{
        valence: 0.7,
        strength: 0.5,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      context_marker = %{
        valence: 0.6,
        strength: 0.5,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"friend" => social_marker},
          context_markers: %{"conversation" => context_marker}
      }

      stimulus = %{type: :social, source: "friend", social_context: :conversation}

      {_updated_somatic, bias} = SomaticMarkers.recall(somatic, stimulus)

      assert bias.markers_activated == 2
      assert bias.bias > 0
    end

    test "reinforces activated markers" do
      original_strength = 0.5

      marker = %{
        valence: 0.7,
        strength: original_strength,
        last_activated: DateTime.add(DateTime.utc_now(:second), -3600, :second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"crush" => marker}
      }

      stimulus = %{type: :social, source: "crush"}

      {updated_somatic, _bias} = SomaticMarkers.recall(somatic, stimulus)

      reinforced_marker = Map.get(updated_somatic.social_markers, "crush")
      assert reinforced_marker.strength > original_strength
    end

    test "generates positive body signal for strong positive bias" do
      marker = %{
        valence: 0.9,
        strength: 0.9,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"crush" => marker}
      }

      stimulus = %{type: :social, source: "crush"}

      {updated_somatic, bias} = SomaticMarkers.recall(somatic, stimulus)

      assert bias.signal =~ "warm"
      assert updated_somatic.body_signal =~ "warm"
    end

    test "generates negative body signal for strong negative bias" do
      marker = %{
        valence: -0.9,
        strength: 0.9,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"conversation_partner" => marker}
      }

      stimulus = %{type: :social, source: "conversation_partner"}

      {updated_somatic, bias} = SomaticMarkers.recall(somatic, stimulus)

      assert bias.signal =~ "tighten" or bias.signal =~ "unease"
      assert updated_somatic.body_signal != nil
    end
  end

  describe "maybe_learn/4" do
    test "does not learn marker when intensity below threshold" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "conversation_partner"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.2, arousal: 0.2}

      updated_somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      assert updated_somatic.markers_formed == 0
      assert map_size(updated_somatic.social_markers) == 0
    end

    test "learns marker when intensity above threshold" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "conversation_partner", social_context: :conversation}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.8, arousal: 0.9}

      updated_somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      assert updated_somatic.markers_formed > 0
    end

    test "learns positive marker from high pleasure high arousal" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "crush"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.9, arousal: 0.8}

      updated_somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      marker = Map.get(updated_somatic.social_markers, "crush")
      assert marker != nil
      assert marker.valence > 0
    end

    test "learns negative marker from low pleasure high arousal" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "conversation_partner"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: -0.8, arousal: 0.9}

      updated_somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      marker = Map.get(updated_somatic.social_markers, "conversation_partner")
      assert marker != nil
      assert marker.valence < 0
    end

    test "learns activity marker for social type" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "friend"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.8, arousal: 0.9}

      updated_somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      assert Map.has_key?(updated_somatic.activity_markers, :social)
    end

    test "learns context marker when social_context present" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "friend", social_context: :conversation}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.8, arousal: 0.9}

      updated_somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      assert Map.has_key?(updated_somatic.context_markers, "conversation")
    end

    test "merges with existing marker" do
      existing_marker = %{
        valence: 0.5,
        strength: 0.4,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"friend" => existing_marker}
      }

      stimulus = %{type: :social, source: "friend"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.9, arousal: 0.9}

      updated_somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      merged_marker = Map.get(updated_somatic.social_markers, "friend")
      assert merged_marker.strength > existing_marker.strength
    end

    test "decays markers when intensity below threshold" do
      marker = %{
        valence: 0.5,
        strength: 0.5,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"friend" => marker}
      }

      stimulus = %{type: :ambient, source: "environment"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.1, arousal: 0.1}

      updated_somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      decayed_marker = Map.get(updated_somatic.social_markers, "friend")
      assert decayed_marker.strength < marker.strength
    end

    test "removes markers that decay below minimum strength" do
      # Strength 0.1009 - 0.001 = 0.0999, which is below @min_strength (0.1)
      marker = %{
        valence: 0.5,
        strength: 0.1009,
        last_activated: DateTime.utc_now(:second),
        context: nil
      }

      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | social_markers: %{"weak_memory" => marker}
      }

      stimulus = %{type: :ambient, source: "environment"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.0, arousal: 0.0}

      updated_somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      assert Map.get(updated_somatic.social_markers, "weak_memory") == nil
    end
  end

  describe "describe/1" do
    test "describes positive bias state" do
      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | current_bias: 0.5,
          body_signal: "A warm feeling"
      }

      description = SomaticMarkers.describe(somatic)
      assert description =~ "warm" or description =~ "drawn"
    end

    test "describes negative bias state" do
      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | current_bias: -0.5,
          body_signal: "Stomach tightens"
      }

      description = SomaticMarkers.describe(somatic)
      assert description =~ "tighten" or description =~ "tense" or description =~ "warning"
    end

    test "describes neutral state with no markers" do
      somatic = SomaticMarkersState.new()

      description = SomaticMarkers.describe(somatic)
      assert description =~ "No strong body memories"
    end

    test "describes neutral state with existing markers" do
      somatic = %SomaticMarkersState{
        SomaticMarkersState.new()
        | current_bias: 0.0,
          markers_formed: 3
      }

      description = SomaticMarkers.describe(somatic)
      assert description =~ "neutral"
    end
  end

  describe "body_warning?/1" do
    test "returns true for negative bias below threshold" do
      somatic = %SomaticMarkersState{SomaticMarkersState.new() | current_bias: -0.3}
      assert SomaticMarkers.body_warning?(somatic)
    end

    test "returns false for neutral bias" do
      somatic = %SomaticMarkersState{SomaticMarkersState.new() | current_bias: 0.0}
      refute SomaticMarkers.body_warning?(somatic)
    end

    test "returns false for positive bias" do
      somatic = %SomaticMarkersState{SomaticMarkersState.new() | current_bias: 0.5}
      refute SomaticMarkers.body_warning?(somatic)
    end
  end

  describe "body_attraction?/1" do
    test "returns true for positive bias above threshold" do
      somatic = %SomaticMarkersState{SomaticMarkersState.new() | current_bias: 0.3}
      assert SomaticMarkers.body_attraction?(somatic)
    end

    test "returns false for neutral bias" do
      somatic = %SomaticMarkersState{SomaticMarkersState.new() | current_bias: 0.0}
      refute SomaticMarkers.body_attraction?(somatic)
    end

    test "returns false for negative bias" do
      somatic = %SomaticMarkersState{SomaticMarkersState.new() | current_bias: -0.5}
      refute SomaticMarkers.body_attraction?(somatic)
    end
  end

  describe "integration scenarios" do
    test "body learns from repeated positive experiences with same source" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "friend"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.8, arousal: 0.8}

      somatic =
        Enum.reduce(1..3, somatic, fn _, acc ->
          SomaticMarkers.maybe_learn(acc, stimulus, bio, emotional)
        end)

      marker = Map.get(somatic.social_markers, "friend")
      assert marker != nil
      assert marker.strength > 0.5
    end

    test "body warning suppresses approach after negative experience" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "conversation_partner"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: -0.9, arousal: 0.9}

      somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      {updated_somatic, bias} = SomaticMarkers.recall(somatic, stimulus)

      assert bias.bias < 0
      assert SomaticMarkers.body_warning?(updated_somatic)
    end

    test "body attraction increases approach after positive experience" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "crush"}
      bio = %BioState{}
      emotional = %EmotionalState{pleasure: 0.9, arousal: 0.9}

      somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, emotional)

      {updated_somatic, bias} = SomaticMarkers.recall(somatic, stimulus)

      assert bias.bias > 0
      assert SomaticMarkers.body_attraction?(updated_somatic)
    end

    test "mixed experiences blend marker valence" do
      somatic = SomaticMarkersState.new()
      stimulus = %{type: :social, source: "friend"}
      bio = %BioState{}

      positive_emotional = %EmotionalState{pleasure: 0.8, arousal: 0.8}
      somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, positive_emotional)

      negative_emotional = %EmotionalState{pleasure: -0.6, arousal: 0.8}
      somatic = SomaticMarkers.maybe_learn(somatic, stimulus, bio, negative_emotional)

      marker = Map.get(somatic.social_markers, "friend")
      assert marker.valence > -0.6
      assert marker.valence < 0.8
    end
  end
end
