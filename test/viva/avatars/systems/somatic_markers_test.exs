defmodule Viva.Avatars.Systems.SomaticMarkersTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.SomaticMarkersState
  alias Viva.Avatars.Systems.SomaticMarkers

  setup do
    somatic = SomaticMarkersState.new()
    bio = %BioState{}
    {:ok, somatic: somatic, bio: bio}
  end

  describe "maybe_learn/4" do
    test "learns new positive marker", %{somatic: s, bio: b} do
      # Intensity = arousal*0.6 + abs(pleasure)*0.4. Threshold = 0.7
      # a=0.8, p=0.8 -> 0.48 + 0.32 = 0.8
      e = %EmotionalState{pleasure: 0.8, arousal: 0.8}
      stimulus = %{source: "friend", type: :social, social_context: "party"}

      updated = SomaticMarkers.maybe_learn(s, stimulus, b, e)

      assert updated.markers_formed == 1
      assert Map.has_key?(updated.social_markers, "friend")
      assert updated.social_markers["friend"].valence > 0
    end

    test "learns new negative marker", %{somatic: s, bio: b} do
      e = %EmotionalState{pleasure: -0.8, arousal: 0.8}
      stimulus = %{source: "conversation_partner", type: :social}

      updated = SomaticMarkers.maybe_learn(s, stimulus, b, e)

      assert updated.social_markers["conversation_partner"].valence < 0
    end

    test "merges with existing markers", %{bio: b} do
      m = %{valence: 0.5, strength: 0.5, last_activated: DateTime.utc_now(), context: nil}
      s = %SomaticMarkersState{social_markers: %{"friend" => m}}

      e = %EmotionalState{pleasure: 0.9, arousal: 0.9}
      stimulus = %{source: "friend"}

      updated = SomaticMarkers.maybe_learn(s, stimulus, b, e)

      # Strength should increase
      assert updated.social_markers["friend"].strength > 0.5
    end

    test "decays markers when intensity low", %{bio: b} do
      # Set strength high enough to survive decay but still decrease
      m = %{valence: 0.5, strength: 0.5, last_activated: DateTime.utc_now(), context: nil}
      s = %SomaticMarkersState{social_markers: %{"friend" => m}}

      e = %EmotionalState{pleasure: 0.0, arousal: 0.0}
      stimulus = %{source: "friend"}

      updated = SomaticMarkers.maybe_learn(s, stimulus, b, e)

      assert updated.social_markers["friend"].strength < 0.5
    end
  end

  describe "recall/2" do
    test "activates matching markers", %{bio: _b} do
      m = %{valence: 0.8, strength: 0.8, last_activated: nil, context: nil}
      s = %SomaticMarkersState{social_markers: %{"friend" => m}}

      stimulus = %{source: "friend", type: :social}
      {updated, result} = SomaticMarkers.recall(s, stimulus)

      assert result.bias > 0
      assert result.markers_activated == 1
      assert updated.current_bias == result.bias

      updated.body_signal
      |> String.downcase()
      |> String.contains?("warm")
      |> assert()
    end

    test "returns neutral if no match", %{somatic: s} do
      {_, result} = SomaticMarkers.recall(s, %{source: "unknown"})
      assert result.bias == 0.0
      assert result.markers_activated == 0
    end
  end

  describe "decision_confidence/2" do
    test "returns levels based on marker valence" do
      m = %{valence: 0.9, strength: 1.0}
      s = %SomaticMarkersState{activity_markers: %{social: m}}

      {conf, warning} = SomaticMarkers.decision_confidence(s, :social)
      assert conf == 1.0
      assert warning == :go_for_it

      m_neg = %{valence: -0.9, strength: 1.0}
      s_neg = %SomaticMarkersState{activity_markers: %{social: m_neg}}
      {conf_n, warning_n} = SomaticMarkers.decision_confidence(s_neg, :social)
      assert conf_n == 0.1
      assert warning_n == :avoid
    end
  end

  describe "apply_confidence_to_desire/3" do
    test "adjusts weight up for positive markers" do
      m = %{valence: 0.9, strength: 1.0}
      s = %SomaticMarkersState{activity_markers: %{social: m}}

      adjusted = SomaticMarkers.apply_confidence_to_desire(s, :social, 0.5)
      assert adjusted > 0.5
    end
  end

  describe "describe/1 and explain_confidence/2" do
    test "returns human readable info" do
      s = %SomaticMarkersState{current_bias: 0.5, body_signal: "Warm sun"}
      assert SomaticMarkers.describe(s) == "Warm sun"

      # No activity marker
      assert SomaticMarkers.explain_confidence(s, :social) =~ "new territory"

      s_with_act = %{s | activity_markers: %{social: %{valence: 0.9, strength: 1.0}}}
      assert SomaticMarkers.explain_confidence(s_with_act, :social) =~ "warmth"
    end
  end

  describe "query helpers" do
    test "body_warning? and body_attraction?" do
      s = %SomaticMarkersState{current_bias: 0.5}
      assert SomaticMarkers.body_attraction?(s)
      refute SomaticMarkers.body_warning?(s)

      s_warn = %SomaticMarkersState{current_bias: -0.5}
      assert SomaticMarkers.body_warning?(s_warn)
      refute SomaticMarkers.body_attraction?(s_warn)
    end
  end
end
