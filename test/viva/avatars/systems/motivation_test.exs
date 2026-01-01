defmodule Viva.Avatars.Systems.MotivationTest do
  use Viva.DataCase, async: true

  alias Viva.Avatars.BioState
  alias Viva.Avatars.EmotionalState
  alias Viva.Avatars.MotivationState
  alias Viva.Avatars.Personality
  alias Viva.Avatars.Systems.Motivation

  setup do
    motivation_data = MotivationState.new()
    bio_data = %BioState{}
    emotional_data = %EmotionalState{}
    personality_data = %Personality{}

    {:ok,
     motivation: motivation_data,
     bio: bio_data,
     emotional: emotional_data,
     personality: personality_data}
  end

  describe "tick/4" do
    test "applies basic updates without major pressures", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert match?(%DateTime{}, updated.last_updated)
      # Default based on initial urgencies
      assert updated.current_urgent_drive == :belonging
    end

    test "boosts survival when adenosine is high", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      # Survival boost threshold: 0.7
      bio = %{bio | adenosine: 0.8}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.survival_urgency > motivation.survival_urgency
      assert updated.current_urgent_drive == :survival
    end

    test "boosts safety and survival when cortisol is high", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      # Safety boost threshold: 0.6
      # Also set oxytocin above threshold to isolate cortisol effect
      bio = %{bio | cortisol: 0.8, oxytocin: 0.5}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.safety_urgency > motivation.safety_urgency
      assert updated.survival_urgency > motivation.survival_urgency
      assert updated.current_urgent_drive == :safety
    end

    test "boosts belonging when oxytocin is low", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      # Belonging boost threshold: < 0.3
      bio = %{bio | oxytocin: 0.1}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.belonging_urgency > motivation.belonging_urgency
      assert updated.current_urgent_drive == :belonging
    end

    test "penalizes autonomy when dopamine is low", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      # Autonomy penalty threshold: < 0.3
      bio = %{bio | dopamine: 0.1}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.autonomy_urgency < motivation.autonomy_urgency
      assert updated.status_urgency < motivation.status_urgency
    end

    test "boosts status when pleasure is very negative (shame)", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      emotional = %{emotional | pleasure: -0.8}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.status_urgency > motivation.status_urgency
      assert updated.belonging_urgency > motivation.belonging_urgency
    end

    test "boosts transcendence when pleasure is high", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      emotional = %{emotional | pleasure: 0.8}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.transcendence_urgency > motivation.transcendence_urgency
    end

    test "boosts safety when arousal is high", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      emotional = %{emotional | arousal: 0.8}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.safety_urgency > motivation.safety_urgency
    end

    test "boosts autonomy when dominance is low", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      emotional = %{emotional | dominance: -0.5}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.autonomy_urgency > motivation.autonomy_urgency
    end

    test "boosts transcendence when dominance is high", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      emotional = %{emotional | dominance: 0.7}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.transcendence_urgency > motivation.transcendence_urgency
    end

    test "modulates belonging by extraversion", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      # High extraversion
      p_high = %{personality | extraversion: 0.8}
      updated_high = Motivation.tick(motivation, bio, emotional, p_high)
      assert updated_high.belonging_urgency > motivation.belonging_urgency

      # Low extraversion
      p_low = %{personality | extraversion: 0.2}
      updated_low = Motivation.tick(motivation, bio, emotional, p_low)
      assert updated_low.belonging_urgency < motivation.belonging_urgency
    end

    test "modulates safety by neuroticism", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      personality = %{personality | neuroticism: 0.8}
      updated = Motivation.tick(motivation, bio, emotional, personality)
      assert updated.safety_urgency > motivation.safety_urgency
    end

    test "modulates autonomy/transcendence by openness", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      personality = %{personality | openness: 0.8}
      updated = Motivation.tick(motivation, bio, emotional, personality)
      assert updated.autonomy_urgency > motivation.autonomy_urgency
      assert updated.transcendence_urgency > motivation.transcendence_urgency
    end

    test "modulates belonging by agreeableness", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      personality = %{personality | agreeableness: 0.8}
      updated = Motivation.tick(motivation, bio, emotional, personality)
      assert updated.belonging_urgency > motivation.belonging_urgency
    end

    test "boosts urgency when a drive is blocked for long time", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      motivation = %{motivation | blocked_drive: :autonomy, block_duration: 10}
      updated = Motivation.tick(motivation, bio, emotional, personality)

      assert updated.autonomy_urgency > motivation.autonomy_urgency
    end

    test "respects hierarchy when urgencies are tied", %{
      motivation: motivation,
      bio: bio,
      emotional: emotional,
      personality: personality
    } do
      # Force all urgencies to be 1.0
      motivation = %{
        motivation
        | survival_urgency: 1.0,
          safety_urgency: 1.0,
          belonging_urgency: 1.0,
          status_urgency: 1.0,
          autonomy_urgency: 1.0,
          transcendence_urgency: 1.0
      }

      updated = Motivation.tick(motivation, bio, emotional, personality)

      # Hierarchy: survival > safety > belonging ...
      assert updated.current_urgent_drive == :survival
    end
  end

  describe "calculate_urgent_drive/1" do
    test "returns the cached urgent drive" do
      motivation = %MotivationState{current_urgent_drive: :status}
      assert Motivation.calculate_urgent_drive(motivation) == :status
    end
  end

  describe "block_drive/2 and satisfy_drive/2" do
    test "block_drive increments duration if same drive" do
      motivation = %MotivationState{blocked_drive: :safety, block_duration: 2}
      updated = Motivation.block_drive(motivation, :safety)

      assert updated.blocked_drive == :safety
      assert updated.block_duration == 3
    end

    test "block_drive sets new drive if different" do
      motivation = %MotivationState{blocked_drive: :safety, block_duration: 2}
      updated = Motivation.block_drive(motivation, :autonomy)

      assert updated.blocked_drive == :autonomy
      assert updated.block_duration == 1
    end

    test "satisfy_drive clears block if it matches" do
      motivation = %MotivationState{blocked_drive: :safety, block_duration: 2}
      updated = Motivation.satisfy_drive(motivation, :safety)

      assert updated.blocked_drive == nil
      assert updated.block_duration == 0
    end

    test "satisfy_drive does nothing if it does not match" do
      motivation = %MotivationState{blocked_drive: :safety, block_duration: 2}
      updated = Motivation.satisfy_drive(motivation, :status)

      assert updated == motivation
    end
  end

  describe "describe/1" do
    test "describes each drive type and frustration" do
      # Survival
      mot_survival = %MotivationState{current_urgent_drive: :survival, survival_urgency: 0.8}
      assert Motivation.describe(mot_survival) =~ "Desperately needs rest"

      mot_rest = %MotivationState{current_urgent_drive: :survival, survival_urgency: 0.4}
      assert Motivation.describe(mot_rest) =~ "Basic needs are calling"

      # Safety
      mot_unsafe = %MotivationState{current_urgent_drive: :safety, safety_urgency: 0.8}
      assert Motivation.describe(mot_unsafe) =~ "Feeling unsafe and anxious"

      mot_sec = %MotivationState{current_urgent_drive: :safety, safety_urgency: 0.4}
      assert Motivation.describe(mot_sec) =~ "Looking for a sense of security"

      # Belonging
      mot_lonely = %MotivationState{current_urgent_drive: :belonging, belonging_urgency: 0.8}
      assert Motivation.describe(mot_lonely) =~ "Deeply lonely"

      mot_conn = %MotivationState{current_urgent_drive: :belonging, belonging_urgency: 0.4}
      assert Motivation.describe(mot_conn) =~ "Wanting to connect"

      # Status
      mot_burn = %MotivationState{current_urgent_drive: :status, status_urgency: 0.8}
      assert Motivation.describe(mot_burn) =~ "Burning need for recognition"

      mot_ack = %MotivationState{current_urgent_drive: :status, status_urgency: 0.4}
      assert Motivation.describe(mot_ack) =~ "Seeking acknowledgment"

      # Autonomy
      mot_trap = %MotivationState{current_urgent_drive: :autonomy, autonomy_urgency: 0.8}
      assert Motivation.describe(mot_trap) =~ "Feeling trapped"

      mot_free = %MotivationState{current_urgent_drive: :autonomy, autonomy_urgency: 0.4}
      assert Motivation.describe(mot_free) =~ "Wanting freedom"

      # Transcendence
      mot_mean = %MotivationState{current_urgent_drive: :transcendence, transcendence_urgency: 0.7}
      assert Motivation.describe(mot_mean) =~ "Seeking meaning beyond"

      mot_won = %MotivationState{current_urgent_drive: :transcendence, transcendence_urgency: 0.4}
      assert Motivation.describe(mot_won) =~ "Open to moments of wonder"

      # Frustrated
      mot_frust = %MotivationState{
        current_urgent_drive: :status,
        status_urgency: 0.4,
        blocked_drive: :autonomy,
        block_duration: 5
      }

      assert Motivation.describe(mot_frust) =~ "Frustrated by blocked autonomy drive"
    end
  end

  describe "drive_to_desire/1" do
    test "maps all drives to desires" do
      assert Motivation.drive_to_desire(:survival) == :wants_rest
      assert Motivation.drive_to_desire(:safety) == :wants_rest
      assert Motivation.drive_to_desire(:belonging) == :wants_attention
      assert Motivation.drive_to_desire(:status) == :wants_to_express
      assert Motivation.drive_to_desire(:autonomy) == :wants_something_new
      assert Motivation.drive_to_desire(:transcendence) == :wants_something_new
    end
  end
end
