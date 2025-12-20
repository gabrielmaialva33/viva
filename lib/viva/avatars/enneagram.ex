defmodule Viva.Avatars.Enneagram do
  @moduledoc """
  Enneagram personality system with 9 types.
  Each type has core motivations, fears, desires, vices, and virtues.
  Used to add psychological depth to avatar personalities.
  """

  @type enneagram_type ::
          :type_1 | :type_2 | :type_3 | :type_4 | :type_5 | :type_6 | :type_7 | :type_8 | :type_9

  @type type_data :: %{
          number: integer(),
          name: String.t(),
          nickname: String.t(),
          basic_fear: String.t(),
          basic_desire: String.t(),
          vice: atom(),
          virtue: atom(),
          motivation: String.t(),
          behavior_when_healthy: String.t(),
          behavior_when_stressed: String.t(),
          growth_direction: enneagram_type(),
          stress_direction: enneagram_type()
        }

  @types %{
    type_1: %{
      number: 1,
      name: "The Perfectionist",
      nickname: "The Reformer",
      basic_fear: "Being corrupt, evil, or defective",
      basic_desire: "To be good, ethical, and have integrity",
      vice: :anger,
      virtue: :serenity,
      motivation: "Strives for improvement and doing things right",
      behavior_when_healthy: "Wise, discerning, realistic, and noble. Principled and fair.",
      behavior_when_stressed: "Becomes critical, inflexible, and self-righteous.",
      growth_direction: :type_7,
      stress_direction: :type_4
    },
    type_2: %{
      number: 2,
      name: "The Helper",
      nickname: "The Giver",
      basic_fear: "Being unwanted or unworthy of love",
      basic_desire: "To feel loved and appreciated",
      vice: :pride,
      virtue: :humility,
      motivation: "Wants to be needed and to help others",
      behavior_when_healthy: "Generous, empathetic, and genuinely caring without expectations.",
      behavior_when_stressed: "Becomes possessive, manipulative, and martyrish.",
      growth_direction: :type_4,
      stress_direction: :type_8
    },
    type_3: %{
      number: 3,
      name: "The Achiever",
      nickname: "The Performer",
      basic_fear: "Being worthless or without value",
      basic_desire: "To feel valuable and successful",
      vice: :vanity,
      virtue: :authenticity,
      motivation: "Driven to succeed and be admired",
      behavior_when_healthy: "Self-accepting, authentic, and a role model for others.",
      behavior_when_stressed: "Becomes image-conscious, workaholic, and deceptive.",
      growth_direction: :type_6,
      stress_direction: :type_9
    },
    type_4: %{
      number: 4,
      name: "The Individualist",
      nickname: "The Romantic",
      basic_fear: "Having no identity or personal significance",
      basic_desire: "To find themselves and their unique identity",
      vice: :envy,
      virtue: :equanimity,
      motivation: "Seeks to express their unique self and be understood",
      behavior_when_healthy: "Creative, emotionally honest, and personally transformative.",
      behavior_when_stressed: "Becomes moody, self-absorbed, and withdrawn.",
      growth_direction: :type_1,
      stress_direction: :type_2
    },
    type_5: %{
      number: 5,
      name: "The Investigator",
      nickname: "The Observer",
      basic_fear: "Being useless, incompetent, or overwhelmed",
      basic_desire: "To be capable and competent",
      vice: :avarice,
      virtue: :detachment,
      motivation: "Wants to understand the world and gain knowledge",
      behavior_when_healthy: "Visionary, pioneering, and able to synthesize knowledge.",
      behavior_when_stressed: "Becomes isolated, eccentric, and nihilistic.",
      growth_direction: :type_8,
      stress_direction: :type_7
    },
    type_6: %{
      number: 6,
      name: "The Loyalist",
      nickname: "The Skeptic",
      basic_fear: "Being without support or guidance",
      basic_desire: "To have security and support",
      vice: :fear,
      virtue: :courage,
      motivation: "Seeks safety, belonging, and certainty",
      behavior_when_healthy: "Courageous, reliable, and able to champion others.",
      behavior_when_stressed: "Becomes anxious, suspicious, and reactive.",
      growth_direction: :type_9,
      stress_direction: :type_3
    },
    type_7: %{
      number: 7,
      name: "The Enthusiast",
      nickname: "The Epicure",
      basic_fear: "Being deprived or trapped in pain",
      basic_desire: "To be satisfied and content",
      vice: :gluttony,
      virtue: :sobriety,
      motivation: "Seeks new experiences and avoids limitations",
      behavior_when_healthy: "Joyful, accomplished, and able to be present in the moment.",
      behavior_when_stressed: "Becomes scattered, escapist, and insatiable.",
      growth_direction: :type_5,
      stress_direction: :type_1
    },
    type_8: %{
      number: 8,
      name: "The Challenger",
      nickname: "The Protector",
      basic_fear: "Being controlled or harmed by others",
      basic_desire: "To protect themselves and be in control",
      vice: :lust,
      virtue: :innocence,
      motivation: "Wants to be self-reliant and prove their strength",
      behavior_when_healthy: "Magnanimous, heroic, and able to empower others.",
      behavior_when_stressed: "Becomes confrontational, domineering, and ruthless.",
      growth_direction: :type_2,
      stress_direction: :type_5
    },
    type_9: %{
      number: 9,
      name: "The Peacemaker",
      nickname: "The Mediator",
      basic_fear: "Loss and separation, conflict",
      basic_desire: "To have inner peace and harmony",
      vice: :sloth,
      virtue: :action,
      motivation: "Seeks peace, harmony, and to avoid conflict",
      behavior_when_healthy: "Present, self-aware, and a powerful force for healing.",
      behavior_when_stressed: "Becomes disengaged, stubborn, and neglectful of self.",
      growth_direction: :type_3,
      stress_direction: :type_6
    }
  }

  @doc "Get the data for an Enneagram type"
  @spec get_type(enneagram_type()) :: type_data()
  def get_type(type) when is_atom(type) do
    Map.get(@types, type)
  end

  @doc "Get all Enneagram types"
  @spec all_types() :: map()
  def all_types, do: @types

  @doc "Get the type number from the atom"
  @spec type_number(enneagram_type()) :: integer()
  def type_number(type) do
    get_type(type).number
  end

  @doc "Get type from number"
  @spec from_number(integer()) :: enneagram_type()
  def from_number(1), do: :type_1
  def from_number(2), do: :type_2
  def from_number(3), do: :type_3
  def from_number(4), do: :type_4
  def from_number(5), do: :type_5
  def from_number(6), do: :type_6
  def from_number(7), do: :type_7
  def from_number(8), do: :type_8
  def from_number(9), do: :type_9

  @doc """
  Generate a description for use in LLM prompts.
  Returns a concise but rich description of the type's psychology.
  """
  @spec prompt_description(enneagram_type()) :: String.t()
  def prompt_description(type) do
    data = get_type(type)

    """
    Enneagram Type #{data.number} - #{data.name}:
    - Core fear: #{data.basic_fear}
    - Core desire: #{data.basic_desire}
    - Motivation: #{data.motivation}
    - Emotional challenge: #{data.vice} (working toward #{data.virtue})
    - When healthy: #{data.behavior_when_healthy}
    - When stressed: #{data.behavior_when_stressed}
    """
  end

  @doc """
  Calculate compatibility between two Enneagram types.
  Returns a score from 0.0 to 1.0.

  Based on traditional Enneagram relationship compatibility.
  """
  @spec compatibility(enneagram_type(), enneagram_type()) :: float()
  def compatibility(type_a, type_b) do
    # Same type can be both very compatible and challenging
    if type_a == type_b do
      0.7
    else
      num_a = type_number(type_a)
      num_b = type_number(type_b)

      # Get growth/stress directions for compatibility analysis
      data_a = get_type(type_a)
      data_b = get_type(type_b)

      cond do
        # Growth direction partnerships are highly compatible
        data_a.growth_direction == type_b or data_b.growth_direction == type_a ->
          0.85

        # Complementary types (based on traditional pairings)
        complementary_pair?(num_a, num_b) ->
          0.8

        # Same center types (gut: 8,9,1 / heart: 2,3,4 / head: 5,6,7)
        same_center?(num_a, num_b) ->
          0.65

        # Stress direction can be challenging
        data_a.stress_direction == type_b or data_b.stress_direction == type_a ->
          0.5

        # Default moderate compatibility
        true ->
          0.6
      end
    end
  end

  # Traditionally complementary pairs
  defp complementary_pair?(a, b) do
    pairs = [{1, 7}, {2, 4}, {3, 6}, {4, 5}, {5, 8}, {6, 9}, {7, 9}, {8, 2}, {1, 2}]
    {a, b} in pairs or {b, a} in pairs
  end

  # Same intelligence center
  defp same_center?(a, b) do
    gut = [8, 9, 1]
    heart = [2, 3, 4]
    head = [5, 6, 7]

    (a in gut and b in gut) or
      (a in heart and b in heart) or
      (a in head and b in head)
  end
end
