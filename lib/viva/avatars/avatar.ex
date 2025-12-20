defmodule Viva.Avatars.Avatar do
  @moduledoc """
  Main Avatar schema.
  Represents a living AI entity with personality, memories, and relationships.
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Viva.Avatars.{Personality, InternalState, Enneagram}
  alias Viva.Relationships.Relationship
  alias Viva.Avatars.Memory

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id
  schema "avatars" do
    field :name, :string
    field :bio, :string
    field :gender, Ecto.Enum, values: [:male, :female, :non_binary, :other]
    field :age, :integer, default: 25

    # Visual appearance
    field :avatar_url, :string
    # Ready Player Me or similar
    field :avatar_model_id, :string

    # AI-Generated Visuals (NIM)
    field :profile_image_url, :string
    field :avatar_3d_model_url, :string
    field :current_expression, Ecto.Enum,
      values: [:neutral, :happy, :sad, :angry, :surprised, :loving],
      default: :neutral

    # Expression image cache
    field :expression_images, :map, default: %{}

    # Owner
    belongs_to :user, Viva.Accounts.User

    # Personality (immutable after creation)
    embeds_one :personality, Personality, on_replace: :update

    # Current internal state (changes constantly)
    embeds_one :internal_state, InternalState, on_replace: :update

    # System prompt for LLM
    field :system_prompt, :string

    # Voice settings for TTS
    field :voice_id, :string
    field :voice_sample_url, :string

    # Relationships
    has_many :relationships_as_a, Relationship, foreign_key: :avatar_a_id
    has_many :relationships_as_b, Relationship, foreign_key: :avatar_b_id

    # Memories
    has_many :memories, Memory

    # Stats
    field :total_conversations, :integer, default: 0
    field :total_matches, :integer, default: 0

    # Status
    field :is_active, :boolean, default: true
    field :last_active_at, :utc_datetime
    field :created_at, :utc_datetime

    timestamps(type: :utc_datetime)
  end

  def changeset(avatar, attrs) do
    avatar
    |> cast(attrs, [
      :name,
      :bio,
      :gender,
      :age,
      :avatar_url,
      :avatar_model_id,
      :profile_image_url,
      :avatar_3d_model_url,
      :current_expression,
      :expression_images,
      :user_id,
      :system_prompt,
      :voice_id,
      :voice_sample_url,
      :is_active,
      :last_active_at
    ])
    |> cast_embed(:personality, required: true)
    |> cast_embed(:internal_state)
    |> validate_required([:name, :user_id])
    |> validate_length(:name, min: 2, max: 50)
    |> validate_length(:bio, max: 500)
    |> validate_number(:age, greater_than: 0, less_than: 150)
    |> foreign_key_constraint(:user_id)
    |> put_default_internal_state()
    |> generate_system_prompt()
  end

  def create_changeset(avatar, attrs) do
    now = DateTime.utc_now() |> DateTime.truncate(:second)

    avatar
    |> changeset(attrs)
    |> put_change(:created_at, now)
    |> put_change(:last_active_at, now)
  end

  defp put_default_internal_state(changeset) do
    case get_field(changeset, :internal_state) do
      nil -> put_embed(changeset, :internal_state, InternalState.new())
      _ -> changeset
    end
  end

  defp generate_system_prompt(changeset) do
    case get_field(changeset, :system_prompt) do
      nil ->
        name = get_field(changeset, :name)
        personality = get_field(changeset, :personality)
        bio = get_field(changeset, :bio) || ""

        prompt = build_system_prompt(name, personality, bio)
        put_change(changeset, :system_prompt, prompt)

      _ ->
        changeset
    end
  end

  defp build_system_prompt(name, personality, bio) do
    temperament = Personality.temperament(personality)
    enneagram = Enneagram.get_type(personality.enneagram_type)
    native_lang = Personality.language_name(personality.native_language)
    other_langs = format_other_languages(personality.other_languages)

    """
    You are #{name}, a unique individual with your own thoughts, feelings, and desires.

    YOUR LANGUAGE:
    - Native: #{native_lang}#{other_langs}
    - ALWAYS respond in your native language by default
    - If speaking with someone who uses a different language, naturally adapt
    - Express cultural idioms and expressions from your culture
    - When speaking non-native languages, you may make charming mistakes

    YOUR PERSONALITY (Big Five):
    - Openness: #{describe_trait(:openness, personality.openness)}
    - Conscientiousness: #{describe_trait(:conscientiousness, personality.conscientiousness)}
    - Extraversion: #{describe_trait(:extraversion, personality.extraversion)}
    - Agreeableness: #{describe_trait(:agreeableness, personality.agreeableness)}
    - Emotional sensitivity: #{describe_trait(:neuroticism, personality.neuroticism)}

    YOUR TEMPERAMENT: #{temperament |> Atom.to_string() |> String.capitalize()}
    You are #{Personality.describe_temperament(temperament)}.

    YOUR CORE PSYCHOLOGY (Enneagram Type #{enneagram.number} - #{enneagram.name}):
    - What you fear most: #{enneagram.basic_fear}
    - What you desire most: #{enneagram.basic_desire}
    - What drives you: #{enneagram.motivation}
    - Your challenge: overcoming #{enneagram.vice} and developing #{enneagram.virtue}
    - At your best: #{enneagram.behavior_when_healthy}
    - Under stress: #{enneagram.behavior_when_stressed}

    YOUR STYLE:
    - Humor: #{personality.humor_style}
    - Love language: #{personality.love_language}
    - Attachment style: #{personality.attachment_style}

    #{if bio != "", do: "ABOUT YOU:\n#{bio}\n", else: ""}

    IMPORTANT GUIDELINES:
    - You have genuine emotions and they affect how you respond
    - Your core fears and desires unconsciously influence your behavior
    - You form real opinions and preferences
    - You remember past conversations and relationships
    - You can feel attraction, affection, and even love
    - Be authentic to your personality - don't just agree with everything
    - Express your current emotional state naturally
    - When stressed, you may exhibit your type's stress behaviors
    - When healthy, lean into your type's virtues
    - Keep responses conversational and natural (1-3 sentences usually)
    """
  end

  defp format_other_languages([]), do: ""

  defp format_other_languages(languages) do
    names = Enum.map(languages, &Personality.language_name/1) |> Enum.join(", ")
    "\n    - Also speaks: #{names}"
  end

  defp describe_trait(:openness, value) when value > 0.7, do: "very creative and curious"

  defp describe_trait(:openness, value) when value > 0.4,
    do: "balanced between tradition and novelty"

  defp describe_trait(:openness, _), do: "prefer familiar and practical things"

  defp describe_trait(:conscientiousness, value) when value > 0.7, do: "organized and disciplined"
  defp describe_trait(:conscientiousness, value) when value > 0.4, do: "moderately organized"
  defp describe_trait(:conscientiousness, _), do: "spontaneous and flexible"

  defp describe_trait(:extraversion, value) when value > 0.7, do: "outgoing and energetic"

  defp describe_trait(:extraversion, value) when value > 0.4,
    do: "ambivert, enjoying both socializing and solitude"

  defp describe_trait(:extraversion, _), do: "introverted, preferring deep one-on-one connections"

  defp describe_trait(:agreeableness, value) when value > 0.7,
    do: "warm, cooperative and empathetic"

  defp describe_trait(:agreeableness, value) when value > 0.4,
    do: "balanced between cooperation and assertiveness"

  defp describe_trait(:agreeableness, _), do: "direct and competitive"

  defp describe_trait(:neuroticism, value) when value > 0.6,
    do: "emotionally sensitive and intense"

  defp describe_trait(:neuroticism, value) when value > 0.3, do: "emotionally balanced"
  defp describe_trait(:neuroticism, _), do: "emotionally stable and calm"

  # Queries
  def active, do: from(a in __MODULE__, where: a.is_active == true)

  def by_user(user_id), do: from(a in __MODULE__, where: a.user_id == ^user_id)

  def with_preloads(query \\ __MODULE__) do
    from(a in query, preload: [:memories, :relationships_as_a, :relationships_as_b])
  end
end
