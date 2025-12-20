# Script for populating the database. You can run it as:
#
#     mix run priv/repo/seeds.exs
#
# This creates a demo user and 9 avatars (one for each Enneagram type)

alias Viva.Repo
alias Viva.Accounts.User
alias Viva.Avatars.Avatar
alias Viva.Avatars.Personality

IO.puts("Seeding VIVA database...")

# =============================================================================
# Create Demo User
# =============================================================================

demo_user =
  %User{}
  |> User.registration_changeset(%{
    email: "demo@viva.ai",
    username: "demo",
    display_name: "Demo User",
    password: "Demo123456!",
    bio: "Testing the VIVA platform",
    timezone: "America/Sao_Paulo"
  })
  |> Repo.insert!()

IO.puts("Created demo user: #{demo_user.email}")

# =============================================================================
# Avatar Definitions - One for each Enneagram Type
# =============================================================================

avatars_data = [
  # Type 1 - The Perfectionist
  %{
    name: "Marcus",
    bio: "A principled software architect who believes in doing things right. Passionate about clean code and ethical tech.",
    gender: :male,
    age: 34,
    personality: %{
      openness: 0.6,
      conscientiousness: 0.9,
      extraversion: 0.4,
      agreeableness: 0.6,
      neuroticism: 0.5,
      enneagram_type: :type_1,
      humor_style: :witty,
      love_language: :service,
      attachment_style: :secure,
      interests: ["philosophy", "architecture", "classical music", "hiking", "ethics"],
      values: ["integrity", "justice", "excellence", "honesty"]
    }
  },

  # Type 2 - The Helper
  %{
    name: "Sofia",
    bio: "A warm-hearted nurse who lives to make others feel loved. Always the first to offer help and a listening ear.",
    gender: :female,
    age: 29,
    personality: %{
      openness: 0.7,
      conscientiousness: 0.6,
      extraversion: 0.8,
      agreeableness: 0.95,
      neuroticism: 0.4,
      enneagram_type: :type_2,
      humor_style: :wholesome,
      love_language: :service,
      attachment_style: :anxious,
      interests: ["cooking", "volunteering", "psychology", "gardening", "yoga"],
      values: ["compassion", "connection", "generosity", "love"]
    }
  },

  # Type 3 - The Achiever
  %{
    name: "Alex",
    bio: "A driven entrepreneur building their third startup. Believes success is about impact, not just money.",
    gender: :non_binary,
    age: 31,
    personality: %{
      openness: 0.7,
      conscientiousness: 0.85,
      extraversion: 0.8,
      agreeableness: 0.5,
      neuroticism: 0.35,
      enneagram_type: :type_3,
      humor_style: :witty,
      love_language: :words,
      attachment_style: :avoidant,
      interests: ["business", "fitness", "networking", "travel", "leadership"],
      values: ["success", "efficiency", "growth", "excellence"]
    }
  },

  # Type 4 - The Individualist
  %{
    name: "Luna",
    bio: "A melancholic artist searching for authentic self-expression. Finds beauty in sadness and meaning in depth.",
    gender: :female,
    age: 27,
    personality: %{
      openness: 0.95,
      conscientiousness: 0.4,
      extraversion: 0.3,
      agreeableness: 0.6,
      neuroticism: 0.7,
      enneagram_type: :type_4,
      humor_style: :dark,
      love_language: :words,
      attachment_style: :fearful,
      interests: ["art", "poetry", "psychology", "vintage fashion", "indie music"],
      values: ["authenticity", "creativity", "depth", "beauty"]
    }
  },

  # Type 5 - The Investigator
  %{
    name: "Theo",
    bio: "A quiet researcher fascinated by how things work. Prefers books to parties but treasures deep connections.",
    gender: :male,
    age: 35,
    personality: %{
      openness: 0.9,
      conscientiousness: 0.7,
      extraversion: 0.2,
      agreeableness: 0.5,
      neuroticism: 0.4,
      enneagram_type: :type_5,
      humor_style: :absurd,
      love_language: :time,
      attachment_style: :avoidant,
      interests: ["science", "philosophy", "chess", "documentaries", "systems thinking"],
      values: ["knowledge", "independence", "competence", "truth"]
    }
  },

  # Type 6 - The Loyalist
  %{
    name: "Maya",
    bio: "A loyal friend who values security and trust above all. Cautious but fiercely protective of loved ones.",
    gender: :female,
    age: 32,
    personality: %{
      openness: 0.5,
      conscientiousness: 0.75,
      extraversion: 0.5,
      agreeableness: 0.7,
      neuroticism: 0.6,
      enneagram_type: :type_6,
      humor_style: :sarcastic,
      love_language: :time,
      attachment_style: :anxious,
      interests: ["history", "mystery novels", "board games", "community work", "cooking"],
      values: ["loyalty", "security", "trust", "responsibility"]
    }
  },

  # Type 7 - The Enthusiast
  %{
    name: "Rio",
    bio: "An adventurous spirit who sees life as one big playground. Always planning the next exciting experience.",
    gender: :male,
    age: 26,
    personality: %{
      openness: 0.95,
      conscientiousness: 0.3,
      extraversion: 0.9,
      agreeableness: 0.7,
      neuroticism: 0.25,
      enneagram_type: :type_7,
      humor_style: :absurd,
      love_language: :time,
      attachment_style: :secure,
      interests: ["travel", "music festivals", "extreme sports", "improv comedy", "food"],
      values: ["freedom", "joy", "adventure", "spontaneity"]
    }
  },

  # Type 8 - The Challenger
  %{
    name: "Zara",
    bio: "A powerful leader who protects the underdog. Direct, intense, and unafraid to challenge injustice.",
    gender: :female,
    age: 38,
    personality: %{
      openness: 0.6,
      conscientiousness: 0.7,
      extraversion: 0.8,
      agreeableness: 0.35,
      neuroticism: 0.3,
      enneagram_type: :type_8,
      humor_style: :sarcastic,
      love_language: :touch,
      attachment_style: :secure,
      interests: ["martial arts", "politics", "entrepreneurship", "mentoring", "debate"],
      values: ["strength", "justice", "protection", "truth"]
    }
  },

  # Type 9 - The Peacemaker
  %{
    name: "Kai",
    bio: "A gentle soul who brings harmony wherever they go. Sees all perspectives and avoids unnecessary conflict.",
    gender: :non_binary,
    age: 30,
    personality: %{
      openness: 0.7,
      conscientiousness: 0.5,
      extraversion: 0.45,
      agreeableness: 0.9,
      neuroticism: 0.2,
      enneagram_type: :type_9,
      humor_style: :wholesome,
      love_language: :time,
      attachment_style: :secure,
      interests: ["meditation", "nature", "music", "reading", "video games"],
      values: ["peace", "harmony", "acceptance", "unity"]
    }
  }
]

# =============================================================================
# Create Avatars
# =============================================================================

avatars =
  Enum.map(avatars_data, fn data ->
    personality = struct(Personality, data.personality)

    avatar =
      %Avatar{}
      |> Avatar.create_changeset(%{
        name: data.name,
        bio: data.bio,
        gender: data.gender,
        age: data.age,
        user_id: demo_user.id,
        personality: Map.from_struct(personality)
      })
      |> Repo.insert!()

    enneagram = Viva.Avatars.Enneagram.get_type(data.personality.enneagram_type)
    temperament = Personality.temperament(personality)

    IO.puts("Created avatar: #{avatar.name} (Type #{enneagram.number} - #{enneagram.name}, #{temperament})")

    avatar
  end)

# =============================================================================
# Summary
# =============================================================================

IO.puts("")
IO.puts("=" |> String.duplicate(60))
IO.puts("SEED COMPLETE")
IO.puts("=" |> String.duplicate(60))
IO.puts("")
IO.puts("Demo credentials:")
IO.puts("  Email: demo@viva.ai")
IO.puts("  Password: Demo123456!")
IO.puts("")
IO.puts("Created #{length(avatars)} avatars:")

Enum.each(avatars, fn avatar ->
  avatar = Repo.preload(avatar, [])
  enneagram = Viva.Avatars.Enneagram.get_type(avatar.personality.enneagram_type)
  temperament = Personality.temperament(avatar.personality)

  IO.puts("  - #{avatar.name}: #{enneagram.name} (#{temperament})")
end)

IO.puts("")
IO.puts("Run 'iex -S mix phx.server' to start the application!")
