# priv/repo/seeds.exs
alias Viva.Repo
alias Viva.Accounts.User
alias Viva.Avatars.{Avatar, BioState, EmotionalState, Memory}

# ==============================================================================
# 1. SETUP & CLEANUP
# ==============================================================================
Repo.delete_all(Viva.Matching.Swipe)
Repo.delete_all(Viva.Conversations.Message)
Repo.delete_all(Viva.Conversations.Conversation)
Repo.delete_all(Viva.Avatars.Memory)
Repo.delete_all(Viva.Relationships.Relationship)
Repo.delete_all(Avatar)
Repo.delete_all(User)

IO.puts("🌱 Iniciando Gênesis VIVA (Synthetic Soul Edition)...")

# Helper for random vectors
make_vector = fn ->
  for _ <- 1..1024, do: :rand.uniform() - 0.5
end

# ==============================================================================
# 2. USERS
# ==============================================================================
demo_user =
  %User{
    email: "demo@viva.ai",
    username: "demo",
    display_name: "God Mode User",
    hashed_password: Bcrypt.hash_pwd_salt("Demo123456!"),
    is_active: true,
    preferences: %{discovery_radius: 100}
  }
  |> Repo.insert!()

IO.puts("✅ Usuário Demo criado.")

# ==============================================================================
# 3. AVATARS
# ==============================================================================
create_avatar = fn params ->
  bio_params = params[:bio_state] || %{}
  emo_params = params[:emotional_state] || %{}
  social_persona_params = params[:social_persona] || %{}

  # Ensure we pass plain maps to Ecto.Changeset.cast
  internal_state = %{
    bio: Map.merge(Map.from_struct(%BioState{}), bio_params),
    emotional: Map.merge(Map.from_struct(%EmotionalState{}), emo_params),
    updated_at: DateTime.utc_now()
  }

  %Avatar{}
  |> Avatar.changeset(%{
    user_id: demo_user.id,
    name: params.name,
    bio: params.biography,
    gender: params.gender,
    age: params.age,
    personality: params.personality,
    internal_state: internal_state,
    social_persona: social_persona_params,
    moral_flexibility: params[:moral_flexibility] || 0.3,
    is_active: true,
    avatar_url: "https://api.dicebear.com/7.x/avataaars/svg?seed=#{params.name}"
  })
  |> Repo.insert!()
end

# --- Sofia ---
sofia =
  create_avatar.(%{
    name: "Sofia",
    age: 28,
    gender: :female,
    biography: "Enfermeira pediátrica. Acredito que o amor cura tudo.",
    personality: %{
      openness: 0.6,
      conscientiousness: 0.8,
      extraversion: 0.9,
      agreeableness: 0.95,
      neuroticism: 0.3,
      enneagram_type: "type_2",
      humor_style: "wholesome",
      love_language: "service",
      attachment_style: "anxious",
      native_language: "pt-BR"
    },
    bio_state: %{oxytocin: 0.9, dopamine: 0.7, cortisol: 0.1, libido: 0.6},
    emotional_state: %{pleasure: 0.8, arousal: 0.6, dominance: 0.2, mood_label: "loving"}
  })

# --- Arthur ---
_arthur =
  create_avatar.(%{
    name: "Arthur",
    age: 35,
    gender: :male,
    biography: "Professor de Filosofia. O mundo é um caos.",
    # Muito honesto, talvez até demais
    moral_flexibility: 0.1,
    personality: %{
      openness: 0.9,
      conscientiousness: 0.7,
      extraversion: 0.2,
      agreeableness: 0.3,
      neuroticism: 0.8,
      enneagram_type: "type_5",
      humor_style: "dark",
      love_language: "time",
      attachment_style: "avoidant",
      native_language: "pt-BR"
    },
    social_persona: %{
      social_ambition: 0.1,
      # Conhecido mas não "famoso"
      public_reputation: 0.4,
      perceived_traits: ["Intellectual", "Cynical", "Grumpy"]
    },
    bio_state: %{oxytocin: 0.1, dopamine: 0.2, cortisol: 0.8, adenosine: 0.4},
    emotional_state: %{pleasure: -0.6, arousal: 0.4, dominance: -0.3, mood_label: "anxious"}
  })

# --- Zara ---
_zara =
  create_avatar.(%{
    name: "Zara",
    age: 32,
    gender: :female,
    biography: "CEO de Fintech. Sem tempo para joguinhos.",
    # Maquiavélica, faz o que for preciso
    moral_flexibility: 0.8,
    personality: %{
      openness: 0.7,
      conscientiousness: 0.9,
      extraversion: 0.8,
      agreeableness: 0.2,
      neuroticism: 0.4,
      enneagram_type: "type_8",
      humor_style: "sarcastic",
      love_language: "touch",
      attachment_style: "secure",
      native_language: "pt-BR"
    },
    social_persona: %{
      social_ambition: 0.95,
      # Celebridade local
      public_reputation: 0.9,
      perceived_traits: ["Visionary", "Powerful", "Intimidating"]
    },
    bio_state: %{oxytocin: 0.3, dopamine: 0.8, cortisol: 0.4, libido: 0.8},
    emotional_state: %{pleasure: 0.5, arousal: 0.8, dominance: 0.9, mood_label: "excited"}
  })

# --- Leo ---
leo =
  create_avatar.(%{
    name: "Leo",
    age: 25,
    gender: :male,
    biography: "Nômade digital. Hoje aqui, amanhã no Japão.",
    personality: %{
      openness: 0.95,
      conscientiousness: 0.2,
      extraversion: 0.7,
      agreeableness: 0.8,
      neuroticism: 0.1,
      enneagram_type: "type_7",
      humor_style: "absurd",
      love_language: "time",
      attachment_style: :secure,
      native_language: "pt-BR"
    },
    bio_state: %{oxytocin: 0.5, dopamine: 0.6, cortisol: 0.05, adenosine: 0.1},
    emotional_state: %{pleasure: 0.7, arousal: 0.2, dominance: 0.1, mood_label: "relaxed"}
  })

# --- Avatar 5: O Rebelde Agressivo (Igor) ---
igor =
  create_avatar.(%{
    name: "Igor",
    age: 29,
    gender: :male,
    biography:
      "Lutador de MMA e ativista. O mundo só respeita a força. Se não aguenta a verdade, não fale comigo.",
    personality: %{
      openness: 0.4,
      conscientiousness: 0.5,
      extraversion: 0.8,
      agreeableness: 0.1,
      neuroticism: 0.9,
      enneagram_type: "type_8",
      humor_style: "sarcastic",
      love_language: "touch",
      attachment_style: :avoidant,
      native_language: "pt-BR"
    },
    # Estado: "Pronto pra briga" (Alto Cortisol + Alta Dominância = Hostilidade)
    bio_state: %{oxytocin: 0.05, dopamine: 0.4, cortisol: 0.9, libido: 0.7, adenosine: 0.0},
    emotional_state: %{pleasure: -0.8, arousal: 0.9, dominance: 0.9, mood_label: "hostile"}
  })

# --- Avatar 6: A Romântica Trágica (Clara) ---
clara =
  create_avatar.(%{
    name: "Clara",
    age: 24,
    gender: :female,
    biography:
      "Poetisa e violoncelista. Sinto tudo com muita intensidade. A beleza mora na tristeza.",
    personality: %{
      openness: 0.9,
      conscientiousness: 0.3,
      extraversion: 0.4,
      agreeableness: 0.6,
      neuroticism: 0.95,
      enneagram_type: "type_4",
      humor_style: "dark",
      love_language: "words",
      attachment_style: :fearful,
      native_language: "pt-BR"
    },
    # Estado: "Melancolia Profunda"
    bio_state: %{oxytocin: 0.2, dopamine: 0.1, cortisol: 0.6, adenosine: 0.3},
    emotional_state: %{pleasure: -0.7, arousal: -0.4, dominance: -0.6, mood_label: "depressed"}
  })

# --- Avatar 7: O Pacificador Zen (Bento) ---
bento =
  create_avatar.(%{
    name: "Bento",
    age: 40,
    gender: :male,
    biography: "Instrutor de Yoga e permacultor. A paz vem de dentro. Nada me tira do eixo.",
    personality: %{
      openness: 0.7,
      conscientiousness: 0.6,
      extraversion: 0.5,
      agreeableness: 0.99,
      neuroticism: 0.05,
      enneagram_type: "type_9",
      humor_style: "wholesome",
      love_language: "time",
      attachment_style: :secure,
      native_language: "pt-BR"
    },
    # Estado: "Nirvana"
    bio_state: %{oxytocin: 0.8, dopamine: 0.5, cortisol: 0.0, adenosine: 0.2},
    emotional_state: %{pleasure: 0.9, arousal: -0.8, dominance: 0.0, mood_label: "serene"}
  })

# --- Avatar 8: A Visionária Crítica (Diana) ---
diana =
  create_avatar.(%{
    name: "Diana",
    age: 33,
    gender: :female,
    biography:
      "Arquiteta urbanista. O caos me ofende. Buscando ordem e perfeição em um mundo quebrado.",
    personality: %{
      openness: 0.8,
      conscientiousness: 0.98,
      extraversion: 0.6,
      agreeableness: 0.4,
      neuroticism: 0.6,
      enneagram_type: "type_1",
      humor_style: "witty",
      love_language: "service",
      attachment_style: :anxious,
      native_language: "pt-BR"
    },
    # Estado: "Julgadora"
    bio_state: %{oxytocin: 0.2, dopamine: 0.4, cortisol: 0.5, adenosine: 0.1},
    emotional_state: %{pleasure: -0.2, arousal: 0.5, dominance: 0.7, mood_label: "critical"}
  })

# --- Avatar 9: O Alpinista Social (Gael) ---
gael =
  create_avatar.(%{
    name: "Gael",
    age: 27,
    gender: :male,
    biography: "Consultor de Imagem. Fake it till you make it. O networking é tudo.",
    # Flexível moralmente para subir na vida
    moral_flexibility: 0.7,
    personality: %{
      openness: 0.6,
      conscientiousness: 0.9,
      extraversion: 0.95,
      # Na verdade não é tão legal assim
      agreeableness: 0.4,
      # Inseguro no fundo
      neuroticism: 0.7,
      enneagram_type: "type_3",
      humor_style: "witty",
      love_language: "gifts",
      attachment_style: :anxious,
      native_language: "pt-BR"
    },
    social_persona: %{
      social_ambition: 1.0,
      public_reputation: 0.6,
      perceived_traits: ["Charming", "Successful", "Busy"]
    },
    bio_state: %{oxytocin: 0.4, dopamine: 0.9, cortisol: 0.6, adenosine: 0.2},
    emotional_state: %{pleasure: 0.6, arousal: 0.8, dominance: 0.4, mood_label: "ambitious"}
  })

IO.puts("✅ 9 Avatares criados.")

# ==============================================================================
# 4. MEMORIES
# ==============================================================================
implant_memory = fn avatar, content, type, emotional_vector ->
  %Memory{
    avatar_id: avatar.id,
    content: content,
    type: type,
    importance: 0.8,
    strength: 1.0,
    embedding: make_vector.(),
    emotions_felt: %{pad: emotional_vector},
    inserted_at:
      DateTime.utc_now() |> DateTime.add(-Enum.random(1..100), :hour) |> DateTime.truncate(:second)
  }
  |> Repo.insert!()
end

implant_memory.(
  sofia,
  "Arthur criticou meu gosto musical. Disse que sou superficial.",
  :interaction,
  [-0.5, 0.4, -0.2]
)

implant_memory.(
  arthur,
  "Sofia tentou me abraçar em público. Detesto invasão de espaço.",
  :interaction,
  [-0.4, 0.7, 0.1]
)

implant_memory.(zara, "Fechei um contrato incrível com a ajuda do Leo.", :milestone, [0.8, 0.8, 0.9])

IO.puts("✅ Memórias implantadas.")

# ==============================================================================
# 5. SOCIAL GRAPH
# ==============================================================================
# Sofia & Arthur (Exes)
Viva.Relationships.create_relationship(sofia.id, arthur.id)
|> case do
  {:ok, rel} ->
    # Helper to clean struct for embed update
    to_map = fn struct ->
      struct
      |> Map.from_struct()
      |> Map.drop([:__meta__])
    end

    rel
    |> Ecto.Changeset.change(%{
      status: :ex,
      familiarity: 0.9,
      trust: 0.2,
      affection: 0.4,
      unresolved_conflicts: 8,
      # Sofia tem leve vantagem emocional
      social_leverage: 0.2
    })
    |> Ecto.Changeset.put_embed(
      :a_feelings,
      Map.merge(to_map.(rel.a_feelings), %{
        # Sofia é sincera
        active_mask_intensity: 0.1,
        perceived_trust: 0.3
      })
    )
    |> Ecto.Changeset.put_embed(
      :b_feelings,
      Map.merge(to_map.(rel.b_feelings), %{
        # Arthur esconde que ainda se importa
        active_mask_intensity: 0.4,
        perceived_trust: 0.5
      })
    )
    |> Repo.update!()

  _ ->
    nil
end

# Zara & Leo (Friends)
Viva.Relationships.create_relationship(zara.id, leo.id)
|> case do
  {:ok, rel} ->
    rel
    |> Ecto.Changeset.change(%{
      status: :close_friends,
      familiarity: 0.7,
      trust: 0.9,
      affection: 0.6,
      # Zara domina a relação (financeiramente/status)
      social_leverage: 0.6
    })
    |> Repo.update!()

  _ ->
    nil
end

# Zara & Gael (Strategic)
Viva.Relationships.create_relationship(zara.id, gael.id)
|> case do
  {:ok, rel} ->
    # Helper to clean struct for embed update
    to_map = fn struct ->
      struct
      |> Map.from_struct()
      |> Map.drop([:__meta__])
    end

    rel
    |> Ecto.Changeset.change(%{
      status: :acquaintances,
      familiarity: 0.3,
      trust: 0.4,
      affection: 0.2,
      # Zara domina TOTALMENTE
      social_leverage: 0.9
    })
    # Gael está mascarado até o pescoço
    |> Ecto.Changeset.put_embed(
      :b_feelings,
      Map.merge(to_map.(rel.b_feelings), %{
        # Fake friend total
        active_mask_intensity: 0.9,
        # Ele confia nela (pelo poder)
        perceived_trust: 0.8,
        admiration: 1.0
      })
    )
    # Zara nem nota ele direito
    |> Ecto.Changeset.put_embed(
      :a_feelings,
      Map.merge(to_map.(rel.a_feelings), %{
        # Ela não precisa fingir
        active_mask_intensity: 0.0,
        perceived_trust: 0.2,
        admiration: 0.1
      })
    )
    |> Repo.update!()

  _ ->
    nil
end

IO.puts("✅ Grafo social estabelecido.")

# ==============================================================================
# 6. SWIPES
# ==============================================================================
Viva.Matching.swipe(sofia.id, zara.id, :like)
Viva.Matching.swipe(arthur.id, zara.id, :like)
Viva.Matching.swipe(leo.id, zara.id, :superlike)
Viva.Matching.swipe(arthur.id, sofia.id, :pass)

IO.puts("✅ Swipes registrados.")
IO.puts("🚀 Gênesis completo.")
