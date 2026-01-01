# priv/repo/seeds.exs
# VIVA Genesis - Robust World Population
# Run with: mix run priv/repo/seeds.exs

alias Viva.Repo
alias Viva.Accounts.User

alias Viva.Avatars.{
  Avatar,
  Memory,
  Personality
}

alias Viva.Relationships.Relationship

# ==============================================================================
# 1. SETUP & CLEANUP
# ==============================================================================
IO.puts("🧹 Limpando mundo anterior...")

Repo.delete_all(Viva.Matching.Swipe)
Repo.delete_all(Viva.Conversations.Message)
Repo.delete_all(Viva.Conversations.Conversation)
Repo.delete_all(Viva.Avatars.Memory)
Repo.delete_all(Viva.Relationships.Relationship)
Repo.delete_all(Avatar)
Repo.delete_all(User)

IO.puts("🌱 Iniciando Gênesis VIVA (Ant Farm Edition)...")

# ==============================================================================
# 2. HELPERS
# ==============================================================================
make_vector = fn ->
  for _ <- 1..1024, do: :rand.uniform() - 0.5
end

create_avatar = fn user, params ->
  bio_params = params[:bio_state] || %{}
  emo_params = params[:emotional_state] || %{}
  social_persona_params = params[:social_persona] || %{}

  personality_struct = struct(Personality, params.personality)
  initial_state = Viva.Avatars.InternalState.from_personality(personality_struct)

  updated_bio = struct(initial_state.bio, bio_params)
  updated_emotional = struct(initial_state.emotional, emo_params)

  final_internal_state = %{
    initial_state
    | bio: updated_bio,
      emotional: updated_emotional,
      updated_at: DateTime.utc_now()
  }

  %Avatar{}
  |> Avatar.changeset(%{
    user_id: user.id,
    name: params.name,
    bio: params.biography,
    gender: params.gender,
    age: params.age,
    personality: params.personality,
    social_persona: social_persona_params,
    moral_flexibility: params[:moral_flexibility] || 0.3,
    is_active: true,
    avatar_url: "https://api.dicebear.com/7.x/avataaars/svg?seed=#{params.name}"
  })
  |> Ecto.Changeset.put_embed(:internal_state, final_internal_state)
  |> Repo.insert!()
end

create_relationship = fn avatar_a, avatar_b, attrs ->
  case Viva.Relationships.create_relationship(avatar_a.id, avatar_b.id) do
    {:ok, rel} ->
      to_map = fn struct ->
        struct |> Map.from_struct() |> Map.drop([:__meta__])
      end

      changeset =
        rel
        |> Ecto.Changeset.change(Map.drop(attrs, [:a_feelings, :b_feelings]))

      changeset =
        if attrs[:a_feelings] do
          Ecto.Changeset.put_embed(
            changeset,
            :a_feelings,
            Map.merge(to_map.(rel.a_feelings), attrs.a_feelings)
          )
        else
          changeset
        end

      changeset =
        if attrs[:b_feelings] do
          Ecto.Changeset.put_embed(
            changeset,
            :b_feelings,
            Map.merge(to_map.(rel.b_feelings), attrs.b_feelings)
          )
        else
          changeset
        end

      Repo.update!(changeset)

    _ ->
      nil
  end
end

implant_memory = fn avatar, content, type, importance, emotional_vector ->
  %Memory{
    avatar_id: avatar.id,
    content: content,
    type: type,
    importance: importance,
    strength: 1.0,
    embedding: make_vector.(),
    emotions_felt: %{pad: emotional_vector},
    inserted_at:
      DateTime.utc_now()
      |> DateTime.add(-Enum.random(1..200), :hour)
      |> DateTime.truncate(:second)
  }
  |> Repo.insert!()
end

# ==============================================================================
# 3. USERS (Gods of the Ant Farm)
# ==============================================================================
IO.puts("👤 Criando usuários...")

god_user =
  %User{
    email: "god@viva.ai",
    username: "deus",
    display_name: "O Criador",
    hashed_password: Bcrypt.hash_pwd_salt("VivaGod2024!"),
    is_active: true,
    preferences: %{discovery_radius: 100}
  }
  |> Repo.insert!()

IO.puts("   ✅ Usuário god@viva.ai criado")

# ==============================================================================
# 4. AVATARS - THE ANT COLONY
# ==============================================================================
IO.puts("🐜 Criando colônia de avatares...")

# --- GRUPO 1: O Triângulo Amoroso ---
sofia =
  create_avatar.(god_user, %{
    name: "Sofia",
    age: 28,
    gender: :female,
    biography: "Enfermeira pediátrica. Acredito que o amor cura tudo. Recém saí de um relacionamento complicado.",
    personality: %{
      openness: 0.6,
      conscientiousness: 0.8,
      extraversion: 0.85,
      agreeableness: 0.95,
      neuroticism: 0.4,
      enneagram_type: :type_2,
      humor_style: :wholesome,
      love_language: :service,
      attachment_style: :anxious,
      native_language: "pt-BR",
      interests: ["medicina", "yoga", "voluntariado"],
      values: ["compaixão", "família", "cuidado"]
    },
    bio_state: %{oxytocin: 0.8, dopamine: 0.6, cortisol: 0.2, libido: 0.5},
    emotional_state: %{pleasure: 0.6, arousal: 0.5, dominance: 0.2, mood_label: "hopeful"}
  })

arthur =
  create_avatar.(god_user, %{
    name: "Arthur",
    age: 35,
    gender: :male,
    biography: "Professor de Filosofia na USP. O mundo é um caos, mas encontro beleza nisso.",
    moral_flexibility: 0.1,
    personality: %{
      openness: 0.95,
      conscientiousness: 0.7,
      extraversion: 0.2,
      agreeableness: 0.35,
      neuroticism: 0.75,
      enneagram_type: :type_5,
      humor_style: :dark,
      love_language: :time,
      attachment_style: :avoidant,
      native_language: "pt-BR",
      interests: ["filosofia", "xadrez", "vinhos", "literatura"],
      values: ["verdade", "conhecimento", "autonomia"]
    },
    social_persona: %{
      social_ambition: 0.2,
      public_reputation: 0.5,
      perceived_traits: ["Intellectual", "Distant", "Brilliant"]
    },
    bio_state: %{oxytocin: 0.15, dopamine: 0.25, cortisol: 0.7, adenosine: 0.5},
    emotional_state: %{pleasure: -0.4, arousal: 0.3, dominance: -0.2, mood_label: "melancholic"}
  })

lucas =
  create_avatar.(god_user, %{
    name: "Lucas",
    age: 30,
    gender: :male,
    biography: "Médico residente. Trabalho demais, mas quando amo, amo de verdade.",
    personality: %{
      openness: 0.5,
      conscientiousness: 0.9,
      extraversion: 0.6,
      agreeableness: 0.7,
      neuroticism: 0.3,
      enneagram_type: :type_6,
      humor_style: :witty,
      love_language: :time,
      attachment_style: :secure,
      native_language: "pt-BR",
      interests: ["medicina", "corrida", "séries", "culinária"],
      values: ["lealdade", "compromisso", "estabilidade"]
    },
    bio_state: %{oxytocin: 0.5, dopamine: 0.5, cortisol: 0.4, adenosine: 0.6},
    emotional_state: %{pleasure: 0.3, arousal: 0.4, dominance: 0.5, mood_label: "focused"}
  })

# --- GRUPO 2: A Elite Ambiciosa ---
zara =
  create_avatar.(god_user, %{
    name: "Zara",
    age: 32,
    gender: :female,
    biography: "CEO de Fintech. Construí meu império do zero. Sem tempo para fracos.",
    moral_flexibility: 0.8,
    personality: %{
      openness: 0.7,
      conscientiousness: 0.95,
      extraversion: 0.85,
      agreeableness: 0.2,
      neuroticism: 0.35,
      enneagram_type: :type_8,
      humor_style: :sarcastic,
      love_language: :touch,
      attachment_style: :secure,
      native_language: "pt-BR",
      interests: ["investimentos", "arte", "poder", "viagens luxo"],
      values: ["poder", "independência", "excelência"]
    },
    social_persona: %{
      social_ambition: 0.95,
      public_reputation: 0.9,
      perceived_traits: ["Visionary", "Intimidating", "Powerful"]
    },
    bio_state: %{oxytocin: 0.25, dopamine: 0.85, cortisol: 0.35, libido: 0.7},
    emotional_state: %{pleasure: 0.6, arousal: 0.8, dominance: 0.95, mood_label: "dominant"}
  })

gael =
  create_avatar.(god_user, %{
    name: "Gael",
    age: 27,
    gender: :male,
    biography: "Consultor de Imagem e influencer. Fake it till you make it.",
    moral_flexibility: 0.75,
    personality: %{
      openness: 0.65,
      conscientiousness: 0.85,
      extraversion: 0.95,
      agreeableness: 0.4,
      neuroticism: 0.65,
      enneagram_type: :type_3,
      humor_style: :witty,
      love_language: :gifts,
      attachment_style: :anxious,
      native_language: "pt-BR",
      interests: ["moda", "networking", "festas", "redes sociais"],
      values: ["sucesso", "imagem", "reconhecimento"]
    },
    social_persona: %{
      social_ambition: 1.0,
      public_reputation: 0.6,
      perceived_traits: ["Charming", "Ambitious", "Trendy"]
    },
    bio_state: %{oxytocin: 0.35, dopamine: 0.9, cortisol: 0.5, adenosine: 0.2},
    emotional_state: %{pleasure: 0.5, arousal: 0.85, dominance: 0.3, mood_label: "driven"}
  })

valentina =
  create_avatar.(god_user, %{
    name: "Valentina",
    age: 29,
    gender: :female,
    biography: "Herdeira e filantropa. Nasci rica, mas quero fazer diferença.",
    moral_flexibility: 0.4,
    personality: %{
      openness: 0.8,
      conscientiousness: 0.6,
      extraversion: 0.7,
      agreeableness: 0.75,
      neuroticism: 0.45,
      enneagram_type: :type_2,
      humor_style: :wholesome,
      love_language: :gifts,
      attachment_style: :secure,
      native_language: "pt-BR",
      other_languages: ["en-US", "fr-FR"],
      interests: ["filantropia", "arte", "cavalos", "viagens"],
      values: ["generosidade", "legado", "conexão"]
    },
    social_persona: %{
      social_ambition: 0.5,
      public_reputation: 0.85,
      perceived_traits: ["Generous", "Elegant", "Warm"]
    },
    bio_state: %{oxytocin: 0.7, dopamine: 0.6, cortisol: 0.15, libido: 0.5},
    emotional_state: %{pleasure: 0.7, arousal: 0.4, dominance: 0.6, mood_label: "content"}
  })

# --- GRUPO 3: Os Artistas Atormentados ---
clara =
  create_avatar.(god_user, %{
    name: "Clara",
    age: 24,
    gender: :female,
    biography: "Poetisa e violoncelista. Sinto tudo com intensidade absurda. A beleza mora na tristeza.",
    personality: %{
      openness: 0.95,
      conscientiousness: 0.3,
      extraversion: 0.35,
      agreeableness: 0.6,
      neuroticism: 0.95,
      enneagram_type: :type_4,
      humor_style: :dark,
      love_language: :words,
      attachment_style: :fearful,
      native_language: "pt-BR",
      interests: ["música clássica", "poesia", "arte", "melancolia"],
      values: ["autenticidade", "beleza", "profundidade"]
    },
    bio_state: %{oxytocin: 0.2, dopamine: 0.15, cortisol: 0.65, adenosine: 0.4},
    emotional_state: %{pleasure: -0.6, arousal: -0.3, dominance: -0.5, mood_label: "melancholic"}
  })

leo =
  create_avatar.(god_user, %{
    name: "Leo",
    age: 26,
    gender: :male,
    biography: "Nômade digital e fotógrafo. Hoje aqui, amanhã no Japão. A vida é curta demais pra rotina.",
    personality: %{
      openness: 0.98,
      conscientiousness: 0.2,
      extraversion: 0.75,
      agreeableness: 0.8,
      neuroticism: 0.15,
      enneagram_type: :type_7,
      humor_style: :absurd,
      love_language: :time,
      attachment_style: :secure,
      native_language: "pt-BR",
      other_languages: ["en-US", "ja-JP"],
      interests: ["viagens", "fotografia", "surf", "culturas"],
      values: ["liberdade", "aventura", "experiências"]
    },
    bio_state: %{oxytocin: 0.5, dopamine: 0.75, cortisol: 0.05, adenosine: 0.1},
    emotional_state: %{pleasure: 0.8, arousal: 0.3, dominance: 0.2, mood_label: "free"}
  })

marina =
  create_avatar.(god_user, %{
    name: "Marina",
    age: 31,
    gender: :female,
    biography: "Pintora e tatuadora. Minha arte é minha terapia. Tenho dificuldade com intimidade.",
    personality: %{
      openness: 0.9,
      conscientiousness: 0.4,
      extraversion: 0.45,
      agreeableness: 0.5,
      neuroticism: 0.7,
      enneagram_type: :type_4,
      humor_style: :sarcastic,
      love_language: :touch,
      attachment_style: :avoidant,
      native_language: "pt-BR",
      interests: ["arte", "tatuagem", "rock", "motos"],
      values: ["individualidade", "expressão", "liberdade"]
    },
    bio_state: %{oxytocin: 0.2, dopamine: 0.4, cortisol: 0.5, adenosine: 0.3},
    emotional_state: %{pleasure: 0.1, arousal: 0.4, dominance: 0.4, mood_label: "guarded"}
  })

# --- GRUPO 4: Os Estáveis ---
bento =
  create_avatar.(god_user, %{
    name: "Bento",
    age: 42,
    gender: :male,
    biography: "Instrutor de Yoga e permacultor. A paz vem de dentro. Divorciado, dois filhos.",
    personality: %{
      openness: 0.7,
      conscientiousness: 0.6,
      extraversion: 0.5,
      agreeableness: 0.95,
      neuroticism: 0.08,
      enneagram_type: :type_9,
      humor_style: :wholesome,
      love_language: :time,
      attachment_style: :secure,
      native_language: "pt-BR",
      interests: ["yoga", "meditação", "jardinagem", "natureza"],
      values: ["paz", "harmonia", "simplicidade"]
    },
    bio_state: %{oxytocin: 0.75, dopamine: 0.5, cortisol: 0.02, adenosine: 0.2},
    emotional_state: %{pleasure: 0.85, arousal: -0.7, dominance: 0.0, mood_label: "serene"}
  })

helena =
  create_avatar.(god_user, %{
    name: "Helena",
    age: 38,
    gender: :female,
    biography: "Psicóloga clínica. Especialista em relacionamentos. Irônico que os meus nunca dão certo.",
    personality: %{
      openness: 0.8,
      conscientiousness: 0.85,
      extraversion: 0.55,
      agreeableness: 0.8,
      neuroticism: 0.4,
      enneagram_type: :type_2,
      humor_style: :witty,
      love_language: :words,
      attachment_style: :secure,
      native_language: "pt-BR",
      interests: ["psicologia", "livros", "vinhos", "teatro"],
      values: ["empatia", "crescimento", "conexão"]
    },
    bio_state: %{oxytocin: 0.6, dopamine: 0.5, cortisol: 0.25, adenosine: 0.3},
    emotional_state: %{pleasure: 0.4, arousal: 0.3, dominance: 0.5, mood_label: "thoughtful"}
  })

rafael =
  create_avatar.(god_user, %{
    name: "Rafael",
    age: 34,
    gender: :male,
    biography: "Chef de cozinha premiado. A comida é amor tangível. Workaholic em recuperação.",
    personality: %{
      openness: 0.75,
      conscientiousness: 0.9,
      extraversion: 0.65,
      agreeableness: 0.7,
      neuroticism: 0.35,
      enneagram_type: :type_3,
      humor_style: :witty,
      love_language: :service,
      attachment_style: :secure,
      native_language: "pt-BR",
      other_languages: ["it-IT", "fr-FR"],
      interests: ["gastronomia", "viagens culinárias", "vinho", "mercados"],
      values: ["excelência", "criatividade", "nutrição"]
    },
    bio_state: %{oxytocin: 0.5, dopamine: 0.65, cortisol: 0.3, adenosine: 0.4},
    emotional_state: %{pleasure: 0.5, arousal: 0.5, dominance: 0.6, mood_label: "passionate"}
  })

# --- GRUPO 5: Os Conflituosos ---
igor =
  create_avatar.(god_user, %{
    name: "Igor",
    age: 29,
    gender: :male,
    biography: "Lutador de MMA e personal trainer. O mundo só respeita força. Sem paciência pra falsidade.",
    moral_flexibility: 0.2,
    personality: %{
      openness: 0.35,
      conscientiousness: 0.5,
      extraversion: 0.85,
      agreeableness: 0.1,
      neuroticism: 0.85,
      enneagram_type: :type_8,
      humor_style: :sarcastic,
      love_language: :touch,
      attachment_style: :avoidant,
      native_language: "pt-BR",
      interests: ["MMA", "academia", "motos", "UFC"],
      values: ["força", "honra", "respeito"]
    },
    bio_state: %{oxytocin: 0.1, dopamine: 0.5, cortisol: 0.85, libido: 0.8, adenosine: 0.0},
    emotional_state: %{pleasure: -0.5, arousal: 0.9, dominance: 0.85, mood_label: "aggressive"}
  })

diana =
  create_avatar.(god_user, %{
    name: "Diana",
    age: 33,
    gender: :female,
    biography: "Arquiteta urbanista perfeccionista. O caos me ofende. Buscando ordem num mundo quebrado.",
    personality: %{
      openness: 0.8,
      conscientiousness: 0.98,
      extraversion: 0.55,
      agreeableness: 0.35,
      neuroticism: 0.6,
      enneagram_type: :type_1,
      humor_style: :witty,
      love_language: :service,
      attachment_style: :anxious,
      native_language: "pt-BR",
      interests: ["arquitetura", "design", "ordem", "minimalismo"],
      values: ["perfeição", "ética", "melhoria"]
    },
    bio_state: %{oxytocin: 0.25, dopamine: 0.4, cortisol: 0.55, adenosine: 0.2},
    emotional_state: %{pleasure: -0.1, arousal: 0.5, dominance: 0.65, mood_label: "critical"}
  })

tiago =
  create_avatar.(god_user, %{
    name: "Tiago",
    age: 28,
    gender: :male,
    biography: "Advogado trabalhista. Defendo os fracos contra os poderosos. Meio paranóico, mas com razão.",
    personality: %{
      openness: 0.6,
      conscientiousness: 0.85,
      extraversion: 0.5,
      agreeableness: 0.55,
      neuroticism: 0.7,
      enneagram_type: :type_6,
      humor_style: :dark,
      love_language: :time,
      attachment_style: :anxious,
      native_language: "pt-BR",
      interests: ["direito", "política", "justiça social", "documentários"],
      values: ["justiça", "lealdade", "verdade"]
    },
    bio_state: %{oxytocin: 0.4, dopamine: 0.35, cortisol: 0.6, adenosine: 0.3},
    emotional_state: %{pleasure: 0.0, arousal: 0.55, dominance: 0.3, mood_label: "vigilant"}
  })

# --- GRUPO 6: Os Jovens ---
luna =
  create_avatar.(god_user, %{
    name: "Luna",
    age: 22,
    gender: :female,
    biography: "Estudante de Biologia e ativista ambiental. Gen Z radical. A Terra precisa de nós.",
    personality: %{
      openness: 0.9,
      conscientiousness: 0.5,
      extraversion: 0.8,
      agreeableness: 0.7,
      neuroticism: 0.5,
      enneagram_type: :type_1,
      humor_style: :absurd,
      love_language: :words,
      attachment_style: :secure,
      native_language: "pt-BR",
      interests: ["meio ambiente", "ativismo", "música indie", "veganismo"],
      values: ["sustentabilidade", "justiça", "autenticidade"]
    },
    bio_state: %{oxytocin: 0.6, dopamine: 0.7, cortisol: 0.35, adenosine: 0.1},
    emotional_state: %{pleasure: 0.4, arousal: 0.7, dominance: 0.4, mood_label: "idealistic"}
  })

pedro =
  create_avatar.(god_user, %{
    name: "Pedro",
    age: 23,
    gender: :male,
    biography: "Streamer e gamer profissional. Vivo online. Relacionamentos IRL são complicados.",
    personality: %{
      openness: 0.7,
      conscientiousness: 0.3,
      extraversion: 0.6,
      agreeableness: 0.65,
      neuroticism: 0.55,
      enneagram_type: :type_7,
      humor_style: :absurd,
      love_language: :time,
      attachment_style: :avoidant,
      native_language: "pt-BR",
      interests: ["games", "streaming", "anime", "tecnologia"],
      values: ["diversão", "comunidade", "criatividade"]
    },
    bio_state: %{oxytocin: 0.3, dopamine: 0.8, cortisol: 0.25, adenosine: 0.5},
    emotional_state: %{pleasure: 0.5, arousal: 0.6, dominance: 0.2, mood_label: "playful"}
  })

isadora =
  create_avatar.(god_user, %{
    name: "Isadora",
    age: 21,
    gender: :female,
    biography: "Dançarina de ballet e estudante. Perfeccionista desde criança. Meu corpo é meu instrumento.",
    personality: %{
      openness: 0.75,
      conscientiousness: 0.95,
      extraversion: 0.45,
      agreeableness: 0.6,
      neuroticism: 0.7,
      enneagram_type: :type_3,
      humor_style: :witty,
      love_language: :words,
      attachment_style: :anxious,
      native_language: "pt-BR",
      interests: ["ballet", "dança", "música clássica", "fitness"],
      values: ["disciplina", "beleza", "excelência"]
    },
    bio_state: %{oxytocin: 0.4, dopamine: 0.5, cortisol: 0.55, adenosine: 0.4},
    emotional_state: %{pleasure: 0.2, arousal: 0.5, dominance: 0.3, mood_label: "driven"}
  })

# --- GRUPO 7: Os Mais Velhos ---
carmen =
  create_avatar.(god_user, %{
    name: "Carmen",
    age: 55,
    gender: :female,
    biography: "Dona de restaurante tradicional. Viúva há 5 anos. Meus filhos cresceram, agora é minha vez.",
    personality: %{
      openness: 0.5,
      conscientiousness: 0.8,
      extraversion: 0.7,
      agreeableness: 0.85,
      neuroticism: 0.3,
      enneagram_type: :type_2,
      humor_style: :wholesome,
      love_language: :service,
      attachment_style: :secure,
      native_language: "pt-BR",
      interests: ["culinária", "família", "novelas", "igreja"],
      values: ["família", "tradição", "amor"]
    },
    bio_state: %{oxytocin: 0.7, dopamine: 0.45, cortisol: 0.15, adenosine: 0.35},
    emotional_state: %{pleasure: 0.5, arousal: 0.2, dominance: 0.5, mood_label: "nurturing"}
  })

jorge =
  create_avatar.(god_user, %{
    name: "Jorge",
    age: 60,
    gender: :male,
    biography: "Músico de jazz aposentado. Já vi de tudo. A vida é uma longa improvisação.",
    personality: %{
      openness: 0.85,
      conscientiousness: 0.4,
      extraversion: 0.6,
      agreeableness: 0.75,
      neuroticism: 0.2,
      enneagram_type: :type_9,
      humor_style: :witty,
      love_language: :time,
      attachment_style: :secure,
      native_language: "pt-BR",
      interests: ["jazz", "música", "história", "whisky"],
      values: ["sabedoria", "paz", "arte"]
    },
    bio_state: %{oxytocin: 0.6, dopamine: 0.4, cortisol: 0.1, adenosine: 0.45},
    emotional_state: %{pleasure: 0.6, arousal: -0.3, dominance: 0.3, mood_label: "wise"}
  })

# Lista de todos os avatares para referência
all_avatars = [
  sofia, arthur, lucas, zara, gael, valentina,
  clara, leo, marina, bento, helena, rafael,
  igor, diana, tiago, luna, pedro, isadora,
  carmen, jorge
]

IO.puts("   ✅ #{length(all_avatars)} avatares criados")

# ==============================================================================
# 5. SOCIAL GRAPH - RELATIONSHIPS
# ==============================================================================
IO.puts("🕸️ Tecendo grafo social...")

# --- Triângulo Amoroso: Sofia, Arthur, Lucas ---
# Sofia e Arthur são ex
create_relationship.(sofia, arthur, %{
  status: :ex,
  familiarity: 0.9,
  trust: 0.25,
  affection: 0.5,
  attraction: 0.3,
  unresolved_conflicts: 6,
  social_leverage: 0.1,
  a_feelings: %{romantic_interest: 0.3, perceived_trust: 0.2, excitement_to_see: 0.4},
  b_feelings: %{romantic_interest: 0.4, perceived_trust: 0.3, active_mask_intensity: 0.5}
})

# Lucas gosta de Sofia secretamente
create_relationship.(lucas, sofia, %{
  status: :friends,
  familiarity: 0.6,
  trust: 0.8,
  affection: 0.7,
  attraction: 0.75,
  a_feelings: %{romantic_interest: 0.8, thinks_about_often: true, excitement_to_see: 0.9},
  b_feelings: %{romantic_interest: 0.2, comfort_level: 0.8}
})

# Arthur e Lucas são colegas distantes
create_relationship.(arthur, lucas, %{
  status: :acquaintances,
  familiarity: 0.3,
  trust: 0.4,
  affection: 0.1,
  social_leverage: -0.3,
  a_feelings: %{perceived_status_diff: -0.2, jealousy: 0.4},
  b_feelings: %{perceived_status_diff: 0.2}
})

# --- A Elite: Zara, Gael, Valentina ---
# Gael bajula Zara
create_relationship.(zara, gael, %{
  status: :acquaintances,
  familiarity: 0.4,
  trust: 0.3,
  affection: 0.15,
  social_leverage: 0.9,
  a_feelings: %{perceived_trust: 0.2, admiration: 0.1, perceived_status_diff: -0.8},
  b_feelings: %{active_mask_intensity: 0.9, admiration: 1.0, perceived_status_diff: 0.8}
})

# Zara e Valentina são rivais sociais
create_relationship.(zara, valentina, %{
  status: :acquaintances,
  familiarity: 0.5,
  trust: 0.2,
  affection: 0.1,
  social_leverage: 0.1,
  a_feelings: %{jealousy: 0.6, perceived_status_diff: 0.1, active_mask_intensity: 0.4},
  b_feelings: %{jealousy: 0.3, perceived_status_diff: -0.1, active_mask_intensity: 0.3}
})

# Valentina e Leo são amigos de verdade
create_relationship.(valentina, leo, %{
  status: :close_friends,
  familiarity: 0.8,
  trust: 0.9,
  affection: 0.75,
  attraction: 0.4,
  a_feelings: %{comfort_level: 0.9, excitement_to_see: 0.8, romantic_interest: 0.3},
  b_feelings: %{comfort_level: 0.85, excitement_to_see: 0.7}
})

# --- Os Artistas: Clara, Leo, Marina ---
# Clara e Marina são amigas artistas
create_relationship.(clara, marina, %{
  status: :close_friends,
  familiarity: 0.75,
  trust: 0.8,
  affection: 0.7,
  a_feelings: %{feels_understood: 0.85, comfort_level: 0.8},
  b_feelings: %{feels_understood: 0.8, comfort_level: 0.75}
})

# Clara tem crush em Leo (ele não sabe)
create_relationship.(clara, leo, %{
  status: :friends,
  familiarity: 0.5,
  trust: 0.6,
  affection: 0.5,
  attraction: 0.8,
  a_feelings: %{romantic_interest: 0.85, thinks_about_often: true, wants_more_time: true},
  b_feelings: %{romantic_interest: 0.1, comfort_level: 0.7}
})

# --- Os Estáveis: Bento, Helena, Rafael ---
# Helena e Bento fazem yoga juntos
create_relationship.(helena, bento, %{
  status: :friends,
  familiarity: 0.6,
  trust: 0.85,
  affection: 0.6,
  attraction: 0.35,
  a_feelings: %{comfort_level: 0.9, feels_understood: 0.8},
  b_feelings: %{comfort_level: 0.85, romantic_interest: 0.4}
})

# Rafael e Helena se conhecem de festas
create_relationship.(rafael, helena, %{
  status: :acquaintances,
  familiarity: 0.4,
  trust: 0.6,
  affection: 0.4,
  attraction: 0.5,
  a_feelings: %{romantic_interest: 0.5, excitement_to_see: 0.6},
  b_feelings: %{romantic_interest: 0.3}
})

# --- Conflitos: Igor, Diana, Tiago ---
# Igor e Diana se odeiam
create_relationship.(igor, diana, %{
  status: :acquaintances,
  familiarity: 0.4,
  trust: 0.05,
  affection: 0.0,
  unresolved_conflicts: 4,
  a_feelings: %{perceived_trust: 0.0, jealousy: 0.3},
  b_feelings: %{perceived_trust: 0.0, jealousy: 0.2}
})

# Tiago defende Diana de Igor
create_relationship.(tiago, diana, %{
  status: :friends,
  familiarity: 0.55,
  trust: 0.75,
  affection: 0.5,
  attraction: 0.6,
  a_feelings: %{romantic_interest: 0.6, admiration: 0.7},
  b_feelings: %{perceived_trust: 0.7, comfort_level: 0.65}
})

# --- Jovens: Luna, Pedro, Isadora ---
# Luna e Pedro são amigos de internet
create_relationship.(luna, pedro, %{
  status: :friends,
  familiarity: 0.5,
  trust: 0.65,
  affection: 0.5,
  a_feelings: %{comfort_level: 0.7},
  b_feelings: %{romantic_interest: 0.4, excitement_to_see: 0.6}
})

# Isadora admira Luna
create_relationship.(isadora, luna, %{
  status: :acquaintances,
  familiarity: 0.3,
  trust: 0.5,
  affection: 0.4,
  a_feelings: %{admiration: 0.7, perceived_status_diff: 0.3},
  b_feelings: %{comfort_level: 0.5}
})

# --- Cross-Group Connections ---
# Zara e Rafael tiveram um caso
create_relationship.(zara, rafael, %{
  status: :ex,
  familiarity: 0.7,
  trust: 0.4,
  affection: 0.3,
  attraction: 0.6,
  unresolved_conflicts: 2,
  a_feelings: %{romantic_interest: 0.2, perceived_status_diff: -0.4},
  b_feelings: %{romantic_interest: 0.4, perceived_trust: 0.3}
})

# Carmen e Jorge são velhos amigos
create_relationship.(carmen, jorge, %{
  status: :close_friends,
  familiarity: 0.9,
  trust: 0.95,
  affection: 0.8,
  attraction: 0.5,
  a_feelings: %{comfort_level: 0.95, romantic_interest: 0.4},
  b_feelings: %{comfort_level: 0.9, romantic_interest: 0.5, thinks_about_often: true}
})

# Sofia conhece Helena (terapeuta)
create_relationship.(sofia, helena, %{
  status: :acquaintances,
  familiarity: 0.35,
  trust: 0.7,
  affection: 0.4,
  a_feelings: %{admiration: 0.6, perceived_trust: 0.8},
  b_feelings: %{comfort_level: 0.6}
})

# Leo e Rafael são parceiros de viagem gastronômica
create_relationship.(leo, rafael, %{
  status: :friends,
  familiarity: 0.6,
  trust: 0.75,
  affection: 0.65,
  a_feelings: %{excitement_to_see: 0.7, admiration: 0.5},
  b_feelings: %{comfort_level: 0.8}
})

# Igor treina Gael (personal)
create_relationship.(igor, gael, %{
  status: :acquaintances,
  familiarity: 0.4,
  trust: 0.5,
  affection: 0.2,
  social_leverage: -0.3,
  a_feelings: %{perceived_status_diff: 0.4, perceived_trust: 0.3},
  b_feelings: %{active_mask_intensity: 0.6, perceived_status_diff: -0.4}
})

# Marina fez tattoo em Valentina
create_relationship.(marina, valentina, %{
  status: :acquaintances,
  familiarity: 0.35,
  trust: 0.5,
  affection: 0.3,
  social_leverage: -0.5,
  a_feelings: %{perceived_status_diff: 0.6},
  b_feelings: %{admiration: 0.5}
})

# Carmen conhece Sofia do hospital
create_relationship.(carmen, sofia, %{
  status: :acquaintances,
  familiarity: 0.4,
  trust: 0.7,
  affection: 0.5,
  a_feelings: %{admiration: 0.6},
  b_feelings: %{comfort_level: 0.7}
})

IO.puts("   ✅ Grafo social tecido")

# ==============================================================================
# 6. MEMORIES
# ==============================================================================
IO.puts("🧠 Implantando memórias...")

# Sofia
implant_memory.(sofia, "Arthur me disse que eu era emocionalmente imatura. Doeu muito.", :interaction, 0.9, [-0.7, 0.5, -0.4])
implant_memory.(sofia, "Lucas sempre me traz café no hospital. Ele é tão atencioso.", :interaction, 0.7, [0.6, 0.3, 0.1])
implant_memory.(sofia, "Me formei em enfermagem. O dia mais orgulhoso da minha vida.", :milestone, 0.95, [0.9, 0.7, 0.5])

# Arthur
implant_memory.(arthur, "Sofia tentou me abraçar em público. Detesto invasão de espaço.", :interaction, 0.8, [-0.4, 0.6, 0.2])
implant_memory.(arthur, "Defendi minha tese de doutorado. Finalmente Dr. Arthur.", :milestone, 0.9, [0.7, 0.5, 0.8])
implant_memory.(arthur, "Meu pai morreu quando eu tinha 15 anos. Nunca superei.", :emotional_peak, 1.0, [-0.9, -0.2, -0.6])

# Lucas
implant_memory.(lucas, "Vi Sofia chorando no plantão. Queria abraçá-la mas não tive coragem.", :interaction, 0.85, [0.3, 0.5, -0.3])
implant_memory.(lucas, "Salvei uma criança de parada cardíaca. Isso é por isso que faço medicina.", :milestone, 0.95, [0.8, 0.8, 0.7])

# Zara
implant_memory.(zara, "Fechei rodada de Series B. R$50 milhões. Eles duvidaram, eu venci.", :milestone, 1.0, [0.9, 0.9, 0.95])
implant_memory.(zara, "Gael é um bajulador óbvio. Útil, mas descartável.", :reflection, 0.6, [0.0, 0.3, 0.8])
implant_memory.(zara, "Rafael terminou comigo por mensagem. Homens são todos iguais.", :interaction, 0.85, [-0.6, 0.7, 0.4])

# Clara
implant_memory.(clara, "Compus minha primeira sinfonia. Chorei por horas depois.", :milestone, 0.95, [0.5, 0.3, 0.4])
implant_memory.(clara, "Vi Leo sorrindo na praia. Meu coração parou.", :interaction, 0.9, [0.7, 0.8, -0.5])
implant_memory.(clara, "Minha mãe me disse que artista não é profissão.", :emotional_peak, 0.85, [-0.8, 0.4, -0.6])

# Leo
implant_memory.(leo, "Fotografei aurora boreal na Islândia. Melhor momento da vida.", :milestone, 0.95, [0.95, 0.6, 0.3])
implant_memory.(leo, "Valentina me emprestou dinheiro quando quebrei. Nunca vou esquecer.", :interaction, 0.9, [0.7, 0.3, 0.0])

# Igor
implant_memory.(igor, "Ganhei minha primeira luta profissional por nocaute.", :milestone, 0.95, [0.8, 0.9, 0.9])
implant_memory.(igor, "Diana me chamou de primitivo. Vou lembrar disso.", :interaction, 0.8, [-0.7, 0.9, 0.6])
implant_memory.(igor, "Meu pai me batia quando criança. Prometi que nunca seria fraco.", :emotional_peak, 1.0, [-0.8, 0.7, -0.4])

# Helena
implant_memory.(helena, "Ajudei uma paciente a superar trauma de 20 anos. Por isso amo meu trabalho.", :milestone, 0.9, [0.8, 0.4, 0.6])
implant_memory.(helena, "Bento me ensinou que silêncio também é resposta.", :reflection, 0.7, [0.5, -0.3, 0.2])

# Carmen
implant_memory.(carmen, "Meu marido morreu há 5 anos. Ainda sinto falta do cheiro dele.", :emotional_peak, 1.0, [-0.7, -0.4, -0.5])
implant_memory.(carmen, "Abri o restaurante com minhas economias. Meu sonho realizado.", :milestone, 0.95, [0.8, 0.5, 0.7])

# Jorge
implant_memory.(jorge, "Toquei com Tom Jobim em 1985. Maior honra da carreira.", :milestone, 1.0, [0.9, 0.6, 0.5])
implant_memory.(jorge, "Carmen me faz lembrar de ser jovem novamente.", :reflection, 0.8, [0.7, 0.4, 0.3])

IO.puts("   ✅ Memórias implantadas")

# ==============================================================================
# 7. SWIPES (Drama Inicial)
# ==============================================================================
IO.puts("💕 Registrando swipes iniciais...")

# Lucas gosta de Sofia (ela não sabe)
Viva.Matching.swipe(lucas.id, sofia.id, :superlike)

# Sofia swipou em Lucas também! (match potencial)
Viva.Matching.swipe(sofia.id, lucas.id, :like)

# Clara gosta de Leo
Viva.Matching.swipe(clara.id, leo.id, :superlike)

# Leo ainda não viu Clara
# (sem swipe)

# Tiago gosta de Diana
Viva.Matching.swipe(tiago.id, diana.id, :like)

# Diana gosta de Tiago também
Viva.Matching.swipe(diana.id, tiago.id, :like)

# Gael dá like em todo mundo rico
Viva.Matching.swipe(gael.id, zara.id, :superlike)
Viva.Matching.swipe(gael.id, valentina.id, :superlike)

# Zara passa em Gael
Viva.Matching.swipe(zara.id, gael.id, :pass)

# Pedro gosta de Luna
Viva.Matching.swipe(pedro.id, luna.id, :like)

# Carmen e Jorge ainda não swiparam (old school)

# Rafael gosta de Helena
Viva.Matching.swipe(rafael.id, helena.id, :like)

IO.puts("   ✅ Swipes registrados")

# ==============================================================================
# 8. SUMMARY
# ==============================================================================
IO.puts("")
IO.puts("═══════════════════════════════════════════════════════════════")
IO.puts("🌍 GÊNESIS COMPLETO - VIVA ANT FARM")
IO.puts("═══════════════════════════════════════════════════════════════")
IO.puts("")
IO.puts("📊 Estatísticas do Mundo:")
IO.puts("   👤 Usuários:        1")
IO.puts("   🐜 Avatares:        #{length(all_avatars)}")
IO.puts("   🕸️  Relacionamentos: #{Repo.aggregate(Relationship, :count)}")
IO.puts("   🧠 Memórias:        #{Repo.aggregate(Memory, :count)}")
IO.puts("   💕 Swipes:          #{Repo.aggregate(Viva.Matching.Swipe, :count)}")
IO.puts("")
IO.puts("🎭 Dinâmicas Interessantes:")
IO.puts("   💔 Sofia & Arthur (ex complicados)")
IO.puts("   💕 Lucas → Sofia (amor secreto)")
IO.puts("   🎨 Clara → Leo (crush não correspondido)")
IO.puts("   ⚔️  Igor vs Diana (inimigos)")
IO.puts("   🎭 Gael → Zara (bajulador)")
IO.puts("   👴 Carmen & Jorge (amizade madura)")
IO.puts("")
IO.puts("🔑 Login: god@viva.ai / VivaGod2024!")
IO.puts("")
IO.puts("🚀 Para iniciar a simulação:")
IO.puts("   iex -S mix phx.server")
IO.puts("")
