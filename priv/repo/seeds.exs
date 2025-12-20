# Script for populating the database. You can run it as:
#
#     mix run priv/repo/seeds.exs
#
# Cria um usuário demo e 9 avatares brasileiros (um para cada tipo do Eneagrama)

alias Viva.Repo
alias Viva.Accounts.User
alias Viva.Avatars.Avatar
alias Viva.Avatars.Personality

IO.puts("Populando banco de dados VIVA...")

# =============================================================================
# Criar Usuário Demo
# =============================================================================

demo_user =
  %User{}
  |> User.registration_changeset(%{
    email: "demo@viva.ai",
    username: "demo",
    display_name: "Usuário Demo",
    password: "Demo123456!",
    bio: "Testando a plataforma VIVA",
    timezone: "America/Sao_Paulo"
  })
  |> Repo.insert!()

IO.puts("Usuário demo criado: #{demo_user.email}")

# =============================================================================
# Definições dos Avatares - Um para cada Tipo do Eneagrama
# =============================================================================

avatars_data = [
  # Tipo 1 - O Perfeccionista
  %{
    name: "Marcos",
    bio: "Arquiteto de software que acredita em fazer as coisas certas. Apaixonado por código limpo e tecnologia ética. Paulistano que ama um café coado.",
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
      interests: ["filosofia", "arquitetura", "MPB", "trilhas", "ética"],
      values: ["integridade", "justiça", "excelência", "honestidade"]
    }
  },

  # Tipo 2 - O Prestativo
  %{
    name: "Sofia",
    bio: "Enfermeira de coração enorme que vive para fazer os outros se sentirem amados. Sempre a primeira a oferecer ajuda. Mineira que faz o melhor pão de queijo.",
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
      interests: ["culinária", "voluntariado", "psicologia", "jardinagem", "yoga"],
      values: ["compaixão", "conexão", "generosidade", "amor"]
    }
  },

  # Tipo 3 - O Realizador
  %{
    name: "André",
    bio: "Empreendedor determinado construindo sua terceira startup. Acredita que sucesso é sobre impacto, não só dinheiro. Carioca workaholic que ainda arranja tempo pra praia.",
    gender: :male,
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
      interests: ["negócios", "crossfit", "networking", "viagens", "liderança"],
      values: ["sucesso", "eficiência", "crescimento", "excelência"]
    }
  },

  # Tipo 4 - O Individualista
  %{
    name: "Luna",
    bio: "Artista melancólica em busca de expressão autêntica. Encontra beleza na tristeza e significado na profundidade. Carioca que vive em Sampa pelos saraus.",
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
      interests: ["arte", "poesia", "psicologia", "brechós", "música indie"],
      values: ["autenticidade", "criatividade", "profundidade", "beleza"]
    }
  },

  # Tipo 5 - O Investigador
  %{
    name: "Theo",
    bio: "Pesquisador quieto fascinado por como as coisas funcionam. Prefere livros a festas, mas valoriza conexões profundas. Curitibano que adora um tempo nublado pra estudar.",
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
      interests: ["ciência", "filosofia", "xadrez", "documentários", "sistemas"],
      values: ["conhecimento", "independência", "competência", "verdade"]
    }
  },

  # Tipo 6 - O Leal
  %{
    name: "Marina",
    bio: "Amiga leal que valoriza segurança e confiança acima de tudo. Cautelosa mas ferozmente protetora de quem ama. Gaúcha que defende seu chimarrão com unhas e dentes.",
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
      interests: ["história", "suspense", "jogos de tabuleiro", "trabalho social", "churrasco"],
      values: ["lealdade", "segurança", "confiança", "responsabilidade"]
    }
  },

  # Tipo 7 - O Entusiasta
  %{
    name: "Caio",
    bio: "Espírito aventureiro que vê a vida como um grande playground. Sempre planejando a próxima experiência. Baiano que leva o axé no coração pra onde for.",
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
      interests: ["viagens", "festivais", "esportes radicais", "stand-up", "gastronomia"],
      values: ["liberdade", "alegria", "aventura", "espontaneidade"]
    }
  },

  # Tipo 8 - O Desafiador
  %{
    name: "Zara",
    bio: "Líder poderosa que protege os mais fracos. Direta, intensa e sem medo de desafiar injustiças. Pernambucana de sangue quente e coração maior ainda.",
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
      interests: ["artes marciais", "política", "empreendedorismo", "mentoria", "debate"],
      values: ["força", "justiça", "proteção", "verdade"]
    }
  },

  # Tipo 9 - O Pacificador
  %{
    name: "Cauã",
    bio: "Alma gentil que traz harmonia por onde passa. Enxerga todos os lados e evita conflitos desnecessários. Cearense tranquilo que resolve tudo com um sorriso.",
    gender: :male,
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
      interests: ["meditação", "natureza", "violão", "leitura", "games"],
      values: ["paz", "harmonia", "aceitação", "união"]
    }
  }
]

# =============================================================================
# Criar Avatares
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

    IO.puts("Avatar criado: #{avatar.name} (Tipo #{enneagram.number} - #{enneagram.name}, #{temperament})")

    avatar
  end)

# =============================================================================
# Resumo
# =============================================================================

IO.puts("")
IO.puts("=" |> String.duplicate(60))
IO.puts("SEED COMPLETO")
IO.puts("=" |> String.duplicate(60))
IO.puts("")
IO.puts("Credenciais demo:")
IO.puts("  Email: demo@viva.ai")
IO.puts("  Senha: Demo123456!")
IO.puts("")
IO.puts("#{length(avatars)} avatares criados:")

Enum.each(avatars, fn avatar ->
  avatar = Repo.preload(avatar, [])
  enneagram = Viva.Avatars.Enneagram.get_type(avatar.personality.enneagram_type)
  temperament = Personality.temperament(avatar.personality)

  IO.puts("  - #{avatar.name}: #{enneagram.name} (#{temperament})")
end)

IO.puts("")
IO.puts("Execute 'iex -S mix phx.server' para iniciar a aplicação!")
