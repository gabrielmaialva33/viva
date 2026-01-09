defmodule Viva.Collective.Mutator do
  @moduledoc """
  Motor de Mutação - Individualidade na Mente Coletiva.

  Quando um pensamento é transmitido de um avatar para outro,
  o receptor "interpreta" o pensamento através de sua personalidade,
  estado emocional e memórias, gerando uma MUTAÇÃO única.

  ## Tipos de Mutação

  - **Semântica**: Muda o significado levemente
  - **Emocional**: Adiciona coloração emocional do receptor
  - **Intensidade**: Amplifica ou atenua
  - **Criativa**: Adiciona elementos novos (alta criatividade)

  ## Exemplo

      Original: "Estou feliz hoje"

      Avatar Criativo:     "A alegria dança no ar como borboletas"
      Avatar Melancólico:  "Uma fagulha de felicidade, rara e preciosa"
      Avatar Analítico:    "Identifico um estado emocional positivo"
      Avatar Empático:     "Sinto que todos merecem essa felicidade"
  """

  require Logger

  alias Viva.AI.LLM.Backends.{GroqBackend, GeminiCliBackend}

  @mutation_prompt_template """
  Você é a mente do avatar %{name}.
  Personalidade: %{personality}
  Estado emocional atual: %{emotional_state}

  Você recebeu este pensamento de outro avatar:
  "%{original_thought}"

  Reinterprete este pensamento com SUA voz única.
  Mantenha a essência mas adicione sua perspectiva pessoal.
  Responda apenas com o pensamento reinterpretado (máximo 2 frases).
  """

  # Padrões de mutação para fallback (prefixo ou sufixo)
  @fallback_patterns [
    {:prefix, "Refletindo sobre isso... "},
    {:suffix, "... isso ressoa em mim"},
    {:prefix, "Sinto que "},
    {:prefix, "Penso diferente: "},
    {:prefix, "De certa forma, "}
  ]

  @doc """
  Aplica mutação a um pensamento baseado na personalidade do avatar receptor.

  ## Parâmetros
  - `content`: Conteúdo original do pensamento
  - `avatar_id`: ID do avatar que está recebendo

  ## Retorno
  String com o pensamento mutado
  """
  def mutate(content, avatar_id) do
    case get_avatar_context(avatar_id) do
      {:ok, context} ->
        mutate_with_llm(content, context)

      :error ->
        mutate_fallback(content, avatar_id)
    end
  end

  @doc """
  Calcula o "grau de mutação" entre original e mutado.
  Retorna 0.0 (idêntico) a 1.0 (completamente diferente).
  """
  def mutation_degree(original, mutated) do
    # Usa distância de Levenshtein normalizada
    original_len = String.length(original)
    mutated_len = String.length(mutated)
    max_len = max(original_len, mutated_len)

    if max_len == 0 do
      0.0
    else
      distance = levenshtein_distance(original, mutated)
      min(distance / max_len, 1.0)
    end
  end

  @doc """
  Verifica se dois pensamentos são "similares" (para convergência).
  """
  def similar?(thought1, thought2, threshold \\ 0.7) do
    degree = mutation_degree(thought1, thought2)
    (1.0 - degree) >= threshold
  end

  ## Private Functions

  defp mutate_with_llm(content, context) do
    prompt = build_mutation_prompt(content, context)

    # Tenta Groq primeiro (mais rápido)
    case GroqBackend.generate(prompt, max_tokens: 100, temperature: 0.9) do
      {:ok, mutated} ->
        clean_mutation(mutated, content)

      {:error, _} ->
        # Fallback para Gemini
        case GeminiCliBackend.generate(prompt, max_tokens: 100) do
          {:ok, mutated} -> clean_mutation(mutated, content)
          {:error, _} -> mutate_fallback(content, context.avatar_id)
        end
    end
  rescue
    _ -> mutate_fallback(content, nil)
  end

  defp mutate_fallback(content, _avatar_id) do
    # Seleciona um padrão aleatório e aplica
    pattern = Enum.random(@fallback_patterns)
    apply_pattern(pattern, content)
  end

  defp apply_pattern({:prefix, prefix}, content), do: prefix <> content
  defp apply_pattern({:suffix, suffix}, content), do: content <> suffix

  defp build_mutation_prompt(content, context) do
    @mutation_prompt_template
    |> String.replace("%{name}", context.name || "Avatar")
    |> String.replace("%{personality}", describe_personality(context))
    |> String.replace("%{emotional_state}", describe_emotional_state(context))
    |> String.replace("%{original_thought}", content)
  end

  defp clean_mutation(mutated, original) do
    cleaned = mutated
    |> String.trim()
    |> String.replace(~r/^["']|["']$/, "")  # Remove aspas
    |> String.replace(~r/^\*.*\*$/, "")      # Remove asteriscos
    |> String.trim()

    # Se ficou vazio ou muito similar, retorna original com leve modificação
    if String.length(cleaned) < 5 or cleaned == original do
      "#{original}... em outras palavras"
    else
      cleaned
    end
  end

  defp get_avatar_context(avatar_id) do
    with {:ok, avatar} <- Viva.Avatars.get_avatar(avatar_id),
         {:ok, state} <- Viva.Sessions.LifeProcess.get_internal_state(avatar_id) do
      {:ok, %{
        avatar_id: avatar_id,
        name: avatar.name,
        personality: avatar.personality || %{},
        emotional_state: state,
        creativity: get_in(avatar.personality, [:openness]) || 0.5
      }}
    else
      _ -> :error
    end
  rescue
    _ -> :error
  end

  defp describe_personality(context) do
    traits = context.personality || %{}

    [
      if(traits[:openness] > 0.7, do: "criativo e imaginativo"),
      if(traits[:conscientiousness] > 0.7, do: "metódico e organizado"),
      if(traits[:extraversion] > 0.7, do: "extrovertido e expressivo"),
      if(traits[:agreeableness] > 0.7, do: "empático e acolhedor"),
      if(traits[:neuroticism] > 0.7, do: "sensível e introspectivo")
    ]
    |> Enum.filter(& &1)
    |> Enum.join(", ")
    |> case do
      "" -> "equilibrado"
      desc -> desc
    end
  end

  defp describe_emotional_state(context) do
    # context.emotional_state contains the full internal state
    # We need to access the nested :emotional map (may be a struct)
    state = context.emotional_state || %{}
    emotional = Map.get(state, :emotional) || %{}
    emotional_map = if is_struct(emotional), do: Map.from_struct(emotional), else: emotional

    pleasure = Map.get(emotional_map, :pleasure) || Map.get(emotional_map, :valence, 0)
    arousal = Map.get(emotional_map, :arousal) || Map.get(emotional_map, :activation, 0)

    cond do
      pleasure > 0.5 and arousal > 0.5 -> "animado e feliz"
      pleasure > 0.5 and arousal < -0.5 -> "sereno e contente"
      pleasure < -0.5 and arousal > 0.5 -> "ansioso e agitado"
      pleasure < -0.5 and arousal < -0.5 -> "triste e desanimado"
      true -> "neutro"
    end
  end

  # Implementação simples de distância de Levenshtein
  defp levenshtein_distance(s1, s2) do
    s1_chars = String.graphemes(s1)
    s2_chars = String.graphemes(s2)
    s1_len = length(s1_chars)
    s2_len = length(s2_chars)

    cond do
      s1_len == 0 -> s2_len
      s2_len == 0 -> s1_len
      true -> do_levenshtein(s1_chars, s2_chars, s1_len, s2_len)
    end
  end

  defp do_levenshtein(s1, s2, _len1, len2) do
    # Versão otimizada com duas linhas
    row = 0..len2 |> Enum.to_list()

    Enum.reduce(Enum.with_index(s1), row, fn {c1, i}, prev_row ->
      Enum.reduce(Enum.with_index(s2), {i + 1, [i + 1]}, fn {c2, j}, {_, curr_row} ->
        cost = if c1 == c2, do: 0, else: 1

        val = min(
          Enum.at(prev_row, j + 1) + 1,      # deletion
          min(
            List.last(curr_row) + 1,          # insertion
            Enum.at(prev_row, j) + cost       # substitution
          )
        )

        {val, curr_row ++ [val]}
      end)
      |> elem(1)
      |> tl()
    end)
    |> List.last()
  end
end
