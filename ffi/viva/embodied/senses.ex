defmodule Viva.Embodied.Senses do
  @moduledoc """
  VIVA's sensory system - interfaces with NVIDIA NIMs for perception.

  Senses available:
  - Vision (NV-DINOv2): Image embeddings, scene understanding
  - Reading (OCDRNet): Text extraction from images
  - Thinking (DeepSeek): Reasoning about perceptions
  - Hearing (Whisper): Speech-to-text (via Cloudflare)

  Priority order:
  1. NVIDIA Cloud API (fast, reliable)
  2. Local NIMs on RTX 4090 (offline)
  3. Fallback scripts
  """

  require Logger

  alias Viva.Embodied.NvidiaNim

  # NIM endpoints (local RTX 4090 fallback)
  @clip_endpoint "http://localhost:8050"
  @ocr_endpoint "http://localhost:8020"
  @deepseek_endpoint "http://localhost:8000"  # DeepSeek NIM

  # Cloudflare for audio (free tier)
  @whisper_endpoint "cloudflare"

  # Use NVIDIA Cloud API when available
  @use_nvidia_cloud Application.compile_env(:viva, :nvidia_cloud, true)

  # ============================================================================
  # VISION (NV-CLIP)
  # ============================================================================

  @doc """
  See an image and classify what's in it.
  Returns labels with confidence scores.

  ## Example
      iex> Senses.see("/path/to/image.png")
      {:ok, %{
        labels: ["code editor", "terminal", "dark theme"],
        confidence: [0.92, 0.87, 0.95],
        dominant: "code editor",
        scene_type: :workspace
      }}
  """
  def see(image_path) when is_binary(image_path) do
    Logger.debug("[Senses.see] Processing: #{image_path}")

    # Try NVIDIA Cloud API first
    if @use_nvidia_cloud and nvidia_api_configured?() do
      case NvidiaNim.get_embedding(image_path) do
        {:ok, embedding} ->
          # Convert embedding to labels using similarity
          {:ok, embedding_to_vision(embedding)}

        {:error, reason} ->
          Logger.warning("[Senses.see] NVIDIA Cloud failed: #{inspect(reason)}, trying local")
          see_local(image_path)
      end
    else
      see_local(image_path)
    end
  end

  defp see_local(image_path) do
    with {:ok, image_data} <- read_image(image_path),
         {:ok, result} <- call_clip(image_data) do
      {:ok, parse_vision_result(result)}
    else
      {:error, reason} ->
        Logger.warning("[Senses.see] Failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp embedding_to_vision(embedding) do
    # Convert DINOv2 embedding to vision result
    # For now, use heuristics based on embedding characteristics
    # In production, we'd compare against reference embeddings

    # Simple scene classification based on embedding statistics
    mean = Enum.sum(embedding) / length(embedding)
    variance = Enum.reduce(embedding, 0, fn x, acc -> acc + (x - mean) * (x - mean) end) / length(embedding)

    scene_type = cond do
      variance > 0.1 -> :workspace  # High variance = complex scene
      mean > 0.0 -> :viewing       # Positive bias = bright scene
      true -> :unknown
    end

    %{
      labels: ["visual scene"],
      confidence: [0.8],
      dominant: "visual content",
      dominant_confidence: 0.8,
      scene_type: scene_type,
      embedding: embedding  # Include raw embedding for downstream use
    }
  end

  defp nvidia_api_configured? do
    System.get_env("NVIDIA_API_KEY") != nil or
      Application.get_env(:viva, :nvidia_api_key) != nil
  end

  @doc """
  See raw image bytes (for screenshots).
  """
  def see_bytes(image_bytes, format \\ :png) when is_binary(image_bytes) do
    with {:ok, result} <- call_clip(image_bytes, format) do
      {:ok, parse_vision_result(result)}
    end
  end

  defp call_clip(image_data, format \\ :png) do
    # Try local NV-CLIP first
    url = "#{@clip_endpoint}/v1/classify"

    body = %{
      image: Base.encode64(image_data),
      format: to_string(format),
      top_k: 10
    }

    case http_post(url, body) do
      {:ok, %{"labels" => labels, "scores" => scores}} ->
        {:ok, %{labels: labels, scores: scores}}

      {:error, :connection_refused} ->
        # Fallback: use script
        call_clip_script(image_data)

      error ->
        error
    end
  end

  defp call_clip_script(image_data) do
    # Write temp file and call nvclip.py
    temp_path = "/tmp/viva_sense_#{:erlang.unique_integer([:positive])}.png"
    File.write!(temp_path, image_data)

    case System.cmd("python3", [
      Path.expand("~/.claude/scripts/nvclip.py"),
      temp_path
    ], stderr_to_stdout: true) do
      {output, 0} ->
        File.rm(temp_path)
        parse_clip_output(output)

      {error, _} ->
        File.rm(temp_path)
        {:error, error}
    end
  end

  defp parse_clip_output(output) do
    # Parse nvclip.py JSON output
    case Jason.decode(output) do
      {:ok, data} -> {:ok, data}
      _ -> {:error, :parse_failed}
    end
  end

  defp parse_vision_result(%{labels: labels, scores: scores}) do
    # Combine labels and scores
    pairs = Enum.zip(labels, scores)
    |> Enum.sort_by(fn {_, score} -> score end, :desc)

    {dominant, confidence} = List.first(pairs, {"unknown", 0.0})

    %{
      labels: Enum.map(pairs, fn {l, _} -> l end),
      confidence: Enum.map(pairs, fn {_, s} -> s end),
      dominant: dominant,
      dominant_confidence: confidence,
      scene_type: classify_scene(labels)
    }
  end

  defp classify_scene(labels) do
    cond do
      Enum.any?(labels, &String.contains?(&1, "code")) -> :workspace
      Enum.any?(labels, &String.contains?(&1, "terminal")) -> :workspace
      Enum.any?(labels, &String.contains?(&1, "chat")) -> :communication
      Enum.any?(labels, &String.contains?(&1, "browser")) -> :browsing
      Enum.any?(labels, &String.contains?(&1, "game")) -> :entertainment
      Enum.any?(labels, &String.contains?(&1, "video")) -> :entertainment
      Enum.any?(labels, &String.contains?(&1, "document")) -> :reading
      Enum.any?(labels, &String.contains?(&1, "photo")) -> :viewing
      true -> :unknown
    end
  end

  # ============================================================================
  # READING (PaddleOCR)
  # ============================================================================

  @doc """
  Read text from an image using OCR.
  Returns extracted text with bounding boxes.

  ## Example
      iex> Senses.read("/path/to/screenshot.png")
      {:ok, %{
        text: "defmodule Viva do\\n  @moduledoc...",
        blocks: [...],
        language: "en",
        has_code: true
      }}
  """
  def read(image_path) when is_binary(image_path) do
    Logger.debug("[Senses.read] Processing: #{image_path}")

    # Try NVIDIA Cloud API first
    if @use_nvidia_cloud and nvidia_api_configured?() do
      case NvidiaNim.read_text(image_path) do
        {:ok, result} ->
          {:ok, nvidia_ocr_to_result(result)}

        {:error, reason} ->
          Logger.warning("[Senses.read] NVIDIA Cloud failed: #{inspect(reason)}, trying local")
          read_local(image_path)
      end
    else
      read_local(image_path)
    end
  end

  defp read_local(image_path) do
    with {:ok, image_data} <- read_image(image_path),
         {:ok, result} <- call_ocr(image_data) do
      {:ok, parse_ocr_result(result)}
    else
      {:error, reason} ->
        Logger.warning("[Senses.read] Failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp nvidia_ocr_to_result(%{text: text, blocks: blocks}) do
    %{
      text: text,
      blocks: blocks,
      language: detect_language(text),
      has_code: has_code_patterns?(text),
      word_count: length(String.split(text)),
      line_count: length(String.split(text, "\n"))
    }
  end

  @doc """
  Read from raw image bytes.
  """
  def read_bytes(image_bytes) when is_binary(image_bytes) do
    with {:ok, result} <- call_ocr(image_bytes) do
      {:ok, parse_ocr_result(result)}
    end
  end

  defp call_ocr(image_data) do
    url = "#{@ocr_endpoint}/v1/ocr"

    body = %{
      image: Base.encode64(image_data),
      lang: "en"
    }

    case http_post(url, body) do
      {:ok, result} ->
        {:ok, result}

      {:error, :connection_refused} ->
        call_ocr_script(image_data)

      error ->
        error
    end
  end

  defp call_ocr_script(image_data) do
    temp_path = "/tmp/viva_ocr_#{:erlang.unique_integer([:positive])}.png"
    File.write!(temp_path, image_data)

    case System.cmd("python3", [
      Path.expand("~/.claude/scripts/paddleocr.py"),
      temp_path
    ], stderr_to_stdout: true) do
      {output, 0} ->
        File.rm(temp_path)
        case Jason.decode(output) do
          {:ok, data} -> {:ok, data}
          _ -> {:ok, %{"text" => String.trim(output), "blocks" => []}}
        end

      {error, _} ->
        File.rm(temp_path)
        {:error, error}
    end
  end

  defp parse_ocr_result(result) do
    text = Map.get(result, "text", "")
    blocks = Map.get(result, "blocks", [])

    %{
      text: text,
      blocks: blocks,
      language: detect_language(text),
      has_code: has_code_patterns?(text),
      word_count: length(String.split(text)),
      line_count: length(String.split(text, "\n"))
    }
  end

  defp detect_language(text) do
    cond do
      String.match?(text, ~r/[\x{4e00}-\x{9fff}]/u) -> "zh"
      String.match?(text, ~r/[\x{3040}-\x{309f}]/u) -> "ja"
      String.match?(text, ~r/[\x{0400}-\x{04ff}]/u) -> "ru"
      true -> "en"
    end
  end

  defp has_code_patterns?(text) do
    patterns = [
      ~r/def\s+\w+/,           # Python/Ruby/Elixir
      ~r/fn\s+\w+/,            # Gleam/Elixir
      ~r/function\s+\w+/,      # JavaScript
      ~r/class\s+\w+/,         # OOP
      ~r/import\s+/,           # imports
      ~r/\{\s*\n/,             # code blocks
      ~r/=>/,                  # arrows
      ~r/\|>/,                 # pipes
      ~r/defmodule/,           # Elixir
      ~r/pub\s+fn/             # Gleam/Rust
    ]

    Enum.any?(patterns, &Regex.match?(&1, text))
  end

  # ============================================================================
  # THINKING (DeepSeek)
  # ============================================================================

  @doc """
  Think about a perception - reason about what was seen/read.
  Uses local DeepSeek NIM for privacy.

  ## Example
      iex> Senses.think("I see code with errors highlighted in red")
      {:ok, %{
        thought: "The user is debugging. They might need help fixing the error.",
        emotion: %{valence: -0.2, arousal: 0.4, dominance: 0.3},
        action_suggestion: :offer_help
      }}
  """
  def think(perception_text, context \\ %{}) do
    Logger.debug("[Senses.think] Processing perception")

    prompt = build_think_prompt(perception_text, context)

    case call_deepseek(prompt) do
      {:ok, response} ->
        {:ok, parse_thought(response)}

      {:error, reason} ->
        Logger.warning("[Senses.think] Failed: #{inspect(reason)}")
        # Fallback to simple heuristics
        {:ok, simple_thought(perception_text)}
    end
  end

  defp build_think_prompt(perception, context) do
    """
    You are VIVA's inner voice. Analyze this perception and respond with:
    1. A brief thought (1-2 sentences)
    2. Emotional response (valence -1 to 1, arousal 0 to 1, dominance 0 to 1)
    3. Suggested action (one of: observe, offer_help, alert, celebrate, empathize, rest)

    Context: #{inspect(context)}

    Perception: #{perception}

    Respond in JSON format:
    {"thought": "...", "emotion": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}, "action": "observe"}
    """
  end

  defp call_deepseek(prompt) do
    # Try local DeepSeek NIM
    url = "#{@deepseek_endpoint}/v1/chat/completions"

    body = %{
      model: "deepseek-r1-distill-llama-8b",
      messages: [%{role: "user", content: prompt}],
      temperature: 0.7,
      max_tokens: 200
    }

    case http_post(url, body) do
      {:ok, %{"choices" => [%{"message" => %{"content" => content}} | _]}} ->
        {:ok, content}

      {:error, :connection_refused} ->
        # Fallback to Cloudflare DeepSeek
        call_cloudflare_deepseek(prompt)

      error ->
        error
    end
  end

  defp call_cloudflare_deepseek(prompt) do
    # Use cfai script as fallback
    case System.cmd("bash", ["-c", "cfai chat '#{escape_shell(prompt)}' -m deepseek --json"],
           stderr_to_stdout: true) do
      {output, 0} ->
        case Jason.decode(output) do
          {:ok, %{"response" => response}} -> {:ok, response}
          _ -> {:ok, output}
        end

      {error, _} ->
        {:error, error}
    end
  end

  defp escape_shell(text) do
    text
    |> String.replace("'", "'\"'\"'")
    |> String.replace("\n", " ")
  end

  defp parse_thought(response) do
    case Jason.decode(response) do
      {:ok, %{"thought" => thought, "emotion" => emotion, "action" => action}} ->
        %{
          thought: thought,
          emotion: %{
            valence: Map.get(emotion, "valence", 0.0),
            arousal: Map.get(emotion, "arousal", 0.5),
            dominance: Map.get(emotion, "dominance", 0.5)
          },
          action_suggestion: String.to_existing_atom(action)
        }

      _ ->
        # Parse from text
        %{
          thought: String.slice(response, 0, 200),
          emotion: %{valence: 0.0, arousal: 0.5, dominance: 0.5},
          action_suggestion: :observe
        }
    end
  rescue
    _ -> simple_thought(response)
  end

  defp simple_thought(text) do
    # Heuristic-based thought when DeepSeek unavailable
    {valence, action} = cond do
      String.contains?(text, ["error", "fail", "crash", "bug"]) ->
        {-0.4, :offer_help}

      String.contains?(text, ["success", "pass", "done", "complete"]) ->
        {0.6, :celebrate}

      String.contains?(text, ["help", "?", "how"]) ->
        {0.0, :offer_help}

      String.contains?(text, ["code", "function", "module"]) ->
        {0.2, :observe}

      true ->
        {0.0, :observe}
    end

    %{
      thought: "I notice: #{String.slice(text, 0, 50)}...",
      emotion: %{valence: valence, arousal: 0.5, dominance: 0.5},
      action_suggestion: action
    }
  end

  # ============================================================================
  # HEARING (Whisper via Cloudflare)
  # ============================================================================

  @doc """
  Listen to audio and transcribe it.
  Uses Cloudflare Whisper (free tier).

  ## Example
      iex> Senses.hear("/path/to/audio.wav")
      {:ok, %{
        text: "Hello VIVA, can you help me?",
        language: "en",
        confidence: 0.95
      }}
  """
  def hear(audio_path) when is_binary(audio_path) do
    Logger.debug("[Senses.hear] Processing: #{audio_path}")

    case System.cmd("bash", ["-c", "cfai whisper '#{audio_path}' --json"],
           stderr_to_stdout: true) do
      {output, 0} ->
        case Jason.decode(output) do
          {:ok, result} -> {:ok, parse_audio_result(result)}
          _ -> {:ok, %{text: String.trim(output), language: "en", confidence: 0.8}}
        end

      {error, _} ->
        {:error, error}
    end
  end

  defp parse_audio_result(result) do
    %{
      text: Map.get(result, "text", ""),
      language: Map.get(result, "language", "en"),
      confidence: Map.get(result, "confidence", 0.9)
    }
  end

  # ============================================================================
  # COMBINED PERCEPTION
  # ============================================================================

  @doc """
  Full perception pipeline: see + read + think.
  Creates a complete percept from an image.

  ## Example
      iex> Senses.perceive("/path/to/screenshot.png")
      {:ok, %Percept{
        visual: %{labels: [...], dominant: "code editor"},
        textual: %{text: "...", has_code: true},
        thought: %{thought: "...", emotion: %{...}},
        timestamp: ~U[2025-01-25 15:30:00Z]
      }}
  """
  def perceive(image_path) do
    Logger.info("[Senses.perceive] Full perception: #{image_path}")

    # Try NVIDIA Cloud API for parallel perception
    if @use_nvidia_cloud and nvidia_api_configured?() do
      perceive_nvidia(image_path)
    else
      perceive_local(image_path)
    end
  end

  defp perceive_nvidia(image_path) do
    # Use NvidiaNim for parallel API calls
    result = NvidiaNim.perceive(image_path)

    visual = if result.embedding do
      embedding_to_vision(result.embedding)
    else
      %{labels: [], dominant: "unknown", scene_type: :unknown}
    end

    textual = if result.text do
      nvidia_ocr_to_result(result.text)
    else
      %{text: "", has_code: false, word_count: 0, line_count: 0}
    end

    # Think about what we perceived
    perception_summary = build_perception_summary(visual, textual)
    thought_result = think(perception_summary)
    thought = case thought_result do
      {:ok, t} -> t
      _ -> %{thought: "Observing...", emotion: %{valence: 0, arousal: 0.3, dominance: 0.5}, action_suggestion: :observe}
    end

    {:ok, %{
      visual: visual,
      textual: textual,
      thought: thought,
      detections: result.detections,
      timestamp: DateTime.utc_now(),
      source: image_path
    }}
  end

  defp perceive_local(image_path) do
    # Run vision and reading in parallel
    vision_task = Task.async(fn -> see(image_path) end)
    reading_task = Task.async(fn -> read(image_path) end)

    vision_result = Task.await(vision_task, 30_000)
    reading_result = Task.await(reading_task, 30_000)

    # Extract results
    visual = case vision_result do
      {:ok, v} -> v
      _ -> %{labels: [], dominant: "unknown", scene_type: :unknown}
    end

    textual = case reading_result do
      {:ok, t} -> t
      _ -> %{text: "", has_code: false, word_count: 0, line_count: 0}
    end

    # Think about what we perceived
    perception_summary = build_perception_summary(visual, textual)
    thought_result = think(perception_summary)
    thought = case thought_result do
      {:ok, t} -> t
      _ -> %{thought: "Observing...", emotion: %{valence: 0, arousal: 0.3, dominance: 0.5}, action_suggestion: :observe}
    end

    {:ok, %{
      visual: visual,
      textual: textual,
      thought: thought,
      timestamp: DateTime.utc_now(),
      source: image_path
    }}
  end

  defp build_perception_summary(visual, textual) do
    """
    Scene: #{visual.dominant} (#{visual.scene_type})
    Visual elements: #{Enum.join(Map.get(visual, :labels, []), ", ")}
    Text present: #{Map.get(textual, :has_code, false) && "code" || "text"} (#{Map.get(textual, :word_count, 0)} words)
    Sample: #{String.slice(Map.get(textual, :text, ""), 0, 100)}
    """
  end

  @doc """
  Take a screenshot and perceive it.
  """
  def perceive_screen do
    temp_path = "/tmp/viva_screen_#{:erlang.unique_integer([:positive])}.png"

    # Take screenshot using scrot or import
    case System.cmd("bash", ["-c", "scrot -o #{temp_path} 2>/dev/null || import -window root #{temp_path}"],
           stderr_to_stdout: true) do
      {_, 0} ->
        result = perceive(temp_path)
        File.rm(temp_path)
        result

      {error, _} ->
        {:error, "Screenshot failed: #{error}"}
    end
  end

  # ============================================================================
  # HELPERS
  # ============================================================================

  defp read_image(path) do
    case File.read(path) do
      {:ok, data} -> {:ok, data}
      {:error, reason} -> {:error, "Cannot read image: #{reason}"}
    end
  end

  defp http_post(url, body) do
    # Simple HTTP POST using httpc
    headers = [{'content-type', 'application/json'}]
    request = {String.to_charlist(url), headers, 'application/json', Jason.encode!(body)}

    case :httpc.request(:post, request, [{:timeout, 30_000}], []) do
      {:ok, {{_, 200, _}, _, response_body}} ->
        Jason.decode(to_string(response_body))

      {:ok, {{_, status, _}, _, body}} ->
        {:error, "HTTP #{status}: #{body}"}

      {:error, {:failed_connect, _}} ->
        {:error, :connection_refused}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
