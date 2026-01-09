defmodule Viva.AI.LLM.Backends.GeminiCliBackend do
  @moduledoc """
  Backend para LLM usando Gemini CLI (gemini-cli).

  Usa o binário `gemini` instalado no sistema para inferência.
  Ideal como fallback do Groq por ter rate limits generosos.

  Configuração via config:
    config :viva, Viva.AI.LLM.Backends.GeminiCliBackend,
      executable: "gemini",
      model: "gemini-2.5-flash-lite",
      timeout: 30_000
  """

  @behaviour Viva.AI.LLM.ClientBehaviour

  require Logger

  @default_executable "gemini"
  @default_model nil  # nil = Auto (Gemini 3)
  @default_timeout 30_000

  @thought_settings %{
    max_tokens: 150
  }

  @doc """
  Gera uma resposta para um prompt simples.
  """
  @impl true
  def generate(prompt, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, config(:timeout, @default_timeout))
    model = Keyword.get(opts, :model, config(:model, @default_model))

    run_gemini(prompt, model, timeout)
  end

  @doc """
  Processa uma conversa com múltiplas mensagens.
  Converte mensagens para um prompt único para o gemini-cli.
  """
  @impl true
  def chat(messages, opts \\ []) when is_list(messages) do
    # Converte mensagens para prompt único
    prompt = messages_to_prompt(messages)
    generate(prompt, opts)
  end

  @doc """
  Gera um pensamento de avatar.
  """
  def generate_thought(prompt, opts \\ []) do
    merged_opts = Keyword.merge([
      max_tokens: @thought_settings.max_tokens
    ], opts)

    generate(prompt, merged_opts)
  end

  @doc """
  Verifica se o gemini-cli está disponível.
  """
  def health_check do
    executable = config(:executable, @default_executable)

    case System.find_executable(executable) do
      nil ->
        {:error, :executable_not_found}

      _path ->
        {:ok, :healthy}
    end
  end

  @doc """
  Retorna a versão do gemini-cli.
  """
  def version do
    executable = config(:executable, @default_executable)

    case System.cmd(executable, ["--version"], stderr_to_stdout: true) do
      {version, 0} -> {:ok, String.trim(version)}
      {error, _} -> {:error, error}
    end
  end

  # Private functions

  defp run_gemini(prompt, model, timeout) do
    executable = config(:executable, @default_executable)

    # Se model for nil, usa Auto (Gemini 3)
    args = case model do
      nil -> ["-o", "json", prompt]
      m -> ["-m", m, "-o", "json", prompt]
    end

    start = System.monotonic_time(:millisecond)

    task = Task.async(fn ->
      System.cmd(executable, args, stderr_to_stdout: true)
    end)

    case Task.yield(task, timeout) || Task.shutdown(task) do
      {:ok, {output, 0}} ->
        elapsed = System.monotonic_time(:millisecond) - start
        parse_response(output, elapsed)

      {:ok, {error, _code}} ->
        Logger.error("[GeminiCliBackend] Command failed: #{error}")
        {:error, {:command_failed, error}}

      nil ->
        Logger.error("[GeminiCliBackend] Command timed out after #{timeout}ms")
        {:error, :timeout}
    end
  end

  defp parse_response(output, elapsed) do
    # Remove linhas antes do JSON (como "Loaded cached credentials.")
    json_output = output
    |> String.split("\n")
    |> Enum.drop_while(fn line -> not String.starts_with?(String.trim(line), "{") end)
    |> Enum.join("\n")

    case Jason.decode(json_output) do
      {:ok, %{"response" => response}} ->
        Logger.debug("[GeminiCliBackend] Request completed in #{elapsed}ms")
        {:ok, String.trim(response)}

      {:ok, %{"error" => error}} ->
        {:error, {:api_error, error}}

      {:error, _} ->
        # Tenta extrair resposta de texto puro se JSON falhar
        if String.contains?(output, "error") do
          {:error, {:parse_error, output}}
        else
          {:ok, String.trim(output)}
        end
    end
  end

  defp messages_to_prompt(messages) do
    messages
    |> Enum.map(fn
      %{role: "system", content: content} -> "Sistema: #{content}"
      %{role: "user", content: content} -> "Usuário: #{content}"
      %{role: "assistant", content: content} -> "Assistente: #{content}"
      %{"role" => "system", "content" => content} -> "Sistema: #{content}"
      %{"role" => "user", "content" => content} -> "Usuário: #{content}"
      %{"role" => "assistant", "content" => content} -> "Assistente: #{content}"
    end)
    |> Enum.join("\n\n")
    |> Kernel.<>("\n\nAssistente:")
  end

  defp config(key, default) do
    Application.get_env(:viva, __MODULE__, [])
    |> Keyword.get(key, default)
  end
end
