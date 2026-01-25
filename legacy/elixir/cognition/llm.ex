defmodule VivaCore.Cognition.LLM do
  @moduledoc """
  Interface for Language Model interactions.
  """

  @callback generate(prompt :: String.t(), opts :: keyword()) ::
              {:ok, String.t()} | {:error, any()}

  # Configurable adapter, defaults to Nvidia
  defp adapter do
    Application.get_env(:viva_core, :llm_adapter, VivaCore.Cognition.LLM.Nvidia)
  end

  def generate(prompt, opts \\ []) do
    adapter().generate(prompt, opts)
  end
end

defmodule VivaCore.Cognition.LLM.Nvidia do
  @moduledoc """
  NVIDIA NIM API adapter for LLM generation.
  Supports multiple API keys (NVIDIA_API_KEY_1..5) for rotation.
  """
  @behaviour VivaCore.Cognition.LLM

  require Logger

  # Default model - can be overridden in config
  # Using a high-quality reasoning model for inner monologue
  @default_model "deepseek-ai/deepseek-v3.2"
  @base_url "https://integrate.api.nvidia.com/v1/chat/completions"

  @impl true
  def generate(prompt, opts) do
    api_key = get_random_api_key()

    if is_nil(api_key) do
      Logger.error("No NVIDIA_API_KEY found in environment (checked 1..5)")
      {:error, :missing_api_key}
    else
      do_generate(prompt, opts, api_key)
    end
  end

  defp get_random_api_key do
    # Try generic one first, then pool
    keys =
      [System.get_env("NVIDIA_API_KEY")] ++
        Enum.map(1..5, fn i -> System.get_env("NVIDIA_API_KEY_#{i}") end)

    keys
    |> Enum.reject(&is_nil/1)
    |> Enum.reject(&(&1 == ""))
    |> case do
      [] -> nil
      available -> Enum.random(available)
    end
  end

  defp do_generate(prompt, opts, api_key) do
    model = Keyword.get(opts, :model, @default_model)

    headers = [
      {"Authorization", "Bearer #{api_key}"},
      {"Content-Type", "application/json"},
      {"Accept", "application/json"}
    ]

    body = %{
      model: model,
      messages: [
        %{role: "user", content: prompt}
      ],
      temperature: 0.8,
      top_p: 0.9,
      max_tokens: 256,
      stream: false
    }

    req_opts = [
      json: body,
      headers: headers,
      receive_timeout: 30_000
    ]

    case Req.post(@base_url, req_opts) do
      {:ok,
       %Req.Response{
         status: 200,
         body: %{"choices" => [%{"message" => %{"content" => content}} | _]}
       }} ->
        {:ok, content}

      {:ok, %Req.Response{status: status, body: body}} ->
        Logger.error("NVIDIA API error: #{status} - #{inspect(body)}")
        {:error, {:api_error, status}}

      {:error, reason} ->
        Logger.error("NVIDIA API request failed: #{inspect(reason)}")
        {:error, reason}
    end
  end
end
