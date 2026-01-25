defmodule Viva.Embodied.NvidiaNim do
  @moduledoc """
  NVIDIA NIM API Client for Vision Models

  Uses NVIDIA's cloud-hosted NIMs:
  - nv-dinov2: Visual embeddings (1024-dim for classification)
  - ocdrnet: OCR (text detection + recognition)
  - nv-grounding-dino: Object detection with text prompts

  API Docs: https://docs.api.nvidia.com/nim/reference/visual-models-apis
  """

  require Logger

  @base_url "https://ai.api.nvidia.com/v1/cv/nvidia"

  # Models
  @dinov2_model "nv-dinov2"
  @ocr_model "ocdrnet"
  @grounding_model "nv-grounding-dino"

  # ============================================================================
  # PUBLIC API
  # ============================================================================

  @doc """
  Get visual embeddings from an image using NV-DINOv2.
  Returns a 1024-dimensional embedding vector.

  ## Examples

      {:ok, embedding} = NvidiaNim.get_embedding("/path/to/image.png")
      # embedding is a list of 1024 floats
  """
  def get_embedding(image_path) do
    with {:ok, base64} <- encode_image(image_path),
         {:ok, response} <- call_dinov2(base64) do
      parse_embedding_response(response)
    end
  end

  @doc """
  Get visual embedding from raw bytes.
  """
  def get_embedding_bytes(image_bytes) when is_binary(image_bytes) do
    base64 = Base.encode64(image_bytes)
    with {:ok, response} <- call_dinov2(base64) do
      parse_embedding_response(response)
    end
  end

  @doc """
  Extract text from an image using OCDRNet.
  Returns detected text blocks with positions.

  ## Examples

      {:ok, result} = NvidiaNim.read_text("/path/to/screenshot.png")
      # result = %{text: "Hello World", blocks: [...]}
  """
  def read_text(image_path) do
    with {:ok, base64} <- encode_image(image_path),
         {:ok, response} <- call_ocr(base64) do
      parse_ocr_response(response)
    end
  end

  @doc """
  Extract text from raw bytes.
  """
  def read_text_bytes(image_bytes) when is_binary(image_bytes) do
    base64 = Base.encode64(image_bytes)
    with {:ok, response} <- call_ocr(base64) do
      parse_ocr_response(response)
    end
  end

  @doc """
  Detect objects in an image using Grounding DINO.
  Provide text prompts for what to detect.

  ## Examples

      {:ok, detections} = NvidiaNim.detect_objects("/path/to/image.png", "person. cat. dog.")
      # detections = [%{label: "person", confidence: 0.95, bbox: [x, y, w, h]}, ...]
  """
  def detect_objects(image_path, prompt) do
    with {:ok, base64} <- encode_image(image_path),
         {:ok, response} <- call_grounding(base64, prompt) do
      parse_detection_response(response)
    end
  end

  @doc """
  Full perception: embedding + OCR + detection in parallel.
  """
  def perceive(image_path, detect_prompt \\ "text. code. button. window.") do
    tasks = [
      Task.async(fn -> get_embedding(image_path) end),
      Task.async(fn -> read_text(image_path) end),
      Task.async(fn -> detect_objects(image_path, detect_prompt) end)
    ]

    [embedding_result, ocr_result, detection_result] =
      Task.await_many(tasks, 30_000)

    %{
      embedding: unwrap_or_nil(embedding_result),
      text: unwrap_or_nil(ocr_result),
      detections: unwrap_or_nil(detection_result)
    }
  end

  # ============================================================================
  # API CALLS
  # ============================================================================

  defp call_dinov2(base64_image) do
    url = "#{@base_url}/#{@dinov2_model}"

    body = %{
      "input" => [
        %{"type" => "image_url", "url" => "data:image/png;base64,#{base64_image}"}
      ]
    }

    post_request(url, body)
  end

  defp call_ocr(base64_image) do
    url = "#{@base_url}/#{@ocr_model}"

    # OCDRNet cloud API expects "image" as data URL
    body = %{
      "image" => "data:image/png;base64,#{base64_image}",
      "render_label" => false
    }

    post_request(url, body)
  end

  defp call_grounding(base64_image, prompt) do
    url = "#{@base_url}/#{@grounding_model}"

    # Grounding DINO uses input array + prompt
    body = %{
      "input" => [
        %{"type" => "image_url", "url" => "data:image/png;base64,#{base64_image}"}
      ],
      "prompt" => prompt,
      "threshold" => 0.3
    }

    post_request(url, body)
  end

  defp post_request(url, body) do
    api_key = get_api_key()

    headers = [
      {'authorization', String.to_charlist("Bearer #{api_key}")},
      {'content-type', 'application/json'},
      {'accept', 'application/json'}
    ]

    json_body = Jason.encode!(body)
    request = {String.to_charlist(url), headers, 'application/json', json_body}

    case :httpc.request(:post, request, [{:timeout, 30_000}], []) do
      {:ok, {{_, 200, _}, _, response_body}} ->
        case Jason.decode(to_string(response_body)) do
          {:ok, parsed} -> {:ok, parsed}
          {:error, _} -> {:error, "Failed to parse response"}
        end

      {:ok, {{_, 202, _}, _, response_body}} ->
        # Async request - poll for result
        case Jason.decode(to_string(response_body)) do
          {:ok, %{"reqId" => req_id}} ->
            poll_for_result(url, req_id)
          _ ->
            {:error, "Missing reqId in async response"}
        end

      {:ok, {{_, status, _}, _, response_body}} ->
        Logger.error("[NvidiaNim] API error: #{status} - #{to_string(response_body)}")
        {:error, "API error: #{status}"}

      {:error, reason} ->
        Logger.error("[NvidiaNim] Request failed: #{inspect(reason)}")
        {:error, "Request failed: #{inspect(reason)}"}
    end
  end

  defp poll_for_result(base_url, req_id, attempts \\ 10) do
    if attempts <= 0 do
      {:error, "Timeout waiting for result"}
    else
      Process.sleep(1000)

      url = "#{base_url}/status/#{req_id}"
      api_key = get_api_key()

      headers = [
        {'authorization', String.to_charlist("Bearer #{api_key}")},
        {'accept', 'application/json'}
      ]

      case :httpc.request(:get, {String.to_charlist(url), headers}, [{:timeout, 30_000}], []) do
        {:ok, {{_, 200, _}, _, response_body}} ->
          case Jason.decode(to_string(response_body)) do
            {:ok, parsed} -> {:ok, parsed}
            {:error, _} -> {:error, "Failed to parse poll response"}
          end

        {:ok, {{_, 202, _}, _, _}} ->
          poll_for_result(base_url, req_id, attempts - 1)

        {:ok, {{_, status, _}, _, response_body}} ->
          {:error, "Poll error: #{status} - #{to_string(response_body)}"}

        {:error, reason} ->
          {:error, "Poll request failed: #{inspect(reason)}"}
      end
    end
  end

  # ============================================================================
  # RESPONSE PARSERS
  # ============================================================================

  defp parse_embedding_response(%{"data" => [%{"embedding" => embedding}]}) do
    {:ok, embedding}
  end

  defp parse_embedding_response(%{"embedding" => embedding}) do
    {:ok, embedding}
  end

  defp parse_embedding_response(response) do
    Logger.warning("[NvidiaNim] Unexpected embedding response: #{inspect(response)}")
    {:error, "Unexpected response format"}
  end

  defp parse_ocr_response(%{"text" => text, "metadata" => %{"text_detections" => blocks}}) do
    parsed_blocks = Enum.map(blocks, fn block ->
      %{
        text: block["text"],
        confidence: block["confidence"],
        bbox: block["bounding_box"]
      }
    end)

    {:ok, %{text: text, blocks: parsed_blocks}}
  end

  defp parse_ocr_response(%{"predictions" => predictions}) do
    # Alternative response format
    text = predictions
    |> Enum.map(& &1["text"])
    |> Enum.join(" ")

    blocks = Enum.map(predictions, fn pred ->
      %{
        text: pred["text"],
        confidence: pred["confidence"] || 1.0,
        bbox: pred["bbox"] || pred["bounding_box"]
      }
    end)

    {:ok, %{text: text, blocks: blocks}}
  end

  defp parse_ocr_response(response) do
    Logger.warning("[NvidiaNim] Unexpected OCR response: #{inspect(response)}")
    {:error, "Unexpected response format"}
  end

  defp parse_detection_response(%{"detections" => detections}) do
    parsed = Enum.map(detections, fn det ->
      %{
        label: det["label"],
        confidence: det["confidence"],
        bbox: det["bbox"] || det["bounding_box"]
      }
    end)

    {:ok, parsed}
  end

  defp parse_detection_response(response) do
    Logger.warning("[NvidiaNim] Unexpected detection response: #{inspect(response)}")
    {:error, "Unexpected response format"}
  end

  # ============================================================================
  # HELPERS
  # ============================================================================

  defp encode_image(path) do
    case File.read(path) do
      {:ok, bytes} -> {:ok, Base.encode64(bytes)}
      {:error, reason} -> {:error, "Failed to read image: #{reason}"}
    end
  end

  defp get_api_key do
    System.get_env("NVIDIA_API_KEY") ||
      Application.get_env(:viva, :nvidia_api_key) ||
      raise "NVIDIA_API_KEY not set"
  end

  defp unwrap_or_nil({:ok, value}), do: value
  defp unwrap_or_nil({:error, _}), do: nil
end
