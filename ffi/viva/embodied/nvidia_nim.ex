defmodule Viva.Embodied.NvidiaNim do
  @moduledoc """
  NVIDIA NIM API Client for Vision Models

  Uses NVIDIA's cloud-hosted NIMs with NVCF Asset API:
  - nv-dinov2: Visual embeddings (1024-dim for classification)
  - ocdrnet: OCR (text detection + recognition)
  - nv-grounding-dino: Object detection with text prompts

  Flow: Upload image to NVCF → Get asset UUID → Call NIM with UUID
  """

  require Logger

  @base_url "https://ai.api.nvidia.com/v1/cv/nvidia"
  @assets_url "https://api.nvcf.nvidia.com/v2/nvcf/assets"

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
    with {:ok, asset_id} <- upload_asset(image_path),
         {:ok, response} <- call_dinov2(asset_id) do
      parse_embedding_response(response)
    end
  end

  @doc """
  Get visual embedding from raw bytes.
  """
  def get_embedding_bytes(image_bytes) when is_binary(image_bytes) do
    with {:ok, asset_id} <- upload_asset_bytes(image_bytes),
         {:ok, response} <- call_dinov2(asset_id) do
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
    with {:ok, asset_id} <- upload_asset(image_path),
         {:ok, response} <- call_ocr(asset_id) do
      parse_ocr_response(response)
    end
  end

  @doc """
  Extract text from raw bytes.
  """
  def read_text_bytes(image_bytes) when is_binary(image_bytes) do
    with {:ok, asset_id} <- upload_asset_bytes(image_bytes),
         {:ok, response} <- call_ocr(asset_id) do
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
    with {:ok, asset_id} <- upload_asset(image_path),
         {:ok, response} <- call_grounding(asset_id, prompt) do
      parse_detection_response(response)
    end
  end

  @doc """
  Full perception: embedding + OCR + detection in parallel.
  Uploads asset once, then runs all models.
  """
  def perceive(image_path, detect_prompt \\ "text. code. button. window.") do
    case upload_asset(image_path) do
      {:ok, asset_id} ->
        tasks = [
          Task.async(fn -> call_dinov2(asset_id) |> parse_result(&parse_embedding_response/1) end),
          Task.async(fn -> call_ocr(asset_id) |> parse_result(&parse_ocr_response/1) end),
          Task.async(fn -> call_grounding(asset_id, detect_prompt) |> parse_result(&parse_detection_response/1) end)
        ]

        [embedding_result, ocr_result, detection_result] = Task.await_many(tasks, 60_000)

        %{
          embedding: unwrap_or_nil(embedding_result),
          text: unwrap_or_nil(ocr_result),
          detections: unwrap_or_nil(detection_result)
        }

      {:error, reason} ->
        Logger.error("[NvidiaNim] Failed to upload asset: #{inspect(reason)}")
        %{embedding: nil, text: nil, detections: nil}
    end
  end

  defp parse_result({:ok, response}, parser), do: parser.(response)
  defp parse_result({:error, _} = err, _parser), do: err

  # ============================================================================
  # ASSET UPLOAD (NVCF)
  # ============================================================================

  defp upload_asset(image_path) do
    case File.read(image_path) do
      {:ok, bytes} -> upload_asset_bytes(bytes)
      {:error, reason} -> {:error, "Failed to read image: #{reason}"}
    end
  end

  defp upload_asset_bytes(image_bytes) do
    api_key = get_api_key()

    # Step 1: Request upload URL
    headers = [
      {~c"authorization", String.to_charlist("Bearer #{api_key}")},
      {~c"content-type", ~c"application/json"},
      {~c"accept", ~c"application/json"}
    ]

    body = Jason.encode!(%{
      "contentType" => "image/jpeg",
      "description" => "input"
    })

    request = {String.to_charlist(@assets_url), headers, ~c"application/json", body}

    case :httpc.request(:post, request, [{:timeout, 30_000}], []) do
      {:ok, {{_, 200, _}, _, response_body}} ->
        case Jason.decode(to_string(response_body)) do
          {:ok, %{"uploadUrl" => upload_url, "assetId" => asset_id}} ->
            # Step 2: Upload image to S3
            upload_to_s3(upload_url, image_bytes, asset_id)

          {:ok, other} ->
            Logger.error("[NvidiaNim] Unexpected asset response: #{inspect(other)}")
            {:error, "Unexpected asset response"}

          {:error, _} ->
            {:error, "Failed to parse asset response"}
        end

      {:ok, {{_, status, _}, _, response_body}} ->
        Logger.error("[NvidiaNim] Asset API error: #{status} - #{to_string(response_body)}")
        {:error, "Asset API error: #{status}"}

      {:error, reason} ->
        Logger.error("[NvidiaNim] Asset request failed: #{inspect(reason)}")
        {:error, "Asset request failed"}
    end
  end

  defp upload_to_s3(upload_url, image_bytes, asset_id) do
    # Convert to JPEG if needed
    jpeg_bytes = ensure_jpeg(image_bytes)

    Logger.debug("[NvidiaNim] Uploading asset #{asset_id}, image size: #{byte_size(jpeg_bytes)}")
    Logger.debug("[NvidiaNim] URL length: #{String.length(upload_url)}")

    # Use curl via shell script to avoid URL escaping issues
    id = :erlang.unique_integer([:positive])
    tmp_image = "/tmp/viva_upload_#{id}.jpg"
    tmp_url = "/tmp/viva_url_#{id}.txt"

    File.write!(tmp_image, jpeg_bytes)
    File.write!(tmp_url, upload_url)

    try do
      # Execute curl directly reading URL from file
      # Build command with explicit string concatenation to avoid %{} interpolation issues
      # Note: description header must match the value used in asset creation
      format_spec = ~S('%{http_code}')
      cmd = "URL=$(cat #{tmp_url}) && curl -s -o /dev/null -w #{format_spec} -X PUT \"$URL\" -H 'Content-Type: image/jpeg' -H 'x-amz-meta-nvcf-asset-description: input' --data-binary @#{tmp_image}"
      {output, _code} = System.cmd("bash", ["-c", cmd])

      status_str = String.trim(output)
      Logger.debug("[NvidiaNim] Upload curl output: #{status_str}")

      status = String.to_integer(status_str)

      if status in 200..299 do
        Logger.debug("[NvidiaNim] Asset uploaded: #{asset_id}")
        {:ok, asset_id}
      else
        Logger.error("[NvidiaNim] S3 upload error: status #{status}")
        {:error, "S3 upload error: #{status}"}
      end
    after
      File.rm(tmp_image)
      File.rm(tmp_url)
    end
  end

  defp ensure_jpeg(image_bytes) do
    # Check if already JPEG (starts with FFD8)
    case image_bytes do
      <<0xFF, 0xD8, _rest::binary>> ->
        image_bytes

      _ ->
        # Convert using ImageMagick
        tmp_in = "/tmp/viva_convert_in_#{:erlang.unique_integer([:positive])}"
        tmp_out = "/tmp/viva_convert_out_#{:erlang.unique_integer([:positive])}.jpg"

        File.write!(tmp_in, image_bytes)

        case System.cmd("convert", [tmp_in, "-quality", "90", tmp_out]) do
          {_, 0} ->
            result = File.read!(tmp_out)
            File.rm(tmp_in)
            File.rm(tmp_out)
            result

          _ ->
            File.rm(tmp_in)
            # Return original if conversion fails
            image_bytes
        end
    end
  end

  # ============================================================================
  # NIM API CALLS
  # ============================================================================

  defp call_dinov2(asset_id) do
    url = "#{@base_url}/#{@dinov2_model}"
    post_with_asset(url, asset_id, %{"messages" => []})
  end

  defp call_ocr(asset_id) do
    url = "#{@base_url}/#{@ocr_model}"
    # OCR returns 302 redirect to ZIP file with results
    post_ocr_with_asset(url, asset_id)
  end

  defp post_ocr_with_asset(url, asset_id) do
    api_key = get_api_key()

    headers = [
      {~c"authorization", String.to_charlist("Bearer #{api_key}")},
      {~c"content-type", ~c"application/json"},
      {~c"accept", ~c"application/json"},
      {~c"nvcf-input-asset-references", String.to_charlist(asset_id)},
      {~c"nvcf-function-asset-ids", String.to_charlist(asset_id)}
    ]

    body = Jason.encode!(%{"image" => asset_id})
    request = {String.to_charlist(url), headers, ~c"application/json", body}

    case :httpc.request(:post, request, [{:timeout, 60_000}, {:autoredirect, false}], []) do
      {:ok, {{_, 302, _}, response_headers, _}} ->
        # Get redirect location
        location = get_header(response_headers, "location")
        if location do
          fetch_ocr_result(location)
        else
          {:error, "No redirect location in OCR response"}
        end

      {:ok, {{_, 200, _}, _, response_body}} ->
        # Direct response (unlikely but handle it)
        case Jason.decode(to_string(response_body)) do
          {:ok, parsed} -> {:ok, parsed}
          {:error, _} -> {:error, "Failed to parse response"}
        end

      {:ok, {{_, status, _}, _, response_body}} ->
        Logger.error("[NvidiaNim] OCR API error: #{status} - #{to_string(response_body)}")
        {:error, "OCR API error: #{status}"}

      {:error, reason} ->
        Logger.error("[NvidiaNim] OCR request failed: #{inspect(reason)}")
        {:error, "OCR request failed"}
    end
  end

  defp fetch_ocr_result(location) do
    # Fetch ZIP from S3 (no auth needed, pre-signed URL)
    case :httpc.request(:get, {String.to_charlist(location), []}, [{:timeout, 30_000}], [body_format: :binary]) do
      {:ok, {{_, 200, _}, _, zip_data}} ->
        extract_ocr_from_zip(zip_data)

      {:ok, {{_, status, _}, _, _}} ->
        {:error, "Failed to fetch OCR result: #{status}"}

      {:error, reason} ->
        {:error, "Failed to fetch OCR result: #{inspect(reason)}"}
    end
  end

  defp extract_ocr_from_zip(zip_data) do
    # Save ZIP to temp file
    tmp_zip = "/tmp/viva_ocr_#{:erlang.unique_integer([:positive])}.zip"
    tmp_dir = "/tmp/viva_ocr_#{:erlang.unique_integer([:positive])}"

    File.write!(tmp_zip, zip_data)
    File.mkdir_p!(tmp_dir)

    try do
      # Extract ZIP
      {_, 0} = System.cmd("unzip", ["-o", "-d", tmp_dir, tmp_zip])

      # Find .response file
      response_files = Path.wildcard("#{tmp_dir}/*.response")

      case response_files do
        [response_file | _] ->
          content = File.read!(response_file)
          Jason.decode(content)

        [] ->
          {:error, "No response file in OCR ZIP"}
      end
    after
      File.rm(tmp_zip)
      File.rm_rf(tmp_dir)
    end
  end

  defp call_grounding(asset_id, prompt) do
    url = "#{@base_url}/#{@grounding_model}"
    # Grounding DINO uses messages format with media_url referencing asset
    body = %{
      "model" => "Grounding-Dino",
      "messages" => [
        %{
          "role" => "user",
          "content" => [
            %{"type" => "text", "text" => prompt},
            %{
              "type" => "media_url",
              "media_url" => %{
                "url" => "data:image/jpeg;asset_id,#{asset_id}"
              }
            }
          ]
        }
      ],
      "threshold" => 0.3
    }
    post_with_asset(url, asset_id, body)
  end

  defp post_with_asset(url, asset_id, body) do
    api_key = get_api_key()

    headers = [
      {~c"authorization", String.to_charlist("Bearer #{api_key}")},
      {~c"content-type", ~c"application/json"},
      {~c"accept", ~c"application/json"},
      {~c"nvcf-input-asset-references", String.to_charlist(asset_id)},
      {~c"nvcf-function-asset-ids", String.to_charlist(asset_id)}
    ]

    json_body = Jason.encode!(body)
    request = {String.to_charlist(url), headers, ~c"application/json", json_body}

    # Disable autoredirect to handle 302 manually (redirect to S3 without auth headers)
    case :httpc.request(:post, request, [{:timeout, 60_000}, {:autoredirect, false}], []) do
      {:ok, {{_, 200, _}, _, response_body}} ->
        case Jason.decode(to_string(response_body)) do
          {:ok, parsed} -> {:ok, parsed}
          {:error, _} -> {:error, "Failed to parse response"}
        end

      {:ok, {{_, 302, _}, response_headers, _}} ->
        # Redirect to S3 - fetch result without auth headers
        location = get_header(response_headers, "location")
        if location do
          fetch_async_result(location)
        else
          {:error, "No redirect location in response"}
        end

      {:ok, {{_, 202, _}, response_headers, response_body}} ->
        # Async request - poll for result
        nvcf_reqid = get_header(response_headers, "nvcf-reqid")
        case {nvcf_reqid, Jason.decode(to_string(response_body))} do
          {nil, {:ok, %{"reqId" => req_id}}} ->
            poll_for_result(url, req_id)
          {req_id, _} when is_binary(req_id) ->
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

  defp fetch_async_result(location) do
    # Fetch result from S3 without auth headers (pre-signed URL)
    case :httpc.request(:get, {String.to_charlist(location), []}, [{:timeout, 60_000}], [body_format: :binary]) do
      {:ok, {{_, 200, _}, response_headers, body}} ->
        content_type = get_header(response_headers, "content-type") || ""

        cond do
          String.contains?(content_type, "application/json") ->
            # Direct JSON response
            case Jason.decode(body) do
              {:ok, parsed} -> {:ok, parsed}
              {:error, _} -> {:error, "Failed to parse JSON response"}
            end

          String.contains?(content_type, "application/zip") or String.starts_with?(body, "PK") ->
            # ZIP file (like OCR) - extract JSON from .response file
            extract_json_from_zip(body)

          true ->
            # Try JSON first, then ZIP
            case Jason.decode(body) do
              {:ok, parsed} -> {:ok, parsed}
              {:error, _} -> extract_json_from_zip(body)
            end
        end

      {:ok, {{_, status, _}, _, body}} ->
        Logger.error("[NvidiaNim] Fetch async result error: #{status}")
        {:error, "Fetch error: #{status} - #{to_string(body)}"}

      {:error, reason} ->
        {:error, "Fetch async result failed: #{inspect(reason)}"}
    end
  end

  defp extract_json_from_zip(zip_data) do
    tmp_zip = "/tmp/viva_result_#{:erlang.unique_integer([:positive])}.zip"
    tmp_dir = "/tmp/viva_result_#{:erlang.unique_integer([:positive])}"

    File.write!(tmp_zip, zip_data)
    File.mkdir_p!(tmp_dir)

    try do
      case System.cmd("unzip", ["-o", "-d", tmp_dir, tmp_zip], stderr_to_stdout: true) do
        {_, 0} ->
          # Find .response or .json file
          response_files = Path.wildcard("#{tmp_dir}/*.response") ++ Path.wildcard("#{tmp_dir}/*.json")

          case response_files do
            [file | _] ->
              content = File.read!(file)
              Jason.decode(content)

            [] ->
              {:error, "No response file in ZIP"}
          end

        {output, _} ->
          {:error, "Failed to extract ZIP: #{output}"}
      end
    after
      File.rm(tmp_zip)
      File.rm_rf(tmp_dir)
    end
  end

  defp get_header(headers, name) do
    name_charlist = String.to_charlist(name)
    Enum.find_value(headers, fn
      {^name_charlist, value} -> to_string(value)
      _ -> nil
    end)
  end

  defp poll_for_result(base_url, req_id, attempts \\ 30) do
    if attempts <= 0 do
      {:error, "Timeout waiting for result"}
    else
      Process.sleep(1000)

      api_key = get_api_key()
      url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/#{req_id}"

      headers = [
        {~c"authorization", String.to_charlist("Bearer #{api_key}")},
        {~c"accept", ~c"application/json"}
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

  defp parse_embedding_response(%{"metadata" => [%{"embedding" => embedding} | _]}) do
    {:ok, embedding}
  end

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

  defp parse_ocr_response(%{"metadata" => detections}) when is_list(detections) do
    # OCDRNet format: metadata is list of detections with label, polygon, confidence
    blocks = Enum.map(detections, fn det ->
      polygon = det["polygon"] || %{}
      %{
        text: det["label"],
        confidence: det["confidence"] || 1.0,
        bbox: [polygon["x1"], polygon["y1"], polygon["x3"], polygon["y3"]]
      }
    end)

    # Combine all labels into text
    text = blocks
    |> Enum.map(& &1.text)
    |> Enum.join(" ")

    {:ok, %{text: text, blocks: blocks}}
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

  # Grounding DINO format with metadata
  defp parse_detection_response(%{"metadata" => metadata}) when is_list(metadata) do
    parsed = Enum.flat_map(metadata, fn item ->
      boxes = item["bboxes"] || []
      labels = item["labels"] || []
      scores = item["scores"] || []

      Enum.zip([labels, scores, boxes])
      |> Enum.map(fn {label, score, bbox} ->
        %{
          label: label,
          confidence: score,
          bbox: normalize_bbox(bbox)
        }
      end)
    end)

    {:ok, parsed}
  end

  # Grounding DINO raw tensor format (pred_boxes, pred_logits)
  defp parse_detection_response(%{"pred_boxes" => boxes, "pred_logits" => logits}) do
    # Boxes are in cxcywh format [center_x, center_y, width, height]
    # Convert to [x, y, w, h] format
    detections =
      Enum.zip(boxes, logits)
      |> Enum.filter(fn {_, logit} -> logit > 0.3 end)  # Threshold
      |> Enum.map(fn {[cx, cy, w, h], confidence} ->
        %{
          label: "object",
          confidence: confidence,
          bbox: [cx - w/2, cy - h/2, w, h]
        }
      end)

    {:ok, detections}
  end

  # Choice format (wrapper) - content can be string or map
  defp parse_detection_response(%{"choices" => [%{"message" => %{"content" => content}} | _]}) when is_binary(content) do
    case Jason.decode(content) do
      {:ok, parsed} -> parse_detection_response(parsed)
      {:error, _} -> {:ok, []}  # No detections if can't parse
    end
  end

  defp parse_detection_response(%{"choices" => [%{"message" => %{"content" => content}} | _]}) when is_map(content) do
    parse_detection_response(content)
  end

  # Grounding DINO cloud format: boundingBoxes with phrase/bboxes/confidence arrays
  defp parse_detection_response(%{"boundingBoxes" => boxes}) when is_list(boxes) do
    parsed = Enum.flat_map(boxes, fn box_group ->
      phrase = box_group["phrase"] || "object"
      bboxes = box_group["bboxes"] || []
      confidences = box_group["confidence"] || []

      # Each phrase can have multiple detections
      Enum.zip(bboxes, confidences)
      |> Enum.map(fn {[x, y, w, h], conf} ->
        %{
          label: phrase,
          confidence: conf,
          bbox: [x, y, w, h]
        }
      end)
    end)

    {:ok, parsed}
  end

  defp parse_detection_response(response) do
    Logger.warning("[NvidiaNim] Unexpected detection response: #{inspect(response)}")
    {:error, "Unexpected response format"}
  end

  defp normalize_bbox(bbox) when is_list(bbox), do: bbox
  defp normalize_bbox(bbox) when is_map(bbox) do
    [bbox["x"] || 0, bbox["y"] || 0, bbox["width"] || 0, bbox["height"] || 0]
  end
  defp normalize_bbox(_), do: [0, 0, 0, 0]

  # ============================================================================
  # HELPERS
  # ============================================================================

  defp get_api_key do
    System.get_env("NVIDIA_API_KEY") ||
      Application.get_env(:viva, :nvidia_api_key) ||
      raise "NVIDIA_API_KEY not set"
  end

  defp unwrap_or_nil({:ok, value}), do: value
  defp unwrap_or_nil({:error, _}), do: nil
end
