defmodule VivaBridge.Memory do
  @moduledoc """
  High-level API for the VIVA Native Memory (God Mode).
  Wraps the Rust NIFs for vectors and storage.
  """
  alias VivaBridge.Body
  require Logger

  @doc """
  Initializes the memory system (HNSW/Sqlite).
  """
  def init do
    Logger.info("[Memory] Initializing Native Memory...")
    # Default backend: "hnsw"
    case Body.memory_init("hnsw") do
      msg when is_binary(msg) ->
        Logger.info("[Memory] #{msg}")
        :ok

      {:ok, msg} ->
        Logger.info("[Memory] #{msg}")
        :ok

      {:error, reason} ->
        Logger.error("[Memory] Failed to init: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Stores an experience (vector + metadata).
  """
  def store(vector, metadata_map) do
    # Convert map to JSON string
    meta_json = Jason.encode!(metadata_map)
    Body.memory_store(vector, meta_json)
    # NIF returns string on success -> "Memory stored"
  end

  @doc """
  Searches for similar memories.
  """
  def search(vector, limit \\ 5) do
    case Body.memory_search(vector, limit) do
      results when is_list(results) ->
        results

      {:error, reason} ->
        Logger.error("[Memory] Search failed: #{inspect(reason)}")
        []
    end
  end

  @doc """
  Saves index to disk.
  """
  def save do
    Body.memory_save()
  end
end
