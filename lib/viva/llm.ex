defmodule Viva.Llm do
  use Rustler, otp_app: :viva_nx_project, crate: "viva_llm"

  def load_model(_path, _gpu_layers), do: :erlang.nif_error(:nif_not_loaded)

  def predict(_resource, _prompt), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Returns a tuple {total_bytes, free_bytes} of system RAM.
  """
  def get_memory_status(), do: :erlang.nif_error(:nif_not_loaded)

  def native_check(), do: :erlang.nif_error(:nif_not_loaded)
end
