defmodule VivaBridge.Firmware.Codegen do
  @moduledoc """
  Generates Arduino firmware (.ino) from EEx templates.

  VIVA uses this to evolve her own body code through genetic programming.
  The genotype (Elixir map) is transformed into phenotype (Arduino C++).

  ## Genotype Structure

      %{
        generation: 0,
        fitness: 0.0,
        pins: %{speaker: 8, buzzer: 9, fan_pwm: 10, fan_tach: 2, led: 13},
        timer: %{prescaler: 1, top: 639},  # 25kHz for Intel fans
        emotions: %{
          joy:     %{pwm: 200, melody: [{262, 100}, {330, 100}, {392, 100}, {523, 200}]},
          sad:     %{pwm: 80,  melody: [{440, 300}, {392, 300}, {330, 300}, {294, 500}]},
          fear:    %{pwm: 255, melody: [{233, 50}, {247, 50}], repeat: 5},
          calm:    %{pwm: 60,  melody: [{262, 300}, {330, 300}, {392, 500}]},
          curious: %{pwm: 150, melody: [{262, 100}, {294, 100}, {330, 100}, {392, 200}]},
          love:    %{pwm: 120, melody: [{262, 150}, {330, 150}, {392, 150}, {330, 150}, {262, 300}]}
        },
        harmony_ratio: 2.0,  # Octave up on buzzer
        serial_baud: 9600,
        serial_timeout: 50
      }
  """

  require VivaLog

  @template_path "priv/templates/viva_music.ino.eex"

  @doc """
  Returns the default genotype - baseline firmware configuration.
  """
  def default_genotype do
    %{
      generation: 0,
      fitness: 0.0,
      pins: %{speaker: 8, buzzer: 9, fan_pwm: 10, fan_tach: 2, led: 13},
      timer: %{prescaler: 1, top: 639},
      emotions: %{
        joy: %{pwm: 200, melody: [{262, 100}, {330, 100}, {392, 100}, {523, 200}]},
        sad: %{pwm: 80, melody: [{440, 300}, {392, 300}, {330, 300}, {294, 500}]},
        fear: %{pwm: 255, melody: [{233, 50}, {247, 50}], repeat: 5},
        calm: %{pwm: 60, melody: [{262, 300}, {330, 300}, {392, 500}]},
        curious: %{pwm: 150, melody: [{262, 100}, {294, 100}, {330, 100}, {392, 200}]},
        love: %{pwm: 120, melody: [{262, 150}, {330, 150}, {392, 150}, {330, 150}, {262, 300}]}
      },
      harmony_ratio: 2.0,
      serial_baud: 9600,
      serial_timeout: 50
    }
  end

  @doc """
  Generates Arduino .ino code from a genotype.

  Returns `{:ok, ino_code}` or `{:error, reason}`.
  """
  def generate(genotype) do
    template_path = Application.app_dir(:viva_bridge, @template_path)

    if File.exists?(template_path) do
      try do
        code = EEx.eval_file(template_path, assigns: [g: genotype])
        {:ok, code}
      rescue
        e ->
          VivaLog.error(:codegen, :template_error, error: inspect(e))
          {:error, {:template_error, e}}
      end
    else
      {:error, {:template_not_found, template_path}}
    end
  end

  @doc """
  Generates .ino and writes to a file.

  Returns `{:ok, file_path}` or `{:error, reason}`.
  """
  def generate_to_file(genotype, output_path) do
    case generate(genotype) do
      {:ok, code} ->
        case File.write(output_path, code) do
          :ok -> {:ok, output_path}
          {:error, reason} -> {:error, {:write_failed, reason}}
        end

      error ->
        error
    end
  end

  @doc """
  Validates a genotype has all required fields and values in bounds.
  """
  def validate_genotype(genotype) do
    with :ok <- validate_pins(genotype.pins),
         :ok <- validate_timer(genotype.timer),
         :ok <- validate_emotions(genotype.emotions),
         :ok <- validate_bounds(genotype) do
      :ok
    end
  end

  # Pin validation - hardware safety
  defp validate_pins(pins) do
    required = [:speaker, :buzzer, :fan_pwm, :fan_tach, :led]
    missing = required -- Map.keys(pins)

    if Enum.empty?(missing) do
      :ok
    else
      {:error, {:missing_pins, missing}}
    end
  end

  # Timer validation - must produce valid PWM frequency
  defp validate_timer(%{prescaler: p, top: t})
       when p in [1, 8, 64, 256, 1024] and t > 0 and t <= 65535 do
    freq = 16_000_000 / (p * (1 + t))

    if freq >= 20_000 and freq <= 30_000 do
      :ok
    else
      {:error, {:pwm_freq_out_of_range, freq}}
    end
  end

  defp validate_timer(_), do: {:error, :invalid_timer_config}

  # Emotion validation
  defp validate_emotions(emotions) do
    required = [:joy, :sad, :fear, :calm, :curious, :love]
    missing = required -- Map.keys(emotions)

    if Enum.empty?(missing) do
      # Check PWM bounds
      invalid =
        emotions
        |> Enum.filter(fn {_name, %{pwm: pwm}} -> pwm < 0 or pwm > 255 end)
        |> Enum.map(&elem(&1, 0))

      if Enum.empty?(invalid) do
        :ok
      else
        {:error, {:pwm_out_of_bounds, invalid}}
      end
    else
      {:error, {:missing_emotions, missing}}
    end
  end

  # General bounds validation
  defp validate_bounds(g) do
    cond do
      g.harmony_ratio < 1.0 or g.harmony_ratio > 4.0 ->
        {:error, {:harmony_ratio_out_of_bounds, g.harmony_ratio}}

      g.serial_baud not in [9600, 19200, 38400, 57600, 115_200] ->
        {:error, {:invalid_baud_rate, g.serial_baud}}

      g.serial_timeout < 10 or g.serial_timeout > 1000 ->
        {:error, {:timeout_out_of_bounds, g.serial_timeout}}

      true ->
        :ok
    end
  end

  @doc """
  Returns a summary of the genotype for logging.
  """
  def summarize(genotype) do
    freq = 16_000_000 / (genotype.timer.prescaler * (1 + genotype.timer.top))

    %{
      generation: genotype.generation,
      fitness: genotype.fitness,
      pwm_freq_khz: Float.round(freq / 1000, 1),
      harmony_ratio: genotype.harmony_ratio,
      emotion_pwms: genotype.emotions |> Enum.map(fn {k, v} -> {k, v.pwm} end) |> Map.new()
    }
  end
end
