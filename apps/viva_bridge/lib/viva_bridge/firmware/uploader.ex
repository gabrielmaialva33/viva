defmodule VivaBridge.Firmware.Uploader do
  @moduledoc """
  Compiles and uploads Arduino firmware using arduino-cli.

  VIVA uses this to deploy evolved firmware to her body.
  Includes safety checks: verify alive after upload, rollback on failure.

  ## Flow

      genotype → codegen → .ino → compile → upload → verify → success
                                                  ↓
                                            rollback on failure
  """

  require Logger

  @arduino_cli System.get_env("ARDUINO_CLI_PATH", "/home/mrootx/.local/bin/arduino-cli")
  @fqbn "arduino:avr:nano"
  @sketch_dir "/tmp/viva_firmware"
  @backup_dir "/tmp/viva_firmware_backup"
  @verify_timeout 5_000
  @upload_timeout 30_000

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Full pipeline: write sketch, compile, upload, verify.

  Returns `{:ok, %{port: port, generation: gen}}` or `{:error, reason}`.
  """
  def deploy(ino_code, opts \\ []) do
    port = Keyword.get(opts, :port)
    generation = Keyword.get(opts, :generation, 0)

    with :ok <- backup_current(),
         {:ok, sketch_path} <- write_sketch(ino_code),
         {:ok, _} <- compile(sketch_path),
         {:ok, detected_port} <- detect_port(port),
         {:ok, _} <- upload(sketch_path, detected_port),
         :ok <- verify_alive(detected_port) do
      Logger.info("[Uploader] Deploy successful: gen=#{generation} port=#{detected_port}")
      {:ok, %{port: detected_port, generation: generation}}
    else
      {:error, reason} = error ->
        Logger.error("[Uploader] Deploy failed: #{inspect(reason)}, attempting rollback")
        rollback()
        error
    end
  end

  @doc """
  Compile only (no upload). Useful for testing genotypes.

  Returns `{:ok, sketch_path}` or `{:error, reason}`.
  """
  def compile_only(ino_code) do
    with {:ok, sketch_path} <- write_sketch(ino_code),
         {:ok, _output} <- compile(sketch_path) do
      {:ok, sketch_path}
    end
  end

  @doc """
  Detect Arduino port automatically.

  Returns `{:ok, port}` or `{:error, :no_arduino_found}`.
  """
  def detect_port(nil), do: detect_port()
  def detect_port(port) when is_binary(port), do: {:ok, port}

  def detect_port do
    case System.cmd(@arduino_cli, ["board", "list", "--format", "json"], stderr_to_stdout: true) do
      {output, 0} ->
        parse_board_list(output)

      {error, _} ->
        Logger.error("[Uploader] Failed to list boards: #{error}")
        {:error, :board_list_failed}
    end
  end

  @doc """
  Check if Arduino is responding to PING command.

  Returns `:ok` or `{:error, reason}`.
  """
  def verify_alive(port) do
    Logger.info("[Uploader] Verifying firmware on #{port}...")

    # Wait for Arduino to boot after upload
    Process.sleep(2_000)

    # Try to connect and ping
    case VivaBridge.Music.connect(port) do
      {:ok, _} ->
        Process.sleep(500)
        case ping_with_retry(3) do
          :ok ->
            Logger.info("[Uploader] Firmware responding on #{port}")
            :ok

          {:error, reason} ->
            Logger.error("[Uploader] Firmware not responding: #{inspect(reason)}")
            {:error, {:verify_failed, reason}}
        end

      {:error, reason} ->
        Logger.error("[Uploader] Failed to connect: #{inspect(reason)}")
        {:error, {:connect_failed, reason}}
    end
  end

  @doc """
  Get info about installed arduino-cli and cores.
  """
  def info do
    version = case System.cmd(@arduino_cli, ["version"], stderr_to_stdout: true) do
      {output, 0} -> String.trim(output)
      _ -> "unknown"
    end

    cores = case System.cmd(@arduino_cli, ["core", "list"], stderr_to_stdout: true) do
      {output, 0} -> String.trim(output)
      _ -> "none"
    end

    %{
      arduino_cli: @arduino_cli,
      version: version,
      fqbn: @fqbn,
      cores: cores
    }
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp write_sketch(ino_code) do
    sketch_path = Path.join(@sketch_dir, "viva_evolved")
    ino_file = Path.join(sketch_path, "viva_evolved.ino")

    with :ok <- File.mkdir_p(sketch_path),
         :ok <- File.write(ino_file, ino_code) do
      Logger.debug("[Uploader] Wrote sketch to #{ino_file}")
      {:ok, sketch_path}
    else
      {:error, reason} ->
        Logger.error("[Uploader] Failed to write sketch: #{inspect(reason)}")
        {:error, {:write_failed, reason}}
    end
  end

  defp compile(sketch_path) do
    Logger.info("[Uploader] Compiling #{sketch_path}...")

    args = ["compile", "--fqbn", @fqbn, sketch_path]

    case System.cmd(@arduino_cli, args, stderr_to_stdout: true) do
      {output, 0} ->
        Logger.info("[Uploader] Compile successful")
        Logger.debug("[Uploader] #{output}")
        {:ok, output}

      {error, code} ->
        Logger.error("[Uploader] Compile failed (code #{code}): #{error}")
        {:error, {:compile_failed, error}}
    end
  end

  defp upload(sketch_path, port) do
    Logger.info("[Uploader] Uploading to #{port}...")

    # Disconnect Music module if connected (release serial port)
    VivaBridge.Music.disconnect()
    Process.sleep(500)

    args = ["upload", "-p", port, "--fqbn", @fqbn, sketch_path]

    case System.cmd(@arduino_cli, args, stderr_to_stdout: true) do
      {output, 0} ->
        Logger.info("[Uploader] Upload successful")
        Logger.debug("[Uploader] #{output}")
        {:ok, output}

      {error, code} ->
        Logger.error("[Uploader] Upload failed (code #{code}): #{error}")
        {:error, {:upload_failed, error}}
    end
  end

  defp parse_board_list(json) do
    case Jason.decode(json) do
      {:ok, %{"detected_ports" => ports}} when is_list(ports) ->
        # Find Arduino Nano or compatible
        arduino = Enum.find(ports, fn port ->
          boards = get_in(port, ["matching_boards"]) || []
          Enum.any?(boards, fn board ->
            fqbn = board["fqbn"] || ""
            String.contains?(fqbn, "arduino:avr")
          end)
        end)

        case arduino do
          %{"port" => %{"address" => address}} ->
            {:ok, address}

          nil ->
            # Fallback: any USB serial port
            fallback_port()
        end

      {:ok, _} ->
        fallback_port()

      {:error, _} ->
        fallback_port()
    end
  end

  defp fallback_port do
    # Try common Arduino ports
    candidates = [
      "/dev/ttyUSB0",
      "/dev/ttyUSB1",
      "/dev/ttyACM0",
      "/dev/ttyACM1"
    ]

    case Enum.find(candidates, &File.exists?/1) do
      nil -> {:error, :no_arduino_found}
      port -> {:ok, port}
    end
  end

  defp ping_with_retry(0), do: {:error, :no_response}
  defp ping_with_retry(attempts) do
    case VivaBridge.Music.ping() do
      :ok -> :ok
      {:error, _} ->
        Process.sleep(1_000)
        ping_with_retry(attempts - 1)
    end
  end

  defp backup_current do
    # Backup current working firmware
    current_ino = Path.join([@sketch_dir, "viva_evolved", "viva_evolved.ino"])

    if File.exists?(current_ino) do
      backup_path = Path.join(@backup_dir, "viva_evolved")
      File.mkdir_p!(backup_path)
      File.cp!(current_ino, Path.join(backup_path, "viva_evolved.ino"))
      Logger.debug("[Uploader] Backed up current firmware")
    end

    :ok
  end

  defp rollback do
    backup_ino = Path.join([@backup_dir, "viva_evolved", "viva_evolved.ino"])

    if File.exists?(backup_ino) do
      Logger.warning("[Uploader] Rolling back to previous firmware...")

      with {:ok, port} <- detect_port(),
           sketch_path = Path.join(@backup_dir, "viva_evolved"),
           {:ok, _} <- upload(sketch_path, port) do
        Logger.info("[Uploader] Rollback successful")
        :ok
      else
        {:error, reason} ->
          Logger.error("[Uploader] Rollback failed: #{inspect(reason)}")
          {:error, {:rollback_failed, reason}}
      end
    else
      Logger.warning("[Uploader] No backup available for rollback")
      {:error, :no_backup}
    end
  end
end
