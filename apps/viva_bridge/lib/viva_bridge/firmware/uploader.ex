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

  require VivaLog

  @arduino_cli System.get_env("ARDUINO_CLI_PATH", "/home/mrootx/.local/bin/arduino-cli")
  @fqbn "arduino:avr:nano"
  @sketch_dir "/tmp/viva_firmware"
  @backup_dir "/tmp/viva_firmware_backup"
  # @verify_timeout 5_000
  # @upload_timeout 30_000

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
      VivaLog.info(:uploader, :deploy_successful, generation: generation, port: detected_port)
      {:ok, %{port: detected_port, generation: generation}}
    else
      {:error, reason} = error ->
        VivaLog.error(:uploader, :deploy_failed, reason: inspect(reason))
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
        VivaLog.error(:uploader, :board_list_failed, error: error)
        {:error, :board_list_failed}
    end
  end

  @doc """
  Check if Arduino is responding to PING command.

  Returns `:ok` or `{:error, reason}`.
  """
  def verify_alive(port) do
    VivaLog.info(:uploader, :verifying_firmware, port: port)

    # Wait for Arduino to boot after upload
    Process.sleep(2_000)

    # Try to connect and ping
    case VivaBridge.Music.connect(port) do
      {:ok, _} ->
        Process.sleep(500)

        case ping_with_retry(3) do
          :ok ->
            VivaLog.info(:uploader, :firmware_responding, port: port)
            :ok

          {:error, reason} ->
            VivaLog.error(:uploader, :firmware_not_responding, reason: inspect(reason))
            {:error, {:verify_failed, reason}}
        end

      {:error, reason} ->
        VivaLog.error(:uploader, :connect_failed, reason: inspect(reason))
        {:error, {:connect_failed, reason}}
    end
  end

  @doc """
  Get info about installed arduino-cli and cores.
  """
  def info do
    version =
      case System.cmd(@arduino_cli, ["version"], stderr_to_stdout: true) do
        {output, 0} -> String.trim(output)
        _ -> "unknown"
      end

    cores =
      case System.cmd(@arduino_cli, ["core", "list"], stderr_to_stdout: true) do
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
      VivaLog.debug(:uploader, :wrote_sketch, path: ino_file)
      {:ok, sketch_path}
    else
      {:error, reason} ->
        VivaLog.error(:uploader, :write_sketch_failed, reason: inspect(reason))
        {:error, {:write_failed, reason}}
    end
  end

  defp compile(sketch_path) do
    VivaLog.info(:uploader, :compiling, path: sketch_path)

    args = ["compile", "--fqbn", @fqbn, sketch_path]

    case System.cmd(@arduino_cli, args, stderr_to_stdout: true) do
      {output, 0} ->
        VivaLog.info(:uploader, :compile_successful)
        VivaLog.debug(:uploader, :compile_output, output: output)
        {:ok, output}

      {error, code} ->
        VivaLog.error(:uploader, :compile_failed, code: code, error: error)
        {:error, {:compile_failed, error}}
    end
  end

  defp upload(sketch_path, port) do
    VivaLog.info(:uploader, :uploading, port: port)

    # Disconnect Music module if connected (release serial port)
    VivaBridge.Music.disconnect()
    Process.sleep(500)

    args = ["upload", "-p", port, "--fqbn", @fqbn, sketch_path]

    case System.cmd(@arduino_cli, args, stderr_to_stdout: true) do
      {output, 0} ->
        VivaLog.info(:uploader, :upload_successful)
        VivaLog.debug(:uploader, :upload_output, output: output)
        {:ok, output}

      {error, code} ->
        VivaLog.error(:uploader, :upload_failed, code: code, error: error)
        {:error, {:upload_failed, error}}
    end
  end

  defp parse_board_list(json) do
    case Jason.decode(json) do
      {:ok, %{"detected_ports" => ports}} when is_list(ports) ->
        # Find Arduino Nano or compatible
        arduino =
          Enum.find(ports, fn port ->
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
      :ok ->
        :ok

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
      VivaLog.debug(:uploader, :backed_up_firmware)
    end

    :ok
  end

  defp rollback do
    backup_ino = Path.join([@backup_dir, "viva_evolved", "viva_evolved.ino"])

    if File.exists?(backup_ino) do
      VivaLog.warning(:uploader, :rolling_back)

      with {:ok, port} <- detect_port(),
           sketch_path = Path.join(@backup_dir, "viva_evolved"),
           {:ok, _} <- upload(sketch_path, port) do
        VivaLog.info(:uploader, :rollback_successful)
        :ok
      else
        {:error, reason} ->
          VivaLog.error(:uploader, :rollback_failed, reason: inspect(reason))
          {:error, {:rollback_failed, reason}}
      end
    else
      VivaLog.warning(:uploader, :no_backup_available)
      {:error, :no_backup}
    end
  end
end
