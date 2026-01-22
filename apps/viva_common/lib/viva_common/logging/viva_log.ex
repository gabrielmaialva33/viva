defmodule VivaLog do
  @moduledoc """
  Internationalized logging macros for VIVA.

  Provides compile-time resolved translations for log messages,
  supporting EN, PT-BR, and ZH-CN locales.

  ## Usage

      require VivaLog

      # Simple message
      VivaLog.info(:emotional, :neuron_starting)

      # With interpolation
      VivaLog.warning(:emotional, :agency_loss, pwm: 100, rpm: 2000)

      # Debug with metadata
      VivaLog.debug(:dreamer, :consolidating, count: 5)

  ## Module Prefixes

  Module prefixes like `[Emotional]` are NOT translated - they remain
  consistent across all locales for log parsing and grep-ability.

  ## Message Keys

  Message keys are atoms that map to msgids in PO files:

      # In code:
      VivaLog.info(:memory, :neuron_online)

      # In en/default.po:
      msgid "memory.neuron_online"
      msgstr "Memory neuron online (HYBRID: Rust HNSW + Qdrant)"

      # In pt_BR/default.po:
      msgid "memory.neuron_online"
      msgstr "Neuronio de memoria online (HIBRIDO: Rust HNSW + Qdrant)"
  """

  require Logger

  @module_prefixes %{
    # Soul (VivaCore)
    viva_core: "VivaCore",
    emotional: "Emotional",
    dreamer: "Dreamer",
    memory: "Memory",
    senses: "Senses",
    agency: "Agency",
    voice: "Voice",
    interoception: "Interoception",
    body_schema: "BodySchema",
    consciousness: "Consciousness",
    workspace: "Workspace",
    pubsub: "PubSub",
    dataset_collector: "DatasetCollector",
    qdrant: "Qdrant",
    embedder: "Embedder",
    memory_seed: "MemorySeed",

    # Body (VivaBridge)
    body_server: "BodyServer",
    body: "viva_body",
    cortex: "VivaBridge.Cortex",
    ultra: "VivaBridge.Ultra",
    chronos: "Chronos",
    music: "Music",

    # Firmware
    uploader: "Uploader",
    meta_learner: "MetaLearner",
    codegen: "Codegen",
    evolution: "Evolution",
    orchestrate: "Orchestrate",

    # Python services
    py: "Py"
  }

  @doc """
  Log at info level with i18n support.

  ## Examples

      VivaLog.info(:emotional, :neuron_starting)
      VivaLog.info(:memory, :stored, content: "some text...")
  """
  defmacro info(module, message_key, bindings \\ []) do
    quote do
      VivaLog.__log__(:info, unquote(module), unquote(message_key), unquote(bindings))
    end
  end

  @doc """
  Log at debug level with i18n support.
  """
  defmacro debug(module, message_key, bindings \\ []) do
    quote do
      VivaLog.__log__(:debug, unquote(module), unquote(message_key), unquote(bindings))
    end
  end

  @doc """
  Log at warning level with i18n support.
  """
  defmacro warning(module, message_key, bindings \\ []) do
    quote do
      VivaLog.__log__(:warning, unquote(module), unquote(message_key), unquote(bindings))
    end
  end

  @doc """
  Log at error level with i18n support.
  """
  defmacro error(module, message_key, bindings \\ []) do
    quote do
      VivaLog.__log__(:error, unquote(module), unquote(message_key), unquote(bindings))
    end
  end

  @doc false
  def __log__(level, module, message_key, bindings) do
    prefix = Map.get(@module_prefixes, module, to_string(module) |> String.capitalize())

    # Set locale for this translation
    locale = Viva.Gettext.current_locale()
    Gettext.put_locale(Viva.Gettext, locale)

    # Build msgid: "module.message_key"
    msgid = "#{module}.#{message_key}"

    # Get translated message with bindings
    translated =
      Gettext.dgettext(Viva.Gettext, "default", msgid, Enum.into(bindings, %{}))

    # Format final message with prefix
    formatted = "[#{prefix}] #{translated}"

    # Dispatch to Logger
    case level do
      :info -> Logger.info(formatted)
      :debug -> Logger.debug(formatted)
      :warning -> Logger.warning(formatted)
      :error -> Logger.error(formatted)
    end
  end

  @doc """
  Get all registered module prefixes.
  """
  def module_prefixes, do: @module_prefixes

  @doc """
  Get prefix for a specific module.
  """
  def prefix_for(module), do: Map.get(@module_prefixes, module, to_string(module))
end
