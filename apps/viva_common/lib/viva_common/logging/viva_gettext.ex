defmodule Viva.Gettext do
  @moduledoc """
  Gettext backend for VIVA internationalization.

  Locale is configured via VIVA_LOCALE environment variable.
  Supported locales: en, pt_BR, zh_CN

  ## Configuration

      # config/config.exs
      config :viva, :locale, System.get_env("VIVA_LOCALE", "en")

  ## Usage

      Gettext.put_locale(Viva.Gettext, "pt_BR")
      Viva.Gettext.dgettext("default", "neuron_starting")
      # => "Neuronio emocional iniciando..."
  """

  use Gettext.Backend, otp_app: :viva_common

  @doc """
  Get current locale from application config.

  Falls back to "en" if not configured.
  """
  def current_locale do
    Application.get_env(:viva, :locale, "en")
  end

  @doc """
  Set locale for the current process.
  """
  def set_locale(locale) when locale in ["en", "pt_BR", "zh_CN"] do
    Gettext.put_locale(__MODULE__, locale)
  end

  def set_locale(_), do: set_locale("en")
end
