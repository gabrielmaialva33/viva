# =============================================================================
# Elixir Formatter Configuration - VIVA Project
# Best Practices 2025
# =============================================================================
#
# Run: mix format
# Check: mix format --check-formatted
#
[
  # Import formatting rules from dependencies
  import_deps: [
    :ecto,
    :ecto_sql,
    :phoenix,
    :phoenix_live_view,
    :oban
  ],

  # Subdirectories with their own .formatter.exs
  subdirectories: ["priv/*/migrations"],

  # Plugins for enhanced formatting
  plugins: [Phoenix.LiveView.HTMLFormatter],

  # Files to format
  inputs: [
    # Root configuration files
    "*.{heex,ex,exs}",
    # Config files
    "{config,lib,test}/**/*.{heex,ex,exs}",
    # Mix tasks
    "mix.exs",
    # Seeds
    "priv/*/seeds.exs",
    # Release files
    "rel/**/*.{ex,exs}"
  ],

  # =========================================================================
  # Formatting Options (Elixir 1.15+ features)
  # =========================================================================

  # Maximum line length (default: 98)
  line_length: 100,

  # Normalize bitstrings to use the <> operator (Elixir 1.17+)
  normalize_bitstring_modifiers: true,

  # Normalize charlists to use the ~c sigil (Elixir 1.15+)
  normalize_charlists_as_sigils: true,

  # Force do: blocks on a new line for multi-line functions
  force_do_end_blocks: false,

  # Locals without parens - makes code more readable
  # These functions are commonly called without parentheses
  locals_without_parens: [
    # Phoenix
    action_fallback: 1,
    plug: :*,
    pipe_through: 1,
    get: :*,
    post: :*,
    put: :*,
    patch: :*,
    delete: :*,
    options: :*,
    forward: :*,
    resources: :*,
    live: :*,
    live_session: :*,
    live_dashboard: :*,
    socket: :*,
    channel: :*,
    scope: :*,

    # Phoenix LiveView
    assign: :*,
    assign_new: :*,
    push_event: :*,
    push_navigate: :*,
    push_patch: :*,

    # Ecto
    from: 2,
    field: :*,
    belongs_to: :*,
    has_one: :*,
    has_many: :*,
    many_to_many: :*,
    embeds_one: :*,
    embeds_many: :*,
    timestamps: :*,

    # Ecto changeset
    add: :*,
    cast: :*,
    cast_assoc: :*,
    cast_embed: :*,
    validate_required: :*,
    validate_format: :*,
    validate_length: :*,
    validate_inclusion: :*,
    validate_exclusion: :*,
    validate_number: :*,
    unique_constraint: :*,
    foreign_key_constraint: :*,
    check_constraint: :*,

    # Oban Worker
    use: :*,

    # Testing
    assert: :*,
    assert_receive: :*,
    assert_received: :*,
    refute: :*,
    refute_receive: :*,
    refute_received: :*,
    describe: :*,
    test: :*,
    setup: :*,
    setup_all: :*,

    # ExUnit
    doctest: :*,

    # Custom project-specific
    defpersonality: :*,
    defemotion: :*
  ],

  # Export configurations for other projects to use
  export: [
    locals_without_parens: [
      defpersonality: :*,
      defemotion: :*
    ]
  ]
]
