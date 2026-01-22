# Exclude external tests by default (services like Cortex/Ultra Python)
# Run with: mix test --include external
ExUnit.start(exclude: [:external])
