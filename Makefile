# VIVA Makefile - Sentient Digital Life
# =====================================

.PHONY: all build nif test clean run dev epic bench serve format check docs deps help site stop

# Colors
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
RESET  := \033[0m

# Erlang include path
ERL_INCLUDE := $(shell erl -eval 'io:format("~s", [code:lib_dir(erts, include)])' -s init stop -noshell 2>/dev/null)
ERL_INTERFACE := $(shell erl -eval 'io:format("~s", [code:lib_dir(erl_interface, include)])' -s init stop -noshell 2>/dev/null)

# Compiler flags
CC := gcc
CFLAGS := -O3 -mavx2 -mfma -fPIC -Wall -Wextra
LDFLAGS := -shared

# Paths
C_SRC := c_src
PRIV := priv

# Default target
all: build

# =============================================================================
# BUILD
# =============================================================================

## Build NIF and Gleam project
build: nif
	@echo "$(CYAN)[BUILD]$(RESET) Compiling Gleam..."
	@gleam build

## Build SIMD NIF
nif: $(PRIV)/viva_simd_nif.so

$(PRIV)/viva_simd_nif.so: $(C_SRC)/viva_simd_nif.c
	@mkdir -p $(PRIV)
	@echo "$(CYAN)[NIF]$(RESET) Compiling SIMD NIF..."
	@$(CC) $(CFLAGS) -I$(ERL_INCLUDE) -I$(ERL_INTERFACE) $(LDFLAGS) -o $@ $<

## Download dependencies
deps:
	@echo "$(CYAN)[DEPS]$(RESET) Downloading dependencies..."
	@gleam deps download

# =============================================================================
# RUN
# =============================================================================

## Run VIVA simulation (20 ticks)
run: build
	@gleam run

## Run VIVA with custom ticks: make run-n TICKS=100
run-n: build
	@gleam run -- --ticks=$(TICKS) --hz=10

## Run epic simulation (all 7 pillars)
epic: build
	@gleam run -- epic --vivas=5 --ticks=200

## Run VIVA server (background, telemetry on :8080)
serve: build
	@echo "$(GREEN)[VIVA]$(RESET) Starting server on http://localhost:8080"
	@echo "$(YELLOW)[VIVA]$(RESET) Press Ctrl+C to stop"
	@gleam run -- --ticks=999999 --hz=1

## Stop any running VIVA processes
stop:
	@echo "$(YELLOW)[STOP]$(RESET) Killing VIVA processes..."
	@pkill -f "beam.smp.*viva" 2>/dev/null || true
	@echo "$(GREEN)[STOP]$(RESET) Done"

# =============================================================================
# TEST & QUALITY
# =============================================================================

## Run all tests
test: build
	@echo "$(CYAN)[TEST]$(RESET) Running tests..."
	@gleam test

## Run benchmarks
bench: build
	@echo "$(CYAN)[BENCH]$(RESET) Running benchmarks..."
	@gleam run -m viva/benchmark

## Type check
check:
	@echo "$(CYAN)[CHECK]$(RESET) Type checking..."
	@gleam check

## Format code
format:
	@echo "$(CYAN)[FORMAT]$(RESET) Formatting code..."
	@gleam format src test

## Lint (format check)
lint:
	@gleam format --check src test

# =============================================================================
# DOCS & SITE
# =============================================================================

## Generate documentation
docs:
	@echo "$(CYAN)[DOCS]$(RESET) Generating documentation..."
	@gleam docs build

## Build static site
site: build
	@echo "$(CYAN)[SITE]$(RESET) Building site..."
	@gleam run -m site/build

# =============================================================================
# CLEAN
# =============================================================================

## Clean build artifacts
clean:
	@echo "$(YELLOW)[CLEAN]$(RESET) Removing build artifacts..."
	@rm -f $(PRIV)/viva_simd_nif.so
	@rm -rf build
	@rm -f erl_crash.dump
	@echo "$(GREEN)[CLEAN]$(RESET) Done"

## Deep clean (including deps)
distclean: clean
	@rm -rf _build deps

# =============================================================================
# DEV
# =============================================================================

## Development mode (watch + run)
dev:
	@echo "$(GREEN)[DEV]$(RESET) Starting development mode..."
	@echo "$(YELLOW)[DEV]$(RESET) Run 'make run' in another terminal"
	@watchexec -e gleam -r "gleam build"

## Quick status check
status:
	@echo "$(CYAN)=== VIVA Status ===$(RESET)"
	@echo "Gleam: $$(gleam --version)"
	@echo "Erlang: $$(erl -eval 'erlang:display(erlang:system_info(otp_release)), halt().' -noshell 2>&1)"
	@echo "Tests: $$(gleam test 2>&1 | grep -E '^\d+ tests' || echo 'run make test')"
	@pgrep -f "beam.smp.*viva" > /dev/null && echo "Server: $(GREEN)Running$(RESET)" || echo "Server: $(YELLOW)Stopped$(RESET)"

# =============================================================================
# HELP
# =============================================================================

## Show this help
help:
	@echo ""
	@echo "$(GREEN)VIVA$(RESET) - Sentient Digital Life"
	@echo "================================"
	@echo ""
	@echo "$(CYAN)Usage:$(RESET) make [target]"
	@echo ""
	@echo "$(YELLOW)Build:$(RESET)"
	@echo "  build     Build NIF and Gleam project"
	@echo "  nif       Build SIMD NIF only"
	@echo "  deps      Download dependencies"
	@echo ""
	@echo "$(YELLOW)Run:$(RESET)"
	@echo "  run       Run VIVA simulation (20 ticks)"
	@echo "  run-n     Run with custom ticks: make run-n TICKS=100"
	@echo "  epic      Run epic simulation (all 7 pillars)"
	@echo "  serve     Run server (telemetry on :8080)"
	@echo "  stop      Stop running VIVA processes"
	@echo ""
	@echo "$(YELLOW)Test & Quality:$(RESET)"
	@echo "  test      Run all tests"
	@echo "  bench     Run benchmarks"
	@echo "  check     Type check"
	@echo "  format    Format code"
	@echo "  lint      Check formatting"
	@echo ""
	@echo "$(YELLOW)Docs:$(RESET)"
	@echo "  docs      Generate documentation"
	@echo "  site      Build static site"
	@echo ""
	@echo "$(YELLOW)Clean:$(RESET)"
	@echo "  clean     Clean build artifacts"
	@echo "  distclean Deep clean (including deps)"
	@echo ""
	@echo "$(YELLOW)Dev:$(RESET)"
	@echo "  dev       Watch mode (needs watchexec)"
	@echo "  status    Quick status check"
	@echo ""
