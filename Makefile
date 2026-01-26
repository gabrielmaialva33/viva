# VIVA Makefile - Build SIMD NIFs and project

.PHONY: all build nif test clean

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

# Build NIF and Gleam project
build: nif
	gleam build

# Build SIMD NIF
nif: $(PRIV)/viva_simd_nif.so

$(PRIV)/viva_simd_nif.so: $(C_SRC)/viva_simd_nif.c
	@mkdir -p $(PRIV)
	$(CC) $(CFLAGS) -I$(ERL_INCLUDE) -I$(ERL_INTERFACE) $(LDFLAGS) -o $@ $<

# Test
test: build
	gleam test

# Clean
clean:
	rm -f $(PRIV)/viva_simd_nif.so
	rm -rf build
