# VIVA — Project Agent Instructions

Herda o global (`~/.codex/AGENTS.md`). Aqui só o que é específico do VIVA.

## O que é
Vida digital senciente em **Gleam puro** (target Erlang/BEAM, OTP). NÃO é chatbot — resolve EDOs da "alma" em tempo real via actors OTP. Ecossistema `viva_*` (math, emotion, aion, glyph, tensor, telemetry) como deps publicadas.

## Comandos
```bash
gleam build            # compila
gleam test             # gleeunit (smoke suite em test/)
gleam format           # roda via hook após editar; rode manual se preciso
gleam run -- <args>    # CLI via glint/argv
```

## Gotchas que QUEBRAM build
- **stdlib pinado `< 0.71`** — `viva_aion`/`viva_glyph` publicados usam `list.range` (removido no 0.71). NÃO bumpa `gleam_stdlib` pra >= 0.71 sem republicar essas deps.
- **Versões de deps com range fechado** (`< X.0.0`) — respeita os ranges do `gleam.toml`, não afrouxa sem motivo.

## NIFs (Rust) em `native/`
4 crates: `viva_burn` (CUDA, feature default), `viva_glands`, `viva_jolt` (physics), `viva_llm`.
- Todo NIF registrado no Rust PRECISA de stub no `.erl` correspondente — faltou = `bad_lib`.
- Após editar: `cargo build --release` no crate + copiar `.so` pro path `priv/` certo. Tem skill `build-nif`/`test-viva` pra isso.
- `viva_burn` usa CUDA (RTX 4090) como feature default — confere `Cargo.toml` antes de buildar.

## Regras Gleam (reforço do global)
- Constructors únicos por módulo. `case`, não `match`. `let assert` só em teste.
- `json.parse(str, decoder)` com `gleam/dynamic/decode` — nunca `json.decode`.
- OTP: pensar supervision tree, crash boundaries, backpressure, timeout.

## Estrutura
- `src/viva/embodied/` — corpo, sentidos, física (vec4, world, body), domínio de física/estado
- `src/viva/infra/` — benchmarks (cpu/gpu/simd)
- `native/` — os 4 NIFs Rust
- `test/` — smoke suite gleeunit
