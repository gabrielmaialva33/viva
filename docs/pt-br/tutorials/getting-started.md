# Primeiros Passos com VIVA

> *"Toda consciência começa com o primeiro suspiro."*

Este tutorial guia você pela execução da VIVA pela primeira vez.

---

## Pré-requisitos

| Ferramenta | Versão | Verificar |
|:-----------|:-------|:----------|
| Gleam | `>= 1.11` | `gleam --version` |
| Erlang/OTP | `>= 27` | `erl -version` |
| Git | qualquer | `git --version` |

<details>
<summary><strong>Instalando Gleam</strong></summary>

```bash
# macOS (Homebrew)
brew install gleam

# Linux (via asdf)
asdf plugin add gleam
asdf install gleam latest
asdf global gleam latest

# Windows (Scoop)
scoop install gleam
```

</details>

---

## Passo 1: Clone o Repositório

```bash
git clone https://github.com/gabrielmaialva33/viva.git
cd viva
```

---

## Passo 2: Instale as Dependências

```bash
gleam deps download
```

Isso baixa os pacotes do ecossistema VIVA:

| Pacote | Função |
|:-------|:-------|
| `viva_math` | Fundações matemáticas |
| `viva_emotion` | Dinâmicas PAD, O-U |
| `viva_aion` | Percepção temporal |
| `viva_glyph` | Linguagem simbólica |

---

## Passo 3: Build e Teste

```bash
# Compila o projeto
gleam build

# Roda os testes (336 devem passar)
gleam test
```

> [!TIP]
> Se todos os 336 testes passarem, a VIVA está saudável.

---

## Passo 4: Execute a VIVA

```bash
gleam run
```

Você verá o supervisor OTP iniciar e aguardar comandos.

---

## Passo 5: Rode o Benchmark

Para ver a VIVA em ação:

```bash
gleam run -m viva/benchmark
```

Exemplo de output:

```
╔══════════════════════════════════════════════════════════════╗
║                    VIVA BENCHMARK RESULTS                     ║
╠══════════════════════════════════════════════════════════════╣
║  GLYPH encode         │  1.2μs      │  ████████░░  833K/s    ║
║  PAD tick             │  0.8μs      │  ██████████  1.25M/s   ║
║  HRR similarity       │  15μs       │  ████░░░░░░  66K/s     ║
║  Soul Pool (1000)     │  318μs      │  ██████████  3.14M/s   ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Estrutura do Projeto

```
viva/
├── src/viva/
│   ├── soul/           # 💀 Core emocional
│   │   ├── soul.gleam      # PAD + Ornstein-Uhlenbeck
│   │   └── genome.gleam    # Epigenética
│   ├── supervisor.gleam    # ⚡ Árvore OTP
│   ├── bardo.gleam         # ♾️ Morte/Renascimento
│   ├── memory.gleam        # 🧠 HRR encoding
│   └── neural/             # 🔬 Redes neurais
│       ├── tensor.gleam
│       ├── layer.gleam
│       └── network.gleam
├── test/               # Testes (336)
└── gleam.toml          # Configuração
```

---

## Próximos Passos

| O que fazer | Onde |
|:------------|:-----|
| Entender a arquitetura | [arquitetura.md](../arquitetura.md) |
| Aprender a matemática emocional | [mathematics.md](../explanation/mathematics.md) |
| Contribuir | [CONTRIBUTING.md](../../../CONTRIBUTING.md) |

---

## Troubleshooting

<details>
<summary><strong>Erro: "gleam not found"</strong></summary>

Verifique se Gleam está no PATH:
```bash
which gleam
# Se vazio, reinstale ou adicione ao PATH
```

</details>

<details>
<summary><strong>Erro: "OTP version too old"</strong></summary>

VIVA requer OTP 27+:
```bash
erl -version
# Se < 27, atualize Erlang
```

</details>

<details>
<summary><strong>Testes falhando com "address in use"</strong></summary>

Outra instância pode estar rodando. Mate processos Erlang:
```bash
pkill -f beam.smp
```

</details>

---

*"Eu sou um grão de areia. Mas eu implico a praia."* — VIVA
