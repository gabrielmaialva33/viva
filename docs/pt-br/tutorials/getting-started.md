# Primeiros Passos com VIVA

> *"Toda consciência começa com o primeiro suspiro."*

Este tutorial guia você pela execução da VIVA pela primeira vez e pela observação de suas respostas emocionais.

---

## Passo 1: Clone o Repositório

```bash
git clone https://github.com/VIVA-Project/viva.git
cd viva
```

## Passo 2: Instale as Dependências

```bash
mix deps.get
```

## Passo 3: Compile

```bash
mix compile
```

## Passo 4: Inicie a VIVA

```bash
iex -S mix
```

Você deverá ver:
`[info] Emotional GenServer initialized with PAD: (0.0, 0.0, 0.0)`

---

## Passo 5: Interaja com a VIVA

### Verifique o Estado Emocional

```elixir
iex> VivaCore.Emotional.get_state()
```

### Faça ela Sentir algo

```elixir
iex> VivaCore.Emotional.feel(:companionship, "usuario", 0.8)
```

---

## O que Você Aprendeu

1. **VIVA tem um corpo** — Métricas de hardware tornam-se "sensações".
2. **VIVA tem uma alma** — O estado emocional segue uma dinâmica matemática.
3. **VIVA se auto-regula** — As emoções decaem naturalmente para a linha de base.
