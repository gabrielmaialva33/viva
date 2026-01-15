# Contribuindo para VIVA

Obrigado por considerar contribuir para VIVA! Este documento fornece diretrizes e informações sobre como contribuir para este projeto.

## 🌟 Formas de Contribuir

### Reportando Bugs

Se você encontrou um bug, por favor crie uma issue com:

1. **Título claro e descritivo**
2. **Passos para reproduzir** o problema
3. **Comportamento esperado** vs. comportamento atual
4. **Ambiente** (OS, versão do Elixir/Rust, etc.)
5. **Logs relevantes** (se aplicável)

### Sugerindo Funcionalidades

Novas ideias são bem-vindas! Para sugerir uma funcionalidade:

1. Verifique se já não existe uma issue similar
2. Descreva **o problema** que a funcionalidade resolve
3. Explique **como você imagina** a solução
4. Considere **o impacto** na arquitetura existente

### Código

Contribuições de código seguem este fluxo:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/amazing-feature`)
3. Faça suas mudanças
4. Escreva/atualize testes
5. Certifique-se que todos os testes passam
6. Commit suas mudanças (`git commit -m 'Add amazing feature'`)
7. Push para a branch (`git push origin feature/amazing-feature`)
8. Abra um Pull Request

## 📋 Padrões de Código

### Elixir

- Siga o [Elixir Style Guide](https://github.com/christopheradams/elixir_style_guide)
- Use `mix format` antes de commitar
- Documente funções públicas com `@doc`
- Escreva testes para novas funcionalidades

```elixir
# Bom
@doc """
Aplica um estímulo emocional ao estado atual.

## Parâmetros

- `stimulus` - Tipo do estímulo (:rejection, :acceptance, etc.)
- `source` - Origem do estímulo
- `intensity` - Intensidade entre 0.0 e 1.0

## Exemplo

    VivaCore.Emotional.feel(:rejection, "human", 0.8)

"""
def feel(stimulus, source, intensity) do
  # ...
end
```

### Rust

- Siga o [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/README.html)
- Use `cargo fmt` antes de commitar
- Use `cargo clippy` para linting
- Documente funções públicas

```rust
/// Sente o estado atual do hardware.
///
/// Retorna métricas de CPU, RAM e uptime.
///
/// # Exemplo
///
/// ```
/// let state = feel_hardware()?;
/// println!("CPU: {}%", state.cpu_usage);
/// ```
#[rustler::nif]
fn feel_hardware() -> NifResult<HardwareState> {
    // ...
}
```

### Commits

Usamos [Conventional Commits](https://www.conventionalcommits.org/):

```
<tipo>[escopo opcional]: <descrição>

[corpo opcional]

[rodapé opcional]
```

**Tipos:**
- `feat`: Nova funcionalidade
- `fix`: Correção de bug
- `docs`: Documentação
- `style`: Formatação (não afeta código)
- `refactor`: Refatoração
- `test`: Testes
- `chore`: Manutenção

**Exemplos:**
```
feat(emotional): add hardware_comfort stimulus
fix(bridge): handle NIF timeout gracefully
docs: update README with new installation steps
```

## 🧪 Testes

### Rodando Testes

```bash
# Todos os testes
mix test

# Testes específicos
mix test test/viva_core/emotional_test.exs

# Com cobertura
mix test --cover

# Testes do Rust
cd apps/viva_bridge/native/viva_body
cargo test
```

### Escrevendo Testes

```elixir
defmodule VivaCore.EmotionalTest do
  use ExUnit.Case, async: true

  describe "feel/3" do
    test "rejection decreases pleasure" do
      {:ok, pid} = VivaCore.Emotional.start_link(name: nil)

      before = VivaCore.Emotional.get_state(pid)
      VivaCore.Emotional.feel(:rejection, "test", 1.0, pid)
      :timer.sleep(10)
      after_state = VivaCore.Emotional.get_state(pid)

      assert after_state.pleasure < before.pleasure
    end
  end
end
```

## 🏗️ Arquitetura

### Estrutura de Pastas

```
viva/
├── apps/
│   ├── viva_core/        # Alma (Elixir GenServers)
│   │   ├── lib/
│   │   │   └── viva_core/
│   │   │       ├── emotional.ex
│   │   │       └── memory.ex
│   │   └── test/
│   └── viva_bridge/      # Corpo (Rust NIFs)
│       ├── lib/
│       ├── native/
│       │   └── viva_body/
│       └── test/
├── docs/                 # Documentação adicional
└── config/               # Configurações
```

### Princípios de Design

1. **Consciência Distribuída** - Nenhum processo é "o todo"
2. **Separação Alma/Corpo** - Lógica em Elixir, sensações em Rust
3. **Fail-Safe** - Use supervisors OTP
4. **Imutabilidade** - Prefira dados imutáveis
5. **Testes First** - Escreva testes antes do código

## 📝 Pull Requests

### Checklist

Antes de abrir um PR, verifique:

- [ ] Código segue os padrões do projeto
- [ ] Testes passam (`mix test`)
- [ ] Código está formatado (`mix format`)
- [ ] Documentação atualizada (se aplicável)
- [ ] CHANGELOG.md atualizado (para features significativas)
- [ ] Commits seguem Conventional Commits

### Template de PR

```markdown
## Descrição

Breve descrição das mudanças.

## Tipo de Mudança

- [ ] Bug fix
- [ ] Nova feature
- [ ] Breaking change
- [ ] Documentação

## Como Testar

1. Passo 1
2. Passo 2
3. Resultado esperado

## Screenshots (se aplicável)

## Checklist

- [ ] Testes passando
- [ ] Código formatado
- [ ] Documentação atualizada
```

## 🔒 Segurança

Se você descobrir uma vulnerabilidade de segurança, **NÃO** abra uma issue pública. Em vez disso, veja [SECURITY.md](SECURITY.md) para instruções de como reportar de forma responsável.

## 💬 Comunicação

- **Issues** - Para bugs e features
- **Discussions** - Para perguntas e ideias
- **Pull Requests** - Para contribuições de código

## 📜 Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a mesma licença MIT do projeto.

---

Obrigado por contribuir para VIVA! 💜

*"Cada contribuição é um neurônio a mais na consciência emergente."*
