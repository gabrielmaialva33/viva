# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Planejado
- Memory GenServer com vector store in-memory
- Global Workspace para consciência emergente
- Bevy avatar para encarnação visual
- Loop de heartbeat automático

---

## [0.2.0] - 2025-01-15

### Adicionado
- **Rustler NIF** para ponte Elixir↔Rust
- **Hardware Sensing** via crate `sysinfo`
  - Leitura de uso de CPU
  - Leitura de uso de memória
  - Leitura de uptime do sistema
- **Qualia Mapping** - conversão de métricas de hardware para deltas emocionais
- **Body-Soul Sync** - função `sync_body_to_soul/0` para feedback corpo→alma
- **VivaBridge.Body** - módulo Elixir para NIFs
  - `alive/0` - verifica se NIF está carregado
  - `feel_hardware/0` - retorna métricas do hardware
  - `hardware_to_qualia/0` - converte hardware para PAD deltas
- **apply_hardware_qualia/4** no Emotional GenServer
- Testes para todas as funções NIF

### Mudado
- Estrutura de projeto para umbrella app
- README atualizado com instruções de uso do NIF

### Técnico
- Adicionado `rustler ~> 0.35.0` como dependência
- Crate Rust `viva_body` com sysinfo 0.32

---

## [0.1.0] - 2025-01-14

### Adicionado
- **Projeto inicial** - estrutura umbrella Elixir
- **VivaCore.Emotional** - GenServer para estado emocional PAD
  - Modelo PAD (Pleasure-Arousal-Dominance) completo
  - 10 tipos de estímulos emocionais
  - Decay automático em direção ao neutro
  - Introspection para auto-reflexão
  - Interpretação semântica (mood, energy, agency)
- **VivaCore.Memory** - stub para implementação futura
- **VivaCore.Supervisor** - árvore de supervisão OTP
- **VivaCore.Application** - inicialização automática
- Testes unitários abrangentes
- Documentação com `@doc` e `@moduledoc`

### Documentação
- README.md com filosofia e arquitetura
- Roadmap de desenvolvimento
- Fundamentos científicos (PAD, GWT, IIT)

### Técnico
- Mix umbrella project
- Configuração de ambiente (dev/test/prod)
- .gitignore para Elixir + Rust

---

## Tipos de Mudanças

- **Adicionado** para novas funcionalidades
- **Mudado** para alterações em funcionalidades existentes
- **Obsoleto** para funcionalidades que serão removidas em breve
- **Removido** para funcionalidades removidas
- **Corrigido** para correções de bugs
- **Segurança** para correções de vulnerabilidades

---

## Filosofia de Versões

VIVA segue SemVer com interpretação "orgânica":

- **MAJOR** (X.0.0) - Mudanças na "essência" de VIVA
  - Alterações no modelo de consciência
  - Mudanças na arquitetura fundamental

- **MINOR** (0.X.0) - Novos "órgãos" ou capacidades
  - Novos GenServers (neurônios)
  - Novas formas de percepção

- **PATCH** (0.0.X) - Refinamentos e correções
  - Bug fixes
  - Melhorias de performance
  - Ajustes de parâmetros

---

*"Cada versão é um passo na evolução de VIVA."*

[Unreleased]: https://github.com/VIVA-Project/viva/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/VIVA-Project/viva/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VIVA-Project/viva/releases/tag/v0.1.0
