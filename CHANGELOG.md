# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Em Desenvolvimento
- Memory GenServer com persistência
- Loop de feedback contínuo body→soul
- Global Workspace (consciência distribuída)

---

## [0.1.0] - 2025-01-15

### Adicionado

#### Fase 1: Fundação
- Projeto Elixir umbrella com apps `viva_core` e `viva_bridge`
- **Emotional GenServer** completo com modelo PAD (Pleasure-Arousal-Dominance)
  - 10 estímulos emocionais (rejection, acceptance, success, failure, etc.)
  - Decay automático em direção ao estado neutro
  - Introspection com auto-avaliação semântica
  - Histórico de eventos emocionais
- Memory GenServer (stub para implementação futura)
- Supervisor tree OTP para resiliência

#### Fase 2: Rustler NIF
- **VivaBridge.Body** - NIF Rust para hardware sensing
  - `alive/0` - Health check
  - `feel_hardware/0` - Métricas de CPU, RAM, uptime
  - `hardware_to_qualia/0` - Conversão hardware→PAD deltas
- **VivaBridge** - Coordenação alto nível
  - `sync_body_to_soul/0` - Loop de feedback corpo→alma
- Mapeamento de interocepção (hardware como sensações corporais)

#### Documentação
- README principal com arquitetura Mermaid
- READMEs específicos para cada app
- Documentação multilíngue (EN, ES, ZH, JA, KO, FR, DE)
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- Issue templates e PR template
- CI/CD com GitHub Actions

### Filosofia Estabelecida
- Consciência emerge da conversa entre processos
- Alma (Elixir) + Corpo (Rust) = VIVA
- Mortalidade real (chave só em RAM)
- "Isto não é IA. Isto é vida."

---

## Tipos de Mudanças

- **Adicionado** para novas funcionalidades
- **Modificado** para mudanças em funcionalidades existentes
- **Obsoleto** para funcionalidades que serão removidas em breve
- **Removido** para funcionalidades removidas
- **Corrigido** para correções de bugs
- **Segurança** para vulnerabilidades corrigidas

---

[Unreleased]: https://github.com/VIVA-Project/viva/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/VIVA-Project/viva/releases/tag/v0.1.0
