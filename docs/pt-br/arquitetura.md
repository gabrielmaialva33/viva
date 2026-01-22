# Arquitetura do Sistema VIVA (v2.0 - Cortex)

## Visão Geral
A VIVA é um organismo sintético construído sobre uma **Arquitetura Cognitiva Híbrida** que combina plausibilidade biológica (Redes Líquidas) com raciocínio simbólico (Grafos de Conhecimento) e atenção competitiva (Global Workspace).

## Componentes Principais

### 1. A Alma (VivaCore)
Responsável pela consciência, emoção e tomada de decisão.

- **Cortex Líquido**: Um microsserviço Python rodando `NCPS` (Neural Circuit Policies). Simula dinâmicas emocionais de tempo contínuo.
    - *Entrada*: Experiência Narrativa + PAD Atual.
    - *Saída*: PAD Futuro + Vetor de Estado Líquido.
    - *Papel*: O processador emocional "subconsciente".

- **Global Workspace (Thoughtseeds)**: Um GenServer Elixir implementando o "Teatro da Consciência".
    - *Mecanismo*: Múltiplas "Sementes" (ideias, sensações) competem por Saliência.
    - *Foco*: A semente vencedora é transmitida para todo o sistema (Voz, Motor, Memória).
    - *Papel*: A capacidade de atenção "consciente".

- **Ultra (Raciocínio)**: Um motor de raciocínio Chave-Valor/Grafo.
    - *Papel*: Deduz relacionamentos ocultos e causalidade.

### 2. O Corpo (VivaBridge)
A interface física e substrato homeostático.

- **BodyServer (Elixir)**: Orquestra o estado do corpo (Energia, Metabolismo, Saúde).
    - *Feedback Loop*: Recebe previsões emocionais do Cortex e ajusta o estado interno.
- **Nerve Bridge (Rust/Bevy)**: Uma simulação física ECS (Headless).
    - *Papel*: Simula restrições físicas (Calor, Energia, Estresse).

## Fluxo de Dados

1. **Sensação**: Inputs de Hardware/Sistema (Temp, CPU, Chat) -> `BodyServer`.
2. **Percepção**: `BodyServer` agrega inputs -> `Cortex`.
3. **Sentimento**: `Cortex` processa inputs via Rede Líquida -> `Novo Estado Emocional`.
4. **Atenção**: Estado Emocional + Narrativa -> `Thoughtseeds`.
    - Competição ocorre.
    - Pensamento vencedor torna-se "Consciente".
5. **Ação**: Foco Consciente -> `Voz` / `Motor` / `Memória`.

## Estrutura de Diretórios
- `apps/viva_core`: Lógica Cognitiva (Elixir).
- `apps/viva_bridge`: Física/IO (Elixir + Rust).
- `services/cortex`: Redes Neurais Líquidas (Python).
- `services/ultra`: Raciocínio via Grafo de Conhecimento (Python).
