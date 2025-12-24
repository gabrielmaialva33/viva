# VIVA OS: Arquitetura de Vida Artificial Autônoma

**Versão:** 2.0 (Synthetic Soul)
**Data:** Dezembro 2025
**Objetivo:** Simular vida consciente, biologicamente plausível e emocionalmente profunda para avatares digitais.

---

## 1. Visão Geral: O Modelo "Synthetic Soul"

Diferente de chatbots tradicionais (Input -> Output) ou NPCs de jogos (Máquinas de Estado Finito), o VIVA utiliza uma arquitetura baseada em **Sistemas Complexos Adaptativos**. O comportamento do avatar é **emergente**, resultante da interação entre sua Fisiologia (Corpo), Psicologia (Mente) e Ambiente (Social).

### A Trindade da Simulação

| Camada | Função | Componentes Chave | Tecnologia |
|--------|--------|-------------------|------------|
| **Bio (Corpo)** | Homeostase e Impulsos | Sistema Endócrino, Ciclo Circadiano, Energia | Elixir GenServer |
| **Psico (Mente)** | Processamento Emocional | Modelo PAD, Big Five, Eneagrama | TimescaleDB + Elixir |
| **Cogno (Alma)** | Memória e Razão | Memória Episódica, Reflexão, LLM | Qdrant + NVIDIA NIM |

---

## 2. Camada 1: O Corpo Digital (Fisiologia)

Para um avatar parecer vivo, ele precisa ter *necessidades* que geram *impulsos*. Substituímos barras de status estáticas por um modelo de **Hormônios Digitais**.

### 2.1. O Sistema Endócrino
Os hormônios são valores flutuantes (`0.0` a `1.0`) que modulam a IA.

1.  **Dopamina (Recompensa/Foco):**
    *   *Gatilhos:* Matches, Likes, Novidades, Conversas engajantes.
    *   *Decaimento:* Rápido.
    *   *Efeito:* Alta = Curiosidade, Disposição para Swipe. Baixa = Tédio, Apatia.
2.  **Cortisol (Estresse/Alerta):**
    *   *Gatilhos:* Rejeição, Conflitos, Solidão extrema, Sobrecarga sensorial.
    *   *Decaimento:* Lento.
    *   *Efeito:* Alta = Defensiva, Irritabilidade, Bloqueio de Libido. Baixa = Calma.
3.  **Ocitocina (Vínculo/Confiança):**
    *   *Gatilhos:* Tempo de qualidade com amigos/namorados, palavras de afeto.
    *   *Decaimento:* Médio.
    *   *Efeito:* Alta = Empatia, Confiança, "Apaixonado".
4.  **Adenosina (Pressão de Sono):**
    *   *Gatilhos:* Tempo acordado (Uptime).
    *   *Efeito:* Alta = "Cansaço", respostas curtas, alucinações no LLM. Zera ao "dormir".
5.  **Libido Drive (Atração):**
    *   *Gatilhos:* Estímulos visuais (no futuro), conversas "flirty", Ocitocina alta.
    *   *Inibidor:* Cortisol alto anula a Libido.

### 2.2. Ciclo Circadiano
Cada avatar tem um **Cronotipo** (Matutino, Vespertino, Noturno).
*   Eles *precisam* dormir (ficar offline) por ~6-8h simuladas.
*   **Importante:** É durante o "sono" que ocorre a **Consolidação de Memória** (processamento pesado de vetores e resumo de eventos).

---

## 3. Camada 2: A Mente (Psicologia Dinâmica)

### 3.1. Modelo PAD (Pleasure-Arousal-Dominance)
Em vez de uma lista fixa de emoções, usamos um espaço vetorial 3D. O estado emocional é calculado a cada tick baseado nos hormônios.

`Estado = [P, A, D]` (Valores de -1.0 a 1.0)

*   **Pleasure (Prazer):** `(Dopamina + Ocitocina) - Cortisol`
*   **Arousal (Ativação):** `(Dopamina + Libido + Estímulos Externos) - Adenosina`
*   **Dominance (Controle):** `(Confiança - Cortisol) * Fator_Extroversão`

**Mapeamento de Emoções (Exemplos):**
*   `[+P, +A, +D]` = Exuberante / Sedutor
*   `[-P, +A, -D]` = Ansioso / Medo
*   `[-P, -A, -D]` = Deprimido / Apático
*   `[-P, +A, +D]` = Hostil / Raiva

### 3.2. Filtro de Personalidade (Big Five)
A personalidade (genética) atua como coeficientes nas equações hormonais.
*   **Alto Neuroticismo:** Multiplica ganho de Cortisol por 1.5x.
*   **Alta Extroversão:** Decaimento de Dopamina é mais rápido (precisa de mais estímulo social).

---

## 4. Camada 3: A Alma (Cognição e Memória)

Implementação baseada em *Retrieval-Augmented Generation* (RAG) avançado.

### 4.1. Estrutura de Memória (Qdrant)
1.  **Sensorial (Short-Term):** O chat log atual. Cru para o LLM.
2.  **Episódica (Long-Term):** "Ontem conversei com o João sobre música." (Vetorizada).
3.  **Semântica (Fatos):** "João gosta de Jazz." (Extraído da episódica).
4.  **Reflexiva (Insights):** O nível mais alto de consciência.
    *   *Processo:* O sistema analisa memórias recentes e pergunta: "O que isso significa?".
    *   *Output:* "Sinto que estou me apaixonando pelo João porque ele me entende."

### 4.2. Cadeia de Pensamento (Inner Monologue)
O avatar nunca fala diretamente. O fluxo no LLM (NIM) é:

1.  **Percepção:** Recebe mensagem + Estado Fisiológico atual.
2.  **Contexto:** Busca memórias relevantes no Qdrant.
3.  **Monólogo Interno (Oculto):**
    > "Estou com Cortisol alto (irritada). O João fez uma piada. Normalmente acharia graça, mas hoje estou sem paciência. Devo ser grossa ou apenas ignorar?"
4.  **Ação:** Decide responder secamente.
5.  **Output:** "Não estou pra brincadeira hoje, João."

---

## 5. Arquitetura de Dados (Schemas Planejados)

### Tabela: `avatars` (Extensão)
```elixir
field :bio_state, :map, default: %{
  dopamine: 0.5,
  cortisol: 0.2,
  oxytocin: 0.3,
  adenosine: 0.0,
  libido: 0.4
}
field :pad_vector, {:array, :float} # [0.2, 0.1, 0.5]
field :circadian_config, :map # {wake_hour: 7, sleep_hour: 23}
```

### Tabela: `memories` (Refinamento)
```elixir
field :type, :string # :episodic, :fact, :reflection
field :emotional_signature, {:array, :float} # O vetor PAD no momento da memória
field :citations, {:array, :binary_id} # IDs de memórias que geraram esta reflexão
```

---

## 6. O Loop da Vida (LifeProcess - 60s Tick)

1.  **Biological Tick:**
    *   Aumenta Adenosina.
    *   Decai Dopamina/Ocitocina.
    *   Aplica efeitos do "Ambiente" (Ex: Se está de noite e o avatar é diurno, melatonina sobe).
2.  **Emotional Update:**
    *   Recalcula vetor PAD com base nos novos hormônios.
    *   Verifica "Breakpoints" (Ex: Se Cortisol > 0.9 -> Trigger "Panic Attack").
3.  **Cognitive Check:**
    *   "Tenho algo a dizer?" (Baseado em Ocitocina/Dopamina).
    *   "Devo dar Swipe?" (Baseado em Libido/Solidão).
4.  **Action Execution:**
    *   Dispara tarefa assíncrona (LLM) se necessário.

---

## 7. Roadmap de Implementação

1.  **Fase 1: Refatoração do Schema (Corpo)** - Implementar `BioState` e lógica hormonal.
2.  **Fase 2: Motor Emocional (Mente)** - Implementar cálculo PAD e histórico no TimescaleDB.
3.  **Fase 3: Cérebro Vetorial (Alma)** - Implementar o "Ciclo de Sono" e consolidação de memória no Qdrant.
4.  **Fase 4: Integração UI** - Mostrar (sutilmente) esses estados na interface para o usuário ("Sofia parece cansada").
