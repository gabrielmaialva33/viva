# Referência da API Thoughtseeds
> *O Teatro da Consciência*

O sistema **Thoughtseeds** implementa a Teoria do Global Workspace (GWT). Ele permite que objetos mentais compitam pela atenção de todo o sistema.

## Conceitos

- **Seed (Semente)**: Uma unidade atômica de pensamento. Contém:
    - `content`: O conteúdo (texto, imagem, struct).
    - `salience`: Saliência/Importância (0.0 - 1.0).
    - `emotion`: Valência emocional associada.
    - `source`: Origem (Voz, Cortex, Corpo).
    - `created_at`: Timestamp biológico.

- **Competição**: A cada 100ms (10Hz), as sementes decaem em saliência. Novos inputs aumentam a saliência. O vencedor assume o "Palco".

- **Transmissão (Broadcasting)**: O vencedor é publicado via `Phoenix.PubSub` para o canal `consciousness:focus`.

## API Elixir (`VivaCore.Consciousness.Workspace`)

### `sow/4`
Planta uma semente.
```elixir
VivaCore.Consciousness.Workspace.sow(content, source, salience, emotion \\ nil)
```

### `current_focus/0`
Obtém o vencedor atual.
```elixir
{:ok, seed} = VivaCore.Consciousness.Workspace.current_focus()
```

### `subscribe/0`
Assina atualizações de consciência.
```elixir
# No init do seu GenServer:
VivaCore.Consciousness.Workspace.subscribe()

# Callback handle_info:
def handle_info({:conscious_focus, seed}, state) do
  Logger.info("Estou consciente de: #{inspect seed.content}")
  {:noreply, state}
end
```
