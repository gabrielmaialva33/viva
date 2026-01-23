# Labyrinth Web Visualization Concept

Conceito de visualização web do labirinto usando Phoenix LiveView.

## Arquivos

- `labyrinth_live.ex` - LiveView com PubSub real-time
- `labyrinth_renderer.js` - Canvas renderer SNES-style

## Conceitos Chave

### Visual
- Canvas 8x8 tiles (SNES resolution)
- CRT scanlines overlay
- Agent com glow effect (shadowBlur)

### Cores
```
WALL = #FF0000 (vermelho)
PATH = #00FF00 (verde)
CORE = #FFFFFF (branco)
AGENT = #FFFFFF + glow
```

### Real-time
- PubSub subscribe em "world:updates"
- `:observer_moved` -> update agent position
- `:universe_reset` -> redraw entire labyrinth (Big Bounce)

### Controls
- WASD / Arrow keys -> Observer.move(direction)

## Uso Futuro

Para reimplementar:
1. `mix phx.new viva_web --umbrella`
2. Copiar esses arquivos
3. Adicionar PubSub config
4. Criar route `/labyrinth`
