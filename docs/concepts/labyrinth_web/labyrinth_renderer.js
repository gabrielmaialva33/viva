export const LabyrinthRenderer = {
  mounted() {
    this.draw();

    // Draw initial agent position
    const ax = parseInt(this.el.dataset.agentX);
    const ay = parseInt(this.el.dataset.agentY);
    if (!isNaN(ax) && !isNaN(ay)) {
      this.drawAgent(ax, ay);
    }

    this.handleEvent("update_agent", ({ x, y }) => {
      this.drawAgent(x, y);
    });

    // Keyboard Input for Agent Movement
    this.el.addEventListener("keydown", (e) => {
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "w", "a", "s", "d"].includes(e.key)) {
        e.preventDefault();
        this.pushEvent("keydown", { key: e.key });
      }
    });

    // Focus the element to capture keys
    this.el.tabIndex = 0;
    this.el.focus();
  },

  updated() {
    this.draw();
    // Redraw agent if pos is in dataset?
    // Usually "update_agent" event handles dynamic movement.
    // But on re-render, we might need to redraw the agent if it's in the DOM.
    // For now, let's rely on the event or initial draw.
    // We should parse initial agent pos from dataset if provided.
  },

  drawAgent(x, y) {
    // We need to clear previous agent?
    // For "Quantum/Snes" style, maybe we just redraw the whole local area or the whole canvas?
    // Redrawing whole canvas is safest for 32x32.
    this.draw();

    const canvas = this.el.querySelector("#snes-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const scale = 8;

    // Agent = White Pixel (Singularity/Consciousness)
    ctx.fillStyle = "#FFFFFF";
    // Add a glow effect?
    ctx.shadowBlur = 10;
    ctx.shadowColor = "white";
    ctx.fillRect(x * scale, y * scale, scale, scale);
    ctx.shadowBlur = 0; // Reset
  },

  draw() {
    const canvas = this.el.querySelector("#snes-canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = parseInt(this.el.dataset.width);
    const height = parseInt(this.el.dataset.height);
    const rawGrid = JSON.parse(this.el.dataset.grid);
    const scale = 8;

    // Clear Canvas
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, width * scale, height * scale);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const tile = rawGrid[y][x];
        const px = x * scale;
        const py = y * scale;

        switch (tile) {
          case 1: // WALL
            ctx.fillStyle = "#FF0000";
            ctx.fillRect(px, py, scale, scale);
            break;
          case 2: // PATH / FLUX
            ctx.fillStyle = "#002200";
            ctx.fillRect(px, py, scale, scale);
            ctx.fillStyle = "#00FF00";
            ctx.fillRect(px + 2, py + 2, scale - 4, scale - 4);
            break;
          case 3: // CORE
            ctx.fillStyle = "#FFFFFF";
            ctx.fillRect(px, py, scale, scale);
            break;
        }
      }
    }
  }
};
