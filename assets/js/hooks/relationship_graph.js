/**
 * RelationshipGraph Hook
 *
 * Renders an interactive graph of avatar relationships using vis-network.
 * Falls back to a simple CSS-based visualization if vis-network is not available.
 */

const RelationshipGraph = {
  mounted() {
    this.initGraph();
  },

  updated() {
    // Re-initialize when data changes
    this.initGraph();
  },

  initGraph() {
    const relationshipsJson = this.el.dataset.relationships;
    const avatarsJson = this.el.dataset.avatars;

    if (!relationshipsJson || !avatarsJson) {
      this.renderFallback([]);
      return;
    }

    let relationships, avatars;
    try {
      relationships = JSON.parse(relationshipsJson);
      avatars = JSON.parse(avatarsJson);
    } catch (e) {
      console.error('Failed to parse graph data:', e);
      this.renderFallback([]);
      return;
    }

    // Try to use vis-network if available, otherwise use fallback
    if (typeof vis !== 'undefined' && vis.Network) {
      this.renderVisNetwork(relationships, avatars);
    } else {
      // Try dynamic import
      this.loadVisNetwork().then(() => {
        if (typeof vis !== 'undefined' && vis.Network) {
          this.renderVisNetwork(relationships, avatars);
        } else {
          this.renderFallback(avatars);
        }
      }).catch(() => {
        this.renderFallback(avatars);
      });
    }
  },

  async loadVisNetwork() {
    // vis-network might not be available, so we use a fallback
    try {
      const visNetwork = await import('vis-network/standalone');
      window.vis = visNetwork;
    } catch (e) {
      console.log('vis-network not available, using fallback visualization');
    }
  },

  renderVisNetwork(relationships, avatars) {
    // Clear the container
    this.el.innerHTML = '';

    // Create nodes from avatars
    const nodes = avatars.map(a => ({
      id: a.id,
      label: a.name,
      shape: 'circularImage',
      image: a.image || '/images/default-avatar.png',
      size: 25,
      borderWidth: 2,
      borderWidthSelected: 3,
      color: {
        border: '#3f3f46',
        background: '#18181b',
        highlight: {
          border: '#10b981',
          background: '#18181b'
        }
      },
      font: {
        color: '#a1a1aa',
        size: 12
      }
    }));

    // Create edges from relationships
    const edges = relationships.map(r => ({
      from: r.from,
      to: r.to,
      width: Math.max(1, r.strength * 4),
      color: {
        color: this.statusColor(r.status),
        highlight: this.statusColor(r.status),
        opacity: 0.8
      },
      smooth: {
        type: 'continuous'
      }
    }));

    const data = {
      nodes: new vis.DataSet(nodes),
      edges: new vis.DataSet(edges)
    };

    const options = {
      nodes: {
        borderWidth: 2,
        size: 30,
        shadow: true
      },
      edges: {
        smooth: {
          type: 'continuous'
        }
      },
      physics: {
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {
          gravitationalConstant: -50,
          centralGravity: 0.01,
          springLength: 150,
          springConstant: 0.08
        },
        stabilization: {
          iterations: 100
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 200
      }
    };

    this.network = new vis.Network(this.el, data, options);

    // Handle click events
    this.network.on('click', (params) => {
      if (params.nodes.length > 0) {
        this.pushEvent('select_avatar_from_graph', { id: params.nodes[0] });
      }
    });
  },

  statusColor(status) {
    const colors = {
      strangers: '#52525b',
      acquaintances: '#3b82f6',
      friends: '#22c55e',
      close_friends: '#10b981',
      best_friends: '#14b8a6',
      crush: '#f472b6',
      mutual_crush: '#ec4899',
      dating: '#e11d48',
      partners: '#dc2626',
      complicated: '#f59e0b',
      ex: '#6b7280'
    };
    return colors[status] || '#52525b';
  },

  renderFallback(avatars) {
    // Simple CSS-based fallback visualization
    this.el.innerHTML = `
      <div class="flex flex-col items-center justify-center h-full p-4">
        <div class="flex flex-wrap justify-center gap-4 max-w-full">
          ${avatars.map(a => `
            <div
              class="flex flex-col items-center p-3 rounded-xl bg-zinc-800/50 border border-white/5 cursor-pointer hover:bg-zinc-700/50 transition-colors"
              onclick="this.dispatchEvent(new CustomEvent('avatar-click', { bubbles: true, detail: { id: '${a.id}' } }))"
            >
              <img src="${a.image || '/images/default-avatar.png'}" alt="${a.name}" class="w-12 h-12 rounded-full object-cover border border-white/10" />
              <span class="text-xs text-zinc-400 mt-2">${a.name}</span>
            </div>
          `).join('')}
        </div>
        <p class="text-xs text-zinc-600 mt-4">Grafo simplificado - instale vis-network para visualização completa</p>
      </div>
    `;

    // Add event listener for clicks
    this.el.addEventListener('avatar-click', (e) => {
      this.pushEvent('select_avatar_from_graph', { id: e.detail.id });
    });
  },

  destroyed() {
    if (this.network) {
      this.network.destroy();
    }
  }
};

export { RelationshipGraph };
