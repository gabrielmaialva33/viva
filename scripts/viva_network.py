#!/usr/bin/env python3
"""
VIVA Network Science Visualization
Analyzes holographic memory space dynamics using NetworkX

Usage: python scripts/viva_network.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform

# Configuration
VIVA_API = "http://localhost:8888"
OUTPUT_DIR = "output"
PROXIMITY_THRESHOLD = 5.0  # Link bodies within this distance

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== VIVA Network Science Visualization (Python) ===\n")

# Fetch data
print(f"Fetching data from {VIVA_API}...")
try:
    df = pd.read_csv(f"{VIVA_API}/api/export/csv")
except Exception as e:
    print(f"Error: Could not connect to VIVA server.")
    print(f"Make sure the server is running: gleam run -m viva/telemetry/demo")
    sys.exit(1)

print(f"Loaded {len(df)} memory bodies\n")

# === Build NetworkX Graph ===
print("Building network graph...")
G = nx.Graph()

# Add nodes
for _, row in df.iterrows():
    G.add_node(
        row['id'],
        label=row['label'],
        pos=(row['x'], row['y']),
        pos_zw=(row['z'], row['w']),
        energy=row['energy'],
        sleeping=row['sleeping'],
        island_id=row['island_id']
    )

# Add edges based on proximity in HRR-4D space
coords = df[['x', 'y', 'z', 'w']].values
distances = squareform(pdist(coords))

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        if distances[i, j] < PROXIMITY_THRESHOLD:
            G.add_edge(
                df.iloc[i]['id'],
                df.iloc[j]['id'],
                weight=1.0 / (1.0 + distances[i, j])
            )

print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# === Network Metrics ===
print("\n=== Network Metrics ===")

# Density
density = nx.density(G)
print(f"Density (ρ): {density:.4f}")

# Degree distribution
degrees = [d for n, d in G.degree()]
avg_degree = np.mean(degrees) if degrees else 0
print(f"Average degree (k̄): {avg_degree:.2f}")

# Clustering coefficient
clustering = nx.average_clustering(G)
print(f"Clustering coefficient (C): {clustering:.4f}")

# Connected components
components = nx.number_connected_components(G)
print(f"Connected components: {components}")

# === 1. Network Graph Visualization ===
print("\nGenerating network visualization...")
fig, ax = plt.subplots(figsize=(12, 10))

# Node positions from HRR coordinates
pos = {row['id']: (row['x'], row['y']) for _, row in df.iterrows()}

# Node colors based on energy
energies = [G.nodes[n]['energy'] for n in G.nodes()]
node_colors = plt.cm.plasma(np.array(energies) / max(energies) if max(energies) > 0 else energies)

# Node sizes based on energy
node_sizes = [300 + e * 1000 for e in energies]

# Draw network
nx.draw_networkx_edges(G, pos, alpha=0.3, width=1, edge_color='gray', ax=ax)
nx.draw_networkx_nodes(G, pos, node_color=energies, cmap=plt.cm.plasma,
                       node_size=node_sizes, alpha=0.8, ax=ax)

# Labels
labels = {row['id']: row['label'] for _, row in df.iterrows()}
nx.draw_networkx_labels(G, pos, labels, font_size=9, font_color='black', ax=ax)

ax.set_title("VIVA Memory Network (HRR-4D → 2D)", fontsize=16, fontweight='bold')
ax.set_xlabel("X (HRR dimension 1)")
ax.set_ylabel("Y (HRR dimension 2)")
ax.axis('equal')

# Colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(energies), vmax=max(energies)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
cbar.set_label('Energy')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viva_network.png"), dpi=150)
print(f"  -> {OUTPUT_DIR}/viva_network.png")
plt.close()

# === 2. Degree Distribution ===
print("Generating degree distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

degree_counts = pd.Series(degrees).value_counts().sort_index()
ax.bar(degree_counts.index, degree_counts.values, color='steelblue', alpha=0.8, edgecolor='white')
ax.axvline(avg_degree, color='coral', linestyle='--', linewidth=2, label=f'Mean = {avg_degree:.2f}')

ax.set_xlabel("Degree (k)")
ax.set_ylabel("Count")
ax.set_title("Degree Distribution", fontsize=14, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viva_degree_dist.png"), dpi=150)
print(f"  -> {OUTPUT_DIR}/viva_degree_dist.png")
plt.close()

# === 3. Distance Matrix Heatmap ===
print("Generating distance matrix heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

labels_list = df['label'].tolist()
im = ax.imshow(distances, cmap='viridis_r', aspect='equal')

ax.set_xticks(range(len(labels_list)))
ax.set_yticks(range(len(labels_list)))
ax.set_xticklabels(labels_list, rotation=45, ha='right')
ax.set_yticklabels(labels_list)

ax.set_title("HRR-4D Distance Matrix", fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Euclidean Distance')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viva_distance_matrix.png"), dpi=150)
print(f"  -> {OUTPUT_DIR}/viva_distance_matrix.png")
plt.close()

# === 4. Energy vs Position ===
print("Generating energy-position plot...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# X vs Energy
axes[0, 0].scatter(df['x'], df['energy'], c=df['energy'], cmap='plasma', s=100, alpha=0.8)
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Energy')
axes[0, 0].set_title('Energy vs X')

# Y vs Energy
axes[0, 1].scatter(df['y'], df['energy'], c=df['energy'], cmap='plasma', s=100, alpha=0.8)
axes[0, 1].set_xlabel('Y')
axes[0, 1].set_ylabel('Energy')
axes[0, 1].set_title('Energy vs Y')

# Z vs Energy
axes[1, 0].scatter(df['z'], df['energy'], c=df['energy'], cmap='plasma', s=100, alpha=0.8)
axes[1, 0].set_xlabel('Z')
axes[1, 0].set_ylabel('Energy')
axes[1, 0].set_title('Energy vs Z')

# W vs Energy
axes[1, 1].scatter(df['w'], df['energy'], c=df['energy'], cmap='plasma', s=100, alpha=0.8)
axes[1, 1].set_xlabel('W')
axes[1, 1].set_ylabel('Energy')
axes[1, 1].set_title('Energy vs W')

plt.suptitle("Energy Distribution Across HRR-4D Dimensions", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "viva_energy_position.png"), dpi=150)
print(f"  -> {OUTPUT_DIR}/viva_energy_position.png")
plt.close()

# === Summary ===
print("\n=== Summary ===")
print(f"Total bodies: {len(df)}")
print(f"Active: {(~df['sleeping']).sum()}")
print(f"Dormant: {df['sleeping'].sum()}")
print(f"Islands: {df['island_id'].nunique()}")
print(f"Mean energy: {df['energy'].mean():.4f}")
print(f"Energy variance: {df['energy'].var():.6f}")

print("\n=== Done! ===")
