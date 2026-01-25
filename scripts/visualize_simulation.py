#!/usr/bin/env python3
"""VIVA Simulation Visualizer"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Find latest simulation
data_dir = Path("data/simulations")
files = sorted(data_dir.glob("*_stats.csv"))
if not files:
    print("No simulation data found!")
    sys.exit(1)

latest = files[-1].stem.replace("_stats", "")
print(f"Visualizing: {latest}")

# Load data
stats = pd.read_csv(data_dir / f"{latest}_stats.csv")
pads = pd.read_csv(data_dir / f"{latest}_pads.csv")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
fig.suptitle(f"VIVA Simulation: {latest}", fontsize=14, fontweight='bold')

# 1. PAD means over time
ax1 = fig.add_subplot(2, 3, 1)
ax1.plot(stats['tick'], stats['mean_p'], 'r-', label='Pleasure', linewidth=2)
ax1.plot(stats['tick'], stats['mean_a'], 'g-', label='Arousal', linewidth=2)
ax1.plot(stats['tick'], stats['mean_d'], 'b-', label='Dominance', linewidth=2)
ax1.fill_between(stats['tick'],
                  stats['mean_p'] - stats['std_p'],
                  stats['mean_p'] + stats['std_p'],
                  color='red', alpha=0.2)
ax1.fill_between(stats['tick'],
                  stats['mean_a'] - stats['std_a'],
                  stats['mean_a'] + stats['std_a'],
                  color='green', alpha=0.2)
ax1.fill_between(stats['tick'],
                  stats['mean_d'] - stats['std_d'],
                  stats['mean_d'] + stats['std_d'],
                  color='blue', alpha=0.2)
ax1.set_xlabel('Tick')
ax1.set_ylabel('PAD Value')
ax1.set_title('PAD Dimensions Over Time')
ax1.legend()
ax1.set_ylim(-1.1, 1.1)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 2. Standard deviation over time (emotional diversity)
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(stats['tick'], stats['std_p'], 'r-', label='P std', linewidth=2)
ax2.plot(stats['tick'], stats['std_a'], 'g-', label='A std', linewidth=2)
ax2.plot(stats['tick'], stats['std_d'], 'b-', label='D std', linewidth=2)
ax2.set_xlabel('Tick')
ax2.set_ylabel('Standard Deviation')
ax2.set_title('Emotional Diversity Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. PA plane trajectory (Pleasure-Arousal)
ax3 = fig.add_subplot(2, 3, 3)
scatter = ax3.scatter(stats['mean_p'], stats['mean_a'],
                       c=stats['tick'], cmap='viridis', s=30)
ax3.plot(stats['mean_p'], stats['mean_a'], 'k-', alpha=0.3, linewidth=1)
ax3.set_xlabel('Pleasure')
ax3.set_ylabel('Arousal')
ax3.set_title('PA Plane Trajectory')
ax3.set_xlim(-1.1, 1.1)
ax3.set_ylim(-1.1, 1.1)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.colorbar(scatter, ax=ax3, label='Tick')
# Quadrant labels
ax3.text(0.5, 0.5, 'Excited\nHappy', ha='center', va='center', alpha=0.5)
ax3.text(-0.5, 0.5, 'Stressed', ha='center', va='center', alpha=0.5)
ax3.text(0.5, -0.5, 'Calm\nHappy', ha='center', va='center', alpha=0.5)
ax3.text(-0.5, -0.5, 'Depressed', ha='center', va='center', alpha=0.5)

# 4. Individual VIVA trajectories (sample)
ax4 = fig.add_subplot(2, 3, 4)
sample_vivas = pads['viva_id'].unique()[:5]  # First 5 VIVAs
for vid in sample_vivas:
    viva_data = pads[pads['viva_id'] == vid]
    ax4.plot(viva_data['tick'], viva_data['pleasure'], alpha=0.7, label=f'VIVA {vid}')
ax4.set_xlabel('Tick')
ax4.set_ylabel('Pleasure')
ax4.set_title('Individual VIVA Pleasure Trajectories')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(-1.1, 1.1)

# 5. Final state distribution
ax5 = fig.add_subplot(2, 3, 5)
final_tick = pads['tick'].max()
final_pads = pads[pads['tick'] == final_tick]
ax5.scatter(final_pads['pleasure'], final_pads['arousal'],
            c=final_pads['dominance'], cmap='coolwarm', s=50, alpha=0.7)
ax5.set_xlabel('Pleasure')
ax5.set_ylabel('Arousal')
ax5.set_title(f'Final State Distribution (tick={final_tick})')
ax5.set_xlim(-1.1, 1.1)
ax5.set_ylim(-1.1, 1.1)
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# 6. 3D PAD space
ax6 = fig.add_subplot(2, 3, 6, projection='3d')
# Sample some ticks to avoid clutter
tick_samples = [0, 250, 500, 750, 1000]
colors = plt.cm.viridis(np.linspace(0, 1, len(tick_samples)))
for i, t in enumerate(tick_samples):
    tick_data = pads[pads['tick'] == t]
    if not tick_data.empty:
        ax6.scatter(tick_data['pleasure'], tick_data['arousal'], tick_data['dominance'],
                   c=[colors[i]], s=20, alpha=0.6, label=f't={t}')
ax6.set_xlabel('P')
ax6.set_ylabel('A')
ax6.set_zlabel('D')
ax6.set_title('3D PAD Space Over Time')
ax6.legend(fontsize=8)

plt.tight_layout()

# Save
output_path = data_dir / f"{latest}_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

# Also show if running interactively
plt.savefig("/tmp/viva_sim_viz.png", dpi=150, bbox_inches='tight')
print(f"Also saved to: /tmp/viva_sim_viz.png")
