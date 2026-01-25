#!/usr/bin/env python3
"""VIVA Simulation v2 - Advanced Visualization"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Style
plt.style.use('dark_background')

# Find latest v2 simulation
data_dir = Path("data/simulations")
files = sorted(data_dir.glob("sim_v2_*_stats.csv"))
if not files:
    print("No v2 simulation found!")
    exit(1)

latest = files[-1].stem.replace("_stats", "")
print(f"Visualizing: {latest}")

# Load data
stats = pd.read_csv(data_dir / f"{latest}_stats.csv")
pads = pd.read_csv(data_dir / f"{latest}_pads.csv")
personalities = pd.read_csv(data_dir / f"{latest}_personalities.csv")
events = pd.read_csv(data_dir / f"{latest}_events.csv")

# Colors for personalities
pers_colors = {
    'optimist': '#FFD700',    # Gold
    'neurotic': '#9932CC',    # Purple
    'calm': '#87CEEB',        # Sky blue
    'energetic': '#FF6B35',   # Orange
    'balanced': '#808080'     # Gray
}

# Create figure
fig = plt.figure(figsize=(20, 16))
fig.suptitle(f'VIVA EMOTIONAL DYNAMICS\n{latest}', fontsize=18, fontweight='bold', color='white')

# ═══════════════════════════════════════════════════════════════
# 1. PAD EVOLUTION (top left - large)
# ═══════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(3, 3, 1)

ax1.fill_between(stats['tick'],
                  stats['mean_p'] - stats['std_p'],
                  stats['mean_p'] + stats['std_p'],
                  color='#FF6B6B', alpha=0.3)
ax1.fill_between(stats['tick'],
                  stats['mean_a'] - stats['std_a'],
                  stats['mean_a'] + stats['std_a'],
                  color='#4ECDC4', alpha=0.3)
ax1.fill_between(stats['tick'],
                  stats['mean_d'] - stats['std_d'],
                  stats['mean_d'] + stats['std_d'],
                  color='#45B7D1', alpha=0.3)

ax1.plot(stats['tick'], stats['mean_p'], '#FF6B6B', lw=2.5, label='Pleasure')
ax1.plot(stats['tick'], stats['mean_a'], '#4ECDC4', lw=2.5, label='Arousal')
ax1.plot(stats['tick'], stats['mean_d'], '#45B7D1', lw=2.5, label='Dominance')

# Mark events
for _, ev in events.iterrows():
    ax1.axvline(x=ev['tick'], color='white', alpha=0.3, linestyle='--', lw=1)
    ax1.text(ev['tick'], 0.95, ev['label'][:4], fontsize=7, rotation=90,
             color='white', alpha=0.7, va='top')

ax1.axhline(y=0, color='white', alpha=0.3, lw=0.5)
ax1.set_xlabel('Tick', fontsize=10)
ax1.set_ylabel('PAD Value', fontsize=10)
ax1.set_title('PAD Evolution Over Time', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim(-1.1, 1.1)
ax1.set_xlim(0, stats['tick'].max())

# ═══════════════════════════════════════════════════════════════
# 2. EMOTIONAL DIVERSITY
# ═══════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(3, 3, 2)

ax2.fill_between(stats['tick'], stats['std_p'], color='#FF6B6B', alpha=0.5, label='P diversity')
ax2.fill_between(stats['tick'], stats['std_a'], color='#4ECDC4', alpha=0.5, label='A diversity')
ax2.fill_between(stats['tick'], stats['std_d'], color='#45B7D1', alpha=0.5, label='D diversity')

ax2.set_xlabel('Tick', fontsize=10)
ax2.set_ylabel('Standard Deviation', fontsize=10)
ax2.set_title('Emotional Diversity (Population Spread)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)

# ═══════════════════════════════════════════════════════════════
# 3. PA PLANE TRAJECTORY
# ═══════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(3, 3, 3)

# Background quadrants
ax3.axhspan(0, 1.1, 0, 0.5, alpha=0.1, color='red')      # Stressed
ax3.axhspan(0, 1.1, 0.5, 1, alpha=0.1, color='yellow')   # Excited/Happy
ax3.axhspan(-1.1, 0, 0, 0.5, alpha=0.1, color='blue')    # Depressed
ax3.axhspan(-1.1, 0, 0.5, 1, alpha=0.1, color='green')   # Calm/Happy

scatter = ax3.scatter(stats['mean_p'], stats['mean_a'],
                       c=stats['tick'], cmap='plasma', s=20, alpha=0.8)
ax3.plot(stats['mean_p'], stats['mean_a'], 'white', alpha=0.3, lw=1)

# Start and end markers
ax3.scatter(stats['mean_p'].iloc[0], stats['mean_a'].iloc[0],
            c='lime', s=150, marker='o', edgecolors='white', lw=2, zorder=5, label='Start')
ax3.scatter(stats['mean_p'].iloc[-1], stats['mean_a'].iloc[-1],
            c='red', s=150, marker='X', edgecolors='white', lw=2, zorder=5, label='End')

ax3.axhline(y=0, color='white', alpha=0.5, lw=1)
ax3.axvline(x=0, color='white', alpha=0.5, lw=1)
ax3.set_xlim(-1.1, 1.1)
ax3.set_ylim(-1.1, 1.1)
ax3.set_xlabel('Pleasure', fontsize=10)
ax3.set_ylabel('Arousal', fontsize=10)
ax3.set_title('PA Plane Trajectory', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Tick', shrink=0.8)
ax3.legend(loc='lower left', fontsize=8)

# Quadrant labels
ax3.text(0.5, 0.8, 'EXCITED\nHAPPY', ha='center', fontsize=8, color='yellow', alpha=0.7)
ax3.text(-0.5, 0.8, 'STRESSED', ha='center', fontsize=8, color='red', alpha=0.7)
ax3.text(0.5, -0.8, 'CALM\nHAPPY', ha='center', fontsize=8, color='green', alpha=0.7)
ax3.text(-0.5, -0.8, 'DEPRESSED', ha='center', fontsize=8, color='blue', alpha=0.7)

# ═══════════════════════════════════════════════════════════════
# 4. PERSONALITY CLUSTERS (Final State)
# ═══════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(3, 3, 4)

final_tick = pads['tick'].max()
final_pads = pads[pads['tick'] == final_tick].copy()

for pers, color in pers_colors.items():
    mask = final_pads['personality'] == pers
    ax4.scatter(final_pads.loc[mask, 'pleasure'],
                final_pads.loc[mask, 'arousal'],
                c=color, s=80, alpha=0.8, label=pers, edgecolors='white', lw=0.5)

ax4.axhline(y=0, color='white', alpha=0.3)
ax4.axvline(x=0, color='white', alpha=0.3)
ax4.set_xlim(-1.1, 1.1)
ax4.set_ylim(-1.1, 1.1)
ax4.set_xlabel('Pleasure', fontsize=10)
ax4.set_ylabel('Arousal', fontsize=10)
ax4.set_title(f'Final State by Personality (t={final_tick})', fontsize=12, fontweight='bold')
ax4.legend(loc='lower left', fontsize=8)

# ═══════════════════════════════════════════════════════════════
# 5. PERSONALITY MEAN TRAJECTORIES
# ═══════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(3, 3, 5)

for pers, color in pers_colors.items():
    pers_data = pads[pads['personality'] == pers].groupby('tick')['pleasure'].mean()
    ax5.plot(pers_data.index, pers_data.values, color=color, lw=2, label=pers, alpha=0.9)

ax5.axhline(y=0, color='white', alpha=0.3)
ax5.set_xlabel('Tick', fontsize=10)
ax5.set_ylabel('Mean Pleasure', fontsize=10)
ax5.set_title('Pleasure by Personality Over Time', fontsize=12, fontweight='bold')
ax5.legend(loc='lower right', fontsize=8)
ax5.set_ylim(-1.1, 1.1)

# ═══════════════════════════════════════════════════════════════
# 6. 3D PAD SPACE
# ═══════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(3, 3, 6, projection='3d')

# Sample ticks for clarity
tick_samples = np.linspace(0, final_tick, 6).astype(int)
colors_3d = plt.cm.viridis(np.linspace(0, 1, len(tick_samples)))

for i, t in enumerate(tick_samples):
    closest_tick = pads['tick'].unique()[np.argmin(np.abs(pads['tick'].unique() - t))]
    tick_data = pads[pads['tick'] == closest_tick]
    ax6.scatter(tick_data['pleasure'], tick_data['arousal'], tick_data['dominance'],
                c=[colors_3d[i]], s=15, alpha=0.6, label=f't={closest_tick}')

ax6.set_xlabel('P', fontsize=9)
ax6.set_ylabel('A', fontsize=9)
ax6.set_zlabel('D', fontsize=9)
ax6.set_title('3D PAD Space Evolution', fontsize=12, fontweight='bold')
ax6.legend(loc='upper left', fontsize=7)

# ═══════════════════════════════════════════════════════════════
# 7. QUADRANT DISTRIBUTION OVER TIME
# ═══════════════════════════════════════════════════════════════
ax7 = fig.add_subplot(3, 3, 7)

# Calculate quadrant percentages over time
def get_quadrant(row):
    if row['pleasure'] > 0 and row['arousal'] > 0:
        return 'Excited/Happy'
    elif row['pleasure'] > 0:
        return 'Calm/Happy'
    elif row['arousal'] > 0:
        return 'Stressed'
    return 'Depressed'

pads['quadrant'] = pads.apply(get_quadrant, axis=1)
quad_over_time = pads.groupby(['tick', 'quadrant']).size().unstack(fill_value=0)
quad_pct = quad_over_time.div(quad_over_time.sum(axis=1), axis=0) * 100

quad_colors = {'Excited/Happy': '#FFD93D', 'Calm/Happy': '#6BCB77',
               'Stressed': '#FF6B6B', 'Depressed': '#4D96FF'}

ax7.stackplot(quad_pct.index,
              [quad_pct[q] if q in quad_pct.columns else np.zeros(len(quad_pct))
               for q in quad_colors.keys()],
              labels=quad_colors.keys(),
              colors=quad_colors.values(),
              alpha=0.8)

ax7.set_xlabel('Tick', fontsize=10)
ax7.set_ylabel('Population %', fontsize=10)
ax7.set_title('Emotional Quadrant Distribution', fontsize=12, fontweight='bold')
ax7.legend(loc='upper right', fontsize=8)
ax7.set_ylim(0, 100)

# ═══════════════════════════════════════════════════════════════
# 8. VOLATILITY BY PERSONALITY
# ═══════════════════════════════════════════════════════════════
ax8 = fig.add_subplot(3, 3, 8)

volatility = pads.groupby(['viva_id', 'personality']).agg({
    'pleasure': 'std',
    'arousal': 'std',
    'dominance': 'std'
}).reset_index()
volatility['total_vol'] = volatility['pleasure'] + volatility['arousal'] + volatility['dominance']

vol_by_pers = volatility.groupby('personality')['total_vol'].agg(['mean', 'std']).reset_index()

x_pos = np.arange(len(vol_by_pers))
bars = ax8.bar(x_pos, vol_by_pers['mean'],
               yerr=vol_by_pers['std'],
               color=[pers_colors[p] for p in vol_by_pers['personality']],
               capsize=5, alpha=0.8, edgecolor='white', lw=1)

ax8.set_xticks(x_pos)
ax8.set_xticklabels(vol_by_pers['personality'], rotation=45, ha='right')
ax8.set_ylabel('Total Volatility', fontsize=10)
ax8.set_title('Emotional Stability by Personality', fontsize=12, fontweight='bold')

# ═══════════════════════════════════════════════════════════════
# 9. HEATMAP - Individual VIVA trajectories
# ═══════════════════════════════════════════════════════════════
ax9 = fig.add_subplot(3, 3, 9)

# Pivot for heatmap (sample 20 VIVAs)
sample_vivas = sorted(pads['viva_id'].unique())[:20]
heatmap_data = pads[pads['viva_id'].isin(sample_vivas)].pivot_table(
    index='viva_id', columns='tick', values='pleasure', aggfunc='first'
)

# Sample ticks for readability
tick_cols = heatmap_data.columns[::10]  # Every 10th tick
heatmap_sample = heatmap_data[tick_cols]

im = ax9.imshow(heatmap_sample.values, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
ax9.set_xlabel('Time (tick)', fontsize=10)
ax9.set_ylabel('VIVA ID', fontsize=10)
ax9.set_title('Individual Pleasure Trajectories (20 VIVAs)', fontsize=12, fontweight='bold')

# Tick labels
ax9.set_xticks(np.arange(0, len(tick_cols), 5))
ax9.set_xticklabels([str(int(t)) for t in tick_cols[::5]], fontsize=8)
ax9.set_yticks(np.arange(len(sample_vivas)))
ax9.set_yticklabels(sample_vivas, fontsize=7)

plt.colorbar(im, ax=ax9, label='Pleasure', shrink=0.8)

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = data_dir / f"{latest}_dashboard.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"Saved: {output_path}")

# Also save to /tmp for quick view
plt.savefig("/tmp/viva_dashboard.png", dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("Saved: /tmp/viva_dashboard.png")
