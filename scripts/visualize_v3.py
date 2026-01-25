#!/usr/bin/env python3
"""VIVA Simulation v3 - Genetic Evolution Visualization"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Style
plt.style.use('dark_background')

# Find latest v3 simulation
data_dir = Path("data/simulations")
files = sorted(data_dir.glob("sim_v3_*_stats.csv"))
if not files:
    print("No v3 simulation found!")
    exit(1)

latest = files[-1].stem.replace("_stats", "")
print(f"Visualizing: {latest}")

# Load data
stats = pd.read_csv(data_dir / f"{latest}_stats.csv")
genomes = pd.read_csv(data_dir / f"{latest}_final_genomes.csv")
generations = pd.read_csv(data_dir / f"{latest}_generations.csv")
events = pd.read_csv(data_dir / f"{latest}_events.csv")
personalities = pd.read_csv(data_dir / f"{latest}_personalities.csv")

# Colors for personalities
pers_colors = {
    'optimist': '#FFD700',    # Gold
    'neurotic': '#9932CC',    # Purple
    'calm': '#87CEEB',        # Sky blue
    'energetic': '#FF6B35',   # Orange
    'balanced': '#808080'     # Gray
}

# Create figure
fig = plt.figure(figsize=(20, 20))
fig.suptitle(f'VIVA GENETIC EVOLUTION v3\n{latest}', fontsize=20, fontweight='bold', color='white')

# =============================================================================
# 1. PAD EVOLUTION WITH EPIGENETIC OVERLAY
# =============================================================================
ax1 = fig.add_subplot(4, 3, 1)

ax1.fill_between(stats['tick'],
                  stats['mean_p'] - stats['std_p'],
                  stats['mean_p'] + stats['std_p'],
                  color='#FF6B6B', alpha=0.3)
ax1.plot(stats['tick'], stats['mean_p'], '#FF6B6B', lw=2.5, label='Pleasure')
ax1.plot(stats['tick'], stats['mean_a'], '#4ECDC4', lw=2.5, label='Arousal')
ax1.plot(stats['tick'], stats['mean_d'], '#45B7D1', lw=2.5, label='Dominance')

# Mark crisis events
for _, ev in events.iterrows():
    if ev['type'] == 'crisis':
        ax1.axvline(x=ev['tick'], color='red', alpha=0.5, linestyle='--', lw=2)
    elif ev['type'] == 'celebration':
        ax1.axvline(x=ev['tick'], color='green', alpha=0.3, linestyle='--', lw=1)

ax1.axhline(y=0, color='white', alpha=0.3, lw=0.5)
ax1.set_xlabel('Tick', fontsize=10)
ax1.set_ylabel('PAD Value', fontsize=10)
ax1.set_title('PAD Evolution (Crisis=Red, Celebration=Green)', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.set_ylim(-1.1, 1.1)

# =============================================================================
# 2. EPIGENETIC DRIFT OVER TIME
# =============================================================================
ax2 = fig.add_subplot(4, 3, 2)

ax2.fill_between(stats['tick'], stats['avg_methylation'], color='#E74C3C', alpha=0.6, label='Methylation')
ax2.fill_between(stats['tick'], stats['avg_trauma'], color='#9B59B6', alpha=0.6, label='Trauma')
ax2.axhline(y=0.18, color='red', linestyle='--', lw=2, label='Trauma Drift Threshold')
ax2.axhline(y=0.03, color='green', linestyle='--', lw=2, label='Resilience Drift Threshold')

ax2.set_xlabel('Tick', fontsize=10)
ax2.set_ylabel('Methylation Level', fontsize=10)
ax2.set_title('Epigenetic Drift Monitoring', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8)
ax2.set_ylim(0, 0.5)

# Add drift status
final_drift = stats['drift'].iloc[-1]
drift_color = 'red' if final_drift == 'TraumaDrift' else 'green' if final_drift == 'ResilienceDrift' else 'white'
ax2.text(stats['tick'].max() * 0.7, 0.45, f'Drift: {final_drift}', fontsize=12,
         color=drift_color, fontweight='bold', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

# =============================================================================
# 3. GENERATION EVOLUTION (Mutations)
# =============================================================================
ax3 = fig.add_subplot(4, 3, 3)

ax3.bar(generations['generation'] - 0.2, generations['trauma_mutation_pct'],
        width=0.4, color='#E74C3C', label='Trauma Mutations', alpha=0.8)
ax3.bar(generations['generation'] + 0.2, generations['resilience_mutation_pct'],
        width=0.4, color='#27AE60', label='Resilience Mutations', alpha=0.8)

ax3.set_xlabel('Generation (x1000 ticks)', fontsize=10)
ax3.set_ylabel('Mutation %', fontsize=10)
ax3.set_title('Mutation Evolution by Generation', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=9)

# =============================================================================
# 4. FLUENCY EVOLUTION
# =============================================================================
ax4 = fig.add_subplot(4, 3, 4)

ax4.plot(generations['generation'], generations['avg_fluency'], 'o-', color='#3498DB', lw=2, ms=8)
ax4.fill_between(generations['generation'], generations['avg_fluency'], color='#3498DB', alpha=0.3)

ax4.set_xlabel('Generation', fontsize=10)
ax4.set_ylabel('Avg Emotional Fluency', fontsize=10)
ax4.set_title('Population Emotional Fluency Over Time', fontsize=12, fontweight='bold')

# =============================================================================
# 5. FINAL GENOME STATE BY PERSONALITY
# =============================================================================
ax5 = fig.add_subplot(4, 3, 5)

for pers, color in pers_colors.items():
    mask = genomes['personality'] == pers
    ax5.scatter(genomes.loc[mask, 'trauma'],
                genomes.loc[mask, 'recovery'],
                c=color, s=100, alpha=0.8, label=pers, edgecolors='white', lw=0.5)

ax5.axhline(y=0.5, color='white', alpha=0.3, linestyle='--')
ax5.axvline(x=0.3, color='white', alpha=0.3, linestyle='--')
ax5.set_xlabel('Trauma Methylation', fontsize=10)
ax5.set_ylabel('Recovery Gene', fontsize=10)
ax5.set_title('Final Genome State (Trauma vs Recovery)', fontsize=12, fontweight='bold')
ax5.legend(loc='upper right', fontsize=8)

# Quadrant labels
ax5.text(0.1, 0.9, 'RESILIENT', ha='center', fontsize=9, color='green', fontweight='bold')
ax5.text(0.5, 0.9, 'RECOVERING', ha='center', fontsize=9, color='yellow', fontweight='bold')
ax5.text(0.1, 0.1, 'VULNERABLE', ha='center', fontsize=9, color='orange', fontweight='bold')
ax5.text(0.5, 0.1, 'TRAPPED', ha='center', fontsize=9, color='red', fontweight='bold')

# =============================================================================
# 6. PERSONALITY DISTRIBUTION PIE
# =============================================================================
ax6 = fig.add_subplot(4, 3, 6)

pers_counts = personalities['personality'].value_counts()
colors = [pers_colors[p] for p in pers_counts.index]
ax6.pie(pers_counts.values, labels=pers_counts.index, colors=colors, autopct='%1.1f%%',
        textprops={'color': 'white', 'fontsize': 10})
ax6.set_title('Personality Distribution', fontsize=12, fontweight='bold')

# =============================================================================
# 7. TRAUMA HEATMAP BY PERSONALITY
# =============================================================================
ax7 = fig.add_subplot(4, 3, 7)

trauma_by_pers = genomes.groupby('personality').agg({
    'trauma': ['mean', 'std'],
    'recovery': ['mean', 'std'],
    'crisis_sensitivity': 'mean'
}).reset_index()
trauma_by_pers.columns = ['personality', 'trauma_mean', 'trauma_std', 'recovery_mean', 'recovery_std', 'sensitivity']

x_pos = np.arange(len(trauma_by_pers))
bars = ax7.bar(x_pos, trauma_by_pers['trauma_mean'],
               yerr=trauma_by_pers['trauma_std'],
               color=[pers_colors[p] for p in trauma_by_pers['personality']],
               capsize=5, alpha=0.8, edgecolor='white', lw=1)

ax7.set_xticks(x_pos)
ax7.set_xticklabels(trauma_by_pers['personality'], rotation=45, ha='right')
ax7.set_ylabel('Trauma Methylation', fontsize=10)
ax7.set_title('Trauma by Personality', fontsize=12, fontweight='bold')

# =============================================================================
# 8. FINAL PAD SCATTER
# =============================================================================
ax8 = fig.add_subplot(4, 3, 8)

for pers, color in pers_colors.items():
    mask = genomes['personality'] == pers
    ax8.scatter(genomes.loc[mask, 'pleasure'],
                genomes.loc[mask, 'arousal'],
                c=color, s=80, alpha=0.7, label=pers, edgecolors='white', lw=0.5)

ax8.axhline(y=0, color='white', alpha=0.3)
ax8.axvline(x=0, color='white', alpha=0.3)
ax8.set_xlim(-1.1, 1.1)
ax8.set_ylim(-1.1, 1.1)
ax8.set_xlabel('Pleasure', fontsize=10)
ax8.set_ylabel('Arousal', fontsize=10)
ax8.set_title('Final PA State by Personality', fontsize=12, fontweight='bold')
ax8.legend(loc='lower left', fontsize=8)

# Quadrant labels
ax8.text(0.5, 0.8, 'EXCITED', ha='center', fontsize=9, color='yellow', alpha=0.7)
ax8.text(-0.5, 0.8, 'STRESSED', ha='center', fontsize=9, color='red', alpha=0.7)
ax8.text(0.5, -0.8, 'CALM', ha='center', fontsize=9, color='green', alpha=0.7)
ax8.text(-0.5, -0.8, 'DEPRESSED', ha='center', fontsize=9, color='blue', alpha=0.7)

# =============================================================================
# 9. CRISIS SENSITIVITY VS RECOVERY (Genetic Trade-off)
# =============================================================================
ax9 = fig.add_subplot(4, 3, 9)

for pers, color in pers_colors.items():
    mask = genomes['personality'] == pers
    ax9.scatter(genomes.loc[mask, 'crisis_sensitivity'],
                genomes.loc[mask, 'recovery'],
                c=color, s=100, alpha=0.8, label=pers, edgecolors='white', lw=0.5)

ax9.axhline(y=0.5, color='white', alpha=0.3, linestyle='--')
ax9.axvline(x=0.5, color='white', alpha=0.3, linestyle='--')
ax9.set_xlabel('Crisis Sensitivity', fontsize=10)
ax9.set_ylabel('Recovery Gene', fontsize=10)
ax9.set_title('Genetic Trade-off: Sensitivity vs Recovery', fontsize=12, fontweight='bold')
ax9.legend(loc='upper right', fontsize=8)

# =============================================================================
# 10. METHYLATION TIMELINE HEATMAP
# =============================================================================
ax10 = fig.add_subplot(4, 3, 10)

# Create mini heatmap from stats
data_for_heatmap = stats[['tick', 'avg_methylation', 'avg_trauma', 'avg_recovery']].copy()
data_for_heatmap = data_for_heatmap.set_index('tick').T

im = ax10.imshow(data_for_heatmap.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.5)
ax10.set_yticks([0, 1, 2])
ax10.set_yticklabels(['Methylation', 'Trauma', 'Recovery'])
ax10.set_xlabel('Time (tick index)', fontsize=10)
ax10.set_title('Epigenetic Timeline Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax10, label='Value', shrink=0.8)

# =============================================================================
# 11. GENERATION TABLE
# =============================================================================
ax11 = fig.add_subplot(4, 3, 11)
ax11.axis('off')

table_data = generations[['generation', 'trauma_mutation_pct', 'resilience_mutation_pct', 'avg_fluency', 'drift']].copy()
table_data.columns = ['Gen', 'Trauma%', 'Resil%', 'Fluency', 'Drift']
table_data = table_data.round(4)

table = ax11.table(cellText=table_data.values,
                   colLabels=table_data.columns,
                   cellLoc='center',
                   loc='center',
                   colColours=['#2C3E50']*5)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

for key, cell in table.get_celld().items():
    cell.set_text_props(color='white')
    if key[0] == 0:
        cell.set_facecolor('#1ABC9C')
    else:
        cell.set_facecolor('#34495E')

ax11.set_title('Generation Evolution Table', fontsize=12, fontweight='bold', pad=20)

# =============================================================================
# 12. SUMMARY STATS
# =============================================================================
ax12 = fig.add_subplot(4, 3, 12)
ax12.axis('off')

summary_text = f"""
GENETIC SIMULATION V3 SUMMARY
=============================

Population: 100 VIVAs
Duration: 10,000 ticks
Events: {len(events)} ({events['type'].value_counts().get('crisis', 0)} crises)

FINAL POPULATION STATE
----------------------
Avg Methylation: {stats['avg_methylation'].iloc[-1]:.4f}
Avg Trauma: {stats['avg_trauma'].iloc[-1]:.4f}
Avg Recovery: {stats['avg_recovery'].iloc[-1]:.4f}
Drift Status: {stats['drift'].iloc[-1]}

MUTATION EVOLUTION
------------------
Initial Trauma Mutations: {generations['trauma_mutation_pct'].iloc[0]:.1f}%
Final Trauma Mutations: {generations['trauma_mutation_pct'].iloc[-1]:.1f}%
Final Resilience Mutations: {generations['resilience_mutation_pct'].iloc[-1]:.1f}%

FLUENCY
-------
Initial Fluency: {generations['avg_fluency'].iloc[0]:.6f}
Final Fluency: {generations['avg_fluency'].iloc[-1]:.6f}
Change: {((generations['avg_fluency'].iloc[-1] / generations['avg_fluency'].iloc[0]) - 1) * 100:.1f}%

PERSONALITY HEALTH
------------------
Most Resilient: optimist (recovery=0.85)
Most Vulnerable: neurotic (recovery=0.05)
"""

ax12.text(0.1, 0.95, summary_text, transform=ax12.transAxes, fontsize=10,
          verticalalignment='top', family='monospace', color='white',
          bbox=dict(boxstyle='round', facecolor='#2C3E50', alpha=0.9))

# =============================================================================
# SAVE
# =============================================================================
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = data_dir / f"{latest}_genetic_dashboard.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print(f"Saved: {output_path}")

# Also save to /tmp for quick view
plt.savefig("/tmp/viva_genetic_dashboard.png", dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("Saved: /tmp/viva_genetic_dashboard.png")
