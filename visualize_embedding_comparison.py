#!/usr/bin/env python3
"""
Visualization Script for Embedding Model Comparison
===================================================
Generates comprehensive graphs and visualizations for embedding model comparison results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Color palette
COLORS = {
    'bge-small-en-v1.5': '#2E86AB',      # Blue
    'all-MiniLM-L6-v2': '#A23B72',       # Purple
    'all-mpnet-base-v2': '#F18F01',     # Orange
    'TF-IDF': '#C73E1D',                 # Red
}

print("=" * 80)
print("GENERATING EMBEDDING MODEL COMPARISON VISUALIZATIONS")
print("=" * 80)

# Load data
print("\n📊 Loading data...")
results_df = pd.read_csv('embedding_model_comparison_results.csv')
summary_df = pd.read_csv('embedding_model_comparison_summary.csv')

print(f"✓ Loaded {len(results_df)} results")
print(f"✓ Loaded {len(summary_df)} model summaries")

# Create output directory
output_dir = Path('evaluation_plots')
output_dir.mkdir(exist_ok=True)
print(f"✓ Output directory: {output_dir}")

# ============================================================================
# 1. Precision@10 Comparison (Bar Chart)
# ============================================================================

print("\n📈 Generating Precision@10 comparison...")
fig, ax = plt.subplots(figsize=(12, 7))

# Extract numeric values - handle both decimal and percentage formats
if summary_df['P@10'].dtype == 'object' and summary_df['P@10'].str.contains('%').any():
    # Percentage format
    summary_df['P@10_num'] = summary_df['P@10'].str.rstrip('%').astype(float)
    summary_df['TFIDF_P@10_num'] = summary_df['vs TF-IDF P@10'].str.rstrip('%').astype(float)
else:
    # Decimal format - convert to percentage
    summary_df['P@10_num'] = summary_df['P@10'].astype(float) * 100
    summary_df['TFIDF_P@10_num'] = summary_df['vs TF-IDF P@10'].astype(float) * 100

# Prepare data
models = summary_df['Model'].tolist()
p10_values = summary_df['P@10_num'].tolist()
tfidf_p10 = summary_df['TFIDF_P@10_num'].iloc[0]  # Same for all

x = np.arange(len(models))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, p10_values, width, label='Semantic Models', 
               color=[COLORS.get(m, '#808080') for m in models], alpha=0.8)
bars2 = ax.bar(x + width/2, [tfidf_p10] * len(models), width, 
               label='TF-IDF Baseline', color=COLORS['TF-IDF'], alpha=0.8)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, p10_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.text(bars2[0].get_x() + bars2[0].get_width()/2., tfidf_p10,
        f'{tfidf_p10:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision@10 (%)', fontsize=12, fontweight='bold')
ax.set_title('Precision@10 Comparison: Embedding Models vs TF-IDF Baseline', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=10, rotation=0)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, max(p10_values + [tfidf_p10]) * 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '1_precision_at_10_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '1_precision_at_10_comparison.png'}")
plt.close()

# ============================================================================
# 2. Recall@10 Comparison (Bar Chart)
# ============================================================================

print("\n📈 Generating Recall@10 comparison...")
fig, ax = plt.subplots(figsize=(12, 7))

# Extract numeric values - handle both decimal and percentage formats
if summary_df['R@10'].dtype == 'object' and summary_df['R@10'].str.contains('%').any():
    # Percentage format
    summary_df['R@10_num'] = summary_df['R@10'].str.rstrip('%').astype(float)
    summary_df['TFIDF_R@10_num'] = summary_df['vs TF-IDF R@10'].str.rstrip('%').astype(float)
else:
    # Decimal format - convert to percentage
    summary_df['R@10_num'] = summary_df['R@10'].astype(float) * 100
    summary_df['TFIDF_R@10_num'] = summary_df['vs TF-IDF R@10'].astype(float) * 100

r10_values = summary_df['R@10_num'].tolist()
tfidf_r10 = summary_df['TFIDF_R@10_num'].iloc[0]  # Same for all

bars1 = ax.bar(x - width/2, r10_values, width, label='Semantic Models',
               color=[COLORS.get(m, '#808080') for m in models], alpha=0.8)
bars2 = ax.bar(x + width/2, [tfidf_r10] * len(models), width,
               label='TF-IDF Baseline', color=COLORS['TF-IDF'], alpha=0.8)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, r10_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.text(bars2[0].get_x() + bars2[0].get_width()/2., tfidf_r10,
        f'{tfidf_r10:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Recall@10 (%)', fontsize=12, fontweight='bold')
ax.set_title('Recall@10 Comparison: Embedding Models vs TF-IDF Baseline',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=10, rotation=0)
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0, max(r10_values + [tfidf_r10]) * 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '2_recall_at_10_comparison.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '2_recall_at_10_comparison.png'}")
plt.close()

# ============================================================================
# 3. Improvement Over TF-IDF (Bar Chart)
# ============================================================================

print("\n📈 Generating improvement analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Precision improvement
summary_df['P_Improvement_num'] = summary_df['P@10 Improvement'].str.replace('+', '').str.rstrip('%').astype(float)
summary_df['R_Improvement_num'] = summary_df['R@10 Improvement'].str.replace('+', '').str.rstrip('%').astype(float)

bars1 = ax1.bar(models, summary_df['P_Improvement_num'], 
                color=[COLORS.get(m, '#808080') for m in models], alpha=0.8)
for bar, val in zip(bars1, summary_df['P_Improvement_num']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'+{val:.1f}%', ha='center', va='bottom', fontweight='bold')

ax1.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
ax1.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax1.set_title('Precision@10 Improvement Over TF-IDF', fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=10)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(axis='y', alpha=0.3)

# Recall improvement
bars2 = ax2.bar(models, summary_df['R_Improvement_num'],
                color=[COLORS.get(m, '#808080') for m in models], alpha=0.8)
for bar, val in zip(bars2, summary_df['R_Improvement_num']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'+{val:.1f}%', ha='center', va='bottom', fontweight='bold')

ax2.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax2.set_title('Recall@10 Improvement Over TF-IDF', fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=10)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / '3_improvement_analysis.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '3_improvement_analysis.png'}")
plt.close()

# ============================================================================
# 4. Per-Query Performance (Line Chart)
# ============================================================================

print("\n📈 Generating per-query performance...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Precision@10 per query
for model in results_df['model'].unique():
    model_data = results_df[results_df['model'] == model]
    ax1.plot(model_data['query'], model_data['semantic_P@10'] * 100, 
             marker='o', label=model, linewidth=2, markersize=8,
             color=COLORS.get(model, '#808080'))

# Add TF-IDF baseline
tfidf_data = results_df[results_df['model'] == results_df['model'].iloc[0]]
ax1.plot(tfidf_data['query'], tfidf_data['tfidf_P@10'] * 100,
         marker='s', label='TF-IDF', linewidth=2, markersize=8,
         color=COLORS['TF-IDF'], linestyle='--', alpha=0.7)

ax1.set_xlabel('Query', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precision@10 (%)', fontsize=12, fontweight='bold')
ax1.set_title('Precision@10 Per Query', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(range(len(tfidf_data)))
ax1.set_xticklabels(tfidf_data['query'], rotation=45, ha='right', fontsize=9)
ax1.legend(loc='best', fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 100)

# Recall@10 per query
for model in results_df['model'].unique():
    model_data = results_df[results_df['model'] == model]
    ax2.plot(model_data['query'], model_data['semantic_R@10'] * 100,
             marker='o', label=model, linewidth=2, markersize=8,
             color=COLORS.get(model, '#808080'))

ax2.plot(tfidf_data['query'], tfidf_data['tfidf_R@10'] * 100,
         marker='s', label='TF-IDF', linewidth=2, markersize=8,
         color=COLORS['TF-IDF'], linestyle='--', alpha=0.7)

ax2.set_xlabel('Query', fontsize=12, fontweight='bold')
ax2.set_ylabel('Recall@10 (%)', fontsize=12, fontweight='bold')
ax2.set_title('Recall@10 Per Query', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(range(len(tfidf_data)))
ax2.set_xticklabels(tfidf_data['query'], rotation=45, ha='right', fontsize=9)
ax2.legend(loc='best', fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, max(results_df['semantic_R@10'].max(), results_df['tfidf_R@10'].max()) * 100 * 1.1)

plt.tight_layout()
plt.savefig(output_dir / '4_per_query_performance.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '4_per_query_performance.png'}")
plt.close()

# ============================================================================
# 5. Heatmap: Precision@10 by Model and Query
# ============================================================================

print("\n📈 Generating precision heatmap...")
fig, ax = plt.subplots(figsize=(14, 6))

# Create pivot table
pivot_p10 = results_df.pivot_table(
    values='semantic_P@10', 
    index='query', 
    columns='model'
) * 100

# Create heatmap
sns.heatmap(pivot_p10, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Precision@10 (%)'}, 
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title('Precision@10 Heatmap: Models vs Queries', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Query', fontsize=12, fontweight='bold')
ax.set_xticklabels([label.get_text().replace('-', '\n') for label in ax.get_xticklabels()], 
                   fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / '5_precision_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '5_precision_heatmap.png'}")
plt.close()

# ============================================================================
# 6. Heatmap: Recall@10 by Model and Query
# ============================================================================

print("\n📈 Generating recall heatmap...")
fig, ax = plt.subplots(figsize=(14, 6))

# Create pivot table
pivot_r10 = results_df.pivot_table(
    values='semantic_R@10',
    index='query',
    columns='model'
) * 100

# Create heatmap
sns.heatmap(pivot_r10, annot=True, fmt='.1f', cmap='YlGnBu',
            cbar_kws={'label': 'Recall@10 (%)'},
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title('Recall@10 Heatmap: Models vs Queries',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Embedding Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Query', fontsize=12, fontweight='bold')
ax.set_xticklabels([label.get_text().replace('-', '\n') for label in ax.get_xticklabels()],
                   fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / '6_recall_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '6_recall_heatmap.png'}")
plt.close()

# ============================================================================
# 7. Comprehensive Dashboard
# ============================================================================

print("\n📈 Generating comprehensive dashboard...")
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 7a. Precision@10 comparison (top left)
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(models))
bars = ax1.bar(x, summary_df['P@10_num'], 
                color=[COLORS.get(m, '#808080') for m in models], alpha=0.8)
ax1.axhline(y=tfidf_p10, color=COLORS['TF-IDF'], linestyle='--', 
            linewidth=2, label=f'TF-IDF ({tfidf_p10:.1f}%)')
for bar, val in zip(bars, summary_df['P@10_num']):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_ylabel('Precision@10 (%)', fontweight='bold')
ax1.set_title('Precision@10 Comparison', fontweight='bold', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=8)
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# 7b. Recall@10 comparison (top middle)
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(x, summary_df['R@10_num'],
                color=[COLORS.get(m, '#808080') for m in models], alpha=0.8)
ax2.axhline(y=tfidf_r10, color=COLORS['TF-IDF'], linestyle='--',
            linewidth=2, label=f'TF-IDF ({tfidf_r10:.1f}%)')
for bar, val in zip(bars, summary_df['R@10_num']):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_ylabel('Recall@10 (%)', fontweight='bold')
ax2.set_title('Recall@10 Comparison', fontweight='bold', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=8)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# 7c. Improvement (top right)
ax3 = fig.add_subplot(gs[0, 2])
x_pos = np.arange(len(models))
bars1 = ax3.bar(x_pos - 0.2, summary_df['P_Improvement_num'], 0.4,
                label='P@10', color='#2E86AB', alpha=0.8)
bars2 = ax3.bar(x_pos + 0.2, summary_df['R_Improvement_num'], 0.4,
                label='R@10', color='#A23B72', alpha=0.8)
ax3.set_ylabel('Improvement (%)', fontweight='bold')
ax3.set_title('Improvement Over TF-IDF', fontweight='bold', fontsize=11)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=8)
ax3.legend(fontsize=8)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax3.grid(axis='y', alpha=0.3)

# 7d. Precision heatmap (middle row, full width)
ax4 = fig.add_subplot(gs[1, :])
pivot_p10_short = pivot_p10.copy()
pivot_p10_short.index = [q[:25] + '...' if len(q) > 25 else q for q in pivot_p10_short.index]
sns.heatmap(pivot_p10_short, annot=True, fmt='.1f', cmap='YlOrRd',
            cbar_kws={'label': 'Precision@10 (%)', 'shrink': 0.8}, 
            linewidths=0.5, linecolor='gray', ax=ax4)
ax4.set_title('Precision@10 by Model and Query', fontweight='bold', fontsize=12, pad=10)
ax4.set_xlabel('Embedding Model', fontweight='bold')
ax4.set_ylabel('Query', fontweight='bold')

# 7e. Per-query precision (bottom left)
ax5 = fig.add_subplot(gs[2, 0])
for model in results_df['model'].unique():
    model_data = results_df[results_df['model'] == model]
    ax5.plot(range(len(model_data)), model_data['semantic_P@10'] * 100,
             marker='o', label=model, linewidth=2, markersize=6,
             color=COLORS.get(model, '#808080'))
ax5.plot(range(len(tfidf_data)), tfidf_data['tfidf_P@10'] * 100,
         marker='s', label='TF-IDF', linewidth=2, markersize=6,
         color=COLORS['TF-IDF'], linestyle='--', alpha=0.7)
ax5.set_xlabel('Query Index', fontweight='bold')
ax5.set_ylabel('Precision@10 (%)', fontweight='bold')
ax5.set_title('Precision@10 Per Query', fontweight='bold', fontsize=11)
ax5.legend(fontsize=7)
ax5.grid(alpha=0.3)

# 7f. Per-query recall (bottom middle)
ax6 = fig.add_subplot(gs[2, 1])
for model in results_df['model'].unique():
    model_data = results_df[results_df['model'] == model]
    ax6.plot(range(len(model_data)), model_data['semantic_R@10'] * 100,
             marker='o', label=model, linewidth=2, markersize=6,
             color=COLORS.get(model, '#808080'))
ax6.plot(range(len(tfidf_data)), tfidf_data['tfidf_R@10'] * 100,
         marker='s', label='TF-IDF', linewidth=2, markersize=6,
         color=COLORS['TF-IDF'], linestyle='--', alpha=0.7)
ax6.set_xlabel('Query Index', fontweight='bold')
ax6.set_ylabel('Recall@10 (%)', fontweight='bold')
ax6.set_title('Recall@10 Per Query', fontweight='bold', fontsize=11)
ax6.legend(fontsize=7)
ax6.grid(alpha=0.3)

# 7g. Summary statistics (bottom right)
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
summary_text = "SUMMARY STATISTICS\n" + "="*30 + "\n\n"
for _, row in summary_df.iterrows():
    summary_text += f"{row['Model']}\n"
    # Use percentage columns if available, otherwise format decimal
    p10_display = row.get('P@10_pct', f"{row['P@10_num']:.2f}%")
    tfidf_p10_display = row.get('vs TF-IDF P@10_pct', f"{summary_df['TFIDF_P@10_num'].iloc[0]:.2f}%")
    r10_display = row.get('R@10_pct', f"{row['R@10_num']:.2f}%")
    tfidf_r10_display = row.get('vs TF-IDF R@10_pct', f"{summary_df['TFIDF_R@10_num'].iloc[0]:.2f}%")
    summary_text += f"  P@10: {p10_display} (vs {tfidf_p10_display})\n"
    summary_text += f"  R@10: {r10_display} (vs {tfidf_r10_display})\n"
    summary_text += f"  Improvement: {row['P@10 Improvement']} / {row['R@10 Improvement']}\n\n"
ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes,
         fontsize=9, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Embedding Model Comparison Dashboard', 
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig(output_dir / '7_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir / '7_comprehensive_dashboard.png'}")
plt.close()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print(f"\n✓ Generated {7} visualization files in '{output_dir}/':")
print("  1. Precision@10 comparison (bar chart)")
print("  2. Recall@10 comparison (bar chart)")
print("  3. Improvement analysis (precision & recall)")
print("  4. Per-query performance (line charts)")
print("  5. Precision@10 heatmap")
print("  6. Recall@10 heatmap")
print("  7. Comprehensive dashboard")
print("\n" + "=" * 80)
