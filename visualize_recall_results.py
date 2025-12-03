#!/usr/bin/env python3
"""
Visualize Recall Evaluation Results - Separate Images
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")

# Load results
df = pd.read_csv("target_evaluation_results.csv")

# Create output directory for images
os.makedirs("evaluation_plots", exist_ok=True)

print("=" * 80)
print("GENERATING RECALL VISUALIZATIONS")
print("=" * 80)

# Calculate metrics
sem_r5_avg = df['semantic_R@5'].mean() * 100
tfidf_r5_avg = df['tfidf_R@5'].mean() * 100
sem_r10_avg = df['semantic_R@10'].mean() * 100
tfidf_r10_avg = df['tfidf_R@10'].mean() * 100
improvement_r5 = ((sem_r5_avg - tfidf_r5_avg) / tfidf_r5_avg * 100) if tfidf_r5_avg > 0 else 0
improvement_r10 = ((sem_r10_avg - tfidf_r10_avg) / tfidf_r10_avg * 100) if tfidf_r10_avg > 0 else 0

# ============================================================================
# 1. Per-Query Comparison - Recall@5
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(14, 7))
queries_short = [q[:25] for q in df['query']]
x = np.arange(len(queries_short))
width = 0.35

bars1 = ax1.bar(x - width/2, df['semantic_R@5']*100, width, label='Semantic Search', 
                color='#3498db', alpha=0.85)
bars2 = ax1.bar(x + width/2, df['tfidf_R@5']*100, width, label='TF-IDF Baseline',
                color='#e67e22', alpha=0.85)

ax1.set_ylabel('Recall@5 (%)', fontweight='bold', fontsize=13)
ax1.set_title('Per-Query Recall@5 Comparison', fontweight='bold', fontsize=16, pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(queries_short, rotation=45, ha='right', fontsize=10)
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(df['semantic_R@5'].max()*110, df['tfidf_R@5'].max()*110, 15))

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('evaluation_plots/recall_1_per_query_recall_at_5.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: evaluation_plots/recall_1_per_query_recall_at_5.png")

# ============================================================================
# 2. Per-Query Comparison - Recall@10
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 7))

bars1 = ax2.bar(x - width/2, df['semantic_R@10']*100, width, label='Semantic Search', 
                color='#2ecc71', alpha=0.85)
bars2 = ax2.bar(x + width/2, df['tfidf_R@10']*100, width, label='TF-IDF Baseline',
                color='#e74c3c', alpha=0.85)

ax2.set_ylabel('Recall@10 (%)', fontweight='bold', fontsize=13)
ax2.set_title('Per-Query Recall@10 Comparison', fontweight='bold', fontsize=16, pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(queries_short, rotation=45, ha='right', fontsize=10)
ax2.legend(fontsize=12, loc='upper right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, max(df['semantic_R@10'].max()*110, df['tfidf_R@10'].max()*110, 20))

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('evaluation_plots/recall_2_per_query_recall_at_10.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: evaluation_plots/recall_2_per_query_recall_at_10.png")

# ============================================================================
# 3. Average Recall@5
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 7))

bars = ax3.bar(['Semantic\nSearch', 'TF-IDF\nBaseline'], [sem_r5_avg, tfidf_r5_avg],
               color=['#3498db', '#e67e22'], alpha=0.8, width=0.5)
ax3.set_ylabel('Recall@5 (%)', fontweight='bold', fontsize=13)
ax3.set_title('Average Recall@5', fontweight='bold', fontsize=16, pad=20)
ax3.set_ylim(0, max(sem_r5_avg, tfidf_r5_avg) * 1.3)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, [sem_r5_avg, tfidf_r5_avg]):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{height:.2f}%', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('evaluation_plots/recall_3_average_recall_at_5.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: evaluation_plots/recall_3_average_recall_at_5.png")

# ============================================================================
# 4. Average Recall@10
# ============================================================================
fig4, ax4 = plt.subplots(figsize=(10, 7))

bars = ax4.bar(['Semantic\nSearch', 'TF-IDF\nBaseline'], [sem_r10_avg, tfidf_r10_avg],
               color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.5)
ax4.set_ylabel('Recall@10 (%)', fontweight='bold', fontsize=13)
ax4.set_title('Average Recall@10', fontweight='bold', fontsize=16, pad=20)
ax4.set_ylim(0, max(sem_r10_avg, tfidf_r10_avg) * 1.3)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, [sem_r10_avg, tfidf_r10_avg]):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{height:.2f}%', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('evaluation_plots/recall_4_average_recall_at_10.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: evaluation_plots/recall_4_average_recall_at_10.png")

# ============================================================================
# 5. Improvement Analysis
# ============================================================================
fig5, ax5 = plt.subplots(figsize=(10, 7))

categories = ['Recall@5', 'Recall@10']
improvements = [improvement_r5, improvement_r10]
bars = ax5.bar(categories, improvements, color=['#3498db', '#9b59b6'], alpha=0.8, width=0.5)
ax5.set_ylabel('Improvement (%)', fontweight='bold', fontsize=13)
ax5.set_title('Relative Improvement: Semantic vs TF-IDF (Recall)', fontweight='bold', fontsize=16, pad=20)
ax5.set_ylim(0, max(max(improvements) * 1.2, 10))
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'+{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('evaluation_plots/recall_5_improvement_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: evaluation_plots/recall_5_improvement_analysis.png")

# ============================================================================
# 6. Win/Loss Analysis for both R@5 and R@10
# ============================================================================
fig6, ax6 = plt.subplots(figsize=(12, 7))

wins_r5 = sum(df['semantic_R@5'] > df['tfidf_R@5'])
wins_r10 = sum(df['semantic_R@10'] > df['tfidf_R@10'])
ties_r5 = sum(df['semantic_R@5'] == df['tfidf_R@5'])
ties_r10 = sum(df['semantic_R@10'] == df['tfidf_R@10'])
losses_r5 = sum(df['semantic_R@5'] < df['tfidf_R@5'])
losses_r10 = sum(df['semantic_R@10'] < df['tfidf_R@10'])

x = np.arange(3)
width = 0.35

bars1 = ax6.bar(x - width/2, [wins_r5, ties_r5, losses_r5], width, label='Recall@5',
                color=['#3498db', '#f39c12', '#e67e22'], alpha=0.8)
bars2 = ax6.bar(x + width/2, [wins_r10, ties_r10, losses_r10], width, label='Recall@10',
                color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)

ax6.set_ylabel('Number of Queries', fontweight='bold', fontsize=13)
ax6.set_title('Win/Tie/Loss Analysis - Recall (Semantic vs TF-IDF)', fontweight='bold', fontsize=16, pad=20)
ax6.set_xticks(x)
ax6.set_xticklabels(['Wins', 'Ties', 'Losses'], fontsize=12)
ax6.set_ylim(0, len(df) + 1)
ax6.legend(fontsize=11)
ax6.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('evaluation_plots/recall_6_win_loss_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: evaluation_plots/recall_6_win_loss_analysis.png")

print("\n" + "=" * 80)
print("ALL RECALL VISUALIZATIONS COMPLETE")
print("=" * 80)
print("\n📁 Saved 6 recall plots in 'evaluation_plots/' directory:")
print("   1. recall_1_per_query_recall_at_5.png   (Per-query R@5 comparison)")
print("   2. recall_2_per_query_recall_at_10.png  (Per-query R@10 comparison)")
print("   3. recall_3_average_recall_at_5.png     (Average R@5)")
print("   4. recall_4_average_recall_at_10.png    (Average R@10)")
print("   5. recall_5_improvement_analysis.png    (Improvement comparison)")
print("   6. recall_6_win_loss_analysis.png       (Win/Tie/Loss)")
print(f"\n📊 Summary:")
print(f"   Recall@5  - Semantic: {sem_r5_avg:.2f}% | TF-IDF: {tfidf_r5_avg:.2f}% | Improvement: +{improvement_r5:.1f}%")
print(f"   Recall@10 - Semantic: {sem_r10_avg:.2f}% | TF-IDF: {tfidf_r10_avg:.2f}% | Improvement: +{improvement_r10:.1f}%")
print("\n" + "=" * 80)

