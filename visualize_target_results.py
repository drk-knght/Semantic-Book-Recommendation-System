#!/usr/bin/env python3
"""
Visualization of Target Evaluation Results
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


print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Calculate metrics
sem_p5_avg = df['semantic_P@5'].mean() * 100
tfidf_p5_avg = df['tfidf_P@5'].mean() * 100
sem_p10_avg = df['semantic_P@10'].mean() * 100
tfidf_p10_avg = df['tfidf_P@10'].mean() * 100
improvement_p5 = ((sem_p5_avg - tfidf_p5_avg) / tfidf_p5_avg * 100)
improvement_p10 = ((sem_p10_avg - tfidf_p10_avg) / tfidf_p10_avg * 100)



### 1. Per-Query Comparison - Precision@5
fig1, ax1 = plt.subplots(figsize=(14, 7))
queries_short = [q[:25] for q in df['query']]
x = np.arange(len(queries_short))
width = 0.35

bars1 = ax1.bar(x - width/2, df['semantic_P@5']*100, width, label='Semantic Search', color='#27ae60', alpha=0.85)
bars2 = ax1.bar(x + width/2, df['tfidf_P@5']*100, width, label='TF-IDF Baseline',color='#e67e22', alpha=0.85)
ax1.set_ylabel('Precision@5 (%)', fontweight='bold', fontsize=13)
ax1.set_title('Per-Query Precision@5 Comparison', fontweight='bold', fontsize=16, pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(queries_short, rotation=45, ha='right', fontsize=10)
ax1.legend(fontsize=12, loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 110)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('evaluation_plots/1_per_query_precision_at_5.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: evaluation_plots/1_per_query_precision_at_5.png")



### 2. Per-Query Comparison - Precision@10
fig2, ax2 = plt.subplots(figsize=(14, 7))

bars1 = ax2.bar(x - width/2, df['semantic_P@10']*100, width, label='Semantic Search', color='#2ecc71', alpha=0.85)
bars2 = ax2.bar(x + width/2, df['tfidf_P@10']*100, width, label='TF-IDF Baseline', color='#e74c3c', alpha=0.85)
ax2.set_ylabel('Precision@10 (%)', fontweight='bold', fontsize=13)
ax2.set_title('Per-Query Precision@10 Comparison', fontweight='bold', fontsize=16, pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(queries_short, rotation=45, ha='right', fontsize=10)
ax2.legend(fontsize=12, loc='upper right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 110)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('evaluation_plots/2_per_query_precision_at_10.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: evaluation_plots/2_per_query_precision_at_10.png")



### 3. Average Precision@5
fig3, ax3 = plt.subplots(figsize=(10, 7))
bars = ax3.bar(['Semantic\nSearch', 'TF-IDF\nBaseline'], [sem_p5_avg, tfidf_p5_avg],color=['#27ae60', '#e67e22'], alpha=0.8, width=0.5)
ax3.set_ylabel('Precision@5 (%)', fontweight='bold', fontsize=13)
ax3.set_title('Average Precision@5', fontweight='bold', fontsize=16, pad=20)
ax3.set_ylim(0, 110)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, [sem_p5_avg, tfidf_p5_avg]):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2, f'{height:.1f}%', ha='center', va='bottom',fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('evaluation_plots/3_average_precision_at_5.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: evaluation_plots/3_average_precision_at_5.png")


### 4. Average Precision@10
fig4, ax4 = plt.subplots(figsize=(10, 7))
bars = ax4.bar(['Semantic\nSearch', 'TF-IDF\nBaseline'], [sem_p10_avg, tfidf_p10_avg],color=['#2ecc71', '#e74c3c'], alpha=0.8, width=0.5)
ax4.set_ylabel('Precision@10 (%)', fontweight='bold', fontsize=13)
ax4.set_title('Average Precision@10', fontweight='bold', fontsize=16, pad=20)
ax4.set_ylim(0, 110)
ax4.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('evaluation_plots/4_average_precision_at_10.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: evaluation_plots/4_average_precision_at_10.png")



### 5. Improvement Analysis
fig5, ax5 = plt.subplots(figsize=(10, 7))
categories = ['Precision@5', 'Precision@10']
improvements = [improvement_p5, improvement_p10]
bars = ax5.bar(categories, improvements, color=['#3498db', '#9b59b6'], alpha=0.8, width=0.5)
ax5.set_ylabel('Improvement (%)', fontweight='bold', fontsize=13)
ax5.set_title('Relative Improvement: Semantic vs TF-IDF', fontweight='bold', fontsize=16, pad=20)
ax5.set_ylim(0, max(140, max(improvements) + 10))
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 2, f'+{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig('evaluation_plots/5_improvement_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: evaluation_plots/5_improvement_analysis.png")


### 6. Analysis for both P@5 and P@10
fig6, ax6 = plt.subplots(figsize=(12, 7))
wins_p5 = sum(df['semantic_P@5'] > df['tfidf_P@5'])
wins_p10 = sum(df['semantic_P@10'] > df['tfidf_P@10'])
ties_p5 = sum(df['semantic_P@5'] == df['tfidf_P@5'])
ties_p10 = sum(df['semantic_P@10'] == df['tfidf_P@10'])
losses_p5 = sum(df['semantic_P@5'] < df['tfidf_P@5'])
losses_p10 = sum(df['semantic_P@10'] < df['tfidf_P@10'])

x = np.arange(3)
width = 0.35

bars1 = ax6.bar(x - width/2, [wins_p5, ties_p5, losses_p5], width, label='Precision@5', color=['#27ae60', '#f39c12', '#e67e22'], alpha=0.8)
bars2 = ax6.bar(x + width/2, [wins_p10, ties_p10, losses_p10], width, label='Precision@10', color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.8)
ax6.set_ylabel('Number of Queries', fontweight='bold', fontsize=13)
ax6.set_title('Analysis (Semantic vs TF-IDF)', fontweight='bold', fontsize=16, pad=20)
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
plt.savefig('evaluation_plots/6_win_loss_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: evaluation_plots/6_win_loss_analysis.png")



print("ALL VISUALIZATIONS COMPLETE")
print("=" * 80)
print("\n Saved 6 separate plots in 'evaluation_plots/' directory:")
print("   1. 1_per_query_precision_at_5.png   (Per-query P@5 comparison)")
print("   2. 2_per_query_precision_at_10.png  (Per-query P@10 comparison)")
print("   3. 3_average_precision_at_5.png     (Average P@5)")
print("   4. 4_average_precision_at_10.png    (Average P@10)")
print("   5. 5_improvement_analysis.png       (Improvement comparison)")
print("   6. 6_win_loss_analysis.png          (Win/Tie/Loss)")
print(f"\n Summary:")
print(f"   Precision@5  - Semantic: {sem_p5_avg:.1f}% | TF-IDF: {tfidf_p5_avg:.1f}% | Improvement: +{improvement_p5:.1f}%")
print(f"   Precision@10 - Semantic: {sem_p10_avg:.1f}% | TF-IDF: {tfidf_p10_avg:.1f}% | Improvement: +{improvement_p10:.1f}%")
print("\n" + "=" * 80)

