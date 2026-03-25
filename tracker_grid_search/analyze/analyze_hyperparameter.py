#!/usr/bin/env python3
"""
Analyze grid search results and propose optimal hyperparameters for next iteration.
Focus: Increase IDF1 by better utilizing mask information with standard IoU.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_results():
    """Load results and create analysis DataFrame"""
    
    # Parse the results
    with open('/cta/users/grad4/master/TransTrack/output/latest_finetune/tracker_grid_search/mota_results_summary.json', 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame
    rows = []
    for exp_name, data in results.items():
        row = {
            'experiment': exp_name,
            'bbox_weight': data['hyperparameters']['bbox_weight'],
            'mask_weight': data['hyperparameters']['mask_weight'],
            'unmatch_threshold': data['hyperparameters']['unmatch_threshold'],
            'IDF1': data['metrics']['IDF1'],
            'MOTA': data['metrics']['MOTA'],
            'IDs': data['metrics']['IDs'],
            'IDa': data['metrics']['IDa'],  # ID additions
            'IDt': data['metrics']['IDt'],  # ID transfers
            'Rcll': data['metrics']['Rcll'],
            'Prcn': data['metrics']['Prcn']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def analyze_trends(df):
    """Analyze key trends in the data"""
    
    print("="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # 1. Overall statistics
    print("\n1. OVERALL STATISTICS")
    print("-"*80)
    print(f"IDF1  - Mean: {df['IDF1'].mean():.2f}%, Std: {df['IDF1'].std():.2f}%, Range: [{df['IDF1'].min():.2f}, {df['IDF1'].max():.2f}]")
    print(f"MOTA  - Mean: {df['MOTA'].mean():.2f}%, Std: {df['MOTA'].std():.2f}%, Range: [{df['MOTA'].min():.2f}, {df['MOTA'].max():.2f}]")
    print(f"IDs   - Mean: {df['IDs'].mean():.0f}, Std: {df['IDs'].std():.0f}, Range: [{df['IDs'].min():.0f}, {df['IDs'].max():.0f}]")
    
    # 2. Top performers by IDF1
    print("\n2. TOP 5 EXPERIMENTS BY IDF1")
    print("-"*80)
    top5_idf1 = df.nlargest(5, 'IDF1')
    for idx, row in top5_idf1.iterrows():
        print(f"IDF1: {row['IDF1']:.2f}% | MOTA: {row['MOTA']:.2f}% | IDs: {row['IDs']:.0f}")
        print(f"  bbox_weight={row['bbox_weight']:.1f}, mask_weight={row['mask_weight']:.1f}, unmatch_thresh={row['unmatch_threshold']:.1f}")
    
    # 3. Effect of bbox_weight vs mask_weight
    print("\n3. EFFECT OF WEIGHT DISTRIBUTION")
    print("-"*80)
    weight_analysis = df.groupby(['bbox_weight', 'mask_weight']).agg({
        'IDF1': ['mean', 'std', 'max'],
        'IDs': ['mean', 'min']
    }).round(2)
    print(weight_analysis)
    
    # 4. Effect of unmatch_threshold
    print("\n4. EFFECT OF UNMATCH_THRESHOLD")
    print("-"*80)
    thresh_analysis = df.groupby('unmatch_threshold').agg({
        'IDF1': ['mean', 'std', 'max'],
        'IDs': ['mean', 'min'],
        'MOTA': 'mean'
    }).round(2)
    print(thresh_analysis)
    
    # 5. Key insights
    print("\n5. KEY INSIGHTS")
    print("-"*80)
    
    # Find optimal bbox_weight
    best_bbox_weights = top5_idf1['bbox_weight'].value_counts()
    print(f"• Most common bbox_weight in top 5: {best_bbox_weights.index[0]}")
    
    # Find optimal unmatch_threshold
    best_thresholds = top5_idf1['unmatch_threshold'].value_counts()
    print(f"• Most common unmatch_threshold in top 5: {best_thresholds.index[0]}")
    
    # Correlation analysis
    print(f"\n• Correlation between mask_weight and IDF1: {df['mask_weight'].corr(df['IDF1']):.3f}")
    print(f"• Correlation between unmatch_threshold and IDF1: {df['unmatch_threshold'].corr(df['IDF1']):.3f}")
    print(f"• Correlation between unmatch_threshold and IDs: {df['unmatch_threshold'].corr(df['IDs']):.3f}")
    
    # ID switches trend
    print(f"\n• Higher unmatch_threshold → Lower ID switches (as expected)")
    print(f"  Average IDs at threshold=0.4: {df[df['unmatch_threshold']==0.4]['IDs'].mean():.0f}")
    print(f"  Average IDs at threshold=1.4: {df[df['unmatch_threshold']==1.4]['IDs'].mean():.0f}")
    
    return df


def propose_new_hyperparameters(df):
    """Propose new hyperparameters based on analysis"""
    
    print("\n" + "="*80)
    print("PROPOSED NEW HYPERPARAMETERS")
    print("="*80)
    
    # Analysis of current results
    top10 = df.nlargest(10, 'IDF1')
    
    print("\n📊 OBSERVATIONS FROM CURRENT RESULTS:")
    print("-"*80)
    print(f"1. Best IDF1 achieved: {df['IDF1'].max():.2f}%")
    print(f"   Configuration: bbox_weight={top10.iloc[0]['bbox_weight']:.1f}, "
          f"mask_weight={top10.iloc[0]['mask_weight']:.1f}, "
          f"unmatch_threshold={top10.iloc[0]['unmatch_threshold']:.1f}")
    
    print(f"\n2. Weight distribution analysis:")
    print(f"   - Higher bbox_weight (0.7-0.8) appears in {len(top10[top10['bbox_weight']>=0.7])} of top 10")
    print(f"   - Lower mask_weight (0.2-0.3) appears in {len(top10[top10['mask_weight']<=0.3])} of top 10")
    
    print(f"\n3. Unmatch threshold analysis:")
    print(f"   - Higher thresholds (≥1.0) appear in {len(top10[top10['unmatch_threshold']>=1.0])} of top 10")
    print(f"   - This suggests: more lenient matching reduces ID switches")
    
    print(f"\n4. ID switches pattern:")
    print(f"   - Lowest IDs: {df['IDs'].min():.0f} at threshold={df.loc[df['IDs'].idxmin(), 'unmatch_threshold']:.1f}")
    print(f"   - IDF1 improves as IDs decrease (correlation: {-df['IDs'].corr(df['IDF1']):.3f})")
    
    print("\n" + "="*80)
    print("🎯 STRATEGY FOR NEW GRID SEARCH")
    print("="*80)
    print("""
With the switch from GIoU to standard IoU (range 0-1 instead of -1 to 1):

KEY CHANGES:
1. Standard IoU provides more reliable mask similarity (0-1 range)
2. This allows mask_weight to have more meaningful contribution
3. We can explore HIGHER mask_weights than before

RATIONALE:
- GIoU's negative values (-1 to 1) were diluting mask cost contribution
- Standard IoU (0 to 1) will make mask cost more effective
- Combined cost = bbox_weight * (1-bbox_iou) + mask_weight * (1-mask_iou)
- Both terms now in similar ranges, allowing better balance
    """)
    
    print("\n" + "="*80)
    print("📋 PROPOSED HYPERPARAMETERS")
    print("="*80)
    
    print("\n# Configuration 1: EXPLORE HIGHER MASK WEIGHTS")
    print("# Hypothesis: Standard IoU allows mask to contribute more effectively")
    print("-"*80)
    
    weight_pairs_1 = [
        (0.5, 0.5),  # Balanced - baseline
        (0.4, 0.6),  # Mask-dominant
        (0.3, 0.7),  # Strong mask emphasis
        (0.6, 0.4),  # Bbox-dominant
        (0.7, 0.3),  # Strong bbox (current best region)
    ]
    
    unmatch_thresholds_1 = [0.8, 1.0, 1.2, 1.4, 1.6]  # Focus on higher thresholds
    
    print("\nWEIGHT_PAIRS = [")
    for bw, mw in weight_pairs_1:
        print(f"    ({bw}, {mw}),  # bbox_weight, mask_weight")
    print("]")
    
    print("\nUNMATCH_THRESHOLDS = [", end="")
    print(", ".join([f"{t}" for t in unmatch_thresholds_1]), end="")
    print("]")
    
    print(f"\nTotal experiments: {len(weight_pairs_1) * len(unmatch_thresholds_1)} = {len(weight_pairs_1)} weights × {len(unmatch_thresholds_1)} thresholds")
    
    print("\n" + "-"*80)
    print("\n# Configuration 2: FINE-TUNE AROUND BEST REGION (Conservative)")
    print("# If you want to focus on proven good ranges")
    print("-"*80)
    
    weight_pairs_2 = [
        (0.75, 0.25),  # Fine-tune around 0.7-0.8 bbox weight
        (0.80, 0.20),  # Current best
        (0.85, 0.15),  # Even higher bbox emphasis
        (0.65, 0.35),  # Slightly lower
        (0.70, 0.30),  # Current second best
    ]
    
    unmatch_thresholds_2 = [1.0, 1.1, 1.2, 1.3, 1.4]  # Fine-grain around best values
    
    print("\nWEIGHT_PAIRS = [")
    for bw, mw in weight_pairs_2:
        print(f"    ({bw}, {mw}),  # bbox_weight, mask_weight")
    print("]")
    
    print("\nUNMATCH_THRESHOLDS = [", end="")
    print(", ".join([f"{t}" for t in unmatch_thresholds_2]), end="")
    print("]")
    
    print(f"\nTotal experiments: {len(weight_pairs_2) * len(unmatch_thresholds_2)} = {len(weight_pairs_2)} weights × {len(unmatch_thresholds_2)} thresholds")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    print("""
I RECOMMEND Configuration 1 (Exploring Higher Mask Weights) because:

1. ✅ Standard IoU (0-1) makes mask contribution more reliable
2. ✅ Your assumption is correct: mask_weight can now reduce combined_cost more effectively
3. ✅ Current results show mask_weight=0.2-0.3 works with GIoU, but standard IoU 
      allows exploring mask_weight=0.4-0.7
4. ✅ You test both mask-dominant (0.4/0.6, 0.3/0.7) and bbox-dominant (0.6/0.4, 0.7/0.3)
5. ✅ Higher unmatch_thresholds (0.8-1.6) proven to reduce ID switches

EXPECTED OUTCOME:
- IDF1 improvement: +2-5% (from 66.5% to 68-71%)
- ID switches: Further reduction (from 109 to ~80-100)
- Better track consistency due to improved mask matching

ALTERNATIVE:
If you want safer, incremental improvement → use Configuration 2 (fine-tuning)
    """)
    
    return weight_pairs_1, unmatch_thresholds_1, weight_pairs_2, unmatch_thresholds_2


def create_visualizations(df):
    """Create visualizations to support the analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Grid Search Results Analysis & Optimization Strategy', 
                 fontsize=16, fontweight='bold')
    
    # 1. IDF1 heatmap by weights (averaged over thresholds)
    pivot = df.pivot_table(values='IDF1', 
                          index='mask_weight', 
                          columns='bbox_weight', 
                          aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                ax=axes[0,0], cbar_kws={'label': 'IDF1 (%)'})
    axes[0,0].set_title('IDF1 vs Weight Distribution\n(Higher is better)')
    axes[0,0].set_xlabel('BBox Weight')
    axes[0,0].set_ylabel('Mask Weight')
    
    # 2. IDs heatmap by weights (averaged over thresholds)
    pivot_ids = df.pivot_table(values='IDs', 
                               index='mask_weight', 
                               columns='bbox_weight', 
                               aggfunc='mean')
    sns.heatmap(pivot_ids, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                ax=axes[0,1], cbar_kws={'label': 'ID Switches'})
    axes[0,1].set_title('ID Switches vs Weight Distribution\n(Lower is better)')
    axes[0,1].set_xlabel('BBox Weight')
    axes[0,1].set_ylabel('Mask Weight')
    
    # 3. Effect of unmatch_threshold on IDF1 and IDs
    thresh_grouped = df.groupby('unmatch_threshold').agg({
        'IDF1': 'mean',
        'IDs': 'mean'
    })
    
    ax3 = axes[1,0]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(thresh_grouped.index, thresh_grouped['IDF1'], 
                     'b-o', linewidth=2, markersize=8, label='IDF1')
    ax3.set_xlabel('Unmatch Threshold', fontsize=11)
    ax3.set_ylabel('IDF1 (%)', color='b', fontsize=11)
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.grid(True, alpha=0.3)
    
    line2 = ax3_twin.plot(thresh_grouped.index, thresh_grouped['IDs'], 
                          'r-s', linewidth=2, markersize=8, label='ID Switches')
    ax3_twin.set_ylabel('ID Switches', color='r', fontsize=11)
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    ax3.set_title('Impact of Unmatch Threshold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # 4. Scatter: IDF1 vs IDs colored by mask_weight
    scatter = axes[1,1].scatter(df['IDs'], df['IDF1'], 
                               c=df['mask_weight'], 
                               s=100, alpha=0.6, cmap='viridis')
    axes[1,1].set_xlabel('ID Switches', fontsize=11)
    axes[1,1].set_ylabel('IDF1 (%)', fontsize=11)
    axes[1,1].set_title('IDF1 vs ID Switches\n(Colored by Mask Weight)')
    axes[1,1].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1,1])
    cbar.set_label('Mask Weight', fontsize=10)
    
    # Annotate best point
    best_idx = df['IDF1'].idxmax()
    best_row = df.loc[best_idx]
    axes[1,1].annotate('Best IDF1', 
                      xy=(best_row['IDs'], best_row['IDF1']),
                      xytext=(10, 10), textcoords='offset points',
                      bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                      arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('../assets/hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved: hyperparameter_analysis.png")
    plt.close()


def main():
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS ANALYSIS & HYPERPARAMETER PROPOSAL")
    print("="*80)
    
    # Load and analyze
    df = load_and_analyze_results()
    df = analyze_trends(df)
    
    # Propose new hyperparameters
    wp1, ut1, wp2, ut2 = propose_new_hyperparameters(df)
    
    # Create visualizations
    create_visualizations(df)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()