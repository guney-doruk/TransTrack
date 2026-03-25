#!/usr/bin/env python3
"""
Visualize MOTA evaluation results from grid search experiments.
Creates comprehensive plots and tables for analysis.
"""


import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

# Configuration
GRID_SEARCH_OUTPUT_DIR = "/cta/users/grad4/master/TransTrack/output/latest_finetune/tracker_grid_search/mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_fixed"
RESULTS_JSON = "mota_results_summary.json"
OUTPUT_DIR = "mota_visualizations"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_dataframe(results):
    """Convert results dictionary to pandas DataFrame."""
    rows = []
    
    for exp_name, data in results.items():
        row = {
            'experiment': exp_name,
            **data['hyperparameters'],
            **data['metrics']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_main_metrics(df, output_dir):
    """Plot main tracking metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Grid Search Results: Main Tracking Metrics', fontsize=16, fontweight='bold')
    
    metrics = ['MOTA', 'IDF1', 'MOTP', 'Rcll', 'Prcn', 'MT']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric in df.columns:
            # Sort by metric value
            sorted_df = df.sort_values(by=metric, ascending=False)
            
            # Plot top 15 experiments
            top_n = min(15, len(sorted_df))
            data = sorted_df.head(top_n)
            
            bars = ax.barh(range(top_n), data[metric].values)
            
            # Color bars by value
            colors = plt.cm.RdYlGn(data[metric].values / data[metric].max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Set labels
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([f"Exp {i+1}" for i in range(top_n)])
            ax.set_xlabel(f'{metric} Value')
            ax.set_title(f'Top 15 by {metric}')
            ax.invert_yaxis()
            
            # Add value labels
            for i, v in enumerate(data[metric].values):
                ax.text(v, i, f' {v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'main_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: main_metrics.png")
    plt.close()


def plot_hyperparameter_heatmaps(df, output_dir):
    """Create heatmaps showing metric values across hyperparameter combinations."""
    
    if not all(col in df.columns for col in ['bbox_weight', 'mask_weight', 'unmatch_threshold']):
        print("⚠️  Missing hyperparameter columns, skipping heatmaps")
        return
    
    metrics_to_plot = ['MOTA', 'IDF1', 'IDs', 'FN']
    
    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{metric} Across Hyperparameter Combinations', fontsize=14, fontweight='bold')
        
        # 1. bbox_weight vs mask_weight (averaged over unmatch_threshold)
        pivot1 = df.pivot_table(
            values=metric,
            index='bbox_weight',
            columns='mask_weight',
            aggfunc='mean'
        )
        sns.heatmap(pivot1, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0], cbar_kws={'label': metric})
        axes[0].set_title('bbox_weight vs mask_weight')
        
        # 2. bbox_weight vs _threshold (averaged over mask_weight)
        pivot2 = df.pivot_table(
            values=metric,
            index='bbox_weight',
            columns='unmatch_threshold',
            aggfunc='mean'
        )
        sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1], cbar_kws={'label': metric})
        axes[1].set_title('bbox_weight vs unmatch_threshold')
        
        # 3. mask_weight vs unmatch_threshold (averaged over bbox_weight)
        pivot3 = df.pivot_table(
            values=metric,
            index='mask_weight',
            columns='unmatch_threshold',
            aggfunc='mean'
        )
        sns.heatmap(pivot3, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[2], cbar_kws={'label': metric})
        axes[2].set_title('mask_weight vs unmatch_threshold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{metric.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: heatmap_{metric.lower()}.png")
        plt.close()


def plot_pareto_frontier(df, output_dir):
    """Plot Pareto frontier for MOTA vs IDF1."""
    
    if 'MOTA' not in df.columns or 'IDF1' not in df.columns:
        print("⚠️  Missing MOTA or IDF1, skipping Pareto plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    scatter = ax.scatter(df['MOTA'], df['IDF1'], 
                        c=df['IDs'] if 'IDs' in df.columns else 'blue',
                        s=100, alpha=0.6, cmap='RdYlGn_r')
    
    # Find and highlight top performers
    top_mota = df.nlargest(3, 'MOTA')
    top_idf1 = df.nlargest(3, 'IDF1')
    top_combined = df.assign(combined=df['MOTA'] + df['IDF1']).nlargest(3, 'combined')
    
    ax.scatter(top_mota['MOTA'], top_mota['IDF1'], 
              color='red', s=200, marker='*', label='Top 3 MOTA', zorder=5)
    ax.scatter(top_idf1['MOTA'], top_idf1['IDF1'], 
              color='blue', s=200, marker='^', label='Top 3 IDF1', zorder=5)
    ax.scatter(top_combined['MOTA'], top_combined['IDF1'], 
              color='green', s=200, marker='s', label='Top 3 Combined', zorder=5)
    
    # Labels
    ax.set_xlabel('MOTA (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('IDF1 (%)', fontsize=12, fontweight='bold')
    ax.set_title('MOTA vs IDF1 Trade-off (Pareto Analysis)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if 'IDs' in df.columns:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('ID Switches', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_mota_idf1.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: pareto_mota_idf1.png")
    plt.close()


# def plot_hyperparameter_effects(df, output_dir):
#     """Plot individual hyperparameter effects on key metrics."""
    
#     if not all(col in df.columns for col in ['bbox_weight', 'mask_weight', 'unmatch_threshold']):
#         print("⚠️  Missing hyperparameter columns, skipping effect plots")
#         return
    
#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))
#     fig.suptitle('Hyperparameter Effects on Key Metrics', fontsize=16, fontweight='bold')
    
#     params = ['bbox_weight', 'mask_weight', 'unmatch_threshold']
#     metrics = ['MOTA', 'IDF1']
    
#     for i, metric in enumerate(metrics):
#         for j, param in enumerate(params):
#             ax = axes[i, j]
            
#             if metric in df.columns:
#                 # Group by parameter and calculate mean and std
#                 grouped = df.groupby(param)[metric].agg(['mean', "min", "max", 'std', 'count'])
                
#                 # Plot with error bars
#                 ax.errorbar(grouped.index, grouped['mean'], 
#                            yerr=grouped['std'], 
#                            marker='o', capsize=5, capthick=2, 
#                            linewidth=2, markersize=8)
                
#                 # Annotate points
#                 for x, y in zip(grouped.index, grouped['mean']):
#                     ax.annotate(f'{y:.1f}', (x, y), 
#                                textcoords="offset points", 
#                                xytext=(0,10), ha='center', fontsize=9)
                
#                 ax.set_xlabel(param.replace('_', ' ').title(), fontsize=11)
#                 ax.set_ylabel(f'{metric} (%)', fontsize=11)
#                 ax.set_title(f'{metric} vs {param.replace("_", " ").title()}')
#                 ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(output_dir / 'hyperparameter_effects.png', dpi=300, bbox_inches='tight')
#     print(f"✓ Saved: hyperparameter_effects.png")
#     plt.close()

def plot_hyperparameter_effects(df, output_dir):
    """Plot individual hyperparameter effects on key metrics with min/max ranges."""
    
    # Determine parameter names
    weight1_col = 'bbox_weight' if 'bbox_weight' in df.columns else 'birth_weight'
    weight2_col = 'mask_weight' if 'mask_weight' in df.columns else 'motion_weight'
    thresh_col = 'unmatch_threshold' if 'unmatch_threshold' in df.columns else 'update_threshold'
    
    if not all(col in df.columns for col in [weight1_col, weight2_col, thresh_col]):
        print("⚠️  Missing hyperparameter columns, skipping effect plots")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Hyperparameter Effects on Key Metrics', fontsize=16, fontweight='bold')
    
    params = [weight1_col, weight2_col, thresh_col]
    metrics = ['MOTA', 'IDF1']
    
    for i, metric in enumerate(metrics):
        for j, param in enumerate(params):
            ax = axes[i, j]
            
            if metric in df.columns:
                # Group by parameter and calculate mean, min, max
                grouped = df.groupby(param)[metric].agg(['mean', 'min', 'max', 'std', 'count'])
                
                # Plot mean line
                ax.plot(grouped.index, grouped['mean'], 
                       'o-', linewidth=2, markersize=8, label='Mean')
                
                # Add shaded region for min-max range
                ax.fill_between(grouped.index, grouped['min'], grouped['max'],
                               alpha=0.2, label='Min-Max Range')
                
                # Annotate points with mean [min-max]
                for x, mean_val, min_val, max_val in zip(grouped.index, grouped['mean'], 
                                                          grouped['min'], grouped['max']):
                    ax.annotate(f'{mean_val:.1f}\n[{min_val:.1f}-{max_val:.1f}]', 
                               (x, mean_val), 
                               textcoords="offset points", 
                               xytext=(0, 15), ha='center', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
                
                ax.set_xlabel(param.replace('_', ' ').title(), fontsize=11)
                ax.set_ylabel(f'{metric} (%)', fontsize=11)
                ax.set_title(f'{metric} vs {param.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_effects.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: hyperparameter_effects.png")
    plt.close()

def create_summary_table(df, output_dir):
    """Create a summary table with top performers."""
    
    # Top 10 by MOTA
    top_10 = df.nlargest(10, 'MOTA')
    
    # Select important columns
    display_cols = ['MOTA', 'IDF1', 'MOTP', 'Rcll', 'Prcn', 'MT', 'IDs', 'FN', 
                   'bbox_weight', 'mask_weight', 'unmatch_threshold']
    display_cols = [col for col in display_cols if col in df.columns]
    
    summary = top_10[display_cols].copy()
    summary.index = range(1, len(summary) + 1)
    
    # Save to CSV
    summary.to_csv(output_dir / 'top_10_results.csv')
    print(f"✓ Saved: top_10_results.csv")
    
    # Create a formatted text table
    with open(output_dir / 'top_10_results.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("TOP 10 EXPERIMENTS BY MOTA\n")
        f.write("="*100 + "\n\n")
        f.write(summary.to_string())
        f.write("\n\n" + "="*100 + "\n")
    
    print(f"✓ Saved: top_10_results.txt")
    
    return summary


def main():
    print("="*80)
    print("MOTA RESULTS VISUALIZATION")
    print("="*80)
    
    # Load results
    json_path = Path(GRID_SEARCH_OUTPUT_DIR) / RESULTS_JSON
    
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        print("Please run the parser script first.")
        return
    
    print(f"\nLoading results from: {json_path}")
    results = load_results(json_path)
    print(f"Loaded {len(results)} experiments\n")
    
    # Create output directory
    output_dir = Path(GRID_SEARCH_OUTPUT_DIR) / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Convert to DataFrame
    df = create_dataframe(results)
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns\n")
    
    print("Generating visualizations...")
    print("-" * 80)
    
    # Generate plots
    plot_main_metrics(df, output_dir)
    plot_hyperparameter_heatmaps(df, output_dir)
    plot_pareto_frontier(df, output_dir)
    plot_hyperparameter_effects(df, output_dir)
    
    # Create summary table
    print("-" * 80)
    print("\nCreating summary table...")
    summary = create_summary_table(df, output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - main_metrics.png")
    print("  - heatmap_*.png (for each metric)")
    print("  - pareto_mota_idf1.png")
    print("  - hyperparameter_effects.png")
    print("  - top_10_results.csv")
    print("  - top_10_results.txt")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if 'MOTA' in df.columns:
        print(f"\nMOTA Statistics:")
        print(f"  Best:  {df['MOTA'].max():.2f}%")
        print(f"  Worst: {df['MOTA'].min():.2f}%")
        print(f"  Mean:  {df['MOTA'].mean():.2f}%")
        print(f"  Std:   {df['MOTA'].std():.2f}%")
    
    if 'IDF1' in df.columns:
        print(f"\nIDF1 Statistics:")
        print(f"  Best:  {df['IDF1'].max():.2f}%")
        print(f"  Worst: {df['IDF1'].min():.2f}%")
        print(f"  Mean:  {df['IDF1'].mean():.2f}%")
        print(f"  Std:   {df['IDF1'].std():.2f}%")


if __name__ == "__main__":
    main()