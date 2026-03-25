#!/usr/bin/env python3
"""
Parse MOTA evaluation results from all experiments and create a summary JSON.
Extracts the OVERALL row metrics from each experiment's stdout.
"""

import json
import re
from pathlib import Path

# Configuration
GRID_SEARCH_OUTPUT_DIR = "/cta/users/grad4/master/TransTrack/output/latest_finetune/tracker_grid_search/mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_fixed_rerun3"
OUTPUT_JSON = "mota_results_summary.json"


def parse_overall_metrics(stdout_content):
    """ 
    Parse the OVERALL row from MOTA output.
    Extracts all metrics from the detailed output table.
    """
    lines = stdout_content.split('\n')
    
    # Find the OVERALL line in the detailed output (with IDF1, IDP, IDR, etc.)
    overall_line = None
    header_line = None
    
    for i, line in enumerate(lines):
        if 'IDF1' in line and 'IDP' in line and 'IDR' in line:
            header_line = line
            # Look for OVERALL in the next few lines
            for j in range(i+1, min(i+10, len(lines))):
                if 'OVERALL' in lines[j]:
                    overall_line = lines[j]
                    break
            if overall_line:
                break
    
    if not overall_line or not header_line:
        return None
    
    # Parse header to get column names
    headers = header_line.split()
    
    # Parse OVERALL values
    # Remove 'OVERALL' label and split the rest
    values_str = overall_line.replace('OVERALL', '').strip()
    values = values_str.split()
    
    # Create metrics dictionary
    metrics = {}
    
    # Map values to headers
    for i, header in enumerate(headers):
        if i < len(values):
            value = values[i]
            
            # Clean percentage signs and convert to float
            if '%' in value:
                try:
                    metrics[header] = float(value.replace('%', ''))
                except ValueError:
                    metrics[header] = value
            else:
                # Try to convert to appropriate type
                try:
                    # Try integer first
                    if '.' not in value:
                        metrics[header] = int(value)
                    else:
                        metrics[header] = float(value)
                except ValueError:
                    metrics[header] = value
    
    return metrics


def extract_hyperparameters(exp_name):
    """
    Extract hyperparameters from experiment name.
    Example: mots_..._bw0.4_mw0.6_ut0.4 -> {'bw': 0.4, 'mw': 0.6, 'ut': 0.4}
    """
    params = {}
    
    # Extract bw (bbox_weight)
    bw_match = re.search(r'bw([\d.]+)', exp_name)
    if bw_match:
        params['bbox_weight'] = float(bw_match.group(1))
    
    # Extract mw (mask_weight)
    mw_match = re.search(r'mw([\d.]+)', exp_name)
    if mw_match:
        params['mask_weight'] = float(mw_match.group(1))
    
    # Extract ut (unmatch_threshold)
    ut_match = re.search(r'ut([\d.]+)', exp_name)
    if ut_match:
        params['unmatch_threshold'] = float(ut_match.group(1))
    
    return params


def find_experiment_dirs(base_dir):
    """Find all experiment directories."""
    base_path = Path(base_dir)
    experiment_dirs = []
    
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("mots_"):
            experiment_dirs.append(item)
    
    return sorted(experiment_dirs)


def main():
    print(f"Parsing MOTA results from: {GRID_SEARCH_OUTPUT_DIR}")
    print("="*80)
    
    # Find all experiment directories
    experiment_dirs = find_experiment_dirs(GRID_SEARCH_OUTPUT_DIR)
    
    if not experiment_dirs:
        print("No experiment directories found!")
        return
    
    print(f"Found {len(experiment_dirs)} experiments\n")
    
    # Parse results from each experiment
    all_results = {}
    parsed_count = 0
    failed_count = 0
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        stdout_path = exp_dir / "mota_evaluation_results" / "mota_stdout.log"
        
        if not stdout_path.exists():
            print(f"⚠️  {exp_name}: No stdout.log found")
            failed_count += 1
            continue
        
        # Read stdout
        with open(stdout_path, 'r') as f:
            stdout_content = f.read()
        
        # Parse metrics
        metrics = parse_overall_metrics(stdout_content)
        
        if metrics:
            # Extract hyperparameters
            hyperparams = extract_hyperparameters(exp_name)
            
            # Combine metrics and hyperparameters
            all_results[exp_name] = {
                'metrics': metrics,
                'hyperparameters': hyperparams
            }
            
            print(f"✓ {exp_name}")
            print(f"  MOTA: {metrics.get('MOTA', 'N/A')}%, IDF1: {metrics.get('IDF1', 'N/A')}%")
            parsed_count += 1
        else:
            print(f"✗ {exp_name}: Failed to parse metrics")
            failed_count += 1
    
    # Save to JSON
    output_path = Path(GRID_SEARCH_OUTPUT_DIR) / OUTPUT_JSON
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully parsed: {parsed_count}")
    print(f"Failed to parse: {failed_count}")
    print(f"\nResults saved to: {output_path}")
    
    # Print top 5 by MOTA
    # if all_results:
    #     print(f"\n{'='*80}")
    #     print("TOP 5 EXPERIMENTS BY MOTA")
    #     print(f"{'='*80}")
        
    #     sorted_results = sorted(
    #         all_results.items(),
    #         key=lambda x: x[1]['metrics'].get('MOTA', 0),
    #         reverse=True
    #     )
        
    #     for i, (name, data) in enumerate(sorted_results[:5], 1):
    #         metrics = data['metrics']
    #         params = data['hyperparameters']
    #         print(f"{i}. MOTA: {metrics.get('MOTA', 'N/A')}% | IDF1: {metrics.get('IDF1', 'N/A')}%")
    #         print(f"   {name}")
    #         print(f"   Params: bw={params.get('bbox_weight', 'N/A')}, "
    #               f"mw={params.get('motion_weight', 'N/A')}, "
    #               f"ut={params.get('unmatch_threshold', 'N/A')}")
    #         print()


if __name__ == "__main__":
    main()