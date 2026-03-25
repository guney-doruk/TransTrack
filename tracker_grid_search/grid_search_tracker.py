#!/usr/bin/env python3
"""
Grid Search Script for Tracker Hyperparameters
Runs inference with different combinations of bbox_weight, mask_weight, and unmatch_threshold
"""

import subprocess
import os
import itertools
from pathlib import Path
import json
import time
from datetime import datetime

WEIGHT_PAIRS = [  
    (1.0, 0.0)   
]
UNMATCH_THRESHOLDS = [1.2, 1.4]

# Base configuration
BASE_OUTPUT_DIR = "/cta/users/grad4/master/TransTrack/output/latest_finetune/tracker_grid_search/mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_fixed_10"
BASE_EXPERIMENT_NAME = "mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_validation_model39_ET"
CHECKPOINT_PATH = "/cta/users/grad4/master/TransTrack/output/latest_finetune/mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_fixed/checkpoint0039.pth"
DATASET_PATH = "/cta/users/grad4/master/datasets/MOTS"

# Fixed arguments
CUDA_DEVICE = "0"
DATASET_FILE = "mots"
BATCH_SIZE = 1
NUM_QUERIES = 500


def create_output_dir_name(bbox_weight, mask_weight, unmatch_threshold):
    """Create a descriptive output directory name with hyperparameter values"""
    dir_name = f"{BASE_EXPERIMENT_NAME}_bw{bbox_weight}_mw{mask_weight}_ut{unmatch_threshold}"
    return os.path.join(BASE_OUTPUT_DIR, dir_name)


def build_command(bbox_weight, mask_weight, unmatch_threshold, output_dir):
    """Build the command string for running inference"""
    cmd = [
        "python3", "main_track.py",
        "--output_dir", output_dir,
        "--dataset_file", DATASET_FILE,
        "--coco_path", DATASET_PATH,
        "--batch_size", str(BATCH_SIZE),
        "--resume", CHECKPOINT_PATH,
        "--eval",
        "--with_box_refine",
        "--num_queries", str(NUM_QUERIES),
        "--masks",
        "--mask",
        #"--bbox_masking",
        "--box_weight", str(bbox_weight),
        "--mask_weight", str(mask_weight),
        "--unmatch_threshold", str(unmatch_threshold)
        #"--use_box_small",
        #"--use_scaled_factor"
    ]
    return cmd


def run_inference(bbox_weight, mask_weight, unmatch_threshold, dry_run=False):
    """Run a single inference experiment"""
    output_dir = create_output_dir_name(bbox_weight, mask_weight, unmatch_threshold)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = build_command(bbox_weight, mask_weight, unmatch_threshold, output_dir)
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    
    print(f"\n{'='*80}")
    print(f"Running experiment:")
    print(f"  bbox_weight={bbox_weight}, mask_weight={mask_weight}, unmatch_threshold={unmatch_threshold}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("DRY RUN - Command that would be executed:")
        print(" ".join(cmd))
        return {"status": "dry_run", "output_dir": output_dir}
    
    # Save hyperparameters to output directory
    config = {
        "bbox_weight": bbox_weight,
        "mask_weight": mask_weight,
        "unmatch_threshold": unmatch_threshold,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "grid_search_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        elapsed_time = time.time() - start_time
        
        print(f"✓ Experiment completed successfully in {elapsed_time:.2f}s")
        
        # Save stdout and stderr
        with open(os.path.join(output_dir, "stdout.log"), "w") as f:
            f.write(result.stdout)
        with open(os.path.join(output_dir, "stderr.log"), "w") as f:
            f.write(result.stderr)
        
        return {
            "status": "success",
            "output_dir": output_dir,
            "elapsed_time": elapsed_time
        }
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Experiment failed after {elapsed_time:.2f}s")
        print(f"Error: {e}")
        
        # Save error logs
        with open(os.path.join(output_dir, "stdout.log"), "w") as f:
            f.write(e.stdout if e.stdout else "")
        with open(os.path.join(output_dir, "stderr.log"), "w") as f:
            f.write(e.stderr if e.stderr else "")
        
        return {
            "status": "failed",
            "output_dir": output_dir,
            "elapsed_time": elapsed_time,
            "error": str(e)
        }


def main(dry_run=False, resume_from=None):
    """
    Main grid search execution
    
    Args:
        dry_run: If True, only print commands without executing
        resume_from: If provided, skip experiments up to this index
    """
    # Generate all combinations
    all_combinations = list(itertools.product(WEIGHT_PAIRS, UNMATCH_THRESHOLDS))
    total_experiments = len(all_combinations)
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH CONFIGURATION")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Weight pairs: {len(WEIGHT_PAIRS)}")
    print(f"Unmatch thresholds: {len(UNMATCH_THRESHOLDS)}")
    print(f"CUDA Device: {CUDA_DEVICE}")
    print(f"Base output directory: {BASE_OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    # Create base output directory
    Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = []
    start_idx = resume_from if resume_from is not None else 0
    
    # Run experiments
    for idx, ((bbox_weight, mask_weight), unmatch_threshold) in enumerate(all_combinations, 1):
        if idx < start_idx:
            print(f"Skipping experiment {idx}/{total_experiments} (resume_from={start_idx})")
            continue
        
        print(f"\n[{idx}/{total_experiments}] Starting experiment...")
        
        result = run_inference(bbox_weight, mask_weight, unmatch_threshold, dry_run)
        result["experiment_id"] = idx
        result["bbox_weight"] = bbox_weight
        result["mask_weight"] = mask_weight
        result["unmatch_threshold"] = unmatch_threshold
        results.append(result)
        
        # Save intermediate results
        results_file = os.path.join(BASE_OUTPUT_DIR, "grid_search_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETED")
    print(f"{'='*80}")
    
    if not dry_run:
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        total_time = sum(r.get("elapsed_time", 0) for r in results)
        
        print(f"Successful: {successful}/{len(results)}")
        print(f"Failed: {failed}/{len(results)}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"\nResults saved to: {results_file}")
        
        if failed > 0:
            print("\nFailed experiments:")
            for r in results:
                if r["status"] == "failed":
                    print(f"  - Experiment {r['experiment_id']}: "
                          f"bw={r['bbox_weight']}, mw={r['mask_weight']}, "
                          f"ut={r['unmatch_threshold']}")
    
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run grid search for tracker hyperparameters")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print commands without executing them")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="Resume from experiment number (skip earlier ones)")
    
    args = parser.parse_args()
    
    main(dry_run=args.dry_run, resume_from=args.resume_from)