#!/usr/bin/env python3
"""
Script to run MOTA evaluation for all grid search experiments.
Iterates through all experiment directories and runs mota.sh for each one.
FIXED: Now properly isolates each experiment's evaluation results.
"""

import os
import subprocess
import json
import shutil
from pathlib import Path

# Configuration
GRID_SEARCH_OUTPUT_DIR = "/cta/users/grad4/master/TransTrack/output/latest_finetune/tracker_grid_search/mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_fixed_rerun3"
GROUNDTRUTH = "/cta/users/grad4/master/datasets/MOTS/train"
GT_TYPE = "_mot_val_half"
THRESHOLD = "-1"

# Python script for evaluation
EVAL_SCRIPT = "../track_tools/eval_motchallenge.py"


def find_experiment_dirs(base_dir):
    """Find all experiment directories (those starting with 'mots_')."""
    base_path = Path(base_dir)  
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    experiment_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("mots_"):
            experiment_dirs.append(item)
    
    return sorted(experiment_dirs)


def check_tracks_dir(exp_dir):
    """Check if the experiment has a val/tracks directory."""
    tracks_dir = exp_dir / "val" / "tracks"
    return tracks_dir.exists()


def create_isolated_eval_workspace(exp_dir):
    """Create an isolated workspace for evaluation to prevent result contamination."""
    eval_workspace = exp_dir / "mota_evaluation_workspace"
    
    # Clean up if exists
    if eval_workspace.exists():
        shutil.rmtree(eval_workspace)
    
    eval_workspace.mkdir(parents=True)
    return eval_workspace


def run_mota_evaluation(exp_dir, results_path, eval_workspace):
    """
    Run MOTA evaluation with proper isolation.
    Changes working directory to the workspace to ensure output files go there.
    """
    
    # Save current directory
    original_cwd = os.getcwd()
    
    # Create symbolic link to results in workspace to avoid path issues
    workspace_results_link = eval_workspace / "tracks"
    if workspace_results_link.exists():
        workspace_results_link.unlink()
    workspace_results_link.symlink_to(results_path.resolve())
    
    cmd = [
        'python3',
        os.path.abspath(EVAL_SCRIPT),
        '--groundtruths', GROUNDTRUTH,
        '--tests', str(workspace_results_link),
        '--gt_type', GT_TYPE,
        '--eval_official',
        '--score_threshold', THRESHOLD
    ]
    
    try:
        # Change to workspace directory so any output files are created there
        os.chdir(eval_workspace)
        
        print(f"    Working directory: {os.getcwd()}")
        print(f"    Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(eval_workspace)  # Explicitly set working directory
        )
        
        stdout = result.stdout
        stderr = result.stderr
        returncode = 0
        
    except subprocess.CalledProcessError as e:
        stdout = e.stdout
        stderr = e.stderr
        returncode = e.returncode
    finally:
        # Always return to original directory
        os.chdir(original_cwd)
    
    return stdout, stderr, returncode


def save_evaluation_results(exp_dir, eval_workspace, stdout, stderr, returncode):
    """Save evaluation results from workspace to permanent location."""
    output_dir = exp_dir / "mota_evaluation_results"
    
    # Remove old results if exist
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Save stdout
    with open(output_dir / "mota_stdout.log", "w") as f:
        f.write(stdout)
    
    # Save stderr
    with open(output_dir / "mota_stderr.log", "w") as f:
        f.write(stderr)
    
    # Copy any generated files from workspace to results directory
    # Common output files from MOT evaluation
    potential_output_files = [
        "*.txt",  # Summary files
        "pedestrian_summary.txt",
        "pedestrian_detailed.csv",
        "*.csv",
        "*.json"
    ]
    
    copied_files = []
    for pattern in potential_output_files:
        for file_path in eval_workspace.glob(pattern):
            if file_path.is_file():
                dest_path = output_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_files.append(file_path.name)
                print(f"    Copied: {file_path.name}")
    
    # Copy any subdirectories that might contain results
    for item in eval_workspace.iterdir():
        if item.is_dir() and item.name != "tracks":  # Skip our symlink
            dest_dir = output_dir / item.name
            shutil.copytree(item, dest_dir)
            copied_files.append(f"{item.name}/")
            print(f"    Copied directory: {item.name}/")
    
    # Save status with list of output files
    status = {
        "returncode": returncode,
        "success": returncode == 0,
        "output_files": copied_files
    }
    with open(output_dir / "mota_status.json", "w") as f:
        json.dump(status, f, indent=2)
    
    # Parse and save MOTA scores if available in stdout
    mota_scores = parse_mota_scores(stdout)
    if mota_scores:
        with open(output_dir / "mota_scores.json", "w") as f:
            json.dump(mota_scores, f, indent=2)
    
    return copied_files


def parse_mota_scores(stdout):
    """Parse MOTA scores from stdout."""
    scores = {}
    
    # Look for common MOTA metric patterns
    lines = stdout.split('\n')
    for line in lines:
        line = line.strip()
        # Common patterns in MOT evaluation output
        if 'MOTA' in line or 'IDF1' in line or 'MT' in line or 'ML' in line:
            scores['raw_output'] = line
        
        # Try to extract numeric scores
        if line.startswith('MOTA'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    scores['MOTA'] = float(parts[1].strip('%'))
                except ValueError:
                    pass
    
    return scores if scores else None


def main():
    print(f"Searching for experiments in: {GRID_SEARCH_OUTPUT_DIR}")
    print(f"Evaluation script: {EVAL_SCRIPT}")
    print("="*80)
    
    # Find all experiment directories
    try:
        experiment_dirs = find_experiment_dirs(GRID_SEARCH_OUTPUT_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not experiment_dirs:
        print("No experiment directories found!")
        return
    
    print(f"\nFound {len(experiment_dirs)} experiment directories\n")
    
    # Process each experiment
    results_summary = []
    for i, exp_dir in enumerate(experiment_dirs, 1):
        exp_name = exp_dir.name
        print(f"{'='*80}")
        print(f"[{i}/{len(experiment_dirs)}] Processing: {exp_name}")
        print(f"{'='*80}")
        
        # Check if val/tracks directory exists
        if not check_tracks_dir(exp_dir):
            print(f"  ⚠️  Skipping: val/tracks directory not found\n")
            results_summary.append({
                "experiment": exp_name,
                "status": "skipped",
                "reason": "val/tracks not found"
            })
            continue
        
        # Get the tracks directory path
        tracks_path = exp_dir / "val" / "tracks"
        print(f"  Tracks path: {tracks_path}")
        
        # Create isolated workspace for this experiment
        eval_workspace = create_isolated_eval_workspace(exp_dir)
        print(f"  Workspace: {eval_workspace}")
        
        # Run MOTA evaluation in isolated workspace
        print(f"  Running MOTA evaluation...")
        stdout, stderr, returncode = run_mota_evaluation(exp_dir, tracks_path, eval_workspace)
        
        # Save results from workspace
        print(f"  Saving results...")
        copied_files = save_evaluation_results(exp_dir, eval_workspace, stdout, stderr, returncode)
        
        # Clean up workspace
        try:
            shutil.rmtree(eval_workspace)
            print(f"  Cleaned up workspace")
        except Exception as e:
            print(f"  Warning: Could not clean workspace: {e}")
        
        if returncode == 0:
            print(f"  ✓ Evaluation completed successfully")
            print(f"  Results saved to: {exp_dir}/mota_evaluation_results/")
            results_summary.append({
                "experiment": exp_name,
                "status": "success",
                "tracks_path": str(tracks_path),
                "output_files": copied_files
            })
        else:
            print(f"  ✗ Evaluation failed with return code {returncode}")
            print(f"  Check logs in: {exp_dir}/mota_evaluation_results/")
            results_summary.append({
                "experiment": exp_name,
                "status": "failed",
                "returncode": returncode
            })
        
        print()  # Empty line for readability
    
    # Save overall summary
    summary_path = Path(GRID_SEARCH_OUTPUT_DIR) / "mota_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print('='*80)
    successful = sum(1 for r in results_summary if r["status"] == "success")
    failed = sum(1 for r in results_summary if r["status"] == "failed")
    skipped = sum(1 for r in results_summary if r["status"] == "skipped")
    
    print(f"Total experiments: {len(experiment_dirs)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⚠️  Skipped: {skipped}")
    print(f"\nDetailed summary saved to: {summary_path}")
    print(f"\nEach experiment's results are in:")
    print(f"  <experiment_dir>/mota_evaluation_results/")
    print('='*80)


if __name__ == "__main__":
    main()