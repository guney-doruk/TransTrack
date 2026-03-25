#!/usr/bin/env python3
import subprocess
import sys
# Designed for multiple runs for the reproducibility issue.
# Runs the same training with defined number of times.

# Configuration
NUM_RUNS = 5
BASE_OUTPUT_DIR = "/cta/users/grad4/master/TransTrack/output/latest_finetune/mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_ste_fixed"

# Base command arguments
base_cmd = [
    "python3", "-m", "torch.distributed.launch",
    "--nproc_per_node=1",
    "--use_env", "main_track.py",
    "--lr", "0.0000125",
    "--lr_backbone", "0.00000125",
    "--dataset_file", "mots",
    "--coco_path", "/cta/users/grad4/master/datasets/MOTS",
    "--batch_size", "1",
    "--resume", "/cta/users/grad4/master/TransTrack/output/finetune/seg_head_cocoperson_40k_pretrain_frozen_transtrack_continue_from_40th_epoch/checkpoint0099.pth",
    "--with_box_refine",
    "--num_queries", "500",
    "--masks",
    "--epochs", "70",
    "--lr_drop", "35",
    "--no_load_optimizer",
    "--wandb"
]

# Environment variables
env_vars = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": "1"
}

def run_training(run_number):
    """Run a single training iteration"""
    print("=" * 50)
    print(f"Starting run {run_number} of {NUM_RUNS}")
    print("=" * 50)
    
    # Create output directory name
    output_dir = f"{BASE_OUTPUT_DIR}_rerun{run_number}"
    print(f"Output directory: {output_dir}")
    
    # Build complete command
    cmd = base_cmd.copy()
    cmd.insert(cmd.index("main_track.py") + 1, "--output_dir")
    cmd.insert(cmd.index("main_track.py") + 2, output_dir)
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            env={**subprocess.os.environ, **env_vars},
            check=True
        )
        print(f"Run {run_number} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Run {run_number} failed with error code {e.returncode}")
        print("Stopping execution")
        return False

def main():
    """Main function to run all training iterations"""
    for i in range(1, NUM_RUNS + 1):
        success = run_training(i)
        if not success:
            sys.exit(1)
        print()
    
    print("=" * 50)
    print("All runs completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()