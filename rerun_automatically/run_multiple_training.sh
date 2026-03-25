#!/bin/bash
# """
# Designed for multiple runs for the reproducibility issue.
# Runs the same training with defined number of times.
# """
# Number of runs
NUM_RUNS=4

# Base output directory
BASE_OUTPUT_DIR="/cta/users/grad4/master/TransTrack/output/latest_finetune/mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_fixed"

# Loop through each run
for i in $(seq 1 $NUM_RUNS); do
    echo "=========================================="
    echo "Starting run $i of $NUM_RUNS"
    echo "=========================================="
    
    # Create output directory name with rerun suffix
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_rerun${i}"
    
    echo "Output directory: $OUTPUT_DIR"
    
    # Run the training command
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main_track.py \
        --lr 0.0000125 \
        --lr_backbone 0.00000125 \
        --output_dir "$OUTPUT_DIR" \
        --dataset_file mots \
        --coco_path /cta/users/grad4/master/datasets/MOTS \
        --batch_size 1 \
        --resume /cta/users/grad4/master/TransTrack/output/finetune/seg_head_cocoperson_40k_pretrain_frozen_transtrack_continue_from_40th_epoch/checkpoint0099.pth \
        --with_box_refine \
        --num_queries 500 \
        --masks \
        --epochs 70 \
        --lr_drop 35 \
        --no_load_optimizer \
        --wandb
    
    # Check if the training completed successfully
    if [ $? -eq 0 ]; then
        echo "Run $i completed successfully"
    else
        echo "Run $i failed with error code $?"
        echo "Stopping execution"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All runs completed!"
echo "=========================================="