#!/usr/bin/env bash


GROUNDTRUTH=/home/arslan/Desktop/master/datasets/mot17/train
RESULTS=/home/arslan/Desktop/master/TransTrack/output/finetune/crowdhuman_plus_mot_trainhalf/val/tracks/
GT_TYPE=_val_half
THRESHOLD=-1

python3 ../track_tools/eval_motchallenge.py \
--groundtruths ${GROUNDTRUTH} \
--tests ${RESULTS} \
--gt_type ${GT_TYPE} \
--eval_official \
--score_threshold ${THRESHOLD}
