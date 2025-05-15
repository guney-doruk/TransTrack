#!/usr/bin/env bash


#GROUNDTRUTH=/cta/users/grad4/master/datasets/mot17/train
GROUNDTRUTH=/cta/users/grad4/master/datasets/MOTS/train/
RESULTS=/cta/users/grad4/master/TransTrack/output/finetune/mots_train_no_ignore_from_cocopersonv2_100thepoch_halftrain_halfval_validation/val/tracks
GT_TYPE=_mot
THRESHOLD=-1
IS_ONE_VIDEO=False

python3 ../track_tools/eval_motchallenge.py \
--groundtruths ${GROUNDTRUTH} \
--tests ${RESULTS} \
--gt_type ${GT_TYPE} \
--eval_official \
--score_threshold ${THRESHOLD} \
--is_one_video ${IS_ONE_VIDEO}
