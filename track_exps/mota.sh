#!/usr/bin/env bash


#GROUNDTRUTH=/cta/users/grad4/master/datasets/mot17/train
GROUNDTRUTH=/cta/users/grad4/master/datasets/MOTS/train #Datasetteki tek video içinse o videonun adıda yazılmalı.
RESULTS=/cta/users/grad4/master/TransTrack/output/latest_finetune/tracker_grid_search/mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_fixed_rerun3/mots_train_from_cocopersonv2_100thepoch_halftrain_halfval_consistency_loss_validation_model39_ET_rerun3_bw0.8_mw0.2_ut1.4/val/tracks
GT_TYPE=_mot_val_half
THRESHOLD=-1
#IS_ONE_VIDEO=FALSE

python3 ../track_tools/eval_motchallenge.py \
--groundtruths ${GROUNDTRUTH} \
--tests ${RESULTS} \
--gt_type ${GT_TYPE} \
--eval_official \
--score_threshold ${THRESHOLD} \
#--is_one_video 

