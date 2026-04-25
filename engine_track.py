# Modified by Peize Sun, Rufeng Zhang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import copy

import torch
from scripts.utils import mean_iou_func, apply_bbox_masking_and_visualize_eval
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, scaler: torch.cuda.amp.GradScaler,
                    epoch: int, max_norm: float = 0, fp16=False, bbox_masking=False):
    model.train()
    criterion.train()

    if bbox_masking:
        if hasattr(criterion, 'reset_epoch_counter'):
            criterion.reset_epoch_counter()
            print(f"Reset epoch counter for epoch {epoch}")

        if hasattr(criterion, 'current_epoch'):
            criterion.current_epoch = epoch

    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        samples.tensors = samples.tensors.type(tensor_type)
        samples.mask = samples.mask.type(tensor_type)

        # with torch.cuda.amp.autocast(enabled=fp16):
        with torch.autocast(device_type="cuda", enabled=fp16):
            outputs, pre_outputs, pre_targets = model([samples, targets])
            loss_dict = criterion(outputs, targets, pre_outputs, pre_targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_uncertainty_loss(model: torch.nn.Module, 
                    criterion: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    scaler: torch.cuda.amp.GradScaler,
                    epoch: int, 
                    max_norm: float = 0, 
                    fp16: bool = False,
                    bbox_masking = False):
    
    model.train()
    criterion.train()

    if bbox_masking:
        if hasattr(criterion, 'reset_epoch_counter'):
            criterion.reset_epoch_counter()
            print(f"Reset epoch counter for epoch {epoch}")

        if hasattr(criterion, 'current_epoch'):
            criterion.current_epoch = epoch

    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = f'Epoch: [{epoch}]'
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        samples.tensors = samples.tensors.type(tensor_type)
        samples.mask = samples.mask.type(tensor_type)

        with torch.autocast(device_type="cuda", enabled=fp16):
            outputs, pre_outputs, pre_targets = model([samples, targets])
        
            total_loss, loss_dict_raw, weighted_losses, learned_log_vars = criterion(
                outputs, targets, pre_outputs, pre_targets
            )
            total_loss *= 0.5

        loss_dict_reduced = utils.reduce_dict(loss_dict_raw)
        loss_value = total_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        # Logging (raw losses + uncertainty vars)
        metric_logger.update(loss=loss_value)
        for k, v in loss_dict_reduced.items():
            metric_logger.update(**{f"{k}_unscaled": v.item()})
        for k, v in weighted_losses.items():
            metric_logger.update(**{f"{k}_weighted": v.item()})
        for k, v in learned_log_vars.items():
            metric_logger.update(**{f"log_var_{k}": v})
        metric_logger.update(class_error=loss_dict_reduced.get('class_error', torch.tensor(0.0)))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        samples, targets = prefetcher.next()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, matcher, data_loader, base_ds, device, output_dir, masks, mask, mask_out, tracker=None, 
             phase='train', det_val=False, fp16=False, bbox_masking=False, infer_masks_without_gt=False):
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    model.eval()
#     criterion.eval()
       
    metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys()) if not infer_masks_without_gt else tuple(k for k in ('bbox',) if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    res_tracks = dict()
    pre_embed = None
    # Correct gt for tracking decoder in matcher for box filtering in inference where tracking and detection decoder works at the same time
    prev_targets = None
    # For Mean IoU calculation
    total_iou = 0.0
    total_objects = 0 # bu kısmı number of Boxesdaki gibide alabilirsin
    #End
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # pre process for track.
        if tracker is not None:
            if phase != 'train':
                assert samples.tensors.shape[0] == 1, "Now only support inference of batchsize 1." 
            frame_id = targets[0].get("frame_id", None)
            assert frame_id is not None
            frame_id = frame_id.item()
            if frame_id == 1:
                tracker.reset_all()
                pre_embed = None
                
        samples = samples.to(device)
        samples.tensors = samples.tensors.type(tensor_type)
        samples.mask = samples.mask.type(tensor_type)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=fp16):
            if det_val:
                outputs = model(samples)
            else:
                outputs, pre_embed = model(samples, pre_embed)
            
        if mask_out:
            _ = apply_bbox_masking_and_visualize_eval(outputs,targets,matcher,save_binary_masks=mask_out)
            
        if mask_out and bbox_masking:
            #Get bbox masked maks
            _ = apply_bbox_masking_and_visualize_eval(outputs,targets,matcher,save_binary_masks=mask_out)
            
#             loss_dict = criterion(outputs, targets)
            
#         weight_dict = criterion.weight_dict

#         reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])
        
        if masks and mask and not infer_masks_without_gt:
            ious, pred_idx, tgt_idx = mean_iou_func(outputs, targets, prev_targets, matcher, mask_out)
            total_iou += ious.sum().item()
            total_objects += ious.numel()
        else:
            _, pred_idx, tgt_idx = mean_iou_func(outputs, targets, prev_targets, matcher, mask_out, return_only_idxs=True)

        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, targets, orig_target_sizes, target_sizes, pred_idx, tgt_idx, bbox_masking)
        
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # post process for track.
        if tracker is not None:
            if frame_id == 1:
                res_track = tracker.init_track(results[0])
            else:
                res_track = tracker.step(results[0])
            res_tracks[targets[0]['image_id'].item()] = res_track

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        #Update prev_targets
        #prev_targets = copy.deepcopy(targets)

    # ##NOTE: Mean IoU last step
    # ##TODO: Find a way to display at the end of the validation and test.
    if masks and mask and not infer_masks_without_gt:
        mean_iou = total_iou / total_objects
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, "mean_iou_log.txt")

        with open(output_file, 'a') as f:
            f.write(f"Mean_IOU: {mean_iou}\n")
        
    # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys() and not infer_masks_without_gt:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator, res_tracks
