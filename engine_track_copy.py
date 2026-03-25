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

import torch
from scripts.utils import pred2mask_save
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from models.matcher import HungarianMatcher

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, scaler: torch.cuda.amp.GradScaler,
                    epoch: int, max_norm: float = 0, fp16=False):
    model.train()
    criterion.train()
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


@torch.no_grad()
def evaluate(model, criterion, postprocessors, matcher, data_loader, base_ds, device, output_dir, mask_out, tracker=None, 
             phase='train', det_val=False, fp16=False):
    tensor_type = torch.cuda.HalfTensor if fp16 else torch.cuda.FloatTensor
    model.eval()
#     criterion.eval()
       
    metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
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
        
        ##NOTE: Calculate MeanIOU start
        ##NOTE: Works only if batchsize is 1. For the test batchsize is always 1. For the train we need to make it 1 to not get out of memory error.
        
        #Bu datasette mean IoU olaylarının 2 veya daha fazla gpu kullanıldığında doğru alınmasının yolu düşünülmesi gerekli Bunun dışında 1 den fazla batch size içinde nasıl kullanılacağının yazılması gerkeli
        #Alttaki kodun aynısı veya benzeri birden fazla GPU durumunda kullanılabilir.
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_boxes = sum(len(t["labels"]) for t in targets)
        # num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        indices = matcher(outputs_without_aux, targets)
        pred_idx = _get_src_permutation_idx(indices)
        tgt_idx = _get_tgt_permutation_idx(indices)

        pred_masks = outputs["pred_masks"] #Shape: 1,500,h,w
        target_masks, valid = utils.nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose() #Shape 1,X,H,W X refers to the object count in the image
        
        target_masks = target_masks.to(pred_masks)
        
        pred_masks = pred_masks[pred_idx]

        pred_masks = utils.interpolate(pred_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        
        pred_masks = pred_masks[:, 0] #Batch size not included

        target_masks = target_masks[tgt_idx] #Batch size not included

        ##NOTE: Batch size ı kaldırmak ister squeeze ile ister dettracktrainde nasıl yapılmışsa öyle. Bunu Yukarıda arada bir yerde de yapmak gerkeebilir ona debug da bak
        pred_binary = (pred_masks.sigmoid() >= 0.4).int()
        target_binary = target_masks.int()
        # if mask_out:
        #     pred2mask_save(targets, pred_binary, target_binary)

        ious = compute_iou(pred_binary, target_binary) #Resulting shape(X,) CHECK THE SHAPE AND LOGİC BEHIND SHAPE

        # #Accumulate IoU and object count CHECK THİS TO MAKE IT CORRECT IF NEEDED
        total_iou += ious.sum().item()
        total_objects += ious.numel()

        ##NOTE: Calculate MeanIOU end

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        
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

    # ##NOTE: Mean IoU last step
    # ##TODO: Find a way to display at the end of the validation and test.
    mean_iou = total_iou / total_objects

    # #Print mean_iou result
    # print("Mean_IOU across dataset: ", mean_iou)

    #save mean_iou
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
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    ##TODO: Use mean_iou for logging pruproses in main track
    return stats, coco_evaluator, res_tracks, mean_iou


def compute_iou(pred_mask, target_mask, offset=1e-6):
    """
    Computes IoU between two binary masks.
    
    Args:
        pred_mask: Tensor of predicted masks, shape (X, H, W).
        target_mask: Tensor of target masks, shape (X, H, W).
    
    Returns:
        iou: Tensor of IoU values, shape (X,).
    """
    intersection = (pred_mask & target_mask).sum(dim=(-2, -1))#shape X
    union = (pred_mask | target_mask).sum(dim=(-2, -1)) #Shape X
    iou = intersection / (union + offset)
    return iou

def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])#Gives the indices belongs to witch batch
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx
