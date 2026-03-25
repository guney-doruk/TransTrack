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
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from torchvision.ops import masks_to_boxes
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .seg_head_detr_backbone import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer_track import build_deforamble_transformer
import copy
from scipy.optimize import linear_sum_assignment


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        num_channels = backbone.num_channels[-3:]
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides) - 1
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_] 
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([nn.Conv2d(num_channels[0], hidden_dim, kernel_size=1)])
        self.combine = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
    
    @torch.no_grad()
    def randshift(self, samples, targets):
        bs = samples.tensors.shape[0]
        
        self.xshift = (100 * torch.rand(bs)).int()
        self.xshift *= (torch.randn(bs) > 0.0).int() * 2 - 1 
        self.yshift = (100 * torch.rand(bs)).int()
        self.yshift *= (torch.randn(bs) > 0.0).int() * 2 - 1
        
        shifted_images = []
        new_targets = copy.deepcopy(targets)
        
        for i, (image, target) in enumerate(zip(samples.tensors, targets)):
            _, h, w = image.shape
            img_h, img_w = target['size']
            nopad_image = image[:, :img_h, :img_w]
            image_patch = \
            nopad_image[:,
                  max(0, -self.yshift[i]) : min(h, h - self.yshift[i]), 
                  max(0, -self.xshift[i]) : min(w, w - self.xshift[i])] 
            
            _, patch_h, patch_w = image_patch.shape
            ratio_h, ratio_w = img_h / patch_h,  img_w / patch_w 
            shifted_image = F.interpolate(image_patch[None], size=(img_h, img_w))[0]
            pad_shifted_image = copy.deepcopy(image)
            pad_shifted_image[:, :img_h, :img_w] = shifted_image
            shifted_images.append(pad_shifted_image)
            
            scale = torch.tensor([img_w, img_h, img_w, img_h], device=image.device)[None]
            bboxes = target['boxes'] * scale
            bboxes -= torch.tensor([max(0, -self.xshift[i]), max(0, -self.yshift[i]), 0, 0], device=image.device)[None]
            bboxes *= torch.tensor([ratio_w, ratio_h, ratio_w, ratio_h], device=image.device)[None]
            shifted_bboxes = bboxes / scale
            new_targets[i]['boxes'] = shifted_bboxes
                        
        new_samples = copy.deepcopy(samples)
        new_samples.tensors = torch.stack(shifted_images, dim=0)
        
        return new_samples, new_targets
            
    def forward(self, samples_targets, unused_embed=None):
        if self.training:
            samples, targets = samples_targets        
            pre_samples, pre_targets = self.randshift(samples, targets)
            prepre_samples, _ = self.randshift(samples, targets)

            pre_out, pre_embed = self.forward_once(pre_samples, prepre_samples, pre_targets, targets)             
            
            if torch.randn(1).item() > 0.0:
                out, _ = self.forward_train(samples, pre_embed)     
            else:
                for key in pre_embed:
                    if key != 'feat':
                        pre_embed[key] = None
                out, _ = self.forward_train(samples, pre_embed)
                pre_out = None
                pre_targets = None
            return out, pre_out, pre_targets
        
        else:
            samples = samples_targets
            out, _ = self.forward_train(samples)         
            return out, None
    
    @torch.no_grad()    
    def forward_once(self, samples: NestedTensor, train_samples: NestedTensor, targets=None, next_targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        features_all = features
        features = features[-3:]
        
        pos_all = pos
        pos = pos[-3:]

        if not isinstance(train_samples, NestedTensor):
            train_samples = nested_tensor_from_tensor_list(train_samples)
        pre_feat, _ = self.backbone(train_samples)
        pre_feat_all = pre_feat
        pre_feat = pre_feat[-3:]
        
        srcs = []
        masks = []
        
        for l, (feat, feat2) in enumerate(zip(features, pre_feat)):
            src, mask = feat.decompose()
            src2, _ = feat2.decompose()
            srcs.append(self.combine(torch.cat([self.input_proj[l](src), self.input_proj[l](src2)], dim=1)))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.combine(torch.cat([self.input_proj[l](features[-1].tensors), self.input_proj[l](pre_feat[-1].tensors)], dim=1))
                else:
                    src = self.input_proj[l](srcs[-1])

                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
            
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, memory = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
               
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        pre_embed = {'reference': outputs_coord[-1], 'tgt': hs[-1], 'feat': features, 'memory': memory}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)        
        
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out, pre_embed
    
    def forward_train(self, samples: NestedTensor, pre_embed=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        
        if pre_embed is not None:
            pre_reference, pre_tgt, pre_feat, pre_memory = pre_embed['reference'], pre_embed['tgt'], pre_embed['feat'], pre_embed['memory']
        else:
            pre_reference = None
            pre_tgt = None
            pre_memory = None
            pre_feat = features
        
        srcs = []
        masks = []
        
        for l, (feat, feat2) in enumerate(zip(features, pre_feat)):
            src, mask = feat.decompose()
            src2, _ = feat2.decompose()
            srcs.append(self.combine(torch.cat([self.input_proj[l](src), self.input_proj[l](src2)], dim=1)))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.combine(torch.cat([self.input_proj[l](features[-1].tensors), self.input_proj[l](pre_feat[-1].tensors)], dim=1))
                else:
                    src = self.input_proj[l](srcs[-1])

                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
            
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight        
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, _ = self.transformer(srcs, masks, pos, query_embeds, pre_reference, pre_tgt)           
            
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage and self.training:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out, None

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, args_):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            ignored_region_handling: whether handle ignored regions or not.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = args_.focal_alpha
        self.ignored_region_handling = args_.ignored_region_handling
        self.bbox_masking = args_.bbox_masking
        self.epochs = args_.epochs
        self.current_epoch = args_.start_epoch

        self.epoch_image_counter = 0
        if args_.uncertainity_loss:
            self.log_vars = nn.ParameterDict({
                "detection": nn.Parameter(torch.zeros(1)),
                "classification": nn.Parameter(torch.zeros(1)),
                "segmentation": nn.Parameter(torch.zeros(1)),
            })

            if "mask_box_consistency_loss" in losses:
                self.log_vars["consistency"] = nn.Parameter(torch.zeros(1))

    # def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
    #     """Classification loss (NLL)
    #     targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    #     """
    #     assert 'pred_logits' in outputs
    #     src_logits = outputs['pred_logits']

    #     idx = self._get_src_permutation_idx(indices)
        
    #     batch_size, num_objects, _ = outputs['pred_boxes'].shape
    #     # if self.ignored_region_handling:
    #     #     unique_idx_range = torch.unique(idx[0])
    #     #     #unmatched_idx = {i: list(range(num_objects)) for i in range(batch_size)}
    #     #     unmatched_idx, unmatched_boxes, batch_ignore = {}, [], {}

    #     #     # Convert idx to a dictionary of sets for fast lookup
    #     #     batch_to_matched = {torch.tensor(batch).item(): set(idx[1][idx[0] == torch.tensor(batch)].tolist()) for batch in range(batch_size)}

    #     #     # Initialize unmatched_idx as a dictionary comprehension
    #     #     unmatched_idx = {batch: list(set(range(num_objects)) - batch_to_matched.get(batch, set())) for batch in range(batch_size)}

    #     #     # Use list comprehension to collect unmatched boxes
    #     #     unmatched_boxes = [outputs['pred_boxes'][batch][unmatched_idx[batch]] for batch in range(batch_size)]
           
    #     #     #mask = torch.ones((batch_size, num_objects), dtype=torch.bool, device=src_logits.device)

    #     #     # Set indices from idx to False
    #     #     #mask[idx] = False

    #     #     # Apply mask to src_boxes
    #     #     #unmatched_src_boxes = [outputs['pred_boxes'][i][mask[i]] for i in range(batch_size)] #[(419,4), (480,4)] for batchsize 2
            
    #     #     #Get sizes after transform
    #     #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    #     #     assert target_sizes.shape[1] == 2
            
    #     #     target_masks_ignored = [t["masks_ignore"] for t in targets]
    #     #     #batch_ignore = {i: [] for i in range(batch_size)}
    #     #     for batch in range(batch_size):
    #     #         # unmatched_idx[batch] = list(range(num_objects))
    #     #         # matched_ids = idx[1][idx[0] == batch].tolist()
    #     #         # for matched_id in matched_ids:
    #     #         #     unmatched_idx[batch].remove(matched_id)
    #     #         # unmatched_boxes.append(outputs['pred_boxes'][batch][unmatched_idx[batch]]) #NOTE: mathcleşmeyen maskelerin bulunması buraya taşındı bu şekilde memoryden ve speed tasarruf edildi

    #     #         if target_masks_ignored[batch].shape[0] != 0: #If image has no ignore region, we should pass
    #     #             batch_ignore[batch] = None
    #     #             if unmatched_boxes == [] or unmatched_boxes is None:
    #     #                 print(f"unmatched boxes: {unmatched_boxes}, unmatched idx: {unmatched_idx}, Batch for: {batch}, unique idx range: {unique_idx_range} ")
    #     #             elif unmatched_idx == {} or unmatched_idx is None:
    #     #                 print(f"unmatched boxes: {unmatched_boxes}, unmatched idx: {unmatched_idx}, Batch for: {batch}, unique idx range: {unique_idx_range} ")
    #     #             converted_boxes = box_ops.box_cxcywh_to_xyxy(unmatched_boxes[batch])
    #     #             img_h, img_w = target_sizes[batch]
    #     #             scale_fct = torch.stack([img_w, img_h, img_w, img_h])  # Shape: (4,)
    #     #             converted_boxes = converted_boxes * scale_fct
    #     #             unmatched_boxes[batch] = converted_boxes
    #     #             #Guard agains overflow for ignored region calculation
    #     #             unmatched_boxes[batch][:, 0::2].clamp_(min=0, max=img_w)
    #     #             unmatched_boxes[batch][:, 1::2].clamp_(min=0, max=img_h)
                    
    #     #             #Getting indexes of %50 or more match with binary mask
    #     #             #idx_for = self.filter_boxes_by_mask_coverage(unmatched_boxes[batch], target_masks_ignored[batch], unmatched_idx, batch)
    #     #             idx_vec = self.filter_boxes_by_mask_coverage_vectorized(unmatched_boxes[batch], target_masks_ignored[batch], unmatched_idx, batch)
    #     #             #idx_for_w = self.filter_boxes_by_mask_coverage_with_where(unmatched_boxes[i], target_masks_ignored[i])
    #     #             batch_ignore[batch] = idx_vec
    #     #         else:
    #     #             pass
    #     # else:
    #     #     batch_ignore = None

    #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    #     target_classes = torch.full(src_logits.shape[:2], self.num_classes,
    #                                 dtype=torch.int64, device=src_logits.device)
    #     target_classes[idx] = target_classes_o

    #     # for batch, unmatched_ids in batch_ignore.items():
    #     #     target_classes[batch, unmatched_ids] = 11 #Ignore region class
    #     #     #src_logits[batch, unmatched_ids] = 10

    #     target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
    #                                         dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    #     target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

    #     target_classes_onehot = target_classes_onehot[:,:,:-1]
        
    #     # if self.ignored_region_handling:
    #     #     t = target_classes_onehot.clone()
    #     #     s = src_logits.clone()
    #     #     #s = s.sigmoid()
    #     #     for batch, unmatched_ids in batch_ignore.items():
    #     #         #target_classes_onehot[batch, unmatched_ids] = 0
    #     #         src_logits[batch, unmatched_ids] = 0

    #     #         s[batch, unmatched_ids] = t[batch, unmatched_ids]

    #     loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, batch_ignore, self.ignored_region_handling, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
    #     #loss_ce_1 = sigmoid_focal_loss(s, t, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
    #     losses = {'loss_ce': loss_ce}

    #     if log:
    #         # TODO this should probably be a separate loss, not hacked in this one here
    #         losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    #     return losses
    
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        ##NOTE: Burada boxların valid gelip gelmediğine bakılması gerekli diye düşünüyorum.""
        if self.bbox_masking:
            # Get target sizes
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            # Get bbox preds
            src_boxes = outputs['pred_boxes'][src_idx]
            src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            src_boxes = src_boxes * scale_fct
            
            # Take sigmoid before box masking
            src_masks_sigmoid = src_masks[:, 0].sigmoid()

            #Arragnge target_mask so that it will have same index with src_mask
            target_masks_plot = target_masks[tgt_idx]
            
            # Apply bbox masking
            # self.reset_epoch_counter()
            src_masks_sigmoid = self.apply_bbox_masking(src_masks_sigmoid, src_boxes, target_masks=target_masks_plot, current_epoch=self.current_epoch, batch_size=len(targets))#use save_binary_mask = False for not saving th box filterek masks as jpg
            
            # Convert back to the format expected by loss functions
            src_masks_masked = src_masks_sigmoid.flatten(1)
        else:
            src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        input_masks = src_masks_masked if self.bbox_masking else src_masks

        losses = {
            "loss_mask": sigmoid_focal_loss(input_masks, target_masks, num_boxes, bbox_masking=self.bbox_masking),
            "loss_dice": dice_loss(input_masks, target_masks, num_boxes, bbox_masking=self.bbox_masking),
        }
        return losses
    
    def mask_box_consistency_loss(self, outputs, targets, indices, num_boxes):
        """"This loss is designed for ensuring the consistency between good box results and masks by getting the box from masks and the gt boxes by L1 and GIoU loss
        The Aim: By looking at the difference between mask box and gt boxes model tries to outputs consistent masks with good box head outputs"""
        
        assert 'pred_boxes' in outputs
        assert 'pred_masks' in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][src_idx]
        src_masks = outputs['pred_masks'][src_idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        img_h, img_w = targets[0]['size'].tolist()
        
        def get_bbox_from_soft_mask(masks):
            """
            Extract bounding boxes from soft masks using variance-based differentiable operations.
            This allows gradients to flow back to the mask predictions.
            
            Args:
                masks: torch.Tensor of shape [M, height, width] with soft masks (values in [0, 1])
                
            Returns:
                bboxes: torch.Tensor of shape [M, 4] where each row is [center_x, center_y, width, height]
            """
            # Handle empty tensor case
            if masks.numel() == 0:
                return torch.zeros((0, 4), device=masks.device)
            
            B, H, W = masks.shape
            device = masks.device
            dtype = masks.dtype
            
            # Create coordinate grids
            y_coords = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
            x_coords = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
            
            # Compute weighted center using mask values as weights
            # Clamp to avoid division by zero
            total_mass = masks.sum(dim=(1, 2)).clamp(min=1e-4)
            
            # Check for nearly empty masks (total mass below threshold)
            # Use 1e-4 as threshold - filters numerical noise and essentially empty masks
            #has_content = total_mass > 1e-4
            
            center_y = (masks * y_coords).sum(dim=(1, 2)) / total_mass
            center_x = (masks * x_coords).sum(dim=(1, 2)) / total_mass
            
            # Compute variance to estimate box size
            cy_expanded = center_y.view(-1, 1, 1)
            cx_expanded = center_x.view(-1, 1, 1)
            
            var_y = (masks * (y_coords - cy_expanded)**2).sum(dim=(1, 2)) / total_mass
            var_x = (masks * (x_coords - cx_expanded)**2).sum(dim=(1, 2)) / total_mass
            
            # Use 2.2*sqrt(variance) to approximate box size for person masks
            # Person masks have medium compactness with extended limbs
            # Multiplier 2.2 works well for articulated human bodies
            height = 2.2 * torch.sqrt(var_y.clamp(min=1e-6))
            width = 2.2 * torch.sqrt(var_x.clamp(min=1e-6))
            
            # Clamp box dimensions to reasonable values
            # Prevent extremely small or large boxes
            height = height.clamp(min=1.0, max=H)
            width = width.clamp(min=1.0, max=W)
            
            # Clamp centers to be within image bounds
            center_y = center_y.clamp(min=0, max=H)
            center_x = center_x.clamp(min=0, max=W)
            
            bboxes = torch.stack([center_x, center_y, width, height], dim=1)
            
            # Normalize so that bboxes have range 0,1
            bboxes_normalized = bboxes / torch.tensor([img_w, img_h, img_w, img_h], device=device, dtype=dtype)
            
            # For masks with no content (sum <= 1e-4), set to zero box
            # default_box = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
            # bboxes_normalized = torch.where(
            #     has_content.view(-1, 1).expand(-1, 4),
            #     bboxes_normalized,
            #     default_box.unsqueeze(0).expand(B, -1)
            # )
            
            return bboxes_normalized
        
        def get_bbox_from_mask(masks):
            """
            Extract bounding boxes from binary masks.
            
            Args:
                masks: torch.Tensor of shape [M, height, width] with binary masks
                
            Returns:
                bboxes: torch.Tensor of shape [M, 4] where each row is [center_x, center_y, width, height]
            """
            # masks_to_boxes returns [x_min, y_min, x_max, y_max]
            xyxy_boxes = box_ops.masks_to_boxes(masks)
            
            # Convert to [center_x, center_y, width, height]
            x_min, y_min, x_max, y_max = xyxy_boxes.unbind(dim=1)
            
            width = x_max - x_min
            height = y_max - y_min
            center_x = x_min + width / 2.0
            center_y = y_min + height / 2.0
            
            bboxes = torch.stack([center_x, center_y, width, height], dim=1)
            # Normalize so that bboxes have range 0,1
            bboxes_normalized = bboxes / torch.tensor([img_w, img_h, img_w, img_h], device=bboxes.device, dtype=bboxes.dtype)
            
            return bboxes_normalized
        
        # Interpolate masks to full image size
        src_masks_up = F.interpolate(
            src_masks.unsqueeze(1),
            size=(img_h, img_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        # Apply sigmoid to get soft masks
        src_masks_soft = src_masks_up.sigmoid()
        
        # Get boxes from soft masks (fully differentiable)
        src_masks_bboxes = get_bbox_from_soft_mask(src_masks_soft)
        
        # Compute L1 loss
        loss_bbox = F.l1_loss(src_masks_bboxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox_consistency'] = loss_bbox.sum() / num_boxes
        
        # Compute GIoU loss
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_masks_bboxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        
        losses['loss_giou_consistency'] = loss_giou.sum() / num_boxes
        
        return losses
    
    # def apply_bbox_masking(
    #     self, 
    #     src_masks_sigmoid, 
    #     src_boxes, 
    #     target_masks=None, 
    #     save_binary_masks=True, 
    #     output_dir="output/assets/bbox_masked_predictions",
    #     current_epoch=None,
    #     batch_size=None,  # Batch içindeki resim sayısı
    # ):
    #     """
    #     Apply bounding box masking to predicted masks.
        
    #     Args:
    #         src_masks_sigmoid: Sigmoid-activated masks of shape [N, H, W] - N = toplam obje sayısı (tüm batch'teki)
    #         src_boxes: Bounding boxes in xyxy format of shape [N, 4]
    #         target_masks: Target masks of shape [N, H, W] for comparison
        
    #     Returns:
    #         Masked predictions where only regions inside bboxes are kept
    #     """
    #     import os
    #     from PIL import Image
    #     import numpy as np

    #     device = src_masks_sigmoid.device
    #     N, H, W = src_masks_sigmoid.shape  # N = batch'teki toplam obje sayısı

    #     if save_binary_masks:
    #         os.makedirs(output_dir, exist_ok=True)
        
    #     # Clamp bounding boxes to valid image boundaries
    #     src_boxes_clamped = src_boxes.clone()
    #     src_boxes_clamped[:, 0] = torch.clamp(src_boxes_clamped[:, 0], min=0, max=W-1)  # x1
    #     src_boxes_clamped[:, 1] = torch.clamp(src_boxes_clamped[:, 1], min=0, max=H-1)  # y1
    #     src_boxes_clamped[:, 2] = torch.clamp(src_boxes_clamped[:, 2], min=0, max=W-1)  # x2
    #     src_boxes_clamped[:, 3] = torch.clamp(src_boxes_clamped[:, 3], min=0, max=H-1)  # y2
        
    #     # Convert to integer coordinates
    #     src_boxes_int = src_boxes_clamped.round().long()
        
    #     # Create binary masks for each bounding box
    #     bbox_masks = torch.zeros_like(src_masks_sigmoid, device=device)
        
    #     for i in range(N):
    #         x1, y1, x2, y2 = src_boxes_int[i]
    #         # Ensure x2 >= x1 and y2 >= y1 (in case of invalid boxes)
    #         x1, x2 = min(x1, x2), max(x1, x2)
    #         y1, y2 = min(y1, y2), max(y1, y2)
            
    #         # Set the bounding box region to 1
    #         bbox_masks[i, y1:y2+1, x1:x2+1] = 1.0
        
    #     # Apply the bbox mask to the predicted masks
    #     masked_predictions = src_masks_sigmoid * bbox_masks

    #     if save_binary_masks and target_masks is not None and current_epoch is not None:
    #         if current_epoch % 10 == 0:
    #             target_indices = [0, 200, 400, 600, 800, 1000, 1200, 1400]
                
    #             # Her obje için işlem yap ama counter'ı sadece batch sonunda artır
    #             for i in range(N):
    #                 # Mevcut epoch içindeki resim indeksini kontrol et
    #                 if self.epoch_image_counter in target_indices:
    #                     # Prepare target mask (top half)
    #                     target_mask_np = (target_masks[i].detach().cpu().numpy() * 255).astype(np.uint8)
    #                     target_rgb = np.stack([target_mask_np, target_mask_np, target_mask_np], axis=-1)
                        
    #                     # Prepare masked prediction (bottom half)
    #                     masked_pred_np = (masked_predictions[i].detach().cpu().numpy() * 255).astype(np.uint8)
    #                     pred_rgb = np.stack([masked_pred_np, masked_pred_np, masked_pred_np], axis=-1)
                        
    #                     # Draw green bounding box on prediction
    #                     x1, y1, x2, y2 = src_boxes_int[i]
    #                     x1, x2 = min(x1, x2), max(x1, x2)
    #                     y1, y2 = min(y1, y2), max(y1, y2)
                        
    #                     # Draw box outline in green (RGB: 0, 255, 0)
    #                     box_thickness = 2
    #                     # Top and bottom lines
    #                     pred_rgb[y1:y1+box_thickness, x1:x2+1, :] = [0, 255, 0]  # Top
    #                     pred_rgb[y2-box_thickness+1:y2+1, x1:x2+1, :] = [0, 255, 0]  # Bottom
    #                     # Left and right lines  
    #                     pred_rgb[y1:y2+1, x1:x1+box_thickness, :] = [0, 255, 0]  # Left
    #                     pred_rgb[y1:y2+1, x2-box_thickness+1:x2+1, :] = [0, 255, 0]  # Right
                        
    #                     # Create combined image: target on top, prediction on bottom
    #                     combined_height = H * 2
    #                     combined_img = np.zeros((combined_height, W, 3), dtype=np.uint8)
                        
    #                     # Top half: target mask
    #                     combined_img[0:H, :, :] = target_rgb
                        
    #                     # Bottom half: masked prediction with green box
    #                     combined_img[H:combined_height, :, :] = pred_rgb
                        
    #                     # Add a white separator line between the two masks
    #                     separator_thickness = 2
    #                     combined_img[H-separator_thickness:H+separator_thickness, :, :] = [255, 255, 255]
                        
    #                     # Create epoch-specific directory
    #                     epoch_dir = os.path.join(output_dir, f"epoch_{current_epoch:03d}")
    #                     img_dir = os.path.join(epoch_dir, f"img_{self.epoch_image_counter:06d}")
    #                     os.makedirs(img_dir, exist_ok=True)

    #                     # Save with object index and counter as filename
    #                     filename = f"obj_{i:03d}.png"
    #                     save_path = os.path.join(img_dir, filename)

    #                     # Save the image
    #                     img = Image.fromarray(combined_img, mode='RGB')
    #                     img.save(save_path)

    #                     print(f"Saved comparison image: {save_path}")
                
    #             # ÖNEMLI: Counter'ı batch size kadar artır (resim sayısı)
    #             batch_size_to_add = batch_size if batch_size is not None else 1
    #             self.epoch_image_counter += batch_size_to_add
                
    #         else:
    #             # Kaydetmediğimiz epoch'larda da counter'ı artırmamız gerekiyor
    #             batch_size_to_add = batch_size if batch_size is not None else 1
    #             self.epoch_image_counter += batch_size_to_add

    #     return masked_predictions

    # NOTE: this is the new version of apply_bbox_masking that saves images with green boxes - original
    def apply_bbox_masking(self, src_masks_sigmoid, src_boxes, target_masks=None, save_binary_masks=True, output_dir="output/debug_masks", current_epoch=None, batch_size=None):
        """
        Apply bounding box masking to predicted masks.
        
        Args:
            src_masks_sigmoid: Sigmoid-activated masks of shape [N, H, W]
            src_boxes: Bounding boxes in xyxy format of shape [N, 4]
            target_masks: Target masks of shape [N, H, W] for comparison
        
        Returns:
            Masked predictions where only regions inside bboxes are kept
        """
        import os
        from PIL import Image
        import numpy as np


        device = src_masks_sigmoid.device
        N, H, W = src_masks_sigmoid.shape

        if save_binary_masks:
            os.makedirs(output_dir, exist_ok=True)
        
        # Clamp bounding boxes to valid image boundaries
        src_boxes_clamped = src_boxes.clone()
        src_boxes_clamped[:, 0] = torch.clamp(src_boxes_clamped[:, 0], min=0, max=W-1)  # x1
        src_boxes_clamped[:, 1] = torch.clamp(src_boxes_clamped[:, 1], min=0, max=H-1)  # y1
        src_boxes_clamped[:, 2] = torch.clamp(src_boxes_clamped[:, 2], min=0, max=W-1)  # x2
        src_boxes_clamped[:, 3] = torch.clamp(src_boxes_clamped[:, 3], min=0, max=H-1)  # y2
        
        # Convert to integer coordinates
        src_boxes_int = src_boxes_clamped.round().long()
        
        # Create binary masks for each bounding box
        bbox_masks = torch.zeros_like(src_masks_sigmoid, device=device)
        
        for i in range(N):
            x1, y1, x2, y2 = src_boxes_int[i]
            # Ensure x2 >= x1 and y2 >= y1 (in case of invalid boxes)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Set the bounding box region to 1
            bbox_masks[i, y1:y2+1, x1:x2+1] = 1.0
        
        # Apply the bbox mask to the predicted masks
        masked_predictions = src_masks_sigmoid * bbox_masks

        if save_binary_masks and target_masks is not None:
            for i in range(N):
                if i == 0: #Save only the 0th image for checking
                    # Prepare target mask (top half)
                    target_mask_np = (target_masks[i].detach().cpu().numpy() * 255).astype(np.uint8)
                    target_rgb = np.stack([target_mask_np, target_mask_np, target_mask_np], axis=-1)
                    
                    # Prepare masked prediction (bottom half)
                    masked_pred_np = (masked_predictions[i].detach().cpu().numpy() * 255).astype(np.uint8)
                    pred_rgb = np.stack([masked_pred_np, masked_pred_np, masked_pred_np], axis=-1)
                    
                    # Draw green bounding box on prediction
                    x1, y1, x2, y2 = src_boxes_int[i]
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    # Draw box outline in green (RGB: 0, 255, 0)
                    box_thickness = 2
                    # Top and bottom lines
                    pred_rgb[y1:y1+box_thickness, x1:x2+1, :] = [0, 255, 0]  # Top
                    pred_rgb[y2-box_thickness+1:y2+1, x1:x2+1, :] = [0, 255, 0]  # Bottom
                    # Left and right lines  
                    pred_rgb[y1:y2+1, x1:x1+box_thickness, :] = [0, 255, 0]  # Left
                    pred_rgb[y1:y2+1, x2-box_thickness+1:x2+1, :] = [0, 255, 0]  # Right
                    
                    # Create combined image: target on top, prediction on bottom
                    combined_height = H * 2
                    combined_img = np.zeros((combined_height, W, 3), dtype=np.uint8)
                    
                    # Top half: target mask
                    combined_img[0:H, :, :] = target_rgb
                    
                    # Bottom half: masked prediction with green box
                    combined_img[H:combined_height, :, :] = pred_rgb
                    
                    # Add a white separator line between the two masks
                    separator_thickness = 2
                    combined_img[H-separator_thickness:H+separator_thickness, :, :] = [255, 255, 255]
                    
                    # Save combined image
                    img = Image.fromarray(combined_img, mode='RGB')
                    filename = f"comparison_{i:03d}_{current_epoch}.png"
                    img.save(os.path.join(output_dir, filename))
                    print(f"Saved comparison image: {os.path.join(output_dir, filename)}")
        
        return masked_predictions

    def reset_epoch_counter(self):
        """Reset the epoch image counter to zero."""
        self.epoch_image_counter = 0

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]) #Gives the indices belongs to witch batch
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
   
    def filter_boxes_by_mask_coverage(self, src_boxes, binary_mask, unmatched_idx, batch, threshold=0.5):
        # Remove batch dimension from binary_mask (shape: 1, H, W)
        binary_mask = binary_mask.squeeze(0)  # Shape: (576, 768)
        
        # Convert bounding box coordinates to integers before computing areas
        src_boxes = src_boxes.int() 

        # Compute areas of the bounding boxes
        box_areas = (src_boxes[:, 2] - src_boxes[:, 0]) * (src_boxes[:, 3] - src_boxes[:, 1])  # (487,)
        

        # Initialize a list to store valid indices
        valid_indices = []

        # Iterate over all bounding boxes
        for idx, (xmin, ymin, xmax, ymax) in enumerate(src_boxes):

            # Extract the region from the binary mask
            mask_region = binary_mask[ymin:ymax, xmin:xmax]

            # Count the number of True pixels in the mask region
            mask_count = mask_region.sum().item()

            # Compute the coverage ratio
            coverage_ratio = mask_count / box_areas[idx]

            # Check if the coverage exceeds the threshold
            if coverage_ratio > threshold:
                valid_indices.append(unmatched_idx[batch][idx])
                #valid_indices.append(idx)

        return torch.tensor(valid_indices, dtype=torch.long, device=src_boxes.device)
 

    def filter_boxes_by_mask_coverage_with_where(self, src_boxes, binary_mask, threshold=0.5):
        binary_mask = binary_mask.squeeze(0)  # Shape: (H, W)
        src_boxes = src_boxes.int()  # Convert to integer

        xmin, ymin, xmax, ymax = src_boxes[:, 0], src_boxes[:, 1], src_boxes[:, 2], src_boxes[:, 3]
        box_areas = (xmax - xmin) * (ymax - ymin)  # Compute areas

        height, width = binary_mask.shape
        y_grid, x_grid = torch.meshgrid(torch.arange(height, device=src_boxes.device),
                                        torch.arange(width, device=src_boxes.device),
                                        indexing="ij")

        valid_indices = []

        for idx in range(len(src_boxes)):
            x_mask = (x_grid >= xmin[idx]) & (x_grid < xmax[idx])
            y_mask = (y_grid >= ymin[idx]) & (y_grid < ymax[idx])

            # Create a boolean mask for the bounding box
            box_mask = x_mask & y_mask  # Shape: (H, W)

            # Count overlapping pixels
            mask_count = (box_mask & binary_mask).sum().item()

            # Compute the coverage ratio
            coverage_ratio = mask_count / box_areas[idx]

            # Store index if coverage is greater than threshold
            if coverage_ratio > threshold:
                valid_indices.append(idx)

        return torch.tensor(valid_indices, dtype=torch.long, device=src_boxes.device)



    def filter_boxes_by_mask_coverage_vectorized(self, src_boxes, binary_mask, unmatched_idx, batch, threshold=0.5):
        # Remove batch dimension from binary_mask (shape: 1, H, W)
        binary_mask = binary_mask.squeeze(0)  # Shape: (576, 768)

        # Ensure bounding box coordinates are in integer format
        src_boxes = src_boxes.int()

        # Extract box coordinates
        xmin, ymin, xmax, ymax = src_boxes[:, 0], src_boxes[:, 1], src_boxes[:, 2], src_boxes[:, 3]

        # Compute bounding box areas
        box_areas = (xmax - xmin) * (ymax - ymin)  # Shape: (487,)

        # Create a coordinate grid for all pixels
        height, width = binary_mask.shape  # (576, 768)
        y_grid, x_grid = torch.meshgrid(torch.arange(height, device=src_boxes.device),
                                        torch.arange(width, device=src_boxes.device),
                                        indexing="ij")

        # Expand coordinates to shape (487, H, W) for broadcasting
        x_grid = x_grid.unsqueeze(0)  # Shape: (1, H, W)
        y_grid = y_grid.unsqueeze(0)  # Shape: (1, H, W)

        # Create masks for each bounding box
        inside_x = (x_grid >= xmin[:, None, None]) & (x_grid < xmax[:, None, None])
        inside_y = (y_grid >= ymin[:, None, None]) & (y_grid < ymax[:, None, None])

        # Get the full bounding box mask: shape (487, H, W)
        box_masks = inside_x & inside_y

        # Count number of True pixels inside each bounding box
        mask_pixels_in_box = (box_masks & binary_mask).sum(dim=(1, 2))

        # Compute the coverage ratio
        coverage_ratios = mask_pixels_in_box / box_areas

        # Select indices where coverage exceeds the threshold
        selected_indices = torch.tensor(unmatched_idx[batch], device= src_boxes.device)[torch.where(coverage_ratios > threshold)[0]]

        return selected_indices

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'mask_box_consistency_loss': self.mask_box_consistency_loss
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, pre_outputs=None, pre_targets=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
#         if pre_outputs is None:
#             outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
#             # Retrieve the matching between the outputs of the last layer and the targets
#             indices = self.matcher(outputs_without_aux, targets)
#         else:
#             outputs_without_aux = {k: v for k, v in pre_outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
#             # Retrieve the matching between the outputs of the last layer and the targets
#             indices = self.matcher(outputs_without_aux, pre_targets)
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
            
#         pre_indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
#                 if pre_outputs is not None:
#                     indices = pre_indices
#                 else:
#                     indices = self.matcher(aux_outputs, targets)
                indices = self.matcher(aux_outputs, targets)

                for loss in self.losses:
                    if loss == 'masks' or loss == 'mask_box_consistency_loss':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks' or loss == 'mask_box_consistency_loss':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        
#         topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
#         scores = topk_values
#         topk_boxes = topk_indexes // out_logits.shape[2]
#         labels = topk_indexes % out_logits.shape[2]
#         boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
#         boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        scores, labels = prob[..., 1:2].max(-1)
        labels = labels + 1
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.dataset_file == 'coco':
        num_classes = 20
    elif args.dataset_file == 'mot':
        num_classes = 20
    elif args.dataset_file == "coco_panoptic":
        num_classes = 250
    elif args.dataset_file == "burst":
        num_classes = 20 # NOTE: max label id + 1 can work to
    else:
        num_classes = 20 
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        #model = DETRsegm(model, freeze_detr=True) # NOTE for continue on training for specifying model on --resume
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
        ##NOTE: Şuan için box loss undaki katsayılar kullanılıyor, sonradan ayrı katsayılar tanımlanabilir.
        weight_dict["loss_bbox_consistency"] = args.bbox_loss_coef 
        weight_dict["loss_giou_consistency"] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
        losses += ["mask_box_consistency_loss"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, args_=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, matcher
