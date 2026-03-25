# import torch
# from scipy.optimize import linear_sum_assignment
# from util import box_ops
# import copy
# import torch.nn.functional as F

# def compute_mask_iou(mask1, mask2):
#     intersection = (mask1 * mask2).sum()
#     union = (mask1 + mask2 - mask1 * mask2).sum()
    
#     if union == 0:
#         return torch.tensor(0.0)
#     return intersection / union

# def compute_mask_iou_batch(masks1, masks2):
#     N, M = masks1.shape[0], masks2.shape[0]
#     iou_matrix = torch.zeros(N, M)
    
#     for i in range(N):
#         for j in range(M):
#             iou_matrix[i, j] = compute_mask_iou(masks1[i], masks2[j])
    
#     return iou_matrix

# class EnhancedTracker(object):
#     def __init__(self, score_thresh, max_age=32, bbox_weight=0.5, mask_weight=0.5):        
#         self.score_thresh = score_thresh
#         self.max_age = max_age
#         self.bbox_weight = bbox_weight
#         self.mask_weight = mask_weight
#         self.id_count = 0
#         self.tracks_dict = dict()
#         self.tracks = list()
#         self.unmatched_tracks = list()
#         self.reset_all()
        
#     def reset_all(self):
#         self.id_count = 0
#         self.tracks_dict = dict()
#         self.tracks = list()
#         self.unmatched_tracks = list()
    
#     def init_track(self, results):
#         scores = results["scores"]
#         classes = results["labels"]
#         bboxes = results["boxes"]  # x1y1x2y2
#         masks = results.get("masks", None)  # [N, H, W]
        
#         ret = list()
#         ret_dict = dict()
#         for idx in range(scores.shape[0]):
#             if scores[idx] >= self.score_thresh:
#                 self.id_count += 1
#                 obj = dict()
#                 obj["score"] = float(scores[idx])
#                 obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
#                 obj["tracking_id"] = self.id_count
#                 obj['active'] = 1
#                 obj['age'] = 1
                
#                 if masks is not None:
#                     obj["mask"] = masks[idx].cpu()
                
#                 ret.append(obj)
#                 ret_dict[idx] = obj
        
#         self.tracks = ret
#         self.tracks_dict = ret_dict
#         return copy.deepcopy(ret)

#     def compute_combined_cost(self, det_boxes, track_boxes, det_masks=None, track_masks=None):
#         bbox_iou = box_ops.generalized_box_iou(det_boxes, track_boxes)
#         cost_bbox = 1.0 - bbox_iou
        
#         if det_masks is not None and track_masks is not None:
#             mask_iou = compute_mask_iou_batch(det_masks, track_masks)
#             cost_mask = 1.0 - mask_iou
            
#             combined_cost = (self.bbox_weight * cost_bbox + 
#                            self.mask_weight * cost_mask)
#         else:
#             combined_cost = cost_bbox
            
#         return combined_cost
    
#     def step(self, output_results):
#         scores = output_results["scores"]
#         classes = output_results["labels"]
#         bboxes = output_results["boxes"]  # x1y1x2y2
#         track_bboxes = output_results.get("track_boxes", None) # x1y1x2y2
        
#         det_masks = output_results.get("masks", None)
#         track_masks = output_results.get("track_masks", None)
        
#         results = list()
#         results_dict = dict()
#         tracks = list()
        
#         for idx in range(scores.shape[0]):
#             if idx in self.tracks_dict and track_bboxes is not None:
#                 self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()

#                 if track_masks is not None:
#                     self.tracks_dict[idx]["mask"] = track_masks[idx].cpu()

#             if scores[idx] >= self.score_thresh:
#                 obj = dict()
#                 obj["score"] = float(scores[idx])
#                 obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                
#                 if det_masks is not None:
#                     obj["mask"] = det_masks[idx].cpu()
                    
#                 results.append(obj)        
#                 results_dict[idx] = obj
        
#         tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
#         N = len(results)
#         M = len(tracks)
        
#         ret = list()
#         unmatched_tracks = [t for t in range(M)]
#         unmatched_dets = [d for d in range(N)]
        
#         if N > 0 and M > 0:
#             det_box = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4        
#             track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4
            
#             det_mask_list = []
#             track_mask_list = []
            
#             masks_available = True
#             for obj in results:
#                 if "mask" in obj:
#                     det_mask_list.append(obj["mask"])
#                 else:
#                     masks_available = False
#                     break
            
#             if masks_available:
#                 for track in tracks:
#                     if "mask" in track:
#                         track_mask_list.append(track["mask"])
#                     else:
#                         masks_available = False
#                         break
            
#             if masks_available and len(det_mask_list) > 0 and len(track_mask_list) > 0:
#                 det_masks_tensor = torch.stack(det_mask_list, dim=0)
#                 track_masks_tensor = torch.stack(track_mask_list, dim=0)
#                 combined_cost = self.compute_combined_cost(
#                     det_box, track_box, det_masks_tensor, track_masks_tensor
#                 )
#             else:
#                 combined_cost = 1.0 - box_ops.generalized_box_iou(det_box, track_box)

#             matched_indices = linear_sum_assignment(combined_cost)
#             unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
#             unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

#             matches = [[],[]]
#             for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
#                 if combined_cost[m0, m1] > 1.2:
#                     unmatched_dets.append(m0)
#                     unmatched_tracks.append(m1)
#                 else:
#                     matches[0].append(m0)
#                     matches[1].append(m1)

#             for (m0, m1) in zip(matches[0], matches[1]):
#                 track = results[m0]
#                 track['tracking_id'] = tracks[m1]['tracking_id']
#                 track['age'] = 1
#                 track['active'] = 1
#                 ret.append(track)

#         for i in unmatched_dets:
#             track = results[i]
#             self.id_count += 1
#             track['tracking_id'] = self.id_count
#             track['age'] = 1
#             track['active'] = 1
#             ret.append(track)
        
#         ret_unmatched_tracks = []
#         for i in unmatched_tracks:
#             track = tracks[i]
#             if track['age'] < self.max_age:
#                 track['age'] += 1
#                 track['active'] = 0
#                 ret.append(track)
#                 ret_unmatched_tracks.append(track)
    
#         self.tracks = ret
#         self.tracks_dict = results_dict
#         self.unmatched_tracks = ret_unmatched_tracks
#         return copy.deepcopy(ret)

import torch, copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment
from util import box_ops

# def compute_mask_iou_batch(masks1, masks2):
#     """Optimized batch mask IoU computation with memory management"""
#     N, M = masks1.shape[0], masks2.shape[0]
#     device = masks1.device
    
#     with torch.no_grad():
#         masks1_flat = masks1.view(N, -1).float()  # [N, H*W]
#         masks2_flat = masks2.view(M, -1).float()  # [M, H*W]
        
#         intersection = torch.mm(masks1_flat, masks2_flat.t())  # [N, M]
        
#         area1 = masks1_flat.sum(dim=1, keepdim=True)  # [N, 1]
#         area2 = masks2_flat.sum(dim=1, keepdim=True)  # [M, 1]
        
#         union = area1 + area2.t() - intersection  # [N, M]
        
#         iou = intersection / (union + 1e-6)
#         iou = torch.where(union > 0, iou, torch.zeros_like(iou))
        
#         return iou

def compute_mask_iou_batch(masks1, masks2):
    """Optimized batch mask Generalized IoU computation following GIoU concept"""
    N, M = masks1.shape[0], masks2.shape[0]
    device = masks1.device
    H, W = masks1.shape[1], masks1.shape[2]
    
    with torch.no_grad():
        masks1_flat = masks1.view(N, -1).float()  # [N, H*W]
        masks2_flat = masks2.view(M, -1).float()  # [M, H*W]
        
        # Standard IoU computation
        intersection = torch.mm(masks1_flat, masks2_flat.t())  # [N, M]
        area1 = masks1_flat.sum(dim=1, keepdim=True)  # [N, 1]
        area2 = masks2_flat.sum(dim=1, keepdim=True)  # [M, 1]
        union = area1 + area2.t() - intersection  # [N, M]
        
        iou = intersection / (union + 1e-6)
        iou = torch.where(union > 0, iou, torch.zeros_like(iou))
        
        #Compute bounding boxes using torch.nonzero for proper indexing
        def get_bbox_from_mask(mask):
            """Get bounding box coordinates from a single 2D mask"""
            if mask.sum() > 0:
                # Find all non-zero positions
                nonzero_pos = torch.nonzero(mask, as_tuple=False)  # [num_points, 2] where each row is [y, x]
                
                # Get min/max coordinates
                y_coords = nonzero_pos[:, 1].float()  # y coordinates
                x_coords = nonzero_pos[:, 2].float()  # x coordinates
                
                y_min, y_max = y_coords.min(), y_coords.max()
                x_min, x_max = x_coords.min(), x_coords.max()
                
                return [x_min.item(), y_min.item(), x_max.item(), y_max.item()]
            else:
                # Handle empty masks
                return [0.0, 0.0, 1.0, 1.0]

        
        # Compute bounding boxes for masks1
        boxes1 = []
        for i in range(N):
            bbox = get_bbox_from_mask(masks1[i])
            boxes1.append(bbox)
        
        # Compute bounding boxes for masks2
        boxes2 = []
        for i in range(M):
            bbox = get_bbox_from_mask(masks2[i])
            boxes2.append(bbox)
        
        # Convert to tensors for vectorized computation
        boxes1_tensor = torch.tensor(boxes1, device=device, dtype=torch.float32)  # [N, 4]
        boxes2_tensor = torch.tensor(boxes2, device=device, dtype=torch.float32)  # [M, 4]

        # Compute enclosing box areas (C in GIoU formula)
        # lt = top-left corner of enclosing box
        lt = torch.min(boxes1_tensor[:, None, :2], boxes2_tensor[:, :2])  # [N, M, 2]
        # rb = bottom-right corner of enclosing box  
        rb = torch.max(boxes1_tensor[:, None, 2:], boxes2_tensor[:, 2:])   # [N, M, 2]
        
        wh = (rb - lt).clamp(min=0)  # [N, M, 2]
        enclosing_area = wh[:, :, 0] * wh[:, :, 1]  # [N, M] - This is C
        penalty = (enclosing_area - union) / (enclosing_area + 1e-6)

        # Generalized IoU = IoU - (C - Union) / C
        # Following the exact GIoU formula from the paper
        penalty_weight = torch.clamp(1.0 - iou, min=0.0, max=1.0)
        adjusted_penalty = penalty * penalty_weight
        generalized_iou = iou - adjusted_penalty
        
        return generalized_iou

class EnhancedTracker(object):
    def __init__(self, score_thresh, max_age=32, bbox_weight=0.7, mask_weight=0.3, unmatch_threshold=1.2, use_box_small=False, use_scaled_factor=False):        
        self.score_thresh = score_thresh
        self.max_age = max_age
        self.bbox_weight = bbox_weight
        self.mask_weight = mask_weight
        self.unmatch_threshold = unmatch_threshold
        self.use_box_small = use_box_small
        self.use_scaled_factor = use_scaled_factor
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
        self.reset_all()
        
    def reset_all(self):
        self.id_count = 0
        self.tracks_dict = dict()
        self.tracks = list()
        self.unmatched_tracks = list()
        
    def init_track(self, results):

        scores = results["scores"]
        classes = results["labels"] 
        bboxes = results["boxes"]
        masks = results.get("masks", None)
        
        ret = list()
        ret_dict = dict()
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                self.id_count += 1
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj["tracking_id"] = self.id_count
                obj['active'] = 1
                obj['age'] = 1
                
                if masks is not None:
                    obj["mask"] = masks[idx].cpu().numpy()
                
                ret.append(obj)
                ret_dict[idx] = obj
    
        self.tracks = ret
        self.tracks_dict = ret_dict
        return copy.deepcopy(ret)

    def compute_combined_cost(self, det_boxes, track_boxes, det_masks=None, track_masks=None):
        bbox_iou = box_ops.generalized_box_iou(det_boxes, track_boxes)
        cost_bbox = 1.0 - bbox_iou
        
        if det_masks is not None and track_masks is not None:
            mask_iou = compute_mask_iou_batch(det_masks, track_masks)
            cost_mask = 1.0 - mask_iou
            
            combined_cost = (self.bbox_weight * cost_bbox + self.mask_weight * cost_mask)
        else:
            combined_cost = cost_bbox
            
        return combined_cost

    def compute_combined_cost_box_small(self, det_boxes, track_boxes, det_masks=None, track_masks=None):
        """
        Compute combined cost with special handling for small objects.
        Small objects (area < 1024 pixels²) use 100% bbox cost.
        """
        print("small box combined cost is using")
        bbox_iou = box_ops.generalized_box_iou(det_boxes, track_boxes)
        cost_bbox = 1.0 - bbox_iou
        
        # boxes (width * height)
        det_widths = det_boxes[:, 2] - det_boxes[:, 0]  # x2 - x1
        det_heights = det_boxes[:, 3] - det_boxes[:, 1]  # y2 - y1
        det_areas = det_widths * det_heights  # [N]
        
        SMALL_OBJECT_THRESHOLD = 1024

        COCO_RESOLUTION = 640 * 480
        OUR_RESOLUTION = 1920 * 1080
        SCALE_FACTOR = OUR_RESOLUTION / COCO_RESOLUTION
        SCALED_SMALL_OBJECT_THRESHOLD = SMALL_OBJECT_THRESHOLD * SCALE_FACTOR

        if self.use_scaled_factor:
            is_small = det_areas < SCALED_SMALL_OBJECT_THRESHOLD
        else:
            is_small = det_areas < SMALL_OBJECT_THRESHOLD
        
        if det_masks is not None and track_masks is not None:
            mask_iou = compute_mask_iou_batch(det_masks, track_masks)
            cost_mask = 1.0 - mask_iou
            
            combined_cost = (self.bbox_weight * cost_bbox + self.mask_weight * cost_mask)
            
            is_small_expanded = is_small.unsqueeze(1).expand_as(combined_cost)
            combined_cost = torch.where(is_small_expanded, cost_bbox, combined_cost)
        else:
            combined_cost = cost_bbox
        
        return combined_cost
    
    def step(self, output_results):
        scores = output_results["scores"]
        classes = output_results["labels"]
        bboxes = output_results["boxes"]  # x1y1x2y2
        track_bboxes = output_results["track_boxes"] if "track_boxes" in output_results else None # x1y1x2y2
        
        det_masks = output_results.get("masks", None)
        track_masks = output_results.get("track_masks", None)
        
        results = list()
        results_dict = dict()

        tracks = list()
        
        for idx in range(scores.shape[0]):
            if idx in self.tracks_dict and track_bboxes is not None:
                self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()

                if track_masks is not None:
                    self.tracks_dict[idx]["mask"] = track_masks[idx].cpu().numpy()

            if scores[idx] >= self.score_thresh:
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                
                if det_masks is not None:
                    obj["mask"] = det_masks[idx].cpu().numpy()
                    
                results.append(obj)        
                results_dict[idx] = obj
        
        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        N = len(results)
        M = len(tracks)
        
        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]
        if N > 0 and M > 0:
            with torch.no_grad():
                device = bboxes.device if hasattr(bboxes, 'device') else torch.device('cpu')
                det_box   = torch.stack([torch.tensor(obj['bbox'], device=device) for obj in results], dim=0)
                track_box = torch.stack([torch.tensor(obj['bbox'], device=device) for obj in tracks], dim=0)
                
                det_mask_list = []
                track_mask_list = []
                
                masks_available = True
                for obj in results:
                    if "mask" in obj:
                        mask_tensor = torch.from_numpy(obj["mask"]).to(device)
                        det_mask_list.append(mask_tensor)
                    else:
                        masks_available = False
                        break
                
                if masks_available:
                    for track in tracks:
                        if "mask" in track:
                            mask_tensor = torch.from_numpy(track["mask"]).to(device)
                            track_mask_list.append(mask_tensor)
                        else:
                            masks_available = False
                            break
                
                if masks_available and len(det_mask_list) > 0 and len(track_mask_list) > 0:
                    det_masks_tensor = torch.stack(det_mask_list, dim=0)
                    track_masks_tensor = torch.stack(track_mask_list, dim=0)
                    if self.use_box_small: 
                        combined_cost = self.compute_combined_cost_box_small(
                            det_box, track_box, det_masks_tensor, track_masks_tensor
                        )
                    else:
                        combined_cost = self.compute_combined_cost(
                            det_box, track_box, det_masks_tensor, track_masks_tensor
                        )
                    
                else:
                    bbox_iou = box_ops.generalized_box_iou(det_box, track_box)
                    combined_cost = 1.0 - bbox_iou

                cpu_cost = combined_cost.cpu().numpy()
                matched_indices = linear_sum_assignment(cpu_cost)
                unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
                unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

                matches = [[],[]]
                for (m0, m1) in zip(matched_indices[0], matched_indices[1]):
                    if cpu_cost[m0, m1] > self.unmatch_threshold:
                        unmatched_dets.append(m0)
                        unmatched_tracks.append(m1)
                    else:
                        matches[0].append(m0)
                        matches[1].append(m1)

                for (m0, m1) in zip(matches[0], matches[1]):
                    track = results[m0]
                    track['tracking_id'] = tracks[m1]['tracking_id']
                    track['age'] = 1
                    track['active'] = 1
                    ret.append(track) 

        for i in unmatched_dets:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] = 1
            ret.append(track)
        
        ret_unmatched_tracks = []
        for i in unmatched_tracks:
            track = tracks[i]
            if track['age'] < self.max_age:
                track['age'] += 1
                track['active'] = 0
                ret.append(track)
                ret_unmatched_tracks.append(track)
    
        self.tracks = ret
        self.tracks_dict = results_dict
        self.unmatched_tracks = ret_unmatched_tracks
        return copy.deepcopy(ret)