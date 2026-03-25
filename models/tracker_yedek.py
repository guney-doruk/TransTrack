# """
# Copyright (c) https://github.com/xingyizhou/CenterTrack
# Modified by Peize Sun, Rufeng Zhang
# """
# # coding: utf-8
# import torch
# from scipy.optimize import linear_sum_assignment
# from util import box_ops
# from scripts.utils import mask2bbox
# import copy

# class Tracker(object):
#     def __init__(self, score_thresh, max_age=32):        
#         self.score_thresh = score_thresh
#         self.max_age = max_age        
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

#         mask_bboxes = []

#         scores = results["scores"]
#         classes = results["labels"]
#         bboxes = results["boxes"]  # x1y1x2y2
#         mask_bboxes = mask2bbox(results["masks"], bboxes, mask_bboxes=mask_bboxes, scores=scores, threshold=self.score_thresh)

#         ret = list()
#         ret_dict = dict()
#         for idx in range(scores.shape[0]):
#             if scores[idx] >= self.score_thresh:
#                 self.id_count += 1
#                 obj = dict()
#                 obj["score"] = float(scores[idx])
#                 obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
#                 obj["mask_bbox"] = mask_bboxes[idx]
#                 obj["tracking_id"] = self.id_count
# #                 obj['vxvy'] = [0.0, 0.0]
#                 obj['active'] = 1
#                 obj['age'] = 1
#                 ret.append(obj)
#                 ret_dict[idx] = obj
        
#         self.tracks = ret
#         self.tracks_dict = ret_dict
#         return copy.deepcopy(ret)

    
#     def step(self, output_results, frame_id):
#         mask_bboxes = []
#         track_mask_bboxes = []

#         scores = output_results["scores"]
#         track_scores = output_results["track_scores"]
#         classes = output_results["labels"]
#         bboxes = output_results["boxes"]  # x1y1x2y2
#         track_bboxes = output_results["track_boxes"] if "track_boxes" in output_results else None # x1y1x2y2
#         mask_bboxes = mask2bbox(output_results["masks"], bboxes, mask_bboxes=mask_bboxes, scores=scores, threshold=self.score_thresh) # x1y1x2y2
#         print(f"Frame_id: {frame_id}")
#         track_mask_bboxes = mask2bbox(output_results["track_masks"], track_bboxes, mask_bboxes=track_mask_bboxes, scores=track_scores, threshold=self.score_thresh, track=True) if "track_masks" in output_results else None # x1y1x2y2
        
#         results = list()
#         results_dict = dict()

#         tracks = list()
        
#         for idx in range(scores.shape[0]):
#             if idx in self.tracks_dict and track_bboxes is not None and track_mask_bboxes is not None:#Trackden yeni obje gelebiliyor ama bu şekilde sadece onceki objelerı bulması sağlanıyor.
#                 self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()
#                 self.tracks_dict[idx]["mask_bbox"] = track_mask_bboxes[idx]#Yukarıdakiyle aynı format olması lazım

#             if scores[idx] >= self.score_thresh:
#                 obj = dict()
#                 obj["score"] = float(scores[idx])
#                 obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist() 
#                 obj["mask_bbox"] = mask_bboxes[idx]#Eklendi              
#                 results.append(obj)        
#                 results_dict[idx] = obj
        
#         tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
#         N = len(results)
#         M = len(tracks)
        
#         ret = list()
#         unmatched_tracks = [t for t in range(M)]
#         unmatched_dets = [d for d in range(N)]
#         if N > 0 and M > 0:
#             det_box   = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4        
#             track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4                
#             cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box) # N x M
            
#             #Eklendi
#             det_mask_box = torch.stack([torch.tensor(obj['mask_bbox']) for obj in results], dim=0)  # N x 4 
#             track_mask_box = torch.stack([torch.tensor(obj['mask_bbox']) for obj in tracks], dim=0) # M x 4 #Doğru aldığına emin ol
#             cost_mask_bbox = 1.0 - box_ops.generalized_box_iou(det_mask_box, track_mask_box) # N x M

#             matched_indices = linear_sum_assignment(cost_bbox)
#             matched_indices_mask = linear_sum_assignment(cost_mask_bbox) # eklendi

#             # unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])]
#             # unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]

#             unmatched_dets, unmatched_tracks = [], []
#             for d in range(N):
#                 if not (d in matched_indices[0]) and (d in matched_indices_mask[0]):
#                     unmatched_dets.append(d)

#             for t in range(M):
#                 if not (t in matched_indices[1]) and (t in matched_indices_mask[1]):
#                     unmatched_tracks.append(t)
            
#             # Step 1: Convert to sets of pairs
#             matches_set = set(zip(matched_indices[0], matched_indices[1]))
#             matches_mask_set = set(zip(matched_indices_mask[0], matched_indices_mask[1]))

#             # Step 2: Find differences
#             only_in_bbox = matches_set - matches_mask_set
#             only_in_mask = matches_mask_set - matches_set
            
#             def process_mismatches(only_in_bbox, only_in_mask):
#                 mismatch_dict = {}
#                 if only_in_bbox or only_in_mask:
#                     for box in only_in_bbox:
#                         key = str(box[0])
#                         value = box[1]
#                         if key not in mismatch_dict:
#                             mismatch_dict[key] = []
#                         mismatch_dict[key].append(value)
                    
#                     for mask in only_in_mask:
#                         key = str(mask[0])
#                         value = mask[1]
#                         if key not in mismatch_dict:
#                             mismatch_dict[key] = []
#                         mismatch_dict[key].append(value)
#                 return mismatch_dict
            
#             # Step 3: Write differences to txt
#             with open('differences_log_utils_debug.txt', 'a') as f:
#                 if only_in_bbox:
#                     f.write("Matches only in bbox matching:\n")
#                     for match in only_in_bbox:
#                         f.write(f"{match}\n")
#                 if only_in_mask:
#                     f.write("Matches only in mask matching:\n")
#                     for match in only_in_mask:
#                         f.write(f"{match}\n")
#                 if not only_in_bbox and not only_in_mask:
#                     f.write("No differences found.\n")
#                 f.write("--------------------------------\n")

#             matches = [[],[]]
#             for m0, m1, m0_mask, m1_mask in zip(matched_indices[0], matched_indices[1], matched_indices_mask[0], matched_indices_mask[1]):
#                 current_mismatches = process_mismatches(only_in_bbox, only_in_mask)
#                 # current_mismatches = {'6': [10, 3], '5': [2, 10]}
                
#                 if cost_bbox[m0, m1] > 1.2 and cost_mask_bbox[m0_mask, m1_mask] > 1.2:
#                     if current_mismatches:
#                         matched_pairs = []
#                         used_tracks = set()
                        
#                         for det, tracks in current_mismatches.items():
#                             det_int = int(det)
#                             track1, track2 = map(int, tracks)
                            
#                             cost1 = cost_bbox[det_int, track1]
#                             cost2 = cost_mask_bbox[det_int, track2]
                            
#                             if cost1 <= cost2:
#                                 selected_track = track1
#                                 alternative_track = track2
#                             else:
#                                 selected_track = track2
#                                 alternative_track = track1
                            
#                             if selected_track not in used_tracks:
#                                 matched_pairs.append((det_int, selected_track))
#                                 used_tracks.add(selected_track)
#                                 # matches[0].append(det_int)
#                                 # matches[1].append(selected_track)
#                             else:
#                                 if alternative_track not in used_tracks:
#                                     matched_pairs.append((det_int, alternative_track))
#                                     used_tracks.add(alternative_track)
#                                     # matches[0].append(det_int)
#                                     # matches[1].append(alternative_track)
                        
#                         all_tracks = set()
#                         for tracks in current_mismatches.values():
#                             all_tracks.update(map(int, tracks))
#                         unmatched_tracks.extend(list(all_tracks - used_tracks))
#                     else:
#                         if cost_bbox[m0, m1] > cost_mask_bbox[m0_mask, m1_mask]:
#                             unmatched_dets.append(m0_mask)
#                             unmatched_tracks.append(m1_mask)
#                         else:
#                             unmatched_dets.append(m0)
#                             unmatched_tracks.append(m1)
#                 else:
#                     if cost_bbox[m0, m1] > cost_mask_bbox[m0_mask, m1_mask]:
#                         matches[0].append(m0_mask)
#                         matches[1].append(m1_mask)
#                     else:
#                         matches[0].append(m0)
#                         matches[1].append(m1)

#             #Buradan sonrası kaldı.Buradan itibaren devam
#             #Bence bitti sadece bir kontrol yapılacak.
#             for (m0, m1) in zip(matches[0], matches[1]):
#                 track = results[m0]
#                 track['tracking_id'] = tracks[m1]['tracking_id']
#                 track['age'] = 1
#                 track['active'] = 1
#                 pre_box = tracks[m1]['bbox']
#                 cur_box = track['bbox']
#                 ret.append(track)

#         for i in unmatched_dets:
#             track = results[i]
#             self.id_count += 1
#             track['tracking_id'] = self.id_count
#             track['age'] = 1
#             track['active'] =  1
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


"""
Copyright (c) https://github.com/xingyizhou/CenterTrack
Modified by Peize Sun, Rufeng Zhang
"""
# coding: utf-8
import torch
from scipy.optimize import linear_sum_assignment
from util import box_ops
from scripts.utils import mask2bbox
import copy

class Tracker(object):
    def __init__(self, score_thresh, max_age=32):        
        self.score_thresh = score_thresh
        self.max_age = max_age        
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
        mask_bboxes = []

        scores = results["scores"]
        classes = results["labels"]
        bboxes = results["boxes"]  # x1y1x2y2
        mask_bboxes = mask2bbox(results["masks"], bboxes, mask_bboxes=mask_bboxes, scores=scores, threshold=self.score_thresh)

        ret = list()
        ret_dict = dict()
        for idx in range(scores.shape[0]):
            if scores[idx] >= self.score_thresh:
                self.id_count += 1
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
                obj["mask_bbox"] = mask_bboxes[idx]
                obj["tracking_id"] = self.id_count
#                 obj['vxvy'] = [0.0, 0.0]
                obj['active'] = 1
                obj['age'] = 1
                ret.append(obj)
                ret_dict[idx] = obj
        
        self.tracks = ret
        self.tracks_dict = ret_dict
        return copy.deepcopy(ret)

    
    def step(self, output_results, frame_id):
        mask_bboxes = []
        track_mask_bboxes = []

        scores = output_results["scores"]
        track_scores = output_results["track_scores"]
        classes = output_results["labels"]
        bboxes = output_results["boxes"]  # x1y1x2y2
        track_bboxes = output_results["track_boxes"] if "track_boxes" in output_results else None # x1y1x2y2
        mask_bboxes = mask2bbox(output_results["masks"], bboxes, mask_bboxes=mask_bboxes, scores=scores, threshold=self.score_thresh) # x1y1x2y2
        print(f"Frame_id: {frame_id}")
        track_mask_bboxes = mask2bbox(output_results["track_masks"], track_bboxes, mask_bboxes=track_mask_bboxes, scores=track_scores, threshold=self.score_thresh, track=True) if "track_masks" in output_results else None # x1y1x2y2
        
        results = list()
        results_dict = dict()

        tracks = list()
        
        for idx in range(scores.shape[0]):
            if idx in self.tracks_dict and track_bboxes is not None and track_mask_bboxes is not None:#Trackden yeni obje gelebiliyor ama bu şekilde sadece onceki objelerı bulması sağlanıyor.
                self.tracks_dict[idx]["bbox"] = track_bboxes[idx, :].cpu().numpy().tolist()
                self.tracks_dict[idx]["mask_bbox"] = track_mask_bboxes[idx]#Yukarıdakiyle aynı format olması lazım

            if scores[idx] >= self.score_thresh:
                obj = dict()
                obj["score"] = float(scores[idx])
                obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist() 
                obj["mask_bbox"] = mask_bboxes[idx]#Eklendi              
                results.append(obj)        
                results_dict[idx] = obj
        
        tracks = [v for v in self.tracks_dict.values()] + self.unmatched_tracks
        N = len(results)
        M = len(tracks)
        
        ret = list()
        unmatched_tracks = [t for t in range(M)]
        unmatched_dets = [d for d in range(N)]
        if N > 0 and M > 0:
            det_box   = torch.stack([torch.tensor(obj['bbox']) for obj in results], dim=0) # N x 4        
            track_box = torch.stack([torch.tensor(obj['bbox']) for obj in tracks], dim=0) # M x 4                
            cost_bbox = 1.0 - box_ops.generalized_box_iou(det_box, track_box) # N x M
            
            #Eklendi
            det_mask_box = torch.stack([torch.tensor(obj['mask_bbox']) for obj in results], dim=0)  # N x 4 
            track_mask_box = torch.stack([torch.tensor(obj['mask_bbox']) for obj in tracks], dim=0) # M x 4 #Doğru aldığına emin ol
            cost_mask_bbox = 1.0 - box_ops.generalized_box_iou(det_mask_box, track_mask_box) # N x M

            matched_indices = linear_sum_assignment(cost_bbox)
            unmatched_dets = [d for d in range(N) if not (d in matched_indices[0])] 
            unmatched_tracks = [d for d in range(M) if not (d in matched_indices[1])]
            if torch.isnan(cost_mask_bbox).any():
                print("🔍 NaN found in cost_mask_bbox — scanning for source...")
                for i in range(det_mask_box.shape[0]):
                    for j in range(track_mask_box.shape[0]):
                        b1 = det_mask_box[i].unsqueeze(0)  # shape [1, 4]
                        b2 = track_mask_box[j].unsqueeze(0)  # shape [1, 4]
                        giou = box_ops.generalized_box_iou(b1, b2)
                        if torch.isnan(giou):
                            print(f"\n🔥 NaN GIoU at det_mask_box[{i}] and track_mask_box[{j}]")
                            print("det_mask_box:", b1)
                            print("track_mask_box:", b2)
                            area1 = (b1[0, 2] - b1[0, 0]) * (b1[0, 3] - b1[0, 1])
                            area2 = (b2[0, 2] - b2[0, 0]) * (b2[0, 3] - b2[0, 1])
                            print(f"area1: {area1.item()}, area2: {area2.item()}")

            print("NaN Check")
            print("NaNs:", torch.isnan(cost_mask_bbox).any().item())
            print("Infs:", torch.isinf(cost_mask_bbox).any().item())
            print("Min:", cost_mask_bbox.min().item(), "Max:", cost_mask_bbox.max().item())
            if torch.isnan(cost_mask_bbox).any().item() == True:
                print("Cost mask matrix", cost_mask_bbox)
            print("Cost matrix shape:", cost_mask_bbox.shape)
            #Eklendi
            matched_indices_mask = linear_sum_assignment(cost_mask_bbox)
            unmatched_dets_mask = [d for d in range(N) if not (d in matched_indices_mask[0])]
            unmatched_tracks_mask = [d for d in range(M) if not (d in matched_indices_mask[1])]

            # Step 1: Convert to sets of pairs
            matches_set = set(zip(matched_indices[0], matched_indices[1]))
            matches_mask_set = set(zip(matched_indices_mask[0], matched_indices_mask[1]))

            # Step 2: Find differences
            only_in_bbox = matches_set - matches_mask_set
            only_in_mask = matches_mask_set - matches_set

            # Step 3: Write differences to txt
            with open('differences_log_utils_debug.txt', 'a') as f:
                if only_in_bbox:
                    f.write("Matches only in bbox matching:\n")
                    for match in only_in_bbox:
                        f.write(f"{match}\n")
                if only_in_mask:
                    f.write("Matches only in mask matching:\n")
                    for match in only_in_mask:
                        f.write(f"{match}\n")
                if not only_in_bbox and not only_in_mask:
                    f.write("No differences found.\n")
                f.write("--------------------------------\n")

            matches = [[],[]]
            #Eklendi
            matches_mask = [[],[]]
            #Burada şöyle bir sıkınıt var class tarafı ortak olduğu için segmentation tarafıyla bbox tarafı her zaman aynı objelri bulacak. Bu duurmda box ın bulamadığı ama segmentasyonun bulduğu bir obje olmayacak. Zaten confidence skorlarıda aynı olduğu için düşük confidencelarıda katamayacaz. Bunları katmak için ByteTrack ı entegre edebiliriz.
            #Aynı sayıda bulmadığı takdirde implementasyonu aşırı zorlaşıyor ve bu durumda aşağıdaki ekler işe yaramıyor.
            #for (m0, m1),(m0_mask, m1_mask) in zip(matched_indices[0], matched_indices[1]):
            for m0, m1, m0_mask, m1_mask in zip(matched_indices[0], matched_indices[1], matched_indices_mask[0], matched_indices_mask[1]):
                if cost_bbox[m0, m1] > 1.2 and cost_mask_bbox[m0_mask, m1_mask] > 1.2:
                    #Eklendi
                    #Amaç unmatchedlere bile aralarından en düşük costa sahip olanı vermek ki sonrasında hem det hem de deaktif tracklar için doğrusunu alalım
                    if cost_bbox[m0, m1] > cost_mask_bbox[m0_mask, m1_mask]:
                        unmatched_dets.append(m0_mask)
                        unmatched_tracks.append(m1_mask)
                    else:
                        unmatched_dets.append(m0)
                        unmatched_tracks.append(m1)
                else:
                    #Eklendi
                    # Asıl mantığımızın eklendiği yer 
                    if cost_bbox[m0, m1] > cost_mask_bbox[m0_mask, m1_mask]:
                        matches[0].append(m0_mask)
                        matches[1].append(m1_mask)
                    else:
                        matches[0].append(m0)
                        matches[1].append(m1)

            #Buradan sonrası kaldı.Buradan itibaren devam
            #Bence bitti sadece bir kontrol yapılacak.
            for (m0, m1) in zip(matches[0], matches[1]):
                track = results[m0]
                track['tracking_id'] = tracks[m1]['tracking_id']
                track['age'] = 1
                track['active'] = 1
                pre_box = tracks[m1]['bbox']
                cur_box = track['bbox']
    #             pre_cx, pre_cy = (pre_box[0] + pre_box[2]) / 2, (pre_box[1] + pre_box[3]) / 2
    #             cur_cx, cur_cy = (cur_box[0] + cur_box[2]) / 2, (cur_box[1] + cur_box[3]) / 2
    #             track['vxvy'] = [cur_cx - pre_cx, cur_cy - pre_cy]
                ret.append(track)

        for i in unmatched_dets:
            track = results[i]
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] =  1
#             track['vxvy'] = [0.0, 0.0]
            ret.append(track)
        
        ret_unmatched_tracks = []
        for i in unmatched_tracks:
            track = tracks[i]
            if track['age'] < self.max_age:#Doruk için çok geç
                track['age'] += 1
                track['active'] = 0
#                 x1, y1, x2, y2 = track['bbox']
#                 vx, vy = track['vxvy']
#                 track['bbox'] = [x1+vx, y1+vy, x2+vx, y2+vy]
                ret.append(track)
                ret_unmatched_tracks.append(track)
    
        self.tracks = ret
        self.tracks_dict = results_dict
        self.unmatched_tracks = ret_unmatched_tracks
        return copy.deepcopy(ret)
