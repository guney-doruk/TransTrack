import os, torch
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes
from PIL import ImageDraw
import util.misc as utils

def pred2mask_save(targets, pred_binary, target_binary, target_boxes, draw_pred_boxes=True):
    """
    Draws bboxes on prediction binary masks and saves including the post process.
    """
    image_id = targets[0]["image_id"].item()
    orig_h, orig_w = targets[0]["orig_size"].tolist()

    img_folder = os.path.join('/cta/users/grad4/master/TransTrack/output/seg_out', f"{image_id:012d}")
    os.makedirs(img_folder, exist_ok=True)

    for idx, (pred_mask, target_mask, target_box) in enumerate(zip(pred_binary, target_binary, target_boxes)):
        pred_resized = TF.resize(pred_mask.unsqueeze(0), size=[orig_h, orig_w], interpolation=TF.InterpolationMode.NEAREST)
        target_resized = TF.resize(target_mask.unsqueeze(0), size=[orig_h, orig_w], interpolation=TF.InterpolationMode.NEAREST)

        target_img = target_resized.squeeze(0)
        pred_img = pred_resized.squeeze(0)

        cx, cy, w, h = target_box
        cx *= orig_w
        cy *= orig_h
        w *= orig_w
        h *= orig_h

        x1_gt = int(cx - w / 2)
        y1_gt = int(cy - h / 2)
        x2_gt = int(cx + w / 2)
        y2_gt = int(cy + h / 2)

        x1_gt = max(0, x1_gt)
        y1_gt = max(0, y1_gt)
        x2_gt = min(orig_w, x2_gt)
        y2_gt = min(orig_h, y2_gt)

        mask_filtered = torch.zeros_like(pred_img)
        mask_filtered[y1_gt:y2_gt, x1_gt:x2_gt] = pred_img[y1_gt:y2_gt, x1_gt:x2_gt]
        pred_img = mask_filtered

        bbox = masks_to_boxes(pred_img.unsqueeze(0))
        x1, y1, x2, y2 = bbox[0].tolist()

        combined = torch.cat([target_img, pred_img], dim=0) * 255
        combined = combined.type(torch.uint8).cpu()

        img_pil = TF.to_pil_image(combined)
        draw = ImageDraw.Draw(img_pil)

        pred_offset = target_img.shape[0]
        draw.rectangle([x1, y1 + pred_offset, x2, y2 + pred_offset], width=5)

        if draw_pred_boxes:
            draw.rectangle([x1_gt, y1_gt + pred_offset, x2_gt, y2_gt + pred_offset], width=3)

        save_path = os.path.join(img_folder, f"mask_{idx}.png")
        img_pil.save(save_path)

def mask2bbox_nodefaultpsotprocess(pred_masks, pred_boxes, mask_bboxes, scores, threshold, track=False, offset=10):
    """
    pred_masks: Masks with original sizes shape QxXHxW, the inside is 0 or 1(binary)
    pred_boxes: Used for postprocess in masks(finding the true mask) Shape: Qx4  format: x1y1x2y2
    mask_bboxes: List for storing maskbboxes
    scores: Confidence scores of the detections Shape: Qx1
    threshold: cofidence threshold
    offset: Offset for the region of interest in post processing

    Bu versiyonu sürekli post process yapmıyor. Eğer box + offset alanının dışında mask varsa postprocess yapıyor.
    """

    pred_masks = pred_masks.squeeze(dim=1)
    #To store the indices greater than threshold
    if track==True:
        print("THIS IS FOR TRACK DECODER")
    else:
        print("THIS IS FOR DETECTION DECODER")
    indices = (scores >= threshold).nonzero(as_tuple=True)[0]
    for idx in indices:
        print(f"score in indices: {scores[idx]}")

    for idx, (pred_mask, pred_box, score) in enumerate(zip(pred_masks, pred_boxes, scores)):

        if score >= threshold:
            x1, y1, x2, y2 = pred_box
            orig_h, orig_w = pred_mask.shape
            #Tensor to Scalar
            x1 = x1.item()
            y1 = y1.item()
            x2 = x2.item()
            y2 = y2.item()
            #Guard against box overflow
            x1 = max(0, int(round(x1)) - offset)
            y1 = max(0, int(round(y1)) - offset)
            x2 = min(orig_w, int(round(x2)) + offset)
            y2 = min(orig_h, int(round(y2)) + offset)
            #Postprocess for masks to get one object in one instance
            #Store the results for debugging
           
            print(f"pred_mask b4 maskfiltered: {pred_mask.sum()}")
            print(f"pred_mask boxes b4 maskfiltered: {masks_to_boxes(pred_mask.unsqueeze(0))[0].tolist()}")
            outside_region = pred_mask.clone()
            outside_region[y1:y2, x1:x2] = 0

            if outside_region.sum() != 0:
                mask_filtered = torch.zeros_like(pred_mask)
                mask_filtered[y1:y2, x1:x2] = pred_mask[y1:y2, x1:x2]
                if mask_filtered.sum() != 0:
                    pred_mask = mask_filtered
                else:
                    pass
            else:
                pass
            #Store the results for debugging
            print(f"pred_mask after maskfiltered: {pred_mask.sum()}")
            print(f"matched indices: {indices}")    
            print(f"track_box: {pred_box}")
            print(f"index: {idx}")
            print(f"score: {score}")

            mask_bbox = masks_to_boxes(pred_mask.unsqueeze(0))
            print(f"track_mask_bboxes: {mask_bbox[0].tolist()}")
                
            mask_bboxes.append(mask_bbox[0].tolist())
        else:
            mask_bboxes.append([0.0, 0.0, 0.0, 0.0])
    if track==True:
        print("-------------------------------------------------------------------------------------------------")

    return mask_bboxes

def mask2bbox(pred_masks, pred_boxes, mask_bboxes, scores, threshold, track=False, offset=10):
    """
    pred_masks: Masks with original sizes shape QxXHxW, the inside is 0 or 1(binary)
    pred_boxes: Used for postprocess in masks(finding the true mask) Shape: Qx4  format: x1y1x2y2
    mask_bboxes: List for storing maskbboxes
    scores: Confidence scores of the detections Shape: Qx1
    threshold: cofidence threshold
    offset: Offset for the region of interest in post processing
    """

    pred_masks = pred_masks.squeeze(dim=1)
    #To store the indices greater than threshold
    if track==True:
        print("THIS IS FOR TRACK DECODER")
    else:
        print("THIS IS FOR DETECTION DECODER")
    indices = (scores >= threshold).nonzero(as_tuple=True)[0]
    for idx in indices:
        print(f"score in indices: {scores[idx]}")

    for idx, (pred_mask, pred_box, score) in enumerate(zip(pred_masks, pred_boxes, scores)):

        if score >= threshold:
            x1, y1, x2, y2 = pred_box
            orig_h, orig_w = pred_mask.shape
            #Tensor to Scalar
            x1 = x1.item()
            y1 = y1.item()
            x2 = x2.item()
            y2 = y2.item()
            #Guard against box overflow
            x1 = max(0, int(round(x1)) - offset)
            y1 = max(0, int(round(y1)) - offset)
            x2 = min(orig_w, int(round(x2)) + offset)
            y2 = min(orig_h, int(round(y2)) + offset)
            #Postprocess for masks to get one object in one instance
            #Store the results for debugging
           
            print(f"pred_mask b4 maskfiltered: {pred_mask.sum()}")
            print(f"pred_mask boxes b4 maskfiltered: {masks_to_boxes(pred_mask.unsqueeze(0))[0].tolist()}")
            mask_filtered = torch.zeros_like(pred_mask)
            mask_filtered[y1:y2, x1:x2] = pred_mask[y1:y2, x1:x2]

            if mask_filtered.sum() != 0:
                pred_mask = mask_filtered
            else:
                pass
            #Store the results for debugging
            print(f"pred_mask after maskfiltered: {pred_mask.sum()}")
            print(f"matched indices: {indices}")    
            print(f"track_box: {pred_box}")
            print(f"index: {idx}")
            print(f"score: {score}")

            mask_bbox = masks_to_boxes(pred_mask.unsqueeze(0))
            print(f"track_mask_bboxes: {mask_bbox[0].tolist()}")

            if mask_bbox[0][0] == mask_bbox[0][2] or mask_bbox[0][1] == mask_bbox[0][3]:
                mask_bbox[0] = [-1000.0, -1000.0, -999.0, -999.0]  
            print(f"track_mask_bboxes After Check: {mask_bbox[0].tolist()}")

            mask_bboxes.append(mask_bbox[0].tolist())
        else:
            mask_bboxes.append([-1000.0, -1000.0, -999.0, -999.0])
    if track==True:
        print("-------------------------------------------------------------------------------------------------")

    return mask_bboxes
##NOTE: BU HATA ALDIĞIMIZ AMA DOĞRU OALRAK DÜŞÜNDÜĞÜMÜZ KISMI. YUKARIDAKİ ONUN BBOX IN MASKLA ORTUSEN YERI OLMAMASI DURUMUNDA FİLTRELEME YAPMADAN MASKENİN BBOX INI AL DEDİĞİM KISIM.
# def mask2bbox(pred_masks, pred_boxes, mask_bboxes, scores, threshold, track=False, offset=10):
#     """
#     pred_masks: Masks with original sizes shape QxXHxW, the inside is 0 or 1(binary)
#     pred_boxes: Used for postprocess in masks(finding the true mask) Shape: Qx4  format: x1y1x2y2
#     mask_bboxes: List for storing maskbboxes
#     scores: Confidence scores of the detections Shape: Qx1
#     threshold: cofidence threshold
#     offset: Offset for the region of interest in post processing
#     """

#     pred_masks = pred_masks.squeeze(dim=1)
#     #To store the indices greater than threshold
#     if track==True:
#         print("THIS IS FOR TRACK DECODER")
#     else:
#         print("THIS IS FOR DETECTION DECODER")
#     indices = (scores >= threshold).nonzero(as_tuple=True)[0]
#     for idx in indices:
#         print(f"score in indices: {scores[idx]}")

#     for idx, (pred_mask, pred_box, score) in enumerate(zip(pred_masks, pred_boxes, scores)):

#         if score >= threshold:
#             x1, y1, x2, y2 = pred_box
#             orig_h, orig_w = pred_mask.shape
#             #Tensor to Scalar
#             x1 = x1.item()
#             y1 = y1.item()
#             x2 = x2.item()
#             y2 = y2.item()
#             #Guard against box overflow
#             x1 = max(0, int(round(x1)) - offset)
#             y1 = max(0, int(round(y1)) - offset)
#             x2 = min(orig_w, int(round(x2)) + offset)
#             y2 = min(orig_h, int(round(y2)) + offset)
#             print(f"y1:y2, x1:x2: {y1}:{y2}, {x1}:{x2}")

#             #Postprocess for masks to get one object in one instance
#             #Store the results for debugging
           
#             print(f"pred_mask b4 maskfiltered: {pred_mask.sum()}")
#             print(f"pred_mask boxes b4 maskfiltered: {masks_to_boxes(pred_mask.unsqueeze(0))[0].tolist()}")
#             mask_filtered = torch.zeros_like(pred_mask)
#             mask_filtered[y1:y2, x1:x2] = pred_mask[y1:y2, x1:x2]
#             pred_mask = mask_filtered

#             #Store the results for debugging
#             print(f"pred_mask after maskfiltered: {pred_mask.sum()}")
#             print(f"matched indices: {indices}")    
#             print(f"track_box: {pred_box}")
#             print(f"index: {idx}")
#             print(f"score: {score}")

#             mask_bbox = masks_to_boxes(pred_mask.unsqueeze(0))
#             print(f"track_mask_bboxes: {mask_bbox[0].tolist()}")
                
#             mask_bboxes.append(mask_bbox[0].tolist())
#         else:
#             mask_bboxes.append([0.0, 0.0, 0.0, 0.0])
#     if track==True:
#         print("-------------------------------------------------------------------------------------------------")

#     return mask_bboxes
#Pazartesi burayı kontrol et ve kaldığın yerden tracker algosuna devam et burası doğru gelıyorsa tracker algosuna yoğunlaş sadece buraların yani oncesının dopru gelmesi zorunlu Bugun baya baktım bruası doğru ama kendime gıvenemıyorum.
#Uygarla bastırın anca öyle anlaşılacak.

def mean_iou_func(outputs, targets, matcher, mask_out, threshold=0.5):
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
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    ##NOTE: Batch size ı kaldırmak ister squeeze ile ister dettracktrainde nasıl yapılmışsa öyle. Bunu Yukarıda arada bir yerde de yapmak gerkeebilir ona debug da bak
    pred_binary = (pred_masks.sigmoid() >= threshold).float()
    target_binary = target_masks.float()
    if mask_out:
        pred2mask_save(targets, pred_binary, target_binary, target_boxes)

    ious = compute_iou(pred_binary, target_binary) #Resulting shape(X,) CHECK THE SHAPE AND LOGİC BEHIND SHAPE
    # #Accumulate IoU and object count CHECK THİS TO MAKE IT CORRECT IF NEEDED
    total_iou += ious.sum().item()
    total_objects += ious.numel()

    return total_iou, total_objects
    ##NOTE: Calculate MeanIOU end


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

def sort_bbox_coords(x1, y1, x2, y2):
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)