import os, torch
import torchvision.transforms.functional as TF
from torchvision.ops import masks_to_boxes
from PIL import ImageDraw
import util.misc as utils
from util import box_ops

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

def mean_iou_func(outputs, targets, prev_targets, matcher, mask_out=False, threshold=0.5, return_only_idxs=False):
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
    #If its the first frame / starting
    ##NOTE: Burası test sırasında bbox masking için tracking decoderinde ayrı matcherdan doğru indexlerin gelmesi için tanımlanmıştı. Bbox mask,ng i kullanmadığımız için şuan commented out
    # if prev_targets == None:
    #     prev_targets = targets

    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
    ##NOTE: Burası test sırasında bbox masking için tracking decoderinde ayrı matcherdan doğru indexlerin gelmesi için tanımlanmıştı. Bbox mask,ng i kullanmadığımız için şuan prev_targets inputtan prev_indices outputtan kaldırıldı
    indices = matcher(outputs_without_aux, targets)
    pred_idx = _get_src_permutation_idx(indices)
    tgt_idx = _get_tgt_permutation_idx(indices)

    if return_only_idxs:
        return None , pred_idx, tgt_idx

    ##NOTE: Burası test sırasında bbox masking için tracking decoderinde ayrı matcherdan doğru indexlerin gelmesi için tanımlanmıştı. Bu yüzden commented out
    #Get the previous idx
    # pred_idx_prev = _get_src_permutation_idx(prev_indices)
    # tgt_idx_prev = _get_tgt_permutation_idx(prev_indices)

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
    pred_binary = (pred_masks.sigmoid() >= threshold).byte()
    target_binary = target_masks.byte()
    if mask_out and False:
        pred2mask_save(targets, pred_binary, target_binary, target_boxes)

    ious = compute_iou(pred_binary, target_binary) #Resulting shape(X,) CHECK THE SHAPE AND LOGİC BEHIND SHAPE
    # #Accumulate IoU and object count CHECK THİS TO MAKE IT CORRECT IF NEEDED
    return ious, pred_idx, tgt_idx

    # return total_iou, total_objects
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

def apply_bbox_masking_and_visualize_eval_old(outputs, targets, matcher, save_binary_masks=True, output_dir="debug_masks"):
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
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        indices = matcher(outputs_without_aux, targets)
        src_idx = _get_src_permutation_idx(indices)
        tgt_idx = _get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        track_masks = outputs["tracking_masks"]

        target_masks, valid = utils.nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        track_masks = track_masks[src_idx]

        # upsample predictions to the target size
        src_masks = utils.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        # upsample predictions to the target size
        track_masks = utils.interpolate(track_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        # Get target sizes
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        # Get bbox preds
        src_boxes = outputs['pred_boxes'][src_idx]
        track_boxes = outputs['tracking_boxes'][src_idx]
        src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
        track_boxes = box_ops.box_cxcywh_to_xyxy(track_boxes)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        src_boxes = src_boxes * scale_fct
        track_boxes = track_boxes * scale_fct

        # Take sigmoid before box masking
        src_masks_sigmoid = src_masks[:, 0].sigmoid()
        track_masks_sigmoid = target_masks[:, 0].sigmoid()

        #Arragnge target_mask so that it will have same index with src_mask
        target_masks = target_masks[tgt_idx]

        device = src_masks_sigmoid.device
        assert src_masks_sigmoid.shape[-2:] == track_masks_sigmoid.shape[-2:]
        N_s, H, W = src_masks_sigmoid.shape
        N_t, _, _ = track_masks_sigmoid.shape

        if save_binary_masks:
            os.makedirs(output_dir, exist_ok=True)
        
        # Clamp bounding boxes to valid image boundaries
        src_boxes_clamped = src_boxes.clone()
        src_boxes_clamped[:, 0] = torch.clamp(src_boxes_clamped[:, 0], min=0, max=W-1)  # x1
        src_boxes_clamped[:, 1] = torch.clamp(src_boxes_clamped[:, 1], min=0, max=H-1)  # y1
        src_boxes_clamped[:, 2] = torch.clamp(src_boxes_clamped[:, 2], min=0, max=W-1)  # x2
        src_boxes_clamped[:, 3] = torch.clamp(src_boxes_clamped[:, 3], min=0, max=H-1)  # y2

        # Clamp bounding boxes to valid image boundaries
        track_boxes_clamped = track_boxes.clone()
        track_boxes_clamped[:, 0] = torch.clamp(track_boxes_clamped[:, 0], min=0, max=W-1)  # x1
        track_boxes_clamped[:, 1] = torch.clamp(track_boxes_clamped[:, 1], min=0, max=H-1)  # y1
        track_boxes_clamped[:, 2] = torch.clamp(track_boxes_clamped[:, 2], min=0, max=W-1)  # x2
        track_boxes_clamped[:, 3] = torch.clamp(track_boxes_clamped[:, 3], min=0, max=H-1)  # y2
        
        # Convert to integer coordinates
        src_boxes_int = src_boxes_clamped.round().long()

        # Convert to integer coordinates
        track_boxes_int = track_boxes_clamped.round().long()

        # Create binary masks for each bounding box
        bbox_masks = torch.zeros_like(src_masks_sigmoid, device=device)
        bbox_masks_track = torch.zeros_like(track_masks_sigmoid, device=device)

        for i in range(N_s):
            x1, y1, x2, y2 = src_boxes_int[i]
            # Ensure x2 >= x1 and y2 >= y1 (in case of invalid boxes)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # Set the bounding box region to 1
            bbox_masks[i, y1:y2+1, x1:x2+1] = 1.0

        
        for i in range(N_t):
            x1_t, y1_t, x2_t, y2_t = track_boxes_int[i]
            # Ensure x2 >= x1 and y2 >= y1 (in case of invalid boxes)
            x1_t, x2_t = min(x1_t, x2_t), max(x1_t, x2_t)
            y1_t, y2_t = min(y1_t, y2_t), max(y1_t, y2_t)

            # Set the bounding box region to 1
            bbox_masks_track[i, y1_t:y2_t+1, x1_t:x2_t+1] = 1.0

        
        # Apply the bbox mask to the predicted masks
        masked_predictions = src_masks_sigmoid * bbox_masks

        # Apply the bbox mask to the predicted masks
        masked_predictions_track = track_masks_sigmoid * bbox_masks_track

        #Save pred masks and their related target masks
        if save_binary_masks and target_masks is not None:
            for i in range(N_s):
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
                filename = f"comparison_{i:03d}.png"
                img.save(os.path.join(output_dir, filename))
                print(f"Saved comparison image: {os.path.join(output_dir, filename)}")
        
        return masked_predictions, masked_predictions_track


def apply_bbox_masking_and_visualize_eval_wbbox(outputs, targets, matcher, save_binary_masks=True, output_dir="debug_masks"):
    """
    Apply bounding box masking to predicted masks and save visualizations.
    
    Args:
        outputs: Model outputs containing pred_masks, pred_boxes, tracking_masks, tracking_boxes
        targets: Ground truth targets
        matcher: Matcher to find correspondences
        save_binary_masks: Whether to save visualization images
        output_dir: Directory to save images
    
    Returns:
        Masked predictions for src and track
    """
    import os
    from PIL import Image
    import numpy as np
    
    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
    indices = matcher(outputs_without_aux, targets)
    src_idx = _get_src_permutation_idx(indices)
    tgt_idx = _get_tgt_permutation_idx(indices)

    src_masks = outputs["pred_masks"]
    track_masks = outputs["tracking_masks"]

    target_masks, valid = utils.nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
    target_masks = target_masks.to(src_masks)

    src_masks = src_masks[src_idx]
    track_masks = track_masks[src_idx]

    # Upsample predictions to the target size
    src_masks = utils.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                            mode="bilinear", align_corners=False)
    track_masks = utils.interpolate(track_masks[:, None], size=target_masks.shape[-2:],
                            mode="bilinear", align_corners=False)
    
    # Get target sizes
    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    
    # Get bbox preds
    src_boxes = outputs['pred_boxes'][src_idx]
    track_boxes = outputs['tracking_boxes'][src_idx]
    src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
    track_boxes = box_ops.box_cxcywh_to_xyxy(track_boxes)
    
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    
    # Repeat scale_fct for each matched prediction
    scale_fct_expanded = scale_fct[src_idx[0]]
    src_boxes = src_boxes * scale_fct_expanded
    track_boxes = track_boxes * scale_fct_expanded

    # Get target boxes using the correct indexing
    # tgt_idx is a tuple of (batch_indices, target_indices)
    target_boxes_list = []
    for batch_id, target_id in zip(tgt_idx[0], tgt_idx[1]):
        if "boxes" in targets[batch_id]:
            target_boxes_list.append(targets[batch_id]["boxes"][target_id])
    
    if len(target_boxes_list) > 0:
        target_boxes = torch.stack(target_boxes_list, dim=0)
        
        if target_boxes.shape[-1] == 4:
            # Assume boxes are in cxcywh format, convert to xyxy
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
            scale_fct_target = scale_fct[tgt_idx[0]]
            target_boxes = target_boxes * scale_fct_target
        else:
            print(f"Warning: target_boxes has unexpected shape: {target_boxes.shape}")
            target_boxes = None
    else:
        target_boxes = None

    # Take sigmoid before box masking
    src_masks_sigmoid = src_masks[:, 0].sigmoid()
    track_masks_sigmoid = track_masks[:, 0].sigmoid()

    # Get target masks using the correct indexing
    # tgt_idx[0] = batch indices, tgt_idx[1] = target indices within each batch
    target_masks_matched = target_masks[tgt_idx[0], tgt_idx[1]]

    device = src_masks_sigmoid.device
    assert src_masks_sigmoid.shape[-2:] == track_masks_sigmoid.shape[-2:]
    N_s, H, W = src_masks_sigmoid.shape
    N_t, _, _ = track_masks_sigmoid.shape

    if save_binary_masks:
        os.makedirs(output_dir, exist_ok=True)
    
    # Clamp bounding boxes to valid image boundaries
    src_boxes_clamped = src_boxes.clone()
    src_boxes_clamped[:, 0] = torch.clamp(src_boxes_clamped[:, 0], min=0, max=W-1)
    src_boxes_clamped[:, 1] = torch.clamp(src_boxes_clamped[:, 1], min=0, max=H-1)
    src_boxes_clamped[:, 2] = torch.clamp(src_boxes_clamped[:, 2], min=0, max=W-1)
    src_boxes_clamped[:, 3] = torch.clamp(src_boxes_clamped[:, 3], min=0, max=H-1)

    track_boxes_clamped = track_boxes.clone()
    track_boxes_clamped[:, 0] = torch.clamp(track_boxes_clamped[:, 0], min=0, max=W-1)
    track_boxes_clamped[:, 1] = torch.clamp(track_boxes_clamped[:, 1], min=0, max=H-1)
    track_boxes_clamped[:, 2] = torch.clamp(track_boxes_clamped[:, 2], min=0, max=W-1)
    track_boxes_clamped[:, 3] = torch.clamp(track_boxes_clamped[:, 3], min=0, max=H-1)

    if target_boxes is not None:
        target_boxes_clamped = target_boxes.clone()
        target_boxes_clamped[:, 0] = torch.clamp(target_boxes_clamped[:, 0], min=0, max=W-1)
        target_boxes_clamped[:, 1] = torch.clamp(target_boxes_clamped[:, 1], min=0, max=H-1)
        target_boxes_clamped[:, 2] = torch.clamp(target_boxes_clamped[:, 2], min=0, max=W-1)
        target_boxes_clamped[:, 3] = torch.clamp(target_boxes_clamped[:, 3], min=0, max=H-1)
        target_boxes_int = target_boxes_clamped.round().long()
    else:
        target_boxes_int = None
    
    # Convert to integer coordinates
    src_boxes_int = src_boxes_clamped.round().long()
    track_boxes_int = track_boxes_clamped.round().long()

    # Create binary masks for each bounding box
    bbox_masks = torch.zeros_like(src_masks_sigmoid, device=device)
    bbox_masks_track = torch.zeros_like(track_masks_sigmoid, device=device)

    for i in range(N_s):
        x1, y1, x2, y2 = src_boxes_int[i]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        bbox_masks[i, y1:y2+1, x1:x2+1] = 1.0

    for i in range(N_t):
        x1_t, y1_t, x2_t, y2_t = track_boxes_int[i]
        x1_t, x2_t = min(x1_t, x2_t), max(x1_t, x2_t)
        y1_t, y2_t = min(y1_t, y2_t), max(y1_t, y2_t)
        bbox_masks_track[i, y1_t:y2_t+1, x1_t:x2_t+1] = 1.0
    
    # Apply the bbox mask to the predicted masks
    masked_predictions = src_masks_sigmoid * bbox_masks
    masked_predictions_track = track_masks_sigmoid * bbox_masks_track

    # Save visualizations grouped by image
    if save_binary_masks and target_masks_matched is not None:
        # Group predictions by image
        obj_count = 0
        
        for batch_id, (tgt_idx_batch, src_idx_batch) in enumerate(indices):
            num_objects = len(tgt_idx_batch)
            if num_objects == 0:
                continue
            
            # Get image name from targets
            image_name = targets[batch_id].get("image_name", f"image_{batch_id:04d}")
            if isinstance(image_name, str):
                image_name = os.path.splitext(os.path.basename(image_name))[0]
            
            for local_obj_id in range(num_objects):
                global_obj_id = obj_count + local_obj_id
                
                # Prepare target mask
                target_mask_np = (target_masks_matched[global_obj_id].detach().cpu().numpy() * 255).astype(np.uint8)
                target_rgb = np.stack([target_mask_np, target_mask_np, target_mask_np], axis=-1)
                
                # Draw red bounding box on target (if available)
                if target_boxes_int is not None:
                    x1_t, y1_t, x2_t, y2_t = target_boxes_int[global_obj_id]
                    x1_t, x2_t = min(x1_t, x2_t), max(x1_t, x2_t)
                    y1_t, y2_t = min(y1_t, y2_t), max(y1_t, y2_t)
                    
                    box_thickness = 2
                    # Red box (RGB: 255, 0, 0)
                    target_rgb[y1_t:y1_t+box_thickness, x1_t:x2_t+1, :] = [255, 0, 0]
                    target_rgb[y2_t-box_thickness+1:y2_t+1, x1_t:x2_t+1, :] = [255, 0, 0]
                    target_rgb[y1_t:y2_t+1, x1_t:x1_t+box_thickness, :] = [255, 0, 0]
                    target_rgb[y1_t:y2_t+1, x2_t-box_thickness+1:x2_t+1, :] = [255, 0, 0]
                
                # Prepare predicted mask
                masked_pred_np = (masked_predictions[global_obj_id].detach().cpu().numpy() * 255).astype(np.uint8)
                pred_rgb = np.stack([masked_pred_np, masked_pred_np, masked_pred_np], axis=-1)
                
                # Draw green bounding box on prediction
                x1, y1, x2, y2 = src_boxes_int[global_obj_id]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                box_thickness = 2
                # Green box (RGB: 0, 255, 0)
                pred_rgb[y1:y1+box_thickness, x1:x2+1, :] = [0, 255, 0]
                pred_rgb[y2-box_thickness+1:y2+1, x1:x2+1, :] = [0, 255, 0]
                pred_rgb[y1:y2+1, x1:x1+box_thickness, :] = [0, 255, 0]
                pred_rgb[y1:y2+1, x2-box_thickness+1:x2+1, :] = [0, 255, 0]
                
                # Create combined image: target on top, prediction on bottom
                combined_height = H * 2
                combined_img = np.zeros((combined_height, W, 3), dtype=np.uint8)
                
                # Top half: target mask with red box
                combined_img[0:H, :, :] = target_rgb
                
                # Bottom half: predicted mask with green box
                combined_img[H:combined_height, :, :] = pred_rgb
                
                # Add white separator line
                separator_thickness = 2
                combined_img[H-separator_thickness:H+separator_thickness, :, :] = [255, 255, 255]
                
                # Save combined image with image name and object id
                img = Image.fromarray(combined_img, mode='RGB')
                filename = f"{image_name}_obj{local_obj_id:02d}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                print(f"Saved comparison image: {filepath}")
            
            obj_count += num_objects
    
    return masked_predictions, masked_predictions_track

def apply_bbox_masking_and_visualize_eval(outputs, targets, matcher, save_binary_masks=True, output_dir="debug_masks_onlymask"):
    """
    Apply bounding box masking to predicted masks and save visualizations.
    
    Args:
        outputs: Model outputs containing pred_masks, pred_boxes, tracking_masks, tracking_boxes
        targets: Ground truth targets
        matcher: Matcher to find correspondences
        save_binary_masks: Whether to save visualization images
        output_dir: Directory to save images
    
    Returns:
        Masked predictions for src and track
    """
    import os
    from PIL import Image
    import numpy as np
    
    outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
    indices = matcher(outputs_without_aux, targets)
    src_idx = _get_src_permutation_idx(indices)
    tgt_idx = _get_tgt_permutation_idx(indices)

    src_masks = outputs["pred_masks"]
    track_masks = outputs["tracking_masks"]

    target_masks, valid = utils.nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
    target_masks = target_masks.to(src_masks)

    src_masks = src_masks[src_idx]
    track_masks = track_masks[src_idx]

    # Upsample predictions to the target size
    src_masks = utils.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                            mode="bilinear", align_corners=False)
    track_masks = utils.interpolate(track_masks[:, None], size=target_masks.shape[-2:],
                            mode="bilinear", align_corners=False)
    
    # Get target sizes
    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    
    # Get bbox preds
    src_boxes = outputs['pred_boxes'][src_idx]
    track_boxes = outputs['tracking_boxes'][src_idx]
    src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
    track_boxes = box_ops.box_cxcywh_to_xyxy(track_boxes)
    
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    
    # Repeat scale_fct for each matched prediction
    scale_fct_expanded = scale_fct[src_idx[0]]
    src_boxes = src_boxes * scale_fct_expanded
    track_boxes = track_boxes * scale_fct_expanded

    # Take sigmoid before box masking
    src_masks_sigmoid = src_masks[:, 0].sigmoid()
    track_masks_sigmoid = track_masks[:, 0].sigmoid()

    # Get target masks using the correct indexing
    target_masks_matched = target_masks[tgt_idx[0], tgt_idx[1]]

    device = src_masks_sigmoid.device
    assert src_masks_sigmoid.shape[-2:] == track_masks_sigmoid.shape[-2:]
    N_s, H, W = src_masks_sigmoid.shape
    N_t, _, _ = track_masks_sigmoid.shape

    if save_binary_masks:
        os.makedirs(output_dir, exist_ok=True)
    
    # Clamp bounding boxes to valid image boundaries
    src_boxes_clamped = src_boxes.clone()
    src_boxes_clamped[:, 0] = torch.clamp(src_boxes_clamped[:, 0], min=0, max=W-1)
    src_boxes_clamped[:, 1] = torch.clamp(src_boxes_clamped[:, 1], min=0, max=H-1)
    src_boxes_clamped[:, 2] = torch.clamp(src_boxes_clamped[:, 2], min=0, max=W-1)
    src_boxes_clamped[:, 3] = torch.clamp(src_boxes_clamped[:, 3], min=0, max=H-1)

    track_boxes_clamped = track_boxes.clone()
    track_boxes_clamped[:, 0] = torch.clamp(track_boxes_clamped[:, 0], min=0, max=W-1)
    track_boxes_clamped[:, 1] = torch.clamp(track_boxes_clamped[:, 1], min=0, max=H-1)
    track_boxes_clamped[:, 2] = torch.clamp(track_boxes_clamped[:, 2], min=0, max=W-1)
    track_boxes_clamped[:, 3] = torch.clamp(track_boxes_clamped[:, 3], min=0, max=H-1)
    
    # Convert to integer coordinates
    src_boxes_int = src_boxes_clamped.round().long()
    track_boxes_int = track_boxes_clamped.round().long()

    # Create binary masks for each bounding box
    bbox_masks = torch.zeros_like(src_masks_sigmoid, device=device)
    bbox_masks_track = torch.zeros_like(track_masks_sigmoid, device=device)

    for i in range(N_s):
        x1, y1, x2, y2 = src_boxes_int[i]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        bbox_masks[i, y1:y2+1, x1:x2+1] = 1.0

    for i in range(N_t):
        x1_t, y1_t, x2_t, y2_t = track_boxes_int[i]
        x1_t, x2_t = min(x1_t, x2_t), max(x1_t, x2_t)
        y1_t, y2_t = min(y1_t, y2_t), max(y1_t, y2_t)
        bbox_masks_track[i, y1_t:y2_t+1, x1_t:x2_t+1] = 1.0
    
    # Apply the bbox mask to the predicted masks
    masked_predictions = src_masks_sigmoid * bbox_masks
    masked_predictions_track = track_masks_sigmoid * bbox_masks_track

    # Save visualizations grouped by image
    if save_binary_masks and target_masks_matched is not None:
        # Group predictions by image
        obj_count = 0
        
        for batch_id, (tgt_idx_batch, src_idx_batch) in enumerate(indices):
            num_objects = len(tgt_idx_batch)
            if num_objects == 0:
                continue
            
            # Get image name from targets
            image_name = targets[batch_id].get("image_name", f"image_{batch_id:04d}")
            if isinstance(image_name, str):
                image_name = os.path.splitext(os.path.basename(image_name))[0]
            
            for local_obj_id in range(num_objects):
                global_obj_id = obj_count + local_obj_id
                
                # Prepare target mask (top half)
                target_mask_np = (target_masks_matched[global_obj_id].detach().cpu().numpy() * 255).astype(np.uint8)
                target_rgb = np.stack([target_mask_np, target_mask_np, target_mask_np], axis=-1)
                
                # Prepare predicted mask (bottom half)
                masked_pred_np = (masked_predictions[global_obj_id].detach().cpu().numpy() * 255).astype(np.uint8)
                pred_rgb = np.stack([masked_pred_np, masked_pred_np, masked_pred_np], axis=-1)
                
                # Create combined image: target on top, prediction on bottom
                combined_height = H * 2
                combined_img = np.zeros((combined_height, W, 3), dtype=np.uint8)
                
                # Top half: target mask
                combined_img[0:H, :, :] = target_rgb
                
                # Bottom half: predicted mask
                combined_img[H:combined_height, :, :] = pred_rgb
                
                # Add white separator line
                separator_thickness = 2
                combined_img[H-separator_thickness:H+separator_thickness, :, :] = [255, 255, 255]
                
                # Save combined image with image name and object id
                img = Image.fromarray(combined_img, mode='RGB')
                filename = f"{image_name}_obj{local_obj_id:02d}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                print(f"Saved comparison image: {filepath}")
            
            obj_count += num_objects
    
    return masked_predictions, masked_predictions_track

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

