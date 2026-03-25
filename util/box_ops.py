# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]
    device = masks.device
    dtype = masks.dtype if masks.is_floating_point() else torch.float32

    y = torch.arange(0, h, dtype=dtype, device=device)
    x = torch.arange(0, w, dtype=dtype, device=device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    #x_max = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    #y_max = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).max(-1)[0] 
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    
    return torch.stack([x_min, y_min, x_max, y_max], 1)


def soft_boxes_from_masks(masks, eps=1e-6):
    """
    masks: [M, H, W], float in [0,1]
    returns: [M, 4] in cx, cy, w, h (pixel)
    """
    M, H, W = masks.shape
    device = masks.device

    ys = torch.linspace(0, H - 1, H, device=device).view(1, H, 1)
    xs = torch.linspace(0, W - 1, W, device=device).view(1, 1, W)

    mass = masks.sum(dim=(1, 2)) + eps  # [M]

    cx = (masks * xs).sum(dim=(1, 2)) / mass
    cy = (masks * ys).sum(dim=(1, 2)) / mass

    # basit bir “spread” hesabı (variance tarzı):
    var_x = (masks * (xs - cx.view(M, 1, 1))**2).sum((1, 2)) / mass
    var_y = (masks * (ys - cy.view(M, 1, 1))**2).sum((1, 2)) / mass

    wx = 4.0 * var_x.sqrt()  # tam keyfi, istersen çarpan değiştir
    hy = 4.0 * var_y.sqrt()

    boxes = torch.stack([cx, cy, wx, hy], dim=1)  # [M,4]
    return boxes

    def get_bbox_from_soft_mask(masks):
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

##NOTE: First verison of soft approach put here for logging
def get_bbox_from_soft_mask(masks):
        """
        Extract bounding boxes from soft masks using differentiable operations.
        This allows gradients to flow back to the mask predictions.
        
        Args:
            masks: torch.Tensor of shape [M, height, width] with soft masks (values in [0, 1])
            
        Returns:
            bboxes: torch.Tensor of shape [M, 4] where each row is [center_x, center_y, width, height]
        """
        B, H, W = masks.shape
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=masks.device, dtype=masks.dtype).view(1, H, 1)
        x_coords = torch.arange(W, device=masks.device, dtype=masks.dtype).view(1, 1, W)
        
        # Compute weighted center using mask values as weights
        # Use a larger epsilon to avoid numerical issues with very small masks
        total_mass = masks.sum(dim=(1, 2)).clamp(min=1e-4)
        
        # Check for nearly empty masks (total mass below threshold)
        min_mass_threshold = 0.01  # Adjust based on your mask resolution
        has_content = total_mass > min_mass_threshold
        
        center_y = (masks * y_coords).sum(dim=(1, 2)) / total_mass
        center_x = (masks * x_coords).sum(dim=(1, 2)) / total_mass
        
        # Compute variance to estimate box size
        cy_expanded = center_y.view(-1, 1, 1)
        cx_expanded = center_x.view(-1, 1, 1)
        
        var_y = (masks * (y_coords - cy_expanded)**2).sum(dim=(1, 2)) / total_mass
        var_x = (masks * (x_coords - cx_expanded)**2).sum(dim=(1, 2)) / total_mass
        
        # Use 2.5*sqrt(variance) to approximate box size
        # This covers approximately the region where mask values are significant
        height = 2.5 * torch.sqrt(var_y.clamp(min=1e-6))
        width = 2.5 * torch.sqrt(var_x.clamp(min=1e-6))
        
        # Clamp box dimensions to reasonable values
        # Prevent extremely small or large boxes
        max_dim = max(H, W)
        height = height.clamp(min=1.0, max=max_dim)
        width = width.clamp(min=1.0, max=max_dim)
        
        # Clamp centers to be within image bounds
        center_y = center_y.clamp(min=0, max=H)
        center_x = center_x.clamp(min=0, max=W)
        
        bboxes = torch.stack([center_x, center_y, width, height], dim=1)
        # Normalize so that bboxes have range 0,1
        bboxes_normalized = bboxes / torch.tensor([img_w, img_h, img_w, img_h], device=bboxes.device, dtype=bboxes.dtype)
        
        # For masks with very low content, set to zero box (consistent with hard mask behavior)
        # These will be filtered out by the valid_mask in loss computation anyway
        default_box = torch.tensor([0.0, 0.0, 0.0, 0.0], device=bboxes.device, dtype=bboxes.dtype)
        bboxes_normalized = torch.where(
            has_content.view(-1, 1).expand(-1, 4),
            bboxes_normalized,
            default_box.unsqueeze(0).expand(B, -1)
        )
        
        return bboxes_normalized