#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

# scripts/ altından çalıştırınca repo root import edebilmek için:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.misc import nested_tensor_from_tensor_list


# ---------------------------
# Image
# ---------------------------
def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_transform(img_size: int = 0) -> T.Compose:
    tfms = []
    if img_size and img_size > 0:
        tfms.append(T.Resize(img_size))
    tfms.append(T.ToTensor())
    return T.Compose(tfms)


def make_samples(pil_img: Image.Image, transform: T.Compose, device: str):
    x = transform(pil_img)  # (3,H,W)
    H, W = x.shape[-2], x.shape[-1]
    vis_u8 = (x * 255.0).clamp(0, 255).to(torch.uint8)
    samples = nested_tensor_from_tensor_list([x]).to(device)
    return samples, vis_u8, H, W


# ---------------------------
# Output parse (DETR-style)
# ---------------------------
def choose_scores_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # sigmoid (multi-label)
    s_sig = logits.sigmoid().max(dim=-1).values

    # softmax (single-label) - drop last as background candidate
    probs = logits.softmax(dim=-1)
    if probs.shape[-1] > 1:
        s_sm = probs[:, :-1].max(dim=-1).values
    else:
        s_sm = probs[:, 0]

    return s_sm if s_sm.max() > s_sig.max() else s_sig


def cxcywh_to_xyxy_px(boxes: torch.Tensor, H: int, W: int) -> torch.Tensor:
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - 0.5 * bw) * W
    y1 = (cy - 0.5 * bh) * H
    x2 = (cx + 0.5 * bw) * W
    y2 = (cy + 0.5 * bh) * H
    b = torch.stack([x1, y1, x2, y2], dim=-1)
    b[:, 0::2] = b[:, 0::2].clamp(0, W - 1)
    b[:, 1::2] = b[:, 1::2].clamp(0, H - 1)
    return b


def parse_output(out: Any, H: int, W: int, score_thr: float, debug: bool):
    if isinstance(out, (list, tuple)):
        out = out[0]
    if not isinstance(out, dict):
        raise RuntimeError(f"Unexpected output type: {type(out)}")

    if debug:
        print("[DEBUG] out keys:", list(out.keys()))

    logits = out["pred_logits"][0]  # (Nq,C)
    boxes = out["pred_boxes"][0]    # (Nq,4) cxcywh norm

    scores = choose_scores_from_logits(logits)

    if debug:
        print(f"[DEBUG] scores max={scores.max().item():.4f} mean={scores.mean().item():.4f}")

    keep = scores >= score_thr
    if debug:
        print(f"[DEBUG] keep {keep.sum().item()}/{scores.numel()} with thr={score_thr}")

    scores = scores[keep]
    boxes = boxes[keep]
    boxes_xyxy = cxcywh_to_xyxy_px(boxes, H, W)

    masks_bool = None
    if "pred_masks" in out and out["pred_masks"] is not None:
        pm = out["pred_masks"][0]  # (Nq,Hm,Wm) or (Nq,1,Hm,Wm)
        if pm.ndim == 4 and pm.shape[1] == 1:
            pm = pm[:, 0]
        pm = pm[keep]

        pm = pm.sigmoid()
        pm = F.interpolate(pm[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        masks_bool = pm > 0.5

    return boxes_xyxy, scores, masks_bool


# ---------------------------
# Build model from ckpt args (inject missing fields)
# ---------------------------
def load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError("Checkpoint must be a dict containing at least key 'model'.")
    return ckpt


def ensure_arg(args, name: str, default):
    # Namespace için güvenli attribute ekleme
    if not hasattr(args, name):
        setattr(args, name, default)


def build_model_from_ckpt_args(ckpt: Dict[str, Any], device: str):
    if "args" not in ckpt or ckpt["args"] is None:
        raise RuntimeError("Checkpoint does not include 'args'. Cannot rebuild the exact architecture.")

    args = ckpt["args"]

    # ---- inject missing flags expected by current code ----
    # build_tracktest_model -> build_tracktest -> uses args.mask, args.frozen_weights, etc.
    ensure_arg(args, "mask", True)        # IMPORTANT: your crash was here
    ensure_arg(args, "mask_out", True)
    ensure_arg(args, "frozen_weights", None)
    ensure_arg(args, "bbox_masking", False)
    ensure_arg(args, "fp16", False)

    # Make sure masks is enabled (some older ckpt may have masks=False)
    ensure_arg(args, "masks", True)

    # inference overrides
    args.device = device
    args.eval = True

    from models import build_tracktest_model
    model, _, _, _ = build_tracktest_model(args)
    model.to(device).eval()
    return model


def fix_state_dict_prefix_if_needed(model: torch.nn.Module, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    If model expects deform_detr.* but ckpt has transformer.* etc, prefix.
    We'll try direct load first; if missing/unexpected huge, rebuild sd.
    """
    # quick test by looking at model keys
    model_keys = list(model.state_dict().keys())
    model_expects_deform = any(k.startswith("deform_detr.") for k in model_keys)

    ckpt_keys = list(sd.keys())
    ckpt_has_deform = any(k.startswith("deform_detr.") for k in ckpt_keys)

    if model_expects_deform and not ckpt_has_deform:
        # prefix all ckpt keys into deform_detr.*
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("deform_detr."):
                new_sd[k] = v
            else:
                new_sd["deform_detr." + k] = v
        return new_sd

    return sd


def load_weights(model: torch.nn.Module, ckpt: Dict[str, Any]):
    sd = ckpt["model"]
    sd = fix_state_dict_prefix_if_needed(model, sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if not (k.endswith("total_params") or k.endswith("total_ops"))]

    print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[LOAD] missing sample:", missing[:20])
    if len(unexpected) > 0:
        print("[LOAD] unexpected sample:", unexpected[:20])


# ---------------------------
# Inference
# ---------------------------
@torch.no_grad()
def infer_one(model, samples, vis_u8, H, W, out_dir: str, score_thr: float, debug: bool):
    os.makedirs(out_dir, exist_ok=True)

    out = model(samples)
    boxes_xyxy, scores, masks_bool = parse_output(out, H, W, score_thr, debug)

    raw_path = os.path.join(out_dir, "pred_raw.pt")
    torch.save(
        {"boxes_xyxy": boxes_xyxy.cpu(),
         "scores": scores.cpu(),
         "masks_bool": masks_bool.cpu() if masks_bool is not None else None},
        raw_path
    )

    img = vis_u8.clone()
    if boxes_xyxy.numel() > 0:
        labels = [f"{s:.2f}" for s in scores.cpu().tolist()]
        img = draw_bounding_boxes(img, boxes_xyxy.cpu().to(torch.int64), labels=labels, width=2)

    if masks_bool is not None and masks_bool.numel() > 0:
        img = draw_segmentation_masks(img, masks_bool.cpu().bool(), alpha=0.5)

    vis_path = os.path.join(out_dir, "pred_vis.png")
    T.ToPILImage()(img).save(vis_path)

    print(f"[OK] saved: {vis_path}")
    print(f"[OK] saved: {raw_path}")
    print(f"[OK] detections kept: {boxes_xyxy.shape[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--score_thr", type=float, default=0.5)
    ap.add_argument("--img_size", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    ckpt = load_checkpoint(args.model)
    model = build_model_from_ckpt_args(ckpt, device=device)
    load_weights(model, ckpt)

    pil = load_image_rgb(args.image)
    transform = build_transform(args.img_size)
    samples, vis_u8, H, W = make_samples(pil, transform, device)

    infer_one(model, samples, vis_u8, H, W, args.out_dir, args.score_thr, args.debug)


if __name__ == "__main__":
    main()
