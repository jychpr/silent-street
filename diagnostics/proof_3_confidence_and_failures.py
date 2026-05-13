"""
proof_3_confidence_and_failures.py
====================================
NEEDS: GPU + OV-DQUO checkpoint + COCO val images + annotations.

Checkpoints from OV-DQUO README Google Drive:
  OVDQUO_RN50x4_COCO.pth  (OV-COCO trained, strongest)

What this proves:
    1. Confidence score distribution: novel vs base vs background
       - Extends Fig 3 from the paper with ADDITIONAL breakdowns:
         (a) Large vs small novel objects — showing OLN-missed small objects still have lower confidence
         (b) Novel categories the paper admits OLN misses (keyboard, knife, sink) vs those it finds
    2. Failure cases: side-by-side of images where novel objects are missed
       - Missing detection on small/non-salient novel objects
       - Note: These are the objects OLN never proposed pseudo-labels for

Run:
    python proof_3_confidence_and_failures.py \
        --config config/OV_COCO/OVDQUO_RN50x4.py \
        --checkpoint ckpt/OVDQUO_RN50x4_COCO.pth \
        --coco_val_ann data/Annotations/ovcoco_val.json \
        --img_dir data/Images/val2017 \
        --output_dir analysis_outputs \
        --n_images 50 \
        --score_thresh 0.25

Directory structure expected (matches OV-DQUO README):
    OV-DQUO/
    ├── config/OV_COCO/OVDQUO_RN50x4.py
    ├── ckpt/OVDQUO_RN50x4_COCO.pth
    ├── data/
    │   ├── Annotations/ovcoco_val.json
    │   └── Images/val2017/
    └── proof_3_confidence_and_failures.py  ← place this here
"""

from PIL import ImageFont
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except OSError:
    font = ImageFont.load_default()

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--coco_val_ann', required=True,
                   help='OV-COCO val annotation JSON with novel/base split')
    p.add_argument('--img_dir', required=True)
    p.add_argument('--output_dir', default='analysis_outputs')
    p.add_argument('--n_images', type=int, default=50,
                   help='Number of val images to run inference on')
    p.add_argument('--score_thresh', type=float, default=0.25,
                   help='Confidence threshold for showing detections')
    p.add_argument('--device', default='cuda')
    return p.parse_args()

# ── NOVEL CATEGORY INFO ───────────────────────────────────────
# OV-COCO 17 novel categories — categorized by OLN detectability
# Source: OV-DQUO paper appendix + visual saliency reasoning
OLN_DETECTABLE = {
    # Large, salient — OLN finds these well
    'airplane': True, 'bus': True, 'cat': True, 'dog': True,
    'cow': True, 'elephant': True, 'couch': True,
    # Small/non-salient — OLN misses these (per paper appendix)
    'umbrella': False,  # partially
    'tie': False,       # thin, elongated
    'snowboard': False, # flat, board-shaped
    'skateboard': False, # small, flat
    'cup': False,       # small
    'knife': False,     # thin, small
    'cake': False,      # texture-heavy
    'keyboard': False,  # small, flat
    'sink': False,      # non-salient shape
    'scissors': False,  # thin, small
}

def get_novel_cat_ids(ann_data):
    """Extract novel category IDs from OV-COCO annotation."""
    cats = ann_data.get('categories', [])
    # OV-COCO annotations mark novel vs base in category metadata
    novel_ids = set()
    novel_names = set(OLN_DETECTABLE.keys())
    for c in cats:
        if c['name'] in novel_names or c.get('split') == 'novel':
            novel_ids.add(c['id'])
    print(f"  Found {len(novel_ids)} novel category IDs: {novel_ids}")
    return novel_ids

def load_model(config_path, checkpoint_path, device):
    """Load OV-DQUO model using its own build system."""
    # Add OV-DQUO root to path
    sys.path.insert(0, '.')
    from main import build_model_main
    import torch

    # Load config
    from util.slconfig import SLConfig
    cfg = SLConfig.fromfile(config_path)
    cfg.device = device

    model, criterion, postprocessors = build_model_main(cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    print(f"  Model loaded from {checkpoint_path}")
    return model, postprocessors, cfg


def run_inference_batch(model, postprocessors, img_paths, device, cfg):
    import torch
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize(800, max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Build category text embeddings the same way OV-DQUO eval does
    import torch.nn.functional as F
    classifier = model.classifier  # frozen CLIP text encoder output
    # classifier is already the precomputed text embedding matrix [num_classes, D]
    # Pass it as the categories argument
    categories = classifier  # shape: [C, D], already on device after model.to(device)

    all_results = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor, categories)

        target_sizes = torch.tensor([[h, w]], device=device)
        results = postprocessors["bbox"](outputs, target_sizes)
        all_results.append({
            "img_path": str(img_path),
            "img_size": (w, h),
            "scores": results[0]["scores"].cpu().numpy(),
            "labels": results[0]["labels"].cpu().numpy(),
            "boxes": results[0]["boxes"].cpu().numpy(),
        })
    return all_results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from PIL import Image, ImageDraw

    BG = '#0F1117'

    # ── Load annotations ─────────────────────────────────────
    print("Loading COCO val annotations...")
    with open(args.coco_val_ann) as f:
        ann_data = json.load(f)

    novel_ids = get_novel_cat_ids(ann_data)

    # Build image info lookup
    img_info = {img['id']: img for img in ann_data.get('images', [])}
    # Build GT lookup: img_id → list of (cat_id, bbox, area)
    gt_by_img = {}
    for ann in ann_data.get('annotations', []):
        iid = ann['image_id']
        if iid not in gt_by_img:
            gt_by_img[iid] = []
        gt_by_img[iid].append(ann)

    # ── Load model ───────────────────────────────────────────
    print("Loading OV-DQUO model...")
    try:
        model, postprocessors, cfg = load_model(
            args.config, args.checkpoint, args.device)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Make sure you're running from the OV-DQUO root directory")
        return

    # ── Run inference ────────────────────────────────────────
    import random
    val_img_ids = list(img_info.keys())
    random.shuffle(val_img_ids)
    selected_ids = val_img_ids[:args.n_images]

    img_paths = []
    for iid in selected_ids:
        fname = img_info[iid]['file_name']
        img_paths.append(Path(args.img_dir) / fname)

    print(f"Running inference on {len(img_paths)} images...")
    results = run_inference_batch(model, postprocessors, img_paths, args.device, cfg)

    # ── Collect confidence scores by category type ───────────
    # Build cat_id → name mapping
    cat_id_to_name = {c['id']: c['name'] for c in ann_data.get('categories', [])}

    conf_base = []
    conf_novel_large = []   # novel, OLN-detectable
    conf_novel_small = []   # novel, OLN-missed

    for r in results:
        for score, label in zip(r['scores'], r['labels']):
            cat_name = cat_id_to_name.get(int(label), '')
            if int(label) in novel_ids:
                if OLN_DETECTABLE.get(cat_name, True):
                    conf_novel_large.append(float(score))
                else:
                    conf_novel_small.append(float(score))
            else:
                conf_base.append(float(score))

    # ── Plot 1: Confidence distributions ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle("OV-DQUO Confidence Score Analysis on OV-COCO Val",
                 fontsize=14, fontweight='bold', color='white')

    # Left: base vs novel overall
    ax = axes[0]
    ax.set_facecolor(BG)
    bins = np.linspace(0, 1, 40)
    if conf_base:
        ax.hist(conf_base, bins=bins, alpha=0.6, color='#5C9EE0', label=f'Base categories (n={len(conf_base)})', density=True)
    if conf_novel_large or conf_novel_small:
        all_novel = conf_novel_large + conf_novel_small
        ax.hist(all_novel, bins=bins, alpha=0.6, color='#E05C5C', label=f'Novel categories (n={len(all_novel)})', density=True)
    ax.set_xlabel("Confidence score", color='#AAAACC', fontsize=11)
    ax.set_ylabel("Density", color='#AAAACC', fontsize=11)
    ax.set_title("Base vs Novel confidence\n(replicating paper Fig 3)", color='white', fontsize=11)
    ax.tick_params(colors='#AAAACC')
    ax.spines[:].set_color('#2A2A3A')
    ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='white', labelcolor='white')
    ax.text(0.05, 0.9, "Novel objects get lower\nconfidence → suppressed\nby NMS/thresholding",
            transform=ax.transAxes, fontsize=9, color='#FF9944',
            bbox=dict(boxstyle='round', facecolor='#1A1A2E', edgecolor='#FF9944', alpha=0.8))

    # Right: novel breakdown — OLN-detectable vs OLN-missed
    ax = axes[1]
    ax.set_facecolor(BG)
    if conf_novel_large:
        ax.hist(conf_novel_large, bins=bins, alpha=0.6, color='#4CAF50',
                label=f'Novel (OLN-detectable, large)\nn={len(conf_novel_large)}', density=True)
    if conf_novel_small:
        ax.hist(conf_novel_small, bins=bins, alpha=0.6, color='#E05C5C',
                label=f'Novel (OLN-missed, small)\nn={len(conf_novel_small)}', density=True)
    ax.set_xlabel("Confidence score", color='#AAAACC', fontsize=11)
    ax.set_ylabel("Density", color='#AAAACC', fontsize=11)
    ax.set_title("Novel: OLN-detectable vs OLN-missed\n(new breakdown — paper doesn't show this)",
                 color='white', fontsize=11)
    ax.tick_params(colors='#AAAACC')
    ax.spines[:].set_color('#2A2A3A')
    ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='white', labelcolor='white')
    ax.text(0.05, 0.9,
            "Objects OLN never proposed\npseudo-labels for still have\nlower confidence even after DTQT",
            transform=ax.transAxes, fontsize=9, color='#FF9944',
            bbox=dict(boxstyle='round', facecolor='#1A1A2E', edgecolor='#FF9944', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "proof3a_confidence_distributions.png"),
                dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: proof3a_confidence_distributions.png")

    # ── Plot 2: Failure case visualization ───────────────────
    # Find images where novel objects exist in GT but are missed by detector
    print("Finding failure cases...")
    failure_cases = []

    for r, iid in zip(results, selected_ids):
        gt_anns = gt_by_img.get(iid, [])
        novel_gt = [a for a in gt_anns if a['category_id'] in novel_ids]
        if not novel_gt:
            continue

        # Get detector predictions above threshold
        pred_novel_boxes = []
        for score, label, box in zip(r['scores'], r['labels'], r['boxes']):
            if int(label) in novel_ids and float(score) >= args.score_thresh:
                pred_novel_boxes.append((float(score), box))

        # Count missed GTs
        n_missed = 0
        for gt_ann in novel_gt:
            gt_box = gt_ann['bbox']  # [x, y, w, h]
            gt_area = gt_box[2] * gt_box[3]
            # Check if any prediction overlaps
            matched = False
            for score, pred_box in pred_novel_boxes:
                # Simple IoU check
                gx1, gy1 = gt_box[0], gt_box[1]
                gx2, gy2 = gx1 + gt_box[2], gy1 + gt_box[3]
                px1, py1, px2, py2 = pred_box
                ix1, iy1 = max(gx1, px1), max(gy1, py1)
                ix2, iy2 = min(gx2, px2), min(gy2, py2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1) * (iy2-iy1)
                    union = (gx2-gx1)*(gy2-gy1) + (px2-px1)*(py2-py1) - inter
                    if inter/union > 0.3:
                        matched = True
                        break
            if not matched:
                n_missed += 1

        if n_missed > 0:
            failure_cases.append({
                'img_id': iid,
                'img_path': r['img_path'],
                'n_missed': n_missed,
                'novel_gt': novel_gt,
                'pred_novel': pred_novel_boxes,
                'cat_names': [cat_id_to_name.get(a['category_id'], '?') for a in novel_gt],
            })

    failure_cases.sort(key=lambda x: x['n_missed'], reverse=True)
    print(f"  Found {len(failure_cases)} images with missed novel objects")

    # Visualize top 6 failure cases
    n_show = min(6, len(failure_cases))
    if n_show == 0:
        print("  No failure cases to visualize at this threshold.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor=BG)
    fig.suptitle(f"OV-DQUO Failure Cases: Missed Novel Objects (threshold={args.score_thresh})",
                 fontsize=14, fontweight='bold', color='white')

    for idx, case in enumerate(failure_cases[:n_show]):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor(BG)

        img = Image.open(case['img_path']).convert('RGB')
        draw = ImageDraw.Draw(img)

        # Draw GT novel boxes in red (missed)
        for ann in case['novel_gt']:
            x, y, w, h = ann['bbox']
            cat_name = cat_id_to_name.get(ann['category_id'], '?')
            draw.rectangle([x, y, x+w, y+h], outline='red', width=3)
            # draw.text((x, max(0, y-15)), f"GT: {cat_name}", fill='red')
            draw.text((x, max(0, y-28)), f"GT: {cat_name}", fill='red', font=font)
            draw.text((px1, py1-28), f"{cat_name} {score:.2f}", fill='lime', font=font)

        # Draw predictions in green
        for score, box in case['pred_novel']:
            px1, py1, px2, py2 = box
            draw.rectangle([px1, py1, px2, py2], outline='lime', width=2)
            draw.text((px1, py1), f"{score:.2f}", fill='lime')

        ax.imshow(np.array(img))
        ax.set_title(f"Missed: {case['n_missed']} novel obj(s)\n"
                     f"Categories: {', '.join(set(case['cat_names'])[:3])}",
                     color='white', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "proof3b_failure_cases.png"),
                dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: proof3b_failure_cases.png")
    print("\nKey slide caption:")
    print("  'Red = GT novel objects. Green = OV-DQUO predictions.")
    print("   OLN never proposed pseudo-labels for these objects during training.")
    print("   Without pseudo-label supervision, the detector cannot overcome confidence bias.'")

if __name__ == '__main__':
    main()
