"""
build_side_by_side.py
=====================
Reads two predictions JSON files (one per checkpoint) produced by
proof_3_confidence_and_failures.py and builds a 2-row composite figure
per image showing published OV-DQUO on top and FastSAM-trained on bottom,
with GT novel boxes overlaid in both panels.

Usage:
    python diagnostic/build_side_by_side.py \
        --published_json diagnostics/output/published_ovdquo/predictions.json \
        --fastsam_json   diagnostics/output/fastsam_k5_w1/predictions.json
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
except OSError:
    font = ImageFont.load_default()


def iou_xywh_xyxy(gt_xywh, pred_xyxy):
    gx, gy, gw, gh = gt_xywh
    gx2, gy2 = gx + gw, gy + gh
    px1, py1, px2, py2 = pred_xyxy
    ix1, iy1 = max(gx, px1), max(gy, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = gw * gh + (px2 - px1) * (py2 - py1) - inter
    return inter / union if union > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--published_json',
                   default='diagnostics/output/published_ovdquo__predictions.json')
    p.add_argument('--fastsam_json',
                   default='diagnostics/output/fastsam_k5_w1__predictions.json')
    p.add_argument('--published_ap', type=float, default=39.2)
    p.add_argument('--fastsam_ap', type=float, default=37.13)
    p.add_argument('--fastsam_epoch', type=int, default=15)
    p.add_argument('--output_dir', default='diagnostics/output')
    return p.parse_args()


def load_by_image_id(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {rec["image_id"]: rec for rec in data["images"]}


def draw_panel(img_path, predictions, novel_gt, title, iou_thresh=0.3):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    pred_boxes = [pred["box"] for pred in predictions]

    gt_matched = [
        any(iou_xywh_xyxy(gt["bbox_xywh"], pb) >= iou_thresh for pb in pred_boxes)
        for gt in novel_gt
    ]
    pred_matched = [
        any(iou_xywh_xyxy(gt["bbox_xywh"], pred["box"]) >= iou_thresh for gt in novel_gt)
        for pred in predictions
    ]

    for gt, matched in zip(novel_gt, gt_matched):
        x, y, w, h = gt["bbox_xywh"]
        color = '#FFA500' if matched else 'red'
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((x, max(0, y - 28)), f"GT: {gt['cat_name']}", fill=color, font=font)

    for pred, matched in zip(predictions, pred_matched):
        x1, y1, x2, y2 = pred["box"]
        color = 'lime' if matched else 'cyan'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 28)), f"{pred['cat_name']} {pred['score']:.2f}",
                  fill=color, font=font)

    title_h = 40
    canvas = Image.new("RGB", (img.width, img.height + title_h), color=(15, 17, 23))
    title_draw = ImageDraw.Draw(canvas)
    title_draw.text((8, 8), title, fill='white', font=font)
    canvas.paste(img, (0, title_h))
    return canvas


def stack_vertically(panels):
    w = max(p.width for p in panels)
    h = sum(p.height for p in panels)
    out = Image.new("RGB", (w, h), color=(15, 17, 23))
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height
    return out


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading prediction JSONs...")
    pub = load_by_image_id(args.published_json)
    fas = load_by_image_id(args.fastsam_json)

    common_ids = sorted(set(pub.keys()) & set(fas.keys()))
    print(f"  {len(common_ids)} images in common: {common_ids}")

    pub_title = f"Published OV-DQUO — AP_novel {args.published_ap}"
    fas_title = f"FastSAM K=5 w=1.0 (epoch {args.fastsam_epoch} best) — AP_novel {args.fastsam_ap}"

    composites = []
    for iid in common_ids:
        pub_rec = pub[iid]
        fas_rec = fas[iid]

        top = draw_panel(pub_rec["image_path"], pub_rec["predictions"],
                         pub_rec["novel_gt"], pub_title)
        bot = draw_panel(fas_rec["image_path"], fas_rec["predictions"],
                         fas_rec["novel_gt"], fas_title)

        composite = stack_vertically([top, bot])
        out_path = os.path.join(args.output_dir, f"side_by_side_{iid}.png")
        composite.save(out_path, dpi=(150, 150))
        print(f"  Saved: {out_path}")
        composites.append(composite)

    if not composites:
        print("No images in common between the two JSONs — nothing to write.")
        return

    # Contact sheet: all images stacked
    sheet = stack_vertically(composites)
    sheet_path = os.path.join(args.output_dir, "side_by_side_all.png")
    sheet.save(sheet_path, dpi=(150, 150))
    print(f"Saved contact sheet: {sheet_path}")


if __name__ == '__main__':
    main()
