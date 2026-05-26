"""
Visual comparison: FastSAM pseudo-label sets side by side.

For each target image, produces a 2-panel PNG (left=baseline, right=right-side).
Also produces a contact sheet stacking all panels vertically.

Usage (original K=5 comparison):
    python diagnostics/compare_pseudo_labels_esam_vs_baseline.py

Usage (K=30 comparison with custom paths):
    python diagnostics/compare_pseudo_labels_esam_vs_baseline.py \
        --baseline-json ow_labels/diag_fastsam_baseline_K30_12img.json \
        --esam-json ow_labels/diag_fastsam_esam_K30_12img.json \
        --ann-json data/Annotations/instances_train2017_12img_diag.json \
        --target-ids 157105 122263 543882 579329 443084 92869 475808 435091 171270 307238 30 34 \
        --out-suffix K30 \
        --left-color 0,255,0 \
        --right-color 0,255,255
"""

import argparse
import json
import os
from collections import defaultdict

from PIL import Image, ImageDraw

_DEFAULT_BASELINE_JSON = "ow_labels/OW_COCO_FASTSAM_K5_conf010_filtered.json"
_DEFAULT_ESAM_JSON     = "ow_labels/diag_fastsam_esam_6img.json"
_DEFAULT_ANN_JSON      = "data/Annotations/instances_train2017_6img_diag.json"
IMG_DIR                = "data/Images/train2017"
OUT_DIR                = "diagnostics/output"

_DEFAULT_TARGET_IDS = [157105, 543882, 443084, 435091, 475808, 34]
BOX_WIDTH  = 3
LABEL_PAD  = 2


def _parse_color(s: str) -> tuple[int, int, int]:
    r, g, b = s.split(",")
    return (int(r), int(g), int(b))


def load_boxes_by_image(json_path: str) -> dict[int, list[list[float]]]:
    with open(json_path) as f:
        data = json.load(f)
    result: dict[int, list[list[float]]] = defaultdict(list)
    for ann in data:
        result[ann["image_id"]].append(ann["bbox"])  # xywh
    return result


def load_image_meta(ann_path: str) -> dict[int, dict]:
    with open(ann_path) as f:
        data = json.load(f)
    return {im["id"]: im for im in data["images"]}


def draw_panel(
    img: Image.Image,
    boxes: list[list[float]],
    title: str,
    box_color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """Draw boxes on a copy of img and add a title bar at the top."""
    panel = img.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)

    for i, (x, y, w, h) in enumerate(boxes):
        x1, y1, x2, y2 = x, y, x + w, y + h
        for t in range(BOX_WIDTH):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=box_color)
        label = str(i)
        tx, ty = x1 + LABEL_PAD, y1 + LABEL_PAD
        # black background for readability
        draw.rectangle([tx - 1, ty - 1, tx + 8, ty + 11], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=(255, 255, 0))

    # title bar
    bar_h = 22
    bar = Image.new("RGB", (panel.width, bar_h), color=(30, 30, 30))
    bar_draw = ImageDraw.Draw(bar)
    bar_draw.text((4, 4), title, fill=(255, 255, 255))

    composite = Image.new("RGB", (panel.width, panel.height + bar_h))
    composite.paste(bar, (0, 0))
    composite.paste(panel, (0, bar_h))
    return composite


def make_side_by_side(
    img: Image.Image,
    base_boxes: list[list[float]],
    esam_boxes: list[list[float]],
    image_id: int,
    left_title: str = "",
    right_title: str = "",
    left_color: tuple[int, int, int] = (0, 255, 0),
    right_color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    left_title  = left_title  or f"FastSAM K=5 baseline ({len(base_boxes)} boxes)"
    right_title = right_title or f"FastSAM K=5 + E-SAM adaptive NMS ({len(esam_boxes)} boxes)"
    left  = draw_panel(img, base_boxes, left_title,  box_color=left_color)
    right = draw_panel(img, esam_boxes, right_title, box_color=right_color)

    # Ensure same height (they always are since same source image, just in case)
    h = max(left.height, right.height)
    if left.height < h:
        pad = Image.new("RGB", (left.width, h), (20, 20, 20))
        pad.paste(left, (0, 0))
        left = pad
    if right.height < h:
        pad = Image.new("RGB", (right.width, h), (20, 20, 20))
        pad.paste(right, (0, 0))
        right = pad

    # ID label strip between panels
    sep_w = 8
    strip = Image.new("RGB", (sep_w, h), (60, 60, 60))

    composite = Image.new("RGB", (left.width + sep_w + right.width, h))
    composite.paste(left,  (0, 0))
    composite.paste(strip, (left.width, 0))
    composite.paste(right, (left.width + sep_w, 0))
    return composite


def make_contact_sheet(panels: list[Image.Image]) -> Image.Image:
    max_w = max(p.width for p in panels)
    total_h = sum(p.height for p in panels) + 4 * (len(panels) - 1)
    sheet = Image.new("RGB", (max_w, total_h), (10, 10, 10))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + 4
    return sheet


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-json", default=_DEFAULT_BASELINE_JSON)
    p.add_argument("--esam-json",     default=_DEFAULT_ESAM_JSON)
    p.add_argument("--ann-json",      default=_DEFAULT_ANN_JSON)
    p.add_argument("--target-ids",    nargs="+", type=int, default=_DEFAULT_TARGET_IDS)
    p.add_argument("--out-suffix",    default="",
                   help="Inserted before image_id in output filenames, e.g. 'K30'")
    p.add_argument("--left-color",    default="0,255,0",
                   help="RGB for left (baseline) boxes, e.g. '0,255,0'")
    p.add_argument("--right-color",   default="0,255,0",
                   help="RGB for right (E-SAM) boxes, e.g. '0,255,255'")
    p.add_argument("--left-label",    default="",
                   help="Override left panel title prefix (default: auto)")
    p.add_argument("--right-label",   default="",
                   help="Override right panel title prefix (default: auto)")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    left_color  = _parse_color(args.left_color)
    right_color = _parse_color(args.right_color)
    suffix      = f"_{args.out_suffix}" if args.out_suffix else ""

    base_boxes = load_boxes_by_image(args.baseline_json)
    esam_boxes = load_boxes_by_image(args.esam_json)
    img_meta   = load_image_meta(args.ann_json)

    panels = []
    for image_id in args.target_ids:
        meta     = img_meta[image_id]
        img_path = os.path.join(IMG_DIR, meta["file_name"])
        img      = Image.open(img_path).convert("RGB")

        b_boxes = base_boxes.get(image_id, [])
        e_boxes = esam_boxes.get(image_id, [])

        left_title  = args.left_label  or f"Baseline ({len(b_boxes)} boxes)"
        right_title = args.right_label or f"E-SAM adaptive NMS ({len(e_boxes)} boxes)"

        panel = make_side_by_side(
            img, b_boxes, e_boxes, image_id,
            left_title=left_title, right_title=right_title,
            left_color=left_color, right_color=right_color,
        )
        panels.append(panel)

        out_path = os.path.join(OUT_DIR, f"esam_vs_baseline{suffix}_{image_id}.png")
        panel.save(out_path)
        print(f"[{image_id}]  baseline={len(b_boxes)}  esam={len(e_boxes)}  → {out_path}")

    sheet_path = os.path.join(OUT_DIR, f"esam_vs_baseline{suffix}_all.png")
    make_contact_sheet(panels).save(sheet_path)
    print(f"\nContact sheet → {sheet_path}")


if __name__ == "__main__":
    main()
