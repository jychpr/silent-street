"""
EMR Step 3 — Visual diagnostic on 12 diagnostic images.

For each image, a single FastSAM call produces both:
  - baseline top-5: raw proposals → geometric filters → top-K (no merge)
  - EMR top-5:      raw proposals → merge loop → geometric filters → top-K

Each EMR box is classified against the baseline:
  identical  — IoU ≥ 0.9 with a baseline box (same raw proposal survived)
  merged     — merge product with no baseline match
  reordered  — non-merged raw box newly promoted (slot freed by a merge)

Outputs per image:
  - diagnostics/output/emr_step3/{image_id}.png  (annotated side-by-side)
  - stdout text table: rank | origin | size | conf | best_bl_iou

Size buckets (COCO standard):
  S  area < 32²  = 1024 px²
  M  1024 ≤ area < 96² = 9216 px²
  L  area ≥ 9216 px²

No characterization of better/worse — geometry and merge-origin only.
"""

import argparse
import json
import logging
import math
import os

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DIAG_IMAGE_IDS = frozenset({
    157105, 122263, 543882, 579329, 443084, 92869,
    475808, 435091, 171270, 307238, 30, 34,
})
_DILATION_STRUCT = np.ones((3, 3), dtype=bool)
_IOU_MATCH_THR = 0.9


def parse_args():
    p = argparse.ArgumentParser(description="EMR Step 3 — visual diagnostic on 12 images")
    p.add_argument("--coco-root", required=True)
    p.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    p.add_argument("--coco-ann", required=True,
                   help="12-image diagnostic annotation JSON")
    p.add_argument("--fastsam-weights", default="FastSAM-x.pt")
    p.add_argument("--output-dir", default="diagnostics/output/emr_step3")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-box-side", type=float, default=4.0)
    p.add_argument("--max-area-ratio", type=float, default=0.4)
    p.add_argument("--max-aspect-ratio", type=float, default=5.0)
    p.add_argument("--merge-iou-lo", type=float, default=0.05)
    p.add_argument("--merge-iou-hi", type=float, default=0.3)
    p.add_argument("--centroid-dist", type=float, default=50.0)
    p.add_argument("--area-ratio-min", type=float, default=0.2)
    p.add_argument("--max-merge-iters", type=int, default=500)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _box_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def _centroid_dist(a: tuple, b: tuple) -> float:
    return math.sqrt(
        ((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) ** 2
        + ((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) ** 2
    )


def _union_box(a: tuple, b: tuple) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _size_compatible(mask_a, mask_b, area_ratio_min: float) -> bool:
    if mask_a is None or mask_b is None:
        return True
    aa, ab = int(mask_a.sum()), int(mask_b.sum())
    if aa == 0 or ab == 0:
        return False
    return area_ratio_min < min(aa, ab) / max(aa, ab) < 1.0


def _size_bucket(w: float, h: float) -> str:
    area = w * h
    if area < 1024:
        return "S"
    if area < 9216:
        return "M"
    return "L"


# ---------------------------------------------------------------------------
# Merge loop with origin tracking
# ---------------------------------------------------------------------------

def _run_merge_loop_tracked(candidates: list, args) -> list:
    """Multi-merge-per-pass loop; each candidate carries a 'merged' bool.

    Raw candidates enter with merged=False. Any candidate produced by a merge
    event gets merged=True. (A merged candidate that gets merged again stays
    merged=True.)
    """
    for _pass in range(args.max_merge_iters):
        candidates.sort(key=lambda c: c["conf"], reverse=True)
        n = len(candidates)

        dilated = [
            binary_dilation(c["mask"], structure=_DILATION_STRUCT)
            if c["mask"] is not None else None
            for c in candidates
        ]

        consumed: set = set()
        new_merged: list = []

        for i in range(n):
            if i in consumed:
                continue
            for j in range(i + 1, n):
                if j in consumed:
                    continue
                box_a, box_b = candidates[i]["box"], candidates[j]["box"]
                mask_a, mask_b = candidates[i]["mask"], candidates[j]["mask"]
                dil_a, dil_b = dilated[i], dilated[j]

                iou = _box_iou(box_a, box_b)
                iou_ok = args.merge_iou_lo <= iou <= args.merge_iou_hi
                cd_ok = _centroid_dist(box_a, box_b) < args.centroid_dist
                bt = False
                if dil_a is not None and mask_b is not None:
                    bt = bool((dil_a & mask_b).any())
                if not bt and dil_b is not None and mask_a is not None:
                    bt = bool((dil_b & mask_a).any())

                if not (iou_ok or cd_ok or bt):
                    continue
                if not _size_compatible(mask_a, mask_b, args.area_ratio_min):
                    continue

                merged_mask = (
                    (mask_a | mask_b)
                    if (mask_a is not None and mask_b is not None)
                    else None
                )
                new_merged.append({
                    "box": _union_box(box_a, box_b),
                    "conf": max(candidates[i]["conf"], candidates[j]["conf"]),
                    "mask": merged_mask,
                    "merged": True,
                })
                consumed.add(i)
                consumed.add(j)
                break

        if not new_merged:
            break

        candidates = [c for k, c in enumerate(candidates) if k not in consumed]
        candidates.extend(new_merged)

    return candidates


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def _apply_filters_and_topk(candidates: list, iw: int, ih: int, args) -> list:
    """Geometric filters + top-K; returns dicts with size_bucket added."""
    image_area = iw * ih
    filtered = []
    for c in candidates:
        x1, y1, x2, y2 = c["box"]
        bw, bh = x2 - x1, y2 - y1
        if bw < args.min_box_side or bh < args.min_box_side:
            continue
        if bw * bh > args.max_area_ratio * image_area:
            continue
        if max(bw / bh, bh / bw) > args.max_aspect_ratio:
            continue
        filtered.append(c)
    filtered.sort(key=lambda c: c["conf"], reverse=True)
    kept = filtered[: args.top_k]
    for c in kept:
        x1, y1, x2, y2 = c["box"]
        c["size_bucket"] = _size_bucket(x2 - x1, y2 - y1)
    return kept


def process_image(model, img_path: str, iw: int, ih: int, args) -> tuple:
    """Single FastSAM call → (baseline_top5, emr_top5).

    baseline_top5: raw → filters → top-K (no merge)
    emr_top5:      raw → merge loop → filters → top-K
    Both lists contain dicts: {box, conf, merged, size_bucket}
    """
    results = model(
        img_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        retina_masks=True,
        verbose=False,
    )

    if not results or results[0].boxes is None:
        return [], []

    res = results[0]

    masks_np = None
    if res.masks is not None:
        mt = res.masks.data
        mH, mW = int(mt.shape[1]), int(mt.shape[2])
        if mH != ih or mW != iw:
            import torch.nn.functional as F  # noqa: PLC0415
            mt = F.interpolate(
                mt.unsqueeze(0).float(), size=(ih, iw), mode="nearest"
            ).squeeze(0)
        masks_np = mt.cpu().numpy() > 0.5

    raw: list = []
    for idx, box in enumerate(res.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        mask = masks_np[idx] if masks_np is not None else None
        raw.append({"box": (x1, y1, x2, y2), "conf": conf, "mask": mask, "merged": False})

    # Baseline: filters + top-K on raw candidates, no merge.
    # Shallow-copy list so merge loop below doesn't see filter-mutated size_bucket.
    baseline_top5 = _apply_filters_and_topk(list(raw), iw, ih, args)

    # EMR: merge loop on a fresh list reference, then filters + top-K.
    emr_candidates = _run_merge_loop_tracked(list(raw), args)
    emr_top5 = _apply_filters_and_topk(emr_candidates, iw, ih, args)

    return baseline_top5, emr_top5


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_emr_box(emr_box: dict, baseline_boxes: list) -> tuple:
    """Return (tag, best_iou) where tag is 'identical' | 'merged' | 'reordered'."""
    best_iou = max(
        (_box_iou(emr_box["box"], b["box"]) for b in baseline_boxes),
        default=0.0,
    )
    if best_iou >= _IOU_MATCH_THR:
        return "identical", best_iou
    if emr_box["merged"]:
        return "merged", best_iou
    return "reordered", best_iou


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# BGR colors — viewer (PNG) sees RGB
_COLORS = {
    "identical":  (0, 200, 0),     # green
    "merged":     (0, 165, 255),   # orange
    "reordered":  (255, 60, 60),   # blue
    "baseline":   (0, 200, 0),     # green
}
_TAG_LABELS = {
    "identical": "=",
    "merged":    "M",
    "reordered": "R",
    "baseline":  "BL",
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_FONT_THICK = 1


def _draw_panel(img: np.ndarray, boxes_tagged: list, title: str) -> np.ndarray:
    """Draw annotated boxes on a copy of img.

    boxes_tagged: list of (box_xyxy, tag, rank, size_bucket, conf_or_None)
    """
    panel = img.copy()
    for box, tag, rank, size_bucket, conf in boxes_tagged:
        x1, y1, x2, y2 = (int(v) for v in box)
        color = _COLORS.get(tag, (200, 200, 200))
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)

        tag_char = _TAG_LABELS.get(tag, tag)
        label = f"{tag_char}{rank} {size_bucket}"
        if conf is not None:
            label += f" {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
        # Place label above the box top; flip below if it would clip the image
        label_y = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 4
        lx = max(0, x1)
        # Filled background rectangle
        cv2.rectangle(panel, (lx, label_y - th - 2), (lx + tw + 4, label_y + 2), color, -1)
        cv2.putText(panel, label, (lx + 2, label_y), _FONT, _FONT_SCALE,
                    (0, 0, 0), _FONT_THICK, cv2.LINE_AA)

    # Title with drop-shadow
    cv2.putText(panel, title, (8, 24), _FONT, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(panel, title, (8, 24), _FONT, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def _make_legend(width: int) -> np.ndarray:
    h = 26
    legend = np.zeros((h, width, 3), dtype=np.uint8)
    items = [
        (5,   (0, 200, 0),   "= identical"),
        (155, (0, 165, 255), "M merged"),
        (280, (255, 60, 60), "R reordered"),
        (390, (200, 200, 200), f"IoU match thr={_IOU_MATCH_THR}"),
    ]
    for x, color, text in items:
        cv2.putText(legend, text, (x, 18), _FONT, 0.5, color, 1, cv2.LINE_AA)
    return legend


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    from ultralytics import FastSAM  # noqa: PLC0415

    with open(args.coco_ann) as f:
        data = json.load(f)
    images = [img for img in data["images"] if img["id"] in _DIAG_IMAGE_IDS]
    log.info(f"Loaded {len(images)} diagnostic images")

    model = FastSAM(args.fastsam_weights)
    log.info(f"FastSAM loaded. conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    log.info(
        f"EMR: merge_iou=[{args.merge_iou_lo},{args.merge_iou_hi}] "
        f"centroid_dist={args.centroid_dist}px "
        f"area_ratio_min={args.area_ratio_min}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Aggregate counters
    agg = {"identical": 0, "merged": 0, "reordered": 0}

    print()
    print("=" * 72)
    print("EMR STEP 3 — VISUAL DIAGNOSTIC")
    print(f"  identical threshold : IoU ≥ {_IOU_MATCH_THR}")
    print(f"  size buckets        : S<1024px²  M<9216px²  L≥9216px²")
    print("=" * 72)

    for img_info in images:
        image_id = img_info["id"]
        img_path = os.path.join(
            args.coco_root, "Images", args.split, img_info["file_name"]
        )
        iw, ih = img_info["width"], img_info["height"]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            log.warning(f"Could not read {img_path} — skipping")
            continue

        baseline_top5, emr_top5 = process_image(model, img_path, iw, ih, args)

        # Classify each EMR box
        classified: list = []  # (box, tag, rank, size_bucket, conf, best_iou)
        counts = {"identical": 0, "merged": 0, "reordered": 0}
        for rank, eb in enumerate(emr_top5, 1):
            tag, best_iou = classify_emr_box(eb, baseline_top5)
            classified.append((eb["box"], tag, rank, eb["size_bucket"], eb["conf"], best_iou))
            counts[tag] += 1
            agg[tag] += 1

        # --- Text table ---
        print()
        print(f"img {image_id:>7}  {img_info['file_name']}")
        print(f"  baseline_n={len(baseline_top5)}  emr_n={len(emr_top5)}")
        print(
            f"  identical={counts['identical']}  "
            f"merged={counts['merged']}  "
            f"reordered={counts['reordered']}"
        )
        print(
            f"  {'rank':>4} | {'origin':>10} | {'sz':>2} | "
            f"{'conf':>5} | {'best_bl_iou':>11} | box_area_px²"
        )
        print(f"  {'----':>4}-+-{'----------':>10}-+-{'--':>2}-+-"
              f"{'-----':>5}-+-{'--------':>11}-+")
        for box, tag, rank, sz, conf, best_iou in classified:
            x1, y1, x2, y2 = box
            area = int((x2 - x1) * (y2 - y1))
            print(
                f"  {rank:>4} | {tag:>10} | {sz:>2} | "
                f"{conf:>5.3f} | {best_iou:>11.3f} | {area}"
            )

        # --- PNG ---
        bl_tagged = [
            (c["box"], "baseline", r, c["size_bucket"], c["conf"])
            for r, c in enumerate(baseline_top5, 1)
        ]
        emr_tagged = [
            (box, tag, rank, sz, conf)
            for box, tag, rank, sz, conf, _ in classified
        ]

        bl_panel = _draw_panel(img_bgr, bl_tagged,
                               f"BASELINE  n={len(baseline_top5)}")
        emr_panel = _draw_panel(img_bgr, emr_tagged,
                                f"EMR  n={len(emr_top5)}")

        side_by_side = np.concatenate([bl_panel, emr_panel], axis=1)
        legend = _make_legend(side_by_side.shape[1])
        panel = np.concatenate([side_by_side, legend], axis=0)

        out_path = os.path.join(args.output_dir, f"{image_id}.png")
        cv2.imwrite(out_path, panel)
        log.info(f"  Saved {out_path}")

    # --- Aggregate summary ---
    total = sum(agg.values())
    print()
    print("=" * 72)
    print("AGGREGATE — all 12 images × top-5 slots")
    print(f"  {'tag':>10} | {'count':>5} | {'pct':>6}")
    print(f"  {'----------':>10}-+-{'-----':>5}-+-{'------':>6}")
    for tag in ("identical", "merged", "reordered"):
        pct = 100 * agg[tag] / total if total else 0
        print(f"  {tag:>10} | {agg[tag]:>5} | {pct:>5.1f}%")
    print(f"  {'total':>10} | {total:>5} |")
    print("=" * 72)


if __name__ == "__main__":
    main()
