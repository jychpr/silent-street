"""
Precompute FastSAM pseudo-labels with EMR (Entity-level Mask Refinement).

EMR merges adjacent part-fragment masks into entity-level boxes before top-K
selection, targeting the part-fragmentation failure mode in FastSAM K=5 output.

Pipeline per image:
  FastSAM(iou=0.7, conf=0.1) raw masks
  → EMR merge loop  (greedy, max --max-merge-iters passes)
  → geometric filters (min_box_side, max_area_ratio, max_aspect_ratio)
  → top-K=5 by confidence
  → write JSON

Output schema: identical to OW_COCO_FASTSAM_K5_conf010_filtered.json
  {bbox:[x,y,w,h], weight:float, image_id:int, id:int, category_id:-1, pseudo:1}
"""

import argparse
import json
import logging
import math
import os
import random
import time

import numpy as np
from scipy.ndimage import binary_dilation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_ID_START = 970_000_000_000
_DIAG_IMAGE_IDS = frozenset({
    157105, 122263, 543882, 579329, 443084, 92869,
    475808, 435091, 171270, 307238, 30, 34,
})
_DILATION_STRUCT = np.ones((3, 3), dtype=bool)


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute FastSAM + EMR pseudo-labels in OV-DQUO OW_COCO format"
    )
    # --- shared with precompute_fastsam_pseudo_labels.py ---
    p.add_argument("--coco-root", required=True,
                   help="COCO root dir (contains Images/ and Annotations/)")
    p.add_argument("--split", default="train2017",
                   choices=["train2017", "val2017"])
    p.add_argument("--coco-ann", required=True,
                   help="Path to COCO instances annotation JSON")
    p.add_argument("--fastsam-weights", default="FastSAM-x.pt")
    p.add_argument("--output", required=True,
                   help="Output JSON path")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--top-k", type=int, default=5,
                   help="Max proposals per image after merging + filtering (default 5)")
    p.add_argument("--weight-mode", default="constant",
                   choices=["constant", "conf"])
    p.add_argument("--min-box-side", type=float, default=4.0)
    p.add_argument("--max-area-ratio", type=float, default=0.4)
    p.add_argument("--max-aspect-ratio", type=float, default=5.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-images", type=int, default=None,
                   help="Subsample to N images (smoke tests only)")
    # --- EMR-specific ---
    p.add_argument("--merge-iou-lo", type=float, default=0.05)
    p.add_argument("--merge-iou-hi", type=float, default=0.3)
    p.add_argument("--centroid-dist", type=float, default=50.0,
                   help="Centroid-distance adjacency threshold in px (default 50.0)")
    p.add_argument("--area-ratio-min", type=float, default=0.2)
    p.add_argument("--max-merge-iters", type=int, default=500,
                   help="Hard cap on merge passes per image — guard only, not throttle (default 500)")
    p.add_argument("--diag-mode", action="store_true",
                   help="Process only the 12 diagnostic image IDs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _box_iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _centroid_dist(a: tuple, b: tuple) -> float:
    cx_a, cy_a = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    cx_b, cy_b = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)


def _union_box(box_a: tuple, box_b: tuple) -> tuple:
    return (
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    )


def _size_compatible(mask_a, mask_b, area_ratio_min: float) -> bool:
    """True if the smaller mask is not less than area_ratio_min of the larger."""
    if mask_a is None or mask_b is None:
        return True  # conservative: assume compatible when masks unavailable
    area_a, area_b = int(mask_a.sum()), int(mask_b.sum())
    if area_a == 0 or area_b == 0:
        return False
    ratio = min(area_a, area_b) / max(area_a, area_b)
    return area_ratio_min < ratio < 1.0


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list:
    return [x1, y1, x2 - x1, y2 - y1]


# ---------------------------------------------------------------------------
# EMR merge loop
# ---------------------------------------------------------------------------

def _run_merge_loop(candidates: list, args) -> list:
    """Iteratively merge adjacent part-fragment pairs.

    Each pass collects ALL non-conflicting merges in one scan (greedy,
    descending-confidence order). Consumed indices are skipped so each
    candidate participates in at most one merge per pass. Passes repeat until
    a full scan yields zero new merges (convergence) or the hard cap fires.
    This converges in O(log N) passes for typical fragment trees.

    candidates: list of dicts {"box": (x1,y1,x2,y2), "conf": float, "mask": ndarray|None}
    Returns the same structure after merging.
    """
    for _pass in range(args.max_merge_iters):
        candidates.sort(key=lambda c: c["conf"], reverse=True)
        n = len(candidates)

        # Precompute dilated masks once per pass.
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

                # --- Adjacency (any one) ---
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

                # --- Size gate ---
                if not _size_compatible(mask_a, mask_b, args.area_ratio_min):
                    continue

                # --- Queue merge; mark both consumed for this pass ---
                merged_mask = (mask_a | mask_b) if (mask_a is not None and mask_b is not None) else None
                new_merged.append({
                    "box": _union_box(box_a, box_b),
                    "conf": max(candidates[i]["conf"], candidates[j]["conf"]),
                    "mask": merged_mask,
                })
                consumed.add(i)
                consumed.add(j)
                break  # i consumed; move to next i

        if not new_merged:
            break  # zero-merge pass — converged

        candidates = [c for k, c in enumerate(candidates) if k not in consumed]
        candidates.extend(new_merged)

    return candidates


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(model, img_path: str, iw: int, ih: int, args) -> tuple:
    """Run FastSAM, EMR merge loop, geometric filters, top-K.

    Returns (proposals, stats).
    proposals: list of ([x, y, w, h], weight) tuples.
    stats keys: raw, merge_events, post_merge, drop_min_side, drop_area,
                drop_aspect, surviving, kept.
    """
    stats = {
        "raw": 0, "merge_events": 0, "post_merge": 0,
        "drop_min_side": 0, "drop_area": 0, "drop_aspect": 0,
        "surviving": 0, "kept": 0,
    }

    # --- FastSAM inference (identical call to precompute_fastsam_pseudo_labels.py) ---
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
        return [], stats

    res = results[0]
    image_area = iw * ih

    # Build mask array at original image resolution
    masks_np = None
    if res.masks is not None:
        mt = res.masks.data  # (N, H_m, W_m)
        mH, mW = int(mt.shape[1]), int(mt.shape[2])
        if mH != ih or mW != iw:
            import torch.nn.functional as F  # noqa: PLC0415
            mt = F.interpolate(
                mt.unsqueeze(0).float(), size=(ih, iw), mode="nearest"
            ).squeeze(0)
        masks_np = mt.cpu().numpy() > 0.5  # (N, ih, iw) bool

    candidates = []
    for idx, box in enumerate(res.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        stats["raw"] += 1
        mask = masks_np[idx] if masks_np is not None else None
        candidates.append({"box": (x1, y1, x2, y2), "conf": conf, "mask": mask})

    # --- EMR merge loop ---
    pre_merge_n = len(candidates)
    candidates = _run_merge_loop(candidates, args)
    stats["merge_events"] = pre_merge_n - len(candidates)
    stats["post_merge"] = len(candidates)

    # --- Geometric filters (same thresholds as precompute_fastsam_pseudo_labels.py) ---
    filtered = []
    for c in candidates:
        x1, y1, x2, y2 = c["box"]
        bw, bh = x2 - x1, y2 - y1
        if bw < args.min_box_side or bh < args.min_box_side:
            stats["drop_min_side"] += 1
            continue
        if bw * bh > args.max_area_ratio * image_area:
            stats["drop_area"] += 1
            continue
        if max(bw / bh, bh / bw) > args.max_aspect_ratio:
            stats["drop_aspect"] += 1
            continue
        filtered.append(c)

    stats["surviving"] = len(filtered)
    filtered.sort(key=lambda c: c["conf"], reverse=True)
    kept = filtered[: args.top_k]
    stats["kept"] = len(kept)

    out = []
    for c in kept:
        x1, y1, x2, y2 = c["box"]
        xywh = _xyxy_to_xywh(x1, y1, x2, y2)
        weight = 1.0 if args.weight_mode == "constant" else c["conf"]
        out.append((xywh, weight))
    return out, stats


# ---------------------------------------------------------------------------
# Reporting helpers (mirrored from precompute_fastsam_pseudo_labels.py)
# ---------------------------------------------------------------------------

def _format_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def print_summary(annotations: list, images_processed: int, agg_stats: dict) -> None:
    total_boxes = len(annotations)
    boxes_per_image: dict = {}
    for ann in annotations:
        iid = ann["image_id"]
        boxes_per_image[iid] = boxes_per_image.get(iid, 0) + 1

    counts = sorted(boxes_per_image.values())
    n = len(counts)

    def pct(p: float) -> float:
        if not counts:
            return 0.0
        idx = max(0, min(n - 1, math.ceil(p / 100 * n) - 1))
        return counts[idx]

    mean_count = total_boxes / images_processed if images_processed else 0
    mean_merges = agg_stats["merge_events"] / max(images_processed, 1)

    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info(f"  Images processed       : {images_processed}")
    log.info(f"  Images with ≥1 box     : {len(counts)}")
    log.info(f"  Total boxes written    : {total_boxes}")
    log.info(f"  Boxes/image (all imgs) : mean={mean_count:.2f}")
    log.info(f"  Boxes/image (covered)  : median={pct(50)}, p90={pct(90)}, max={counts[-1] if counts else 0}")
    log.info(f"  Total merge events     : {agg_stats['merge_events']} (mean={mean_merges:.2f}/img)")
    log.info(f"  Total raw FastSAM masks: {agg_stats['raw']}")
    log.info("=" * 60)


def validate_sample(annotations: list, rng: random.Random) -> None:
    sample = rng.sample(annotations, min(5, len(annotations)))
    expected_keys = {"bbox", "weight", "image_id", "id", "category_id", "pseudo"}
    anomalies = []
    for i, ann in enumerate(sample):
        if set(ann.keys()) != expected_keys:
            anomalies.append(f"entry {i}: unexpected keys {set(ann.keys())}")
            continue
        bbox = ann["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            anomalies.append(f"entry {i}: bbox not a 4-list")
        elif not (bbox[0] >= 0 and bbox[1] >= 0 and bbox[2] > 0 and bbox[3] > 0):
            anomalies.append(f"entry {i}: bbox out of range {bbox}")
        if not (0.0 <= ann["weight"] <= 1.0):
            anomalies.append(f"entry {i}: weight out of [0,1]")
        if not isinstance(ann["image_id"], int):
            anomalies.append(f"entry {i}: image_id not int")
        if ann["category_id"] != -1:
            anomalies.append(f"entry {i}: category_id != -1")
        if ann["pseudo"] != 1:
            anomalies.append(f"entry {i}: pseudo != 1")
    log.info(f"Validation: sampled {len(sample)} entries")
    if anomalies:
        for a in anomalies:
            log.warning(f"  ANOMALY: {a}")
    else:
        log.info("  All sampled entries pass schema + value checks")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = random.Random(args.seed)

    from ultralytics import FastSAM  # noqa: PLC0415 — deferred so --help works without GPU

    log.info(f"Loading image list from {args.coco_ann}")
    with open(args.coco_ann) as f:
        data = json.load(f)
    images = data["images"]
    log.info(f"  {len(images)} images in annotation file")

    if args.diag_mode:
        images = [img for img in images if img["id"] in _DIAG_IMAGE_IDS]
        log.info(f"  --diag-mode: restricted to {len(images)} diagnostic images")

    if args.num_images is not None:
        images = rng.sample(images, min(args.num_images, len(images)))
        log.info(f"  Subsampled to {len(images)} images (--num-images {args.num_images})")

    model = FastSAM(args.fastsam_weights)
    log.info(
        f"FastSAM loaded. conf={args.conf} iou={args.iou} imgsz={args.imgsz} "
        f"top_k={args.top_k} weight_mode={args.weight_mode}"
    )
    log.info(
        f"EMR: merge_iou=[{args.merge_iou_lo},{args.merge_iou_hi}] "
        f"centroid_dist={args.centroid_dist}px "
        f"area_ratio_min={args.area_ratio_min} max_iters={args.max_merge_iters}"
    )

    images_dir = os.path.join(args.coco_root, "Images", args.split)
    annotations: list = []
    id_counter = _ID_START
    total_images = len(images)
    t_start = time.time()
    agg_stats = {
        "raw": 0, "merge_events": 0, "post_merge": 0,
        "drop_min_side": 0, "drop_area": 0, "drop_aspect": 0,
        "surviving": 0, "kept": 0,
    }

    for i, img_info in enumerate(images):
        image_id = img_info["id"]
        img_path = os.path.join(images_dir, img_info["file_name"])
        iw, ih = img_info["width"], img_info["height"]

        t0 = time.time()
        proposals, img_stats = process_image(model, img_path, iw, ih, args)
        elapsed = time.time() - t0

        for k in agg_stats:
            agg_stats[k] += img_stats[k]

        for xywh, weight in proposals:
            annotations.append({
                "bbox":        [round(v, 4) for v in xywh],
                "weight":      round(weight, 8),
                "image_id":    image_id,
                "id":          id_counter,
                "category_id": -1,
                "pseudo":      1,
            })
            id_counter += 1

        imgs_done = i + 1
        log_this = (
            args.diag_mode
            or total_images <= 100
            or imgs_done % 1000 == 0
            or imgs_done == total_images
        )
        if log_this:
            rate = imgs_done / max(time.time() - t_start, 1e-6)
            eta = (total_images - imgs_done) / rate if rate > 0 else 0
            log.info(
                f"  [{imgs_done}/{total_images}] img {image_id:>7}:  "
                f"raw={img_stats['raw']:>3}  "
                f"merges={img_stats['merge_events']:>3}  "
                f"post_merge={img_stats['post_merge']:>3}  "
                f"kept={img_stats['kept']:>2}  "
                f"t={elapsed:.1f}s  ETA={_format_eta(eta)}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(annotations, f, separators=(",", ":"))
    log.info(f"Wrote {len(annotations)} annotations to {args.output}")

    print_summary(annotations, total_images, agg_stats)
    validate_sample(annotations, rng)


if __name__ == "__main__":
    main()
