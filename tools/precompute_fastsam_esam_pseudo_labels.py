"""
Precompute FastSAM pseudo-labels with E-SAM-adapted MMG-style adaptive NMS.

Module 1 of 2 — Adaptive NMS only. EMR split-then-merge is Module 2.

Produces a flat JSON array byte-compatible with OW_COCO_R*.json schema.

Output schema (one dict per bounding box):
  {
    "bbox":        [x, y, w, h],   # COCO xywh absolute pixels
    "weight":      float,
    "image_id":    int,
    "id":          int,             # globally unique within this file
    "category_id": -1,
    "pseudo":      1
  }

ID namespace: 960_000_000_000 (avoids collision with existing 950B file).
Reference: E-SAM (Zhang et al., arXiv:2503.12094), Section 3.2 (MMG).
"""

import argparse
import json
import logging
import math
import os
import random
import time

import numpy as np
from PIL import Image as PILImage
from skimage.segmentation import felzenszwalb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_ID_START = 960_000_000_000


def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute FastSAM pseudo-labels with E-SAM adaptive NMS"
    )
    p.add_argument("--coco-root", required=True,
                   help="COCO root dir (contains Images/ and Annotations/)")
    p.add_argument("--split", default="train2017",
                   choices=["train2017", "val2017"])
    p.add_argument("--coco-ann", required=True,
                   help="Path to instances annotation JSON")
    p.add_argument("--fastsam-weights", default="FastSAM-x.pt")
    p.add_argument("--output",
                   default="ow_labels/OW_COCO_FASTSAM_K5_ESAMadapt_v1.json")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--weight-mode", default="constant",
                   choices=["constant", "conf"])
    p.add_argument("--min-box-side", type=float, default=4.0)
    p.add_argument("--max-area-ratio", type=float, default=0.4)
    p.add_argument("--max-aspect-ratio", type=float, default=5.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-images", type=int, default=None,
                   help="Subsample to this many images (smoke tests only)")
    # Adaptive NMS flags (E-SAM MMG)
    p.add_argument("--use-adaptive-nms", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Toggle adaptive NMS post-processing (default: on)")
    p.add_argument("--felzenszwalb-scale", type=int, default=100,
                   help="Felzenszwalb scale parameter (default 100)")
    p.add_argument("--felzenszwalb-sigma", type=float, default=0.8,
                   help="Felzenszwalb sigma (default 0.8)")
    p.add_argument("--felzenszwalb-min-size", type=int, default=50,
                   help="Felzenszwalb min component size (default 50)")
    p.add_argument("--nms-thresh-low", type=float, default=0.5,
                   help="NMS threshold for high-density regions — more suppression (default 0.5)")
    p.add_argument("--nms-thresh-high", type=float, default=0.8,
                   help="NMS threshold for low-density regions — keep more (default 0.8)")
    p.add_argument("--density-bins", type=int, default=3,
                   help="Number of density bins for adaptive threshold mapping (default 3)")
    return p.parse_args()


def load_image_list(coco_ann_path: str) -> list[dict]:
    log.info(f"Loading image list from {coco_ann_path}")
    with open(coco_ann_path) as f:
        data = json.load(f)
    images = data["images"]
    log.info(f"  {len(images)} images in annotation file")
    return images


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    return [x1, y1, x2 - x1, y2 - y1]


def _mask_iou_nms(confs: np.ndarray, masks: np.ndarray, threshold: float) -> list[int]:
    """Greedy mask-IoU NMS. Returns list of kept local indices (original order)."""
    order = list(np.argsort(confs)[::-1])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        remaining = []
        for j in order:
            inter = int((masks[i] & masks[j]).sum())
            union = int((masks[i] | masks[j]).sum())
            iou = inter / union if union > 0 else 0.0
            if iou <= threshold:
                remaining.append(j)
        order = remaining
    return keep


def process_image_adaptive_nms(
    model, img_path: str, iw: int, ih: int, args
) -> tuple[list[tuple[list[float], float]], dict[str, int]]:
    """
    Run FastSAM on one image and return (proposals, filter_stats).

    With adaptive NMS enabled, applies per-density-bin mask-IoU NMS using
    Felzenszwalb superpixel density before top-K selection.

    filter_stats keys: raw, drop_min_side, drop_area, drop_aspect,
                       surviving, dropped_by_adaptive_nms, kept.
    """
    stats: dict[str, int] = {
        "raw": 0, "drop_min_side": 0, "drop_area": 0, "drop_aspect": 0,
        "surviving": 0, "dropped_by_adaptive_nms": 0, "kept": 0,
    }

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
    if res.masks is None:
        return [], stats

    image_area = iw * ih

    # Ensure masks are at original image resolution (retina_masks=True usually does this)
    masks_tensor = res.masks.data  # (N, H_m, W_m)
    mH, mW = int(masks_tensor.shape[1]), int(masks_tensor.shape[2])
    if mH != ih or mW != iw:
        import torch.nn.functional as F
        masks_tensor = F.interpolate(
            masks_tensor.unsqueeze(0).float(),
            size=(ih, iw),
            mode="nearest",
        ).squeeze(0)
    masks_all_np = (masks_tensor.cpu().numpy() > 0.5)  # (N, ih, iw) bool

    # Geometric filtering — keep track of original mask index
    candidates = []  # (x1, y1, x2, y2, conf, orig_idx)
    for idx, box in enumerate(res.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        bw = x2 - x1
        bh = y2 - y1
        stats["raw"] += 1
        if bw < args.min_box_side or bh < args.min_box_side:
            stats["drop_min_side"] += 1
            continue
        if bw * bh > args.max_area_ratio * image_area:
            stats["drop_area"] += 1
            continue
        if max(bw / bh, bh / bw) > args.max_aspect_ratio:
            stats["drop_aspect"] += 1
            continue
        candidates.append((x1, y1, x2, y2, conf, idx))

    stats["surviving"] = len(candidates)

    if not candidates:
        return [], stats

    # Extract masks for surviving candidates
    cand_orig_indices = [c[5] for c in candidates]
    masks_np = masks_all_np[cand_orig_indices]  # (M, ih, iw)
    confs = np.array([c[4] for c in candidates])

    if args.use_adaptive_nms and len(candidates) > 1:
        # Load image for Felzenszwalb (PIL always opens correct ih×iw)
        img_np = np.array(PILImage.open(img_path).convert("RGB"))

        # Felzenszwalb superpixel segmentation
        segments = felzenszwalb(
            img_np,
            scale=args.felzenszwalb_scale,
            sigma=args.felzenszwalb_sigma,
            min_size=args.felzenszwalb_min_size,
        )  # (ih, iw) int, each pixel gets a superpixel label

        # Build density map: for each superpixel, mean number of masks overlapping each pixel.
        # Vectorized via bincount on flattened arrays.
        pixel_mask_count = masks_np.sum(axis=0).astype(float)  # (ih, iw)
        flat_seg = segments.ravel()
        flat_pmc = pixel_mask_count.ravel()
        label_sums = np.bincount(flat_seg, weights=flat_pmc)
        label_counts = np.bincount(flat_seg)
        label_density = label_sums / np.maximum(label_counts, 1)
        density_map = label_density[flat_seg].reshape(segments.shape)

        # Per-mask mean density (average density-map value inside each mask)
        mask_densities = np.array([
            density_map[masks_np[i]].mean() if masks_np[i].any() else 0.0
            for i in range(len(masks_np))
        ])

        # Assign each mask to a density bin (0 = lowest, n_bins-1 = highest)
        n_bins = max(1, args.density_bins)
        d_min, d_max = mask_densities.min(), mask_densities.max()
        if d_max > d_min:
            raw_bins = (mask_densities - d_min) / (d_max - d_min) * n_bins
            bin_indices = np.floor(raw_bins).astype(int).clip(0, n_bins - 1)
        else:
            bin_indices = np.zeros(len(mask_densities), dtype=int)

        # NMS threshold per bin: linearly from nms_thresh_high (low density, bin 0)
        # to nms_thresh_low (high density, bin n_bins-1)
        if n_bins == 1:
            bin_thresholds = [args.nms_thresh_high]
        else:
            bin_thresholds = np.linspace(
                args.nms_thresh_high, args.nms_thresh_low, n_bins
            ).tolist()

        # Greedy mask-IoU NMS within each density bin
        kept_indices: list[int] = []
        for b in range(n_bins):
            in_bin = np.where(bin_indices == b)[0]
            if len(in_bin) == 0:
                continue
            local_keep = _mask_iou_nms(confs[in_bin], masks_np[in_bin], bin_thresholds[b])
            kept_indices.extend(in_bin[local_keep].tolist())

        stats["dropped_by_adaptive_nms"] = len(candidates) - len(kept_indices)
        candidates = [candidates[i] for i in kept_indices]

    # Top-K by confidence
    candidates.sort(key=lambda c: c[4], reverse=True)
    candidates = candidates[:args.top_k]
    stats["kept"] = len(candidates)

    out = []
    for x1, y1, x2, y2, conf, _ in candidates:
        xywh = _xyxy_to_xywh(x1, y1, x2, y2)
        weight = 1.0 if args.weight_mode == "constant" else conf
        out.append((xywh, weight))
    return out, stats


def _format_eta(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def print_summary(annotations: list[dict], images_processed: int) -> None:
    total_boxes = len(annotations)
    boxes_per_image: dict[int, int] = {}
    for ann in annotations:
        iid = ann["image_id"]
        boxes_per_image[iid] = boxes_per_image.get(iid, 0) + 1

    counts = sorted(boxes_per_image.values())
    images_with_boxes = len(counts)
    n = len(counts)

    def percentile(data: list[int], p: float) -> float:
        if not data:
            return 0.0
        idx = max(0, min(n - 1, math.ceil(p / 100 * n) - 1))
        return data[idx]

    mean_count = total_boxes / images_processed if images_processed else 0
    median_count = percentile(counts, 50)
    p90 = percentile(counts, 90)
    p99 = percentile(counts, 99)
    max_count = counts[-1] if counts else 0

    weights = [ann["weight"] for ann in annotations]
    w_mean = sum(weights) / len(weights) if weights else 0.0
    w_std = math.sqrt(sum((w - w_mean) ** 2 for w in weights) / len(weights)) if weights else 0.0

    bboxes = [ann["bbox"] for ann in annotations]
    mean_w = sum(b[2] for b in bboxes) / len(bboxes) if bboxes else 0.0
    mean_h = sum(b[3] for b in bboxes) / len(bboxes) if bboxes else 0.0

    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info(f"  Images processed       : {images_processed}")
    log.info(f"  Images with ≥1 box     : {images_with_boxes}")
    log.info(f"  Total boxes written    : {total_boxes}")
    log.info(f"  Boxes/image (all imgs) : mean={mean_count:.2f}")
    log.info(f"  Boxes/image (covered)  : median={median_count}, p90={p90}, p99={p99}, max={max_count}")
    log.info(f"  Weight                 : mean={w_mean:.4f}, std={w_std:.4f}")
    log.info(f"  BBox dims (mean)       : w={mean_w:.1f}px  h={mean_h:.1f}px")
    log.info("=" * 60)


def validate_sample(annotations: list[dict], rng: random.Random) -> None:
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
            anomalies.append(f"entry {i}: weight {ann['weight']} out of [0,1]")
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


def print_filter_stats(agg: dict[str, int], final_written: int) -> None:
    raw = agg["raw"]
    pct = lambda n: f"{100 * n / raw:.1f}%" if raw else "N/A"
    log.info("=" * 60)
    log.info("FILTER STATS")
    log.info(f"  Total raw FastSAM boxes      : {raw}")
    log.info(f"  Dropped by min-box-side      : {agg['drop_min_side']} ({pct(agg['drop_min_side'])})")
    log.info(f"  Dropped by max-area-ratio    : {agg['drop_area']} ({pct(agg['drop_area'])})")
    log.info(f"  Dropped by max-aspect-ratio  : {agg['drop_aspect']} ({pct(agg['drop_aspect'])})")
    log.info(f"  Surviving after filters      : {agg['surviving']}")
    log.info(f"  Dropped by adaptive NMS      : {agg['dropped_by_adaptive_nms']} ({pct(agg['dropped_by_adaptive_nms'])})")
    log.info(f"  Kept after top-K             : {agg['kept']}")
    log.info(f"  Final box count (written)    : {final_written}")
    log.info("=" * 60)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    from ultralytics import FastSAM  # noqa: PLC0415 — deferred so --help works without GPU

    images = load_image_list(args.coco_ann)

    if args.num_images is not None:
        images = rng.sample(images, min(args.num_images, len(images)))
        log.info(f"Subsampled to {len(images)} images (--num-images {args.num_images})")

    model = FastSAM(args.fastsam_weights)
    log.info(f"FastSAM loaded from {args.fastsam_weights}")
    log.info(
        f"Config: conf={args.conf} iou={args.iou} imgsz={args.imgsz} "
        f"top_k={args.top_k} weight_mode={args.weight_mode} "
        f"min_box_side={args.min_box_side} "
        f"max_area_ratio={args.max_area_ratio} max_aspect_ratio={args.max_aspect_ratio} "
        f"use_adaptive_nms={args.use_adaptive_nms}"
    )
    if args.use_adaptive_nms:
        log.info(
            f"Adaptive NMS: felzenszwalb_scale={args.felzenszwalb_scale} "
            f"sigma={args.felzenszwalb_sigma} min_size={args.felzenszwalb_min_size} "
            f"nms_thresh=[{args.nms_thresh_high} (low-density) → {args.nms_thresh_low} (high-density)] "
            f"density_bins={args.density_bins}"
        )

    images_dir = os.path.join(args.coco_root, "Images", args.split)
    annotations: list[dict] = []
    id_counter = _ID_START
    total_images = len(images)
    t_start = time.time()
    agg_stats: dict[str, int] = {
        "raw": 0, "drop_min_side": 0, "drop_area": 0, "drop_aspect": 0,
        "surviving": 0, "dropped_by_adaptive_nms": 0, "kept": 0,
    }

    for i, img_info in enumerate(images):
        image_id = img_info["id"]
        img_path = os.path.join(images_dir, img_info["file_name"])
        iw, ih = img_info["width"], img_info["height"]

        proposals, img_stats = process_image_adaptive_nms(model, img_path, iw, ih, args)
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

        if (i + 1) % 1000 == 0 or (i + 1) == total_images:
            elapsed = time.time() - t_start
            imgs_done = i + 1
            density = len(annotations) / imgs_done
            rate = imgs_done / elapsed
            eta = (total_images - imgs_done) / rate if rate > 0 else 0
            log.info(
                f"[{imgs_done}/{total_images}] "
                f"boxes={len(annotations)} density={density:.2f} boxes/img "
                f"elapsed={_format_eta(elapsed)} ETA={_format_eta(eta)}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(annotations, f, separators=(",", ":"))
    log.info(f"Wrote {len(annotations)} annotations to {args.output}")

    print_filter_stats(agg_stats, len(annotations))
    print_summary(annotations, total_images)
    validate_sample(annotations, rng)


if __name__ == "__main__":
    main()
