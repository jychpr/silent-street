"""
EMR Step 0 — Pre-implementation sanity check.

Counts mergeable mask pairs at FastSAM iou=0.7 across the 12 diagnostic images.
Does NOT implement any merge logic — purely counting.

Adjacency (any one of):
  1. bbox-pair IoU in [merge_iou_lo, merge_iou_hi]
  2. centroid_distance < centroid_dist px  (swept over multiple thresholds)
  3. dilated mask_a overlaps mask_b  (captures boundary-touching adjacent fragments)

Size (AND with adjacency):
  area_ratio_min < mask_area_smaller / mask_area_larger < 1.0

Confidence gate: guaranteed by FastSAM conf=0.1 threshold.

Kill condition: mean mergeable pairs/image < 5 at the 150px threshold → do not proceed to Step 1.
Saves: diagnostics/output/emr_step0/step0_report.json + stdout summary.
"""

import argparse
import json
import logging
import math
import os
import time

import numpy as np
from scipy.ndimage import binary_dilation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DILATION_STRUCT = np.ones((3, 3), dtype=bool)


def parse_args():
    p = argparse.ArgumentParser(description="EMR Step 0 — count mergeable mask pairs")
    p.add_argument("--coco-root", required=True,
                   help="COCO root dir (contains Images/ and Annotations/)")
    p.add_argument("--split", default="train2017",
                   choices=["train2017", "val2017"])
    p.add_argument("--coco-ann", required=True,
                   help="Path to 12-image diagnostic annotation JSON")
    p.add_argument("--fastsam-weights", default="FastSAM-x.pt",
                   help="Path to FastSAM-x.pt")
    p.add_argument("--output-dir", default="diagnostics/output/emr_step0",
                   help="Directory for step0_report.json")
    p.add_argument("--iou", type=float, default=0.7,
                   help="FastSAM NMS IoU threshold (default 0.7)")
    p.add_argument("--conf", type=float, default=0.1,
                   help="FastSAM confidence threshold (default 0.1)")
    p.add_argument("--imgsz", type=int, default=1024,
                   help="FastSAM input image size (default 1024)")
    p.add_argument("--device", default="cuda:0")
    # Geometric filters — mirror precompute_fastsam_pseudo_labels.py exactly
    p.add_argument("--min-box-side", type=float, default=4.0)
    p.add_argument("--max-area-ratio", type=float, default=0.4)
    p.add_argument("--max-aspect-ratio", type=float, default=5.0)
    # EMR adjacency / size thresholds
    p.add_argument("--merge-iou-lo", type=float, default=0.05,
                   help="Lower bound of bbox-IoU adjacency window (default 0.05)")
    p.add_argument("--merge-iou-hi", type=float, default=0.3,
                   help="Upper bound of bbox-IoU adjacency window (default 0.3)")
    p.add_argument("--centroid-dist", type=float, nargs="+",
                   default=[50.0, 100.0, 150.0],
                   help="Centroid-distance thresholds in pixels; one or more values (default: 50 100 150)")
    p.add_argument("--area-ratio-min", type=float, default=0.2,
                   help="Min mask area ratio for size compatibility (default 0.2)")
    return p.parse_args()


def _box_iou(a: tuple, b: tuple) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _centroid_dist(a: tuple, b: tuple) -> float:
    """Euclidean distance between centroids of two (x1, y1, x2, y2) boxes."""
    cx_a = (a[0] + a[2]) / 2
    cy_a = (a[1] + a[3]) / 2
    cx_b = (b[0] + b[2]) / 2
    cy_b = (b[1] + b[3]) / 2
    return math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)


def _pair_adj_flags(
    box_a: tuple,
    box_b: tuple,
    dil_a: "np.ndarray | None",
    mask_b: "np.ndarray | None",
    dil_b: "np.ndarray | None",
    mask_a: "np.ndarray | None",
    args,
) -> tuple:
    """Return (iou_ok, centroid_dist_px, boundary_touch) for a pair.

    Separating raw flags from the threshold comparison lets the centroid
    threshold vary across the sweep without re-running IoU or boundary checks.
    """
    iou = _box_iou(box_a, box_b)
    iou_ok = args.merge_iou_lo <= iou <= args.merge_iou_hi
    cd = _centroid_dist(box_a, box_b)
    bt = False
    if dil_a is not None and mask_b is not None:
        bt = bool((dil_a & mask_b).any())
    if not bt and dil_b is not None and mask_a is not None:
        bt = bool((dil_b & mask_a).any())
    return iou_ok, cd, bt


def _size_compatible(
    mask_a: "np.ndarray | None",
    mask_b: "np.ndarray | None",
    args,
) -> bool:
    """True if the smaller mask is not less than area_ratio_min of the larger."""
    if mask_a is None or mask_b is None:
        return True  # conservative: assume compatible when masks unavailable
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    if area_a == 0 or area_b == 0:
        return False
    ratio = min(area_a, area_b) / max(area_a, area_b)
    return args.area_ratio_min < ratio < 1.0


def analyze_image(model, img_info: dict, args) -> dict:
    """Run FastSAM, apply geometric filters, count adj/mergeable pairs per threshold."""
    image_id = img_info["id"]
    img_path = os.path.join(
        args.coco_root, "Images", args.split, img_info["file_name"]
    )
    iw, ih = img_info["width"], img_info["height"]
    image_area = iw * ih

    results = model(
        img_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        retina_masks=True,
        verbose=False,
    )

    raw_count = 0
    # Each candidate: (x1, y1, x2, y2, conf, mask_or_None)
    candidates: list[tuple] = []

    if results and results[0].boxes is not None:
        res = results[0]

        # Build mask array at original image resolution
        masks_np = None
        if res.masks is not None:
            mt = res.masks.data  # (N, H_m, W_m)
            mH, mW = int(mt.shape[1]), int(mt.shape[2])
            if mH != ih or mW != iw:
                import torch.nn.functional as F  # noqa: PLC0415
                mt = F.interpolate(
                    mt.unsqueeze(0).float(),
                    size=(ih, iw),
                    mode="nearest",
                ).squeeze(0)
            masks_np = mt.cpu().numpy() > 0.5  # (N, ih, iw) bool

        for idx, box in enumerate(res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            bw, bh = x2 - x1, y2 - y1
            raw_count += 1
            # Geometric filters — match precompute_fastsam_pseudo_labels.py exactly
            if bw < args.min_box_side or bh < args.min_box_side:
                continue
            if bw * bh > args.max_area_ratio * image_area:
                continue
            if max(bw / bh, bh / bw) > args.max_aspect_ratio:
                continue
            mask = masks_np[idx] if masks_np is not None else None
            candidates.append((x1, y1, x2, y2, conf, mask))

    n = len(candidates)
    total_pairs = n * (n - 1) // 2

    # Precompute dilated masks once per candidate
    dilated: list = [
        binary_dilation(c[5], structure=_DILATION_STRUCT) if c[5] is not None else None
        for c in candidates
    ]

    by_threshold: dict[str, dict] = {
        str(int(t)): {"adj_pairs": 0, "mergeable_pairs": 0}
        for t in args.centroid_dist
    }

    for i in range(n):
        for j in range(i + 1, n):
            box_a = candidates[i][:4]
            box_b = candidates[j][:4]
            mask_a, mask_b = candidates[i][5], candidates[j][5]
            dil_a, dil_b = dilated[i], dilated[j]

            iou_ok, cd, bt = _pair_adj_flags(
                box_a, box_b, dil_a, mask_b, dil_b, mask_a, args
            )
            size_ok = _size_compatible(mask_a, mask_b, args)

            for t in args.centroid_dist:
                t_key = str(int(t))
                if iou_ok or (cd < t) or bt:
                    by_threshold[t_key]["adj_pairs"] += 1
                    if size_ok:
                        by_threshold[t_key]["mergeable_pairs"] += 1

    return {
        "image_id": image_id,
        "raw_masks": raw_count,
        "masks_after_filter": n,
        "total_pairs": total_pairs,
        "by_threshold": by_threshold,
    }


def main():
    args = parse_args()
    from ultralytics import FastSAM  # noqa: PLC0415 — deferred so --help works without GPU

    with open(args.coco_ann) as f:
        data = json.load(f)
    images = data["images"]
    log.info(f"Loaded {len(images)} images from {args.coco_ann}")

    model = FastSAM(args.fastsam_weights)
    log.info(
        f"FastSAM loaded. iou={args.iou} conf={args.conf} imgsz={args.imgsz}"
    )
    sweep_str = ", ".join(f"{int(t)}px" for t in sorted(args.centroid_dist))
    log.info(
        f"Adjacency: bbox_iou=[{args.merge_iou_lo},{args.merge_iou_hi}]  "
        f"centroid_dist sweep=[{sweep_str}]  boundary_touch=3x3_dilation"
    )
    log.info(f"Size: area_ratio_min={args.area_ratio_min}")

    t_start = time.time()
    per_image = []
    min_t_key = str(int(min(args.centroid_dist)))
    max_t_key = str(int(max(args.centroid_dist)))

    for img_info in images:
        t0 = time.time()
        stats = analyze_image(model, img_info, args)
        elapsed = time.time() - t0
        stats["wall_seconds"] = round(elapsed, 2)
        per_image.append(stats)
        log.info(
            f"  img {stats['image_id']:>7}:  "
            f"raw={stats['raw_masks']:>3}  "
            f"filt={stats['masks_after_filter']:>3}  "
            f"pairs={stats['total_pairs']:>4}  "
            f"mergeable@{min_t_key}={stats['by_threshold'][min_t_key]['mergeable_pairs']:>3}  "
            f"@{max_t_key}={stats['by_threshold'][max_t_key]['mergeable_pairs']:>3}  "
            f"t={elapsed:.1f}s"
        )

    n_img = len(per_image)
    total_pairs_all = sum(r["total_pairs"] for r in per_image)

    by_threshold_agg: dict[str, dict] = {}
    for t in args.centroid_dist:
        t_key = str(int(t))
        total_adj = sum(r["by_threshold"][t_key]["adj_pairs"] for r in per_image)
        total_mrg = sum(r["by_threshold"][t_key]["mergeable_pairs"] for r in per_image)
        by_threshold_agg[t_key] = {
            "mean_adj_pairs": round(total_adj / n_img, 1),
            "mean_mergeable_pairs": round(total_mrg / n_img, 2),
            "pct_pairs_adj": round(total_adj / max(total_pairs_all, 1) * 100, 1),
            "pct_pairs_mergeable": round(
                total_mrg / max(total_pairs_all, 1) * 100, 1
            ),
        }

    kill_threshold = str(int(max(args.centroid_dist)))
    kill_triggered = by_threshold_agg[kill_threshold]["mean_mergeable_pairs"] < 5.0

    agg = {
        "n_images": n_img,
        "mean_raw_masks": round(sum(r["raw_masks"] for r in per_image) / n_img, 1),
        "mean_masks_after_filter": round(
            sum(r["masks_after_filter"] for r in per_image) / n_img, 1
        ),
        "mean_total_pairs": round(total_pairs_all / n_img, 1),
        "by_threshold": by_threshold_agg,
        "kill_condition_threshold_px": int(max(args.centroid_dist)),
        "kill_triggered": kill_triggered,
        "total_wall_seconds": round(time.time() - t_start, 1),
    }

    report = {"per_image": per_image, "aggregate": agg}
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "step0_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Wrote report to {out_path}")

    print()
    print("=" * 60)
    print("EMR STEP 0 — MERGE CANDIDATE REPORT")
    print("=" * 60)
    print(f"  Images analyzed             : {n_img}")
    print(f"  Mean raw masks / image      : {agg['mean_raw_masks']}")
    print(f"  Mean filtered masks / image : {agg['mean_masks_after_filter']}")
    print(f"  Mean total pairs / image    : {agg['mean_total_pairs']}")
    print()
    print(f"  {'thresh':>6} | {'mean_adj':>8} | {'mean_mrg':>8} | {'%_adj':>5} | {'%_mrg':>5} |")
    print(f"  {'------':>6}-+-{'--------':>8}-+-{'--------':>8}-+-{'-----':>5}-+-{'-----':>5}-+")
    for t in sorted(args.centroid_dist):
        t_key = str(int(t))
        ta = by_threshold_agg[t_key]
        kill_marker = "  ← kill check" if int(t) == int(max(args.centroid_dist)) else ""
        print(
            f"  {int(t):>4}px | {ta['mean_adj_pairs']:>8.1f} | "
            f"{ta['mean_mergeable_pairs']:>8.2f} | "
            f"{ta['pct_pairs_adj']:>4.1f}% | "
            f"{ta['pct_pairs_mergeable']:>4.1f}% |"
            f"{kill_marker}"
        )
    print()
    print(f"  Total wall time             : {agg['total_wall_seconds']:.0f}s")
    print("=" * 60)

    if kill_triggered:
        print()
        print("[KILL CONDITION TRIGGERED]")
        print(
            f"  mean_mergeable_pairs@{kill_threshold}px = "
            f"{by_threshold_agg[kill_threshold]['mean_mergeable_pairs']:.2f} < 5.0"
        )
        print("  EMR has nothing meaningful to act on at FastSAM iou=0.7.")
        print("  Do NOT proceed to Step 1. STOP and report to human.")
    else:
        print()
        print("[KILL CONDITION: PASSED]")
        print(
            f"  mean_mergeable_pairs@{kill_threshold}px = "
            f"{by_threshold_agg[kill_threshold]['mean_mergeable_pairs']:.2f} >= 5.0"
        )
        print(
            "  Sufficient merge candidates found. Awaiting human review before Step 1."
        )


if __name__ == "__main__":
    main()
