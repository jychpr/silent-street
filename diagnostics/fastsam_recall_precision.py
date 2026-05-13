"""
Compute recall and precision of FastSAM proposals against COCO GT.

Reads the JSON written by fastsam_proposals.py and a COCO annotation file.
Outputs a Markdown report to --output and also prints it to stdout.
"""

import argparse
import json
import logging
import math
import os

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# OV-COCO class splits — values sourced from datasets/coco_eval.py (base_catids / target_catids).
# Hardcoded here so we don't import OV-DQUO code; cross-check that file if the splits ever change.
OV_COCO_BASE_IDS = frozenset([
    70, 2, 53, 7, 73, 57, 4, 79, 62, 74, 9, 38, 20, 19, 54, 85, 72, 27, 80, 51,
    78, 15, 84, 55, 16, 59, 48, 34, 23, 86, 90, 50, 25, 31, 56, 82, 75, 42, 3,
    65, 52, 60, 35, 1, 8, 44, 33, 24,
])
OV_COCO_NOVEL_IDS = frozenset([
    28, 21, 47, 6, 76, 41, 18, 63, 32, 36, 81, 22, 61, 87, 5, 17, 49,
])
_SPLITS_SOURCE = "datasets/coco_eval.py (base_catids / target_catids)"


def parse_args():
    p = argparse.ArgumentParser(description="FastSAM proposal recall and precision vs COCO GT")
    p.add_argument("--proposals", required=True, help="Path to proposals JSON from fastsam_proposals.py")
    p.add_argument("--coco-ann", required=True, help="Path to COCO annotation JSON")
    p.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for match (default 0.5)")
    p.add_argument("--output", required=True, help="Path to output Markdown report")
    p.add_argument("--split-name", default="unknown", help="Human-readable split name for the report")
    return p.parse_args()


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of xyxy boxes. Returns (N, M) matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def scale_label(area: float) -> str:
    """COCO scale bins using box area as proxy for mask area (noted in report)."""
    if area < 1024:    # < 32²
        return "small"
    if area < 9216:    # < 96²
        return "medium"
    return "large"


def pct(n: int, d: int) -> str:
    if d == 0:
        return "N/A"
    return f"{100.0 * n / d:.1f}%"


def fmt_f(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x:.3f}"


def count_stats(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {k: "N/A" for k in ("mean", "p50", "p90", "p99", "max")}
    return {
        "mean": f"{arr.mean():.1f}",
        "p50": f"{np.percentile(arr, 50):.0f}",
        "p90": f"{np.percentile(arr, 90):.0f}",
        "p99": f"{np.percentile(arr, 99):.0f}",
        "max": str(int(arr.max())),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def build_report(args, prop_data, has_novel_gt, gt_stats, prop_stats, per_image_counts, iou_thresh):
    L = []

    def row(*cells):
        L.append("| " + " | ".join(str(c) for c in cells) + " |")

    def sep(n):
        L.append("| " + " | ".join(["---"] * n) + " |")

    num_images = len(per_image_counts)
    total_proposals = sum(c["total"] for c in per_image_counts)
    fastsam_cfg = prop_data.get("fastsam_config", {})

    # --- Header ---
    L.append(f"# FastSAM Proposal Recall & Precision — {args.split_name}")
    L.append("")
    L.append("## Header")
    L.append("")
    row("Key", "Value")
    sep(2)
    row("Split name", args.split_name)
    row("Num images evaluated", num_images)
    row("Total proposals", total_proposals)
    row("has\\_novel\\_gt", str(has_novel_gt))
    row("IoU threshold", iou_thresh)
    row("FastSAM weights", fastsam_cfg.get("weights", "N/A"))
    row("FastSAM conf", fastsam_cfg.get("conf", "N/A"))
    row("FastSAM IoU (NMS)", fastsam_cfg.get("iou", "N/A"))
    row("FastSAM imgsz", fastsam_cfg.get("imgsz", "N/A"))
    row("OV-COCO splits source", _SPLITS_SOURCE)
    L.append("")

    # --- Table A: Recall by scale × base/novel ---
    L.append("## Table A — Recall by Scale × Base/Novel")
    L.append("")
    L.append("> **Note:** box area (w×h) is used as a proxy for COCO mask area to classify scale.")
    L.append("")
    row("Scale", "Base Recall", "Novel Recall", "Num Base GT", "Num Novel GT")
    sep(5)
    for scale in ("small", "medium", "large"):
        bs = gt_stats[(scale, "base")]
        ns = gt_stats[(scale, "novel")]
        if has_novel_gt:
            novel_recall = pct(ns["recalled"], ns["total"])
            novel_gt_n = str(ns["total"])
        else:
            novel_recall = "N/A (base-only ann)"
            novel_gt_n = "N/A"
        row(scale, pct(bs["recalled"], bs["total"]), novel_recall, bs["total"], novel_gt_n)
    L.append("")

    # --- Table B: Precision by predicted scale ---
    L.append("## Table B — Precision by Predicted Scale")
    L.append("")
    if not has_novel_gt:
        L.append("> **Caveat:** annotation file is base-only. Novel-class TPs count as FPs, so precision is understated.")
        L.append("")
    row("Scale", "Precision", "Num Proposals", "Mean Conf (all)", "Mean Conf (precise)", "Mean Conf (imprecise)")
    sep(6)
    for scale in ("small", "medium", "large"):
        ps = prop_stats[scale]
        conf_all = np.array(ps["conf_precise"] + ps["conf_imprecise"])
        conf_prec = np.array(ps["conf_precise"])
        conf_imp = np.array(ps["conf_imprecise"])
        row(
            scale,
            pct(ps["precise"], ps["total"]),
            ps["total"],
            fmt_f(float(conf_all.mean()) if len(conf_all) else float("nan")),
            fmt_f(float(conf_prec.mean()) if len(conf_prec) else float("nan")),
            fmt_f(float(conf_imp.mean()) if len(conf_imp) else float("nan")),
        )
    L.append("")

    # --- Table C: Per-image proposal counts ---
    L.append("## Table C — Per-Image Proposal Counts")
    L.append("")
    row("Bin", "Mean", "P50", "P90", "P99", "Max")
    sep(6)
    for label, arr in [
        ("overall", np.array([c["total"] for c in per_image_counts])),
        ("small",   np.array([c["small"] for c in per_image_counts])),
        ("medium",  np.array([c["medium"] for c in per_image_counts])),
        ("large",   np.array([c["large"] for c in per_image_counts])),
    ]:
        s = count_stats(arr)
        row(label, s["mean"], s["p50"], s["p90"], s["p99"], s["max"])
    L.append("")

    # --- Table D: Score vs precision ---
    all_prec = np.array([c for ps in prop_stats.values() for c in ps["conf_precise"]])
    all_imp = np.array([c for ps in prop_stats.values() for c in ps["conf_imprecise"]])
    gap = float(all_prec.mean() - all_imp.mean()) if (len(all_prec) and len(all_imp)) else float("nan")
    d = cohens_d(all_prec, all_imp)

    L.append("## Table D — Score vs Precision (Filter Quality)")
    L.append("")
    row("Group", "N", "Mean Conf", "Stdev Conf")
    sep(4)
    row("Precise proposals", len(all_prec),
        fmt_f(float(all_prec.mean()) if len(all_prec) else float("nan")),
        fmt_f(float(all_prec.std(ddof=1)) if len(all_prec) >= 2 else float("nan")))
    row("Imprecise proposals", len(all_imp),
        fmt_f(float(all_imp.mean()) if len(all_imp) else float("nan")),
        fmt_f(float(all_imp.std(ddof=1)) if len(all_imp) >= 2 else float("nan")))
    row("Gap (precise − imprecise)", "—", fmt_f(gap), "—")
    row("Cohen's d", "—", fmt_f(d), "—")
    L.append("")

    # --- Prose summary ---
    L.append("## Summary")
    L.append("")

    if not has_novel_gt:
        L.append("**Novel recall cannot be assessed from this run** — the annotation file contains only base-class GT.")
        L.append("Novel-class objects that FastSAM correctly localises count as false positives in the precision numbers,")
        L.append("so precision is biased low. Re-run with `instances_val2017_basetarget.json` for the full picture.")
        L.append("")

    # Small-novel recall
    if has_novel_gt:
        sn = gt_stats[("small", "novel")]
        if sn["total"] > 0:
            sn_rate = sn["recalled"] / sn["total"]
            flag = "**GREEN**" if sn_rate > 0.30 else ("**YELLOW**" if sn_rate >= 0.10 else "**RED**")
            L.append(
                f"**Small-novel recall:** {pct(sn['recalled'], sn['total'])} "
                f"({sn['recalled']}/{sn['total']}) — {flag}. "
                "(>30% green, 10–30% yellow, <10% red.)"
            )
        else:
            L.append("**Small-novel recall:** No small novel GT objects in this evaluated set.")
        L.append("")

    # Small-proposal precision
    sp = prop_stats["small"]
    if sp["total"] > 0:
        sp_rate = sp["precise"] / sp["total"]
        if sp_rate > 0.50:
            sp_flag = "**GREEN** — raw FastSAM usable as supervision signal."
        elif sp_rate >= 0.20:
            sp_flag = "**YELLOW** — a filter mechanism is needed (likely the contribution path)."
        else:
            sp_flag = "**RED** — raw FastSAM small-box output is too noisy to use directly."
        L.append(
            f"**Small-proposal precision:** {pct(sp['precise'], sp['total'])} "
            f"({sp['precise']}/{sp['total']}) — {sp_flag}"
        )
    else:
        L.append("**Small-proposal precision:** No small proposals in this evaluated set.")
    L.append("")

    # Score-precision gap
    if not math.isnan(gap):
        if gap > 0.10:
            gap_note = "FastSAM confidence is a **usable filter signal**."
        else:
            gap_note = (
                "FastSAM confidence is **not a reliable filter** — "
                "consider mask geometry, region CLIPness, or other signals."
            )
        L.append(f"**Score–precision gap:** {fmt_f(gap)} confidence units (Cohen's d = {fmt_f(d)}). {gap_note}")
    else:
        L.append("**Score–precision gap:** Insufficient data to assess.")
    L.append("")

    return "\n".join(L)


def main():
    args = parse_args()

    with open(args.proposals) as f:
        prop_data = json.load(f)
    with open(args.coco_ann) as f:
        coco_data = json.load(f)

    log.info(f"Loaded {len(prop_data['proposals'])} proposal entries")
    log.info(f"Loaded COCO ann with {len(coco_data['annotations'])} annotations")

    ann_by_image = {}
    for ann in coco_data["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    ann_cat_ids = {cat["id"] for cat in coco_data["categories"]}
    has_novel_gt = bool(ann_cat_ids & OV_COCO_NOVEL_IDS)
    log.info(
        f"has_novel_gt={has_novel_gt} "
        f"(ann categories={len(ann_cat_ids)}, novel overlap={len(ann_cat_ids & OV_COCO_NOVEL_IDS)})"
    )

    gt_stats = {
        (scale, split): {"recalled": 0, "total": 0}
        for scale in ("small", "medium", "large")
        for split in ("base", "novel")
    }
    prop_stats = {
        scale: {"precise": 0, "total": 0, "conf_precise": [], "conf_imprecise": []}
        for scale in ("small", "medium", "large")
    }
    per_image_counts = []

    iou_thresh = args.iou_thresh

    for entry in prop_data["proposals"]:
        image_id = entry["image_id"]
        prop_boxes = np.array(entry["boxes"], dtype=float) if entry["boxes"] else np.zeros((0, 4))
        prop_scores = np.array(entry["scores"], dtype=float) if entry["scores"] else np.zeros(0)

        gt_anns = ann_by_image.get(image_id, [])
        raw_boxes, raw_cats, raw_areas = [], [], []
        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            raw_boxes.append([x, y, x + w, y + h])
            raw_cats.append(ann["category_id"])
            raw_areas.append(w * h)

        gt_boxes = np.array(raw_boxes, dtype=float) if raw_boxes else np.zeros((0, 4))
        gt_cat_ids = np.array(raw_cats, dtype=int) if raw_cats else np.zeros(0, dtype=int)
        gt_areas = np.array(raw_areas, dtype=float) if raw_areas else np.zeros(0)

        iou_mat = box_iou(gt_boxes, prop_boxes)  # (num_gt, num_prop)

        # GT recall: for each GT, did any proposal match?
        for i in range(len(gt_boxes)):
            cat_id = int(gt_cat_ids[i])
            scale = scale_label(float(gt_areas[i]))
            if cat_id in OV_COCO_BASE_IDS:
                split = "base"
            elif cat_id in OV_COCO_NOVEL_IDS:
                split = "novel"
            else:
                continue
            gt_stats[(scale, split)]["total"] += 1
            if len(prop_boxes) > 0 and iou_mat[i].max() >= iou_thresh:
                gt_stats[(scale, split)]["recalled"] += 1

        # Proposal precision: for each proposal, does it match any GT?
        counts = {"total": len(prop_boxes), "small": 0, "medium": 0, "large": 0}
        for j in range(len(prop_boxes)):
            x1, y1, x2, y2 = prop_boxes[j]
            area = max(0.0, (x2 - x1) * (y2 - y1))
            scale = scale_label(area)
            counts[scale] += 1
            conf = float(prop_scores[j])
            # short-circuit avoids .max() on empty array when gt_boxes is empty
            precise = len(gt_boxes) > 0 and iou_mat[:, j].max() >= iou_thresh
            prop_stats[scale]["total"] += 1
            if precise:
                prop_stats[scale]["precise"] += 1
                prop_stats[scale]["conf_precise"].append(conf)
            else:
                prop_stats[scale]["conf_imprecise"].append(conf)

        per_image_counts.append(counts)

    report = build_report(args, prop_data, has_novel_gt, gt_stats, prop_stats, per_image_counts, iou_thresh)
    print(report)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)
    log.info(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
