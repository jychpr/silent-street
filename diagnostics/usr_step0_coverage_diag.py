"""
USR Step 0 — Coverage diagnostic (no USR logic, no training output).

For each of the 12 diagnostic images:
  1. Run FastSAM pass-1 (identical call to precompute_fastsam_pseudo_labels.py).
  2. Build a union mask = OR of all returned masks.
  3. Compute Felzenszwalb superpixel map.
  4. Compute per-superpixel coverage fraction; apply three cutoffs (0.10, 0.30, 0.50).
     A superpixel is "uncovered" at threshold T if FastSAM covered < T of its pixels.
  5. Report per-image and aggregate uncovered_area% at all three thresholds.

Kill condition: mean uncovered area < 5% at the STRICT 0.10 threshold.
Rationale: thr=0.10 counts only genuinely blank regions (FastSAM touched <10% of the
superpixel). Partial-coverage superpixels would regenerate overlapping masks that
fusion discards — they are not real gaps for USR.
"""

import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
from skimage.segmentation import felzenszwalb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DIAG_IMAGE_IDS = frozenset({
    157105, 122263, 543882, 579329, 443084, 92869,
    475808, 435091, 171270, 307238, 30, 34,
})
_COVERAGE_THRS = [0.10, 0.30, 0.50]   # sweep; kill evaluated at 0.10
_KILL_UNCOVERED_PCT = 5.0              # applied at thr=0.10 (strictest)


def parse_args():
    p = argparse.ArgumentParser(
        description="USR Step 0: FastSAM coverage diagnostic on 12 diagnostic images"
    )
    p.add_argument("--coco-root", required=True,
                   help="COCO root dir (contains Images/ and Annotations/)")
    p.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    p.add_argument("--coco-ann", required=True,
                   help="Path to COCO annotation JSON (e.g. instances_train2017_12img_diag.json)")
    p.add_argument("--fastsam-weights", default="FastSAM-x.pt")
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--felz-scale", type=float, default=200.0,
                   help="Felzenszwalb scale (default 200.0)")
    p.add_argument("--felz-sigma", type=float, default=0.8,
                   help="Felzenszwalb sigma (default 0.8)")
    p.add_argument("--felz-min-size", type=int, default=50,
                   help="Felzenszwalb min_size in pixels (default 50)")
    p.add_argument("--output-dir", default="diagnostics/output/usr_step0",
                   help="Directory to write step0_report.json")
    return p.parse_args()


def _build_union_mask(res, ih: int, iw: int) -> np.ndarray:
    """Return bool array (ih, iw): True wherever any FastSAM mask fires."""
    union = np.zeros((ih, iw), dtype=bool)
    if res is None or res.masks is None:
        return union
    mt = res.masks.data  # (N, H_m, W_m) tensor
    mH, mW = int(mt.shape[1]), int(mt.shape[2])
    if mH != ih or mW != iw:
        import torch.nn.functional as F  # noqa: PLC0415
        mt = F.interpolate(
            mt.unsqueeze(0).float(), size=(ih, iw), mode="nearest"
        ).squeeze(0)
    masks_np = mt.cpu().numpy() > 0.5  # (N, ih, iw) bool
    for m in masks_np:
        union |= m
    return union


def _superpixel_coverage_raw(union_mask: np.ndarray,
                              sp_map: np.ndarray) -> tuple:
    """
    Compute per-superpixel coverage fraction. O(H*W), not O(N_sp * H*W).
    Returns (sp_area, coverage_frac) as 1-D arrays of length n_labels.
    """
    flat_sp = sp_map.ravel()
    flat_mask = union_mask.ravel().astype(np.int32)
    n_labels = int(flat_sp.max()) + 1
    sp_area = np.bincount(flat_sp, minlength=n_labels)
    covered_px = np.bincount(flat_sp, weights=flat_mask, minlength=n_labels)
    coverage_frac = covered_px / np.maximum(sp_area, 1)
    return sp_area, coverage_frac


def process_image(model, img_path: str, iw: int, ih: int, args) -> dict:
    """FastSAM pass-1 + superpixel coverage analysis. Returns stats dict."""
    # Identical FastSAM call to precompute_fastsam_pseudo_labels.py
    results = model(
        img_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        retina_masks=True,
        verbose=False,
    )

    if results and results[0].boxes is not None:
        n_raw = len(results[0].boxes)
        union_mask = _build_union_mask(results[0], ih, iw)
    else:
        n_raw = 0
        union_mask = np.zeros((ih, iw), dtype=bool)

    image_area = iw * ih
    mask_covered_px = int(union_mask.sum())

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    sp_map = felzenszwalb(
        img_rgb,
        scale=args.felz_scale,
        sigma=args.felz_sigma,
        min_size=args.felz_min_size,
    )

    sp_area, coverage_frac = _superpixel_coverage_raw(union_mask, sp_map)
    n_superpixels = len(sp_area)

    thr_stats = {}
    for thr in _COVERAGE_THRS:
        uncov = coverage_frac < thr
        uncov_px = int(sp_area[uncov].sum())
        thr_stats[thr] = {
            "n_uncovered": int(uncov.sum()),
            "uncovered_area_px": uncov_px,
            "uncovered_area_pct": round(100.0 * uncov_px / image_area, 2),
        }

    return {
        "n_raw_masks": n_raw,
        "image_area_px": image_area,
        "mask_covered_px": mask_covered_px,
        "mask_covered_pct": round(100.0 * mask_covered_px / image_area, 2),
        "n_superpixels": n_superpixels,
        "by_thr": thr_stats,
    }


def main():
    args = parse_args()

    from ultralytics import FastSAM  # noqa: PLC0415 — deferred so --help works without GPU

    with open(args.coco_ann) as f:
        data = json.load(f)
    images = [img for img in data["images"] if img["id"] in _DIAG_IMAGE_IDS]
    log.info(f"Loaded {len(images)} diagnostic images from {args.coco_ann}")

    model = FastSAM(args.fastsam_weights)
    log.info(f"FastSAM loaded. conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    log.info(
        f"Felzenszwalb: scale={args.felz_scale} sigma={args.felz_sigma} "
        f"min_size={args.felz_min_size}  thresholds={_COVERAGE_THRS}"
    )

    images_dir = os.path.join(args.coco_root, "Images", args.split)
    rows = []

    hdr = (
        f"{'img_id':>8}  {'iw':>5}x{'ih':<5}  {'n_raw':>5}  "
        f"{'cov%':>6}  {'n_sp':>5}  {'uncov@10%':>9}  {'uncov@30%':>9}  {'uncov@50%':>9}  {'t':>5}"
    )
    log.info(hdr)
    log.info("-" * len(hdr))

    for img_info in images:
        image_id = img_info["id"]
        img_path = os.path.join(images_dir, img_info["file_name"])
        iw, ih = img_info["width"], img_info["height"]

        t0 = time.time()
        stats = process_image(model, img_path, iw, ih, args)
        elapsed = time.time() - t0

        rows.append({"image_id": image_id, "iw": iw, "ih": ih,
                     "elapsed_s": round(elapsed, 3), **stats})

        bt = stats["by_thr"]
        log.info(
            f"{image_id:>8}  {iw:>5}x{ih:<5}  {stats['n_raw_masks']:>5}  "
            f"{stats['mask_covered_pct']:>5.1f}%  {stats['n_superpixels']:>5}  "
            f"{bt[0.10]['uncovered_area_pct']:>8.1f}%  "
            f"{bt[0.30]['uncovered_area_pct']:>8.1f}%  "
            f"{bt[0.50]['uncovered_area_pct']:>8.1f}%  "
            f"{elapsed:.2f}s"
        )

    n = len(rows)
    mean_raw = sum(r["n_raw_masks"] for r in rows) / n
    mean_cov = sum(r["mask_covered_pct"] for r in rows) / n
    mean_n_sp = sum(r["n_superpixels"] for r in rows) / n
    mean_by_thr = {
        thr: {
            "mean_n_uncovered": sum(r["by_thr"][thr]["n_uncovered"] for r in rows) / n,
            "mean_uncovered_area_pct": sum(r["by_thr"][thr]["uncovered_area_pct"] for r in rows) / n,
        }
        for thr in _COVERAGE_THRS
    }

    log.info("=" * 60)
    log.info("AGGREGATE (12 images)")
    log.info(f"  mean raw masks/image       : {mean_raw:.1f}")
    log.info(f"  mean mask coverage         : {mean_cov:.1f}%")
    log.info(f"  mean superpixels/image     : {mean_n_sp:.1f}")
    for thr in _COVERAGE_THRS:
        m = mean_by_thr[thr]
        log.info(
            f"  thr={thr:.2f}  mean_uncovered_sp={m['mean_n_uncovered']:.1f}  "
            f"mean_uncovered_area={m['mean_uncovered_area_pct']:.1f}%"
        )
    log.info("-" * 60)

    # Kill condition evaluated at strictest threshold (0.10)
    kill_pct = mean_by_thr[0.10]["mean_uncovered_area_pct"]
    kill = kill_pct < _KILL_UNCOVERED_PCT
    if kill:
        log.warning(
            f"KILL CONDITION TRIGGERED: mean uncovered area at thr=0.10 is "
            f"{kill_pct:.1f}% < {_KILL_UNCOVERED_PCT:.1f}% — USR has almost nothing to fill. Stop."
        )
    else:
        log.info(
            f"Kill condition NOT triggered: mean uncovered area at thr=0.10 is "
            f"{kill_pct:.1f}% >= {_KILL_UNCOVERED_PCT:.1f}% — proceed to Step 1."
        )
    log.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Serialise by_thr with string keys (JSON requires string keys)
    def _serialise_rows(rows_in):
        out = []
        for r in rows_in:
            row = {k: v for k, v in r.items() if k != "by_thr"}
            row["by_thr"] = {str(thr): v for thr, v in r["by_thr"].items()}
            out.append(row)
        return out

    report = {
        "args": {
            "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
            "coverage_thrs": _COVERAGE_THRS,
            "felz_scale": args.felz_scale, "felz_sigma": args.felz_sigma,
            "felz_min_size": args.felz_min_size,
        },
        "per_image": _serialise_rows(rows),
        "aggregate": {
            "n_images": n,
            "mean_raw_masks": round(mean_raw, 2),
            "mean_mask_covered_pct": round(mean_cov, 2),
            "mean_n_superpixels": round(mean_n_sp, 2),
            "by_thr": {
                str(thr): {k: round(v, 2) for k, v in m.items()}
                for thr, m in mean_by_thr.items()
            },
        },
        "kill_condition_triggered": kill,
        "kill_condition_thr": 0.10,
        "kill_condition_threshold_pct": _KILL_UNCOVERED_PCT,
    }
    out_path = os.path.join(args.output_dir, "step0_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
