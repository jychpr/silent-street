"""
SAM AMG Cost Probe — Phase-2 decision input.

Step 0a: segment-anything install + checkpoint verification (done externally).
Step 0b: Time SamAutomaticMaskGenerator.generate() on 12 diagnostic images
         for SAM-H, SAM-L, SAM-B. Report per-image n_masks + wall-clock.
Step 0c: Save raw AMG mask overlay PNGs for the largest model that ran without OOM.

Hard rules:
- NO MMG/EMR/USR. NO pseudo-label output. Cost probe only.
- OOM is caught per model; run continues with next smaller model.
- Timing is the headline number.
"""

import argparse
import json
import logging
import os
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DIAG_IMAGE_IDS = frozenset({
    157105, 122263, 543882, 579329, 443084, 92869,
    475808, 435091, 171270, 307238, 30, 34,
})

# Paper AMG settings (E-SAM uses 32x32 grid, all other params at SAM defaults)
_AMG_POINTS_PER_SIDE = 32

_MODELS = [
    ("SAM-H", "sam_vit_h_4b8939.pth", "vit_h"),
    ("SAM-L", "sam_vit_l_0b3195.pth", "vit_l"),
    ("SAM-B", "sam_vit_b_01ec64.pth", "vit_b"),
]

# COCO train2017 ~ 107,237 images
_FULL_TRAIN_N = 107_000


def parse_args():
    p = argparse.ArgumentParser(description="SAM AMG cost probe on 12 diagnostic images")
    p.add_argument("--coco-root", required=True,
                   help="COCO root dir (contains Images/ and Annotations/)")
    p.add_argument("--split", default="train2017", choices=["train2017", "val2017"])
    p.add_argument("--coco-ann", required=True,
                   help="Path to diagnostic annotation JSON")
    p.add_argument("--weights-dir", default="weights",
                   help="Directory containing sam_vit_*.pth files (default: weights/)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="diagnostics/output/sam_amg_probe",
                   help="Directory for PNGs and timing JSON")
    return p.parse_args()


def _random_color():
    return tuple(int(x) for x in np.random.randint(64, 220, size=3))


def save_overlay_png(img_bgr: np.ndarray, masks: list, out_path: str):
    """Draw semi-transparent colored masks on image, save to out_path."""
    overlay = img_bgr.copy().astype(np.float32)
    rng = np.random.default_rng(seed=42)
    for m in masks:
        seg = m["segmentation"]  # bool (H, W)
        color = rng.integers(64, 220, size=3).astype(np.float32)
        for c in range(3):
            overlay[:, :, c] = np.where(seg, overlay[:, :, c] * 0.5 + color[c] * 0.5,
                                         overlay[:, :, c])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, overlay.astype(np.uint8))


def run_model(model_name: str, ckpt_path: str, model_type: str,
              images: list, images_dir: str, args,
              save_pngs: bool) -> dict:
    """
    Load model, run AMG on all 12 images, return timing rows.
    Returns None if model fails to load (e.g. checkpoint missing).
    Catches OOM per image; marks that image as oom=True.
    """
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry  # noqa: PLC0415

    log.info(f"Loading {model_name} from {ckpt_path} ...")
    try:
        sam = sam_model_registry[model_type](checkpoint=ckpt_path)
        sam.to(device=args.device)
        generator = SamAutomaticMaskGenerator(sam, points_per_side=_AMG_POINTS_PER_SIDE)
    except Exception as e:
        log.error(f"{model_name}: failed to load — {e}")
        return None

    rows = []
    hdr = f"  {'img_id':>8}  {'n_masks':>7}  {'time_s':>7}  note"
    log.info(hdr)
    log.info("  " + "-" * (len(hdr) - 2))

    for img_info in images:
        image_id = img_info["id"]
        img_path = os.path.join(images_dir, img_info["file_name"])
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            import torch  # noqa: PLC0415
            if "cuda" in args.device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            masks = generator.generate(img_rgb)
            if "cuda" in args.device:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_masks = len(masks)
            note = ""
            oom = False
        except torch.cuda.OutOfMemoryError:
            elapsed = float("nan")
            n_masks = 0
            note = "OOM"
            oom = True
            log.warning(f"  {image_id:>8}  OOM — skipping")

        rows.append({
            "image_id": image_id,
            "n_masks": n_masks,
            "time_s": round(elapsed, 3) if not oom else None,
            "oom": oom,
        })
        if not oom:
            log.info(f"  {image_id:>8}  {n_masks:>7}  {elapsed:>7.2f}s  {note}")
            if save_pngs:
                png_path = os.path.join(args.output_dir,
                                        f"{model_name.replace('-','_').lower()}_{image_id}.png")
                save_overlay_png(img_bgr, masks, png_path)

    # Free GPU memory before next model
    del generator, sam
    import torch  # noqa: PLC0415
    if "cuda" in args.device:
        torch.cuda.empty_cache()

    return rows


def summarise(model_name: str, rows: list, fastsam_h: float = 6.0,
              oln_fe_lo: float = 72.0, oln_fe_hi: float = 96.0):
    """Print aggregate stats + full-set projection."""
    valid = [r for r in rows if not r["oom"]]
    n_oom = sum(1 for r in rows if r["oom"])

    if not valid:
        log.info(f"{model_name}: ALL OOM — no timing data.")
        return {"mean_time_s": None, "mean_n_masks": None, "n_oom": n_oom}

    mean_time = sum(r["time_s"] for r in valid) / len(valid)
    mean_masks = sum(r["n_masks"] for r in valid) / len(valid)
    proj_h = mean_time * _FULL_TRAIN_N / 3600

    log.info("")
    log.info(f"  {model_name}: mean {mean_time:.2f}s/image, {mean_masks:.0f} masks/image"
             + (f"  [{n_oom} OOM]" if n_oom else ""))
    log.info(f"  Projected full precompute ({_FULL_TRAIN_N:,} images): {proj_h:.1f} h")
    log.info(f"  Compare: FastSAM ~{fastsam_h:.0f}h | OLN+FE ~{oln_fe_lo:.0f}–{oln_fe_hi:.0f}h")

    return {"mean_time_s": round(mean_time, 3), "mean_n_masks": round(mean_masks, 1),
            "n_oom": n_oom, "proj_full_h": round(proj_h, 1)}


def main():
    args = parse_args()

    with open(args.coco_ann) as f:
        data = json.load(f)
    images = [img for img in data["images"] if img["id"] in _DIAG_IMAGE_IDS]
    log.info(f"Loaded {len(images)} diagnostic images from {args.coco_ann}")

    images_dir = os.path.join(args.coco_root, "Images", args.split)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}
    best_model_name = None  # largest that ran without all-OOM

    log.info(f"AMG settings: points_per_side={_AMG_POINTS_PER_SIDE} (paper default)")
    log.info("=" * 70)

    for model_name, ckpt_file, model_type in _MODELS:
        ckpt_path = os.path.join(args.weights_dir, ckpt_file)
        if not os.path.isfile(ckpt_path):
            log.warning(f"{model_name}: checkpoint not found at {ckpt_path} — skipping")
            continue

        # Save PNGs for the largest (first) model that runs successfully
        save_pngs = (best_model_name is None)

        log.info(f"\n{'='*70}")
        log.info(f"  {model_name}")
        log.info(f"{'='*70}")

        rows = run_model(model_name, ckpt_path, model_type,
                         images, images_dir, args, save_pngs)
        if rows is None:
            continue

        summary = summarise(model_name, rows)
        all_results[model_name] = {"rows": rows, "summary": summary}

        if summary["mean_time_s"] is not None and best_model_name is None:
            best_model_name = model_name

    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    for mn, res in all_results.items():
        s = res["summary"]
        if s["mean_time_s"] is not None:
            log.info(
                f"  {mn}: {s['mean_time_s']:.2f}s/image, {s['mean_n_masks']:.0f} masks/image"
                f"  → {s['proj_full_h']:.1f}h full precompute"
                + (f"  [{s['n_oom']} OOM]" if s["n_oom"] else "")
            )
        else:
            log.info(f"  {mn}: ALL OOM or failed")

    if best_model_name:
        log.info(f"\n  PNGs saved for {best_model_name} → {args.output_dir}/")

    out_path = os.path.join(args.output_dir, "amg_timing.json")
    with open(out_path, "w") as f:
        json.dump({
            "amg_points_per_side": _AMG_POINTS_PER_SIDE,
            "n_diag_images": len(images),
            "full_train_n": _FULL_TRAIN_N,
            "results": {
                mn: {"summary": res["summary"], "per_image": res["rows"]}
                for mn, res in all_results.items()
            },
        }, f, indent=2)
    log.info(f"\n  Timing report → {out_path}")


if __name__ == "__main__":
    main()
