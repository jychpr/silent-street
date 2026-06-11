"""
E-SAM Gate B — Multi-level Mask Generation (MMG) on 12 diagnostic images.

Implements E-SAM §3.2 Eq.1-2 using SAM-H directly via SamPredictor.
Dropped from this stage (per Gate A approval): Eq.3 density map, Eq.4 adaptive
NMS — those feed USR which is deferred.

MMG produces:
  M̂32_O  — refined object map (Eq.2)
  64-grid gallery — M64_O, M64_B per point at 256×256 (for EMR at Gate C)
  M_S     — Felzenszwalb superpixel map (K count reported; for Gate C S_C)

Eq.1 indexing (revision confirmed):
  - M32_O/P/SP use area-sort indices (order) applied to original masks tensor
  - M32_B uses iou_preds.argmax() on the ORIGINAL (unsorted) iou_preds
  - Both index into the same SAM tensor; operations are independent

Gallery storage (revision confirmed):
  - Stored at 256×256 uint8 via low_res_logits > 0 (SAM's third predict_torch return)
  - Upsample to image resolution only in Eq.5/6 at Gate C

Weight field for Gate D (confirmed):
  - weight = SAM predicted IoU score for that object-level mask (scores_hat_O[i])
  - For EMR-merged entities: max(score) across contributing gallery entries

Run:
  python diagnostics/esam_gate_b.py \\
    --coco-root data \\
    --coco-ann data/Annotations/instances_train2017_12img_diag.json \\
    --output-dir diagnostics/output/esam_gateB
"""

import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
import torch
from skimage.segmentation import felzenszwalb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DIAG_IMAGE_IDS = frozenset({
    157105, 122263, 543882, 579329, 443084, 92869,
    475808, 435091, 171270, 307238, 30, 34,
})

# E-SAM Table 4 hyperparameters used by MMG
_THETA_O    = 0.8   # naive NMS threshold for M32_O (Eq.2 step 1)
_GAMMA_O    = 0.6   # best-map IoU match threshold (Eq.2 step 2)

# Felzenszwalb parameters (Ambiguity D — no paper value; standard scene defaults)
_FELZ_SCALE    = 300
_FELZ_SIGMA    = 0.8
_FELZ_MIN_SIZE = 50

_FULL_TRAIN_N = 107_000
_BATCH_SIZE   = 64        # SAM predict_torch batch size (matches SAM default)

_RNG = np.random.default_rng(seed=42)


def parse_args():
    p = argparse.ArgumentParser(description="E-SAM Gate B: MMG on 12 diagnostic images")
    p.add_argument("--coco-root",   required=True)
    p.add_argument("--coco-ann",    required=True)
    p.add_argument("--split",       default="train2017")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--output-dir",  default="diagnostics/output/esam_gateB")
    return p.parse_args()


# ---------------------------------------------------------------------------
# SAM grid and inference helpers
# ---------------------------------------------------------------------------

def _build_pixel_grid(n_per_side: int, image_h: int, image_w: int) -> np.ndarray:
    """Return (n^2, 2) float array of (x, y) pixel coordinates."""
    from segment_anything.utils.amg import build_point_grid  # noqa: PLC0415
    pts_norm = build_point_grid(n_per_side)                  # (N, 2) in [0,1], (x,y)
    return pts_norm * np.array([[image_w, image_h]], dtype=float)  # (N, 2) pixel (x,y)


def _run_point_grid(
    predictor, n_per_side: int, image_h: int, image_w: int, device: str
) -> tuple:
    """
    Run SAM predict_torch on an n×n uniform point grid.

    Returns:
      masks_full  (N, 3, H, W) bool cpu  — binary at image resolution
      masks_256   (N, 3, 256, 256) uint8 cpu — SAM native low-res
      iou_preds   (N, 3) float cpu
      pts_px      (N, 2) float            — pixel (x, y) of each grid point
    """
    pts_px = _build_pixel_grid(n_per_side, image_h, image_w)   # (N, 2) float
    pts_xf = predictor.transform.apply_coords(pts_px, (image_h, image_w))
    pts_t  = torch.from_numpy(pts_xf).float().to(device)
    lbl_t  = torch.ones(len(pts_t), dtype=torch.int, device=device)

    all_full, all_256, all_iou = [], [], []
    for s in range(0, len(pts_t), _BATCH_SIZE):
        bp = pts_t[s:s + _BATCH_SIZE]
        bl = lbl_t[s:s + _BATCH_SIZE]
        with torch.no_grad():
            m_full, iou, m_low = predictor.predict_torch(
                bp[:, None, :], bl[:, None],
                multimask_output=True,
                return_logits=True,
            )
        # m_full: (B, 3, H, W) logits   — threshold > 0 for binary
        # m_low:  (B, 3, 256, 256) logits — SAM native low-res
        all_full.append((m_full > 0.0).cpu())
        all_256.append((m_low  > 0.0).to(torch.uint8).cpu())
        all_iou.append(iou.cpu())

    return (
        torch.cat(all_full, dim=0),   # (N, 3, H, W) bool
        torch.cat(all_256,  dim=0),   # (N, 3, 256, 256) uint8
        torch.cat(all_iou,  dim=0),   # (N, 3) float
        pts_px,                        # (N, 2) float pixel (x, y)
    )


def _categorize(
    masks_full: torch.Tensor,
    masks_256:  torch.Tensor,
    iou_preds:  torch.Tensor,
):
    """
    Eq.1: assign O/P/SP levels by area; pick best level by iou_pred score.

    Eq.1 indexing guarantee:
      - area sort (order) and iou_preds.argmax() (best) both index the ORIGINAL
        SAM output tensor independently. M32_B is the mask SAM scored highest,
        NOT the mask in the area-sorted slot with the highest index.

    Returns 6-tuple: masks_O_full, masks_B_full, masks_O_256, masks_B_256,
                     scores_O, scores_B — each (N, ...).
    """
    N   = masks_full.shape[0]
    idx = torch.arange(N)

    areas = masks_full.sum(dim=(-2, -1))           # (N, 3) pixel counts
    order = areas.argsort(dim=1, descending=True)  # (N, 3) [O, P, SP] indices
    best  = iou_preds.argmax(dim=1)                # (N,)  original-order argmax

    masks_O_full = masks_full[idx, order[:, 0]]   # (N, H, W) — largest area
    masks_B_full = masks_full[idx, best]           # (N, H, W) — highest SAM score
    masks_O_256  = masks_256 [idx, order[:, 0]]   # (N, 256, 256)
    masks_B_256  = masks_256 [idx, best]           # (N, 256, 256)
    scores_O     = iou_preds [idx, order[:, 0]]   # (N,) predicted IoU for O level
    scores_B     = iou_preds [idx, best]           # (N,) predicted IoU for B level

    return masks_O_full, masks_B_full, masks_O_256, masks_B_256, scores_O, scores_B


# ---------------------------------------------------------------------------
# Eq.2 — object-map refinement
# ---------------------------------------------------------------------------

def _naive_nms(
    masks: torch.Tensor, scores: torch.Tensor, threshold: float, device: str
) -> list[int]:
    """
    Eq.2 step 1: greedy mask-IoU NMS.
    Vectorised: builds full pairwise IoU matrix via matmul on GPU, then greedy.
    masks:  (N, H, W) bool cpu
    scores: (N,) float cpu
    """
    N = masks.shape[0]
    if N == 0:
        return []
    H, W = masks.shape[-2:]

    m_gpu = masks.to(device).float().view(N, H * W)   # (N, H*W)
    areas = m_gpu.sum(dim=1)                           # (N,)

    inter = torch.mm(m_gpu, m_gpu.t())                          # (N, N)
    union = areas[:, None] + areas[None, :] - inter             # (N, N)
    iou_mat = (inter / union.clamp(min=1)).cpu()                # (N, N)

    del m_gpu, inter, union
    if "cuda" in device:
        torch.cuda.empty_cache()

    order      = scores.argsort(descending=True).tolist()
    suppressed = torch.zeros(N, dtype=torch.bool)
    keep       = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou_mat[i] > threshold

    return keep


def _best_map_filter(
    masks_O: torch.Tensor, masks_B: torch.Tensor, gamma: float, device: str
) -> list[int]:
    """
    Eq.2 step 2: keep M_{i,O} if max mask-IoU with any M_{j,B} >= gamma.
    masks_O: (N_O, H, W) bool cpu — post-NMS object masks
    masks_B: (N_B, H, W) bool cpu — ALL 1024 best-level masks (no NMS)
    """
    N_O, N_B = masks_O.shape[0], masks_B.shape[0]
    if N_O == 0 or N_B == 0:
        return []
    H, W = masks_O.shape[-2:]

    mO = masks_O.to(device).float().view(N_O, H * W)  # (N_O, H*W)
    mB = masks_B.to(device).float().view(N_B, H * W)  # (N_B, H*W)
    aO = mO.sum(dim=1)   # (N_O,)
    aB = mB.sum(dim=1)   # (N_B,)

    inter = torch.mm(mO, mB.t())                               # (N_O, N_B)
    union = aO[:, None] + aB[None, :] - inter                  # (N_O, N_B)
    iou_mat = inter / union.clamp(min=1)                       # (N_O, N_B)
    max_ious = iou_mat.max(dim=1).values.cpu()                 # (N_O,)

    del mO, mB, inter, union, iou_mat
    if "cuda" in device:
        torch.cuda.empty_cache()

    return (max_ious >= gamma).nonzero(as_tuple=True)[0].tolist()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _save_overlay(img_bgr: np.ndarray, masks, out_path: str) -> None:
    """Semi-transparent colored mask overlay. masks: iterable of (H,W) bool/uint8."""
    overlay = img_bgr.copy().astype(np.float32)
    for m in masks:
        m_np = m.numpy().astype(bool) if isinstance(m, torch.Tensor) else m.astype(bool)
        color = _RNG.integers(64, 220, size=3).astype(np.float32)
        for c in range(3):
            overlay[:, :, c] = np.where(
                m_np, overlay[:, :, c] * 0.5 + color[c] * 0.5, overlay[:, :, c]
            )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, overlay.astype(np.uint8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.coco_ann) as f:
        data = json.load(f)
    images = [img for img in data["images"] if img["id"] in _DIAG_IMAGE_IDS]
    log.info(f"Loaded {len(images)} diagnostic images")

    images_dir = os.path.join(args.coco_root, "Images", args.split)

    # GPU safety check
    if "cuda" in args.device:
        used  = torch.cuda.memory_allocated(args.device)
        total = torch.cuda.get_device_properties(args.device).total_memory
        log.info(f"GPU: {used/1e6:.0f} MB used / {total/1e6:.0f} MB total")
        if used > 2e9:
            raise RuntimeError(f"GPU has {used/1e9:.1f} GB already allocated — is another process running?")

    # Load SAM-H (fall back to SAM-L on missing checkpoint)
    from segment_anything import SamPredictor, sam_model_registry  # noqa: PLC0415

    ckpt_h = os.path.join(args.weights_dir, "sam_vit_h_4b8939.pth")
    ckpt_l = os.path.join(args.weights_dir, "sam_vit_l_0b3195.pth")
    if os.path.isfile(ckpt_h):
        sam = sam_model_registry["vit_h"](checkpoint=ckpt_h)
        model_tag = "SAM-H"
    elif os.path.isfile(ckpt_l):
        log.warning("SAM-H not found; falling back to SAM-L")
        sam = sam_model_registry["vit_l"](checkpoint=ckpt_l)
        model_tag = "SAM-L"
    else:
        raise FileNotFoundError(f"No SAM checkpoint found in {args.weights_dir}")

    sam.to(device=args.device)
    sam.eval()
    predictor = SamPredictor(sam)
    log.info(f"Loaded {model_tag}")

    log.info(f"Hyperparameters: theta_O={_THETA_O} gamma_O={_GAMMA_O} "
             f"felz(scale={_FELZ_SCALE} sigma={_FELZ_SIGMA} min_size={_FELZ_MIN_SIZE})")
    log.info("=" * 70)

    all_stats = []

    for img_info in images:
        image_id  = img_info["id"]
        img_path  = os.path.join(images_dir, img_info["file_name"])
        img_bgr   = cv2.imread(img_path)
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W      = img_rgb.shape[:2]

        log.info(f"\nImage {image_id}  ({W}×{H}  {img_info['file_name']})")

        # ── Encoder ───────────────────────────────────────────────────────
        if "cuda" in args.device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.set_image(img_rgb)
        if "cuda" in args.device:
            torch.cuda.synchronize()
        t_enc = time.perf_counter() - t0

        # ── 32-grid pass ──────────────────────────────────────────────────
        if "cuda" in args.device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m32_full, m32_256, iou32, pts32_px = _run_point_grid(
            predictor, 32, H, W, args.device)
        if "cuda" in args.device:
            torch.cuda.synchronize()
        t_32 = time.perf_counter() - t0
        # m32_full: (1024, 3, H, W) bool

        # ── Eq.1: categorize 32-grid ──────────────────────────────────────
        m32_O, m32_B, _, _, scores32_O, _ = _categorize(m32_full, m32_256, iou32)
        # m32_O: (1024, H, W) bool   — object level (largest area per point)
        # m32_B: (1024, H, W) bool   — best level   (highest SAM score per point)
        n_raw = len(m32_O)   # always 1024

        # ── Eq.2 step 1: naive NMS at θ_O=0.8 ────────────────────────────
        t0 = time.perf_counter()
        nms_idx   = _naive_nms(m32_O, scores32_O, _THETA_O, args.device)
        m_O_nms   = m32_O[nms_idx]
        sc_O_nms  = scores32_O[nms_idx]
        n_nms     = len(nms_idx)

        # ── Eq.2 step 2: best-map IoU filter at γ_O=0.6 ──────────────────
        flt_idx   = _best_map_filter(m_O_nms, m32_B, _GAMMA_O, args.device)
        m_hat_O   = m_O_nms[flt_idx]    # M̂32_O — refined object map
        sc_hat_O  = sc_O_nms[flt_idx]   # predicted IoU scores (used as Gate-D weight)
        n_hat_O   = len(flt_idx)

        t_mmg_proc = time.perf_counter() - t0

        # ── Felzenszwalb M_S (for EMR S_C at Gate C) ─────────────────────
        t0 = time.perf_counter()
        M_S    = felzenszwalb(img_rgb, scale=_FELZ_SCALE, sigma=_FELZ_SIGMA,
                              min_size=_FELZ_MIN_SIZE)
        felz_K = int(M_S.max()) + 1
        t_felz = time.perf_counter() - t0

        # ── 64-grid pass ──────────────────────────────────────────────────
        if "cuda" in args.device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m64_full, m64_256, iou64, pts64_px = _run_point_grid(
            predictor, 64, H, W, args.device)
        if "cuda" in args.device:
            torch.cuda.synchronize()
        t_64 = time.perf_counter() - t0
        # m64_full: (4096, 3, H, W) bool

        # ── Build 64-grid gallery ─────────────────────────────────────────
        # Gallery stores M64_O and M64_B at 256×256 (uint8) per grid point.
        # Upsample to image resolution only in Eq.5/6 at Gate C.
        t0 = time.perf_counter()
        _, _, m64_O_256, m64_B_256, scores64_O, scores64_B = _categorize(
            m64_full, m64_256, iou64)
        # m64_O_256: (4096, 256, 256) uint8
        # m64_B_256: (4096, 256, 256) uint8
        gallery_size = m64_O_256.shape[0]   # = 4096
        t_gallery    = time.perf_counter() - t0

        # ── Timing totals ──────────────────────────────────────────────────
        t_mmg_total = t_32 + t_mmg_proc + t_felz + t_64 + t_gallery
        t_total     = t_enc + t_mmg_total

        log.info(f"  encoder:      {t_enc:.3f}s")
        log.info(f"  32-grid:      {t_32:.3f}s")
        log.info(f"  Eq.2 proc:    {t_mmg_proc:.3f}s  "
                 f"(raw={n_raw} → NMS={n_nms} → M̂32_O={n_hat_O})")
        log.info(f"  Felzenszwalb: {t_felz:.3f}s  K={felz_K}")
        log.info(f"  64-grid:      {t_64:.3f}s")
        log.info(f"  gallery:      {t_gallery:.3f}s  ({gallery_size} entries)")
        log.info(f"  TOTAL:        {t_total:.3f}s  (MMG={t_mmg_total:.3f}s)")

        # ── Overlay PNGs ──────────────────────────────────────────────────
        # raw_amg.png  — M32_O (1024 object-level masks, pre-NMS)
        # mmg_obj.png  — M̂32_O (post-NMS + IoU-filter)
        raw_png = os.path.join(args.output_dir, f"{image_id}_raw_amg.png")
        mmg_png = os.path.join(args.output_dir, f"{image_id}_mmg_obj.png")
        _save_overlay(img_bgr, m32_O, raw_png)
        _save_overlay(img_bgr, m_hat_O if n_hat_O else [], mmg_png)
        log.info(f"  PNGs → {os.path.basename(raw_png)}, {os.path.basename(mmg_png)}")

        all_stats.append({
            "image_id":     image_id,
            "image_wh":     [W, H],
            "t_encoder_s":  round(t_enc,        3),
            "t_32grid_s":   round(t_32,          3),
            "t_mmg_proc_s": round(t_mmg_proc,    3),
            "t_felz_s":     round(t_felz,        3),
            "t_64grid_s":   round(t_64,          3),
            "t_gallery_s":  round(t_gallery,     3),
            "t_mmg_total_s":round(t_mmg_total,   3),
            "t_total_s":    round(t_total,       3),
            "n_raw_32":     n_raw,
            "n_post_nms":   n_nms,
            "n_hat_O":      n_hat_O,
            "felz_K":       felz_K,
            "gallery_size": gallery_size,
        })

        # Free 64-grid tensors (large) before next image
        del m32_full, m32_256, m64_full, m64_256, m64_O_256, m64_B_256
        del m32_O, m32_B, m_O_nms, m_hat_O
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    # ── Aggregate report ──────────────────────────────────────────────────
    n = len(all_stats)
    def _mean(key):
        return sum(r[key] for r in all_stats) / n

    mean_enc      = _mean("t_encoder_s")
    mean_32       = _mean("t_32grid_s")
    mean_mmgp     = _mean("t_mmg_proc_s")
    mean_felz     = _mean("t_felz_s")
    mean_64       = _mean("t_64grid_s")
    mean_gal      = _mean("t_gallery_s")
    mean_mmg_tot  = _mean("t_mmg_total_s")
    mean_total    = _mean("t_total_s")
    mean_hat_O    = _mean("n_hat_O")
    mean_felz_K   = _mean("felz_K")
    proj_h        = mean_total * _FULL_TRAIN_N / 3600

    log.info("\n" + "=" * 70)
    log.info("GATE B FINAL SUMMARY")
    log.info("=" * 70)
    hdr = f"  {'img_id':>8}  {'enc':>6}  {'32g':>6}  {'Eq2':>6}  {'felz':>5}  "
    hdr += f"{'64g':>6}  {'gal':>5}  {'total':>7}  {'M̂32_O':>6}  {'K':>5}"
    log.info(hdr)
    log.info("  " + "-" * (len(hdr) - 2))
    for r in all_stats:
        log.info(
            f"  {r['image_id']:>8}  {r['t_encoder_s']:>6.2f}  {r['t_32grid_s']:>6.2f}  "
            f"{r['t_mmg_proc_s']:>6.2f}  {r['t_felz_s']:>5.2f}  "
            f"{r['t_64grid_s']:>6.2f}  {r['t_gallery_s']:>5.2f}  "
            f"{r['t_total_s']:>7.2f}s  {r['n_hat_O']:>6}  {r['felz_K']:>5}"
        )
    log.info("  " + "-" * (len(hdr) - 2))
    log.info(
        f"  {'mean':>8}  {mean_enc:>6.2f}  {mean_32:>6.2f}  "
        f"{mean_mmgp:>6.2f}  {mean_felz:>5.2f}  "
        f"{mean_64:>6.2f}  {mean_gal:>5.2f}  "
        f"{mean_total:>7.2f}s  {mean_hat_O:>6.1f}  {mean_felz_K:>5.0f}"
    )
    log.info("")
    log.info(f"  MMG breakdown (mean):  32-grid={mean_32:.2f}s  Eq.2={mean_mmgp:.2f}s  "
             f"Felz={mean_felz:.2f}s  64-grid={mean_64:.2f}s  gallery={mean_gal:.2f}s  "
             f"→ MMG total={mean_mmg_tot:.2f}s")
    log.info(f"  Encoder + MMG total:   {mean_total:.2f}s/image")
    log.info(f"  M̂32_O mean:           {mean_hat_O:.1f} masks/image")
    log.info(f"  Felzenszwalb mean K:   {mean_felz_K:.0f} superpixels/image")
    log.info(f"  OBSERVED projection:   {mean_total:.2f}s × {_FULL_TRAIN_N:,} = {proj_h:.1f}h")
    log.info(f"  [Paper A40 reference:  9.84s total E-SAM → {9.84*107000/3600:.0f}h]")

    out_json = os.path.join(args.output_dir, "gate_b_stats.json")
    with open(out_json, "w") as f:
        json.dump({
            "gate":           "B",
            "model":          model_tag,
            "theta_O":        _THETA_O,
            "gamma_O":        _GAMMA_O,
            "felz_scale":     _FELZ_SCALE,
            "felz_sigma":     _FELZ_SIGMA,
            "felz_min_size":  _FELZ_MIN_SIZE,
            "n_images":       n,
            "mean_total_s":   round(mean_total,   3),
            "mean_mmg_s":     round(mean_mmg_tot, 3),
            "mean_hat_O":     round(mean_hat_O,   1),
            "proj_107k_h":    round(proj_h,       1),
            "per_image":      all_stats,
        }, f, indent=2)
    log.info(f"\n  Stats → {out_json}")
    log.info(f"  PNGs  → {args.output_dir}/")


if __name__ == "__main__":
    main()
