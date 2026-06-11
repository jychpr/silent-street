"""
E-SAM Gate C — Entity-level Mask Refinement (EMR) on 12 diagnostic images.

Implements E-SAM §3.3 Eq.5-8 on top of the Gate B MMG pipeline (Eq.1-2).
The MMG stage is re-run in-memory per image (the Gate B gallery was not
persisted) to rebuild M̂32_O + the 64-grid mask gallery + M_S, then EMR's
"split-then-merge" strategy refines M̂32_O into the entity-level map M_E.

Authority: references/E-SAM_2503.12094v1.pdf, Figure 4 + Eq.5-8.

Split phase (Eq.5-6):
  Eq.5  OR^q_p = M^32_p ∩ M^32_q.  Process M̂32_O in DESCENDING score order.
        ratio = area(OR) / max(area_p, area_q).
          ratio <  δ  → remove OR from the LARGER mask          (minor carve)
          ratio >= δ  → find 64-grid prompts P^64 inside OR,
                        pick gallery guidance G_p (Eq.6), assign
                        OR to whichever mask G_p matches better   (guided carve)
  Eq.6  G_p = M^64O_p if (S^64B_p − S^64O_p < τ) else M^64B_p.
        Most-frequent O/B decision across prompts; highest-conf prompt's mask.

Merge phase (Eq.7-8):
  Eq.7  S_C  = cosine sim of SAM-encoder features sampled at SUPERPIXEL
        centroids of M_S  (Correction 1 — superpixel centroids, NOT one
        mask centroid).  S_M(i,j) = (1/|C_i|) Σ_{c∈C_i}
                                     |{ c'∈C_j : c'∈Top_k(S_C[c,·]) }|.
        C_i = superpixel centroids falling inside entity mask i.
  Eq.8  Candidate pair (S_M >= τ_SM).  MERGE iff ∃ a single gallery mask
        covering >= τ_enc of BOTH candidates; else KEEP SEPARATE.
        (Default keep-separate — this is the over-merge rail.)

Correction 2: NO pseudo-label weight is assigned in Gate C. Entity masks are
produced for visual + quantitative inspection only. Weight is a Gate D decision.

Correction 3: images 34 (n_hat_O=4) and 30 (n_hat_O=31) are flagged as expected
near-no-ops, not bugs.

Run:
  python diagnostics/esam_gate_c.py \\
    --coco-root data \\
    --coco-ann data/Annotations/instances_train2017_12img_diag.json \\
    --output-dir diagnostics/output/esam_gateC
"""

import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.segmentation import felzenszwalb

# Reuse Gate B MMG helpers verbatim (this is our own Gate B code, not FastSAM).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esam_gate_b import (  # noqa: E402
    _run_point_grid, _categorize, _naive_nms, _best_map_filter,
    _DIAG_IMAGE_IDS, _THETA_O, _GAMMA_O,
    _FELZ_SCALE, _FELZ_SIGMA, _FELZ_MIN_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── EMR locked settings (Gate C approval) ──────────────────────────────────
_DELTA   = 0.05   # Eq.5 minor-overlap ratio threshold        (E-SAM Table 4)
_TAU     = 0.1    # Eq.6 O-vs-B score-difference tolerance     (E-SAM Table 4)
_TOPK    = 3      # Eq.7 top-k similar centroids               (Gate A acknowledged guess)
_TAU_SM  = 0.5    # Eq.8 S_M candidate threshold               (Gate A acknowledged guess)
_TAU_ENC = 0.5    # Eq.8 gallery encompass coverage threshold  (Gate A acknowledged guess)

_FULL_TRAIN_N = 107_000
_RNG = np.random.default_rng(seed=42)

# Degeneracy flags (Correction 3) — expected near-no-ops, not bugs.
_DEGENERATE = {34: "n_hat_O=4 trivial scene", 30: "n_hat_O=31 low complexity"}


def parse_args():
    p = argparse.ArgumentParser(description="E-SAM Gate C: EMR on 12 diagnostic images")
    p.add_argument("--coco-root",   required=True)
    p.add_argument("--coco-ann",    required=True)
    p.add_argument("--split",       default="train2017")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--output-dir",  default="diagnostics/output/esam_gateC")
    return p.parse_args()


# ===========================================================================
# Eq.7 — superpixel-centroid similarity  (Correction 1)
# ===========================================================================

def _superpixel_centroids(M_S: np.ndarray) -> np.ndarray:
    """Return (K, 2) float pixel (x, y) centroid of each Felzenszwalb label."""
    K = int(M_S.max()) + 1
    ys, xs = np.nonzero(np.ones_like(M_S))          # all pixel coords
    labels = M_S.ravel()
    flatx = xs.astype(np.float64)
    flaty = ys.astype(np.float64)
    cnt   = np.bincount(labels, minlength=K).astype(np.float64)
    sumx  = np.bincount(labels, weights=flatx, minlength=K)
    sumy  = np.bincount(labels, weights=flaty, minlength=K)
    cnt[cnt == 0] = 1.0
    return np.stack([sumx / cnt, sumy / cnt], axis=1)  # (K, 2) (x, y)


def _sample_centroid_feats(predictor, centroids_xy: np.ndarray, H: int, W: int,
                           device: str) -> torch.Tensor:
    """
    Bilinearly sample the SAM image-encoder feature map at each superpixel
    centroid, returning L2-normalised (K, C) feature vectors.

    predictor.features is (1, C, Hf, Wf) for the 1024-padded input; centroid
    pixel coords are mapped into that frame via the predictor's resize transform.
    """
    feats    = predictor.features                                   # (1, C, Hf, Wf)
    img_size = predictor.model.image_encoder.img_size               # 1024
    coords   = predictor.transform.apply_coords(centroids_xy.copy(), (H, W))  # resized frame
    g        = coords / img_size * 2.0 - 1.0                        # → [-1, 1] (gx, gy)
    grid     = torch.from_numpy(g).float().to(device).view(1, -1, 1, 2)
    samp     = F.grid_sample(feats, grid, mode="bilinear", align_corners=False)  # (1,C,K,1)
    f        = samp.squeeze(0).squeeze(-1).permute(1, 0).contiguous()            # (K, C)
    return F.normalize(f, dim=1)


def _adjacent_mask_similarity(
    S_C: torch.Tensor, cent_entity: np.ndarray, n_ent: int, topk: int,
) -> np.ndarray:
    """
    Eq.7 — build the adjacent mask similarity matrix S_M (n_ent, n_ent).

    S_C         : (K, K) centroid-centroid cosine similarity.
    cent_entity : (K,) entity index each centroid falls inside, or -1.
    Returns S_M[i,j] = (1/|C_i|) Σ_{c∈C_i} #{ top-k neighbours of c that land in C_j }.
    Asymmetric by construction.
    """
    K = S_C.shape[0]
    sc = S_C.clone()
    sc.fill_diagonal_(-1e9)                          # exclude self from Top-k
    topk_idx = sc.topk(min(topk, K - 1), dim=1).indices.cpu().numpy()  # (K, k)

    sizeC = np.bincount(cent_entity[cent_entity >= 0], minlength=n_ent).astype(np.float64)
    count = np.zeros((n_ent, n_ent), dtype=np.float64)
    for c in range(K):
        i = cent_entity[c]
        if i < 0:
            continue
        for t in topk_idx[c]:
            j = cent_entity[t]
            if j < 0 or j == i:
                continue
            count[i, j] += 1.0
    safe = sizeC.copy()
    safe[safe == 0] = 1.0
    return count / safe[:, None]                     # (n_ent, n_ent)


# ===========================================================================
# Eq.8 — gallery encompass gate
# ===========================================================================

def _gallery_encompass(
    gallery_flat: torch.Tensor, gallery_area: torch.Tensor,
    mi_256: torch.Tensor, mj_256: torch.Tensor, tau_enc: float,
) -> bool:
    """
    Eq.8 rail: does a SINGLE gallery mask cover >= tau_enc of BOTH candidates?

    gallery_flat : (G, 256*256) float16 on GPU — aggregated M64_O ∪ M64_B.
    mi_256/mj_256: (256*256,) float16 candidate masks at gallery resolution.
    Returns True → MERGE permitted; False → keep separate.
    """
    ai = mi_256.sum().clamp(min=1.0)
    aj = mj_256.sum().clamp(min=1.0)
    cover_i = (gallery_flat @ mi_256) / ai           # (G,)
    cover_j = (gallery_flat @ mj_256) / aj           # (G,)
    both    = torch.minimum(cover_i, cover_j)        # (G,)
    return bool((both >= tau_enc).any().item())


# ===========================================================================
# Union-find for transitive merges
# ===========================================================================

class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb
            return True
        return False


# ===========================================================================
# Visualisation
# ===========================================================================

def _overlay(img_bgr: np.ndarray, masks, seed: int = 0) -> np.ndarray:
    """Semi-transparent per-mask colour overlay (one random colour per mask)."""
    rng = np.random.default_rng(seed)
    out = img_bgr.copy().astype(np.float32)
    for m in masks:
        m_np = m.cpu().numpy().astype(bool) if isinstance(m, torch.Tensor) else m.astype(bool)
        if not m_np.any():
            continue
        color = rng.integers(64, 220, size=3).astype(np.float32)
        for c in range(3):
            out[:, :, c] = np.where(m_np, out[:, :, c] * 0.5 + color[c] * 0.5, out[:, :, c])
    return out.astype(np.uint8)


def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _save_side_by_side(img_bgr, hat_O_masks, mE_masks, out_path, note=""):
    p0 = _label_panel(img_bgr, "original")
    p1 = _label_panel(_overlay(img_bgr, hat_O_masks, seed=1),
                      f"M-hat32_O  ({len(hat_O_masks)})")
    p2 = _label_panel(_overlay(img_bgr, mE_masks, seed=7),
                      f"M_E  ({len(mE_masks)}){'  ' + note if note else ''}")
    sep = np.full((img_bgr.shape[0], 4, 3), 255, np.uint8)
    canvas = np.hstack([p0, sep, p1, sep, p2])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, canvas)


# ===========================================================================
# EMR
# ===========================================================================

def _run_emr(
    m_hat_O: torch.Tensor, sc_hat_O: torch.Tensor,
    m64_O_256: torch.Tensor, m64_B_256: torch.Tensor,
    sc64_O: torch.Tensor, sc64_B: torch.Tensor, pts64_px: np.ndarray,
    M_S: np.ndarray, predictor, H: int, W: int, device: str,
) -> dict:
    """Apply EMR Eq.5-8. Returns entity masks + audit counts."""
    n = m_hat_O.shape[0]
    stats = dict(eq5_minor=0, eq6_guided=0, guided_no_prompt=0,
                 n_candidates=0, merge_fired=0, merge_rejected=0)
    if n == 0:
        return dict(entities=[], **stats)

    masks = m_hat_O.to(device).clone()               # (n, H, W) bool — mutable working set

    # ── Split phase (Eq.5-6) ────────────────────────────────────────────────
    flat0   = masks.view(n, -1).float()
    inter0  = torch.mm(flat0, flat0.t())             # (n, n) initial intersections
    del flat0
    order   = sc_hat_O.argsort(descending=True).tolist()   # descending score
    rank    = {idx: r for r, idx in enumerate(order)}

    # candidate overlapping pairs (p ranked higher than q), from initial state
    ov = (inter0 > 0).cpu().numpy()
    del inter0
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            if ov[a, b]:
                p, q = (a, b) if rank[a] < rank[b] else (b, a)   # p = higher score
                pairs.append((rank[p], p, q))
    pairs.sort()                                     # resolve in descending-score order

    # 64-grid prompt lookup grid (round to pixel)
    px = np.clip(np.round(pts64_px[:, 0]).astype(int), 0, W - 1)
    py = np.clip(np.round(pts64_px[:, 1]).astype(int), 0, H - 1)

    for _, p, q in pairs:
        OR = masks[p] & masks[q]
        a_or = float(OR.sum())
        if a_or == 0.0:
            continue
        a_p, a_q = float(masks[p].sum()), float(masks[q].sum())
        larger = p if a_p >= a_q else q
        ratio = a_or / max(a_p, a_q)

        if ratio < _DELTA:
            masks[larger] &= ~OR                     # Eq.5 minor carve
            stats["eq5_minor"] += 1
            continue

        # ratio >= δ → guided carve (Eq.6)
        stats["eq6_guided"] += 1
        OR_cpu = OR.cpu().numpy()
        in_or = OR_cpu[py, px]                        # (4096,) bool prompts inside OR
        pidx = np.nonzero(in_or)[0]
        if pidx.size == 0:
            masks[larger] &= ~OR                      # no prompt → fall back to minor carve
            stats["guided_no_prompt"] += 1
            continue

        # Eq.6 per-prompt O/B decision; majority decision, highest-conf prompt's mask
        use_obj = (sc64_B[pidx] - sc64_O[pidx]) < _TAU      # True → object-level guidance
        maj_obj = bool(use_obj.sum() >= (len(pidx) - use_obj.sum()))
        cand    = pidx[use_obj.cpu().numpy()] if maj_obj else pidx[(~use_obj).cpu().numpy()]
        if cand.size == 0:
            cand = pidx
        conf    = torch.maximum(sc64_O[cand], sc64_B[cand])
        best_p  = int(cand[int(conf.argmax())])
        g256    = (m64_O_256[best_p] if maj_obj else m64_B_256[best_p]).to(device)
        Gp = F.interpolate(g256.float()[None, None], size=(H, W), mode="nearest")[0, 0].bool()

        # assign OR to whichever current mask G_p matches better (higher IoU)
        gi = (Gp & masks[p]).sum().float()
        gj = (Gp & masks[q]).sum().float()
        iou_p = gi / (Gp.sum() + masks[p].sum() - gi).clamp(min=1)
        iou_q = gj / (Gp.sum() + masks[q].sum() - gj).clamp(min=1)
        loser = q if iou_p >= iou_q else p
        masks[loser] &= ~OR

    # drop emptied masks; remember mapping to original M̂32_O index for sentinel
    areas = masks.view(n, -1).sum(dim=1)
    surv  = [i for i in range(n) if int(areas[i]) > 0]
    if not surv:
        return dict(entities=[], **stats)
    split_masks = masks[surv]                        # (m, H, W) bool — M̃32
    m = len(surv)

    # ── entity label map for centroid membership (split masks ~ disjoint) ───
    label_map = -np.ones((H, W), dtype=np.int64)
    sm_cpu = split_masks.cpu().numpy()
    for e in range(m):
        label_map[sm_cpu[e]] = e

    # ── Eq.7: superpixel centroids → S_C → S_M ──────────────────────────────
    centroids = _superpixel_centroids(M_S)           # (K, 2)
    feats     = _sample_centroid_feats(predictor, centroids, H, W, device)  # (K, C)
    S_C       = feats @ feats.t()                    # (K, K) cosine
    cx = np.clip(np.round(centroids[:, 0]).astype(int), 0, W - 1)
    cy = np.clip(np.round(centroids[:, 1]).astype(int), 0, H - 1)
    cent_entity = label_map[cy, cx]                  # (K,) entity idx or -1
    S_M = _adjacent_mask_similarity(S_C, cent_entity, m, _TOPK)
    del feats, S_C

    # ── Eq.8: candidate pairs (symmetric S_M) → encompass gate ──────────────
    G = torch.cat([m64_O_256, m64_B_256], dim=0).to(device)          # (8192,256,256) uint8
    gallery_flat = G.view(G.shape[0], -1).half()                     # (8192, 65536) f16
    del G

    # candidate split masks at gallery resolution (256×256)
    sm_256 = F.interpolate(split_masks.float()[:, None], size=(256, 256),
                           mode="nearest")[:, 0].view(m, -1).half()   # (m, 65536)

    uf = _UF(m)
    for i in range(m):
        for j in range(i + 1, m):
            sij = max(S_M[i, j], S_M[j, i])
            if sij < _TAU_SM:
                continue
            stats["n_candidates"] += 1
            if _gallery_encompass(gallery_flat, None, sm_256[i], sm_256[j], _TAU_ENC):
                uf.union(i, j)
                stats["merge_fired"] += 1
            else:
                stats["merge_rejected"] += 1
    del gallery_flat, sm_256

    # ── assemble M_E entities from union-find components ────────────────────
    comp = {}
    for e in range(m):
        comp.setdefault(uf.find(e), []).append(e)
    entities = []
    comp_sizes = []                                  # #input M̂32_O masks fused per entity
    for members in comp.values():
        em = split_masks[members[0]].clone()
        for mem in members[1:]:
            em |= split_masks[mem]
        entities.append(em.cpu())
        comp_sizes.append(len(members))

    return dict(entities=entities, comp_sizes=comp_sizes, split_count=m, **stats)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.coco_ann) as f:
        data = json.load(f)
    images = sorted([img for img in data["images"] if img["id"] in _DIAG_IMAGE_IDS],
                    key=lambda x: x["id"])
    log.info(f"Loaded {len(images)} diagnostic images")
    images_dir = os.path.join(args.coco_root, "Images", args.split)

    # ── GPU safety check (hard rule 7) ──────────────────────────────────────
    if "cuda" in args.device:
        used  = torch.cuda.memory_allocated(args.device)
        total = torch.cuda.get_device_properties(args.device).total_memory
        log.info(f"GPU: {used/1e6:.0f} MB allocated / {total/1e6:.0f} MB total")
        if used > 2e9:
            raise RuntimeError(f"GPU has {used/1e9:.1f} GB already allocated — stop.")

    # ── Load SAM-H ──────────────────────────────────────────────────────────
    from segment_anything import SamPredictor, sam_model_registry
    ckpt_h = os.path.join(args.weights_dir, "sam_vit_h_4b8939.pth")
    if not os.path.isfile(ckpt_h):
        raise FileNotFoundError(f"SAM-H checkpoint missing: {ckpt_h}")
    sam = sam_model_registry["vit_h"](checkpoint=ckpt_h)
    sam.to(device=args.device).eval()
    predictor = SamPredictor(sam)
    model_tag = "SAM-H"
    log.info(f"Loaded {model_tag}")
    log.info(f"EMR locked: delta={_DELTA} tau={_TAU} k={_TOPK} "
             f"S_M>={_TAU_SM} encompass>={_TAU_ENC} | theta_O={_THETA_O} gamma_O={_GAMMA_O}")
    log.info("=" * 78)

    all_stats = []

    for img_info in images:
        image_id = img_info["id"]
        img_bgr  = cv2.imread(os.path.join(images_dir, img_info["file_name"]))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W     = img_rgb.shape[:2]
        flag     = _DEGENERATE.get(image_id, "")
        log.info(f"\nImage {image_id}  ({W}×{H}){'  [DEGENERATE: ' + flag + ']' if flag else ''}")

        # ── Encoder ─────────────────────────────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.set_image(img_rgb)
        torch.cuda.synchronize()
        t_enc = time.perf_counter() - t0

        # ── MMG re-run (Gate B Eq.1-2) ──────────────────────────────────────
        t0 = time.perf_counter()
        m32_full, m32_256, iou32, _ = _run_point_grid(predictor, 32, H, W, args.device)
        m32_O, m32_B, _, _, scores32_O, _ = _categorize(m32_full, m32_256, iou32)
        nms_idx  = _naive_nms(m32_O, scores32_O, _THETA_O, args.device)
        m_O_nms  = m32_O[nms_idx]
        sc_O_nms = scores32_O[nms_idx]
        flt_idx  = _best_map_filter(m_O_nms, m32_B, _GAMMA_O, args.device)
        m_hat_O  = m_O_nms[flt_idx]                  # (n, H, W) bool — M̂32_O
        sc_hat_O = sc_O_nms[flt_idx]
        n_hat_O  = m_hat_O.shape[0]

        M_S = felzenszwalb(img_rgb, scale=_FELZ_SCALE, sigma=_FELZ_SIGMA, min_size=_FELZ_MIN_SIZE)

        m64_full, m64_256, iou64, pts64_px = _run_point_grid(predictor, 64, H, W, args.device)
        _, _, m64_O_256, m64_B_256, sc64_O, sc64_B = _categorize(m64_full, m64_256, iou64)
        torch.cuda.synchronize()
        t_mmg = time.perf_counter() - t0

        del m32_full, m32_256, m64_full, m64_256, m32_O, m32_B, m_O_nms
        torch.cuda.empty_cache()

        # ── EMR (Eq.5-8) ────────────────────────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        emr = _run_emr(m_hat_O, sc_hat_O, m64_O_256, m64_B_256, sc64_O, sc64_B,
                       pts64_px, M_S, predictor, H, W, args.device)
        torch.cuda.synchronize()
        t_emr = time.perf_counter() - t0

        entities = emr["entities"]
        n_M_E = len(entities)
        t_total = t_enc + t_mmg + t_emr

        # ── Over-merge sentinel (PRIMARY): entity fusing >2 input M̂32_O masks.
        # Split is 1:1 with M̂32_O (carving only removes pixels), so an entity's
        # union-find component size == #input objects it swallowed. >2 = over-merge.
        comp_sizes = emr.get("comp_sizes", [])
        sentinel  = sum(1 for s in comp_sizes if s > 2)
        max_comp  = max(comp_sizes) if comp_sizes else 0

        # bbox-span (SECONDARY, confounded): #input M̂32_O centers inside an entity
        # bbox. Fires on spatial nesting (large mask bbox holds small-object
        # centers) even with zero merges — kept only for transparency, not trusted.
        hat_boxes = []
        hcpu = m_hat_O.cpu().numpy()
        for i in range(n_hat_O):
            ys, xs = np.nonzero(hcpu[i])
            if xs.size:
                hat_boxes.append(((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2))
        bbox_span = 0
        for em in entities:
            ys, xs = np.nonzero(em.numpy())
            if xs.size == 0:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            ncov = sum(1 for (bx, by) in hat_boxes if x0 <= bx <= x1 and y0 <= by <= y1)
            if ncov > 2:
                bbox_span += 1

        # ── Side-by-side PNG ────────────────────────────────────────────────
        png = os.path.join(args.output_dir, f"{image_id}_side_by_side.png")
        _save_side_by_side(img_bgr, list(m_hat_O.cpu()), entities, png,
                           note=("DEGEN" if flag else ""))

        log.info(f"  enc={t_enc:.2f}s  MMG={t_mmg:.2f}s  EMR={t_emr:.2f}s  total={t_total:.2f}s")
        log.info(f"  M̂32_O={n_hat_O} → M_E={n_M_E} | split: minor={emr['eq5_minor']} "
                 f"guided={emr['eq6_guided']} (no-prompt={emr['guided_no_prompt']})")
        log.info(f"  Eq.8: candidates={emr['n_candidates']} MERGE={emr['merge_fired']} "
                 f"REJECTED={emr['merge_rejected']} | over-merge(>2 fused)={sentinel} "
                 f"max_fused={max_comp}  [bbox-span confounded={bbox_span}]")

        all_stats.append({
            "image_id": image_id, "image_wh": [W, H], "degenerate": flag or None,
            "n_hat_O_in": n_hat_O, "n_M_E_out": n_M_E,
            "eq5_minor": emr["eq5_minor"], "eq6_guided": emr["eq6_guided"],
            "guided_no_prompt": emr["guided_no_prompt"],
            "n_candidates": emr["n_candidates"],
            "merge_fired": emr["merge_fired"], "merge_rejected": emr["merge_rejected"],
            "over_merge_components": sentinel, "max_fused": max_comp,
            "bbox_span_confounded": bbox_span,
            "t_encoder_s": round(t_enc, 3), "t_mmg_s": round(t_mmg, 3),
            "t_emr_s": round(t_emr, 3), "t_total_s": round(t_total, 3),
        })

        del m_hat_O, m64_O_256, m64_B_256, entities
        torch.cuda.empty_cache()

    # ── Aggregate report ────────────────────────────────────────────────────
    n = len(all_stats)
    mean = lambda k: sum(r[k] for r in all_stats) / n
    mean_total = mean("t_total_s")
    proj_h = mean_total * _FULL_TRAIN_N / 3600

    log.info("\n" + "=" * 100)
    log.info("GATE C — EMR AUDIT TABLE")
    log.info("=" * 100)
    hdr = (f"  {'img_id':>8} {'in':>4} {'out':>4} {'minor':>6} {'guid':>5} "
           f"{'cand':>5} {'MERGE':>6} {'REJ':>5} {'OM>2':>5} {'mxF':>4} "
           f"{'enc':>5} {'MMG':>6} {'EMR':>6} {'total':>6}  flag")
    log.info(hdr)
    log.info("  " + "-" * (len(hdr) + 2))
    for r in all_stats:
        log.info(
            f"  {r['image_id']:>8} {r['n_hat_O_in']:>4} {r['n_M_E_out']:>4} "
            f"{r['eq5_minor']:>6} {r['eq6_guided']:>5} {r['n_candidates']:>5} "
            f"{r['merge_fired']:>6} {r['merge_rejected']:>5} {r['over_merge_components']:>5} "
            f"{r['max_fused']:>4} "
            f"{r['t_encoder_s']:>5.2f} {r['t_mmg_s']:>6.2f} {r['t_emr_s']:>6.2f} "
            f"{r['t_total_s']:>6.2f}  {r['degenerate'] or ''}")
    log.info("  " + "-" * (len(hdr) + 2))
    log.info(f"  mean total/image: {mean_total:.2f}s   "
             f"merge_fired={mean('merge_fired'):.1f}  merge_rejected={mean('merge_rejected'):.1f}  "
             f"over_merge(>2)_total={sum(r['over_merge_components'] for r in all_stats)}  "
             f"[bbox-span confounded total={sum(r['bbox_span_confounded'] for r in all_stats)}]")
    log.info(f"  OBSERVED 107k projection: {mean_total:.2f}s × {_FULL_TRAIN_N:,} = {proj_h:.1f}h")

    out_json = os.path.join(args.output_dir, "gate_c_stats.json")
    with open(out_json, "w") as f:
        json.dump({
            "gate": "C", "model": model_tag,
            "delta": _DELTA, "tau": _TAU, "topk": _TOPK,
            "tau_sm": _TAU_SM, "tau_encompass": _TAU_ENC,
            "theta_O": _THETA_O, "gamma_O": _GAMMA_O,
            "n_images": n, "mean_total_s": round(mean_total, 3),
            "proj_107k_h": round(proj_h, 1),
            "per_image": all_stats,
        }, f, indent=2)
    log.info(f"\n  Stats → {out_json}")
    log.info(f"  PNGs  → {args.output_dir}/  (12 × *_side_by_side.png)")


if __name__ == "__main__":
    main()
