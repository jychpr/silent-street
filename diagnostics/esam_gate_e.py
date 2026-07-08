"""
E-SAM Gate E — Under-Segmentation Refinement (USR) on 12 diagnostic images.

Pipeline:  MMG(+P/SP)  →  EMR (Gate C verbatim)  →  USR (§3.4 Eq.9-10)

Authority: references/E-SAM_2503.12094v1.pdf, §3.4 + Eq.9-10, Figure 3 USR panel,
Table 4 (rho).  MMG (Eq.1-2) and EMR (Eq.5-8) are imported UNCHANGED from Gate B/C;
the only new code is USR.

USR (Eq.9-10) — faithfulness ledger (paper-quote | CHOICE):
  - Eq.1 O/P/SP area order  : order[:,0]=O,[:,1]=P,[:,2]=SP            PAPER (Eq.1)
  - "uncovered" region S_R  : superpixel with <10% pixels covered by M_E
                              CHOICE (paper silent), strict reading of "regions
                              not covered by M_E"; value=0.10
  - "contained" S_R ⊆ P∪SP  : >=90% of S_R pixels inside (M_P∪M_SP)   CHOICE (0.90)
  - multiple containers     : smallest-area containing P/SP mask      CHOICE
  - prompt placement (Eq.9) : contained -> containing part/subpart centroid;
                              else -> superpixel centroid              PAPER (Eq.9)
  - additional-mask SAM lvl : best-level (highest-conf) mask per prompt CHOICE
  - greedy "use fewer masks": largest-uncovered-first; skip a region already
                              filled (>=10% covered) by a prior M_A    PAPER ("use
                              fewer masks") + CHOICE (skip threshold = firing thr)
  - Eq.10 fusion            : max mask-IoU(M_A, entity) > rho -> merge into that
                              entity; else add as new independent entity  PAPER (Eq.10)
  - rho                     : 0.1                                      PAPER (Table 4)

This module ADDS gap-filling entities; we EXPECT it to worsen the Gate D box-flood,
not fix it. Reported as such — no tuning. Novel-GT coverage is measured with the
Gate D convention (entity tight box, box IoU>=0.5) BEFORE vs AFTER USR.

Run:
  python diagnostics/esam_gate_e.py
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from skimage.segmentation import felzenszwalb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esam_gate_b import (  # noqa: E402
    _run_point_grid, _categorize, _naive_nms, _best_map_filter,
    _THETA_O, _GAMMA_O, _FELZ_SCALE, _FELZ_SIGMA, _FELZ_MIN_SIZE,
)
from esam_gate_c import (  # noqa: E402
    _run_emr, _DELTA, _TAU, _TOPK, _TAU_SM, _TAU_ENC,
)
# Gate D GT convention reused verbatim (entity tight box, box IoU>=0.5).
from gate_d_eval import (  # noqa: E402
    NOVEL, DIAG, ROOT, iou_matrix, mask_to_box, recalled_flags,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda:0"

# ── USR locked settings ────────────────────────────────────────────────────
_RHO          = 0.1    # Eq.10 fusion IoU threshold                 (Table 4)
_UNCOVERED    = 0.10   # CHOICE: S_R fires if M_E covers < this frac (strict override)
_CONTAIN_MIN  = 0.90   # CHOICE: S_R "contained" if >= this frac in (M_P∪M_SP)

# Locked EMR target M_E_before counts (Gate C); abort on any mismatch.
_M_E_TARGET = {30: 26, 34: 3, 92869: 45, 122263: 166, 157105: 135, 171270: 177,
               307238: 92, 435091: 58, 443084: 105, 475808: 92, 543882: 93, 579329: 107}

_OUTDIR = f"{ROOT}/diagnostics/output/esam_gateE"


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = a.sum() + b.sum() - inter
    return float(inter) / float(union)


# ===========================================================================
# USR — Eq.9-10
# ===========================================================================

def _run_usr(entities_before, m32_P, m32_SP, M_S, predictor, H, W):
    """
    Apply USR (Eq.9-10) on top of EMR entities.

    entities_before : list of (H,W) bool numpy — EMR M_E.
    m32_P, m32_SP   : (1024,H,W) bool numpy — Eq.1 part / subpart level masks.
    M_S             : (H,W) int — Felzenszwalb superpixel map.
    predictor       : SAM predictor with set_image already called for this image.

    Returns dict: entities_after (list), audit counts.
    """
    stats = dict(felz_K=int(M_S.max()) + 1, n_remaining=0, skipped_greedy=0,
                 prompts_part=0, prompts_superpixel=0, additional_masks=0,
                 fused_into_existing=0, new_entity=0)

    # M_E coverage (union of EMR entities)
    ME_cov = np.zeros((H, W), dtype=bool)
    for e in entities_before:
        ME_cov |= e

    # ── find remaining regions S_R (Eq.9 "not covered by M_E") ──────────────
    K = int(M_S.max()) + 1
    remaining = []  # (uncovered_px, label, sp_mask)
    for L in range(K):
        sp = (M_S == L)
        sp_size = int(sp.sum())
        if sp_size == 0:
            continue
        covered = int(np.logical_and(sp, ME_cov).sum())
        if covered / sp_size < _UNCOVERED:          # CHOICE strict: <10% covered
            remaining.append((sp_size - covered, L, sp))
    stats["n_remaining"] = len(remaining)
    if not remaining:
        return dict(entities_after=[e.copy() for e in entities_before], **stats)

    # greedy "use fewer masks": largest-uncovered-first
    remaining.sort(key=lambda t: t[0], reverse=True)

    # P/SP stack for containment (Eq.9): only materialise when needed
    psp = np.concatenate([m32_P, m32_SP], axis=0)        # (2048,H,W) bool
    psp_union = psp.any(axis=0)                          # (H,W) bool
    psp_area = psp.reshape(psp.shape[0], -1).sum(axis=1)  # (2048,) per-mask area

    entities = [e.copy() for e in entities_before]
    cur_cov = ME_cov.copy()

    for _uncov, L, sp in remaining:
        sp_size = int(sp.sum())
        ys, xs = np.nonzero(sp)
        # greedy skip: region already filled (>=10% covered) by a prior M_A
        if int(np.logical_and(sp, cur_cov).sum()) / sp_size >= _UNCOVERED:
            stats["skipped_greedy"] += 1
            continue

        # Eq.9 containment test: frac of S_R pixels inside (M_P∪M_SP)
        contained_frac = float(psp_union[ys, xs].mean())
        if contained_frac >= _CONTAIN_MIN:
            # smallest-area P/SP mask that contains S_R (>=90% of its pixels)
            inside = psp[:, ys, xs].mean(axis=1)         # (2048,) frac of S_R inside each
            cand = np.nonzero(inside >= _CONTAIN_MIN)[0]
            if cand.size == 0:                           # union covers but no single mask does
                px, py = float(xs.mean()), float(ys.mean())   # fall back to SP centroid
                placement = "superpixel"
            else:
                chosen = cand[int(np.argmin(psp_area[cand]))]
                cys, cxs = np.nonzero(psp[chosen])
                px, py = float(cxs.mean()), float(cys.mean())  # part/subpart centroid
                placement = "part"
        else:
            px, py = float(xs.mean()), float(ys.mean())  # superpixel centroid
            placement = "superpixel"

        # feed prompt to SAM; take best-level (highest-conf) mask
        masks_pred, scores_pred, _ = predictor.predict(
            point_coords=np.array([[px, py]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        MA = masks_pred[int(np.argmax(scores_pred))].astype(bool)
        if MA.sum() == 0:
            continue
        stats["additional_masks"] += 1
        stats["prompts_part" if placement == "part" else "prompts_superpixel"] += 1

        # Eq.10 fusion: max mask-IoU(M_A, entity) > rho -> merge; else new entity
        best_iou, best_j = 0.0, -1
        for j, e in enumerate(entities):
            io = _mask_iou(e, MA)
            if io > best_iou:
                best_iou, best_j = io, j
        if best_iou > _RHO and best_j >= 0:
            entities[best_j] = np.logical_or(entities[best_j], MA)
            stats["fused_into_existing"] += 1
        else:
            entities.append(MA)
            stats["new_entity"] += 1
        cur_cov |= MA

    return dict(entities_after=entities, **stats)


# ===========================================================================
# MMG + EMR rebuild (Gate C path), additionally surfacing P/SP
# ===========================================================================

def _mmg_emr_with_parts(predictor, img_rgb, H, W):
    """Re-run MMG(+P/SP) and EMR. Returns (EMR entities, m32_P, m32_SP numpy)."""
    predictor.set_image(img_rgb)
    m32_full, m32_256, iou32, _ = _run_point_grid(predictor, 32, H, W, DEVICE)
    (m32_O, m32_B, _, _, sc32_O, _,
     m32_P, m32_SP) = _categorize(m32_full, m32_256, iou32, return_parts=True)
    nms_idx = _naive_nms(m32_O, sc32_O, _THETA_O, DEVICE)
    m_O_nms, sc_O_nms = m32_O[nms_idx], sc32_O[nms_idx]
    flt = _best_map_filter(m_O_nms, m32_B, _GAMMA_O, DEVICE)
    m_hat_O, sc_hat_O = m_O_nms[flt], sc_O_nms[flt]

    M_S = felzenszwalb(img_rgb, scale=_FELZ_SCALE, sigma=_FELZ_SIGMA, min_size=_FELZ_MIN_SIZE)

    m64_full, m64_256, iou64, pts64 = _run_point_grid(predictor, 64, H, W, DEVICE)
    _, _, m64_O_256, m64_B_256, sc64_O, sc64_B = _categorize(m64_full, m64_256, iou64)

    emr = _run_emr(m_hat_O, sc_hat_O, m64_O_256, m64_B_256, sc64_O, sc64_B,
                   pts64, M_S, predictor, H, W, DEVICE)

    P_np = m32_P.numpy().astype(bool)
    SP_np = m32_SP.numpy().astype(bool)
    entities = [e.numpy().astype(bool) for e in emr["entities"]]

    del m32_full, m32_256, m64_full, m64_256, m64_O_256, m64_B_256, m32_P, m32_SP
    torch.cuda.empty_cache()
    return entities, P_np, SP_np, M_S


# ===========================================================================
# Visualisation
# ===========================================================================

def _overlay(img_bgr, masks, seed):
    rng = np.random.default_rng(seed)
    out = img_bgr.copy().astype(np.float32)
    for m in masks:
        if not m.any():
            continue
        c = rng.integers(64, 220, size=3).astype(np.float32)
        for k in range(3):
            out[:, :, k] = np.where(m, out[:, :, k] * 0.5 + c[k] * 0.5, out[:, :, k])
    return out.astype(np.uint8)


def _panel(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _save_triptych(img_bgr, before, after, added, path):
    p0 = _panel(img_bgr, "original")
    p1 = _panel(_overlay(img_bgr, before, 1), f"M_E before (EMR)  ({len(before)})")
    # after panel: existing in muted, NEW entities highlighted via separate seed
    p2 = _panel(_overlay(img_bgr, after, 7),
                f"M_E after (USR)  ({len(after)})  +{added} new")
    sep = np.full((img_bgr.shape[0], 4, 3), 255, np.uint8)
    canvas = np.hstack([p0, sep, p1, sep, p2])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, canvas)


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(_OUTDIR, exist_ok=True)

    # ── GT (Gate D convention: full 80-class json, novel boxes) ─────────────
    log.info("Loading full instances_train2017.json (~30s)…")
    full = json.load(open(f"{ROOT}/data/Annotations/instances_train2017.json"))
    cat = {c["id"]: c["name"] for c in full["categories"]}
    diag_set = set(DIAG)
    novel_gt = defaultdict(list)
    for a in full["annotations"]:
        if a["image_id"] in diag_set and cat[a["category_id"]] in NOVEL:
            x, y, w, h = a["bbox"]
            novel_gt[a["image_id"]].append([x, y, x + w, y + h])
    del full
    img_meta = {im["id"]: im for im in json.load(
        open(f"{ROOT}/data/Annotations/instances_train2017_12img_diag.json"))["images"]}

    # ── GPU safety + SAM-H ──────────────────────────────────────────────────
    used = torch.cuda.memory_allocated(DEVICE)
    log.info(f"GPU: {used/1e6:.0f} MB allocated / "
             f"{torch.cuda.get_device_properties(DEVICE).total_memory/1e6:.0f} MB")
    if used > 2e9:
        raise RuntimeError("GPU already busy — stop.")
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=f"{ROOT}/weights/sam_vit_h_4b8939.pth")
    sam.to(DEVICE).eval()
    predictor = SamPredictor(sam)
    log.info("Loaded SAM-H")
    log.info(f"USR locked: rho={_RHO} uncovered<{_UNCOVERED} contain>={_CONTAIN_MIN} | "
             f"EMR: delta={_DELTA} tau={_TAU} k={_TOPK} S_M>={_TAU_SM} enc>={_TAU_ENC} | "
             f"theta_O={_THETA_O} gamma_O={_GAMMA_O}")
    log.info("=" * 96)

    rows = []
    for iid in DIAG:
        im = img_meta[iid]
        H, W = im["height"], im["width"]
        img_bgr = cv2.imread(f"{ROOT}/data/Images/train2017/{im['file_name']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        before, m32_P, m32_SP, M_S = _mmg_emr_with_parts(predictor, img_rgb, H, W)
        m_e_before = len(before)
        # ── HARD ASSERT: EMR M_E reproduces the Gate C locked target ────────
        if m_e_before != _M_E_TARGET[iid]:
            raise SystemExit(f"ABORT {iid}: M_E_before={m_e_before} != Gate C target "
                             f"{_M_E_TARGET[iid]} — EMR perturbed, fix before USR.")

        usr = _run_usr(before, m32_P, m32_SP, M_S, predictor, H, W)
        after = usr["entities_after"]
        m_e_after = len(after)
        delta = m_e_after - m_e_before
        dt = time.perf_counter() - t0

        # sanity: delta == new_entity (fused don't add count)
        assert delta == usr["new_entity"], \
            f"{iid}: delta {delta} != new_entity {usr['new_entity']}"

        # ── novel-GT coverage (Gate D box convention) BEFORE vs AFTER ───────
        ng = novel_gt.get(iid, [])
        boxes_before = [b for b in (mask_to_box(e) for e in before) if b is not None]
        boxes_after = [b for b in (mask_to_box(e) for e in after) if b is not None]
        nov_before = int(recalled_flags(ng, boxes_before).sum())
        nov_after = int(recalled_flags(ng, boxes_after).sum())

        # ── save recoverable masks + triptych ──────────────────────────────
        np.savez_compressed(
            f"{_OUTDIR}/{iid}_entities.npz",
            before=np.stack(before) if before else np.zeros((0, H, W), bool),
            after=np.stack(after) if after else np.zeros((0, H, W), bool),
        )
        _save_triptych(img_bgr, before, after, usr["new_entity"],
                       f"{_OUTDIR}/{iid}_usr.png")

        rows.append(dict(
            iid=iid, H=H, W=W, dt=round(dt, 2),
            m_e_before=m_e_before, felz_K=usr["felz_K"],
            n_remaining=usr["n_remaining"], skipped_greedy=usr["skipped_greedy"],
            prompts_part=usr["prompts_part"], prompts_superpixel=usr["prompts_superpixel"],
            additional_masks=usr["additional_masks"],
            fused_into_existing=usr["fused_into_existing"], new_entity=usr["new_entity"],
            m_e_after=m_e_after, delta=delta,
            n_novel=len(ng), nov_before=nov_before, nov_after=nov_after,
        ))
        log.info(
            f"  {iid:>7}: M_E {m_e_before}->{m_e_after} (+{delta}) | S_R={usr['n_remaining']} "
            f"skip={usr['skipped_greedy']} prompts part/sp={usr['prompts_part']}/"
            f"{usr['prompts_superpixel']} addM={usr['additional_masks']} "
            f"fuse/new={usr['fused_into_existing']}/{usr['new_entity']} | "
            f"novel {len(ng)}: {nov_before}->{nov_after} [{dt:.1f}s]")

    # ── aggregate ───────────────────────────────────────────────────────────
    agg = lambda k: sum(r[k] for r in rows)
    log.info("\n" + "=" * 96)
    log.info("GATE E — USR AUDIT TABLE  (entities ADDED by USR; novel coverage before/after)")
    log.info("=" * 96)
    hdr = (f"  {'img':>7} {'Kfz':>4} {'S_R':>4} {'skip':>4} {'p.part':>6} {'p.sp':>5} "
           f"{'addM':>4} {'fuse':>4} {'new':>4} {'before':>6} {'after':>6} {'Δ':>4} "
           f"{'novGT':>5} {'nB':>3} {'nA':>3}")
    log.info(hdr)
    log.info("  " + "-" * (len(hdr) - 2))
    for r in rows:
        log.info(f"  {r['iid']:>7} {r['felz_K']:>4} {r['n_remaining']:>4} {r['skipped_greedy']:>4} "
                 f"{r['prompts_part']:>6} {r['prompts_superpixel']:>5} {r['additional_masks']:>4} "
                 f"{r['fused_into_existing']:>4} {r['new_entity']:>4} {r['m_e_before']:>6} "
                 f"{r['m_e_after']:>6} {r['delta']:>4} {r['n_novel']:>5} {r['nov_before']:>3} "
                 f"{r['nov_after']:>3}")
    log.info("  " + "-" * (len(hdr) - 2))
    NOV = agg("n_novel")
    log.info(f"  {'TOTAL':>7} {'':>4} {agg('n_remaining'):>4} {agg('skipped_greedy'):>4} "
             f"{agg('prompts_part'):>6} {agg('prompts_superpixel'):>5} {agg('additional_masks'):>4} "
             f"{agg('fused_into_existing'):>4} {agg('new_entity'):>4} {agg('m_e_before'):>6} "
             f"{agg('m_e_after'):>6} {agg('delta'):>4} {NOV:>5} {agg('nov_before'):>3} "
             f"{agg('nov_after'):>3}")

    log.info("")
    log.info(f"  ENTITIES ADDED by USR (Σ new_entity / Σ delta): {agg('new_entity')}  "
             f"({agg('m_e_before')} -> {agg('m_e_after')} entities, "
             f"{100*agg('new_entity')/max(agg('m_e_before'),1):+.1f}%)")
    log.info(f"  additional masks generated: {agg('additional_masks')}  "
             f"(fused into existing: {agg('fused_into_existing')}, new independent: {agg('new_entity')})")
    log.info(f"  prompt placement: part-centroid={agg('prompts_part')}  "
             f"superpixel-centroid={agg('prompts_superpixel')}")
    log.info(f"  NOVEL-GT coverage @IoU0.5 (Gate D box conv):  "
             f"BEFORE {agg('nov_before')}/{NOV} ({agg('nov_before')/NOV:.1%})  ->  "
             f"AFTER {agg('nov_after')}/{NOV} ({agg('nov_after')/NOV:.1%})  "
             f"[Δ novel = {agg('nov_after')-agg('nov_before'):+d}]")
    direction = ("MORE entities = WORSE flood (as expected)" if agg("new_entity") > 0
                 else "no entities added")
    log.info(f"  DIRECTION: {direction}")

    out = f"{_OUTDIR}/gate_e_stats.json"
    json.dump({
        "gate": "E", "model": "SAM-H",
        "locks": {"rho": _RHO, "uncovered_lt": _UNCOVERED, "contain_ge": _CONTAIN_MIN,
                  "delta": _DELTA, "tau": _TAU, "topk": _TOPK, "tau_sm": _TAU_SM,
                  "tau_encompass": _TAU_ENC, "theta_O": _THETA_O, "gamma_O": _GAMMA_O},
        "n_images": len(rows),
        "agg": {"m_e_before": agg("m_e_before"), "m_e_after": agg("m_e_after"),
                "entities_added": agg("new_entity"),
                "additional_masks": agg("additional_masks"),
                "fused_into_existing": agg("fused_into_existing"),
                "prompts_part": agg("prompts_part"),
                "prompts_superpixel": agg("prompts_superpixel"),
                "novel_gt": NOV, "nov_before": agg("nov_before"), "nov_after": agg("nov_after")},
        "per_image": rows,
    }, open(out, "w"), indent=2)
    log.info(f"\n  Stats → {out}")
    log.info(f"  Masks → {_OUTDIR}/{{id}}_entities.npz   PNGs → {_OUTDIR}/{{id}}_usr.png")
    log.info("STOP — USR run + ledger complete. No Gate D re-eval, no tuning.")


if __name__ == "__main__":
    main()
