"""
E-SAM CONSOLIDATED PIPELINE — single source of truth for the full faithful
end-to-end E-SAM:  MMG(+P/SP)  →  EMR  →  USR.

This module does NOT re-derive or re-implement any equation. It IMPORTS the
already-approved, gate-verified implementations verbatim and only adds the
consolidation API (run_esam), per-stage timing, and USR provenance tracking:

    MMG (Eq.1-2)  : esam_gate_b._run_point_grid / _categorize / _naive_nms /
                    _best_map_filter      (imported unchanged)
    EMR (Eq.5-8)  : esam_gate_c._run_emr  (imported unchanged)
    USR (Eq.9-10) : esam_gate_e._run_usr  (imported unchanged)

Because the equation code is imported (not copied), it is byte-identical to the
gate originals by construction — there is no diff to report. The faithfulness
self-test (`python esam_pipeline.py`) HARD-ASSERTS that run_esam reproduces the
locked Gate C M_E counts and the Gate E post-USR counts exactly.

Authority for every quote in the ledger below: references/E-SAM_2503.12094v1.pdf
(pp. 4-5, Eq.1-2 / Eq.5-8 / Eq.9-10; Table 4 for ρ). The CHOICE flags are
carried forward VERBATIM from the Gate E ledger — none are silently re-decided.
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from skimage.segmentation import felzenszwalb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── MMG (Eq.1-2) — imported verbatim from Gate B ────────────────────────────
from esam_gate_b import (  # noqa: E402
    _run_point_grid, _categorize, _naive_nms, _best_map_filter,
    _THETA_O, _GAMMA_O, _FELZ_SCALE, _FELZ_SIGMA, _FELZ_MIN_SIZE,
)
# ── EMR (Eq.5-8) — imported verbatim from Gate C ────────────────────────────
from esam_gate_c import (  # noqa: E402
    _run_emr, _DELTA, _TAU, _TOPK, _TAU_SM, _TAU_ENC,
)
# ── USR (Eq.9-10) — imported verbatim from Gate E ───────────────────────────
from esam_gate_e import (  # noqa: E402
    _run_usr, _RHO, _UNCOVERED, _CONTAIN_MIN,
)
from gate_d_eval import ROOT, DIAG  # noqa: E402

DEVICE = "cuda:0"


# ===========================================================================
# STEP 1 — the single consolidated entry point
# ===========================================================================

def run_esam(predictor, img_rgb, H, W):
    """
    Full E-SAM pipeline on one image. Pipeline work ONLY — no file I/O, no
    plotting, no GT. `predictor` is a SamPredictor; this function calls
    set_image itself (timed as the encoder stage).

    Returns dict:
      'mmg_masks'   : list of M̂32_O object masks            (bool HxW numpy)
      'emr_masks'   : list of M_E entities after EMR          (bool HxW numpy)
      'usr_added'   : list of entities USR appended as NEW independent entities
                      (Eq.10 else-branch, iou<=rho ONLY; fused-in-place excluded)
      'final_masks' : list of final entities (EMR + USR)      (bool HxW numpy)
      'timing'      : {t_encoder, t_mmg, t_emr, t_usr, t_total} seconds, each
                      wrapped in torch.cuda.synchronize(), excluding I/O/plotting.
    """
    def _sync():
        if "cuda" in DEVICE:
            torch.cuda.synchronize()

    # ── Encoder ─────────────────────────────────────────────────────────────
    _sync(); t0 = time.perf_counter()
    predictor.set_image(img_rgb)
    _sync(); t_encoder = time.perf_counter() - t0

    # ── MMG (Eq.1-2) + P/SP surfacing + Felzenszwalb M_S + 64-grid gallery ──
    # Call sequence is byte-identical to esam_gate_e._mmg_emr_with_parts, minus
    # the hoisted set_image (timed above) and plus the read-only M̂32_O surfacing.
    _sync(); t0 = time.perf_counter()
    m32_full, m32_256, iou32, _ = _run_point_grid(predictor, 32, H, W, DEVICE)
    (m32_O, m32_B, _, _, sc32_O, _,
     m32_P, m32_SP) = _categorize(m32_full, m32_256, iou32, return_parts=True)
    nms_idx = _naive_nms(m32_O, sc32_O, _THETA_O, DEVICE)
    m_O_nms, sc_O_nms = m32_O[nms_idx], sc32_O[nms_idx]
    flt = _best_map_filter(m_O_nms, m32_B, _GAMMA_O, DEVICE)
    m_hat_O, sc_hat_O = m_O_nms[flt], sc_O_nms[flt]

    M_S = felzenszwalb(img_rgb, scale=_FELZ_SCALE, sigma=_FELZ_SIGMA,
                       min_size=_FELZ_MIN_SIZE)

    m64_full, m64_256, iou64, pts64 = _run_point_grid(predictor, 64, H, W, DEVICE)
    _, _, m64_O_256, m64_B_256, sc64_O, sc64_B = _categorize(m64_full, m64_256, iou64)

    mmg_masks = [m.numpy().astype(bool) for m in m_hat_O.cpu()]   # M̂32_O (Eq.2)
    m32_P_np = m32_P.numpy().astype(bool)                        # Eq.1 part level
    m32_SP_np = m32_SP.numpy().astype(bool)                      # Eq.1 subpart level
    _sync(); t_mmg = time.perf_counter() - t0

    del m32_full, m32_256, m64_full, m64_256, m32_P, m32_SP
    if "cuda" in DEVICE:
        torch.cuda.empty_cache()

    # ── EMR (Eq.5-8) ────────────────────────────────────────────────────────
    _sync(); t0 = time.perf_counter()
    emr = _run_emr(m_hat_O, sc_hat_O, m64_O_256, m64_B_256, sc64_O, sc64_B,
                   pts64, M_S, predictor, H, W, DEVICE)
    _sync(); t_emr = time.perf_counter() - t0
    emr_masks = [e.numpy().astype(bool) for e in emr["entities"]]

    del m_hat_O, m64_O_256, m64_B_256
    if "cuda" in DEVICE:
        torch.cuda.empty_cache()

    # ── USR (Eq.9-10) ───────────────────────────────────────────────────────
    _sync(); t0 = time.perf_counter()
    usr = _run_usr(emr_masks, m32_P_np, m32_SP_np, M_S, predictor, H, W)
    _sync(); t_usr = time.perf_counter() - t0
    final_masks = usr["entities_after"]

    # USR provenance — NOT a post-hoc set-difference against EMR. _run_usr fuses
    # into existing entities IN PLACE (entities[best_j] |= M_A) and APPENDS the
    # Eq.10 else-branch (iou<=rho) new-independent entities to the tail. So the
    # tail slice [len(emr):] is exactly the appended new entities, captured by
    # the append order produced DURING USR. (Validated by the Gate E +5 trace,
    # which classified entities purely by this index alignment.) Length check
    # below ties it to USR's own new_entity counter.
    n_added = usr["new_entity"]
    usr_added = final_masks[len(emr_masks):]
    assert len(usr_added) == n_added, (
        f"USR provenance mismatch: tail slice {len(usr_added)} != new_entity {n_added}")

    t_total = t_encoder + t_mmg + t_emr + t_usr
    return {
        "mmg_masks": mmg_masks,
        "emr_masks": emr_masks,
        "usr_added": usr_added,
        "final_masks": final_masks,
        "timing": {
            "t_encoder": t_encoder, "t_mmg": t_mmg, "t_emr": t_emr,
            "t_usr": t_usr, "t_total": t_total,
        },
    }


# ===========================================================================
# STEP 0 — re-verification ledger (printed by the self-test)
# ===========================================================================

# (equation, short paper quote [pp.4-5], "matches <file>:<func>", flag)
_LEDGER = [
    ("Eq.1  O/P/SP",
     "M_i^32={M_{i,O},M_{i,P},M_{i,SP}}, A_{i,O}>=A_{i,P}>=A_{i,SP}; "
     "eps_{i,B}=argmax s_{i,eps}",
     "esam_gate_b.py:_categorize (area-descending order[:,0]=O,[:,1]=P,"
     "[:,2]=SP; best=iou_preds.argmax)", "PAPER"),
    ("Eq.2  M̂32_O",
     "MMG first applies naive NMS with a high threshold theta_O ... "
     "max IoU(M_{i,O},M_{*,B})>=gamma_O => M_{i,O} in M̂_O^32",
     "esam_gate_b.py:_naive_nms (theta_O=0.8) + :_best_map_filter "
     "(gamma_O=0.6)", "PAPER (theta_O=0.8, gamma_O=0.6 Table 4)"),
    ("Eq.5  split",
     "OR_p^q=M_p^32 ∩ M_q^32 ... if area of OR relative to the largest mask "
     "is below threshold delta, the overlapping region in the larger mask is "
     "removed",
     "esam_gate_c.py:_run_emr split phase (ratio=a_or/max(a_p,a_q); "
     "ratio<delta -> masks[larger] &= ~OR)", "PAPER (delta=0.05 Table 4)"),
    ("Eq.6  guidance",
     "G_p=M_p^64O if S_p^64B-S_p^64O<tau else M_p^64B; when multiple prompts "
     "present, the mask that appears most frequently is chosen",
     "esam_gate_c.py:_run_emr ((sc64_B-sc64_O)<tau; majority O/B; "
     "highest-conf prompt mask)", "PAPER (tau=0.1 Table 4)"),
    ("Eq.7  S_M",
     "S_M(i,j)=(1/|C_i|) Σ_{c_i∈C_i} |{c_j | c_j∈C_j ∧ c_j∈Top_k(S_C(c_i,·))}|",
     "esam_gate_c.py:_adjacent_mask_similarity (superpixel centroids, "
     "Top_k=3)", "k=3 CHOICE (paper silent on k)"),
    ("Eq.8  merge gate",
     "M_E=M_a^64O ∪ M_*B^64 if M_a^64,M_*^64∈G and match exists; "
     "{M_a^32,M_b^32} otherwise (keep separate)",
     "esam_gate_c.py:_gallery_encompass (single gallery mask covers >=tau_enc "
     "of BOTH; candidate if S_M>=tau_sm)",
     "tau_sm=0.5, tau_enc=0.5 CHOICE (paper silent); keep-separate default"),
    ("Eq.9  USR prompt",
     "P_A^i=centroid(M_P∪M_SP) if S_R^i ⊆ (M_P∪M_SP); else centroid(S_R^i). "
     "USR considers regions in M_S not covered by M_E -> S_R^k",
     "esam_gate_e.py:_run_usr (uncovered<0.10; contained>=0.90 -> "
     "smallest containing P/SP centroid; else superpixel centroid)",
     "placement=PAPER; uncovered<0.10 CHOICE (strict override); "
     "contained>=0.90 CHOICE; smallest container CHOICE"),
    ("Eq.10 fusion",
     "M̌_E^k=M_E ∪ M_A^k if iou(M_E^k,M_A^k)>rho else M_A^k. Naive greedy "
     "algorithm ... use fewer masks from M_A^k",
     "esam_gate_e.py:_run_usr (max mask-IoU(M_A,entity)>rho -> merge in place; "
     "else append new; greedy largest-uncovered-first + skip)",
     "rho=0.1 PAPER (Table 4); best-level M_A CHOICE; greedy-skip CHOICE"),
]


def print_ledger():
    print("=" * 96)
    print("E-SAM CONSOLIDATED PIPELINE — STEP 0 RE-VERIFICATION LEDGER")
    print("authority: references/E-SAM_2503.12094v1.pdf  pp.4-5 (Eq.1-2/5-8/9-10), Table 4")
    print("equation code IMPORTED verbatim from gates -> byte-identical, no diff to report")
    print("=" * 96)
    for eq, quote, code, flag in _LEDGER:
        print(f"\n  [{eq}]")
        print(f"    paper : \"{quote}\"")
        print(f"    code  : matches {code}")
        print(f"    flag  : {flag}")
    print("\n" + "=" * 96)
    print("  CHOICE flags carried forward verbatim from the Gate E ledger:")
    print(f"    k(top)={_TOPK}  tau_sm={_TAU_SM}  tau_enc={_TAU_ENC}  "
          f"uncovered<{_UNCOVERED}  contain>={_CONTAIN_MIN}  (rho={_RHO} PAPER)")
    print("=" * 96)


# ===========================================================================
# STEP 2 — faithfulness self-test (HARD-ASSERT vs locked gate results)
# ===========================================================================

def _self_test():
    # Targets loaded directly from the persisted gate stats (source of truth).
    gc = json.load(open(f"{ROOT}/diagnostics/output/esam_gateC/gate_c_stats.json"))
    ge = json.load(open(f"{ROOT}/diagnostics/output/esam_gateE/gate_e_stats.json"))
    emr_target = {r["image_id"]: r["n_M_E_out"] for r in gc["per_image"]}
    final_target = {r["iid"]: r["m_e_after"] for r in ge["per_image"]}

    img_meta = {im["id"]: im for im in json.load(
        open(f"{ROOT}/data/Annotations/instances_train2017_12img_diag.json"))["images"]}

    # GPU safety (Rule 7)
    used = torch.cuda.memory_allocated(DEVICE)
    total = torch.cuda.get_device_properties(DEVICE).total_memory
    print(f"\nGPU: {used/1e6:.0f} MB allocated / {total/1e6:.0f} MB total")
    if used > 2e9:
        raise RuntimeError("GPU already busy — stop.")

    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=f"{ROOT}/weights/sam_vit_h_4b8939.pth")
    sam.to(DEVICE).eval()
    predictor = SamPredictor(sam)
    print("Loaded SAM-H\n")
    print(f"{'img':>8} {'mmg':>4} {'emr':>4}/{'tgt':<4} {'final':>5}/{'tgt':<5} "
          f"{'+usr':>4} {'enc':>5} {'mmg':>5} {'emr':>5} {'usr':>5} {'tot':>6}  ok")
    print("-" * 88)

    fails = []
    for iid in DIAG:
        im = img_meta[iid]
        H, W = im["height"], im["width"]
        img_rgb = cv2.cvtColor(
            cv2.imread(f"{ROOT}/data/Images/train2017/{im['file_name']}"),
            cv2.COLOR_BGR2RGB)
        r = run_esam(predictor, img_rgb, H, W)
        n_emr, n_fin, n_add = len(r["emr_masks"]), len(r["final_masks"]), len(r["usr_added"])
        t = r["timing"]
        emr_ok = n_emr == emr_target[iid]
        fin_ok = n_fin == final_target[iid]
        ok = emr_ok and fin_ok
        if not ok:
            fails.append((iid, n_emr, emr_target[iid], n_fin, final_target[iid]))
        print(f"{iid:>8} {len(r['mmg_masks']):>4} {n_emr:>4}/{emr_target[iid]:<4} "
              f"{n_fin:>5}/{final_target[iid]:<5} {n_add:>4} "
              f"{t['t_encoder']:>5.2f} {t['t_mmg']:>5.2f} {t['t_emr']:>5.2f} "
              f"{t['t_usr']:>5.2f} {t['t_total']:>6.2f}  "
              f"{'OK' if ok else 'FAIL'}")

    print("-" * 88)
    if fails:
        for iid, ne, te, nf, tf in fails:
            print(f"  MISMATCH {iid}: EMR {ne} vs {te} | FINAL {nf} vs {tf}")
        raise SystemExit(
            f"FAITHFULNESS SELF-TEST FAILED on {len(fails)} image(s) — "
            f"consolidated module does NOT reproduce gate results. Fix before any runner uses it.")
    print("FAITHFULNESS SELF-TEST PASSED — all 12 images reproduce Gate C M_E and "
          "Gate E post-USR counts EXACTLY.")
    print("STOP — ledger + self-test complete. Runner scripts are a separate instruction.")


if __name__ == "__main__":
    print_ledger()
    _self_test()
