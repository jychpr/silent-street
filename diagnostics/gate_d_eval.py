"""
E-SAM Gate D — proposal-recall PROXY on 12 diagnostic images. NOT an AP_novel result.

FILTERED-vs-FILTERED comparison with asymmetric filters:
  - OLN comparator  = OW_COCO_R2.json as shipped (learned FE/Weibull over 3 rounds,
                      then thresholded → 54 pseudo-label boxes across the 12 IDs).
  - E-SAM           = Gate C M_E entities (re-run in-memory; masks were not persisted),
                      boxed tight axis-aligned. Two sets:
                        raw       — every entity
                        filtered  — single geometric filter
                                    (min_side=4, max_area_ratio=0.4, max_aspect=5.0)

Metrics at IoU 0.5 (box IoU), per image + aggregate:
  TABLE A  recall of 198 novel + 341 base GT, three columns (OLN | raw | filtered),
           box counts beside each; novel gain (filt∖OLN) and regression (OLN∖filt).
  TABLE B  cost side — foreground-merge (one box ≥2 GT THINGs @IoU≥0.3), 92869
           horse+rider flag, 92869 dog/umbrella survival, junk proxy (boxes hitting
           zero GT of any class @IoU≥0.5).

Run:
  python diagnostics/gate_d_eval.py
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
from esam_gate_c import _run_emr  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DIAG = [157105, 122263, 543882, 579329, 443084, 92869, 475808, 435091, 171270, 307238, 30, 34]

NOVEL = {"airplane", "bus", "cat", "dog", "cow", "elephant", "umbrella", "tie", "snowboard",
         "skateboard", "cup", "knife", "cake", "couch", "keyboard", "sink", "scissors"}
BASE48 = {"person", "bicycle", "car", "motorcycle", "train", "truck", "boat", "bench", "bird",
          "horse", "sheep", "bear", "zebra", "giraffe", "backpack", "handbag", "suitcase",
          "frisbee", "skis", "kite", "surfboard", "bottle", "fork", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "chair", "bed",
          "toilet", "tv", "laptop", "mouse", "remote", "microwave", "oven", "toaster",
          "refrigerator", "book", "clock", "vase", "toothbrush"}
THING = {"person", "dog", "cat", "horse", "cow", "sheep", "elephant", "bear", "zebra",
         "giraffe", "bird"}

# Gate D geometric filter (asymmetric vs OLN's learned filter)
MIN_SIDE, MAX_AREA_RATIO, MAX_ASPECT = 4, 0.4, 5.0
IOU_REC, IOU_MERGE = 0.5, 0.3

DEVICE = "cuda:0"
ROOT = "/home/akihito/JC/lab/silent-street"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes → (len(A), len(B))."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.float64)
    A = np.asarray(A, float); B = np.asarray(B, float)
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def mask_to_box(m: np.ndarray):
    """Tight axis-aligned xyxy (continuous: xmax+1, ymax+1) or None if empty."""
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def passes_geo(b, W, H) -> bool:
    w, h = b[2] - b[0], b[3] - b[1]
    if min(w, h) < MIN_SIDE:
        return False
    if (w * h) / (W * H) > MAX_AREA_RATIO:
        return False
    if max(w, h) / max(min(w, h), 1e-9) > MAX_ASPECT:
        return False
    return True


def recalled_flags(gt_boxes, prop_boxes, thr=IOU_REC):
    """Per-GT bool: is it recalled by any proposal at IoU>=thr."""
    if len(gt_boxes) == 0:
        return np.zeros(0, dtype=bool)
    if len(prop_boxes) == 0:
        return np.zeros(len(gt_boxes), dtype=bool)
    return iou_matrix(np.array(gt_boxes), np.array(prop_boxes)).max(axis=1) >= thr


def greedy_survival(gt_boxes, prop_boxes, thr=IOU_REC) -> int:
    """#GT matched to DISTINCT proposals (greedy by IoU desc)."""
    if not gt_boxes or not prop_boxes:
        return 0
    M = iou_matrix(np.array(gt_boxes), np.array(prop_boxes))
    pairs = sorted(((M[i, j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1])
                    if M[i, j] >= thr), reverse=True)
    used_g, used_p = set(), set()
    for _, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j)
    return len(used_g)


# ---------------------------------------------------------------------------
# STEP 1 — rebuild M_E in-memory and extract E-SAM boxes
# ---------------------------------------------------------------------------

def esam_entities(predictor, img_rgb, H, W):
    """Re-run Gate C MMG+EMR; return list of (H,W) bool entity masks."""
    predictor.set_image(img_rgb)
    m32_full, m32_256, iou32, _ = _run_point_grid(predictor, 32, H, W, DEVICE)
    m32_O, m32_B, _, _, sc32_O, _ = _categorize(m32_full, m32_256, iou32)
    nms_idx = _naive_nms(m32_O, sc32_O, _THETA_O, DEVICE)
    m_O_nms, sc_O_nms = m32_O[nms_idx], sc32_O[nms_idx]
    flt = _best_map_filter(m_O_nms, m32_B, _GAMMA_O, DEVICE)
    m_hat_O, sc_hat_O = m_O_nms[flt], sc_O_nms[flt]
    M_S = felzenszwalb(img_rgb, scale=_FELZ_SCALE, sigma=_FELZ_SIGMA, min_size=_FELZ_MIN_SIZE)
    m64_full, m64_256, iou64, pts64 = _run_point_grid(predictor, 64, H, W, DEVICE)
    _, _, m64_O_256, m64_B_256, sc64_O, sc64_B = _categorize(m64_full, m64_256, iou64)
    emr = _run_emr(m_hat_O, sc_hat_O, m64_O_256, m64_B_256, sc64_O, sc64_B,
                   pts64, M_S, predictor, H, W, DEVICE)
    del m32_full, m32_256, m64_full, m64_256, m64_O_256, m64_B_256
    torch.cuda.empty_cache()
    return [e.numpy().astype(bool) for e in emr["entities"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    diag_set = set(DIAG)

    # ── load GT (full, 80-class) + OLN-R2 comparator ────────────────────────
    log.info("Loading full instances_train2017.json (~30s)…")
    full = json.load(open(f"{ROOT}/data/Annotations/instances_train2017.json"))
    cat = {c["id"]: c["name"] for c in full["categories"]}
    gt = defaultdict(lambda: {"novel": [], "base": [], "all": [], "thing": [], "by": defaultdict(list)})
    for a in full["annotations"]:
        if a["image_id"] not in diag_set:
            continue
        x, y, w, h = a["bbox"]
        box = [x, y, x + w, y + h]
        nm = cat[a["category_id"]]
        g = gt[a["image_id"]]
        g["all"].append(box)
        if nm in NOVEL: g["novel"].append(box)
        elif nm in BASE48: g["base"].append(box)
        if nm in THING: g["thing"].append(box)
        g["by"][nm].append(box)
    del full

    log.info("Loading OW_COCO_R2.json…")
    r2 = json.load(open(f"{ROOT}/ow_labels/OW_COCO_R2.json"))
    oln = defaultdict(list)
    for o in r2:
        if o["image_id"] in diag_set:
            x, y, w, h = o["bbox"]
            oln[o["image_id"]].append([x, y, x + w, y + h])
    del r2

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
    log.info(f"Geo filter: min_side={MIN_SIDE} max_area_ratio={MAX_AREA_RATIO} "
             f"max_aspect={MAX_ASPECT} | IoU_rec={IOU_REC} IoU_merge={IOU_MERGE}")
    log.info("=" * 90)

    rows = []
    for iid in DIAG:
        im = img_meta[iid]
        H, W = im["height"], im["width"]
        img_rgb = cv2.cvtColor(cv2.imread(f"{ROOT}/data/Images/train2017/{im['file_name']}"),
                               cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        ents = esam_entities(predictor, img_rgb, H, W)
        dt = time.perf_counter() - t0

        raw_boxes = [b for b in (mask_to_box(e) for e in ents) if b is not None]
        filt_boxes = [b for b in raw_boxes if passes_geo(b, W, H)]
        oln_boxes = oln.get(iid, [])
        g = gt[iid]

        # recall flags per GT instance
        nov_oln = recalled_flags(g["novel"], oln_boxes)
        nov_raw = recalled_flags(g["novel"], raw_boxes)
        nov_flt = recalled_flags(g["novel"], filt_boxes)
        bas_oln = recalled_flags(g["base"], oln_boxes)
        bas_raw = recalled_flags(g["base"], raw_boxes)
        bas_flt = recalled_flags(g["base"], filt_boxes)
        gain = int(np.sum(nov_flt & ~nov_oln))
        regr = int(np.sum(nov_oln & ~nov_flt))

        # cost side: foreground-merge (one box ≥2 distinct GT THINGs @IoU≥0.3)
        def merge_count(boxes):
            if not boxes or not g["thing"]:
                return 0
            M = iou_matrix(np.array(boxes), np.array(g["thing"]))  # (nbox, nthing)
            return int(np.sum((M >= IOU_MERGE).sum(axis=1) >= 2))
        merge_raw, merge_flt = merge_count(raw_boxes), merge_count(filt_boxes)

        # junk proxy: boxes hitting zero GT of ANY class @IoU≥0.5
        def junk_frac(boxes):
            if not boxes:
                return 0.0, 0
            M = iou_matrix(np.array(boxes), np.array(g["all"]))
            j = int(np.sum(M.max(axis=1) < IOU_REC))
            return j / len(boxes), j
        junk_raw_f, junk_raw_n = junk_frac(raw_boxes)
        junk_flt_f, junk_flt_n = junk_frac(filt_boxes)

        rows.append(dict(
            iid=iid, H=H, W=W, dt=dt,
            n_oln=len(oln_boxes), n_raw=len(raw_boxes), n_filt=len(filt_boxes),
            n_nov=len(g["novel"]), n_bas=len(g["base"]),
            nov_oln=int(nov_oln.sum()), nov_raw=int(nov_raw.sum()), nov_flt=int(nov_flt.sum()),
            bas_oln=int(bas_oln.sum()), bas_raw=int(bas_raw.sum()), bas_flt=int(bas_flt.sum()),
            gain=gain, regr=regr,
            merge_raw=merge_raw, merge_flt=merge_flt,
            junk_raw_f=junk_raw_f, junk_raw_n=junk_raw_n, junk_raw_tot=len(raw_boxes),
            junk_flt_f=junk_flt_f, junk_flt_n=junk_flt_n, junk_flt_tot=len(filt_boxes),
        ))
        log.info(f"  {iid:>7}: ents={len(ents):>3} raw={len(raw_boxes):>3} "
                 f"filt={len(filt_boxes):>3} oln={len(oln_boxes):>2} | "
                 f"novel {len(g['novel'])}: oln={int(nov_oln.sum())} raw={int(nov_raw.sum())} "
                 f"filt={int(nov_flt.sum())} (gain={gain} regr={regr}) | "
                 f"merge r/f={merge_raw}/{merge_flt} junk r/f={junk_raw_f:.0%}/{junk_flt_f:.0%} "
                 f"[{dt:.1f}s]")

        # ── 92869 special probes ────────────────────────────────────────────
        if iid == 92869:
            horses = g["by"].get("horse", [])
            persons = g["by"].get("person", [])
            def horse_rider(boxes):
                if not boxes or not horses or not persons:
                    return False
                Mh = iou_matrix(np.array(boxes), np.array(horses)).max(axis=1)
                Mp = iou_matrix(np.array(boxes), np.array(persons)).max(axis=1)
                return bool(np.any((Mh >= IOU_MERGE) & (Mp >= IOU_MERGE)))
            hr_raw, hr_flt = horse_rider(raw_boxes), horse_rider(filt_boxes)
            dogs, umbs = g["by"].get("dog", []), g["by"].get("umbrella", [])
            dog_surv = greedy_survival(dogs, filt_boxes)
            umb_surv = greedy_survival(umbs, filt_boxes)
            rows[-1].update(hr_raw=hr_raw, hr_flt=hr_flt,
                            dog_surv=dog_surv, dog_tot=len(dogs),
                            umb_surv=umb_surv, umb_tot=len(umbs))

    # ===================================================================== A
    def agg(k): return sum(r[k] for r in rows)
    log.info("\n" + "=" * 96)
    log.info("TABLE A — RECALL @ IoU 0.5  (FILTERED-vs-FILTERED; box counts beside recall)")
    log.info("=" * 96)
    log.info(f"  {'img':>7} | {'GTn':>3} {'OLN n/rec':>10} {'RAW n/rec':>11} {'FILT n/rec':>11} "
             f"| {'gain':>4} {'regr':>4} || {'GTb':>3} {'oln':>3} {'raw':>3} {'flt':>3}")
    log.info("  " + "-" * 92)
    for r in rows:
        log.info(f"  {r['iid']:>7} | {r['n_nov']:>3} "
                 f"{r['n_oln']:>4}/{r['nov_oln']:<5} {r['n_raw']:>5}/{r['nov_raw']:<5} "
                 f"{r['n_filt']:>5}/{r['nov_flt']:<5} | {r['gain']:>4} {r['regr']:>4} || "
                 f"{r['n_bas']:>3} {r['bas_oln']:>3} {r['bas_raw']:>3} {r['bas_flt']:>3}")
    log.info("  " + "-" * 92)
    NOV, BAS = agg("n_nov"), agg("n_bas")
    log.info(f"  {'TOTAL':>7} | {NOV:>3} "
             f"{agg('n_oln'):>4}/{agg('nov_oln'):<5} {agg('n_raw'):>5}/{agg('nov_raw'):<5} "
             f"{agg('n_filt'):>5}/{agg('nov_flt'):<5} | {agg('gain'):>4} {agg('regr'):>4} || "
             f"{BAS:>3} {agg('bas_oln'):>3} {agg('bas_raw'):>3} {agg('bas_flt'):>3}")
    log.info(f"\n  NOVEL recall:  OLN {agg('nov_oln')}/{NOV} ({agg('nov_oln')/NOV:.1%}, {agg('n_oln')} boxes)"
             f"  |  RAW {agg('nov_raw')}/{NOV} ({agg('nov_raw')/NOV:.1%}, {agg('n_raw')} boxes)"
             f"  |  FILT {agg('nov_flt')}/{NOV} ({agg('nov_flt')/NOV:.1%}, {agg('n_filt')} boxes)")
    log.info(f"  BASE  recall:  OLN {agg('bas_oln')}/{BAS} ({agg('bas_oln')/BAS:.1%})"
             f"  |  RAW {agg('bas_raw')}/{BAS} ({agg('bas_raw')/BAS:.1%})"
             f"  |  FILT {agg('bas_flt')}/{BAS} ({agg('bas_flt')/BAS:.1%})")
    log.info(f"  novel GAIN (filt∖OLN) = {agg('gain')}   REGRESSION (OLN∖filt) = {agg('regr')}")

    # ===================================================================== B
    log.info("\n" + "=" * 96)
    log.info("TABLE B — COST SIDE")
    log.info("=" * 96)
    log.info(f"  {'img':>7} | {'fg-merge raw':>12} {'fg-merge filt':>13} | "
             f"{'junk raw':>14} {'junk filt':>14}")
    log.info("  " + "-" * 88)
    for r in rows:
        log.info(f"  {r['iid']:>7} | {r['merge_raw']:>12} {r['merge_flt']:>13} | "
                 f"{r['junk_raw_n']:>4}/{r['junk_raw_tot']:<4} ({r['junk_raw_f']:>4.0%}) "
                 f"{r['junk_flt_n']:>4}/{r['junk_flt_tot']:<4} ({r['junk_flt_f']:>4.0%})")
    log.info("  " + "-" * 88)
    tot_jr, tot_jrn = agg("junk_raw_tot"), agg("junk_raw_n")
    tot_jf, tot_jfn = agg("junk_flt_tot"), agg("junk_flt_n")
    log.info(f"  {'TOTAL':>7} | {'merge raw='+str(agg('merge_raw')):>12} "
             f"{'filt='+str(agg('merge_flt')):>13} | "
             f"junk raw {tot_jrn}/{tot_jr} ({tot_jrn/max(tot_jr,1):.0%})  "
             f"filt {tot_jfn}/{tot_jf} ({tot_jfn/max(tot_jf,1):.0%})")

    r92 = next(r for r in rows if r["iid"] == 92869)
    log.info("\n  92869 PROBES:")
    log.info(f"    horse+rider single-box merge:  raw={r92['hr_raw']}  filt={r92['hr_flt']}  "
             f"(one E-SAM box covering GT-horse AND GT-person @IoU≥{IOU_MERGE})")
    log.info(f"    novel survival (distinct E-SAM-filtered @IoU≥0.5):  "
             f"dogs {r92['dog_surv']}/{r92['dog_tot']}   umbrellas {r92['umb_surv']}/{r92['umb_tot']}")

    out = f"{ROOT}/diagnostics/output/esam_gateC/gate_d_eval.json"
    json.dump({"filter": dict(min_side=MIN_SIDE, max_area_ratio=MAX_AREA_RATIO,
                              max_aspect=MAX_ASPECT, iou_rec=IOU_REC, iou_merge=IOU_MERGE),
               "rows": rows,
               "agg": {"novel": NOV, "base": BAS,
                       "nov_oln": agg("nov_oln"), "nov_raw": agg("nov_raw"), "nov_flt": agg("nov_flt"),
                       "bas_oln": agg("bas_oln"), "bas_raw": agg("bas_raw"), "bas_flt": agg("bas_flt"),
                       "gain": agg("gain"), "regr": agg("regr"),
                       "n_oln": agg("n_oln"), "n_raw": agg("n_raw"), "n_filt": agg("n_filt")}},
              open(out, "w"), indent=2, default=str)
    log.info(f"\n  → {out}")


if __name__ == "__main__":
    main()
