"""
E-SAM end-to-end VISUAL runner — full pipeline (MMG+EMR+USR) on 12 diagnostic IDs.

Thin runner: imports esam_pipeline.run_esam. GT loading + plotting live ONLY here
(never in the timing runner). For each ID produces a 5-panel strip plus each panel
saved full native resolution, and a per-image funnel line.

Panels:  1 Original | 2 MMG (M̂32_O) | 3 EMR (M_E) |
         4 USR (all final_masks, USR-added in bright highlight) |
         5 final boxes (tight per entity, white) vs Gate D GT
           novel(green)/base(orange)/other(grey)/stuff(red — absent in
           instances_train2017, no stuff annotations).

FAITHFULNESS GUARD: assert len(emr_masks)==gate_c_stats and
len(final_masks)==gate_e_stats per image; ABORT on mismatch.

Run:  python diagnostics/esam_e2e_viz.py
"""

import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esam_pipeline import run_esam, DEVICE  # noqa: E402
from gate_d_eval import (  # noqa: E402
    ROOT, DIAG, NOVEL, BASE48, mask_to_box, recalled_flags,
)

_OUT = f"{ROOT}/diagnostics/output/esam_e2e"

# BGR colours for GT category buckets (panel 5)
_C_NOVEL = (0, 200, 0)      # green
_C_BASE  = (0, 140, 255)    # orange
_C_OTHER = (160, 160, 160)  # grey
_C_STUFF = (0, 0, 220)      # red (no stuff in instances_train2017 → won't fire)
_C_PROP  = (255, 255, 255)  # white — final-entity proposal boxes


def _gt_color(name):
    if name in NOVEL:
        return _C_NOVEL
    if name in BASE48:
        return _C_BASE
    return _C_OTHER


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


def _overlay_usr(img_bgr, final_masks, usr_added):
    """final masks muted; USR-added entities in bright yellow + white contour."""
    out = _overlay(img_bgr, final_masks, seed=7).astype(np.float32)
    for m in usr_added:
        if not m.any():
            continue
        for k, cv in enumerate((0, 255, 255)):   # bright yellow (BGR)
            out[:, :, k] = np.where(m, out[:, :, k] * 0.3 + cv * 0.7, out[:, :, k])
    out = out.astype(np.uint8)
    for m in usr_added:
        if not m.any():
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), 2)
    return out


def _boxes_panel(img_bgr, final_masks, gt_list):
    out = img_bgr.copy()
    # final-entity proposal boxes (thin white)
    for m in final_masks:
        b = mask_to_box(m)
        if b is None:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x0, y0), (x1, y1), _C_PROP, 1)
    # GT boxes coloured by bucket (thick)
    for g in gt_list:
        x0, y0, x1, y1 = [int(round(v)) for v in g["box"]]
        cv2.rectangle(out, (x0, y0), (x1, y1), _gt_color(g["name"]), 2)
    return out


def _panel(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                1, cv2.LINE_AA)
    return out


def main():
    os.makedirs(_OUT, exist_ok=True)

    # ── faithfulness targets ────────────────────────────────────────────────
    gc = json.load(open(f"{ROOT}/diagnostics/output/esam_gateC/gate_c_stats.json"))
    ge = json.load(open(f"{ROOT}/diagnostics/output/esam_gateE/gate_e_stats.json"))
    emr_target = {r["image_id"]: r["n_M_E_out"] for r in gc["per_image"]}
    final_target = {r["iid"]: r["m_e_after"] for r in ge["per_image"]}

    # ── GT (Gate D convention: full 80-class json) ──────────────────────────
    print("Loading full instances_train2017.json (~30s)…")
    full = json.load(open(f"{ROOT}/data/Annotations/instances_train2017.json"))
    cat = {c["id"]: c["name"] for c in full["categories"]}
    diag_set = set(DIAG)
    gt_all = defaultdict(list)        # all GT (box + name) per image, for panel 5
    novel_gt = defaultdict(list)      # novel boxes per image, for funnel
    for a in full["annotations"]:
        if a["image_id"] not in diag_set:
            continue
        x, y, w, h = a["bbox"]
        box = [x, y, x + w, y + h]
        name = cat[a["category_id"]]
        gt_all[a["image_id"]].append({"box": box, "name": name})
        if name in NOVEL:
            novel_gt[a["image_id"]].append(box)
    del full
    img_meta = {im["id"]: im for im in json.load(
        open(f"{ROOT}/data/Annotations/instances_train2017_12img_diag.json"))["images"]}

    # ── GPU safety + SAM-H (as the gates) ───────────────────────────────────
    used = torch.cuda.memory_allocated(DEVICE)
    total = torch.cuda.get_device_properties(DEVICE).total_memory
    print(f"GPU: {used/1e6:.0f} MB allocated / {total/1e6:.0f} MB total")
    if used > 2e9:
        raise RuntimeError("GPU already busy — stop.")
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=f"{ROOT}/weights/sam_vit_h_4b8939.pth")
    sam.to(DEVICE).eval()
    predictor = SamPredictor(sam)
    print("Loaded SAM-H\n")

    rows = []
    for iid in DIAG:
        im = img_meta[iid]
        H, W = im["height"], im["width"]
        img_bgr = cv2.imread(f"{ROOT}/data/Images/train2017/{im['file_name']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        r = run_esam(predictor, img_rgb, H, W)
        mmg, emr, added, final = (r["mmg_masks"], r["emr_masks"],
                                  r["usr_added"], r["final_masks"])

        # ── FAITHFULNESS GUARD ──────────────────────────────────────────────
        if len(emr) != emr_target[iid] or len(final) != final_target[iid]:
            raise SystemExit(
                f"ABORT {iid}: EMR {len(emr)} vs {emr_target[iid]} | "
                f"FINAL {len(final)} vs {final_target[iid]} — pipeline perturbed.")

        # ── novel-GT coverage per stage (Gate D box conv) ───────────────────
        ng = novel_gt.get(iid, [])
        mmg_boxes = [b for b in (mask_to_box(m) for m in mmg) if b is not None]
        emr_boxes = [b for b in (mask_to_box(m) for m in emr) if b is not None]
        fin_boxes = [b for b in (mask_to_box(m) for m in final) if b is not None]
        nov_mmg = int(recalled_flags(ng, mmg_boxes).sum())
        nov_emr = int(recalled_flags(ng, emr_boxes).sum())
        nov_fin = int(recalled_flags(ng, fin_boxes).sum())

        # ── panels (native resolution) ──────────────────────────────────────
        p1 = _panel(img_bgr, "1 original")
        p2 = _panel(_overlay(img_bgr, mmg, 1), f"2 MMG  M32_O ({len(mmg)})")
        p3 = _panel(_overlay(img_bgr, emr, 3), f"3 EMR  M_E ({len(emr)})")
        p4 = _panel(_overlay_usr(img_bgr, final, added),
                    f"4 USR  final ({len(final)})  +{len(added)} added")
        p5 = _panel(_boxes_panel(img_bgr, final, gt_all.get(iid, [])),
                    f"5 boxes  {len(fin_boxes)} prop | GT nov(g)/base(o)/other")

        # individual full-res
        idir = f"{_OUT}/{iid}"
        os.makedirs(idir, exist_ok=True)
        cv2.imwrite(f"{idir}/{iid}_1_original.png", p1)
        cv2.imwrite(f"{idir}/{iid}_2_mmg.png", p2)
        cv2.imwrite(f"{idir}/{iid}_3_emr.png", p3)
        cv2.imwrite(f"{idir}/{iid}_4_usr.png", p4)
        cv2.imwrite(f"{idir}/{iid}_5_boxes.png", p5)

        # combined strip (same height H → no downscaling)
        sep = np.full((H, 4, 3), 255, np.uint8)
        strip = np.hstack([p1, sep, p2, sep, p3, sep, p4, sep, p5])
        cv2.imwrite(f"{_OUT}/{iid}_e2e.png", strip)

        funnel = (f"MMG {len(mmg)} -> EMR {len(emr)} -> USR +{len(added)} -> "
                  f"final boxes {len(fin_boxes)} -> novel GT covered@0.5 "
                  f"{nov_fin}/{len(ng)} (MMG {nov_mmg} / EMR {nov_emr} / final {nov_fin})")
        print(f"  {iid:>7}: {funnel}")

        rows.append(dict(iid=iid, n_mmg=len(mmg), n_emr=len(emr), n_added=len(added),
                         n_final=len(final), n_final_boxes=len(fin_boxes),
                         n_novel=len(ng), nov_mmg=nov_mmg, nov_emr=nov_emr, nov_fin=nov_fin))

    # ── aggregate ───────────────────────────────────────────────────────────
    agg = lambda k: sum(r[k] for r in rows)
    print("\n" + "=" * 92)
    print("E-SAM E2E VIZ — AGGREGATE (entities per stage; novel-GT covered per stage)")
    print("=" * 92)
    print(f"  MMG M̂32_O total : {agg('n_mmg')}")
    print(f"  EMR M_E total    : {agg('n_emr')}")
    print(f"  USR added total  : {agg('n_added')}   -> final entities {agg('n_final')} "
          f"(final boxes {agg('n_final_boxes')})")
    NOV = agg("n_novel")
    print(f"  novel-GT covered @IoU0.5:  MMG {agg('nov_mmg')}/{NOV}  ->  "
          f"EMR {agg('nov_emr')}/{NOV}  ->  final {agg('nov_fin')}/{NOV}  "
          f"[USR recovery = {agg('nov_fin') - agg('nov_emr'):+d}]")
    out = f"{_OUT}/esam_e2e_viz.json"
    json.dump({"model": "SAM-H", "n_images": len(rows),
               "agg": {"mmg": agg("n_mmg"), "emr": agg("n_emr"), "usr_added": agg("n_added"),
                       "final": agg("n_final"), "final_boxes": agg("n_final_boxes"),
                       "novel_gt": NOV, "nov_mmg": agg("nov_mmg"),
                       "nov_emr": agg("nov_emr"), "nov_fin": agg("nov_fin")},
               "per_image": rows}, open(out, "w"), indent=2)
    print(f"\n  Strips -> {_OUT}/{{id}}_e2e.png   Panels -> {_OUT}/{{id}}/{{id}}_*.png")
    print(f"  Stats  -> {out}")
    print("STOP — viz runner complete. No tuning.")


if __name__ == "__main__":
    main()
