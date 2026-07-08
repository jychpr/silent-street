"""
E-SAM PANEL-5 VARIANTS — three box views per diagnostic image. Visualization only;
reuses esam_pipeline.run_esam and the EXACT box logic from esam_e2e_viz. No pipeline
re-run beyond the inference call, no tuning, no metric change.

Per image, three versions of the panel-5 box view:
  (a) both  — model proposal boxes (white) + GT boxes (green novel/orange base/grey other)
  (b) model — ONLY the white E-SAM final-entity proposal boxes (mask_to_box(final_masks))
  (c) gt    — ONLY the GT boxes, coloured by bucket

FAITHFULNESS GUARD: assert len(emr_masks)==gate_c_stats and len(final_masks)==gate_e_stats
per image; ABORT on mismatch.

Run:  python diagnostics/esam_boxes_variants.py
"""

import json
import os
import sys
from collections import defaultdict

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esam_pipeline import run_esam, DEVICE  # noqa: E402
from gate_d_eval import ROOT, DIAG, NOVEL, mask_to_box  # noqa: E402
# Reuse the EXACT box-drawing primitives from the e2e viz runner.
from esam_e2e_viz import _gt_color, _panel, _boxes_panel, _C_PROP  # noqa: E402

_OUT = f"{ROOT}/diagnostics/output/esam_boxes_variants"


def _model_only_panel(img_bgr, final_masks):
    """ONLY white final-entity proposal boxes — same draw as _boxes_panel's model loop."""
    out = img_bgr.copy()
    for m in final_masks:
        b = mask_to_box(m)
        if b is None:
            continue
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x0, y0), (x1, y1), _C_PROP, 1)
    return out


def _gt_only_panel(img_bgr, gt_list):
    """ONLY GT boxes coloured by bucket — same draw as _boxes_panel's GT loop."""
    out = img_bgr.copy()
    for g in gt_list:
        x0, y0, x1, y1 = [int(round(v)) for v in g["box"]]
        cv2.rectangle(out, (x0, y0), (x1, y1), _gt_color(g["name"]), 2)
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
    gt_all = defaultdict(list)
    for a in full["annotations"]:
        if a["image_id"] not in diag_set:
            continue
        x, y, w, h = a["bbox"]
        gt_all[a["image_id"]].append({"box": [x, y, x + w, y + h], "name": cat[a["category_id"]]})
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

    for iid in DIAG:
        im = img_meta[iid]
        H, W = im["height"], im["width"]
        img_bgr = cv2.imread(f"{ROOT}/data/Images/train2017/{im['file_name']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        r = run_esam(predictor, img_rgb, H, W)
        emr, final = r["emr_masks"], r["final_masks"]

        # ── FAITHFULNESS GUARD ──────────────────────────────────────────────
        if len(emr) != emr_target[iid] or len(final) != final_target[iid]:
            raise SystemExit(
                f"ABORT {iid}: EMR {len(emr)} vs {emr_target[iid]} | "
                f"FINAL {len(final)} vs {final_target[iid]} — pipeline perturbed.")

        gts = gt_all.get(iid, [])
        n_prop = sum(1 for m in final if mask_to_box(m) is not None)

        # (a) both — reuse _boxes_panel VERBATIM
        both = _panel(_boxes_panel(img_bgr, final, gts),
                      f"E-SAM proposals + GT  ({n_prop} prop | GT nov/base/other)")
        # (b) model only
        model = _panel(_model_only_panel(img_bgr, final),
                       f"E-SAM proposals only (no GT)  ({n_prop} boxes)")
        # (c) GT only
        gt = _panel(_gt_only_panel(img_bgr, gts),
                    f"GT only: novel(green)+base(orange)+other(grey)  ({len(gts)} GT)")

        cv2.imwrite(f"{_OUT}/{iid}_boxes_both.png", both)
        cv2.imwrite(f"{_OUT}/{iid}_boxes_model.png", model)
        cv2.imwrite(f"{_OUT}/{iid}_boxes_gt.png", gt)
        n_nov = sum(1 for g in gts if g["name"] in NOVEL)
        print(f"  {iid:>7}: {n_prop} proposals | {len(gts)} GT ({n_nov} novel) "
              f"-> both/model/gt png")

    print(f"\n  Images -> {_OUT}/{{id}}_boxes_{{both,model,gt}}.png")
    print("STOP — 12×3 box variants complete. No tuning.")


if __name__ == "__main__":
    main()
