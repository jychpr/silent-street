"""
E-SAM FINAL SEGMENTATION VIZ — clean, standalone. One image per ID showing ONLY
the final EMR+USR entity set as a normal segmentation result (no highlight, no
delta, no stage comparison, no boxes).

Thin runner: imports esam_pipeline.run_esam — no pipeline re-implementation.

FAITHFULNESS GUARD: assert len(final_masks) == gate_e_stats.json post-USR count
per image; ABORT on mismatch.

Run:  python diagnostics/esam_final_seg.py
"""

import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esam_pipeline import run_esam, DEVICE  # noqa: E402
from gate_d_eval import ROOT, DIAG  # noqa: E402

_OUT = f"{ROOT}/diagnostics/output/esam_final_seg"


def _overlay(img_bgr, masks, seed=7):
    """Translucent per-entity colour overlay (same style as MMG/EMR panels)."""
    rng = np.random.default_rng(seed)
    out = img_bgr.copy().astype(np.float32)
    for m in masks:
        if not m.any():
            continue
        c = rng.integers(64, 220, size=3).astype(np.float32)
        for k in range(3):
            out[:, :, k] = np.where(m, out[:, :, k] * 0.5 + c[k] * 0.5, out[:, :, k])
    return out.astype(np.uint8)


def _title(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                1, cv2.LINE_AA)
    return out


def main():
    os.makedirs(_OUT, exist_ok=True)

    ge = json.load(open(f"{ROOT}/diagnostics/output/esam_gateE/gate_e_stats.json"))
    final_target = {r["iid"]: r["m_e_after"] for r in ge["per_image"]}
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
        final = r["final_masks"]

        if len(final) != final_target[iid]:
            raise SystemExit(
                f"ABORT {iid}: final {len(final)} vs gate_e {final_target[iid]} — perturbed.")

        panel = _title(_overlay(img_bgr, final),
                       f"E-SAM final segmentation: {len(final)} entities")
        cv2.imwrite(f"{_OUT}/{iid}_final_seg.png", panel)
        print(f"  {iid:>7}: {len(final)} entities -> {iid}_final_seg.png")

    print(f"\n  Images -> {_OUT}/{{id}}_final_seg.png")
    print("STOP — final segmentation viz complete. No tuning.")


if __name__ == "__main__":
    main()
