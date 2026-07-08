"""
E-SAM end-to-end TIMING runner — full pipeline (MMG+EMR+USR) on 12 diagnostic IDs.

Thin runner: imports esam_pipeline.run_esam and reports its per-stage timing.
ZERO plotting, ZERO GT loading — so the timing number is uncontaminated.

NOTE: MMG is re-run in-memory per image (the 64-grid gallery was never persisted),
so t_total includes the redundant MMG cost and is directly comparable to the
Gate B / Gate C timing (same re-run convention).

Run:  python diagnostics/esam_e2e_timing.py
"""

import json
import os
import sys
import time

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esam_pipeline import run_esam, DEVICE  # noqa: E402
from gate_d_eval import ROOT, DIAG, mask_to_box  # noqa: E402

_OUT = f"{ROOT}/diagnostics/output/esam_e2e"
_FULL_TRAIN_N = 107_000


def main():
    os.makedirs(_OUT, exist_ok=True)
    img_meta = {im["id"]: im for im in json.load(
        open(f"{ROOT}/data/Annotations/instances_train2017_12img_diag.json"))["images"]}

    # ── GPU safety (same check the gates use) ───────────────────────────────
    used = torch.cuda.memory_allocated(DEVICE)
    total = torch.cuda.get_device_properties(DEVICE).total_memory
    print(f"GPU: {used/1e6:.0f} MB allocated / {total/1e6:.0f} MB total")
    if used > 2e9:
        raise RuntimeError("GPU already busy — stop.")

    # ── SAM-H exactly as the gates load it ──────────────────────────────────
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=f"{ROOT}/weights/sam_vit_h_4b8939.pth")
    sam.to(DEVICE).eval()
    predictor = SamPredictor(sam)
    print("Loaded SAM-H")
    print("NOTE: MMG re-run in-memory (gallery not persisted) -> t_total includes "
          "redundant MMG cost, comparable to Gate B/C.\n")
    print(f"{'img':>8} {'enc':>6} {'mmg':>6} {'emr':>6} {'usr':>6} {'total':>7} "
          f"{'n_ent':>5} {'n_box':>5}")
    print("-" * 64)

    rows = []
    for iid in DIAG:
        im = img_meta[iid]
        H, W = im["height"], im["width"]
        img_rgb = cv2.cvtColor(
            cv2.imread(f"{ROOT}/data/Images/train2017/{im['file_name']}"),
            cv2.COLOR_BGR2RGB)
        r = run_esam(predictor, img_rgb, H, W)
        t = r["timing"]
        n_ent = len(r["final_masks"])
        n_box = sum(1 for m in r["final_masks"] if mask_to_box(m) is not None)
        rows.append(dict(
            iid=iid, H=H, W=W,
            t_encoder=round(t["t_encoder"], 3), t_mmg=round(t["t_mmg"], 3),
            t_emr=round(t["t_emr"], 3), t_usr=round(t["t_usr"], 3),
            t_total=round(t["t_total"], 3),
            n_final_entities=n_ent, n_final_boxes=n_box))
        print(f"{iid:>8} {t['t_encoder']:>6.2f} {t['t_mmg']:>6.2f} {t['t_emr']:>6.2f} "
              f"{t['t_usr']:>6.2f} {t['t_total']:>7.2f} {n_ent:>5} {n_box:>5}")

    # ── aggregate ───────────────────────────────────────────────────────────
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    m_enc, m_mmg, m_emr, m_usr, m_tot = (
        mean("t_encoder"), mean("t_mmg"), mean("t_emr"), mean("t_usr"), mean("t_total"))
    proj_h = m_tot * _FULL_TRAIN_N / 3600

    print("-" * 64)
    print(f"{'mean':>8} {m_enc:>6.2f} {m_mmg:>6.2f} {m_emr:>6.2f} {m_usr:>6.2f} "
          f"{m_tot:>7.2f} {mean('n_final_entities'):>5.0f} {mean('n_final_boxes'):>5.0f}")
    print()
    print(f"  full E-SAM incl. USR — mean per-stage:  enc={m_enc:.2f}s  mmg={m_mmg:.2f}s  "
          f"emr={m_emr:.2f}s  usr={m_usr:.2f}s")
    print(f"  full E-SAM incl. USR — mean t_total:    {m_tot:.2f}s/image  "
          f"(MMG dominates; re-run in-memory)")
    print(f"  107k projection (full E-SAM incl. USR): {m_tot:.2f}s x {_FULL_TRAIN_N:,} "
          f"= {proj_h:.1f}h")

    out = f"{_OUT}/esam_e2e_timing.json"
    json.dump({
        "model": "SAM-H", "device": DEVICE, "n_images": n,
        "note": "MMG re-run in-memory (gallery not persisted); t_total includes "
                "redundant MMG cost, comparable to Gate B/C. ZERO plotting/GT.",
        "mean": {"t_encoder": round(m_enc, 3), "t_mmg": round(m_mmg, 3),
                 "t_emr": round(m_emr, 3), "t_usr": round(m_usr, 3),
                 "t_total": round(m_tot, 3)},
        "proj_107k_h_full_esam_incl_usr": round(proj_h, 1),
        "per_image": rows,
    }, open(out, "w"), indent=2)
    print(f"\n  Stats -> {out}")
    print("STOP — timing runner complete. No tuning.")


if __name__ == "__main__":
    main()
