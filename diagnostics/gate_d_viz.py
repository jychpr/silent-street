"""
E-SAM Gate D — VISUALIZATION ONLY. No metric changes.

Renders the existing gate_d_eval.json result as 4-panel overlays so the boxes can
be inspected. Boxes are rebuilt in-memory through the SAME code path Gate D used
(gate_d_eval.esam_entities + mask_to_box + passes_geo), and every per-image count is
ASSERTED against gate_d_eval.json. Any mismatch ABORTS — the viz must be faithful.

Per image -> diagnostics/output/gateD_viz/{id}_gateD.png  (2x2 sub-panels):
  1 GT             — base-class GT green, novel-class GT (17 OV-COCO novel) blue
  2 OLN R2 filt    — OW_COCO_R2.json boxes red, each labeled with its weight w
  3 E-SAM raw      — all raw entity boxes orange  (+ foreground-merge markers)
  4 E-SAM filtered — geometric-filtered boxes magenta (+ foreground-merge markers)

Foreground-merge marker = thin red dashed box on any E-SAM box covering >=2 GT THING
instances at IoU>=0.3 (the 13 from Table B), annotated with the things it covers.
For 92869 the horse+rider region is labeled to confirm it is two separate boxes.

Run: python diagnostics/gate_d_viz.py
"""

import json
import logging
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_d_eval import (  # noqa: E402  — SAME box-generation path as Gate D
    esam_entities, mask_to_box, passes_geo, iou_matrix,
    DIAG, NOVEL, BASE48, THING, IOU_MERGE, DEVICE, ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gateD_viz")

OUT = f"{ROOT}/diagnostics/output/gateD_viz"
EVAL_JSON = f"{ROOT}/diagnostics/output/esam_gateC/gate_d_eval.json"


def load_gt():
    """Replicates gate_d_eval.main()'s GT build; adds thing names for merge labels."""
    full = json.load(open(f"{ROOT}/data/Annotations/instances_train2017.json"))
    cat = {c["id"]: c["name"] for c in full["categories"]}
    diag_set = set(DIAG)
    gt = defaultdict(lambda: {"novel": [], "base": [], "all": [],
                              "thing": [], "thing_named": [], "by": defaultdict(list)})
    for a in full["annotations"]:
        if a["image_id"] not in diag_set:
            continue
        x, y, w, h = a["bbox"]
        box = [x, y, x + w, y + h]
        nm = cat[a["category_id"]]
        g = gt[a["image_id"]]
        g["all"].append(box)
        if nm in NOVEL:
            g["novel"].append(box)
        elif nm in BASE48:
            g["base"].append(box)
        if nm in THING:
            g["thing"].append(box)
            g["thing_named"].append((box, nm))
        g["by"][nm].append(box)
    return gt


def load_oln():
    """OW_COCO_R2 boxes per diag image as (xyxy, weight) — same conversion as Gate D."""
    r2 = json.load(open(f"{ROOT}/ow_labels/OW_COCO_R2.json"))
    diag_set = set(DIAG)
    oln = defaultdict(list)
    for o in r2:
        if o["image_id"] in diag_set:
            x, y, w, h = o["bbox"]
            oln[o["image_id"]].append(([x, y, x + w, y + h], o.get("weight")))
    return oln


def draw_boxes(ax, boxes, color, lw=0.9, labels=None):
    for i, b in enumerate(boxes):
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                               fill=False, edgecolor=color, linewidth=lw))
        if labels is not None and labels[i] is not None:
            ax.text(b[0], b[1] - 2, labels[i], color="black", fontsize=6,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.7))


def merge_boxes_info(boxes, thing_named):
    """Box indices covering >=2 GT THING instances @IoU>=0.3, with covered names.

    Mirrors gate_d_eval.main().merge_count exactly; len() must equal the JSON count.
    """
    if not boxes or not thing_named:
        return []
    tb = np.array([t[0] for t in thing_named])
    tn = [t[1] for t in thing_named]
    M = iou_matrix(np.array(boxes), tb)
    hits = M >= IOU_MERGE
    out = []
    for bi in range(len(boxes)):
        cov = np.where(hits[bi])[0]
        if len(cov) >= 2:
            out.append((bi, [tn[c] for c in cov]))
    return out


def mark_merges(ax, boxes, merges):
    for bi, names in merges:
        b = boxes[bi]
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                               edgecolor="red", linewidth=1.5, linestyle="--"))
        ax.text((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, "MERGE\n" + "+".join(names),
                color="red", fontsize=6, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="yellow", ec="red", alpha=0.75))


def annotate_horse_rider(ax_raw, ax_flt, g, filt_boxes, row):
    """Label the 92869 horse+rider region; confirm it is two separate E-SAM boxes."""
    horses = g["by"].get("horse", [])
    persons = g["by"].get("person", [])
    if not horses or not persons:
        return "92869 horse/person GT missing"
    HM = iou_matrix(np.array(horses), np.array(persons))
    hi, pi = np.unravel_index(int(np.argmax(HM)), HM.shape)
    horse, rider = horses[hi], persons[pi]
    for ax in (ax_raw, ax_flt):
        ax.add_patch(Rectangle((horse[0], horse[1]), horse[2] - horse[0], horse[3] - horse[1],
                               fill=False, edgecolor="cyan", linewidth=1.8, linestyle=":"))
        ax.add_patch(Rectangle((rider[0], rider[1]), rider[2] - rider[0], rider[3] - rider[1],
                               fill=False, edgecolor="yellow", linewidth=1.8, linestyle=":"))
        ax.text(horse[0], horse[3] + 3, "GT horse", color="cyan", fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))
        ax.text(rider[0], rider[1] - 3, "GT rider", color="yellow", fontsize=7, va="bottom",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", ec="none", alpha=0.5))

    def best(tgt):
        if not filt_boxes:
            return -1, 0.0
        v = iou_matrix(np.array([tgt]), np.array(filt_boxes))[0]
        j = int(np.argmax(v))
        return j, float(v[j])

    bh, ih = best(horse)
    br, ir = best(rider)
    distinct = bh >= 0 and br >= 0 and bh != br
    if bh >= 0:
        b = filt_boxes[bh]
        ax_flt.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                   edgecolor="cyan", linewidth=2.2))
    if br >= 0:
        b = filt_boxes[br]
        ax_flt.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                   edgecolor="yellow", linewidth=2.2))
    return (f"horse+rider: hr_flt={row.get('hr_flt')} | best-filt horse#{bh}(IoU{ih:.2f}) "
            f"rider#{br}(IoU{ir:.2f}) -> {'TWO SEPARATE BOXES' if distinct else 'SAME/none'}")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = {r["iid"]: r for r in json.load(open(EVAL_JSON))["rows"]}
    img_meta = {im["id"]: im for im in json.load(
        open(f"{ROOT}/data/Annotations/instances_train2017_12img_diag.json"))["images"]}

    log.info("Loading full instances_train2017.json (~30s)…")
    gt = load_gt()
    log.info("Loading OW_COCO_R2.json…")
    oln = load_oln()

    used = torch.cuda.memory_allocated(DEVICE)
    log.info(f"GPU allocated {used / 1e6:.0f} MB / "
             f"{torch.cuda.get_device_properties(DEVICE).total_memory / 1e6:.0f} MB")
    if used > 2e9:
        raise RuntimeError("GPU already busy — stop.")
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=f"{ROOT}/weights/sam_vit_h_4b8939.pth")
    sam.to(DEVICE).eval()
    predictor = SamPredictor(sam)
    log.info("Loaded SAM-H")
    log.info("=" * 90)

    manifest = []
    for iid in DIAG:
        row = rows[iid]
        im = img_meta[iid]
        H, W = im["height"], im["width"]
        img_rgb = cv2.cvtColor(cv2.imread(f"{ROOT}/data/Images/train2017/{im['file_name']}"),
                               cv2.COLOR_BGR2RGB)
        g = gt[iid]

        # rebuild boxes via the SAME Gate D path
        ents = esam_entities(predictor, img_rgb, H, W)
        raw_boxes = [b for b in (mask_to_box(e) for e in ents) if b is not None]
        filt_boxes = [b for b in raw_boxes if passes_geo(b, W, H)]
        oln_pairs = oln.get(iid, [])
        oln_boxes = [p[0] for p in oln_pairs]

        # ── FAITHFULNESS ASSERTS vs gate_d_eval.json ───────────────────────
        problems = []
        if len(filt_boxes) != row["n_filt"]:
            problems.append(f"filt {len(filt_boxes)}!={row['n_filt']}")
        if len(raw_boxes) != row["n_raw"]:
            problems.append(f"raw {len(raw_boxes)}!={row['n_raw']}")
        if len(oln_boxes) != row["n_oln"]:
            problems.append(f"oln {len(oln_boxes)}!={row['n_oln']}")
        if len(g["novel"]) != row["n_nov"]:
            problems.append(f"novel {len(g['novel'])}!={row['n_nov']}")
        if len(g["base"]) != row["n_bas"]:
            problems.append(f"base {len(g['base'])}!={row['n_bas']}")
        if problems:
            raise AssertionError(f"[{iid}] count mismatch vs gate_d_eval.json: {problems}")

        merges_raw = merge_boxes_info(raw_boxes, g["thing_named"])
        merges_flt = merge_boxes_info(filt_boxes, g["thing_named"])
        if len(merges_raw) != row["merge_raw"] or len(merges_flt) != row["merge_flt"]:
            raise AssertionError(
                f"[{iid}] merge mismatch: raw {len(merges_raw)}!={row['merge_raw']} "
                f"flt {len(merges_flt)}!={row['merge_flt']}")

        # ── render 2x2 ─────────────────────────────────────────────────────
        panel_w = 6.5
        panel_h = panel_w * H / W
        fig, axes = plt.subplots(2, 2, figsize=(2 * panel_w + 0.6, 2 * panel_h + 1.4))
        for ax in axes.ravel():
            ax.imshow(img_rgb)
            ax.set_xticks([])
            ax.set_yticks([])

        ax = axes[0, 0]
        draw_boxes(ax, g["base"], "lime", lw=1.0)
        draw_boxes(ax, g["novel"], "deepskyblue", lw=1.3)
        ax.set_title(f"GT (base={len(g['base'])} green, novel={len(g['novel'])} blue)")

        ax = axes[0, 1]
        labels = [f"w={w:.2f}" if isinstance(w, (int, float)) else "w=?" for (_, w) in oln_pairs]
        draw_boxes(ax, oln_boxes, "red", lw=1.4, labels=labels)
        ax.set_title(f"OLN R2 (filtered): {len(oln_boxes)} boxes")

        ax = axes[1, 0]
        draw_boxes(ax, raw_boxes, "orange", lw=0.8)
        mark_merges(ax, raw_boxes, merges_raw)
        ax.set_title(f"E-SAM raw: {len(raw_boxes)} boxes  (fg-merge={len(merges_raw)})")

        ax = axes[1, 1]
        draw_boxes(ax, filt_boxes, "magenta", lw=0.8)
        mark_merges(ax, filt_boxes, merges_flt)
        ax.set_title(f"E-SAM filtered: {len(filt_boxes)} boxes  (fg-merge={len(merges_flt)})")

        hr_note = ""
        if iid == 92869:
            hr_note = annotate_horse_rider(axes[1, 0], axes[1, 1], g, filt_boxes, row)

        fig.suptitle(f"Gate D viz — image {iid} ({W}x{H})  "
                     f"[OLN {len(oln_boxes)} | E-SAM raw {len(raw_boxes)} | filt {len(filt_boxes)}]"
                     + (f"\n{hr_note}" if hr_note else ""), fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(f"{OUT}/{iid}_gateD.png", dpi=110)
        plt.close(fig)

        manifest.append(dict(iid=iid, nb=len(g["base"]), nn=len(g["novel"]),
                             no=len(oln_boxes), nr=len(raw_boxes), nf=len(filt_boxes),
                             mr=len(merges_raw), mf=len(merges_flt), row=row))
        log.info(f"  wrote {iid}_gateD.png  oln={len(oln_boxes)} raw={len(raw_boxes)} "
                 f"filt={len(filt_boxes)} (fg-merge r/f={len(merges_raw)}/{len(merges_flt)})"
                 + (f"  | {hr_note}" if hr_note else ""))

    # ── manifest ───────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("GATE D VIZ MANIFEST — rendered counts vs gate_d_eval.json (source of truth)")
    print("=" * 96)
    print(f"{'img':>7} | {'GT b/nov':>9} | {'rendered oln/raw/filt':>22} | "
          f"{'json oln/raw/filt':>20} | {'fg-merge r/f':>12} | match")
    print("-" * 96)
    all_ok = True
    for m in manifest:
        r = m["row"]
        ok = (m["no"] == r["n_oln"] and m["nr"] == r["n_raw"] and m["nf"] == r["n_filt"]
              and m["mr"] == r["merge_raw"] and m["mf"] == r["merge_flt"])
        all_ok &= ok
        print(f"{m['iid']:>7} | {m['nb']:>3}/{m['nn']:<5} | "
              f"{m['no']:>6} {m['nr']:>6} {m['nf']:>6}      | "
              f"{r['n_oln']:>5} {r['n_raw']:>6} {r['n_filt']:>6}   | "
              f"{m['mr']:>5}/{m['mf']:<5} | {'OK' if ok else 'MISMATCH'}")
    print("-" * 96)
    tot = lambda k: sum(m[k] for m in manifest)
    print(f"{'TOTAL':>7} |           | "
          f"{tot('no'):>6} {tot('nr'):>6} {tot('nf'):>6}      | "
          f"{'54':>5} {'1099':>6} {'1020':>6}   | "
          f"{tot('mr'):>5}/{tot('mf'):<5} |")
    print("=" * 96)
    print(f"\n{'ALL IMAGES MATCH gate_d_eval.json — viz is faithful.' if all_ok else '*** MISMATCH — viz NOT faithful, investigate. ***'}")
    print(f"12 PNGs written to {OUT}/")


if __name__ == "__main__":
    main()
